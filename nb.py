import os
import time
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from pose import POSES

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Error: API Key not found. Check your .env file!")
    exit()

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=600_000)
)

# 1. PATHS
REF_FACE_PATH = "reference image face"
PROMPTS_FOLDER = "prompts"
OUTPUT_BASE = "output folder"

os.makedirs(PROMPTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# 2. MODEL DEFINITIONS
MODELS = {
    "1": "gemini-2.5-flash-image",       # Reliable, good at consistency
    "2": "gemini-3-pro-image-preview"     # Experimental, high detail
}

# 3. SAFETY CONFIG
# Note: 'OFF' often causes 400 errors. Changed to BLOCK_ONLY_HIGH for stability.
config = types.GenerateContentConfig(
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH")
    ],
    response_modalities=["IMAGE"],
)

def save_image(response, path):
    """Saves image and returns bytes. Returns None if failed."""
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                img = Image.open(BytesIO(part.inline_data.data))
                img.save(path)
                print(f"   ✅ Saved: {path}")
                return part.inline_data.data
    return None

# ==========================================
# MAIN INTERACTIVE LOOP
# ==========================================

# Check Reference Face
if not os.path.exists(REF_FACE_PATH):
    print(f"❌ ERROR: Face reference not found at {REF_FACE_PATH}")
    exit()

with open(REF_FACE_PATH, "rb") as f:
    face_bytes = f.read()

# Get text files
files = [f for f in os.listdir(PROMPTS_FOLDER) if f.endswith(".txt")]
if not files:
    print(f"⚠️ No .txt files in {PROMPTS_FOLDER}")
    exit()

print("\n--- 🤖 MODEL SELECTION ---")
print("1: Gemini 2.5 Flash (Recommended: Fast, Best Consistency)")
print("2: Gemini 3 Pro     (Experimental: Higher Detail, Slower)")
choice = input("Select Model (1 or 2): ").strip()

SELECTED_MODEL = MODELS.get(choice, "gemini-2.5-flash-image") # Default to Flash if invalid
print(f"✅ Using Model: {SELECTED_MODEL}\n")

print(f"🚀 Found {len(files)} prompts. Starting Interactive Session...\n")

for filename in files:
    clean_name = os.path.splitext(filename)[0]
    txt_path = os.path.join(PROMPTS_FOLDER, filename)

    # 1. Generate a unique ID based on the current time
    run_id = int(time.time())

    # 2. Create the unique folder
    current_output_dir = os.path.join(OUTPUT_BASE, clean_name, f"run_{run_id}")
    os.makedirs(current_output_dir, exist_ok=True)

    with open(txt_path, "r") as f:
        prompt_text = f.read().strip()

    print("------------------------------------------------------")
    print(f"📂 Processing: {filename}")
    print(f"📝 Prompt: {prompt_text[:60]}...")
    print(f"💾 Saving to: .../{clean_name}/run_{run_id}/")
    
    # --- STEP 1: GENERATE MASTER IMAGE (Face + Text) ---
    print(f"   🎨 Generating Concept Image using {SELECTED_MODEL}...")
    
    master_prompt = f"""
    Fashion photography of [the person in the reference image].
    {prompt_text}
    CRITICAL: Maintain the facial identity of the reference image exactly.
    """
    
    master_path = os.path.join(current_output_dir, "00_MASTER_CONCEPT.png")
    master_image_bytes = None

    try:
        response = client.models.generate_content(
            model=SELECTED_MODEL,  # <--- USES YOUR SELECTION
            contents=[
                types.Part.from_bytes(data=face_bytes, mime_type="image/jpeg"), 
                master_prompt
            ],
            config=config
        )
        master_image_bytes = save_image(response, master_path)
    except Exception as e:
        print(f"   ❌ Error generating concept: {e}")
        continue

    if not master_image_bytes:
        print("   ⚠️ Failed to generate concept image. Skipping.")
        continue

    # --- STEP 2: ASK THE USER ---
    print(f"\n   👀 Check the image at: {master_path}")
    user_choice = input(f"   ❓ Generate poses for '{clean_name}'? (y/n): ").strip().lower()

    if user_choice != 'y':
        print("   ⏩ Skipping poses. Moving to next prompt.")
        continue

    # --- STEP 3: GENERATE POSES (Using the NEW Master Image) ---
    print(f"   📸 Generating {len(POSES)} poses using {SELECTED_MODEL}...")

    for i, pose in enumerate(POSES):
        print(f"      [{i+1}/{len(POSES)}] {pose}...")
        
        pose_prompt = f"""
        TASK: Create a variation of the [ATTACHED IMAGE].
        
        INSTRUCTIONS:
        1. Keep the exact character (face, hair, clothes) from the [ATTACHED IMAGE].
        2. Keep the exact background environment from the [ATTACHED IMAGE].
        3. CHANGE ONLY THE POSE to: {pose}.
        """

        try:
            response = client.models.generate_content(
                model=SELECTED_MODEL, # <--- USES YOUR SELECTION
                contents=[
                    types.Part.from_bytes(data=master_image_bytes, mime_type="image/png"), 
                    pose_prompt
                ],
                config=config
            )
            
            pose_filename = os.path.join(current_output_dir, f"pose_{i+1}.png")
            save_image(response, pose_filename)
            time.sleep(2) # Increased pause for Pro model safety

        except Exception as e:
            print(f"      ❌ Pose failed: {e}")

    print(f"   ✨ All poses finished for {clean_name}.\n")

print("✅ Session Complete.")