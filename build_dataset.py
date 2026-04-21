import os
import shutil
import ast
import pandas as pd
from pathlib import Path

##########################################################################################
# Brings in data from Animal Kingdom dataset and organizes it into folders by animal type.
##########################################################################################

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH    = r"C:\\Users\\amosa\\Downloads\\image\\AR_metadata.csv"   # ← update filename
IMAGE_DIR   = r"C:\\Users\\amosa\\Downloads\\image\\image"
OUTPUT_DIR  = "./train_data"

TARGET_ANIMALS = {"Lion", "Young Lion", "Common Crane", "Butterfly", "Sheep", "Frog", "Horse"}

# Common image extensions to search for
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
# ───────────────────────────────────────────────────────────────────────────────

def parse_animal_list(raw):
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(a).strip() for a in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def main():
    # ── Load CSV ───────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH, sep="\t")
    df.columns = df.columns.str.strip()

    if "video_id" not in df.columns:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()

    print(f"Columns found : {df.columns.tolist()}")
    print(f"Rows loaded   : {len(df)}\n")

    copied  = 0
    skipped = 0
    missing = 0

    for _, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        animals  = parse_animal_list(row["list_animal"])

        matched_animals = TARGET_ANIMALS.intersection(set(animals))
        if not matched_animals:
            continue

        # Each video_id is a subfolder e.g. Downloads/image/AAACXZTV/
        video_folder = os.path.join(IMAGE_DIR, video_id)

        if not os.path.isdir(video_folder):
            print(f"  [MISSING]  {video_id} — folder not found")
            missing += 1
            continue

        # Get all jpg frames inside that subfolder
        frames = sorted([
            f for f in os.listdir(video_folder)
            if f.lower().endswith(".jpg")
        ])

        if not frames:
            print(f"  [EMPTY]    {video_id} — folder exists but has no .jpg files")
            missing += 1
            continue

        for animal in matched_animals:
            folder_name = animal.replace(" ", "_")   # "Young Lion" → "Young_Lion"
            dest_dir    = os.path.join(OUTPUT_DIR, folder_name)
            os.makedirs(dest_dir, exist_ok=True)

            for frame in frames:
                src_path  = os.path.join(video_folder, frame)
                dest_path = os.path.join(dest_dir, frame)

                if os.path.exists(dest_path):
                    skipped += 1
                    continue

                shutil.copy2(src_path, dest_path)
                copied += 1

            print(f"  [COPIED]  {video_id} ({len(frames)} frames)  →  {folder_name}/")

    print("\n── Summary ────────────────────────────────────")
    print(f"  Frames copied  : {copied}")
    print(f"  Frames skipped : {skipped}  (already existed)")
    print(f"  Videos missing : {missing}  (folder not on disk)")
    print(f"  Output dir     : {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()