import os
import pandas as pd
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm

# Step 1: Load CSV and extract filename
df = pd.read_csv("bddx_flattened.csv")

def extract_filename(s3_url):
    return os.path.basename(urlparse(s3_url).path)

df["filename"] = df["Input.Video"].apply(extract_filename)

unique_filenames = df["filename"].unique()


search_root = Path("/data/nxtaimraid02/datasets/BDD100K/videos/unzipped_files")
found_paths = {}


search_dirs = []

# Train folders: train/bdd100k_videos_train_00 → _69
for i in range(70):
    sub = f"bdd100k_videos_train_{i:02d}"
    path = Path(f"/data/nxtaimraid02/datasets/BDD100K/videos/unzipped_files/train/{sub}/bdd100k/videos/train")
    search_dirs.append(path)

# Val folders: val/bdd100k_videos_val_00 → _08
for i in range(9):
    sub = f"bdd100k_videos_val_{i:02d}"
    path = Path(f"/data/nxtaimraid02/datasets/BDD100K/videos/unzipped_files/val/{sub}/bdd100k/videos/val")
    search_dirs.append(path)

# Test folder
search_dirs.append(Path("/data/nxtaimraid02/datasets/BDD100K/videos/unzipped_files/test/videos"))

found_paths = {}
missing_files = []
#print(search_dirs)
for fname in tqdm(unique_filenames, desc="Searching only in .mov folders"):
    for mov_dir in search_dirs:
        if mov_dir== search_dirs[-1]:
            print(f"Searching in test folder: {mov_dir}")
        target = mov_dir / fname
        if target.exists():
            found_paths[fname] = str(target)
            break
    else:
        print(f"File not found: {fname}")
        missing_files.append(fname)
        found_paths[fname] = None  # not found


output_file = "found_paths_2.json"
missing_files_file = "missing_files.txt"
with open(missing_files_file, "w") as f:
    for missing in missing_files:
        f.write(f"{missing}\n")
with open(output_file, "w") as f:
    import json
    json.dump(found_paths, f)

df["local_path"] = df["filename"].map(found_paths)

df.to_csv("bddx_with_local_paths_last.csv", index=False)
print(f"Missing files: {len(missing_files)}")
print("✅ Mapping done. File saved.")