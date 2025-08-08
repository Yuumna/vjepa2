import os
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
import tqdm

csv_path = "bddx_flattened.csv"
df = pd.read_csv(csv_path)

def extract_filename(s3_url):
    return os.path.basename(urlparse(s3_url).path)

df["filename"] = df["Input.Video"].apply(extract_filename)

video_root = Path("/data/nxtaimraid02/datasets/BDD100K/videos/unzipped_files")
video_index = {}

for split in ["train", "val", "test"]:
    split_dir = video_root / split
    for dirpath, _, filenames in tqdm.tqdm(os.walk(split_dir), desc=f"Indexing {split}"):
        for f in filenames:
            if f.endswith(".mov"):
                full_path = os.path.join(dirpath, f)
                video_index[f] = full_path

def map_to_local_path(filename):
    return video_index.get(filename, None)

print(video_index)
df["local_path"] = df["filename"].apply(map_to_local_path)

df.to_csv("bddx_with_local_paths.csv", index=False)

print("✅ Mapping complete. Results saved to 'bddx_with_local_paths.csv'")