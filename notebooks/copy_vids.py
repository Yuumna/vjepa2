import os
import shutil
import pandas as pd

train_csv = "bddx_train_split.csv"
val_csv = "bddx_val_split.csv"
train_dir = "/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset/videos/train"
val_dir = "/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset/videos/val"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def copy_files(csv_file, out_dir):
    df = pd.read_csv(csv_file, header=None, delimiter=" ")
    for fname in df[0]:
        src = os.path.join(fname)
        dst = os.path.join(out_dir, os.path.basename(fname))
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Missing: {src}")

copy_files(train_csv, train_dir)
copy_files(val_csv, val_dir)