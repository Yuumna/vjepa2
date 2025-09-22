import os
import shutil
import pandas as pd

train_csv = "bddx_gps_train_split.csv"
val_csv = "bddx_gps_val_split.csv"
train_dir = "/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset_gps/videos/train"
val_dir = "/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset_gps/videos/val"

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
"""
def replace_path_column(df, column, new_base_path):
    df[column] = df[column].apply(lambda x: os.path.join(new_base_path, os.path.basename(x)))
    return df


tr = pd.read_csv("/work/dlclarge2/aliy-vjepa/vjepa2/notebooks/bddx_gps_train_split.csv", header=None, delimiter=" ")
tr = replace_path_column(tr, 0, "/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset_gps/videos/train")
tr.to_csv("bddx_gps_train_new_path.csv", index=False, header=False, sep=" ")
val = pd.read_csv("/work/dlclarge2/aliy-vjepa/vjepa2/notebooks/bddx_gps_val_split.csv", header=None, delimiter=" ")
val = replace_path_column(val, 0, "/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Datase_gpst/videos/val")
val.to_csv("bddx_gps_val_new_path.csv", index=False, header=False, sep=" ")

"""