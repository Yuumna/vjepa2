import pandas as pd
import re

def clean_filename(x):
    x = x.strip().lower()
    x = re.sub(r"^\d+_", "", x)
    return x

with open("/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset/train.txt", "r") as f:
    train_set = set(clean_filename(line) for line in f)

with open("/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset/val.txt", "r") as f:
    val_set = set(clean_filename(line) for line in f)

with open("/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset/test.txt", "r") as f:
    test_set = set(clean_filename(line) for line in f)

print(test_set)
df = pd.read_csv("filtered_bddx_clean_6classes.csv")

df["filename"] = df["local_path"].apply(lambda x: x.split("/")[-1])
#without .mov extension
df["filename"] = df["filename"].str.replace(".mov", "", regex=False)

def get_split(fname):
    if fname in train_set:
        return "train"
    elif fname in val_set:
        return "val"
    elif fname in test_set:
        return "test"
    else:
        return "unknown"

df["split"] = df["filename"].apply(get_split)

df.to_csv("annotated_with_split.csv", index=False)

#number of each split
split_counts = df["split"].value_counts()
print(split_counts)