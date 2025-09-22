import os
import pandas as pd
from decord import VideoReader, cpu
# ---- config ----
OUTPUT_DIR  = "evals/action_anticipation_frozen"
ID_PREFIX   = "bddx_"
PAD         = 5                      # -> bddx_00001

#df = pd.read_csv("/work/dlclarge2/aliy-vjepa/vjepa2/notebooks/bddx_train_split.csv",sep=" " ,names = ["local_path", "action_class", "start_time", "end_time"])
"""


def reformat_anticipation(file,output_file):
    df = pd.read_csv(file,sep=" " ,names = ["local_path", "action_class", "start_time", "end_time"])
    unique_paths = pd.Series(sorted(df["local_path"].unique()), name="local_path")
    mapping = pd.DataFrame(unique_paths)
    mapping["video_id"] = mapping["local_path"].apply(lambda x : x.split("/")[-1].split(".")[0])
    
    all_df = df.merge(mapping, on="local_path", how="left")
    vr = VideoReader(df["local_path"][0], num_threads=-1, ctx=cpu(0))
    fps = vr.get_avg_fps()
    
    all_df["start_frame"] =  (all_df["start_time"] * fps).astype(int)
    all_df["end_frame"] = (all_df["end_time"] * fps).astype(int)

    all_df.to_csv(os.path.join(OUTPUT_DIR, f"{output_file}.csv"), index=False)
    print("mapping saved:", os.path.join(OUTPUT_DIR, f"{output_file}.csv"))


reformat_anticipation("/work/dlclarge2/aliy-vjepa/vjepa2/notebooks/bddx_train_split.csv","bddx_train_anticipation" )
reformat_anticipation("/work/dlclarge2/aliy-vjepa/vjepa2/notebooks/bddx_val_split.csv","bddx_val_anticipation" )


"""




def replace_path_column(df, column, new_base_path):
    df[column] = df[column].apply(lambda x: os.path.join(new_base_path, os.path.basename(x)))
    return df


tr = pd.read_csv("/work/dlclarge2/aliy-vjepa/vjepa2/evals/action_anticipation_frozen/bddx_train_anticipation.csv")
tr = replace_path_column(tr, "local_path", "/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset/videos/train")
tr.to_csv("./evals/action_anticipation_frozen/bddx_train_anticipation_new_path.csv", index=False)
val = pd.read_csv("/work/dlclarge2/aliy-vjepa/vjepa2/evals/action_anticipation_frozen/bddx_val_anticipation.csv")
val = replace_path_column(val, "local_path", "/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset/videos/val")
val.to_csv("./evals/action_anticipation_frozen/bddx_val_anticipation_new_path.csv", index=False)


