import pandas as pd

df = pd.read_csv("bddx_with_local_paths_dedup.csv")

df["local_path"] = df["local_path"].fillna("").astype(str)
valid_paths = df["local_path"].apply(lambda x: x.strip().lower() not in ["", "none"])

df["action_class"] = df["action_class"].astype(str).str.strip()
valid_classes_list = ["accelerate", "stop", "turn_left", "turn_right", "decelerate", "steady_speed"]

valid_classes = df["action_class"].isin(valid_classes_list)


filtered_df = df[valid_paths & valid_classes].copy()

filtered_df = filtered_df[filtered_df["start"] < filtered_df["end"]]

filtered_df = filtered_df[["local_path", "action_class", "start", "end"]]

filtered_df.to_csv("filtered_bddx_clean_6classes_updated.csv", index=False)
print("Clean filtered CSV saved.")