import pandas as pd

#text_to_idx = {'accelerate':0, 'steady_speed':1, 'decelerate':2, 'stop':3, 'switch_lane':4,'turn_right':5, 'reverse':6, 'turn_left':7, 'intersection':8, 'u_turn':9,}
text_to_idx = {'accelerate':0, 'steady_speed':1, 'decelerate':2, 'stop':3, 'turn_right':4, 'turn_left':5}

df = pd.read_csv("/work/dlclarge2/aliy-vjepa/vjepa2/notebooks/filtered_bddx_clean_6classes_updated.csv")  # Adjust filename



df["label"] = df["action_class"].map(text_to_idx)


#df["label"] = df["label"].astype(int)

df_out = df[["local_path", "label","start","end"]]

df_out.to_csv("bddx_vjepa2_format_6classes_updated.csv", sep=" ", header=False, index=False)