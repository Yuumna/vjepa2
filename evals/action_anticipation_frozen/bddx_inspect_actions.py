import os
import pandas as pd
import json
from PIL import Image
from torch.utils.data import Dataset
from logging import getLogger
from collections import defaultdict
annot_df = pd.read_csv("/work/dlclarge2/aliy-vjepa/datasets/BDD-X-Dataset/BDD-X-Annotations_v1.csv")



def get_unique_actions(df):
    actions_set = set()
    actions_dict = defaultdict(int)
    total = 0
    for idx, row in df.iterrows():
        for i in range(1, 16):  
            action_key = f"Answer.{i}action"
            if pd.notna(row.get(action_key)):
                action = str(row[action_key]).strip().lower()
                if action:
                    total += 1
                    actions_set.add(action)
                    actions_dict[action] += 1
    actions_dict = dict(sorted(actions_dict.items(), key=lambda item: item[1], reverse=True))
    actions_dict = {k: v for k, v in actions_dict.items() if v >= 100}

    print(f"Total actions found: {total}")
    print(actions_dict)
    return actions_set

##Total actions found: 26538
##Unique actions: 6198

print(len(get_unique_actions(annot_df))) 
