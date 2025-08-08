# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
import math 
import imageio
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))
GIF_DIR = "/work/dlclarge2/aliy-vjepa/vjepa2/evals/video_classification_frozen/vitl-256/bddx/video_classification_frozen/bddx-vitl16-16x1-16f_updated_split/gifs"
os.makedirs(GIF_DIR,exist_ok=True)

def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 encoder
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 classifier
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform

model_pt, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
model_pt.cuda().eval()

# Build PyTorch preprocessing transform
pt_video_transform = build_pt_video_transform(img_size=256)

def get_video(fname, start_time=None, end_time=None):
    vr = VideoReader(fname)
    # choosing some frames here, you can define more complex sampling strategy
    video_fps = math.ceil(vr.get_avg_fps())
    start_frame = max(0, int(start_time * video_fps)) if start_time is not None else 0
    end_frame = min(len(vr), int(end_time * video_fps)) if end_time is not None else len(vr)
    frame_idx = np.arange(start_frame, end_frame, 2)
    video = vr.get_batch(frame_idx).asnumpy()
    base_name = os.path.splitext(os.path.basename(fname))[0]
    gif_path = os.path.join(GIF_DIR, f"{base_name}.gif")
    imageio.mimsave(gif_path, video, fps=video_fps)
    return video


def forward_vjepa_video( model_pt, pt_transform, fname, start_time, end_time):
    # Run a sample inference with VJEPA
    with torch.inference_mode():
        # Read and pre-process the image
        video = get_video(fname, start_time= start_time,end_time=end_time )  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        x_pt = pt_transform(video).cuda().unsqueeze(0)
        # Extract the patch-wise features from the last layer
        out_patch_features_pt = model_pt(x_pt)

    return  out_patch_features_pt


def get_vjepa_video_classification_results(classifier, out_patch_features_pt):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))

    with torch.inference_mode():
        out_classifier = classifier(out_patch_features_pt)


    print("Top 5 predicted class names:")
    top5_indices = out_classifier.topk(5).indices[0]
    top5_probs = F.softmax(out_classifier.topk(5).values[0]) * 100.0  # convert to percentage
    for idx, prob in zip(top5_indices, top5_probs):
        str_idx = str(idx.item())
        print(f"{SOMETHING_SOMETHING_V2_CLASSES[str_idx]} ({prob}%)")

    return top5_indices, top5_probs


def run_sample_inference(probe_path, fname, start_time, end_time):
  


    # Inference on video
    out_patch_features_pt = forward_vjepa_video(
         model_pt, pt_video_transform, fname, start_time, end_time
    )

    # Initialize the classifier
    classifier_model_path = probe_path
    classifier = (
        AttentiveClassifier(embed_dim=model_pt.embed_dim, num_heads=16, depth=1, num_classes=6).cuda().eval()
    )
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

    # Download SSV2 classes if not already present
    ssv2_classes_path = "ssv2_classes.json"

    ind, probs = get_vjepa_video_classification_results(classifier, out_patch_features_pt)
    return ind, probs


if __name__ == "__main__":
    # Run with: `python -m notebooks.vjepa2_demo`
    results = defaultdict(list)
    samples_per_action = 5
    actions = SOMETHING_SOMETHING_V2_CLASSES.values()
    probe_path = "/work/dlclarge2/aliy-vjepa/vjepa2/evals/video_classification_frozen/vitl-256/bddx/video_classification_frozen/bddx-vitl16-16x1-16f_updated_split/latest.pt"
    column_names = ["fname", "action", "start_time", "end_time"]
    val = pd.read_csv("notebooks/bddx_val_split.csv",  header=None, delimiter=" ", names = column_names)
    five_samples = val.groupby('action').sample(n=samples_per_action)
    
    rows = len(actions)
    cols = samples_per_action
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.tight_layout()
    for idx, (i, row) in enumerate(five_samples.iterrows()):
        r = row["action"]
        c = idx % cols
        ax = axs[r, c]
        ax.axis('off')
        base_name = os.path.splitext(os.path.basename(row["fname"]))[0]
        gif_path = os.path.join(GIF_DIR,f"{base_name}.gif")
        
        ind, probs = run_sample_inference(probe_path=probe_path, fname=row["fname"], start_time=row["start_time"], end_time=row["end_time"])
        
        gif_img = mpimg.imread(gif_path)
        ax.imshow(gif_img)
        ax.set_title(f"{SOMETHING_SOMETHING_V2_CLASSES[str(row["action"])]},{base_name}", fontsize=10, pad=5)
        results["gif_path"].append(gif_path)
        results["true_label"].append(SOMETHING_SOMETHING_V2_CLASSES[str(row["action"])])
        results["start_time"].append(row["start_time"])
        results["end_time"].append(row["end_time"])
        probs_text = "\n".join([
            f"{SOMETHING_SOMETHING_V2_CLASSES[str(int(i))]} ({p:.1f}%)"
            for i, p in zip(ind, probs)
            ])
        results["preds"].append(probs_text)

        ax.text(
            0.5, -0.05, probs_text,
            fontsize=7,
            ha='center',
            va='top',
            transform=ax.transAxes,
            color="black"
        )
        
    with open("probe_results_5samples_second.json", "w") as f:
        json.dump(results, f, indent=4)
        
    plt.subplots_adjust(hspace=0.8)  # Increase vertical space

        
    plt.savefig("bddx_probe_grid_2.png", dpi=200)
        
