import os
import json
import imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_caption_image(caption, width, font_size=20, bold_label=None):
    """Create a caption image with optional bold label highlight."""
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    lines = caption.split("\n")
    spacing = 5
    line_height = font.getbbox("A")[3] + spacing
    height = line_height * len(lines)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for i, line in enumerate(lines):
        if bold_label and bold_label in line:
            pre, sep, post = line.partition(bold_label)
            draw.text((5, i * line_height), pre, fill="black", font=font)
            w_pre = font.getbbox(pre)[2]
            draw.text((5 + w_pre, i * line_height), bold_label, fill="red", font=font)
            w_bold = font.getbbox(bold_label)[2]
            draw.text((5 + w_pre + w_bold, i * line_height), post, fill="black", font=font)
        else:
            draw.text((5, i * line_height), line, fill="black", font=font)

    return image


def load_gif_frames(path, resize=(160, 160)):
    """Read all frames from a GIF and resize them."""
    reader = imageio.get_reader(path, memtest=False)  # avoids memory check errors
    frames = [Image.fromarray(frame).resize(resize, Image.BILINEAR) for frame in reader]
    reader.close()
    return frames

def compose_gif_grid(gif_paths, true_labels, preds, output_path, cols=5):
    assert len(gif_paths) == len(true_labels) == len(preds)
    rows = len(gif_paths) // cols
    all_grid_frames = []

    print(f"Composing grid with {rows} rows × {cols} cols")

    gifs = [load_gif_frames(p, resize=(160, 160)) for p in gif_paths]
    max_frames = max(len(g) for g in gifs)  # allow max length

    width, height = gifs[0][0].size
    print(f"GIF frame size: {width}x{height}")

    margin_x, margin_y = 10, 10  # margin around each cell

    for frame_idx in range(max_frames):
        row_images = []
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                idx = r * cols + c
                gif = gifs[idx]
                # Repeat last frame if GIF is shorter than max_frames
                gif_frame = gif[frame_idx] if frame_idx < len(gif) else gif[-1]

                base = os.path.splitext(os.path.basename(gif_paths[idx]))[0]
                if len(base) > 18:
                    base = base[:17] + "…"
                top_caption = f"{base}\nTrue: {true_labels[idx]}"
                top_img = create_caption_image(top_caption, width, font_size=18, bold_label=true_labels[idx])
                bottom_img = create_caption_image(preds[idx], width, font_size=14)

                # Stack vertically
                stacked = Image.new("RGB", (width, top_img.height + height + bottom_img.height), "white")
                stacked.paste(top_img, (0, 0))
                stacked.paste(gif_frame, (0, top_img.height))
                stacked.paste(bottom_img, (0, top_img.height + height))

                # Padding around each cell
                padded = Image.new("RGB", (stacked.width + 2 * margin_x, stacked.height + 2 * margin_y), "white")
                padded.paste(stacked, (margin_x, margin_y))
                row_cells.append(np.array(padded))

            # Combine cells horizontally in a row
            row_images.append(np.hstack(row_cells))

        # Stack all rows vertically for this frame
        grid_frame = np.vstack(row_images)
        all_grid_frames.append(Image.fromarray(grid_frame))

    # Save final animated GIF
    all_grid_frames[0].save(
        output_path,
        save_all=True,
        append_images=all_grid_frames[1:],
        duration=100,
        loop=0
    )
    print(f"Saved grid GIF to {output_path}")


if __name__ == "__main__":
    with open("/work/dlclarge2/aliy-vjepa/vjepa2/probe_results_5samples_second.json", "r") as f:
        data = json.load(f)

    compose_gif_grid(
        gif_paths=data["gif_path"],
        true_labels=data["true_label"],
        preds=data["preds"],
        output_path="grid_output_second.gif",
        cols=5
    )