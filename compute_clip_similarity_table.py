#!/usr/bin/env python3
"""
Render 10 pickup frames (one per ingredient) and compute the 10×10
CLIP similarity matrix between images and "picking up X" text prompts.

Outputs:
  output/clip_tables/raw_similarity.txt     – cosine similarities
  output/clip_tables/normalized_similarity.txt – row-wise softmax probabilities
  output/clip_tables/pickup_frames/          – the 10 rendered frames
"""

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_overcooked_src = os.path.join(_this_dir, "overcooked_ai", "src")
_miniclips_dir = os.path.join(_this_dir, "miniclips_python")
for p in [_this_dir, _overcooked_src, _miniclips_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import pygame
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

STAY = Action.STAY
INTERACT = Action.INTERACT
N, S, E, W = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST

INGREDIENTS = [
    "onion", "tomato", "cucumber", "rice", "olive",
    "feta_cheese", "hamburger_bun", "soy_sauce",
    "frozen_peas", "frozen_carrots",
]

INGREDIENT_DISPLAY = {
    "onion": "onion",
    "tomato": "tomato",
    "cucumber": "cucumber",
    "rice": "rice",
    "olive": "olive",
    "feta_cheese": "feta cheese",
    "hamburger_bun": "hamburger bun",
    "soy_sauce": "soy sauce",
    "frozen_peas": "frozen peas",
    "frozen_carrots": "frozen carrots",
}

DISPENSER_ACCESS = {
    "onion":          ((1, 1), N),
    "tomato":         ((3, 1), N),
    "cucumber":       ((5, 1), N),
    "rice":           ((7, 1), N),
    "olive":          ((1, 1), W),
    "feta_cheese":    ((1, 3), W),
    "hamburger_bun":  ((9, 1), E),
    "soy_sauce":      ((9, 3), E),
    "frozen_peas":    ((3, 3), S),
    "frozen_carrots": ((5, 3), S),
}

P1_START = (2, 2)


def move_to(start, end):
    sx, sy = start
    ex, ey = end
    actions = []
    h_dir = E if ex > sx else W
    v_dir = S if ey > sy else N
    for _ in range(abs(ex - sx)):
        actions.append(h_dir)
    for _ in range(abs(ey - sy)):
        actions.append(v_dir)
    return actions


def pygame_surface_to_pil(surface):
    raw = pygame.image.tostring(surface, "RGB")
    size = surface.get_size()
    return Image.frombytes("RGB", size, raw)


def extract_features(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unexpected output type: {type(output)}")


def render_pickup_frames():
    """For each ingredient, simulate player walking to dispenser and picking up.
    Return dict: ingredient -> PIL Image of the frame right after pickup.

    Returns both the full frame and a cropped version centered on the player
    (±2 tiles) for better CLIP discrimination.
    """

    print("Setting up Overcooked environment...")
    mdp = OvercookedGridworld.from_layout_name("four_goals", cook_time=5)
    viz = StateVisualizer(tile_size=75, is_rendering_hud=False)
    grid = mdp.terrain_mtx
    tile_size = 75

    full_frames = {}
    cropped_frames = {}
    for ing in INGREDIENTS:
        # Fresh state each time so the player has empty hands
        state = mdp.get_standard_start_state()
        cur = P1_START

        access_pos, face_dir = DISPENSER_ACCESS[ing]
        actions = move_to(cur, access_pos)
        actions.append(face_dir)
        actions.append(INTERACT)  # pickup

        for act in actions:
            joint = (act, STAY)
            state, _ = mdp.get_state_transition(state, joint)

        # Render the frame after pickup (player is holding the ingredient)
        surface = viz.render_state(state, grid)
        full_img = pygame_surface_to_pil(surface)
        full_frames[ing] = full_img

        # Crop around the player (±2 tiles, clamped to image bounds)
        px, py = access_pos
        margin = 2
        x1 = max(0, (px - margin) * tile_size)
        y1 = max(0, (py - margin) * tile_size)
        x2 = min(full_img.width, (px + margin + 1) * tile_size)
        y2 = min(full_img.height, (py + margin + 1) * tile_size)
        cropped = full_img.crop((x1, y1, x2, y2))
        cropped_frames[ing] = cropped

        print(f"  Rendered: pick_up({ing}) at ({px},{py}), crop=({x1},{y1},{x2},{y2}) -> {cropped.size}")

    return full_frames, cropped_frames


def compute_similarity_matrix(frames_dir, model_name="openai/clip-vit-base-patch32"):
    """Compute 10×10 cosine similarity matrix.
    Rows = images (one per ingredient pickup), Cols = text prompts ("picking up X").
    Also returns the model's logit_scale for proper softmax."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading CLIP model: {model_name} (device={device})...")
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    logit_scale = model.logit_scale.exp().item()
    print(f"CLIP model loaded. logit_scale={logit_scale:.2f}")

    # Build text prompts (pickup only, one per ingredient)
    prompts = []
    for ing in INGREDIENTS:
        display = INGREDIENT_DISPLAY[ing]
        prompts.append(
            f"The blue hat chef is picking up {display} from the {display} dispenser"
        )

    # Encode texts
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_emb = extract_features(model.get_text_features(**text_inputs))
        text_emb = F.normalize(text_emb, dim=-1)  # (10, D)

    # Encode images one by one (load from disk to avoid buffer reuse)
    img_embeds = []
    for ing in INGREDIENTS:
        img = Image.open(os.path.join(frames_dir, f"pickup_{ing}.png")).convert("RGB")
        img_inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = extract_features(model.get_image_features(**img_inputs))
            emb = F.normalize(emb, dim=-1)
        img_embeds.append(emb)
    img_emb = torch.cat(img_embeds, dim=0)  # (10, D)

    # Cosine similarity matrix
    sim_matrix = (img_emb @ text_emb.T).cpu().numpy()  # (10, 10)

    return sim_matrix, logit_scale


def softmax_rows(matrix, temperature=1.0):
    """Apply softmax to each row (with temperature)."""
    scaled = matrix / temperature
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def format_table(matrix, title, row_labels, col_labels, fmt=".4f"):
    """Format a matrix as a readable text table."""
    # Header
    col_width = max(max(len(c) for c in col_labels), 8)
    row_label_width = max(len(r) for r in row_labels) + 2

    lines = [title, "=" * len(title), ""]

    # Description row
    header_label = "Image \\ Text"
    lines.append(f"{header_label:<{row_label_width}}" +
                 "".join(f"{c:>{col_width + 2}}" for c in col_labels))
    lines.append("-" * (row_label_width + (col_width + 2) * len(col_labels)))

    for i, row_label in enumerate(row_labels):
        vals = "".join(f"{matrix[i, j]:{col_width + 2}{fmt}}" for j in range(matrix.shape[1]))
        lines.append(f"{row_label:<{row_label_width}}{vals}")

    lines.append("")
    return "\n".join(lines)


def main():
    out_dir = os.path.join(_this_dir, "output", "clip_tables")
    frames_dir = os.path.join(out_dir, "pickup_frames")
    cropped_dir = os.path.join(out_dir, "pickup_frames_cropped")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    # Step 1: Render pickup frames
    print("=== Step 1: Rendering pickup frames ===")
    full_frames, cropped_frames = render_pickup_frames()

    # Save frames
    for ing in INGREDIENTS:
        full_frames[ing].save(os.path.join(frames_dir, f"pickup_{ing}.png"))
        cropped_frames[ing].save(os.path.join(cropped_dir, f"pickup_{ing}.png"))
    print(f"  Saved full frames to {frames_dir}")
    print(f"  Saved cropped frames to {cropped_dir}")

    # Step 2: Compute similarity matrices (full and cropped)
    print("\n=== Step 2: Computing CLIP similarity matrices ===")
    print("--- Full frames ---")
    sim_full, logit_scale = compute_similarity_matrix(frames_dir)
    print("--- Cropped frames (around player) ---")
    sim_cropped, _ = compute_similarity_matrix(cropped_dir)

    # Step 3: Compute row-normalized versions using CLIP's logit_scale
    # The logit_scale acts as 1/temperature (typically ~100)
    print(f"\nUsing CLIP logit_scale = {logit_scale:.2f} (effective temperature = {1/logit_scale:.4f})")
    norm_full = softmax_rows(sim_full * logit_scale, temperature=1.0)
    norm_cropped = softmax_rows(sim_cropped * logit_scale, temperature=1.0)

    # Labels
    labels = [INGREDIENT_DISPLAY[ing] for ing in INGREDIENTS]

    # Step 4: Format and save tables
    print("\n=== Step 3: Saving tables ===")

    raw_table = format_table(
        sim_cropped,
        "CLIP Cosine Similarity: Image (row) vs Text Prompt (column)",
        labels, labels, fmt=".4f"
    )
    raw_table += "\nText prompts: \"The blue hat chef is picking up {ingredient} from the {ingredient} dispenser\"\n"
    raw_table += "Images: Cropped game frame (±2 tiles around player) after picking up {ingredient}\n"
    raw_table += "Model: openai/clip-vit-base-patch32\n"

    norm_table = format_table(
        norm_cropped,
        "Row-Normalized (Softmax) Probabilities: P(text | image)",
        labels, labels, fmt=".4f"
    )
    norm_table += f"\nEach row is softmax(logit_scale * cosine_similarity) with logit_scale={logit_scale:.2f}\n"
    norm_table += "Rows sum to 1.0. Higher diagonal values indicate CLIP correctly identifies the ingredient.\n"

    raw_path = os.path.join(out_dir, "raw_similarity.txt")
    norm_path = os.path.join(out_dir, "normalized_similarity.txt")

    with open(raw_path, "w") as f:
        f.write(raw_table)
    with open(norm_path, "w") as f:
        f.write(norm_table)

    print(f"  Raw similarities:  {raw_path}")
    print(f"  Normalized probs:  {norm_path}")

    # Print to console too
    print(f"\n{raw_table}")
    print(f"\n{norm_table}")

    # Also save the full-frame tables for comparison
    raw_full_table = format_table(
        sim_full,
        "CLIP Cosine Similarity (FULL FRAME): Image (row) vs Text Prompt (column)",
        labels, labels, fmt=".4f"
    )
    norm_full_table = format_table(
        norm_full,
        "Row-Normalized (FULL FRAME): P(text | image)",
        labels, labels, fmt=".4f"
    )
    with open(os.path.join(out_dir, "raw_similarity_full_frame.txt"), "w") as f:
        f.write(raw_full_table)
    with open(os.path.join(out_dir, "normalized_similarity_full_frame.txt"), "w") as f:
        f.write(norm_full_table)

    # Print summary stats
    print("=== Summary (cropped frames) ===")
    diag = np.diag(norm_cropped)
    print(f"Diagonal (correct match) probabilities:")
    for i, ing in enumerate(INGREDIENTS):
        print(f"  {INGREDIENT_DISPLAY[ing]:>15s}: {diag[i]:.4f}")
    print(f"  Mean diagonal: {diag.mean():.4f}")
    print(f"  Uniform baseline: {1/len(INGREDIENTS):.4f}")


if __name__ == "__main__":
    main()
