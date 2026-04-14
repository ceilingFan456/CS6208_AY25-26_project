#!/usr/bin/env python3
"""
Generate a side-by-side video:
  Left:  Overcooked game with one agent making "fried rice soup" (onion+cucumber+rice)
  Right: Goal posterior probability curves updated by Qwen3-VL vision model

The observer agent uses Qwen3-VL to look at the last 10 rendered frames
and estimate which recipe the acting player is pursuing.

Usage:
    python render_inference_video.py                    # symbolic-only (fast, no GPU)
    python render_inference_video.py --use-qwen         # with Qwen3-VL estimation
    python render_inference_video.py --use-qwen --model Qwen/Qwen3-VL-2B-Instruct
"""

import argparse
import math
import os
import re
import sys
import tempfile

# ── path setup ──
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir) if os.path.basename(_this_dir) == "miniclips_python" else _this_dir
_overcooked_src = os.path.join(_project_dir, "overcooked_ai", "src")
for p in [_this_dir, _project_dir, _overcooked_src]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PIL import Image

import pygame
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# ── Constants ──
STAY = Action.STAY
INTERACT = Action.INTERACT
N = Direction.NORTH
S = Direction.SOUTH
E = Direction.EAST
W = Direction.WEST

# The two candidate recipes in mixed_ingredients layout
RECIPE_NAMES = ["Fried Rice Soup\n(onion+cucumber+rice)", "Tomato Cucumber Soup\n(tomato×2+cucumber)"]
RECIPE_SHORT = ["Fried Rice Soup", "Tomato Cucumber Soup"]
RECIPE_COLORS = ["#2196F3", "#F44336"]  # blue, red

# ── Layout: mixed_ingredients ──
# Grid:
#   Row 0: X X P X X        P=(2,0) pot
#   Row 1: O _ _ 2 C        O=(0,1) onion, C=(4,1) cucumber, P2=(3,1)
#   Row 2: R 1 _ _ T        R=(0,2) rice, T=(4,2) tomato, P1=(1,2)
#   Row 3: X D X S X        D=(1,3) dish, S=(3,3) serve


def build_p1_action_sequence():
    """Scripted actions for Player 1 to make fried rice soup (onion+cucumber+rice)."""
    actions = []

    # -- Pick up ONION from O(0,1) --
    actions.append(N)          # (1,2) → (1,1)
    actions.append(W)          # (1,1) → stay, face W [O at (0,1)]
    actions.append(INTERACT)   # pick up onion

    # -- Add onion to pot P(2,0) --
    actions.append(E)          # (1,1) → (2,1)
    actions.append(N)          # (2,1) → stay, face N [P at (2,0)]
    actions.append(INTERACT)   # drop onion in pot

    # -- Pick up CUCUMBER from C(4,1) --
    actions.append(E)          # (2,1) → (3,1)  [P2 already moved to (3,2)]
    actions.append(E)          # (3,1) → stay, face E [C at (4,1)]
    actions.append(INTERACT)   # pick up cucumber

    # -- Add cucumber to pot --
    actions.append(W)          # (3,1) → (2,1)
    actions.append(N)          # (2,1) → stay, face N
    actions.append(INTERACT)   # drop cucumber in pot

    # -- Pick up RICE from R(0,2) --
    actions.append(S)          # (2,1) → (2,2)
    actions.append(W)          # (2,2) → (1,2)
    actions.append(W)          # (1,2) → stay, face W [R at (0,2)]
    actions.append(INTERACT)   # pick up rice

    # -- Add rice to pot (pot becomes full→3 ingredients) --
    actions.append(N)          # (1,2) → (1,1)
    actions.append(E)          # (1,1) → (2,1)
    actions.append(N)          # (2,1) → stay, face N
    actions.append(INTERACT)   # drop rice in pot

    # -- Start cooking --
    actions.append(INTERACT)   # interact with pot → start cooking

    # -- Wait for cooking (cook_time=5) --
    for _ in range(5):
        actions.append(STAY)

    # -- Pick up DISH from D(1,3) --
    actions.append(S)          # (2,1) → (2,2)
    actions.append(W)          # (2,2) → (1,2)
    actions.append(S)          # (1,2) → stay, face S [D at (1,3)]
    actions.append(INTERACT)   # pick up dish

    # -- Scoop soup from pot --
    actions.append(N)          # (1,2) → (1,1)
    actions.append(E)          # (1,1) → (2,1)
    actions.append(N)          # (2,1) → stay, face N
    actions.append(INTERACT)   # scoop soup

    # -- Go to serve at S(3,3) --
    actions.append(E)          # (2,1) → (3,1)
    actions.append(S)          # (3,1) → (3,2)  [P2 already moved to (2,2)]
    actions.append(S)          # (3,2) → stay, face S [S at (3,3)]
    actions.append(INTERACT)   # serve!

    # -- End padding --
    for _ in range(5):
        actions.append(STAY)

    return actions


def build_p2_action_sequence(total_steps):
    """Player 2 actions: move out of the way, stay idle.

    P2 starts at (3,1). Moves to (3,2) immediately, then to (2,2) before
    P1 needs (3,2) for serving.
    """
    actions = [None] * total_steps
    # Step 0: move south to (3,2)
    actions[0] = S
    # At step 34 (before P1 needs to go through (3,2) for serving), move west
    # P1 reaches serve path starting at around step 32-35
    serve_start = 32  # P1 starts heading to serve around this step
    if serve_start < total_steps:
        actions[serve_start] = W   # (3,2) → (2,2)

    # Fill remaining with STAY
    for i in range(total_steps):
        if actions[i] is None:
            actions[i] = STAY
    return actions


def symbolic_action_from_events(events_infos, player_idx):
    """Extract a human-readable action description from event_infos."""
    ei = events_infos
    pi = player_idx

    if ei.get("soup_delivery", [False, False])[pi]:
        return "SERVE SOUP"
    if ei.get("soup_pickup", [False, False])[pi]:
        return "PICK UP SOUP"
    for ing in ["onion", "tomato", "cucumber", "rice"]:
        if ei.get(f"potting_{ing}", [False, False])[pi]:
            return f"ADD {ing.upper()} TO POT"
    if ei.get("dish_pickup", [False, False])[pi]:
        return "PICK UP DISH"
    for ing in ["onion", "tomato", "cucumber", "rice"]:
        if ei.get(f"{ing}_pickup", [False, False])[pi]:
            return f"PICK UP {ing.upper()}"
    return None


# ══════════════════════════════════════════════════════════════
#  Qwen3-VL based estimation
# ══════════════════════════════════════════════════════════════

QWEN_PROMPT = """\
You are watching a cooperative cooking game from a top-down view. \
Two chefs (blue hat and green hat) are in a kitchen. \
The blue-hat chef (Player 1) is trying to make one of two possible recipes:

Recipe A - "Fried Rice Soup": requires 1 onion, 1 cucumber, and 1 rice.
Recipe B - "Tomato Cucumber Soup": requires 2 tomatoes and 1 cucumber.

The kitchen has dispensers on the walls:
- Left side row 1: Onion dispenser
- Right side row 1: Cucumber dispenser
- Left side row 2: Rice dispenser
- Right side row 2: Tomato dispenser
- Top center: Cooking pot
- Bottom: Dish dispenser and Serving window

I'm showing you the last few frames of the game in chronological order. \
Based on what ingredients the blue-hat player has been picking up and adding to the pot, \
estimate the probability that they are making each recipe.

Answer ONLY in this exact format (numbers must sum to 1.0):
Recipe A: <probability>
Recipe B: <probability>"""


def load_qwen_vl_model(model_name="Qwen/Qwen3-VL-2B-Instruct"):
    """Load Qwen3-VL model and processor."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"Loading Qwen3-VL model: {model_name} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print("Model loaded.")
    return model, processor


def query_qwen_vl(model, processor, frame_images, prompt=QWEN_PROMPT):
    """Query Qwen3-VL with a list of PIL images and return raw text response."""
    import torch

    content = []
    for img in frame_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def parse_qwen_response(response_text):
    """Parse probability response from Qwen3-VL.

    Expected format:
      Recipe A: 0.7
      Recipe B: 0.3
    Returns [prob_A, prob_B] or None on failure.
    """
    probs = []
    for label in ["Recipe A", "Recipe B"]:
        pattern = rf"{label}\s*[:=]\s*([0-9]*\.?[0-9]+)"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            probs.append(float(match.group(1)))
        else:
            return None

    total = sum(probs)
    if total <= 0:
        return None
    # Normalize
    return [p / total for p in probs]


# ══════════════════════════════════════════════════════════════
#  Symbolic likelihood (no Qwen, uses action-based Bayesian)
# ══════════════════════════════════════════════════════════════

def symbolic_likelihood_update(action_str, prior, noise=0.05):
    """Update posterior based on observed symbolic action.

    Recipe 0 (Fried Rice): onion, cucumber, rice
    Recipe 1 (Tomato Cucumber): tomato, tomato, cucumber
    """
    n_goals = len(prior)
    likelihood = np.ones(n_goals)

    if action_str is None:
        return prior  # no informative action

    action_upper = action_str.upper()

    # Ingredient-based evidence
    if "ONION" in action_upper:
        likelihood[0] = 1.0 - noise   # fried rice has onion
        likelihood[1] = noise          # tomato soup doesn't
    elif "RICE" in action_upper:
        likelihood[0] = 1.0 - noise
        likelihood[1] = noise
    elif "TOMATO" in action_upper:
        likelihood[0] = noise
        likelihood[1] = 1.0 - noise
    elif "CUCUMBER" in action_upper:
        likelihood[0] = 0.5   # both recipes have cucumber
        likelihood[1] = 0.5
    else:
        return prior  # non-informative (dish, serve, etc.)

    unnorm = prior * likelihood
    total = unnorm.sum()
    if total <= 0:
        return prior
    return unnorm / total


# ══════════════════════════════════════════════════════════════
#  Video composition
# ══════════════════════════════════════════════════════════════

def render_posterior_plot(timesteps, posterior_history, current_step,
                          width=500, height=375, dpi=100):
    """Render the posterior probability curves as a PIL image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    t_range = list(range(len(posterior_history)))

    for goal_idx in range(len(RECIPE_SHORT)):
        probs = [ph[goal_idx] for ph in posterior_history]
        ax.plot(t_range, probs, color=RECIPE_COLORS[goal_idx],
                linewidth=2.5, label=RECIPE_SHORT[goal_idx])

    ax.set_xlim(0, max(timesteps[-1], 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("P(goal)", fontsize=12)
    ax.set_title("Goal Posterior (Observer Agent)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)

    # Mark current step
    if current_step < len(posterior_history):
        for goal_idx in range(len(RECIPE_SHORT)):
            y = posterior_history[current_step][goal_idx]
            ax.plot(current_step, y, "o", color=RECIPE_COLORS[goal_idx],
                    markersize=8, zorder=5)

    fig.tight_layout()

    # Convert to PIL
    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA → RGB
    plt.close(fig)
    return Image.fromarray(buf)


def compose_frame(game_img, plot_img, action_label=None, step=0):
    """Compose the side-by-side frame: game (left) + plot (right)."""
    # Resize game image to match plot height
    target_h = plot_img.height
    scale = target_h / game_img.height
    new_w = int(game_img.width * scale)
    game_resized = game_img.resize((new_w, target_h), Image.NEAREST)

    # Create combined image
    total_w = game_resized.width + plot_img.width
    combined = Image.new("RGB", (total_w, target_h), (255, 255, 255))
    combined.paste(game_resized, (0, 0))
    combined.paste(plot_img, (game_resized.width, 0))

    # Add action label at bottom of game area
    if action_label:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()

        text = f"Step {step}: {action_label}"
        # Draw text background
        bbox = draw.textbbox((10, target_h - 30), text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                       fill=(0, 0, 0, 180))
        draw.text((10, target_h - 30), text, fill=(255, 255, 0), font=font)

    return combined


def pygame_surface_to_pil(surface):
    """Convert a pygame Surface to a PIL Image."""
    raw = pygame.image.tostring(surface, "RGB")
    size = surface.get_size()
    return Image.frombytes("RGB", size, raw)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Render inference video")
    parser.add_argument("--use-qwen", action="store_true",
                        help="Use Qwen3-VL for goal likelihood estimation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Qwen3-VL model name")
    parser.add_argument("--qwen-interval", type=int, default=3,
                        help="Query Qwen every N steps (default: 3)")
    parser.add_argument("--fps", type=int, default=3,
                        help="Video frame rate (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: auto)")
    args = parser.parse_args()

    # Output paths
    if args.output:
        video_path = args.output
    else:
        suffix = "_qwen" if args.use_qwen else "_symbolic"
        video_path = os.path.join(_project_dir, "output", f"inference_video{suffix}.mp4")
    frames_dir = os.path.join(_project_dir, "output", "inference_video_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # ── Set up Overcooked ──
    print("Setting up Overcooked environment (mixed_ingredients layout)...")
    mdp = OvercookedGridworld.from_layout_name("mixed_ingredients", cook_time=5)
    state = mdp.get_standard_start_state()
    grid = mdp.terrain_mtx

    viz = StateVisualizer(tile_size=75, is_rendering_hud=False)

    # ── Build action sequences ──
    p1_actions = build_p1_action_sequence()
    total_steps = len(p1_actions)
    p2_actions = build_p2_action_sequence(total_steps)

    print(f"Total steps: {total_steps}")
    print(f"Recipes: {RECIPE_SHORT}")
    print(f"Agent 1 is making: {RECIPE_SHORT[0]} (fried rice soup)")

    # ── Load Qwen if requested ──
    qwen_model, qwen_processor = None, None
    if args.use_qwen:
        qwen_model, qwen_processor = load_qwen_vl_model(args.model)

    # ── Render initial frame ──
    game_frames_pil = []   # store PIL images for Qwen
    posterior_history = []  # list of [prob_recipe_A, prob_recipe_B]
    action_labels = []     # action description per step

    prior = np.array([0.5, 0.5])
    posterior = prior.copy()
    posterior_history.append(posterior.tolist())

    surface = viz.render_state(state, grid)
    init_img = pygame_surface_to_pil(surface)
    game_frames_pil.append(init_img)
    action_labels.append("Initial state")

    print("\n--- Simulating game ---")

    # ── Run game loop ──
    for step in range(total_steps):
        p1_action = p1_actions[step]
        p2_action = p2_actions[step]
        joint_action = (p1_action, p2_action)

        try:
            new_state, infos = mdp.get_state_transition(state, joint_action)
        except Exception as e:
            print(f"  Step {step}: ERROR - {e}")
            # Pad remaining with current state
            for _ in range(step, total_steps):
                posterior_history.append(posterior.tolist())
                game_frames_pil.append(game_frames_pil[-1])
                action_labels.append("ERROR")
            break

        state = new_state

        # Render game frame
        surface = viz.render_state(state, grid)
        frame_img = pygame_surface_to_pil(surface)
        game_frames_pil.append(frame_img)

        # Extract symbolic action
        events = infos.get("event_infos", {})
        action_str = symbolic_action_from_events(events, player_idx=0)
        action_labels.append(action_str or "moving...")

        # ── Update posterior ──
        if args.use_qwen and qwen_model is not None:
            # Use Qwen-VL every qwen_interval steps
            if step % args.qwen_interval == 0 or action_str is not None:
                # Get last 10 frames
                recent_frames = game_frames_pil[-10:]
                print(f"  Step {step:2d}: Querying Qwen-VL with {len(recent_frames)} frames...", end=" ")
                try:
                    response = query_qwen_vl(qwen_model, qwen_processor, recent_frames)
                    parsed = parse_qwen_response(response)
                    if parsed is not None:
                        posterior = np.array(parsed)
                        print(f"→ A:{parsed[0]:.2f} B:{parsed[1]:.2f}")
                    else:
                        # Fallback to symbolic if parsing fails
                        posterior = symbolic_likelihood_update(action_str, posterior)
                        print(f"→ parse failed, symbolic fallback. Raw: {response[:80]}")
                except Exception as e:
                    posterior = symbolic_likelihood_update(action_str, posterior)
                    print(f"→ Qwen error: {e}, symbolic fallback")
            else:
                # Keep previous posterior between Qwen queries
                pass
        else:
            # Pure symbolic update
            posterior = symbolic_likelihood_update(action_str, posterior)

        posterior_history.append(posterior.tolist())

        # Log
        sparse_rew = sum(infos.get("sparse_reward_by_agent", [0, 0]))
        log_parts = [f"  Step {step:2d}: {action_labels[-1]:25s}"]
        log_parts.append(f"P(FR)={posterior[0]:.3f} P(TC)={posterior[1]:.3f}")
        if sparse_rew > 0:
            log_parts.append(f"REWARD={sparse_rew}")
        print(" | ".join(log_parts))

    # ── Compose video frames ──
    print(f"\n--- Composing {len(game_frames_pil)} frames ---")
    all_timesteps = list(range(len(posterior_history)))
    composed_frames = []

    for i in range(len(game_frames_pil)):
        plot_img = render_posterior_plot(
            all_timesteps, posterior_history[:i + 1], i
        )
        label = action_labels[i] if i < len(action_labels) else ""
        composed = compose_frame(game_frames_pil[i], plot_img, label, i)
        composed_frames.append(composed)

        # Save individual frame
        composed.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))

    # ── Write video ──
    print(f"\nWriting video to {video_path} ...")
    try:
        import imageio.v3 as iio
        frame_arrays = [np.array(f) for f in composed_frames]
        iio.imwrite(video_path, frame_arrays, fps=args.fps, codec="libx264")
        print(f"Video saved: {video_path}")
    except Exception as e:
        print(f"Video encoding failed ({e}), trying GIF...")
        gif_path = video_path.replace(".mp4", ".gif")
        try:
            composed_frames[0].save(
                gif_path, save_all=True, append_images=composed_frames[1:],
                duration=int(1000 / args.fps), loop=0,
            )
            print(f"GIF saved: {gif_path}")
        except Exception as e2:
            print(f"GIF also failed: {e2}")
            print(f"Individual frames are in: {frames_dir}/")

    # ── Also write a GIF ──
    gif_path = video_path.replace(".mp4", ".gif")
    try:
        composed_frames[0].save(
            gif_path, save_all=True, append_images=composed_frames[1:],
            duration=int(1000 / args.fps), loop=0,
        )
        print(f"GIF also saved: {gif_path}")
    except Exception:
        pass

    print("\nDone!")


if __name__ == "__main__":
    main()
