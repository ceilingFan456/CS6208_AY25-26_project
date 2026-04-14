#!/usr/bin/env python3
"""
Four-goal Overcooked experiment with dual posterior plots.

Left panel:   Overcooked game (Player 1 makes Fried Rice Soup)
Middle panel: Symbolic action–based Bayesian posterior
Right panel:  Qwen3-VL vision–based posterior

Layout: four_goals (11×5 grid, all 10 ingredient dispensers)
  Row 0: X O X T X C X R X P X    (top dispensers + pot)
  Row 1: V                   B    (olive / hamburger_bun)
  Row 2: X  P1          P2   X
  Row 3: E                   Y    (feta_cheese / soy_sauce)
  Row 4: X D X Z X G X X X S X    (dish, frozen_peas, frozen_carrots, serve)

Four candidate recipes (5 ingredients each, adapted from miniclip goals):
  A – Greek Salad Soup:    tomato, olive, cucumber, onion, feta_cheese
  B – Veggie Burger Soup:  hamburger_bun, tomato, onion, cucumber, frozen_carrots
  C – Fried Rice Soup:     rice, onion, soy_sauce, frozen_peas, frozen_carrots  ← Player 1 makes this
  D – Burrito Bowl Soup:   rice, olive, feta_cheese, onion, tomato

Usage:
    python render_four_goals_video.py                     # symbolic only (no GPU)
    python render_four_goals_video.py --use-qwen          # + Qwen3-VL vision
    python render_four_goals_video.py --use-qwen --model Qwen/Qwen3-VL-4B-Instruct
"""

import argparse
import math
import os
import re
import sys

# ── path setup ──
_this_dir = os.path.dirname(os.path.abspath(__file__))
_overcooked_src = os.path.join(_this_dir, "overcooked_ai", "src")
_miniclips_dir = os.path.join(_this_dir, "miniclips_python")
for p in [_this_dir, _overcooked_src, _miniclips_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PIL import Image

import pygame
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from bayesian_inference import posterior_update
from overcooked_goal_model import build_overcooked_goals
from overcooked_state_parser import extract_action_from_events

# ── Constants ──
STAY = Action.STAY
INTERACT = Action.INTERACT
N = Direction.NORTH
S = Direction.SOUTH
E = Direction.EAST
W = Direction.WEST

# The four candidate recipes
RECIPE_NAMES = [
    "Greek Salad Soup",
    "Veggie Burger Soup",
    "Fried Rice Soup",
    "Burrito Bowl Soup",
]
RECIPE_INGREDIENTS = [
    ["tomato", "olive", "cucumber", "onion", "feta_cheese"],          # greek_salad
    ["hamburger_bun", "tomato", "onion", "cucumber", "frozen_carrots"],# veggie_burger (lettuce→cucumber, frozen_patty→frozen_carrots)
    ["rice", "onion", "soy_sauce", "frozen_peas", "frozen_carrots"],   # fried_rice (all original)
    ["rice", "olive", "feta_cheese", "onion", "tomato"],              # burrito_bowl (black_beans→olive, cotija→feta_cheese)
]
RECIPE_COLORS = ["#4CAF50", "#FF9800", "#2196F3", "#F44336"]  # green, orange, blue, red

GROUND_TRUTH_IDX = 2  # Player 1 is making Fried Rice Soup

# ── Dispenser positions in the four_goals layout ──
# Row 0: X(0,0) O(1,0) X(2,0) T(3,0) X(4,0) C(5,0) X(6,0) R(7,0) X(8,0) P(9,0) X(10,0)
# Row 1: V(0,1)                                                                    B(10,1)
# Row 2: X(0,2)  P1(2,2)                                             P2(8,2)       X(10,2)
# Row 3: E(0,3)                                                                    Y(10,3)
# Row 4: X(0,4) D(1,4) X(2,4) Z(3,4) X(4,4) G(5,4) X(6,4) X(7,4) X(8,4) S(9,4) X(10,4)

# Access positions and face directions for dispensers
DISPENSER_ACCESS = {
    "onion":          ((1, 1), N),    # O at (1,0)
    "tomato":         ((3, 1), N),    # T at (3,0)
    "cucumber":       ((5, 1), N),    # C at (5,0)
    "rice":           ((7, 1), N),    # R at (7,0)
    "olive":          ((1, 1), W),    # V at (0,1)
    "feta_cheese":    ((1, 3), W),    # E at (0,3)
    "hamburger_bun":  ((9, 1), E),    # B at (10,1)
    "soy_sauce":      ((9, 3), E),    # Y at (10,3)
    "frozen_peas":    ((3, 3), S),    # Z at (3,4)
    "frozen_carrots": ((5, 3), S),    # G at (5,4)
}
POT_ACCESS = ((9, 1), N)       # P at (9,0)
DISH_ACCESS = ((1, 3), S)      # D at (1,4)
SERVE_ACCESS = ((9, 3), S)     # S at (9,4)

P1_START = (2, 2)


def move_to(start, end):
    """Generate action list to move from start to end (horizontal first)."""
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


def build_p1_action_sequence():
    """Scripted actions for Player 1 to make Fried Rice Soup (rice, onion, soy_sauce, frozen_peas, frozen_carrots)."""
    ingredients_order = ["rice", "onion", "soy_sauce", "frozen_peas", "frozen_carrots"]
    actions = []
    cur = P1_START

    for ing in ingredients_order:
        access_pos, face_dir = DISPENSER_ACCESS[ing]
        # Move to dispenser
        actions.extend(move_to(cur, access_pos))
        cur = access_pos
        actions.append(face_dir)
        actions.append(INTERACT)  # pick up

        # Move to pot
        pot_pos, pot_face = POT_ACCESS
        actions.extend(move_to(cur, pot_pos))
        cur = pot_pos
        actions.append(pot_face)
        actions.append(INTERACT)  # add to pot

    # Start cooking (already at pot, facing N)
    actions.append(INTERACT)

    # Wait for cooking (cook_time=5)
    for _ in range(5):
        actions.append(STAY)

    # Pick up dish
    dish_pos, dish_face = DISH_ACCESS
    actions.extend(move_to(cur, dish_pos))
    cur = dish_pos
    actions.append(dish_face)
    actions.append(INTERACT)

    # Pick up soup from pot
    pot_pos, pot_face = POT_ACCESS
    actions.extend(move_to(cur, pot_pos))
    cur = pot_pos
    actions.append(pot_face)
    actions.append(INTERACT)

    # Serve
    serve_pos, serve_face = SERVE_ACCESS
    actions.extend(move_to(cur, serve_pos))
    cur = serve_pos
    actions.append(serve_face)
    actions.append(INTERACT)

    # Trailing padding
    for _ in range(5):
        actions.append(STAY)

    return actions


# ══════════════════════════════════════════════════════════════
#  Symbolic likelihood (ingredient-based Bayesian, same action
#  model as miniclip code)
# ══════════════════════════════════════════════════════════════

def symbolic_action_from_events(events_infos, player_idx):
    """Extract high-level action string from event_infos."""
    ei = events_infos
    pi = player_idx

    if ei.get("soup_delivery", [False, False])[pi]:
        return "serve_soup"
    if ei.get("soup_pickup", [False, False])[pi]:
        return "pick_up_soup"
    for ing in ["onion", "tomato", "cucumber", "rice", "olive",
                "feta_cheese", "hamburger_bun", "soy_sauce",
                "frozen_peas", "frozen_carrots"]:
        if ei.get(f"potting_{ing}", [False, False])[pi]:
            return f"add_to_pot({ing})"
    if ei.get("dish_pickup", [False, False])[pi]:
        return "pick_up_dish"
    for ing in ["onion", "tomato", "cucumber", "rice", "olive",
                "feta_cheese", "hamburger_bun", "soy_sauce",
                "frozen_peas", "frozen_carrots"]:
        if ei.get(f"{ing}_pickup", [False, False])[pi]:
            return f"pick_up({ing})"
    return None


def symbolic_likelihood_for_action(action_str, goals, plans, states, act_noise=0.05):
    """Compute P(action | goal) for each goal using the miniclip action model.

    Uses the same action model as overcooked_inference.py:
    planned actions get (1-noise)/|planned| + noise/|all|,
    unplanned actions get noise/|all|.
    """
    from overcooked_goal_model import OVERCOOKED_ACTIONS

    n_goals = len(goals)
    likelihood = np.ones(n_goals, dtype=float)

    if action_str is None:
        return likelihood  # non-informative

    n_actions = len(OVERCOOKED_ACTIONS)

    for i, goal in enumerate(goals):
        plan = plans[goal]
        state = states[goal]

        # Find planned (available) actions
        planned = []
        for act in plan:
            if act in state:
                continue
            if all(dep in state for dep in plan[act]):
                planned.append(act)
        if not planned:
            planned = ["wait"]

        # Match observed action to plan action (strip _N suffix)
        matched = None
        for plan_act in plan:
            if plan_act in state:
                continue
            base = plan_act.rsplit("_", 1)[0] if "_" in plan_act else plan_act
            if base == action_str or plan_act == action_str:
                matched = plan_act
                break

        if matched and matched in planned:
            likelihood[i] = (1.0 - act_noise) / len(planned) + act_noise / n_actions
        else:
            likelihood[i] = act_noise / n_actions

    likelihood = np.maximum(likelihood, 1e-300)
    return likelihood


def update_symbolic_states(action_str, goals, plans, states):
    """Update per-goal completed-action sets after an observed action."""
    for goal in goals:
        plan = plans[goal]
        for plan_act in plan:
            if plan_act in states[goal]:
                continue
            base = plan_act.rsplit("_", 1)[0] if "_" in plan_act else plan_act
            if base == action_str or plan_act == action_str:
                states[goal].add(plan_act)
                break


# ══════════════════════════════════════════════════════════════
#  Qwen3-VL vision likelihood
# ══════════════════════════════════════════════════════════════

QWEN_PROMPT = """\
You are watching a cooperative cooking game from a top-down view. \
Two chefs (blue hat and green hat) are in a kitchen with ingredient dispensers on the walls. \
The blue-hat chef (Player 1) is trying to make one of four possible soups. \
Each soup requires exactly 5 ingredients picked up from dispensers and added to a pot.

The four candidate goals and their recipes are:

Recipe A - "Greek Salad Soup":
  Ingredients: tomato, olive, cucumber, onion, feta cheese
  (Player picks from: Tomato dispenser on top wall, Olive dispenser on left wall, \
Cucumber dispenser on top wall, Onion dispenser on top wall, Feta Cheese dispenser on left wall)

Recipe B - "Veggie Burger Soup":
  Ingredients: hamburger bun, tomato, onion, cucumber, frozen carrots
  (Player picks from: Hamburger Bun dispenser on right wall, Tomato on top wall, \
Onion on top wall, Cucumber on top wall, Frozen Carrots on bottom wall)

Recipe C - "Fried Rice Soup":
  Ingredients: rice, onion, soy sauce, frozen peas, frozen carrots
  (Player picks from: Rice dispenser on top wall, Onion on top wall, \
Soy Sauce on right wall, Frozen Peas on bottom wall, Frozen Carrots on bottom wall)

Recipe D - "Burrito Bowl Soup":
  Ingredients: rice, olive, feta cheese, onion, tomato
  (Player picks from: Rice on top wall, Olive on left wall, \
Feta Cheese on left wall, Onion on top wall, Tomato on top wall)

Kitchen layout:
- Top wall (left to right): Onion (white bulb), Tomato (red), Cucumber (green), Rice (white grains)
- Left wall (top to bottom): Olive (purple/dark), Feta Cheese (yellow block)
- Right wall (top to bottom): Hamburger Bun (brown), Soy Sauce (dark bottle)
- Bottom wall: Dish dispenser, Frozen Peas (green bag), Frozen Carrots (orange bag), Serving window
- Top right corner: Cooking pot

I'm showing you the last several frames of the game in chronological order. \
Observe which dispensers the blue-hat player visits, which ingredients they pick up, \
and which items they add to the pot. Based on these observations, \
estimate the probability that they are making each recipe.

Answer ONLY in this exact format (numbers must sum to 1.0):
Recipe A: <probability>
Recipe B: <probability>
Recipe C: <probability>
Recipe D: <probability>"""


def load_qwen_vl_model(model_name="Qwen/Qwen3-VL-4B-Instruct"):
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


def parse_qwen_response(response_text, n_recipes=4):
    """Parse probability response from Qwen3-VL.

    Expected format:
      Recipe A: 0.3
      Recipe B: 0.2
      Recipe C: 0.4
      Recipe D: 0.1
    Returns list of probabilities, or None on failure.
    """
    labels = [chr(ord("A") + i) for i in range(n_recipes)]
    probs = []
    for label in labels:
        pattern = rf"Recipe {label}\s*[:=]\s*([0-9]*\.?[0-9]+)"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            probs.append(float(match.group(1)))
        else:
            return None

    total = sum(probs)
    if total <= 0:
        return None
    return [p / total for p in probs]


# ══════════════════════════════════════════════════════════════
#  Plotting & composition
# ══════════════════════════════════════════════════════════════

def render_posterior_plot(timesteps, posterior_history, current_step,
                         title, width=420, height=375, dpi=100):
    """Render a posterior probability curve plot as a PIL image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    t_range = list(range(len(posterior_history)))
    n_goals = len(RECIPE_NAMES)

    for goal_idx in range(n_goals):
        probs = [ph[goal_idx] for ph in posterior_history]
        ax.plot(t_range, probs, color=RECIPE_COLORS[goal_idx],
                linewidth=2.5, label=RECIPE_NAMES[goal_idx])

    ax.set_xlim(0, max(timesteps[-1], 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("P(goal)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.4)

    # Mark current step
    if current_step < len(posterior_history):
        for goal_idx in range(n_goals):
            y = posterior_history[current_step][goal_idx]
            ax.plot(current_step, y, "o", color=RECIPE_COLORS[goal_idx],
                    markersize=7, zorder=5)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(buf)


def compose_frame(game_img, sym_plot, qwen_plot, action_label=None, step=0):
    """Compose three-panel frame: game | symbolic plot | qwen plot."""
    target_h = sym_plot.height
    scale = target_h / game_img.height
    new_w = int(game_img.width * scale)
    game_resized = game_img.resize((new_w, target_h), Image.NEAREST)

    total_w = game_resized.width + sym_plot.width + qwen_plot.width
    combined = Image.new("RGB", (total_w, target_h), (255, 255, 255))
    combined.paste(game_resized, (0, 0))
    combined.paste(sym_plot, (game_resized.width, 0))
    combined.paste(qwen_plot, (game_resized.width + sym_plot.width, 0))

    if action_label:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        text = f"Step {step}: {action_label}"
        bbox = draw.textbbox((10, target_h - 28), text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                       fill=(0, 0, 0, 180))
        draw.text((10, target_h - 28), text, fill=(255, 255, 0), font=font)

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
    parser = argparse.ArgumentParser(description="Four-goal Overcooked inference video")
    parser.add_argument("--use-qwen", action="store_true",
                        help="Use Qwen3-VL for vision-based likelihood estimation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Qwen3-VL model name")
    parser.add_argument("--qwen-interval", type=int, default=3,
                        help="Query Qwen every N steps (default: 3)")
    parser.add_argument("--fps", type=int, default=3,
                        help="Video frame rate (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: auto)")
    args = parser.parse_args()

    # Output paths
    suffix = "_qwen" if args.use_qwen else "_symbolic"
    if args.output:
        video_path = args.output
    else:
        video_path = os.path.join(_this_dir, "output", f"four_goals_video{suffix}.mp4")
    frames_dir = os.path.join(_this_dir, "output", "four_goals_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # ── Set up Overcooked ──
    print("Setting up Overcooked environment (four_goals layout)...")
    mdp = OvercookedGridworld.from_layout_name("four_goals", cook_time=5)
    state = mdp.get_standard_start_state()
    grid = mdp.terrain_mtx

    viz = StateVisualizer(tile_size=75, is_rendering_hud=False)

    # ── Build goals/plans using overcooked_goal_model ──
    goals, plans, prior = build_overcooked_goals(RECIPE_INGREDIENTS)
    print(f"Goals ({len(goals)}): {goals}")
    print(f"Ground truth: Player 1 is making {RECIPE_NAMES[GROUND_TRUTH_IDX]}")

    # ── Build action sequences ──
    p1_actions = build_p1_action_sequence()
    total_steps = len(p1_actions)
    print(f"Total steps: {total_steps}")

    # ── Load Qwen-VL if requested ──
    qwen_model, qwen_processor = None, None
    if args.use_qwen:
        qwen_model, qwen_processor = load_qwen_vl_model(args.model)

    # ── Inference state ──
    sym_posterior = prior.copy()
    qwen_posterior = prior.copy()
    sym_states = {g: set() for g in goals}

    sym_history = [sym_posterior.tolist()]
    qwen_history = [qwen_posterior.tolist()]
    game_frames_pil = []
    qwen_action_frames = []  # frames at action steps only, for Qwen
    action_labels = []

    # Render initial frame
    surface = viz.render_state(state, grid)
    init_img = pygame_surface_to_pil(surface)
    game_frames_pil.append(init_img)
    action_labels.append("Initial state")

    print("\n--- Simulating game ---")

    # ── Game loop ──
    for step in range(total_steps):
        p1_action = p1_actions[step]
        joint_action = (p1_action, STAY)  # P2 stays

        try:
            new_state, infos = mdp.get_state_transition(state, joint_action)
        except Exception as e:
            print(f"  Step {step}: ERROR - {e}")
            for _ in range(step, total_steps):
                sym_history.append(sym_posterior.tolist())
                qwen_history.append(qwen_posterior.tolist())
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
        label = action_str.upper().replace("_", " ") if action_str else "moving..."
        action_labels.append(label)

        # ── Symbolic posterior update ──
        if action_str is not None:
            sym_lik = symbolic_likelihood_for_action(
                action_str, goals, plans, sym_states, act_noise=0.05
            )
            sym_posterior = posterior_update(sym_posterior, sym_lik)
            update_symbolic_states(action_str, goals, plans, sym_states)
        sym_history.append(sym_posterior.tolist())

        # ── Qwen-VL posterior update ──
        if args.use_qwen and qwen_model is not None:
            if action_str is not None:
                # Pass all history frames (one per step) to Qwen
                frames_for_qwen = game_frames_pil[:]
                print(f"  Step {step:2d}: Querying Qwen-VL with {len(frames_for_qwen)} frames...", end=" ")
                try:
                    response = query_qwen_vl(qwen_model, qwen_processor, frames_for_qwen)
                    parsed = parse_qwen_response(response)
                    if parsed is not None:
                        qwen_posterior = np.array(parsed)
                        print(f"→ A:{parsed[0]:.2f} B:{parsed[1]:.2f} C:{parsed[2]:.2f} D:{parsed[3]:.2f}")
                    else:
                        print(f"→ parse failed. Raw: {response[:100]}")
                except Exception as e:
                    print(f"→ Qwen error: {e}")
        else:
            # Without Qwen, mirror the symbolic posterior as placeholder
            qwen_posterior = prior.copy()
        qwen_history.append(qwen_posterior.tolist())

        # Log
        sparse_rew = sum(infos.get("sparse_reward_by_agent", [0, 0]))
        goal_probs = " ".join(f"{RECIPE_NAMES[i][:5]}={sym_posterior[i]:.3f}" for i in range(len(goals)))
        log_parts = [f"  Step {step:2d}: {label:30s}", goal_probs]
        if sparse_rew > 0:
            log_parts.append(f"REWARD={sparse_rew}")
        print(" | ".join(log_parts))

    # ── Compose video frames ──
    print(f"\n--- Composing {len(game_frames_pil)} frames ---")
    all_timesteps = list(range(max(len(sym_history), len(qwen_history))))
    composed_frames = []

    for i in range(len(game_frames_pil)):
        sym_plot = render_posterior_plot(
            all_timesteps, sym_history[:i + 1], i,
            title="Symbolic (Miniclip Actions)",
        )
        qwen_title = "Qwen3-VL (Vision)" if args.use_qwen else "Qwen3-VL (not enabled)"
        qwen_plot = render_posterior_plot(
            all_timesteps, qwen_history[:i + 1], i,
            title=qwen_title,
        )
        label = action_labels[i] if i < len(action_labels) else ""
        composed = compose_frame(game_frames_pil[i], sym_plot, qwen_plot, label, i)
        composed_frames.append(composed)
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
