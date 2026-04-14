"""
Render a test video showing one agent fetching all 10 ingredients,
cooking a mega-soup, plating, and serving.

Layout: all_ten_ingredients (13x10, open interior)
  Row 0: X O X T X C X R X P X X X   (top dispensers + pot)
  Row 4: V . . . . . . . . . . . B   (left/right dispensers)
  Row 7: E . . . . . . . . . . . Y   (left/right dispensers)
  Row 9: X D X Z X G X X X X X S X   (dish, bottom dispensers, serve)

Usage: python render_test_video_all10.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "overcooked_ai", "src"))

import pygame
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


STAY = Action.STAY
INTERACT = Action.INTERACT
N = Direction.NORTH
S = Direction.SOUTH
E = Direction.EAST
W = Direction.WEST


def move_to(start, end):
    """Generate action list to move from start to end in the open interior."""
    sx, sy = start
    ex, ey = end
    actions = []
    dx = 1 if ex > sx else -1
    dy = 1 if ey > sy else -1
    h_dir = E if dx == 1 else W
    v_dir = S if dy == 1 else N
    # Move horizontally first, then vertically
    for _ in range(abs(ex - sx)):
        actions.append(h_dir)
    for _ in range(abs(ey - sy)):
        actions.append(v_dir)
    return actions


def main():
    mdp = OvercookedGridworld.from_layout_name(
        "all_ten_ingredients",
        cook_time=10,
    )
    state = mdp.get_standard_start_state()
    grid = mdp.terrain_mtx
    viz = StateVisualizer(tile_size=75, is_rendering_hud=False)

    out_dir = os.path.join(os.path.dirname(__file__), "output", "test_video_all10_frames")
    os.makedirs(out_dir, exist_ok=True)

    # Ingredient pickup plan:
    # (dispenser_access_pos, face_direction, ingredient_name)
    # Player 1 starts at (2, 2)
    ingredients = [
        # Top wall dispensers (access from row 1)
        ((1, 1), N, "onion"),        # O at (1,0)
        ((3, 1), N, "tomato"),       # T at (3,0)
        ((5, 1), N, "cucumber"),     # C at (5,0)
        ((7, 1), N, "rice"),         # R at (7,0)
        # Left wall dispensers
        ((1, 4), W, "olive"),        # V at (0,4)
        ((1, 7), W, "feta_cheese"),  # E at (0,7)
        # Right wall dispensers
        ((11, 4), E, "hamburger_bun"),  # B at (12,4)
        ((11, 7), E, "soy_sauce"),     # Y at (12,7)
        # Bottom wall dispensers (access from row 8)
        ((3, 8), S, "frozen_peas"),    # Z at (3,9)
        ((5, 8), S, "frozen_carrots"), # G at (5,9)
    ]

    pot_access = (9, 1)    # stand here to interact with P at (9,0)
    pot_face = N
    dish_access = (1, 8)   # stand here to interact with D at (1,9)
    dish_face = S
    serve_access = (11, 8) # stand here to interact with S at (11,9)
    serve_face = S

    p1_actions = []
    cur_pos = (2, 2)  # Player 1 start

    # Pick up each ingredient and put in pot
    for access_pos, face_dir, name in ingredients:
        # Move to dispenser access position
        p1_actions.extend(move_to(cur_pos, access_pos))
        cur_pos = access_pos
        # Face the dispenser and interact
        p1_actions.append(face_dir)
        p1_actions.append(INTERACT)
        # Move to pot
        p1_actions.extend(move_to(cur_pos, pot_access))
        cur_pos = pot_access
        # Face pot and drop ingredient
        p1_actions.append(pot_face)
        p1_actions.append(INTERACT)

    # Pot now has all 10 ingredients; interact again to start cooking
    p1_actions.append(INTERACT)

    # Wait for cooking (cook_time=10)
    for _ in range(10):
        p1_actions.append(STAY)

    # Pick up dish
    p1_actions.extend(move_to(cur_pos, dish_access))
    cur_pos = dish_access
    p1_actions.append(dish_face)
    p1_actions.append(INTERACT)

    # Go to pot and scoop
    p1_actions.extend(move_to(cur_pos, pot_access))
    cur_pos = pot_access
    p1_actions.append(pot_face)
    p1_actions.append(INTERACT)

    # Go to serve
    p1_actions.extend(move_to(cur_pos, serve_access))
    cur_pos = serve_access
    p1_actions.append(serve_face)
    p1_actions.append(INTERACT)

    # A few trailing frames
    for _ in range(5):
        p1_actions.append(STAY)

    # Render
    frame_idx = 0
    img_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")
    surface = viz.render_state(state, grid)
    pygame.image.save(surface, img_path)
    print(f"Frame {frame_idx}: initial state")

    for step, p1_action in enumerate(p1_actions):
        joint_action = (p1_action, STAY)
        try:
            state, infos = mdp.get_state_transition(state, joint_action)
        except Exception as e:
            print(f"Step {step}: Error - {e}")
            break

        frame_idx += 1
        img_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")
        surface = viz.render_state(state, grid)
        pygame.image.save(surface, img_path)

        sparse_rew = sum(infos.get("sparse_reward_by_agent", [0, 0]))
        events = infos.get("event_infos", {})
        desc = f"Frame {frame_idx}: P1 action={p1_action}"
        if sparse_rew > 0:
            desc += f" | REWARD: {sparse_rew}"
        for event_name, player_flags in events.items():
            if any(player_flags):
                desc += f" | {event_name}"
        print(desc)

    total_frames = frame_idx + 1
    print(f"\nRendered {total_frames} frames to {out_dir}/")

    # Create video
    video_path = os.path.join(os.path.dirname(__file__), "output", "test_video_all10.mp4")
    print(f"\nCreating video: {video_path}")
    try:
        import imageio.v3 as iio
        frames = []
        for i in range(total_frames):
            fpath = os.path.join(out_dir, f"frame_{i:04d}.png")
            frames.append(iio.imread(fpath))
        iio.imwrite(video_path, frames, fps=3, codec="libx264")
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Video creation failed: {e}")

    # Create GIF
    gif_path = os.path.join(os.path.dirname(__file__), "output", "test_video_all10.gif")
    print(f"Creating GIF: {gif_path}")
    try:
        import imageio.v3 as iio
        frames = []
        for i in range(total_frames):
            fpath = os.path.join(out_dir, f"frame_{i:04d}.png")
            frames.append(iio.imread(fpath))
        iio.imwrite(gif_path, frames, duration=333, loop=0)
        print(f"GIF saved to {gif_path}")
    except Exception as e:
        print(f"GIF creation failed: {e}")


if __name__ == "__main__":
    main()
