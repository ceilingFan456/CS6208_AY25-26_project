"""
Render a test video showing one agent fetching ingredients,
cooking soup, plating, and serving.

Usage: python render_test_video.py
"""

import os
import sys

# Add the src directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "overcooked_ai", "src"))

import pygame
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
    Recipe,
)
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


def main():
    # Use the test layout:
    # Row 0: X V X E X Z X P X
    # Row 1: X _ _ _ _ _ _ _ X
    # Row 2: X _ 1 _ _ _ 2 _ X
    # Row 3: X _ _ _ _ _ _ _ X
    # Row 4: X D X X X X X S X
    #
    # V(1,0)=olive, E(3,0)=feta, Z(5,0)=frozen_peas, P(7,0)=pot
    # D(1,4)=dish, S(7,4)=serve
    # Player1(2,2), Player2(6,2)

    mdp = OvercookedGridworld.from_layout_name(
        "all_ingredients_test",
        cook_time=5,  # Short cook time for demo
    )

    state = mdp.get_standard_start_state()
    grid = mdp.terrain_mtx

    viz = StateVisualizer(tile_size=75, is_rendering_hud=False)

    # Output directory
    out_dir = os.path.join(os.path.dirname(__file__), "output", "test_video_frames")
    os.makedirs(out_dir, exist_ok=True)

    # Player 2 always stays in place
    STAY = Action.STAY
    INTERACT = Action.INTERACT
    N = Direction.NORTH
    S = Direction.SOUTH
    E = Direction.EAST
    W = Direction.WEST

    # Scripted actions for Player 1 (Player 2 just does STAY)
    # Player 1 starts at (2,2)
    p1_actions = []

    # === Pick up olive from V at (1,0) ===
    # Move from (2,2) to (1,1):
    p1_actions.append(N)       # (2,2) -> (2,1)
    p1_actions.append(W)       # (2,1) -> (1,1)
    # Face north to interact with V at (1,0)
    p1_actions.append(N)       # stays at (1,1), face N (blocked by V)
    p1_actions.append(INTERACT) # pick up olive
    # Move to (7,1) to reach pot P at (7,0)
    p1_actions.append(E)       # (1,1) -> (2,1)
    p1_actions.append(E)       # (2,1) -> (3,1)
    p1_actions.append(E)       # (3,1) -> (4,1)
    p1_actions.append(E)       # (4,1) -> (5,1)
    p1_actions.append(E)       # (5,1) -> (6,1)
    p1_actions.append(E)       # (6,1) -> (7,1)
    # Face north to interact with P at (7,0)
    p1_actions.append(N)       # stays at (7,1), face N
    p1_actions.append(INTERACT) # place olive in pot

    # === Pick up feta from E at (3,0) ===
    p1_actions.append(W)       # (7,1) -> (6,1)
    p1_actions.append(W)       # (6,1) -> (5,1)
    p1_actions.append(W)       # (5,1) -> (4,1)
    p1_actions.append(W)       # (4,1) -> (3,1)
    # Face north to interact with E at (3,0)
    p1_actions.append(N)       # stays at (3,1), face N
    p1_actions.append(INTERACT) # pick up feta_cheese
    # Move back to pot
    p1_actions.append(E)       # (3,1) -> (4,1)
    p1_actions.append(E)       # (4,1) -> (5,1)
    p1_actions.append(E)       # (5,1) -> (6,1)
    p1_actions.append(E)       # (6,1) -> (7,1)
    p1_actions.append(N)       # face N
    p1_actions.append(INTERACT) # place feta in pot

    # === Pick up frozen peas from Z at (5,0) ===
    p1_actions.append(W)       # (7,1) -> (6,1)
    p1_actions.append(W)       # (6,1) -> (5,1)
    # Face north to interact with Z at (5,0)
    p1_actions.append(N)       # stays at (5,1), face N
    p1_actions.append(INTERACT) # pick up frozen_peas
    # Move back to pot
    p1_actions.append(E)       # (5,1) -> (6,1)
    p1_actions.append(E)       # (6,1) -> (7,1)
    p1_actions.append(N)       # face N
    p1_actions.append(INTERACT) # place peas in pot -> soup now full with 3 ingredients

    # === Start cooking (interact with pot while empty-handed) ===
    p1_actions.append(INTERACT) # start cooking

    # === Wait for cooking (cook_time=5) ===
    for _ in range(5):
        p1_actions.append(STAY)

    # === Pick up dish from D at (1,4) ===
    # Move from (7,1) to near D
    p1_actions.append(S)       # (7,1) -> (7,2)
    p1_actions.append(S)       # (7,2) -> (7,3)
    p1_actions.append(W)       # (7,3) -> (6,3)
    p1_actions.append(W)       # (6,3) -> (5,3)
    p1_actions.append(W)       # (5,3) -> (4,3)
    p1_actions.append(W)       # (4,3) -> (3,3)
    p1_actions.append(W)       # (3,3) -> (2,3)
    p1_actions.append(W)       # (2,3) -> (1,3)
    # Face south to interact with D at (1,4)
    p1_actions.append(S)       # stays at (1,3), face S
    p1_actions.append(INTERACT) # pick up dish

    # === Go to pot and scoop soup ===
    p1_actions.append(E)       # (1,3) -> (2,3)
    p1_actions.append(E)       # (2,3) -> (3,3)
    p1_actions.append(E)       # (3,3) -> (4,3)
    p1_actions.append(E)       # (4,3) -> (5,3)
    p1_actions.append(E)       # (5,3) -> (6,3)
    p1_actions.append(E)       # (6,3) -> (7,3)
    p1_actions.append(N)       # (7,3) -> (7,2)
    p1_actions.append(N)       # (7,2) -> (7,1)
    p1_actions.append(N)       # stays at (7,1), face N
    p1_actions.append(INTERACT) # scoop soup from pot with dish

    # === Serve at S (7,4) ===
    p1_actions.append(S)       # (7,1) -> (7,2)
    p1_actions.append(S)       # (7,2) -> (7,3)
    p1_actions.append(S)       # stays at (7,3), face S (blocked by S counter at (7,4))
    p1_actions.append(INTERACT) # serve soup!

    # A few more stay actions to see the result
    for _ in range(5):
        p1_actions.append(STAY)

    # Render initial state
    frame_idx = 0
    img_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")
    surface = viz.render_state(state, grid)
    pygame.image.save(surface, img_path)
    print(f"Frame {frame_idx}: initial state")

    # Execute actions and render each frame
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

        # Print progress info
        sparse_rew = sum(infos.get("sparse_reward_by_agent", [0, 0]))
        events = infos.get("event_infos", {})
        desc = f"Frame {frame_idx}: P1 action={p1_action}"
        if sparse_rew > 0:
            desc += f" | REWARD: {sparse_rew}"
        for event_name, player_flags in events.items():
            if any(player_flags):
                desc += f" | {event_name}"
        print(desc)

    print(f"\nRendered {frame_idx + 1} frames to {out_dir}/")

    # Combine into video using imageio
    video_path = os.path.join(os.path.dirname(__file__), "output", "test_video.mp4")
    print(f"\nCreating video: {video_path}")
    try:
        import imageio.v3 as iio
        frames = []
        for i in range(frame_idx + 1):
            fpath = os.path.join(out_dir, f"frame_{i:04d}.png")
            frames.append(iio.imread(fpath))
        iio.imwrite(video_path, frames, fps=3, codec="libx264")
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Video creation failed: {e}")
        print(f"Frames are available in {out_dir}/")

    # Also create a GIF
    gif_path = os.path.join(os.path.dirname(__file__), "output", "test_video.gif")
    print(f"Creating GIF: {gif_path}")
    try:
        import imageio.v3 as iio
        frames = []
        for i in range(frame_idx + 1):
            fpath = os.path.join(out_dir, f"frame_{i:04d}.png")
            frames.append(iio.imread(fpath))
        iio.imwrite(gif_path, frames, duration=333, loop=0)
        print(f"GIF saved to {gif_path}")
    except Exception as e:
        print(f"GIF creation failed: {e}")


if __name__ == "__main__":
    main()
