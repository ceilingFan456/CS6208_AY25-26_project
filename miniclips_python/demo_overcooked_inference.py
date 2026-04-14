#!/usr/bin/env python3
"""Demo: Bayesian goal inference in Overcooked-AI using Qwen.

Demonstrates the integration by:
1. Setting up an Overcooked environment with multiple possible recipes
2. Simulating a player performing actions toward one recipe
3. Running Bayesian inference to see how the posterior updates
4. Optionally using Qwen to estimate action likelihoods

Usage:
    # Symbolic-only inference (fast, no GPU needed)
    python demo_overcooked_inference.py

    # With Qwen likelihood estimation
    python demo_overcooked_inference.py --use-qwen

    # Custom layout
    python demo_overcooked_inference.py --layout mixed_ingredients

    # Custom Qwen model
    python demo_overcooked_inference.py --use-qwen --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure imports work
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
_project_dir = os.path.dirname(_this_dir)
_overcooked_src = os.path.join(_project_dir, "overcooked_ai", "src")
if _overcooked_src not in sys.path:
    sys.path.insert(0, _overcooked_src)

import numpy as np

from overcooked_goal_model import build_overcooked_goals, recipe_to_goal_name
from overcooked_state_parser import OvercookedObservation
from overcooked_inference import run_overcooked_inference


def demo_symbolic_inference():
    """Demo with synthetic observations and symbolic likelihood only."""
    print("=" * 60)
    print("Demo: Symbolic Bayesian Goal Inference in Overcooked")
    print("=" * 60)

    # Define possible recipes (orders) for a layout
    all_orders = [
        ["onion", "onion", "onion"],       # 3 onion soup
        ["tomato", "tomato", "tomato"],     # 3 tomato soup
    ]

    goals, plans, prior = build_overcooked_goals(all_orders)
    print(f"\nPossible goals: {goals}")
    print(f"Prior: {dict(zip(goals, prior))}\n")

    # Simulate a player making onion soup
    observations = [
        OvercookedObservation("act", "pick_up(onion)", 0, 0),
        OvercookedObservation("act", "add_to_pot(onion)", 0, 1),
        OvercookedObservation("act", "pick_up(onion)", 0, 2),
        OvercookedObservation("act", "add_to_pot(onion)", 0, 3),
        OvercookedObservation("act", "pick_up(onion)", 0, 4),
        OvercookedObservation("act", "add_to_pot(onion)", 0, 5),
        OvercookedObservation("act", "start_cooking", 0, 6),
        OvercookedObservation("act", "pick_up_dish", 0, 7),
        OvercookedObservation("act", "pick_up_soup", 0, 8),
        OvercookedObservation("act", "serve_soup", 0, 9),
    ]

    result = run_overcooked_inference(
        observations=observations,
        goals=goals,
        plans=plans,
        prior=prior,
        act_noise=0.05,
        use_symbolic_actions=True,
    )

    print("Step-by-step posterior updates:")
    print("-" * 50)
    for i, (obs, tp) in enumerate(zip(result.observations, result.timestep_posteriors)):
        probs_str = "  ".join(f"{g}: {p:.3f}" for g, p in tp.items())
        print(f"  T={i:2d} [{obs.value:25s}]  {probs_str}")

    print(f"\n{result.summary()}")


def demo_mixed_ingredients():
    """Demo with more complex multi-recipe layout."""
    print("\n" + "=" * 60)
    print("Demo: Mixed Ingredients Layout")
    print("=" * 60)

    all_orders = [
        ["onion", "cucumber", "rice"],
        ["tomato", "tomato", "cucumber"],
    ]

    goals, plans, prior = build_overcooked_goals(all_orders)
    print(f"\nPossible goals: {goals}")
    print(f"Prior: {dict(zip(goals, prior))}\n")

    # Player starts picking tomatoes -> evidence for tomato soup
    observations = [
        OvercookedObservation("act", "pick_up(tomato)", 0, 0),
        OvercookedObservation("act", "add_to_pot(tomato)", 0, 1),
        OvercookedObservation("act", "pick_up(cucumber)", 0, 2),
        OvercookedObservation("act", "add_to_pot(cucumber)", 0, 3),
    ]

    result = run_overcooked_inference(
        observations=observations,
        goals=goals,
        plans=plans,
        prior=prior,
        act_noise=0.05,
        use_symbolic_actions=True,
    )

    print("Step-by-step posterior updates:")
    print("-" * 50)
    for i, (obs, tp) in enumerate(zip(result.observations, result.timestep_posteriors)):
        probs_str = "  ".join(f"{g}: {p:.3f}" for g, p in tp.items())
        print(f"  T={i:2d} [{obs.value:25s}]  {probs_str}")

    print(f"\n{result.summary()}")


def demo_qwen_inference(model_name: str = "Qwen/Qwen3-0.6B"):
    """Demo using Qwen for action likelihood estimation."""
    print("\n" + "=" * 60)
    print(f"Demo: Qwen-based Goal Inference (model: {model_name})")
    print("=" * 60)

    from qwen_likelihood import build_qwen_action_likelihood_fn

    all_orders = [
        ["onion", "onion", "onion"],
        ["tomato", "tomato", "tomato"],
    ]

    goals, plans, prior = build_overcooked_goals(all_orders)
    print(f"\nPossible goals: {goals}")
    print(f"Loading Qwen model {model_name}...")

    qwen_fn = build_qwen_action_likelihood_fn(model_name=model_name)
    print("Model loaded.\n")

    # Simulate observations
    observations = [
        OvercookedObservation("act", "pick_up(onion)", 0, 0),
        OvercookedObservation("act", "add_to_pot(onion)", 0, 1),
        OvercookedObservation("act", "pick_up(onion)", 0, 2),
        OvercookedObservation("act", "add_to_pot(onion)", 0, 3),
    ]

    result = run_overcooked_inference(
        observations=observations,
        goals=goals,
        plans=plans,
        prior=prior,
        act_noise=0.05,
        qwen_likelihood_fn=qwen_fn,
        use_symbolic_actions=False,
        use_qwen_for_actions=True,
    )

    print("Step-by-step posterior updates (Qwen-based):")
    print("-" * 50)
    for i, (obs, tp) in enumerate(zip(result.observations, result.timestep_posteriors)):
        probs_str = "  ".join(f"{g}: {p:.3f}" for g, p in tp.items())
        print(f"  T={i:2d} [{obs.value:25s}]  {probs_str}")

    print(f"\n{result.summary()}")


def demo_live_game(layout_name: str = "simple_o_t", use_qwen: bool = False,
                   model_name: str = "Qwen/Qwen3-0.6B"):
    """Demo using the full Overcooked environment with live simulation."""
    print("\n" + "=" * 60)
    print(f"Demo: Live Game Inference (layout: {layout_name})")
    print("=" * 60)

    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.actions import Action, Direction

    from overcooked_inference import OvercookedGoalInferenceEngine

    engine = OvercookedGoalInferenceEngine(
        layout_name=layout_name,
        qwen_model=model_name,
        use_qwen=use_qwen,
        act_noise=0.05,
    )
    engine.print_goals()

    # Set up the environment
    mdp = engine.mdp
    env = OvercookedEnv.from_mdp(mdp, horizon=100)

    state = mdp.get_standard_start_state()
    print(f"\nInitial state:")
    for i, p in enumerate(state.players):
        print(f"  Player {i}: pos={p.position}, orient={p.orientation}")

    print(f"\nInitial posterior: {engine.current_posterior}")

    # Simulate some steps: player 0 moves toward onion dispenser and picks up
    # (actions depend on the specific layout)
    print("\nSimulating game steps...")
    print("(In a real integration, actions come from agents or human input)")
    print(f"\nFinal posterior: {engine.current_posterior}")


def main():
    parser = argparse.ArgumentParser(description="Overcooked Goal Inference Demo")
    parser.add_argument("--use-qwen", action="store_true",
                        help="Use Qwen model for likelihood estimation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Qwen model name (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--layout", type=str, default="simple_o_t",
                        help="Overcooked layout name")
    parser.add_argument("--live", action="store_true",
                        help="Run live game simulation demo")
    args = parser.parse_args()

    # Always run symbolic demos
    demo_symbolic_inference()
    demo_mixed_ingredients()

    # Qwen demo if requested
    if args.use_qwen:
        demo_qwen_inference(model_name=args.model)

    # Live game demo
    if args.live:
        demo_live_game(
            layout_name=args.layout,
            use_qwen=args.use_qwen,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
