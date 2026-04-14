"""Goal definitions and plans for Overcooked-AI Bayesian inference.

Maps Overcooked recipes to the miniclips goal/plan format:
- Goals are recipes (e.g., "3_onion_soup", "tomato_cucumber_rice_soup")
- Plans are sequences of symbolic actions with dependency ordering
- Actions: pick_up(ingredient), add_to_pot(ingredient), start_cooking,
           pick_up_dish, pick_up_soup, serve_soup
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def recipe_to_goal_name(ingredients: List[str]) -> str:
    """Convert a recipe ingredient list to a readable goal name.

    E.g. ["onion","onion","onion"] -> "3_onion_soup"
         ["tomato","cucumber","rice"] -> "cucumber_rice_tomato_soup"
    """
    from collections import Counter
    counts = Counter(ingredients)
    parts = []
    for ing in sorted(counts.keys()):
        c = counts[ing]
        if c > 1:
            parts.append(f"{c}_{ing}")
        else:
            parts.append(ing)
    return "_".join(parts) + "_soup"


def build_plan_for_recipe(ingredients: List[str]) -> Dict[str, List[str]]:
    """Build a miniclips-style plan dict for a recipe.

    The plan encodes dependency ordering:
    1. pick_up(X) - no dependencies (can happen in any order)
    2. add_to_pot(X) - depends on pick_up(X)
    3. start_cooking - depends on all add_to_pot actions
    4. pick_up_dish - no dependencies (can happen anytime)
    5. pick_up_soup - depends on start_cooking + pick_up_dish
    6. serve_soup - depends on pick_up_soup
    """
    from collections import Counter
    counts = Counter(ingredients)

    plan: Dict[str, List[str]] = {}
    add_to_pot_actions = []

    # For each ingredient instance, create pick_up and add_to_pot actions
    idx = 0
    for ing in sorted(counts.keys()):
        for i in range(counts[ing]):
            idx += 1
            pick_action = f"pick_up({ing})_{idx}"
            pot_action = f"add_to_pot({ing})_{idx}"

            plan[pick_action] = []  # no deps
            plan[pot_action] = [pick_action]  # depends on picking it up
            add_to_pot_actions.append(pot_action)

    # start_cooking depends on all ingredients being in pot
    plan["start_cooking"] = add_to_pot_actions[:]

    # pick_up_dish has no dependencies
    plan["pick_up_dish"] = []

    # pick_up_soup depends on cooking done + having dish
    plan["pick_up_soup"] = ["start_cooking", "pick_up_dish"]

    # serve_soup depends on picking up the soup
    plan["serve_soup"] = ["pick_up_soup"]

    return plan


def build_overcooked_goals(
    all_orders: List[List[str]],
) -> Tuple[List[str], Dict[str, Dict[str, List[str]]], np.ndarray]:
    """Build miniclips-compatible goal model from Overcooked orders.

    Parameters
    ----------
    all_orders : list of ingredient lists
        E.g. [["onion","onion","onion"], ["tomato","tomato","tomato"]]

    Returns
    -------
    goals : list of goal name strings
    plans : dict mapping goal name -> plan dict
    prior : uniform prior over goals
    """
    goals = []
    plans = {}

    for ingredients in all_orders:
        name = recipe_to_goal_name(ingredients)
        if name not in plans:  # deduplicate
            goals.append(name)
            plans[name] = build_plan_for_recipe(ingredients)

    prior = np.ones(len(goals), dtype=float) / len(goals)
    return goals, plans, prior


# ── Overcooked action space (symbolic, not directional) ──

BASE_INGREDIENTS = [
    "onion", "tomato", "cucumber", "rice", "olive",
    "feta_cheese", "hamburger_bun", "soy_sauce",
    "frozen_peas", "frozen_carrots",
]

# All possible symbolic actions in the Overcooked domain
OVERCOOKED_ACTIONS = (
    [f"pick_up({ing})" for ing in BASE_INGREDIENTS]
    + [f"add_to_pot({ing})" for ing in BASE_INGREDIENTS]
    + ["start_cooking", "pick_up_dish", "pick_up_soup", "serve_soup", "wait"]
)
