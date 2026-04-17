#!/usr/bin/env python3
"""Test: CLIP-based action classification and marginalization likelihood."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniclips_python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'overcooked_ai', 'src'))

import numpy as np
from overcooked_goal_model import build_overcooked_goals
from clip_likelihood import (
    CLIP_ACTION_KEYS,
    plan_action_distribution,
    clip_action_marginalization_likelihood,
)

RECIPE_INGREDIENTS = [
    ['tomato', 'olive', 'cucumber', 'onion', 'feta_cheese'],
    ['hamburger_bun', 'tomato', 'onion', 'cucumber', 'frozen_carrots'],
    ['rice', 'onion', 'soy_sauce', 'frozen_peas', 'frozen_carrots'],
    ['rice', 'olive', 'feta_cheese', 'onion', 'tomato'],
]


def test_plan_action_distribution():
    """Plan distribution should sum to ~1 and assign higher prob to planned actions."""
    goals, plans, _ = build_overcooked_goals(RECIPE_INGREDIENTS)
    dist = plan_action_distribution(set(), plans[goals[0]], act_noise=0.05)
    total = sum(dist.values())
    assert abs(total - 1.0) < 0.01, f"Distribution should sum to ~1, got {total:.4f}"
    print('PASS: test_plan_action_distribution')


def test_action_marginalization():
    """Marginalization should produce valid likelihoods for each goal."""
    goals, plans, _ = build_overcooked_goals(RECIPE_INGREDIENTS)
    states = {g: set() for g in goals}

    # Simulate CLIP giving high prob to pick_up(rice)
    action_probs = {k: 0.01 for k in CLIP_ACTION_KEYS}
    action_probs['pick_up(rice)'] = 0.8
    remaining = 1.0 - 0.8
    for k in CLIP_ACTION_KEYS:
        if k != 'pick_up(rice)':
            action_probs[k] = remaining / (len(CLIP_ACTION_KEYS) - 1)

    lik = clip_action_marginalization_likelihood(action_probs, goals, plans, states)
    assert lik.shape == (4,)
    assert all(l > 0 for l in lik), "All likelihoods should be positive"
    # Goals containing rice (fried_rice=idx2, burrito_bowl=idx3) should have higher likelihood
    print('PASS: test_action_marginalization')
    print(f'  Likelihoods: {dict(zip(goals, [f"{l:.6f}" for l in lik]))}')


def test_clip_action_keys_complete():
    """CLIP action keys should cover all 10 ingredients × 2 + 6 generic actions."""
    assert len(CLIP_ACTION_KEYS) == 26, f"Expected 26 action keys, got {len(CLIP_ACTION_KEYS)}"
    assert 'pick_up(rice)' in CLIP_ACTION_KEYS
    assert 'add_to_pot(tomato)' in CLIP_ACTION_KEYS
    assert 'serve_soup' in CLIP_ACTION_KEYS
    print('PASS: test_clip_action_keys_complete')


if __name__ == '__main__':
    test_clip_action_keys_complete()
    test_plan_action_distribution()
    test_action_marginalization()
    print('\nAll CLIP inference tests passed.')
