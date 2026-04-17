#!/usr/bin/env python3
"""Test: Dummy CLIP (ideal classifier) Bayesian inference."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniclips_python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'overcooked_ai', 'src'))

import numpy as np
from overcooked_goal_model import build_overcooked_goals
from clip_likelihood import DummyCLIPRecipeInference

RECIPE_INGREDIENTS = [
    ['tomato', 'olive', 'cucumber', 'onion', 'feta_cheese'],
    ['hamburger_bun', 'tomato', 'onion', 'cucumber', 'frozen_carrots'],
    ['rice', 'onion', 'soy_sauce', 'frozen_peas', 'frozen_carrots'],
    ['rice', 'olive', 'feta_cheese', 'onion', 'tomato'],
]
RECIPE_NAMES = ['Greek Salad', 'Veggie Burger', 'Fried Rice', 'Burrito Bowl']


def _run_dummy_clip(diag_prob):
    goals, plans, _ = build_overcooked_goals(RECIPE_INGREDIENTS)
    engine = DummyCLIPRecipeInference(recipe_names=goals, plans=plans, diag_prob=diag_prob)

    ingredient_order = ['rice', 'onion', 'soy_sauce', 'frozen_peas', 'frozen_carrots']
    actions = []
    for ing in ingredient_order:
        actions += [f'pick_up({ing})', f'add_to_pot({ing})']
    actions += ['start_cooking', 'pick_up_dish', 'pick_up_soup', 'serve_soup']

    for a in actions:
        for _ in range(3):
            engine.set_current_action(None)
            engine.observe_frame()
        engine.set_current_action(a)
        engine.observe_frame()
        engine.observe_action(a)

    return engine, goals


def test_high_accuracy_convergence():
    """diag=0.95 should converge strongly to Fried Rice."""
    engine, goals = _run_dummy_clip(0.95)
    fr_idx = 2
    assert engine.posterior[fr_idx] > 0.9, f"Expected >0.9 for fried rice, got {engine.posterior[fr_idx]:.3f}"
    print('PASS: test_high_accuracy_convergence')
    print(f'  Final: {dict(zip(RECIPE_NAMES, [f"{p:.3f}" for p in engine.posterior]))}')


def test_moderate_accuracy():
    """diag=0.5 should still favor Fried Rice."""
    engine, goals = _run_dummy_clip(0.5)
    fr_idx = 2
    assert engine.posterior[fr_idx] == max(engine.posterior), "Fried Rice should have highest posterior"
    print('PASS: test_moderate_accuracy')
    print(f'  Final: {dict(zip(RECIPE_NAMES, [f"{p:.3f}" for p in engine.posterior]))}')


def test_posterior_history_grows():
    """History should have 1 (prior) + N_frames entries."""
    engine, _ = _run_dummy_clip(0.8)
    assert len(engine.posterior_history) > 1
    assert abs(sum(engine.posterior) - 1.0) < 1e-6, "Posterior should sum to 1"
    print('PASS: test_posterior_history_grows')


if __name__ == '__main__':
    test_posterior_history_grows()
    test_moderate_accuracy()
    test_high_accuracy_convergence()
    print('\nAll Dummy CLIP tests passed.')
