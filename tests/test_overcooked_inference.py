#!/usr/bin/env python3
"""Test: Overcooked-AI symbolic Bayesian goal inference."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniclips_python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'overcooked_ai', 'src'))

from overcooked_goal_model import build_overcooked_goals
from overcooked_inference import run_overcooked_inference
from overcooked_state_parser import OvercookedObservation

RECIPE_INGREDIENTS = [
    ['tomato', 'olive', 'cucumber', 'onion', 'feta_cheese'],
    ['hamburger_bun', 'tomato', 'onion', 'cucumber', 'frozen_carrots'],
    ['rice', 'onion', 'soy_sauce', 'frozen_peas', 'frozen_carrots'],
    ['rice', 'olive', 'feta_cheese', 'onion', 'tomato'],
]
RECIPE_NAMES = ['Greek Salad', 'Veggie Burger', 'Fried Rice', 'Burrito Bowl']


def test_goal_model_setup():
    goals, plans, prior = build_overcooked_goals(RECIPE_INGREDIENTS)
    assert len(goals) == 4
    assert abs(prior.sum() - 1.0) < 1e-9
    for g in goals:
        assert 'serve_soup' in plans[g], f"plan for {g} missing serve_soup"
    print('PASS: test_goal_model_setup')


def test_fried_rice_convergence():
    goals, plans, prior = build_overcooked_goals(RECIPE_INGREDIENTS)
    ingredient_order = ['rice', 'onion', 'soy_sauce', 'frozen_peas', 'frozen_carrots']

    obs = []
    t = 0
    for ing in ingredient_order:
        obs.append(OvercookedObservation('act', f'pick_up({ing})', 0, t)); t += 1
        obs.append(OvercookedObservation('act', f'add_to_pot({ing})', 0, t)); t += 1
    for a in ['start_cooking', 'pick_up_dish', 'pick_up_soup', 'serve_soup']:
        obs.append(OvercookedObservation('act', a, 0, t)); t += 1

    result = run_overcooked_inference(observations=obs, goals=goals, plans=plans, prior=prior, act_noise=0.05)

    fr_idx = 2  # Fried Rice is index 2
    assert result.posterior[fr_idx] > 0.8, f"Fried Rice posterior should be >0.8, got {result.posterior[fr_idx]:.3f}"
    assert len(result.posterior_trajectory) > 1

    print('PASS: test_fried_rice_convergence')
    print(f'  Final: {dict(zip(RECIPE_NAMES, [f"{p:.3f}" for p in result.posterior]))}')


def test_posterior_trajectory_length():
    goals, plans, prior = build_overcooked_goals(RECIPE_INGREDIENTS)
    obs = [
        OvercookedObservation('act', 'pick_up(rice)', 0, 0),
        OvercookedObservation('act', 'add_to_pot(rice)', 0, 1),
    ]
    result = run_overcooked_inference(observations=obs, goals=goals, plans=plans, prior=prior)
    # trajectory = prior + one entry per observation
    assert len(result.posterior_trajectory) == len(obs) + 1
    print('PASS: test_posterior_trajectory_length')


if __name__ == '__main__':
    test_goal_model_setup()
    test_posterior_trajectory_length()
    test_fried_rice_convergence()
    print('\nAll Overcooked inference tests passed.')
