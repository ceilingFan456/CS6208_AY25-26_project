#!/usr/bin/env python3
"""Test: Mini-CLIPS symbolic action-based Bayesian inference."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniclips_python'))

from goal_model import GOALS, PLANS, PRIOR
from simulation import Observation, run_inference_loop
from utils import goal_prob_dict, rounded


def test_two_actions():
    """After get(tomato), get(onion) — veggie_burger should lead."""
    obs = [Observation('act', 'get(tomato)'), Observation('act', 'get(onion)')]
    result = run_inference_loop(obs, act_noise=0.05)
    post = dict(zip(GOALS, result['posterior']))
    assert post['veggie_burger'] > post['greek_salad'], "veggie_burger should lead after tomato+onion"
    print('PASS: test_two_actions')
    print(f'  Posterior: {goal_prob_dict(GOALS, rounded(result["posterior"], 3))}')


def test_fried_rice_sequence():
    """Full fried_rice sequence should converge to fried_rice."""
    obs = [
        Observation('act', 'get(rice)'),
        Observation('act', 'get(onion)'),
        Observation('act', 'get(soy_sauce)'),
        Observation('act', 'get(frozen_peas)'),
        Observation('act', 'get(frozen_carrots)'),
        Observation('act', 'checkout()'),
    ]
    result = run_inference_loop(obs, act_noise=0.05)
    post = dict(zip(GOALS, result['posterior']))
    assert post['fried_rice'] > 0.9, f"fried_rice should be >0.9, got {post['fried_rice']:.3f}"
    assert len(result['posterior_trajectory']) == len(obs) + 1
    print('PASS: test_fried_rice_sequence')
    print(f'  Final: {goal_prob_dict(GOALS, rounded(result["posterior"], 3))}')


def test_uniform_prior():
    """Prior should be uniform over 4 goals."""
    assert len(PRIOR) == 4
    assert abs(PRIOR.sum() - 1.0) < 1e-9
    assert all(abs(p - 0.25) < 1e-9 for p in PRIOR)
    print('PASS: test_uniform_prior')


if __name__ == '__main__':
    test_uniform_prior()
    test_two_actions()
    test_fried_rice_sequence()
    print('\nAll Mini-CLIPS symbolic tests passed.')
