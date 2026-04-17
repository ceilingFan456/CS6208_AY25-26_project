#!/usr/bin/env python3
"""Test: Qwen3-VL response parsing and prompt construction (no GPU required)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniclips_python'))

# Only test parsing/prompt logic — model loading requires GPU
from qwen_likelihood import _recipe_description, _completed_actions_str


def test_recipe_description():
    from overcooked_goal_model import build_overcooked_goals
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'overcooked_ai', 'src'))
    goals, plans, _ = build_overcooked_goals([['rice', 'onion', 'soy_sauce']])
    desc = _recipe_description(goals[0], plans[goals[0]])
    assert 'rice' in desc.lower() and 'onion' in desc.lower()
    print(f'PASS: test_recipe_description → "{desc}"')


def test_completed_actions_str():
    assert _completed_actions_str(set()) == 'none yet'
    result = _completed_actions_str({'pick_up(rice)_1', 'add_to_pot(rice)_1'})
    assert 'pick_up' in result and 'add_to_pot' in result
    print(f'PASS: test_completed_actions_str → "{result}"')


def test_qwen_response_parsing():
    """Test the response parser from render_four_goals_video.py."""
    # Inline the parsing logic (same as render_four_goals_video.parse_qwen_response)
    import re
    def parse_qwen_response(text, n=4):
        probs = []
        for i in range(n):
            label = chr(ord('A') + i)
            m = re.search(rf'Recipe {label}\s*[:=]\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
            if m: probs.append(float(m.group(1)))
            else: return None
        total = sum(probs)
        return [p / total for p in probs] if total > 0 else None

    # Valid response
    result = parse_qwen_response("Recipe A: 0.1\nRecipe B: 0.2\nRecipe C: 0.6\nRecipe D: 0.1")
    assert result is not None
    assert abs(sum(result) - 1.0) < 1e-6
    assert result[2] == max(result)  # Recipe C highest
    print(f'PASS: test_qwen_response_parsing → {[f"{r:.2f}" for r in result]}')

    # Invalid response
    assert parse_qwen_response("I don't know") is None
    print('PASS: test_qwen_response_parsing (invalid input)')


if __name__ == '__main__':
    test_completed_actions_str()
    test_recipe_description()
    test_qwen_response_parsing()
    print('\nAll Qwen VLM tests passed.')
