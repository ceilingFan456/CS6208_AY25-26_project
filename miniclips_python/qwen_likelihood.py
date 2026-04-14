"""Qwen-based likelihood estimation for Overcooked goal inference.

Uses Qwen (text LLM) to estimate P(observation | goal) by:
1. Constructing a prompt describing the game state and candidate goals
2. Asking Qwen to rate how likely each action/state is under each goal
3. Extracting log-probabilities from the model output

Two modes:
- action_likelihood_qwen: P(symbolic_action | goal, state) via completion log-probs
- state_likelihood_qwen: P(state_description | goal) via prompted scoring
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Set, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


# ── Prompt templates ──

ACTION_LIKELIHOOD_PROMPT = """\
You are observing a cooperative cooking game. Players prepare soups by:
1. Picking up ingredients from dispensers
2. Adding ingredients to a pot
3. Starting the cooking process
4. Picking up a dish, then the cooked soup
5. Serving the soup at the serving window

The target recipe is: {recipe_description}

So far, the player has completed these actions: {completed_actions}

The next action the player takes is:"""


STATE_LIKELIHOOD_PROMPT = """\
You are observing a cooperative cooking game where players prepare soups.

The target recipe requires these ingredients: {recipe_description}

Current game state: {state_description}

Based on this state, rate how likely the player is working toward this recipe.\
 Answer with a single number from 0 to 10, where 10 means very likely:"""


def _recipe_description(goal_name: str, plan: Dict[str, List[str]]) -> str:
    """Generate a human-readable recipe description from goal name."""
    # Parse goal name like "3_onion_soup" or "cucumber_rice_tomato_soup"
    parts = goal_name.replace("_soup", "").split("_")
    ingredients = []
    i = 0
    while i < len(parts):
        # Check if part is a number (count prefix)
        if parts[i].isdigit() and i + 1 < len(parts):
            count = int(parts[i])
            ing = parts[i + 1]
            ingredients.extend([ing] * count)
            i += 2
        else:
            ingredients.append(parts[i])
            i += 1
    return f"a soup with {', '.join(ingredients)}"


def _completed_actions_str(state: Set[str]) -> str:
    """Format completed actions for prompt."""
    if not state:
        return "none yet"
    return ", ".join(sorted(state))


def _local_log_prob(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    completion: str,
) -> float:
    """Compute log P(completion | prompt) using a local HuggingFace causal LM."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    completion_start = len(prompt_ids)

    if completion_start >= len(full_ids):
        return -1e9  # completion produced no tokens

    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    total = 0.0
    for t in range(completion_start, len(full_ids)):
        token_id = full_ids[t]
        total += float(log_probs[0, t - 1, token_id])

    return total


def build_qwen_action_likelihood_fn(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str | None = None,
) -> Callable[[str, str, Dict[str, List[str]], Set[str]], float]:
    """Build a likelihood function P(action | goal, plan, state) using Qwen.

    The returned callable has signature:
        (observed_action, goal_name, plan, completed_state) -> float (probability)
    """
    if torch is None:
        raise ImportError("torch and transformers required. pip install torch transformers")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto",
    ).to(device)
    model.eval()

    cache: Dict[Tuple[str, str, Tuple[str, ...]], float] = {}

    def _likelihood(
        observed_action: str,
        goal_name: str,
        plan: Dict[str, List[str]],
        state: Set[str],
    ) -> float:
        state_key = tuple(sorted(state))
        key = (observed_action, goal_name, state_key)
        if key in cache:
            return cache[key]

        recipe_desc = _recipe_description(goal_name, plan)
        completed = _completed_actions_str(state)

        prompt = ACTION_LIKELIHOOD_PROMPT.format(
            recipe_description=recipe_desc,
            completed_actions=completed,
        )
        # Use the action as completion text, preceded by a space
        completion = f" {observed_action}"

        logp = _local_log_prob(model, tokenizer, prompt, completion)
        prob = max(math.exp(logp), 1e-300)
        cache[key] = prob
        return prob

    return _likelihood


def build_qwen_state_likelihood_fn(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str | None = None,
) -> Callable[[str, str, Dict[str, List[str]], Set[str]], float]:
    """Build a likelihood function P(state_description | goal) using Qwen.

    Uses the model to score how consistent a state description is with
    each candidate recipe/goal.
    """
    if torch is None:
        raise ImportError("torch and transformers required. pip install torch transformers")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto",
    ).to(device)
    model.eval()

    cache: Dict[Tuple[str, str], float] = {}

    def _likelihood(
        state_description: str,
        goal_name: str,
        plan: Dict[str, List[str]],
        state: Set[str],
    ) -> float:
        key = (state_description, goal_name)
        if key in cache:
            return cache[key]

        recipe_desc = _recipe_description(goal_name, plan)
        prompt = STATE_LIKELIHOOD_PROMPT.format(
            recipe_description=recipe_desc,
            state_description=state_description,
        )
        # Score completions "10" (very likely) vs "0" (unlikely)
        # Use the ratio of likelihoods as a soft score
        high_logp = _local_log_prob(model, tokenizer, prompt, " 10")
        low_logp = _local_log_prob(model, tokenizer, prompt, " 0")

        # Convert to a probability-like score via softmax between high and low
        max_lp = max(high_logp, low_logp)
        exp_high = math.exp(high_logp - max_lp)
        exp_low = math.exp(low_logp - max_lp)
        prob = exp_high / (exp_high + exp_low)

        cache[key] = max(prob, 1e-300)
        return cache[key]

    return _likelihood


def build_qwen_combined_likelihood_fn(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str | None = None,
    action_weight: float = 0.7,
    state_weight: float = 0.3,
) -> Callable:
    """Build a combined likelihood that uses both action and state signals.

    Returns a callable:
        (obs_value, obs_kind, goal_name, plan, state, state_desc) -> float
    """
    action_fn = build_qwen_action_likelihood_fn(model_name, device)
    state_fn = build_qwen_state_likelihood_fn(model_name, device)

    def _combined(
        obs_value: str,
        obs_kind: str,
        goal_name: str,
        plan: Dict[str, List[str]],
        state: Set[str],
        state_description: str = "",
    ) -> float:
        if obs_kind == "act":
            return action_fn(obs_value, goal_name, plan, state)
        elif obs_kind == "state_description":
            return state_fn(obs_value, goal_name, plan, state)
        else:
            raise ValueError(f"Unknown observation kind: {obs_kind}")

    return _combined
