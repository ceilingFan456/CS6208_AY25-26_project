#!/usr/bin/env python3
"""Test: Unlikely utterances produce more uncertain command distributions.

For each goal we compute P(c | u, pi, g) for a likely utterance and an unlikely
utterance.  The likely utterance should concentrate probability mass on a few
matching commands (low entropy), while the unlikely utterance should spread
mass more evenly (high entropy).
"""

import math
import os
import sys
from typing import Dict, List, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "miniclips_python"))

from likelihood_models import (
    construct_utterance_prompt,
    enumerate_command_candidates,
    get_future_actions,
    local_prompt_completion_log_likelihood,
)
from goal_model import GOALS, PLANS

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "Qwen/Qwen3-0.6B"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto"
    ).to(device)
    model.eval()
    return model, tokenizer


def compute_command_posterior(
    utterance: str,
    plan: Dict[str, List[str]],
    state: Set[str],
    model,
    tokenizer,
) -> List[tuple]:
    """Compute P(command | utterance, plan) for all command candidates.

    Returns a sorted list of (command_str, probability) tuples.
    """
    future_acts = get_future_actions(state, plan)
    commands = enumerate_command_candidates(future_acts, max_actions_per_command=2)
    if not commands:
        return []

    # Compute log P(u | prompt_c) for each command
    log_probs = []
    for cmd in commands:
        prompt = construct_utterance_prompt(cmd)
        lp = local_prompt_completion_log_likelihood(model, tokenizer, prompt, utterance)
        log_probs.append(lp)

    # Uniform prior over commands -> posterior ∝ P(u | prompt_c)
    max_lp = max(log_probs)
    weights = [math.exp(lp - max_lp) for lp in log_probs]
    total = sum(weights)
    probs = [w / total for w in weights]

    result = [(" ".join(cmd), p) for cmd, p in zip(commands, probs)]
    result.sort(key=lambda x: -x[1])
    return result


def entropy(probs: list) -> float:
    """Shannon entropy in nats."""
    return -sum(p * math.log(p) for p in probs if p > 1e-30)


LIKELY_UTTERANCES = {
    "greek_salad": " Get the olives and cucumber.",
    "veggie_burger": " Can you get me the hamburger bun?",
    "fried_rice": " We need soy sauce and onions.",
    "burrito_bowl": " Could you grab the black beans?",
}

UNLIKELY_UTTERANCES = [
    " Let's get some soap.",
    " I need to buy a new laptop.",
]


def main():
    model_name = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-0.6B")
    print(f"Loading model {model_name} ...")
    model, tokenizer = load_model(model_name)
    print("Model loaded.\n")

    state: Set[str] = set()
    all_passed = True

    print("=" * 80)
    print("TEST: Unlikely utterances should have HIGHER entropy (more uncertain)")
    print("      over the command posterior P(c | u, pi, g)")
    print("=" * 80)

    for goal in GOALS:
        plan = PLANS[goal]
        likely_utt = LIKELY_UTTERANCES[goal]

        # -- Likely utterance --
        likely_posterior = compute_command_posterior(
            likely_utt, plan, state, model, tokenizer
        )
        likely_probs = [p for _, p in likely_posterior]
        likely_entropy = entropy(likely_probs)
        max_entropy = math.log(len(likely_probs))  # uniform = maximum entropy

        print(f"\nGoal: {goal}  (num commands = {len(likely_probs)}, "
              f"max entropy = {max_entropy:.3f})")
        print(f"  Likely: {likely_utt!r}")
        print(f"    entropy = {likely_entropy:.3f}  "
              f"(normalized = {likely_entropy / max_entropy:.3f})")
        print(f"    top-5 commands:")
        for cmd_str, p in likely_posterior[:5]:
            print(f"      {p:.4f}  {cmd_str}")

        for unlikely_utt in UNLIKELY_UTTERANCES:
            unlikely_posterior = compute_command_posterior(
                unlikely_utt, plan, state, model, tokenizer
            )
            unlikely_probs = [p for _, p in unlikely_posterior]
            unlikely_entropy = entropy(unlikely_probs)

            passed = unlikely_entropy > likely_entropy
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False

            print(f"\n  Unlikely: {unlikely_utt!r}")
            print(f"    entropy = {unlikely_entropy:.3f}  "
                  f"(normalized = {unlikely_entropy / max_entropy:.3f})  "
                  f"[{status}] diff = {unlikely_entropy - likely_entropy:+.3f}")
            print(f"    top-5 commands:")
            for cmd_str, p in unlikely_posterior[:5]:
                print(f"      {p:.4f}  {cmd_str}")

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED: Unlikely utterances produce higher-entropy "
              "(more uncertain) command distributions.")
    else:
        print("SOME TESTS FAILED: See FAIL entries above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
