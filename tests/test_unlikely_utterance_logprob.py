#!/usr/bin/env python3
"""Test: Unlikely utterances have lower log P(u | pi, g) than likely ones.

For each goal, we compare a *likely* utterance (one that matches the plan) with
an *unlikely* utterance (irrelevant to any plan) and verify that the likely
utterance always scores higher under the mixture-of-prompts model.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "miniclips_python"))

from likelihood_models import (
    construct_utterance_prompt,
    enumerate_command_candidates,
    get_future_actions,
    local_prompt_completion_log_likelihood,
    mixture_prompt_utterance_log_likelihood_local,
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


# Likely utterances per goal (matching the plan)
LIKELY_UTTERANCES = {
    "greek_salad": " Get the olives and cucumber.",
    "veggie_burger": " Can you get me the hamburger bun?",
    "fried_rice": " We need soy sauce and onions.",
    "burrito_bowl": " Could you grab the black beans?",
}

# Unlikely utterances (irrelevant to any recipe)
UNLIKELY_UTTERANCES = [
    " Let's get some soap.",
    " I need to buy a new laptop.",
    " Can you find the dog food?",
    " We should pick up some shampoo.",
]


def main():
    model_name = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-0.6B")
    print(f"Loading model {model_name} ...")
    model, tokenizer = load_model(model_name)
    print("Model loaded.\n")

    state = set()  # empty initial state
    all_passed = True

    print("=" * 80)
    print("TEST: Likely utterances should have HIGHER log-prob than unlikely ones")
    print("=" * 80)

    for goal in GOALS:
        plan = PLANS[goal]
        likely_utt = LIKELY_UTTERANCES[goal]

        # Score the likely utterance
        likely_logp = mixture_prompt_utterance_log_likelihood_local(
            likely_utt, plan, state, model, tokenizer
        )

        print(f"\nGoal: {goal}")
        print(f"  Likely:   {likely_utt!r:55s}  log P = {likely_logp:.3f}")

        for unlikely_utt in UNLIKELY_UTTERANCES:
            unlikely_logp = mixture_prompt_utterance_log_likelihood_local(
                unlikely_utt, plan, state, model, tokenizer
            )
            passed = likely_logp > unlikely_logp
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            print(
                f"  Unlikely: {unlikely_utt!r:55s}  log P = {unlikely_logp:.3f}  "
                f"[{status}] diff = {likely_logp - unlikely_logp:+.3f}"
            )

    print("\n" + "=" * 80)
    # Also test per-command log-probs directly
    print("\nTEST: Per-command log P(u|prompt) for likely vs unlikely utterances")
    print("=" * 80)

    goal = "fried_rice"
    plan = PLANS[goal]
    future_acts = get_future_actions(state, plan)
    commands = enumerate_command_candidates(future_acts, max_actions_per_command=2)

    likely_utt = " We need soy sauce and onions."
    unlikely_utt = " Let's get some soap."

    print(f"\nGoal: {goal}")
    print(f"  Likely utterance:   {likely_utt!r}")
    print(f"  Unlikely utterance: {unlikely_utt!r}\n")

    print(f"  {'Command':50s} {'logP(likely)':>14s} {'logP(unlikely)':>14s} {'diff':>10s}")
    print("  " + "-" * 92)
    for cmd in commands[:10]:  # show top 10
        prompt = construct_utterance_prompt(cmd)
        lp_likely = local_prompt_completion_log_likelihood(
            model, tokenizer, prompt, likely_utt
        )
        lp_unlikely = local_prompt_completion_log_likelihood(
            model, tokenizer, prompt, unlikely_utt
        )
        cmd_str = " ".join(cmd)
        print(f"  {cmd_str:50s} {lp_likely:>14.3f} {lp_unlikely:>14.3f} {lp_likely - lp_unlikely:>+10.3f}")

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED: Likely utterances consistently score higher.")
    else:
        print("SOME TESTS FAILED: See FAIL entries above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
