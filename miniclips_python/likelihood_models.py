"""Symbolic likelihood models for Mini-CLIPS baseline.

This module is a direct translation of the Julia action model blocks:
- `get_planned_actions(state, plan)`
- action sampling section inside `act_only_user_model`

For action observations, we compute P(act_t | goal, state_{t-1}) exactly as implied by:
- `next_acts = vcat(planned_acts, possible_acts)`
- `next_act_probs = vcat(planned_probs, possible_probs)`
where duplicate action labels accumulate probability mass.
"""

from __future__ import annotations

import math
import os
from itertools import combinations
from random import Random
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

from goal_model import ACTIONS

try:
    from openai import AzureOpenAI
except ImportError:  # pragma: no cover - optional dependency
    AzureOpenAI = None  # type: ignore[assignment]

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

COMMAND_EXAMPLES: List[Tuple[str, str]] = [
    ("get(apple)", "Can you get the apple?"),
    ("get(bread)", "Could you find some bread?"),
    ("get(cheddar_cheese)", "Go grab a block of that cheese."),
    ("get(green_tea)", "Add some tea to the cart."),
    ("checkout()", "Let's checkout."),
    ("get(tofu) get(seitan)", "I need some tofu and seitan."),
    ("get(frozen_mango) get(ice_cream)", "Get the mango and ice cream."),
    ("get(strawberries) get(milk)", "Find me strawberries and milk."),
    ("get(frozen_broccoli) get(frozen_cauliflower)", "We'll need frozen broccoli and cauliflower."),
    ("get(fries) checkout()", "Let's get some fries then checkout."),
]
_rnd = Random(0)
_rnd.shuffle(COMMAND_EXAMPLES)


def get_planned_actions(state: Set[str], plan: Dict[str, List[str]]) -> List[str]:
    """Same as the `get_planned_actions`.

    Keeps actions that are not completed and whose dependencies are all satisfied.
    Returns sorted actions, or `["wait()"]` if none are available.
    """

    planned_acts = []
    for act in plan.keys():
        if act in state:
            continue
        if not all(dep in state for dep in plan[act]):
            continue
        planned_acts.append(act)

    planned_acts = sorted(planned_acts)
    if len(planned_acts) == 0:
        planned_acts.append("wait()")
    return planned_acts


def action_likelihood(
    observed_action: str,
    state: Set[str],
    plan: Dict[str, List[str]],
    act_noise: float = 0.05,
    action_space: Iterable[str] = ACTIONS,
) -> float:
    """P(observed_action | goal, state) 

    original code in the notebook is also adding the probability up. 
    """

    planned_acts = get_planned_actions(state, plan)
    possible_acts = [a for a in action_space if a not in state]

    p = 0.0

    # Mass from planned component: (1 - act_noise) / |planned_acts|
    if observed_action in planned_acts:
        p += (1.0 - act_noise) / float(len(planned_acts))

    # Mass from noise component: act_noise / |possible_acts|
    if observed_action in possible_acts:
        p += act_noise / float(len(possible_acts))

    return p


def construct_utterance_prompt(
    command: Sequence[str],
    examples: Sequence[Tuple[str, str]] = COMMAND_EXAMPLES,
) -> str:
    """Few-shot prompt matching the notebook format."""

    example_strs = [f"Input: {cmd}\nOutput: {utt}" for (cmd, utt) in examples]
    example_str = "\n".join(example_strs)
    command_str = " ".join(command)
    return f"{example_str}\nInput: {command_str}\nOutput:"


def get_future_actions(state: Set[str], plan: Dict[str, List[str]]) -> List[str]:
    """Notebook-equivalent DFS topological order of unfinished actions."""

    future_acts: List[str] = []
    visited: Set[str] = set()
    finished: Set[str] = set()
    queue: List[str] = list(plan.keys())

    while queue:
        act = queue[-1]
        if act in finished:
            queue.pop()
            continue
        if act in visited:
            queue.pop()
            finished.add(act)
            if act not in state:
                future_acts.append(act)
            continue

        visited.add(act)
        for dep in plan[act]:
            if dep in finished:
                continue
            if dep in visited:
                raise ValueError("Cycle detected!")
            queue.append(dep)

    return future_acts


def enumerate_command_candidates(
    future_actions: Sequence[str],
    max_actions_per_command: int = 2,
) -> List[List[str]]:
    """Enumerate subsets of future actions with size in [1, max_actions_per_command]."""

    commands: List[List[str]] = []
    upper = min(max_actions_per_command, len(future_actions))
    for k in range(1, upper + 1):
        for acts in combinations(future_actions, k):
            commands.append(list(acts))
    return commands


def azure_prompt_completion_log_likelihood(
    client: "AzureOpenAI",
    deployment: str,
    prompt: str,
    completion: str,
) -> float:
    """Compute log P(completion | prompt) from token logprobs with echoed text.

    This uses the Completions API scoring pattern:
    - `prompt = prompt + completion`
    - `max_tokens = 0`
    - `echo = true`
    Then sums token logprobs for tokens whose character offsets are in `completion`.
    """

    if not hasattr(client, "completions") or not hasattr(client.completions, "create"):
        raise RuntimeError(
            "Azure client does not expose completions.create; "
            "use a deployment/API version that supports completion logprobs scoring."
        )

    full_text = prompt + completion
    response = client.completions.create(
        model=deployment,
        prompt=full_text,
        max_tokens=0,
        temperature=0,
        echo=True,
        logprobs=0,
    )

    choice = response.choices[0]
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        raise RuntimeError("No logprobs returned by API response")

    token_logprobs = getattr(logprobs, "token_logprobs", None)
    text_offsets = getattr(logprobs, "text_offset", None)
    if token_logprobs is None or text_offsets is None:
        raise RuntimeError("Response logprobs missing token_logprobs/text_offset")

    prefix_len = len(prompt)
    total_logprob = 0.0
    found = False
    for lp, off in zip(token_logprobs, text_offsets):
        if off >= prefix_len and lp is not None:
            total_logprob += float(lp)
            found = True

    if not found:
        raise RuntimeError("No completion tokens were found in echoed logprobs")

    return total_logprob


def local_prompt_completion_log_likelihood(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    completion: str,
) -> float:
    """Compute log P(completion | prompt) using a local HuggingFace causal LM.

    Tokenises prompt and prompt+completion, runs a single forward pass,
    then sums the log-probabilities of every completion token conditioned
    on all preceding tokens.
    """

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    completion_start = len(prompt_ids)

    if completion_start >= len(full_ids):
        raise RuntimeError("Completion produced no extra tokens after the prompt.")

    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab)

    # log-softmax over vocab dimension
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    # For token at position t, logits[0, t-1] predicted it
    for t in range(completion_start, len(full_ids)):
        token_id = full_ids[t]
        total_logprob += float(log_probs[0, t - 1, token_id])

    return total_logprob


def mixture_prompt_utterance_log_likelihood(
    observed_utterance: str,
    plan: Dict[str, List[str]],
    state: Set[str],
    client: "AzureOpenAI",
    deployment: str,
) -> float:
    """Notebook-style mixture log-likelihood over prompts induced by command subsets."""

    future_acts = get_future_actions(state, plan)
    commands = enumerate_command_candidates(future_acts, max_actions_per_command=2)
    if len(commands) == 0:
        return -1e9

    lps = []
    for command in commands:
        prompt = construct_utterance_prompt(command)
        lp = azure_prompt_completion_log_likelihood(
            client=client,
            deployment=deployment,
            prompt=prompt,
            completion=observed_utterance,
        )
        lps.append(lp)

    max_lp = max(lps)
    return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in lps)) - math.log(len(lps))


def mixture_prompt_utterance_log_likelihood_local(
    observed_utterance: str,
    plan: Dict[str, List[str]],
    state: Set[str],
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
) -> float:
    """Mixture log-likelihood using a local HuggingFace model."""

    future_acts = get_future_actions(state, plan)
    commands = enumerate_command_candidates(future_acts, max_actions_per_command=2)
    if len(commands) == 0:
        return -1e9

    lps = []
    for command in commands:
        prompt = construct_utterance_prompt(command)
        lp = local_prompt_completion_log_likelihood(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            completion=observed_utterance,
        )
        lps.append(lp)

    max_lp = max(lps)
    return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in lps)) - math.log(len(lps))


def build_azure_utterance_likelihood_fn(
    endpoint: str,
    api_key: str,
    deployment: str,
    api_version: str = "2024-12-01-preview",
) -> Callable[[str, str, Dict[str, List[str]], Set[str]], float]:
    """Factory returning notebook-style `P(utterance | goal, plan, state)` callable."""

    if AzureOpenAI is None:
        raise ImportError(
            "openai package is required for Azure API calls. Install with: pip install openai"
        )

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    cache: Dict[Tuple[str, str, Tuple[str, ...]], float] = {}

    def _likelihood(
        observed_utterance: str,
        goal_name: str,
        plan: Dict[str, List[str]],
        state: Set[str],
    ) -> float:
        state_key = tuple(sorted(state))
        key = (observed_utterance, goal_name, state_key)
        if key not in cache:
            logp = mixture_prompt_utterance_log_likelihood(
                observed_utterance=observed_utterance,
                plan=plan,
                state=state,
                client=client,
                deployment=deployment,
            )
            cache[key] = max(math.exp(logp), 1e-300)
        return float(cache[key])

    return _likelihood


def build_local_utterance_likelihood_fn(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str | None = None,
    torch_dtype: str = "auto",
) -> Callable[[str, str, Dict[str, List[str]], Set[str]], float]:
    """Factory returning `P(utterance | goal, plan, state)` using a local model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to ``Qwen/Qwen3-0.6B``.
    device : str | None
        PyTorch device string (``"cuda"``, ``"cpu"``, …).  ``None`` picks CUDA
        when available, else CPU.
    torch_dtype : str
        Data-type hint forwarded to ``from_pretrained``.
    """

    if torch is None or AutoModelForCausalLM is None:
        raise ImportError(
            "torch and transformers are required.  "
            "Install with: pip install torch transformers"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    cache: Dict[Tuple[str, str, Tuple[str, ...]], float] = {}

    def _likelihood(
        observed_utterance: str,
        goal_name: str,
        plan: Dict[str, List[str]],
        state: Set[str],
    ) -> float:
        state_key = tuple(sorted(state))
        key = (observed_utterance, goal_name, state_key)
        if key not in cache:
            logp = mixture_prompt_utterance_log_likelihood_local(
                observed_utterance=observed_utterance,
                plan=plan,
                state=state,
                model=model,
                tokenizer=tokenizer,
            )
            cache[key] = max(math.exp(logp), 1e-300)
        return float(cache[key])

    return _likelihood


def _test_prompt_loglikelihood() -> None:
    """Small manual test for prompt+utterance likelihood scoring.

    """

    endpoint = os.getenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://e0271-miptdstj-eastus2.cognitiveservices.azure.com/",
    )
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not api_key:
        print(
            "Skip smoke test: set AZURE_OPENAI_API_KEY.\n"
            "Optional overrides: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, "
            "AZURE_OPENAI_API_VERSION."
        )
        return

    if AzureOpenAI is None:
        print("Skip smoke test: openai package not installed.")
        return

    command = ["get(rice)", "get(onion)"]
    observed_utterance = " Can you grab rice and onions?"
    prompt = construct_utterance_prompt(command)

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )

    try:
        logp = azure_prompt_completion_log_likelihood(
            client=client,
            deployment=deployment,
            prompt=prompt,
            completion=observed_utterance,
        )
        print("Pseudo prompt command:", command)
        print("Observed utterance:", repr(observed_utterance))
        print("log P(utterance | prompt):", logp)
        print("P(utterance | prompt):", math.exp(logp))
    except Exception as exc:
        print("Smoke test failed:", exc)
        print(
            "If your deployment is chat-only, completions logprobs scoring may be unsupported."
        )


def _test_local_prompt_loglikelihood() -> None:
    """Smoke test using a local open-source model (e.g. Qwen3)."""

    model_name = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-0.6B")

    if torch is None:
        print("Skip local smoke test: torch not installed.")
        return

    print(f"Loading model {model_name} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
    except Exception as exc:
        print(f"Could not load model: {exc}")
        return

    command = ["get(rice)", "get(onion)"]
    observed_utterance = " Can you grab rice and onions?"
    prompt = construct_utterance_prompt(command)

    logp = local_prompt_completion_log_likelihood(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        completion=observed_utterance,
    )
    print("Pseudo prompt command:", command)
    print("Observed utterance:", repr(observed_utterance))
    print("log P(utterance | prompt):", logp)
    print("P(utterance | prompt):", math.exp(logp))


if __name__ == "__main__":
    import sys
    if "--local" in sys.argv:
        _test_local_prompt_loglikelihood()
    else:
        _test_prompt_loglikelihood()
