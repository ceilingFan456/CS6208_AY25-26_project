"""Run sequential Bayesian goal inference for symbolic observations.

This module corresponds to the Julia notebook flow:
1) Initialize a prior over goals
2) For each observation, compute likelihood per goal
3) Update posterior with Bayes rule
4) Record posterior trajectory over time

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from bayesian_inference import posterior_update
from goal_model import GOALS, PLANS, PRIOR
from likelihood_models import action_likelihood
from utils import goal_prob_dict, rounded


@dataclass(frozen=True)
class Observation:
    """Symbolic observation at one timestep.

    `kind` should be either:
    - "act" for action observations
    - "utterance" for symbolic utterance observations
    """

    kind: str
    value: str


def run_inference_loop(
    observations: List[Observation],
    act_noise: float = 0.05,
    utterance_likelihood_fn: Optional[Callable[..., float]] = None,
) -> Dict[str, object]:
    """Sequentially update goal posterior for the given observation list.
    
        It is rather simple, just need to calculate the likelihood for each observation and update the posterior accordingly.
    """

    posterior = PRIOR.copy()

    # Per-goal state mirrors the Julia model's goal-conditioned state evolution.
    states: Dict[str, Set[str]] = {goal: set() for goal in GOALS}

    posterior_trajectory = [posterior.copy()]

    for obs in observations:
        likelihood = np.zeros(len(GOALS), dtype=float)

        for i, goal in enumerate(GOALS):
            plan = PLANS[goal]
            state = states[goal]

            if obs.kind == "act":
                likelihood[i] = action_likelihood(obs.value, state, plan, act_noise=act_noise)
            elif obs.kind == "utterance":
                if utterance_likelihood_fn is None:
                    raise ValueError(
                        "For utterance observations, provide utterance_likelihood_fn"
                    )
                try:
                    likelihood[i] = float(
                        utterance_likelihood_fn(obs.value, goal, plan, state)
                    )
                except TypeError:
                    # Backward compatibility for older 2-arg callables.
                    likelihood[i] = float(utterance_likelihood_fn(obs.value, goal))
            else:
                raise ValueError(f"Unknown observation kind: {obs.kind}")

        posterior = posterior_update(posterior, likelihood)
        posterior_trajectory.append(posterior.copy())

        # State update corresponds to Julia's action update in act/full user models.
        if obs.kind == "act" and obs.value != "wait()":
            for goal in GOALS:
                states[goal].add(obs.value)

    return {
        "posterior": posterior,
        "posterior_trajectory": posterior_trajectory,
    }


def _test_notebook_action_examples() -> None:
    """Regression test for notebook action-only posterior examples."""

    """
    T=2, noise=0.05
        P(goal)		goal
        0.272		greek_salad
        0.450		veggie_burger
        0.006		fried_rice
        0.272		burrito_bowl
    """

    # Julia cell: observations = [get(tomato), get(onion)], T=2, act_noise=0.05
    obs1 = [
        Observation("act", "get(tomato)"),
        Observation("act", "get(onion)"),
    ]
    out1 = run_inference_loop(obs1, act_noise=0.05)
    p1 = rounded(out1["posterior"], 3)

    print("Posterior after 2 observations:", goal_prob_dict(GOALS, p1))

    
    """example from the notebook
    T=3, noise=0.05 
        P(goal)		goal
        0.011		greek_salad
        0.017		veggie_burger
        0.030		fried_rice
        0.942		burrito_bowl
    """

    # Julia cell: observations = [get(tomato), get(onion), get(olives)], T=3
    # In notebook this call used default act_noise=0.05 via `(T,)`.
    obs2 = [
        Observation("act", "get(tomato)"),
        Observation("act", "get(onion)"),
        Observation("act", "get(rice)"),
    ]
    out2 = run_inference_loop(obs2, act_noise=0.05)
    p2 = rounded(out2["posterior"], 3)

    print("Posterior after 3 observations:", goal_prob_dict(GOALS, p2))


def main() -> None:
    ## Standalone example run.
    
    ## test the action likelihood and posterior update with the same example as the notebook.
    _test_notebook_action_examples()


if __name__ == "__main__":
    main()
