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

from typing import Dict, Iterable, List, Set

from goal_model import ACTIONS


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


def symbolic_utterance_likelihood(
    observed_utterance: str,
    goal_name: str,
    utterance_likelihood_table: Dict[str, Dict[str, float]],
) -> float:
    """Lookup-table symbolic utterance likelihood P(u | g).

    The table is expected in the form:
    {
      utterance_symbol: {
        goal_name: probability,
        ...
      },
      ...
    }
    """

    if observed_utterance not in utterance_likelihood_table:
        raise KeyError(f"Unknown utterance symbol: {observed_utterance}")

    by_goal = utterance_likelihood_table[observed_utterance]
    if goal_name not in by_goal:
        raise KeyError(f"Missing utterance likelihood for goal '{goal_name}'")

    return float(by_goal[goal_name])
