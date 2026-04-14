"""Main integration: Bayesian goal inference over Overcooked-AI trajectories.

Connects the Overcooked environment with the miniclips Bayesian inference
engine, using Qwen for likelihood estimation.

Usage modes:
1. Online: observe an Overcooked game step-by-step, update beliefs in real time
2. Offline: process a recorded trajectory and produce posterior over time
3. Hybrid: use symbolic action likelihood + Qwen state description scoring
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# Add parent paths so imports work when run standalone
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
_project_dir = os.path.dirname(_this_dir)
_overcooked_src = os.path.join(_project_dir, "overcooked_ai", "src")
if _overcooked_src not in sys.path:
    sys.path.insert(0, _overcooked_src)

from bayesian_inference import normalize, posterior_update
from overcooked_goal_model import (
    OVERCOOKED_ACTIONS,
    build_overcooked_goals,
    build_plan_for_recipe,
    recipe_to_goal_name,
)
from overcooked_state_parser import (
    OvercookedObservation,
    describe_state_text,
    extract_action_from_events,
    extract_trajectory_observations,
)
from likelihood_models import action_likelihood, get_planned_actions


@dataclass
class OvercookedInferenceResult:
    """Result of goal inference over an Overcooked game."""
    goals: List[str]
    posterior: np.ndarray
    posterior_trajectory: List[np.ndarray]
    observations: List[OvercookedObservation]
    timestep_posteriors: List[Dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Goal inference after {len(self.observations)} observations:"]
        for g, p in sorted(
            zip(self.goals, self.posterior), key=lambda x: -x[1]
        ):
            lines.append(f"  {p:.3f}  {g}")
        return "\n".join(lines)


def run_overcooked_inference(
    observations: List[OvercookedObservation],
    goals: List[str],
    plans: Dict[str, Dict[str, List[str]]],
    prior: np.ndarray,
    act_noise: float = 0.05,
    qwen_likelihood_fn: Callable | None = None,
    use_symbolic_actions: bool = True,
    use_qwen_for_actions: bool = False,
) -> OvercookedInferenceResult:
    """Run sequential Bayesian inference over Overcooked observations.

    Parameters
    ----------
    observations : list of OvercookedObservation
        Symbolic actions and/or state descriptions from the game.
    goals, plans, prior : from build_overcooked_goals()
    act_noise : float
        Noise parameter for symbolic action likelihood.
    qwen_likelihood_fn : callable, optional
        Qwen-based likelihood function for state descriptions or actions.
    use_symbolic_actions : bool
        If True, use the plan-based symbolic action likelihood for "act" observations.
    use_qwen_for_actions : bool
        If True, use Qwen to score "act" observations instead of symbolic model.
    """
    posterior = prior.copy()

    # Per-goal completed-action sets (mirrors miniclips state tracking)
    states: Dict[str, Set[str]] = {goal: set() for goal in goals}

    # Track which plan actions map to unnumbered action names
    # (plan uses pick_up(onion)_1, but observations use pick_up(onion))
    def _match_action_to_plan(action: str, plan: Dict[str, List[str]], state: Set[str]) -> str | None:
        """Find the first matching numbered plan action for an unnumbered observation."""
        for plan_action in plan.keys():
            if plan_action in state:
                continue
            # Strip trailing _N to compare
            base = plan_action.rsplit("_", 1)[0] if "_" in plan_action else plan_action
            if base == action or plan_action == action:
                return plan_action
        return None

    posterior_trajectory = [posterior.copy()]
    timestep_posteriors = []
    all_obs = []

    for obs in observations:
        if obs.kind == "act":
            likelihood = np.zeros(len(goals), dtype=float)

            for i, goal in enumerate(goals):
                plan = plans[goal]
                state = states[goal]

                if use_qwen_for_actions and qwen_likelihood_fn is not None:
                    likelihood[i] = qwen_likelihood_fn(
                        obs.value, goal, plan, state
                    )
                elif use_symbolic_actions:
                    # Map observation action to plan action
                    planned = get_planned_actions(state, plan)
                    matched = _match_action_to_plan(obs.value, plan, state)

                    if matched and matched in planned:
                        # Action is currently planned and available
                        likelihood[i] = (1.0 - act_noise) / len(planned)
                        likelihood[i] += act_noise / max(len(OVERCOOKED_ACTIONS), 1)
                    else:
                        # Action is not planned - only noise probability
                        likelihood[i] = act_noise / max(len(OVERCOOKED_ACTIONS), 1)
                else:
                    likelihood[i] = 1.0 / len(goals)  # uniform fallback

            # Ensure no zero likelihoods
            likelihood = np.maximum(likelihood, 1e-300)
            posterior = posterior_update(posterior, likelihood)
            posterior_trajectory.append(posterior.copy())

            # Update states: add matched plan action for each goal
            for goal in goals:
                matched = _match_action_to_plan(obs.value, plans[goal], states[goal])
                if matched:
                    states[goal].add(matched)

            all_obs.append(obs)
            timestep_posteriors.append(
                {g: float(p) for g, p in zip(goals, posterior)}
            )

        elif obs.kind == "state_description" and qwen_likelihood_fn is not None:
            likelihood = np.zeros(len(goals), dtype=float)
            for i, goal in enumerate(goals):
                likelihood[i] = qwen_likelihood_fn(
                    obs.value, goal, plans[goal], states[goal]
                )
            likelihood = np.maximum(likelihood, 1e-300)
            posterior = posterior_update(posterior, likelihood)
            posterior_trajectory.append(posterior.copy())

            all_obs.append(obs)
            timestep_posteriors.append(
                {g: float(p) for g, p in zip(goals, posterior)}
            )

    return OvercookedInferenceResult(
        goals=goals,
        posterior=posterior,
        posterior_trajectory=posterior_trajectory,
        observations=all_obs,
        timestep_posteriors=timestep_posteriors,
    )


class OvercookedGoalInferenceEngine:
    """High-level engine for goal inference in Overcooked games.

    Wraps the full pipeline: environment setup -> observation extraction ->
    Bayesian inference with Qwen likelihood.
    """

    def __init__(
        self,
        layout_name: str = "simple_o_t",
        qwen_model: str = "Qwen/Qwen3-0.6B",
        use_qwen: bool = True,
        act_noise: float = 0.05,
        device: str | None = None,
    ):
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.layout_name = layout_name

        # Extract orders from the MDP to define goals
        start_state = self.mdp.get_standard_start_state()
        all_orders = []
        if hasattr(start_state, "all_orders") and start_state.all_orders:
            for recipe in start_state.all_orders:
                all_orders.append(list(recipe.ingredients))
        else:
            # Fallback: use all recipes from MDP configuration
            for r in self.mdp.start_all_orders:
                all_orders.append(list(r.ingredients))

        self.goals, self.plans, self.prior = build_overcooked_goals(all_orders)
        self.act_noise = act_noise

        # Build Qwen likelihood if requested
        self.qwen_likelihood_fn = None
        if use_qwen:
            from qwen_likelihood import build_qwen_action_likelihood_fn
            self.qwen_likelihood_fn = build_qwen_action_likelihood_fn(
                model_name=qwen_model, device=device,
            )

        # Online inference state
        self._posterior = self.prior.copy()
        self._states: Dict[str, Set[str]] = {g: set() for g in self.goals}
        self._history: List[OvercookedObservation] = []

    def reset(self):
        """Reset inference state for a new episode."""
        self._posterior = self.prior.copy()
        self._states = {g: set() for g in self.goals}
        self._history = []

    @property
    def current_posterior(self) -> Dict[str, float]:
        return {g: float(p) for g, p in zip(self.goals, self._posterior)}

    def observe_step(
        self,
        prev_state,
        new_state,
        event_infos: Dict,
        player_index: int = 0,
    ) -> Dict[str, float]:
        """Process one game step and return updated posterior.

        Call this after each OvercookedGridworld.get_state_transition().
        """
        action = extract_action_from_events(
            event_infos, player_index, prev_state, new_state
        )

        if action is None:
            return self.current_posterior

        obs = OvercookedObservation("act", action, player_index, len(self._history))
        result = run_overcooked_inference(
            observations=[obs],
            goals=self.goals,
            plans=self.plans,
            prior=self._posterior,
            act_noise=self.act_noise,
            qwen_likelihood_fn=self.qwen_likelihood_fn,
            use_symbolic_actions=True,
            use_qwen_for_actions=self.qwen_likelihood_fn is not None,
        )
        self._posterior = result.posterior
        self._history.append(obs)

        return self.current_posterior

    def infer_from_trajectory(
        self,
        trajectory: Dict,
        player_index: int = 0,
    ) -> OvercookedInferenceResult:
        """Run inference over a full recorded trajectory."""
        observations = extract_trajectory_observations(
            trajectory,
            player_index=player_index,
            include_state_descriptions=False,
            mdp=self.mdp,
        )
        return run_overcooked_inference(
            observations=observations,
            goals=self.goals,
            plans=self.plans,
            prior=self.prior,
            act_noise=self.act_noise,
            qwen_likelihood_fn=self.qwen_likelihood_fn,
            use_symbolic_actions=True,
            use_qwen_for_actions=self.qwen_likelihood_fn is not None,
        )

    def print_goals(self):
        """Print the available goals and their plans."""
        print(f"Layout: {self.layout_name}")
        print(f"Goals ({len(self.goals)}):")
        for g in self.goals:
            plan_actions = [k for k in self.plans[g].keys() if not k.startswith("serve")]
            print(f"  {g}: {', '.join(plan_actions[:5])}...")
