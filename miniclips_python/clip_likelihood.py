"""CLIP-based likelihood estimation for Overcooked goal inference.

Uses CLIP as a frame-to-action classifier combined with the plan-based
action model for proper Bayesian goal inference via action marginalization:

1. Encode each game frame with CLIP's image encoder (once per frame).
2. Encode a fixed set of action/ingredient text prompts with CLIP's text encoder.
3. Compute cosine similarity between image and text embeddings.
4. Apply temperature-scaled softmax to get P(action | frame).
5. Compute P(frame | goal) = sum_a P(a | frame) * P(a | goal, state) via
   action marginalization using the plan-based action distribution.
6. Accumulate across frames via Bayesian posterior updates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None


# ── Ingredient list (must match overcooked_goal_model.BASE_INGREDIENTS) ──
BASE_INGREDIENTS = [
    "onion", "tomato", "cucumber", "rice", "olive",
    "feta_cheese", "hamburger_bun", "soy_sauce",
    "frozen_peas", "frozen_carrots",
]

# Human-readable names for prompts
INGREDIENT_DISPLAY = {
    "onion": "onion",
    "tomato": "tomato",
    "cucumber": "cucumber",
    "rice": "rice",
    "olive": "olive",
    "feta_cheese": "feta cheese",
    "hamburger_bun": "hamburger bun",
    "soy_sauce": "soy sauce",
    "frozen_peas": "frozen peas",
    "frozen_carrots": "frozen carrots",
}


def _build_action_prompts() -> Tuple[List[str], List[str]]:
    """Build text prompts for all recognizable actions.

    Returns (prompts, action_keys) where action_keys maps each prompt
    index to a symbolic action name like "pick_up(tomato)" or "idle".
    """
    prompts = []
    action_keys = []

    for ing in BASE_INGREDIENTS:
        display = INGREDIENT_DISPLAY[ing]
        # Picking up from dispenser
        prompts.append(
            f"The blue hat chef is picking up {display} from the {display} dispenser"
        )
        action_keys.append(f"pick_up({ing})")

        # Adding to pot
        prompts.append(
            f"The blue hat chef is adding {display} to the cooking pot"
        )
        action_keys.append(f"add_to_pot({ing})")

    # Generic non-ingredient actions
    prompts.append("The blue hat chef is walking in the kitchen")
    action_keys.append("moving")

    prompts.append("The blue hat chef is standing still and waiting")
    action_keys.append("idle")

    prompts.append("The blue hat chef is starting to cook the soup in the pot")
    action_keys.append("start_cooking")

    prompts.append("The blue hat chef is picking up a dish from the dish dispenser")
    action_keys.append("pick_up_dish")

    prompts.append("The blue hat chef is picking up the cooked soup from the pot")
    action_keys.append("pick_up_soup")

    prompts.append("The blue hat chef is serving the soup at the serving window")
    action_keys.append("serve_soup")

    return prompts, action_keys


class CLIPActionClassifier:
    """Frame-level action classifier using CLIP similarity scoring.

    For each frame, produces P(action | frame) over all defined actions.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        temperature: float = 0.5,
        device: str | None = None,
    ):
        if torch is None or CLIPModel is None:
            raise ImportError(
                "torch and transformers required. "
                "pip install torch transformers"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.temperature = temperature

        print(f"Loading CLIP model: {model_name} ...")
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("CLIP model loaded.")

        # Build and encode text prompts (done once)
        self.prompts, self.action_keys = _build_action_prompts()
        self._encode_text_prompts()

        # Build action-to-index mapping
        self.action_to_indices: Dict[str, List[int]] = {}
        for idx, key in enumerate(self.action_keys):
            self.action_to_indices.setdefault(key, []).append(idx)

    @staticmethod
    def _extract_features(output):
        """Extract tensor from model output (handles both old and new transformers)."""
        if isinstance(output, torch.Tensor):
            return output
        # New transformers returns BaseModelOutputWithPooling
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        raise TypeError(f"Unexpected output type: {type(output)}")

    def _encode_text_prompts(self):
        """Pre-encode all text prompts (run once at init)."""
        inputs = self.processor(
            text=self.prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            raw = self.model.get_text_features(**inputs)
            self.text_embeddings = self._extract_features(raw)
            self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)

    def classify_frame(self, image: Image.Image) -> Dict[str, float]:
        """Classify a single frame into action probabilities.

        Parameters
        ----------
        image : PIL.Image
            A game frame.

        Returns
        -------
        dict mapping action_key -> probability
            P(action | frame) for each defined action.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw = self.model.get_image_features(**inputs)
            image_embedding = self._extract_features(raw)
            image_embedding = F.normalize(image_embedding, dim=-1)

        # Cosine similarity (image_embedding is 1×D, text_embeddings is N×D)
        similarities = (image_embedding @ self.text_embeddings.T).squeeze(0)

        # Temperature-scaled softmax
        logits = similarities / self.temperature
        probs = F.softmax(logits, dim=0).cpu().numpy()

        # Aggregate: if multiple prompts map to the same action, sum their probs
        result = {}
        for key in set(self.action_keys):
            indices = self.action_to_indices[key]
            result[key] = float(sum(probs[i] for i in indices))

        return result

    def classify_frames_batch(
        self, images: List[Image.Image]
    ) -> List[Dict[str, float]]:
        """Classify multiple frames in a single batch.

        Parameters
        ----------
        images : list of PIL.Image

        Returns
        -------
        list of dicts, each mapping action_key -> probability
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw = self.model.get_image_features(**inputs)
            image_embeddings = self._extract_features(raw)
            image_embeddings = F.normalize(image_embeddings, dim=-1)

        # Batch similarities: (B, D) @ (N, D).T -> (B, N)
        similarities = image_embeddings @ self.text_embeddings.T
        logits = similarities / self.temperature
        probs = F.softmax(logits, dim=1).cpu().numpy()

        results = []
        unique_keys = set(self.action_keys)
        for b in range(probs.shape[0]):
            result = {}
            for key in unique_keys:
                indices = self.action_to_indices[key]
                result[key] = float(sum(probs[b][i] for i in indices))
            results.append(result)

        return results


def _get_planned_actions_for_clip(
    state: Set[str], plan: Dict[str, List[str]]
) -> List[str]:
    """Get currently planned actions from a plan DAG, returning base action names.

    Returns unnumbered action names (e.g. 'pick_up(onion)' not 'pick_up(onion)_1')
    suitable for matching against CLIP action keys.
    """
    planned = []
    for act in plan:
        if act in state:
            continue
        if all(dep in state for dep in plan[act]):
            base = act.rsplit("_", 1)[0] if "_" in act else act
            # Avoid duplicates (e.g. pick_up(onion)_1 and pick_up(onion)_2)
            if base not in planned:
                planned.append(base)
    return sorted(planned) if planned else ["wait"]


# All CLIP action keys (must match what CLIPActionClassifier produces)
CLIP_ACTION_KEYS = (
    [f"pick_up({ing})" for ing in BASE_INGREDIENTS]
    + [f"add_to_pot({ing})" for ing in BASE_INGREDIENTS]
    + ["moving", "idle", "start_cooking", "pick_up_dish", "pick_up_soup", "serve_soup"]
)

# Map CLIP's "moving"/"idle" to the plan model's non-ingredient actions
# "moving" and "idle" are not in any plan, so they always get noise probability
_CLIP_TO_PLAN_ACTION = {a: a for a in CLIP_ACTION_KEYS}


def plan_action_distribution(
    state: Set[str],
    plan: Dict[str, List[str]],
    act_noise: float = 0.05,
) -> Dict[str, float]:
    """Compute P(a | goal, state) over CLIP action keys using the plan model.

    Planned actions get (1-noise)/|planned| + noise/|A|.
    Unplanned actions get noise/|A|.
    """
    planned = _get_planned_actions_for_clip(state, plan)
    n_actions = len(CLIP_ACTION_KEYS)
    dist = {}

    for action_key in CLIP_ACTION_KEYS:
        if action_key in planned:
            dist[action_key] = (1.0 - act_noise) / len(planned) + act_noise / n_actions
        else:
            dist[action_key] = act_noise / n_actions

    return dist


def clip_action_marginalization_likelihood(
    action_probs: Dict[str, float],
    goals: List[str],
    plans: Dict[str, Dict[str, List[str]]],
    states: Dict[str, Set[str]],
    act_noise: float = 0.05,
) -> np.ndarray:
    """Compute P(frame | goal) via action marginalization.

    P(I_t | g, s_{t-1}) = sum_a P(a | I_t) * P(a | g, s_{t-1})

    This is the dot product between CLIP's action distribution and the
    plan model's action distribution for each goal.

    Parameters
    ----------
    action_probs : dict
        P(action | frame) from CLIPActionClassifier.classify_frame()
    goals : list of str
        Goal names.
    plans : dict
        Mapping goal name -> plan DAG dict.
    states : dict
        Mapping goal name -> set of completed plan actions.
    act_noise : float
        Action noise parameter for the plan model.

    Returns
    -------
    np.ndarray of shape (n_goals,)
        Likelihood P(frame | goal) for each goal.
    """
    n_goals = len(goals)
    likelihoods = np.zeros(n_goals, dtype=float)

    for i, goal in enumerate(goals):
        plan_dist = plan_action_distribution(states[goal], plans[goal], act_noise)

        # Dot product: sum_a P(a|I_t) * P(a|g, s_{t-1})
        dot = 0.0
        for action_key in CLIP_ACTION_KEYS:
            clip_prob = action_probs.get(action_key, 0.0)
            plan_prob = plan_dist[action_key]
            dot += clip_prob * plan_prob

        likelihoods[i] = dot

    # Ensure no zeros
    likelihoods = np.maximum(likelihoods, 1e-300)
    return likelihoods


class CLIPRecipeInference:
    """End-to-end CLIP-based recipe inference engine.

    Maintains a Bayesian posterior over recipes, updated frame-by-frame
    using CLIP action classification + plan-based action marginalization.
    """

    def __init__(
        self,
        recipe_names: List[str],
        plans: Dict[str, Dict[str, List[str]]],
        model_name: str = "openai/clip-vit-base-patch32",
        temperature: float = 0.5,
        act_noise: float = 0.05,
        device: str | None = None,
    ):
        self.recipe_names = recipe_names
        self.plans = plans
        self.n_recipes = len(recipe_names)
        self.act_noise = act_noise

        self.classifier = CLIPActionClassifier(
            model_name=model_name,
            temperature=temperature,
            device=device,
        )

        # Bayesian state
        self.posterior = np.ones(self.n_recipes) / self.n_recipes
        self.posterior_history: List[np.ndarray] = [self.posterior.copy()]
        self.action_log_probs: Dict[str, float] = {}
        self.frame_count = 0

        # Per-goal completed-action sets (tracks plan progress)
        self.states: Dict[str, Set[str]] = {g: set() for g in self.recipe_names}

    def reset(self):
        """Reset inference state."""
        self.posterior = np.ones(self.n_recipes) / self.n_recipes
        self.posterior_history = [self.posterior.copy()]
        self.action_log_probs = {}
        self.frame_count = 0
        self.states = {g: set() for g in self.recipe_names}

    def observe_action(self, action_str: str):
        """Notify the engine that a symbolic action was observed.

        Updates per-goal plan states so that future plan-based distributions
        reflect which actions have been completed.
        """
        if action_str is None:
            return
        for goal in self.recipe_names:
            plan = self.plans[goal]
            for plan_act in plan:
                if plan_act in self.states[goal]:
                    continue
                base = plan_act.rsplit("_", 1)[0] if "_" in plan_act else plan_act
                if base == action_str or plan_act == action_str:
                    self.states[goal].add(plan_act)
                    break

    def observe_frame(self, image: Image.Image) -> np.ndarray:
        """Process one frame and return updated posterior.

        Parameters
        ----------
        image : PIL.Image
            A game frame.

        Returns
        -------
        np.ndarray
            Updated posterior over recipes.
        """
        action_probs = self.classifier.classify_frame(image)

        # Accumulate log-probabilities for action tracking
        for key, prob in action_probs.items():
            if prob > 1e-10:
                self.action_log_probs[key] = (
                    self.action_log_probs.get(key, 0.0) + np.log(prob)
                )

        # Compute likelihood via action marginalization
        likelihood = clip_action_marginalization_likelihood(
            action_probs,
            self.recipe_names,
            self.plans,
            self.states,
            self.act_noise,
        )

        # Bayesian update
        unnormalized = self.posterior * likelihood
        total = unnormalized.sum()
        if total > 0:
            self.posterior = unnormalized / total
        else:
            self.posterior = np.ones(self.n_recipes) / self.n_recipes

        self.posterior_history.append(self.posterior.copy())
        self.frame_count += 1

        return self.posterior.copy()

    def observe_frames(self, images: List[Image.Image]) -> np.ndarray:
        """Process multiple frames (batch) and return updated posterior.

        Parameters
        ----------
        images : list of PIL.Image

        Returns
        -------
        np.ndarray
            Updated posterior after processing all frames.
        """
        all_action_probs = self.classifier.classify_frames_batch(images)

        for action_probs in all_action_probs:
            for key, prob in action_probs.items():
                if prob > 1e-10:
                    self.action_log_probs[key] = (
                        self.action_log_probs.get(key, 0.0) + np.log(prob)
                    )

            likelihood = clip_action_marginalization_likelihood(
                action_probs,
                self.recipe_names,
                self.plans,
                self.states,
                self.act_noise,
            )

            unnormalized = self.posterior * likelihood
            total = unnormalized.sum()
            if total > 0:
                self.posterior = unnormalized / total
            else:
                self.posterior = np.ones(self.n_recipes) / self.n_recipes

            self.posterior_history.append(self.posterior.copy())
            self.frame_count += 1

        return self.posterior.copy()

    def get_top_actions(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return the top-N actions by accumulated log-probability."""
        sorted_actions = sorted(
            self.action_log_probs.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_actions[:n]
