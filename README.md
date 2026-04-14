# Mini-CLIPS Python Baseline

A Python implementation of [Mini-CLIPS](https://github.com/probcomp/GenGPT3.jl) (Cooperative Language-Guided Inverse Plan Search) for Bayesian goal inference in a cooperative grocery shopping scenario.

An AI assistant observes a user's **actions** (e.g. picking up items) and **natural language utterances** (e.g. "Can you grab a tomato?"), then performs sequential Bayesian posterior updates to infer which recipe the user is making.

## Goals

The system reasons over four candidate recipes:

| Goal | Key Ingredients |
|---|---|
| `greek_salad` | tomato, olives, cucumber, onion, feta cheese |
| `veggie_burger` | hamburger bun, tomato, onion, lettuce, frozen patty |
| `fried_rice` | rice, onion, soy sauce, frozen peas, frozen carrots |
| `burrito_bowl` | rice, black beans, cotija cheese, onion, tomato |

## Project Structure

```
miniclips_python/
├── goal_model.py            # Goal definitions, plans (action DAGs), prior
├── bayesian_inference.py    # Bayes rule: posterior ∝ likelihood × prior
├── likelihood_models.py     # Action likelihood + LLM utterance likelihood
├── simulation.py            # Inference loop, observation types, test examples
└── utils.py                 # Formatting helpers
```

## How It Works

At each timestep, the system:

1. Observes an action or utterance
2. Computes a likelihood $P(\text{obs} \mid g)$ for each goal $g$
3. Updates the posterior: $P(g \mid \text{obs}_{1:t}) \propto P(\text{obs}_t \mid g) \cdot P(g \mid \text{obs}_{1:t-1})$

### Action Likelihood

$$P(a_t \mid g, s_{t-1}) = (1 - \epsilon) \cdot \frac{\mathbb{1}[a_t \in \text{planned}]}{|\text{planned}|} + \epsilon \cdot \frac{\mathbb{1}[a_t \in \text{possible}]}{|\text{possible}|}$$

where $\epsilon = 0.05$ is action noise, **planned** actions have all dependencies met, and **possible** actions are all uncompleted actions.

### Utterance Likelihood

Utterance likelihood is computed as a mixture over command-induced prompts:

$$P(u_t \mid g, \pi, s) = \frac{1}{|C|} \sum_{c \in C} P(u_t \mid \text{prompt}(c))$$

where $C$ enumerates subsets (size 1–2) of future actions, and each prompt is a few-shot translation from command to natural language.

Two backends are supported:

| | Azure OpenAI | Local (Qwen3) |
|---|---|---|
| Scoring method | Completions API with `echo=True` + logprobs | Forward pass + log-softmax |
| Default model | `gpt-4.1` | `Qwen/Qwen3-0.6B` |
| Factory | `build_azure_utterance_likelihood_fn()` | `build_local_utterance_likelihood_fn()` |

## Setup

```bash
conda create -n cs6208 python=3.11 -y
conda activate cs6208
pip install numpy
```

For the local Qwen3 utterance model:
```bash
pip install torch transformers accelerate
```

For the Azure OpenAI utterance model:
```bash
pip install openai
```

> **Note:** On V100 GPUs (compute capability 7.0), use `torch==2.4.0` with CUDA 12.1:
> ```bash
> pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
> ```

## Usage

Run from the `miniclips_python/` directory:

```bash
cd miniclips_python/
```

### Action-only inference (no LLM needed)

```bash
python simulation.py
```

Runs two action-observation examples matching the original Julia notebook:

```
Posterior after 2 observations: {'greek_salad': 0.272, 'veggie_burger': 0.45, 'fried_rice': 0.006, 'burrito_bowl': 0.272}
Posterior after 3 observations: {'greek_salad': 0.011, 'veggie_burger': 0.017, 'fried_rice': 0.03, 'burrito_bowl': 0.942}
```

### Utterance + action inference (local Qwen3)

```bash
python simulation.py --utterance
```

Loads `Qwen/Qwen3-0.6B` and runs three mixed observation examples:

1. **Utterance only:** "Can you get the stuff in the frozen section?" → `fried_rice` dominates
2. **Utterance then action:** "Can you grab a tomato?" → `get(onion)` → `veggie_burger` leads
3. **Action then utterance:** `get(rice)` → "Can you get the stuff in the frozen section?" → `fried_rice` at 0.963

Override the model with `LOCAL_MODEL_NAME`:
```bash
LOCAL_MODEL_NAME=Qwen/Qwen3-1.7B python simulation.py --utterance
```

### Likelihood model smoke tests

```bash
# Local model
python likelihood_models.py --local

# Azure OpenAI (requires AZURE_OPENAI_API_KEY)
python likelihood_models.py
```

## Reference

This implementation is based on the Julia tutorial notebook (`tutorial_colab.ipynb`) from:

> Wong, L., Grand, G., Lew, A., Goodman, N., Mansinghka, V., Andreas, J., & Tenenbaum, J. (2023). From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought. *arXiv:2306.12672*.
