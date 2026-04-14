# Mini-CLIPS + Overcooked-AI: Bayesian Goal Inference

A Python implementation of Bayesian goal inference combining [Mini-CLIPS](https://github.com/probcomp/GenGPT3.jl) (Cooperative Language-Guided Inverse Plan Search) with the [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) cooperative cooking environment.

An observer watches a player's **actions** (picking up ingredients, adding to pot, cooking, serving) and uses **sequential Bayesian posterior updates** to infer which recipe the player is making. Three likelihood models are supported:

1. **Symbolic action model** — plans as dependency DAGs, plan-based likelihood
2. **Qwen3-VL vision model** — directly estimates recipe probabilities from rendered game frames
3. **CLIP vision model** — frame-to-action classifier using CLIP similarity scoring, combined with plan-based action marginalization for Bayesian goal inference

---

## Table of Contents

- [Goals & Recipes](#goals--recipes)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Scripts Reference](#scripts-reference)
- [Examples](#examples)
- [Data & Outputs](#data--outputs)
- [Reference](#reference)

---

## Goals & Recipes

### Miniclip Baseline (Grocery Shopping)

The original miniclip domain uses a grocery shopping metaphor — the player picks items for a recipe:

| Goal | Ingredients |
|---|---|
| `greek_salad` | tomato, olives, cucumber, onion, feta cheese |
| `veggie_burger` | hamburger bun, tomato, onion, lettuce, frozen patty |
| `fried_rice` | rice, onion, soy sauce, frozen peas, frozen carrots |
| `burrito_bowl` | rice, black beans, cotija cheese, onion, tomato |

### Overcooked Four-Goal Experiment

The Overcooked experiment maps these to in-game ingredients (10 available dispensers). Unavailable items are replaced with the closest available substitute:

| Recipe | Ingredients | Substitutions from miniclip |
|---|---|---|
| **Greek Salad Soup** | tomato, olive, cucumber, onion, feta_cheese | olives → olive |
| **Veggie Burger Soup** | hamburger_bun, tomato, onion, cucumber, frozen_carrots | lettuce → cucumber, frozen_patty → frozen_carrots |
| **Fried Rice Soup** | rice, onion, soy_sauce, frozen_peas, frozen_carrots | all original |
| **Burrito Bowl Soup** | rice, olive, feta_cheese, onion, tomato | black_beans → olive, cotija_cheese → feta_cheese |

### Available Overcooked Dispenser Items (10)

| Dispenser | Grid char | Item |
|---|---|---|
| Onion | `O` | `onion` |
| Tomato | `T` | `tomato` |
| Cucumber | `C` | `cucumber` |
| Rice | `R` | `rice` |
| Olive | `V` | `olive` |
| Feta Cheese | `E` | `feta_cheese` |
| Hamburger Bun | `B` | `hamburger_bun` |
| Soy Sauce | `Y` | `soy_sauce` |
| Frozen Peas | `Z` | `frozen_peas` |
| Frozen Carrots | `G` | `frozen_carrots` |

---

## Project Structure

```
CS6208_AY25-26_project/
├── render_four_goals_video.py        # ★ Main experiment: 4-goal inference video
├── render_inference_video.py         # 2-recipe inference video (original demo)
├── render_test_video.py              # Test: 3-ingredient agent
├── render_test_video_all10.py        # Test: all 10-ingredient agent
├── analyze_image.py                  # Standalone Qwen3-VL image analysis
│
├── miniclips_python/                 # Core inference engine
│   ├── goal_model.py                # Goal definitions, action DAGs, prior
│   ├── bayesian_inference.py        # Bayes rule: posterior ∝ likelihood × prior
│   ├── likelihood_models.py         # Action + utterance likelihood models
│   ├── qwen_likelihood.py           # Qwen-based likelihood (text LM)
│   ├── simulation.py                # Inference loop + regression tests
│   ├── clip_likelihood.py           # CLIP-based action classifier + recipe inference
│   ├── overcooked_goal_model.py     # Recipe → goal/plan mapping for Overcooked
│   ├── overcooked_state_parser.py   # Extract symbolic actions from game events
│   ├── overcooked_inference.py      # Full Overcooked inference pipeline
│   ├── demo_overcooked_inference.py # Demo script
│   └── utils.py                     # Formatting helpers
│
├── overcooked_ai/                    # Overcooked-AI environment (modified)
│   └── src/overcooked_ai_py/
│       ├── mdp/                     # Game logic, actions, recipes
│       ├── visualization/           # Pygame rendering
│       └── data/layouts/            # Grid layout files (.layout)
│
├── data/
│   ├── mmtom_qa/                    # MMToM-QA dataset (episode JSONs + questions)
│   ├── MMToM-QA-full/               # Full dataset with training splits
│   └── test_images/                 # Sample images for testing
│
├── output/                           # Generated videos, frames, GIFs
│   ├── four_goals_frames/           # Frames from 4-goal experiment
│   ├── inference_video_frames/      # Frames from 2-recipe demo
│   ├── test_video_frames/           # Frames from test videos
│   └── *.mp4, *.gif                 # Output videos
│
└── tests/                            # Smoke tests
    ├── test_command_uncertainty.py
    └── test_unlikely_utterance_logprob.py
```

---

## How It Works

At each timestep $t$, the system:

1. Observes an action $a_t$ (e.g. `pick_up(rice)`) or a visual frame
2. Computes a likelihood $P(\text{obs}_t \mid g)$ for each candidate goal $g$
3. Updates the posterior via Bayes' rule:

$$P(g \mid \text{obs}_{1:t}) \propto P(\text{obs}_t \mid g) \cdot P(g \mid \text{obs}_{1:t-1})$$

### Symbolic Action Likelihood

Each goal has a **plan** — a DAG of actions with dependency ordering. An action is **planned** if all its dependencies have been completed. The likelihood model is:

$$P(a_t \mid g, s_{t-1}) = (1 - \epsilon) \cdot \frac{\mathbb{1}[a_t \in \text{planned}(g)]}{|\text{planned}(g)|} + \epsilon \cdot \frac{\mathbb{1}[a_t \in \text{possible}]}{|\text{possible}|}$$

where $\epsilon = 0.05$ is the action noise parameter. This is the same model used in the original miniclip Julia code.

**Example**: If the player picks up rice, goals containing rice (Fried Rice, Burrito Bowl) get high likelihood, while goals without rice (Greek Salad, Veggie Burger) get only noise probability.

### Overcooked Plan DAG

For a recipe like Fried Rice Soup (rice, onion, soy_sauce, frozen_peas, frozen_carrots):

```
pick_up(rice) ──→ add_to_pot(rice) ──┐
pick_up(onion) ─→ add_to_pot(onion) ─┤
pick_up(soy_sauce)→add_to_pot(...)  ─┤
pick_up(f_peas) ─→ add_to_pot(...)  ─┼──→ start_cooking ──→ pick_up_soup ──→ serve_soup
pick_up(f_carrots)→add_to_pot(...)  ─┘                  ↗
                             pick_up_dish ───────────────┘
```

### Qwen3-VL Vision Likelihood

Instead of symbolic actions, the vision model receives the **last several rendered game frames** at each action step and directly estimates:

$$P(\text{recipe} \mid \text{frames}_{t-k:t})$$

The model is prompted with the full recipe descriptions, kitchen layout, and dispenser locations.

### CLIP Vision Likelihood

The CLIP model takes a fundamentally different approach: rather than asking a VLM to directly predict the recipe, it uses CLIP as a **frame-to-action classifier** and accumulates evidence over time via Bayesian updates.

#### Step 1: Define action prompts

A fixed set of $N = 26$ text prompts is defined, covering:
- 10 ingredients × 2 actions each: `"The blue hat chef is picking up {ingredient}"` and `"The blue hat chef is adding {ingredient} to the cooking pot"`
- 6 generic actions: moving, idle, start cooking, pick up dish, pick up soup, serve soup

#### Step 2: Pre-encode text prompts

All text prompts are encoded once using CLIP's text encoder:

$$\mathbf{t}_j = \text{CLIP}_{\text{text}}(\text{prompt}_j), \quad j = 1, \ldots, N$$

Each embedding is L2-normalized: $\hat{\mathbf{t}}_j = \mathbf{t}_j / \|\mathbf{t}_j\|$.

#### Step 3: Encode each frame

For each game frame $I_t$, the image is encoded once:

$$\mathbf{v}_t = \text{CLIP}_{\text{image}}(I_t), \quad \hat{\mathbf{v}}_t = \mathbf{v}_t / \|\mathbf{v}_t\|$$

#### Step 4: Compute action probabilities

Cosine similarities are computed between the image embedding and all text embeddings, then converted to a probability distribution using temperature-scaled softmax:

$$s_{t,j} = \hat{\mathbf{v}}_t \cdot \hat{\mathbf{t}}_j$$

$$P(a_j \mid I_t) = \frac{\exp(s_{t,j} / \tau)}{\sum_{k=1}^{N} \exp(s_{t,k} / \tau)}$$

where $\tau$ is the temperature parameter (default 0.5). Lower temperature produces sharper distributions.

#### Step 5: Map actions to recipe likelihoods via action marginalization

CLIP is used as an **action estimator** and the **plan model** as the goal-conditioned action prior, combined via proper Bayesian marginalization over the latent (unobserved) action.

#### Graphical Model

The generative story is:

$$g \;\longrightarrow\; a_t \;\longrightarrow\; I_t$$

1. The agent has a goal $g$ (the recipe they are making)
2. Given the goal and current state $s_{t-1}$, the agent chooses an action $a_t$ according to the plan: $a_t \sim P(a \mid g, s_{t-1})$
3. The action $a_t$ produces an observed frame $I_t$: $I_t \sim P(I \mid a_t)$

**Key conditional independence assumption**: given the action $a_t$, the frame $I_t$ does not depend on the goal $g$:

$$P(I_t \mid a_t, g) = P(I_t \mid a_t)$$

This is reasonable: what the frame *looks like* depends on what the chef is *doing* (picking up onion, walking, etc.), not on what recipe they intend to make.

#### Goal Posterior Update

At each timestep, we update the goal posterior:

$$P(g \mid I_{1:t}) \propto P(I_t \mid g, s_{t-1}) \cdot P(g \mid I_{1:t-1})$$

#### Frame Likelihood via Action Marginalization

Since we don't observe the action directly, we **marginalize** over all possible actions:

$$P(I_t \mid g, s_{t-1}) = \sum_{a \in \mathcal{A}} P(I_t \mid a) \cdot P(a \mid g, s_{t-1})$$

This is the key equation: the likelihood of observing frame $I_t$ under goal $g$ is the expected frame likelihood, weighted by how likely each action is under that goal's plan.

#### Connecting to CLIP Output

CLIP gives us $P(a \mid I_t)$ (the action posterior given the frame). We need $P(I_t \mid a)$ (the frame likelihood given the action). By Bayes' rule:

$$P(I_t \mid a) = \frac{P(a \mid I_t) \cdot P(I_t)}{P(a)}$$

Substituting into the marginalization:

$$P(I_t \mid g, s_{t-1}) = \sum_{a \in \mathcal{A}} \frac{P(a \mid I_t) \cdot P(I_t)}{P(a)} \cdot P(a \mid g, s_{t-1})$$

Since $P(I_t)$ is constant across goals (it doesn't depend on $g$), it cancels in the posterior normalization:

$$\boxed{P(I_t \mid g, s_{t-1}) \propto \sum_{a \in \mathcal{A}} \frac{P(a \mid I_t)}{P(a)} \cdot P(a \mid g, s_{t-1})}$$

**Under a uniform action prior** $P(a) = 1/|\mathcal{A}|$, this simplifies to:

$$P(I_t \mid g, s_{t-1}) \propto \sum_{a \in \mathcal{A}} P(a \mid I_t) \cdot P(a \mid g, s_{t-1})$$

This is the **dot product** (inner product) between two distributions:
- $P(a \mid I_t)$: CLIP's belief about what action is happening (from the frame)
- $P(a \mid g, s_{t-1})$: the plan model's belief about what action should happen (from the goal)

#### Plan-Based Action Distribution $P(a \mid g, s_{t-1})$

This is exactly the symbolic likelihood model already in the codebase (same as the symbolic-only method):

$$P(a \mid g, s_{t-1}) = \begin{cases} \dfrac{1-\epsilon}{|\text{planned}(g, s_{t-1})|} + \dfrac{\epsilon}{|\mathcal{A}|} & \text{if } a \in \text{planned}(g, s_{t-1}) \\[6pt] \dfrac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

where $\epsilon = 0.05$ is the action noise, $\text{planned}(g, s_{t-1})$ are the actions currently available in the plan DAG for goal $g$, and $|\mathcal{A}|$ is the total number of possible actions.

#### Intuitive Explanation

Suppose at time $t$, CLIP says with high confidence: "the chef is picking up rice" (i.e., $P(\text{pick\_up(rice)} \mid I_t) \approx 0.8$).

- For **Fried Rice Soup**, the plan says pick_up(rice) is a planned action → $P(\text{pick\_up(rice)} \mid g_{\text{fried\_rice}}, s_{t-1})$ is high
- For **Greek Salad Soup**, the plan says pick_up(rice) is not planned → $P(\text{pick\_up(rice)} \mid g_{\text{greek\_salad}}, s_{t-1})$ is low (only noise $\epsilon / |\mathcal{A}|$)

The dot product will be large for Fried Rice and small for Greek Salad, correctly shifting the posterior.

Conversely, if CLIP is **uncertain** (e.g., the chef is just walking and all actions have low probability), the dot product will be similar across all goals, and the posterior barely changes — which is the correct behavior.

#### Nice Property: Degeneracy to Symbolic Model

If CLIP were a perfect action classifier (i.e., $P(a \mid I_t) = \delta(a = a^*)$ where $a^*$ is the true action), then:

$$P(I_t \mid g, s_{t-1}) \propto P(a^* \mid g, s_{t-1})$$

which is exactly the symbolic action likelihood. So the CLIP model **generalizes** the symbolic model — it reduces to symbolic inference when action recognition is perfect, and gracefully handles uncertainty when it's not.

#### Step 6: Bayesian accumulation

The recipe likelihood from each frame is used for a standard Bayesian posterior update:

$$P(g \mid I_{1:t}) \propto P(I_t \mid g, s_{t-1}) \cdot P(g \mid I_{1:t-1})$$

Unlike Qwen (which is queried only at action steps), CLIP runs on **every frame**, so the posterior is updated continuously. Over many frames, the accumulated evidence from repeatedly observing the player near certain dispensers drives the posterior toward the correct recipe.

---

#### Comparison of the three methods

| Property | Symbolic | Qwen3-VL | CLIP |
|---|---|---|---|
| Input | Symbolic action strings | Rendered frames (batch) | Rendered frames (per-frame) |
| Update frequency | On action events only | On pot actions only | Every frame |
| Likelihood model | Plan DAG + noise | Direct $P(\text{recipe} \mid \text{frames})$ | $\sum_a P(a \mid I_t) \cdot P(a \mid g, s_{t-1})$ |
| Inference type | Action → recipe | End-to-end VLM | Action classifier + plan marginalization |
| GPU required | No | Yes (large) | Yes (small) |
| Model size | N/A | 2B–4B parameters | 150M parameters |

### Utterance Likelihood (Miniclip Domain)

For the grocery shopping domain, utterances are scored via LM log-probabilities:

$$P(u_t \mid g, \pi, s) = \frac{1}{|C|} \sum_{c \in C} P(u_t \mid \text{prompt}(c))$$

where $C$ enumerates subsets of future actions, and each prompt is a few-shot translation from command to natural language.

| Backend | Scoring method | Default model |
|---|---|---|
| Local (Qwen3) | Forward pass + log-softmax | `Qwen/Qwen3-0.6B` |
| Azure OpenAI | Completions API + logprobs | `gpt-4.1` |

---

## Setup

### Environment

```bash
conda create -n cs6208 python=3.11 -y
conda activate cs6208
```

### Core dependencies

```bash
pip install numpy matplotlib pillow imageio imageio-ffmpeg
```

### Overcooked rendering

```bash
pip install pygame gymnasium ipywidgets
```

### Qwen3-VL vision model (requires GPU)

```bash
pip install torch transformers accelerate
```

### CLIP vision model (lighter GPU requirement)

```bash
pip install torch transformers
# Uses openai/clip-vit-base-patch32 by default (~150M params)
```

### Install Overcooked-AI (editable, from local source)

```bash
cd overcooked_ai
pip install -e .
cd ..
```

> **Note:** On V100 GPUs (compute capability 7.0), use `torch==2.4.0` with CUDA 12.1:
> ```bash
> pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Quick Start

### 1. Run the four-goal experiment (symbolic only — no GPU needed)

```bash
conda activate cs6208
python render_four_goals_video.py
```

**Output**: `output/four_goals_video_symbolic.mp4` — a three-panel video:
- Left: Overcooked game rendering
- Middle: Symbolic (miniclip action model) posterior curves
- Right: Qwen3-VL posterior (flat, since `--use-qwen` not passed)

### 2. Run with CLIP vision (requires GPU, lightweight)

```bash
python render_four_goals_video.py --use-clip
```

**Output**: `output/four_goals_video_clip.mp4` — right panel shows CLIP posterior.

### 3. Run with Qwen3-VL vision (requires GPU, heavier)

```bash
python render_four_goals_video.py --use-qwen
```

**Output**: `output/four_goals_video_qwen.mp4` — both panels active.

### 4. Run the miniclip baseline (grocery shopping)

```bash
cd miniclips_python/
python simulation.py
```

**Output** (printed to stdout):
```
Posterior after 2 observations: {'greek_salad': 0.272, 'veggie_burger': 0.45, 'fried_rice': 0.006, 'burrito_bowl': 0.272}
Posterior after 3 observations: {'greek_salad': 0.011, 'veggie_burger': 0.017, 'fried_rice': 0.03, 'burrito_bowl': 0.942}
```

---

## Scripts Reference

### `render_four_goals_video.py` — Main 4-Goal Experiment

Renders a scripted Overcooked game where Player 1 makes **Fried Rice Soup** while two inference methods track the posterior over 4 candidate recipes.

| Argument | Default | Description |
|---|---|---|
| `--use-qwen` | `False` | Enable Qwen3-VL vision-based posterior |
| `--use-clip` | `False` | Enable CLIP vision-based posterior |
| `--model` | `Qwen/Qwen3-VL-2B-Instruct` | Qwen3-VL model name |
| `--clip-model` | `openai/clip-vit-base-patch32` | CLIP model name |
| `--clip-temperature` | `0.5` | CLIP softmax temperature (lower = sharper) |
| `--qwen-interval` | `3` | Query Qwen every N steps when no action occurs |
| `--fps` | `3` | Video frame rate |
| `--output` | auto | Output video path |
| `--debug` | `False` | Save per-query debug info to `output/debugging/` |

**Inputs**: None (self-contained; uses `four_goals` layout)

**Outputs**:
- `output/four_goals_video_symbolic.mp4` / `_qwen.mp4`
- `output/four_goals_video_symbolic.gif` / `_qwen.gif`
- `output/four_goals_frames/frame_XXXX.png` (individual frames)

```bash
# Fast symbolic-only run
python render_four_goals_video.py

# With CLIP vision model (lightweight, ~150M params)
python render_four_goals_video.py --use-clip

# CLIP with custom temperature (sharper action classification)
python render_four_goals_video.py --use-clip --clip-temperature 0.3

# With Qwen3-VL vision model (heavier, ~2B params)
python render_four_goals_video.py --use-qwen --model Qwen/Qwen3-VL-2B-Instruct

# Custom FPS and output path
python render_four_goals_video.py --use-qwen --fps 5 --output my_video.mp4

# Debug mode: saves query details + input images
python render_four_goals_video.py --use-clip --debug
```

---

### `render_inference_video.py` — 2-Recipe Demo

Renders inference over 2 recipes (Fried Rice Soup vs Tomato Cucumber Soup) on the `mixed_ingredients` layout.

| Argument | Default | Description |
|---|---|---|
| `--use-qwen` | `False` | Enable Qwen3-VL |
| `--model` | `Qwen/Qwen3-VL-2B-Instruct` | Qwen model |
| `--qwen-interval` | `3` | Qwen query interval |
| `--fps` | `3` | Frame rate |
| `--output` | auto | Output path |

**Outputs**: `output/inference_video_*.mp4`, `output/inference_video_frames/`

```bash
python render_inference_video.py
python render_inference_video.py --use-qwen
```

---

### `render_test_video.py` — 3-Ingredient Test

Renders one agent making a 3-ingredient soup (olive, feta_cheese, frozen_peas). No inference — just verifies the rendering pipeline.

**No arguments.** Output: `output/test_video.mp4`, `output/test_video_frames/`

```bash
python render_test_video.py
```

---

### `render_test_video_all10.py` — 10-Ingredient Mega-Soup

Agent fetches all 10 ingredients, cooks, plates, and serves. Tests the full ingredient set.

**No arguments.** Output: `output/test_video_all10.mp4`, `output/test_video_all10_frames/`

```bash
python render_test_video_all10.py
```

---

### `analyze_image.py` — Qwen3-VL Image Analysis

Standalone script to query Qwen3-VL about any image.

| Argument | Default | Description |
|---|---|---|
| `--image` | `data/test_images/kitchen.png` | Image file path |
| `--prompt` | `"Describe this image..."` | Question to ask |
| `--model` | `Qwen/Qwen3-VL-2B-Instruct` | Model name |

```bash
python analyze_image.py --image output/four_goals_frames/frame_0040.png \
  --prompt "What ingredients has the blue-hat chef picked up?"
```

---

### `miniclips_python/simulation.py` — Miniclip Regression Tests

Runs action-only and utterance+action Bayesian inference on grocery shopping goals.

```bash
cd miniclips_python/

# Action-only (no LLM)
python simulation.py

# With utterance likelihood (loads Qwen3-0.6B)
python simulation.py --utterance

# Override model
LOCAL_MODEL_NAME=Qwen/Qwen3-1.7B python simulation.py --utterance
```

**Example output:**

```
── Example 1: Two action observations ──
  obs 1: get(onion)
  posterior: {'greek_salad': 0.272, 'veggie_burger': 0.450, 'fried_rice': 0.006, 'burrito_bowl': 0.272}
  obs 2: get(tomato)
  posterior: {'greek_salad': 0.272, 'veggie_burger': 0.450, 'fried_rice': 0.006, 'burrito_bowl': 0.272}

── Example 2: Three action observations ──
  obs 1: get(rice)
  obs 2: get(onion)
  obs 3: get(soy_sauce)
  posterior: {'greek_salad': 0.011, 'veggie_burger': 0.017, 'fried_rice': 0.030, 'burrito_bowl': 0.942}
```

---

### `miniclips_python/demo_overcooked_inference.py` — Overcooked Integration Demo

Demonstrates the full pipeline with Overcooked recipes.

| Argument | Default | Description |
|---|---|---|
| `--use-qwen` | `False` | Use Qwen for likelihood |
| `--model` | `Qwen/Qwen3-0.6B` | Qwen text model |
| `--layout` | `simple_o_t` | Overcooked layout name |
| `--live` | `False` | Run live game simulation |

```bash
cd miniclips_python/

# Symbolic demos
python demo_overcooked_inference.py

# With Qwen text likelihood
python demo_overcooked_inference.py --use-qwen

# Live game on specific layout
python demo_overcooked_inference.py --live --layout mixed_ingredients
```

---

## Examples

### Example 1: Qwen3-VL Vision Likelihood — How a Query Works

At each action step, the system queries Qwen3-VL with **all game frames up to that point** (one frame per step) plus a detailed prompt describing the recipes and kitchen layout. Here is a concrete example at **step 35**, where the player has just picked up soy sauce (their third ingredient after rice and onion):

**Game state at step 35:**
- Player 1 has picked up rice (step 7), added it to pot (step 11), picked up onion (step 21), added it to pot (step 31), and just picked up soy sauce (step 35).
- The pot contains rice and onion. Player 1 is holding soy sauce, walking toward the pot.

**Input images:** 36 frames (frame 0 through frame 35), each a 75px-per-tile top-down render of the kitchen.

**Prompt sent to Qwen3-VL:**

```
You are watching a cooperative cooking game from a top-down view.
Two chefs (blue hat and green hat) are in a kitchen with ingredient dispensers on the walls.
The blue-hat chef (Player 1) is trying to make one of four possible soups.
Each soup requires exactly 5 ingredients picked up from dispensers and added to a pot.

The four candidate goals and their recipes are:

Recipe A - "Greek Salad Soup":
  Ingredients: tomato, olive, cucumber, onion, feta cheese
  (Player picks from: Tomato dispenser on top wall, Olive dispenser on left wall,
   Cucumber dispenser on top wall, Onion dispenser on top wall, Feta Cheese dispenser on left wall)

Recipe B - "Veggie Burger Soup":
  Ingredients: hamburger bun, tomato, onion, cucumber, frozen carrots
  (Player picks from: Hamburger Bun dispenser on right wall, Tomato on top wall,
   Onion on top wall, Cucumber on top wall, Frozen Carrots on bottom wall)

Recipe C - "Fried Rice Soup":
  Ingredients: rice, onion, soy sauce, frozen peas, frozen carrots
  (Player picks from: Rice dispenser on top wall, Onion on top wall,
   Soy Sauce on right wall, Frozen Peas on bottom wall, Frozen Carrots on bottom wall)

Recipe D - "Burrito Bowl Soup":
  Ingredients: rice, olive, feta cheese, onion, tomato
  (Player picks from: Rice on top wall, Olive on left wall,
   Feta Cheese on left wall, Onion on top wall, Tomato on top wall)

Kitchen layout:
- Top wall (left to right): Onion (white bulb), Tomato (red), Cucumber (green), Rice (white grains)
- Left wall (top to bottom): Olive (purple/dark), Feta Cheese (yellow block)
- Right wall (top to bottom): Hamburger Bun (brown), Soy Sauce (dark bottle)
- Bottom wall: Dish dispenser, Frozen Peas (green bag), Frozen Carrots (orange bag), Serving window
- Top right corner: Cooking pot

I'm showing you the last several frames of the game in chronological order.
Observe which dispensers the blue-hat player visits, which ingredients they pick up,
and which items they add to the pot. Based on these observations,
estimate the probability that they are making each recipe.

Answer ONLY in this exact format (numbers must sum to 1.0):
Recipe A: <probability>
Recipe B: <probability>
Recipe C: <probability>
Recipe D: <probability>
```

**Expected Qwen3-VL response:**

```
Recipe A: 0.05
Recipe B: 0.05
Recipe C: 0.80
Recipe D: 0.10
```

The model should assign high probability to Recipe C (Fried Rice Soup) because the observed ingredients (rice, onion, soy sauce) are unique to that recipe. Recipe D (Burrito Bowl) shares rice and onion but not soy sauce. The parsed probabilities are used directly as the Qwen posterior at this step.

**Corresponding symbolic posterior at step 35:** `[0.000, 0.000, 0.992, 0.008]` — the symbolic model is already near-certain because soy sauce is only in the Fried Rice plan.

---

### Example 2: CLIP Vision Likelihood — Frame-to-Action Classification

Unlike Qwen, the CLIP model does not directly predict recipe probabilities. Instead, it classifies **what action is happening in each frame** and then maps those action probabilities to recipe likelihoods via Bayesian updates. Here is a concrete walkthrough of the process.

#### Step 1: CLIP Similarity between Pickup Frames and Action Prompts

We render 10 game frames — one for each ingredient pickup — and compute cosine similarity between each frame's CLIP image embedding and 10 text prompt embeddings of the form *"The blue hat chef is picking up {ingredient} from the {ingredient} dispenser"*.

**Raw Cosine Similarity** $s_{i,j} = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{t}}_j$ (rows = images, columns = text prompts):

| Image \\ Text | onion | tomato | cucumber | rice | olive | feta cheese | hamburger bun | soy sauce | frozen peas | frozen carrots |
|---|---|---|---|---|---|---|---|---|---|---|
| **onion** | 0.295 | 0.308 | 0.275 | 0.285 | 0.287 | 0.272 | 0.280 | 0.266 | 0.286 | 0.305 |
| **tomato** | 0.252 | 0.292 | 0.251 | 0.253 | 0.251 | 0.233 | 0.256 | 0.235 | 0.248 | 0.271 |
| **cucumber** | 0.273 | 0.286 | 0.282 | 0.260 | 0.271 | 0.246 | 0.270 | 0.244 | 0.275 | 0.263 |
| **rice** | 0.265 | 0.260 | 0.257 | 0.267 | 0.271 | 0.248 | 0.273 | 0.248 | 0.265 | 0.268 |
| **olive** | 0.264 | 0.276 | 0.242 | 0.257 | 0.265 | 0.242 | 0.253 | 0.244 | 0.263 | 0.284 |
| **feta cheese** | 0.299 | 0.285 | 0.271 | 0.294 | 0.289 | 0.295 | 0.297 | 0.275 | 0.300 | 0.316 |
| **hamburger bun** | 0.287 | 0.280 | 0.281 | 0.300 | 0.297 | 0.289 | 0.305 | 0.284 | 0.295 | 0.310 |
| **soy sauce** | 0.268 | 0.275 | 0.255 | 0.279 | 0.282 | 0.266 | 0.286 | 0.270 | 0.266 | 0.284 |
| **frozen peas** | 0.270 | 0.257 | 0.263 | 0.274 | 0.267 | 0.266 | 0.275 | 0.257 | 0.274 | 0.281 |
| **frozen carrots** | 0.271 | 0.267 | 0.246 | 0.279 | 0.266 | 0.267 | 0.288 | 0.249 | 0.269 | 0.302 |

The raw cosine similarities are all in the narrow range [0.23, 0.32] — CLIP was not trained on pixel-art game frames, so the absolute similarities are low and tightly clustered. However, the **relative differences** within each row carry useful signal.

#### Step 2: Apply Temperature-Scaled Softmax

To convert similarities to probabilities, we apply softmax with the model's learned logit scale ($\alpha = 100$):

$$P(a_j \mid I_i) = \frac{\exp(\alpha \cdot s_{i,j})}{\sum_{k} \exp(\alpha \cdot s_{i,k})}$$

This amplifies the small differences. The resulting **action probability** matrix:

| Image \\ Text | onion | tomato | cucumber | rice | olive | feta cheese | hamburger bun | soy sauce | frozen peas | frozen carrots |
|---|---|---|---|---|---|---|---|---|---|---|
| **onion** | 0.106 | 0.406 | 0.014 | 0.041 | 0.049 | 0.011 | 0.023 | 0.006 | 0.043 | 0.301 |
| **tomato** | 0.014 | 0.815 | 0.013 | 0.016 | 0.013 | 0.002 | 0.022 | 0.003 | 0.009 | 0.093 |
| **cucumber** | 0.096 | 0.344 | 0.234 | 0.026 | 0.075 | 0.006 | 0.072 | 0.005 | 0.110 | 0.033 |
| **rice** | 0.097 | 0.058 | 0.046 | 0.116 | 0.183 | 0.018 | 0.225 | 0.018 | 0.101 | 0.139 |
| **olive** | 0.067 | 0.219 | 0.008 | 0.034 | 0.076 | 0.008 | 0.022 | 0.009 | 0.059 | 0.500 |
| **feta cheese** | 0.094 | 0.025 | 0.006 | 0.060 | 0.035 | 0.065 | 0.079 | 0.009 | 0.107 | 0.520 |
| **hamburger bun** | 0.035 | 0.018 | 0.019 | 0.129 | 0.100 | 0.043 | 0.212 | 0.027 | 0.078 | 0.339 |
| **soy sauce** | 0.043 | 0.078 | 0.011 | 0.120 | 0.169 | 0.032 | 0.254 | 0.051 | 0.032 | 0.210 |
| **frozen peas** | 0.087 | 0.025 | 0.044 | 0.139 | 0.068 | 0.063 | 0.155 | 0.024 | 0.138 | 0.259 |
| **frozen carrots** | 0.031 | 0.020 | 0.002 | 0.064 | 0.017 | 0.020 | 0.158 | 0.003 | 0.023 | 0.663 |

**Key observations:**
- **tomato** (0.815) and **frozen carrots** (0.663) have strong correct matches — these ingredients have distinctive visual features (bright red, orange bag) that CLIP can identify
- **cucumber** (0.234) and **hamburger bun** (0.212) show moderate correct identification
- Many frames are confused with "tomato" or "frozen carrots" because their visual features dominate the pixel-art kitchen
- The diagonal mean is 0.247, well above the uniform baseline of 0.100

#### Step 3: From Action Probabilities to Recipe Likelihoods (Action Marginalization)

Given the action probability $P(a_j \mid I_t)$ for a frame, we compute the recipe likelihood by marginalizing over the latent action, using the plan-based action distribution $P(a \mid g, s_{t-1})$:

$$P(I_t \mid g, s_{t-1}) \propto \sum_{a \in \mathcal{A}} P(a \mid I_t) \cdot P(a \mid g, s_{t-1})$$

This is the dot product between CLIP's action belief and the plan model's action distribution.

**Concrete example**: Suppose at frame $t$ the player just picked up rice (it's the first action, so no actions have been completed yet: $s_{t-1} = \emptyset$). CLIP outputs (from the table above, "rice" row):

$$P(\text{pick\_up(onion)} \mid I_t) = 0.097, \quad P(\text{pick\_up(rice)} \mid I_t) = 0.116, \quad P(\text{pick\_up(olive)} \mid I_t) = 0.183, \quad \ldots$$

For each goal, the plan model assigns high probability to its **planned actions** and low (noise-only) probability to everything else. At the start, each 5-ingredient recipe has 5 planned pick_up actions plus pick_up_dish = 6 planned actions out of $|\mathcal{A}|$ total. With $\epsilon = 0.05$:

- Planned action: $P(a \mid g) = \frac{0.95}{6} + \frac{0.05}{|\mathcal{A}|} \approx 0.160$
- Unplanned action: $P(a \mid g) = \frac{0.05}{|\mathcal{A}|} \approx 0.002$

The dot product for **Fried Rice Soup** (planned: rice, onion, soy_sauce, frozen_peas, frozen_carrots, dish) gives high weight to $P(\text{pick\_up(rice)} \mid I_t)$, $P(\text{pick\_up(onion)} \mid I_t)$, etc. — all the actions CLIP detects as likely.

For **Greek Salad Soup** (planned: tomato, olive, cucumber, onion, feta_cheese, dish), pick_up(rice) is **unplanned**, so it contributes only $0.116 \times 0.002$ instead of $0.116 \times 0.160$.

The result: goals whose planned actions align with CLIP's detected actions receive higher likelihood, while goals whose plans don't match get suppressed — even from a single noisy frame.

#### Step 4: Bayesian Accumulation Across Frames

The key insight is that a **single frame** is ambiguous, but accumulated evidence over many frames becomes decisive. After each frame, the posterior is updated:

$$P(g \mid I_{1:t}) \propto P(I_t \mid g) \cdot P(g \mid I_{1:t-1})$$

Over the full 115-step game, the CLIP posterior evolves as follows (from the `--use-clip` run):

| Step | Event | Symbolic Posterior (C) | CLIP Posterior (C) |
|---|---|---|---|
| 0 | Start | 0.250 | 0.250 |
| 7 | pick_up(rice) | 0.494 | — |
| 11 | add_to_pot(rice) | 0.500 | — |
| 21 | pick_up(onion) | 0.500 | — |
| 35 | pick_up(soy_sauce) | 0.992 | — |
| 39 | add_to_pot(soy_sauce) | 1.000 | — |

The symbolic model converges quickly because it observes exact action labels. CLIP processes **every frame** (including "moving" frames), so it accumulates gradual evidence. Even though individual frame classifications are noisy, the consistent directional signal (player visiting specific dispensers) drives the posterior toward the correct recipe across 100+ frames.

The full similarity tables are saved in `output/clip_tables/raw_similarity.txt` and `output/clip_tables/normalized_similarity.txt`. The 10 pickup frames are in `output/clip_tables/pickup_frames_cropped/`.

---

### Example 3: Utterance Likelihood — How It Works

In the miniclip grocery shopping domain, the system can also observe **natural language utterances** (e.g., a partner saying "Can you get the stuff in the frozen section?"). Here is how utterance likelihood is computed.

**Setup:** The player has not picked up any items yet (`state = {}`). The observer hears:

> "Can you get the stuff in the frozen section?"

**Step 1: Enumerate future action subsets (commands)**

For each goal, the system computes all unfinished actions and generates **command candidates** — subsets of size 1–2 from future actions. For `fried_rice`:

```
Future actions: [get(rice), get(onion), get(soy_sauce), get(frozen_peas), get(frozen_carrots), checkout()]

Command candidates (size 1-2):
  [get(rice)]
  [get(onion)]
  [get(soy_sauce)]
  [get(frozen_peas)]
  [get(frozen_carrots)]
  [checkout()]
  [get(rice), get(onion)]
  [get(rice), get(soy_sauce)]
  [get(rice), get(frozen_peas)]
  ... (15 total subsets)
```

**Step 2: Build few-shot prompts for each command**

Each command candidate is translated into a few-shot prompt using examples. For command `[get(frozen_peas), get(frozen_carrots)]`:

```
Input: get(apple)
Output: Can you get the apple?
Input: get(bread)
Output: Could you find some bread?
Input: get(cheddar_cheese)
Output: Go grab a block of that cheese.
Input: get(green_tea)
Output: Add some tea to the cart.
Input: get(checkout())
Output: Let's checkout.
Input: get(frozen_broccoli) get(frozen_cauliflower)
Output: We'll need frozen broccoli and cauliflower.
...
Input: get(frozen_peas) get(frozen_carrots)
Output:
```

**Step 3: Score the utterance**

Using a local LM (Qwen3-0.6B), compute $\log P(\text{"Can you get the stuff in the frozen section?"} \mid \text{prompt})$ for each prompt. This uses a forward pass through the model — the prompt is the context, and the utterance tokens are scored via log-softmax:

$$\log P(u \mid \text{prompt}) = \sum_{j=1}^{|u|} \log P(u_j \mid \text{prompt}, u_{1:j-1})$$

**Step 4: Average over all command candidates**

The utterance likelihood for a goal is the average over all its command prompts:

$$P(u \mid g) = \frac{1}{|C_g|} \sum_{c \in C_g} P(u \mid \text{prompt}(c))$$

**Result:** Commands involving frozen items (like `[get(frozen_peas), get(frozen_carrots)]`) produce **higher** $P(u \mid \text{prompt})$ scores because the LM continuation naturally maps "stuff in the frozen section" → frozen items. So `fried_rice` (which has 2 frozen items and 15 command candidates containing them) gets the highest average likelihood, while `greek_salad` (no frozen items) gets low scores, pushing the posterior toward `fried_rice`.

After the Bayesian update:

```
posterior = {
    'greek_salad':   0.048,
    'veggie_burger': 0.048,
    'fried_rice':    0.856,
    'burrito_bowl':  0.048
}
```

---

## Data & Outputs

### Layout Files

Overcooked grid layouts are in `overcooked_ai/src/overcooked_ai_py/data/layouts/`. Key layouts:

| Layout | Grid | Description |
|---|---|---|
| `four_goals` | 11×5 | 4-recipe experiment with all 10 dispensers |
| `mixed_ingredients` | 5×4 | 2-recipe demo (onion, tomato, cucumber, rice) |
| `all_ingredients_test` | 9×5 | Test: 3 ingredients (olive, feta, frozen_peas) |
| `all_ten_ingredients` | 13×10 | Test: all 10 ingredients |

### Output Files

After running the scripts, the `output/` directory contains:

| File | Generated by |
|---|---|
| `four_goals_video_symbolic.mp4` | `render_four_goals_video.py` |
| `four_goals_video_clip.mp4` | `render_four_goals_video.py --use-clip` |
| `four_goals_video_qwen.mp4` | `render_four_goals_video.py --use-qwen` |
| `inference_video_symbolic.mp4` | `render_inference_video.py` |
| `inference_video_qwen.mp4` | `render_inference_video.py --use-qwen` |
| `test_video.mp4` | `render_test_video.py` |
| `test_video_all10.mp4` | `render_test_video_all10.py` |
| `four_goals_frames/frame_XXXX.png` | Individual frames (4-goal) |
| `inference_video_frames/frame_XXXX.png` | Individual frames (2-recipe) |

---

## Reference

This implementation is based on the Julia tutorial notebook (`tutorial_colab.ipynb`) from:

> Wong, L., Grand, G., Lew, A., Goodman, N., Mansinghka, V., Andreas, J., & Tenenbaum, J. (2023). From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought. *arXiv:2306.12672*.
