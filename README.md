# Mini-CLIPS + Overcooked-AI: Bayesian Goal Inference

A Python implementation of Bayesian goal inference combining [Mini-CLIPS](https://github.com/probcomp/GenGPT3.jl) (Cooperative Language-Guided Inverse Plan Search) with the [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) cooperative cooking environment.

An observer watches a player's **actions** (picking up ingredients, adding to pot, cooking, serving) and uses **sequential Bayesian posterior updates** to infer which recipe the player is making. Two likelihood models are supported:

1. **Symbolic action model** — plans as dependency DAGs, plan-based likelihood
2. **Qwen3-VL vision model** — directly estimates recipe probabilities from rendered game frames

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

Instead of symbolic actions, the vision model receives the **last 10 rendered game frames** at each action step and directly estimates:

$$P(\text{recipe} \mid \text{frames}_{t-9:t})$$

The model is prompted with the full recipe descriptions, kitchen layout, and dispenser locations.

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

### 2. Run with Qwen3-VL vision (requires GPU)

```bash
python render_four_goals_video.py --use-qwen
```

**Output**: `output/four_goals_video_qwen.mp4` — both panels active.

### 3. Run the miniclip baseline (grocery shopping)

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
| `--model` | `Qwen/Qwen3-VL-2B-Instruct` | Qwen3-VL model name |
| `--qwen-interval` | `3` | Query Qwen every N steps when no action occurs |
| `--fps` | `3` | Video frame rate |
| `--output` | auto | Output video path |

**Inputs**: None (self-contained; uses `four_goals` layout)

**Outputs**:
- `output/four_goals_video_symbolic.mp4` / `_qwen.mp4`
- `output/four_goals_video_symbolic.gif` / `_qwen.gif`
- `output/four_goals_frames/frame_XXXX.png` (individual frames)

```bash
# Fast symbolic-only run
python render_four_goals_video.py

# With Qwen3-VL vision model
python render_four_goals_video.py --use-qwen --model Qwen/Qwen3-VL-2B-Instruct

# Custom FPS and output path
python render_four_goals_video.py --use-qwen --fps 5 --output my_video.mp4
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

### Example 2: Utterance Likelihood — How It Works

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
