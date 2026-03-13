#!/usr/bin/env python3
"""Goal definitions and priors for our baseline.

- `GOALS = [...]`
- `PLANS = Dict(...)`
- `ACTIONS = ...; push!(ACTIONS, "wait()")`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class Goal:
    """Julia-equivalent goal (name + plan)."""

    name: str
    plan: Dict[str, List[str]]


## Set of possible goals (copied from Julia notebook)
GOALS = [
    "greek_salad",
    "veggie_burger",
    "fried_rice",
    "burrito_bowl",
]


PLANS: Dict[str, Dict[str, List[str]]] = {
    "greek_salad": {
        "get(tomato)": [],
        "get(olives)": [],
        "get(cucumber)": [],
        "get(onion)": [],
        "get(feta_cheese)": [],
        "checkout()": [
            "get(tomato)",
            "get(olives)",
            "get(cucumber)",
            "get(onion)",
            "get(feta_cheese)",
        ],
    },
    "veggie_burger": {
        "get(hamburger_bun)": [],
        "get(tomato)": [],
        "get(onion)": [],
        "get(lettuce)": [],
        "get(frozen_patty)": [
            "get(hamburger_bun)",
            "get(tomato)",
            "get(onion)",
            "get(lettuce)",
        ],
        "checkout()": [
            "get(hamburger_bun)",
            "get(tomato)",
            "get(onion)",
            "get(lettuce)",
            "get(frozen_patty)",
        ],
    },
    "fried_rice": {
        "get(rice)": [],
        "get(onion)": [],
        "get(soy_sauce)": [],
        "get(frozen_peas)": ["get(rice)", "get(onion)", "get(soy_sauce)"],
        "get(frozen_carrots)": ["get(rice)", "get(onion)", "get(soy_sauce)"],
        "checkout()": [
            "get(rice)",
            "get(onion)",
            "get(soy_sauce)",
            "get(frozen_peas)",
            "get(frozen_carrots)",
        ],
    },
    "burrito_bowl": {
        "get(rice)": [],
        "get(black_beans)": [],
        "get(cotija_cheese)": [],
        "get(onion)": [],
        "get(tomato)": [],
        "checkout()": [
            "get(rice)",
            "get(black_beans)",
            "get(cotija_cheese)",
            "get(onion)",
            "get(tomato)",
        ],
    },
}

## Actions objects
ACTIONS = sorted({act for plan in PLANS.values() for act in plan.keys()})
ACTIONS.append("wait()") ## same as the notebook. 

# Candidate goal objects
CANDIDATE_GOALS = [Goal(name=g, plan=PLANS[g]) for g in GOALS]

## Uniform prior over goals 
PRIOR = np.full(len(GOALS), 1.0 / len(GOALS), dtype=float)
