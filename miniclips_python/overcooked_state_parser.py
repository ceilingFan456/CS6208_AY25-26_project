"""Extract symbolic observations from Overcooked state transitions.

Converts low-level Overcooked game events (move, interact) into
high-level symbolic actions that the miniclips Bayesian inference
engine can process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class OvercookedObservation:
    """A symbolic observation extracted from one Overcooked timestep.

    kind : "act" or "state_description"
    value : symbolic action string or textual state description
    player_index : which player this observation is about
    timestep : the game timestep
    """
    kind: str
    value: str
    player_index: int
    timestep: int


def extract_action_from_events(
    events_infos: Dict,
    player_index: int,
    prev_state,
    new_state,
) -> Optional[str]:
    """Derive a high-level symbolic action from Overcooked event_infos.

    Returns one of:
      pick_up(onion), pick_up(tomato), ...
      add_to_pot(onion), add_to_pot(tomato), ...
      start_cooking, pick_up_dish, pick_up_soup, serve_soup, wait
    """
    ei = events_infos
    pi = player_index

    # Soup delivery
    if ei.get("soup_delivery", [False, False])[pi]:
        return "serve_soup"

    # Soup pickup (player picks cooked soup from pot with dish)
    if ei.get("soup_pickup", [False, False])[pi]:
        return "pick_up_soup"

    # Ingredient potting (adding ingredient to pot)
    for ing in ["onion", "tomato", "cucumber", "rice", "olive",
                "feta_cheese", "hamburger_bun", "soy_sauce",
                "frozen_peas", "frozen_carrots"]:
        key = f"potting_{ing}"
        if ei.get(key, [False, False])[pi]:
            return f"add_to_pot({ing})"

    # Dish pickup
    if ei.get("dish_pickup", [False, False])[pi]:
        return "pick_up_dish"

    # Ingredient pickups
    for ing in ["onion", "tomato", "cucumber", "rice", "olive",
                "feta_cheese", "hamburger_bun", "soy_sauce",
                "frozen_peas", "frozen_carrots"]:
        key = f"{ing}_pickup"
        if ei.get(key, [False, False])[pi]:
            return f"pick_up({ing})"

    return None  # movement or no-op


def describe_state_text(state, mdp, player_index: int = 0) -> str:
    """Produce a natural-language description of the current Overcooked state.

    This text is used as input for Qwen-based likelihood estimation.
    """
    player = state.players[player_index]
    lines = []

    # Player info
    held = "nothing"
    if player.has_object():
        obj = player.get_object()
        if obj.name == "soup":
            held = f"a cooked soup ({', '.join(obj.ingredients)})"
        else:
            held = obj.name
    lines.append(f"Player {player_index} is at position {player.position}, holding {held}.")

    # Pot states
    pot_locs = mdp.get_pot_locations()
    for pot_pos in pot_locs:
        if state.has_object(pot_pos):
            soup = state.get_object(pot_pos)
            if soup.is_ready:
                lines.append(
                    f"Pot at {pot_pos}: soup is READY ({', '.join(soup.ingredients)})."
                )
            elif soup.is_cooking:
                lines.append(
                    f"Pot at {pot_pos}: cooking ({', '.join(soup.ingredients)}), "
                    f"tick {soup._cooking_tick}/{soup.cook_time}."
                )
            else:
                lines.append(
                    f"Pot at {pot_pos}: contains {', '.join(soup.ingredients)} (idle, not cooking yet)."
                )
        else:
            lines.append(f"Pot at {pot_pos}: empty.")

    # Available orders
    if hasattr(state, "all_orders") and state.all_orders:
        order_strs = []
        for recipe in state.all_orders:
            order_strs.append(", ".join(recipe.ingredients))
        lines.append(f"Current orders: {'; '.join(order_strs)}.")

    # Counter objects
    counter_objects = []
    for pos, obj in state.objects.items():
        if pos not in pot_locs:
            counter_objects.append(f"{obj.name} at {pos}")
    if counter_objects:
        lines.append(f"Items on counters: {', '.join(counter_objects)}.")

    # Other player
    for i, p in enumerate(state.players):
        if i != player_index:
            other_held = "nothing"
            if p.has_object():
                o = p.get_object()
                other_held = o.name if o.name != "soup" else f"soup ({', '.join(o.ingredients)})"
            lines.append(f"Other player (Player {i}) at {p.position}, holding {other_held}.")

    return " ".join(lines)


def extract_trajectory_observations(
    trajectory: Dict,
    player_index: int = 0,
    include_state_descriptions: bool = True,
    mdp=None,
) -> List[OvercookedObservation]:
    """Extract symbolic observations from an Overcooked trajectory dict.

    Parameters
    ----------
    trajectory : dict
        Standard Overcooked trajectory with ep_states, ep_actions, ep_infos.
    player_index : int
        Which player to track.
    include_state_descriptions : bool
        If True, also emit "state_description" observations for Qwen.
    mdp : OvercookedGridworld, optional
        Required if include_state_descriptions is True.

    Returns
    -------
    List of OvercookedObservation
    """
    states = trajectory["ep_states"]
    infos = trajectory.get("ep_infos", [])
    observations = []

    for t in range(len(infos)):
        prev_state = states[t]
        new_state = states[t + 1]
        event_infos = infos[t].get("event_infos", {})

        # Extract symbolic action
        action = extract_action_from_events(
            event_infos, player_index, prev_state, new_state
        )
        if action:
            observations.append(
                OvercookedObservation("act", action, player_index, t)
            )

        # State description for Qwen
        if include_state_descriptions and mdp is not None:
            desc = describe_state_text(new_state, mdp, player_index)
            observations.append(
                OvercookedObservation("state_description", desc, player_index, t)
            )

    return observations
