"""
drives.py — Phase 8.1 / 8.15 / 8.2: Drive Architecture

Implements the motivational drive architecture for Project Genesis agents.

Architecture (Phase 8 road-map):
  Biological needs (hunger, thirst, energy, injury, temperature)
        ↓  EMA smoothing — every tick
  Biological Drive Tensions   [0.0, 1.0]       ← Phase 8.1
        ↓
  Memory Importance Scoring                    ← Phase 8.15
        ↓
  Emotional Dynamics           [0.0, 1.0]       ← Phase 8.2  (every 10 ticks)
        ↓  via drive interactions, not discrete states
  Relationship Graph           [-1.0, 1.0]      ← Phase 8.3  (every 30 ticks)
        ↓
  Internal Value System                         ← Phase 8.4
        ↓
  Identity Formation                            ← Phase 8.5
        ↓
  evaluate_utility()   (existing planner — unchanged logic)
        ↓
  Emergent behavior

Design rules
------------
1.  Never hardcode behaviour.  Drives produce pressure; behavior emerges.
2.  Everything is continuous.  No boolean emotion flags.
3.  The genome is unchanged.  Genes influence *sensitivity*, *decay*, and
    *threshold* — not the existence of drives.
4.  Clock-tiered update:
      Biological   — every tick           (Phase 8.1)
      Emotional    — every 10 ticks       (Phase 8.2)
      Relationship — every 30 ticks       (Phase 8.3)
      Personality  — every 100 ticks      (Phase 8.5)
5.  Emotions are outputs of drive interactions, not independent variables.
      High hunger + repeated failure → frustration rises
      High fear + isolation          → anxiety-like pressure
      High contentment + known kin   → secure attachment

Phase 8.2: Drive Interaction Table
-----------------------------------
| Drive combination                    | Emergent pressure              |
|--------------------------------------|--------------------------------|
| high hunger_tension + unmet_ticks    | frustration builds             |
| high fear + no nearby known agents   | isolation amplifies fear       |
| all tensions low + recent kin nearby | contentment high, longing low  |
| boredom high + fear low              | exploration utility amplified  |
| grief high + attachment high         | rest utility amplified         |
| longing high + no known kin seen     | loneliness rises               |
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


# ---------------------------------------------------------------------------
# Rolling history helper
# ---------------------------------------------------------------------------

_MAX_HISTORY = 20   # ticks of history retained per emotional drive


def _append_history(hist: List[float], value: float) -> None:
    """Appends value to a capped history list in-place."""
    hist.append(float(value))
    if len(hist) > _MAX_HISTORY:
        del hist[0]


def _rolling_mean(hist: List[float]) -> float:
    return float(np.mean(hist)) if hist else 0.0


def _rolling_max(hist: List[float]) -> float:
    return float(max(hist)) if hist else 0.0


# ---------------------------------------------------------------------------
# Biological Drive Tensions  (Phase 8.1)
# ---------------------------------------------------------------------------

@dataclass
class DriveState:
    """
    Per-agent motivational drive state.

    Phase 8.1  — biological tensions (smoothed EMA of raw needs).
    Phase 8.15 — memory importance scoring (see compute_memory_importance).
    Phase 8.2  — emotional dynamics via drive interactions, every 10 ticks.
                 Drive histories: last 20 emotional-clock values each.
    Phase 8.3  — relationship graph added.
    Phase 8.4  — value_weights added.
    Phase 8.5  — identity weights added.

    Contentment is NOT stored here — it is an *output* (absence of pressure)
    computed on demand via the `contentment` property.

    Valence and arousal are likewise computed on-demand properties so they
    never fall out of sync with the underlying tensions.
    """

    # ------------------------------------------------------------------ #
    # Biological tensions  (Phase 8.1)                                    #
    # Smoothed EMA of raw physiological needs.  Range [0.0, 1.0].        #
    # ------------------------------------------------------------------ #
    hunger_tension:    float = 0.0   # ← agent.hunger / 100
    thirst_tension:    float = 0.0   # ← agent.thirst / 100
    exhaustion_tension: float = 0.0  # ← (100 - agent.energy) / 100
    pain_tension:      float = 0.0   # ← agent.injury_level / 100
    thermal_stress:    float = 0.0   # deviation from comfort band [15–25 °C]

    # ------------------------------------------------------------------ #
    # Emotional drives  (Phase 8.2)                                       #
    # Updated every 10 ticks by update_emotional_drives().                #
    # All float in [0.0, 1.0].                                            #
    # ------------------------------------------------------------------ #
    fear:          float = 0.0   # danger proximity + pain spikes
    frustration:   float = 0.0   # sustained unmet needs + repeated failure
    longing:       float = 0.0   # attachment pull (kin, mate, home, group)
    grief:         float = 0.0   # spikes when a known agent dies nearby
    boredom:       float = 0.0   # same action repeated many ticks

    # ------------------------------------------------------------------ #
    # Emotional drive histories  (Phase 8.2)                              #
    # Last _MAX_HISTORY values sampled every 10 ticks.                   #
    # Used to distinguish brief spikes from chronic states.               #
    # ------------------------------------------------------------------ #
    fear_history:        List[float] = field(default_factory=list)
    frustration_history: List[float] = field(default_factory=list)
    longing_history:     List[float] = field(default_factory=list)
    grief_history:       List[float] = field(default_factory=list)
    boredom_history:     List[float] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Supporting counters                                                  #
    # ------------------------------------------------------------------ #
    _same_action_streak: int = 0    # ticks current action has been repeated
    _unmet_hunger_ticks: int = 0    # consecutive ticks hunger_tension > 0.5
    _unmet_thirst_ticks: int = 0    # consecutive ticks thirst_tension > 0.5
    _ticks_since_known_agent: int = 0  # ticks since a known agent was perceived
    _prev_pain: float = 0.0         # pain tension at previous tick (for delta)

    # ------------------------------------------------------------------ #
    # Contentment — read-only output (absence of pressure)                #
    # ------------------------------------------------------------------ #
    @property
    def contentment(self) -> float:
        """
        High when all biological drives are low AND no acute emotional pressure.
        Not stored — computed from the current state to avoid synchronisation.
        Range [0.0, 1.0].
        """
        bio_pressure = (
            self.hunger_tension    * 0.25 +
            self.thirst_tension    * 0.25 +
            self.exhaustion_tension * 0.15 +
            self.pain_tension      * 0.10 +
            self.thermal_stress    * 0.10
        )
        emo_pressure = (
            self.fear        * 0.08 +
            self.frustration * 0.05 +
            self.grief       * 0.02
        )
        return float(np.clip(1.0 - (bio_pressure + emo_pressure) * 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Valence  — emotional positivity [-1, 1]  (computed on-demand)       #
    # ------------------------------------------------------------------ #
    @property
    def valence(self) -> float:
        """
        Overall emotional positivity.
        Positive: contentment drives positive valence.
        Negative: frustration, fear, grief pull valence negative.
        Range [-1.0, 1.0].
        """
        positive = self.contentment * 0.6
        negative = (
            self.fear        * 0.25 +
            self.frustration * 0.40 +
            self.grief       * 0.35
        )
        return float(np.clip(positive - negative, -1.0, 1.0))

    # ------------------------------------------------------------------ #
    # Arousal  — motivational activation [0, 1]  (computed on-demand)     #
    # ------------------------------------------------------------------ #
    @property
    def arousal(self) -> float:
        """
        How activated/energised the agent is.
        High when any drive is strongly elevated.
        Range [0.0, 1.0].
        """
        return float(np.clip(
            max(
                self.fear,
                self.frustration * 0.8,
                self.hunger_tension * 0.9,
                self.thirst_tension * 0.9,
                self.boredom * 0.5,
                self.longing * 0.4,
            ),
            0.0, 1.0
        ))

    # ------------------------------------------------------------------ #
    # Rolling stats (properties — read from histories)                    #
    # ------------------------------------------------------------------ #
    @property
    def fear_mean(self) -> float:
        return _rolling_mean(self.fear_history)

    @property
    def fear_max(self) -> float:
        return _rolling_max(self.fear_history)

    @property
    def frustration_mean(self) -> float:
        return _rolling_mean(self.frustration_history)

    @property
    def frustration_max(self) -> float:
        return _rolling_max(self.frustration_history)

    @property
    def grief_mean(self) -> float:
        return _rolling_mean(self.grief_history)

    @property
    def longing_mean(self) -> float:
        return _rolling_mean(self.longing_history)

    @property
    def boredom_mean(self) -> float:
        return _rolling_mean(self.boredom_history)

    @property
    def chronic_fear(self) -> bool:
        """True when fear has been elevated (>0.3) for most of the last 20 samples."""
        return self.fear_mean > 0.3

    @property
    def chronic_frustration(self) -> bool:
        """True when frustration mean exceeds threshold."""
        return self.frustration_mean > 0.35


# ---------------------------------------------------------------------------
# Motivation & Adaptive Priorities (Phase 8.4)
# ---------------------------------------------------------------------------

@dataclass
class MotivationDimension:
    """
    Represents a single dimension in the agent's MotivationProfile.
    Tracks running lifetime statistics (mean, variance) in O(1) space
    using Welford's Algorithm.
    """
    current:      float = 0.5
    mean:         float = 0.5
    variance:     float = 0.0
    update_count: int = 0

    def update(self, new_val: float) -> None:
        """Updates current, mean, and variance using Welford's Algorithm."""
        self.current = float(np.clip(new_val, 0.0, 1.0))
        self.update_count += 1
        k = self.update_count
        old_mean = self.mean
        self.mean = old_mean + (self.current - old_mean) / k
        if k > 1:
            self.variance = self.variance + (self.current - old_mean) * (self.current - self.mean)
        else:
            self.variance = 0.0

    @property
    def std_dev(self) -> float:
        """Returns running standard deviation."""
        if self.update_count > 0:
            var_val = self.variance / self.update_count
            return float(np.sqrt(max(0.0, var_val)))
        return 0.0


@dataclass
class MotivationProfile:
    """
    Tracks the agent's six dynamic motivational priorities.
    Starting values are expressed from genes at birth and drift based
    on lifetime experience.
    """
    safety:      MotivationDimension = field(default_factory=MotivationDimension)
    family:      MotivationDimension = field(default_factory=MotivationDimension)
    exploration: MotivationDimension = field(default_factory=MotivationDimension)
    knowledge:   MotivationDimension = field(default_factory=MotivationDimension)
    comfort:     MotivationDimension = field(default_factory=MotivationDimension)
    dominance:   MotivationDimension = field(default_factory=MotivationDimension)

    def to_dict(self) -> dict:
        """Converts profile to a flat dictionary for telemetry serialization."""
        res = {}
        for name in ("safety", "family", "exploration", "knowledge", "comfort", "dominance"):
            dim = getattr(self, name)
            res[f"{name}_current"]  = round(dim.current, 4)
            res[f"{name}_mean"]     = round(dim.mean, 4)
            res[f"{name}_variance"] = round(dim.variance / max(1, dim.update_count), 4)
        return res


# ---------------------------------------------------------------------------
# Relationship  (skeleton for Phase 8.3)
# ---------------------------------------------------------------------------

@dataclass
class Relationship:
    """
    Per-dyad social relationship record.

    All dimensions are continuous floats.  None of them is a boolean flag.

    Phase 8.3 will implement update_relationships() called every 30 ticks:
      - trust updated via share/dispute/reproduce events
      - attachment built from co-location frequency
      - dominance updated from repeated win/loss records
      - reciprocity from symmetry of sharing
      - competition from resource conflicts

    Phase 8.6 (Social Hierarchy) will use dominance to produce emergent
    status ranks without ever assigning them explicitly.
    """
    trust:       float = 0.0    # [-1, 1]   negative = distrust
    attachment:  float = 0.0    # [0, 1]    co-location bond
    respect:     float = 0.0    # [0, 1]    behavioural admiration
    fear:        float = 0.0    # [0, 1]    perceived threat from other
    dominance:   float = 0.0    # [-1, 1]   positive = I dominate other
    reciprocity: float = 0.0    # [0, 1]    symmetry of mutual aid
    competition: float = 0.0    # [0, 1]    resource rivalry

    last_seen_tick: int = -1    # world tick of most recent encounter


# ---------------------------------------------------------------------------
# Drive update — Biological Clock  (every tick, Phase 8.1)
# ---------------------------------------------------------------------------

# EMA smoothing factor: 0.85 old + 0.15 new → ~7-tick lag
_BIO_EMA = 0.85


def update_biological_drives(agent, local_temp: float) -> None:
    """
    Biological Clock — runs every tick inside the Physiology Clock section
    of simulate_agent_tick().

    Updates all biological drive tensions via EMA smoothing.
    Also increments streak/counter fields used by emotional dynamics.

    Parameters
    ----------
    agent      : Agent instance
    local_temp : current local temperature (°C) at the agent's location
    """
    ds = agent.drives

    # 1. Smooth biological tensions ----------------------------------------
    ds.hunger_tension     = _BIO_EMA * ds.hunger_tension     + (1 - _BIO_EMA) * (agent.hunger / 100.0)
    ds.thirst_tension     = _BIO_EMA * ds.thirst_tension     + (1 - _BIO_EMA) * (agent.thirst / 100.0)
    ds.exhaustion_tension = _BIO_EMA * ds.exhaustion_tension + (1 - _BIO_EMA) * ((100.0 - agent.energy) / 100.0)
    ds.pain_tension       = _BIO_EMA * ds.pain_tension       + (1 - _BIO_EMA) * (agent.injury_level / 100.0)

    # 2. Thermal stress [0, 1] — comfort band 15–25 °C --------------------
    if 15.0 <= local_temp <= 25.0:
        raw_thermal = 0.0
    elif local_temp < 15.0:
        raw_thermal = min(1.0, (15.0 - local_temp) / 25.0)   # 0 at 15°C → 1 at -10°C
    else:
        raw_thermal = min(1.0, (local_temp - 25.0) / 25.0)   # 0 at 25°C → 1 at 50°C
    ds.thermal_stress = _BIO_EMA * ds.thermal_stress + (1 - _BIO_EMA) * raw_thermal

    # Clamp all tensions to [0.0, 1.0]
    ds.hunger_tension     = float(np.clip(ds.hunger_tension,     0.0, 1.0))
    ds.thirst_tension     = float(np.clip(ds.thirst_tension,     0.0, 1.0))
    ds.exhaustion_tension = float(np.clip(ds.exhaustion_tension, 0.0, 1.0))
    ds.pain_tension       = float(np.clip(ds.pain_tension,       0.0, 1.0))
    ds.thermal_stress     = float(np.clip(ds.thermal_stress,     0.0, 1.0))

    # 3. Same-action streak ------------------------------------------------
    if agent.current_action == getattr(agent, "_last_drive_action", None):
        ds._same_action_streak += 1
    else:
        ds._same_action_streak = 0
    agent._last_drive_action = agent.current_action

    # 4. Unmet need counters (used by Phase 8.2 frustration) ---------------
    if ds.hunger_tension > 0.5:
        ds._unmet_hunger_ticks += 1
    else:
        ds._unmet_hunger_ticks = max(0, ds._unmet_hunger_ticks - 2)

    if ds.thirst_tension > 0.5:
        ds._unmet_thirst_ticks += 1
    else:
        ds._unmet_thirst_ticks = max(0, ds._unmet_thirst_ticks - 2)

    # 5. Fear: builds from pain spikes + thermal extremes, decays at rest --
    pain_delta = max(0.0, ds.pain_tension - ds._prev_pain)
    threat = pain_delta * 0.4 + ds.thermal_stress * 0.2
    ds.fear = float(np.clip(
        ds.fear * 0.92 + threat,    # natural decay × 0.92 per tick
        0.0, 1.0
    ))
    ds._prev_pain = ds.pain_tension

    # 6. Social proximity counter -----------------------------------------
    # Count nearby known agents (populated in Phase 8.2 update; default increment here)
    # _ticks_since_known_agent is decremented in update_emotional_drives
    ds._ticks_since_known_agent += 1


# ---------------------------------------------------------------------------
# Drive update — Emotional Clock  (every 10 ticks, Phase 8.2)
# ---------------------------------------------------------------------------

def update_emotional_drives(agent, world) -> None:
    """
    Emotional Clock — runs every 10 ticks.

    Computes emotional drives as emergent patterns from interacting biological
    tensions, environmental context, and expectation prediction errors.
    """
    ds = agent.drives
    if getattr(world, "ablation", {}).get("emotion", True) == False:
        ds.fear = 0.0
        ds.frustration = 0.0
        ds.boredom = 0.0
        ds.grief = 0.0
        ds.longing = 0.0
        _append_history(ds.fear_history, 0.0)
        _append_history(ds.frustration_history, 0.0)
        _append_history(ds.longing_history, 0.0)
        _append_history(ds.grief_history, 0.0)
        _append_history(ds.boredom_history, 0.0)
        return
    cy, cx = agent.location

    # 1. Prediction Error routing (Phase 8.4)
    pred_err = getattr(agent, "last_prediction_error", 0.0)
    pred_cat = getattr(agent, "last_prediction_category", "Idle")

    pred_frustration = 0.0
    pred_longing = 0.0
    pred_fear = 0.0

    # High negative prediction error means reality was worse than expected
    if pred_err < -0.4:
        err_mag = abs(pred_err)
        if pred_cat in ("Drinking", "Eating", "Withdraw Food", "Withdraw Water", "Drink Stored Water", "Eat Stored Food"):
            pred_frustration = 0.15 * err_mag
        elif pred_cat == "Reproduce":
            pred_longing = 0.20 * err_mag
        elif pred_cat in ("Resting", "Sheltering", "Building Shelter"):
            pred_fear = 0.18 * err_mag
    # High positive prediction error helps decay frustration / fear
    elif pred_err > 0.4:
        pred_frustration = -0.10 * pred_err
        pred_fear = -0.08 * pred_err
        pred_longing = -0.08 * pred_err

    # ------------------------------------------------------------------ #
    # 1. Frustration                                                       #
    # Builds from sustained unmet needs, failure, and negative prediction error.
    # Decays when needs are met or positive prediction error occurs.
    # ------------------------------------------------------------------ #
    unmet_score = (
        min(1.0, ds._unmet_hunger_ticks / 50.0) * 0.45 +
        min(1.0, ds._unmet_thirst_ticks / 50.0) * 0.45 +
        ds.thermal_stress * 0.10
    )
    failed_visits = getattr(agent, "failed_water_visits", 0) + getattr(agent, "failed_food_visits", 0)
    failure_pressure = min(1.0, failed_visits / 30.0) * 0.3
    satisfaction_event = 1.0 if (
        getattr(agent, "_consumed_food", False) or getattr(agent, "_consumed_water", False)
    ) else 0.0

    frustration_delta = (
        0.08 * unmet_score +
        0.05 * failure_pressure +
        pred_frustration
        - 0.12 * satisfaction_event
        - 0.02  # natural decay toward 0
    )
    ds.frustration = float(np.clip(ds.frustration + frustration_delta, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # 2. Loneliness → Longing                                             #
    # Builds from social isolation and mate prediction error.
    # ------------------------------------------------------------------ #
    # Count known agents visible within vision radius
    nearby_known = 0
    alive_map = getattr(world, "alive_agents_map", None)
    vision_radius_sq = agent.vision_radius ** 2
    if alive_map is not None:
        for other_id in agent.known_agents:
            other = alive_map.get(other_id)
            if other is not None and other.id != agent.id:
                oy, ox = other.location
                dist_sq = (oy - cy)**2 + (ox - cx)**2
                if dist_sq <= vision_radius_sq:
                    nearby_known += 1
    else:
        for other in world.agents:
            if not other.dead and other.id != agent.id and other.id in agent.known_agents:
                oy, ox = other.location
                dist_sq = (oy - cy)**2 + (ox - cx)**2
                if dist_sq <= vision_radius_sq:
                    nearby_known += 1


    if nearby_known > 0:
        ds._ticks_since_known_agent = 0
    isolation_pressure = min(1.0, ds._ticks_since_known_agent / 500.0)

    has_known_kin = len(getattr(agent, "known_agents", set())) > 0
    kin_pull = 0.3 if (has_known_kin and nearby_known == 0) else 0.0

    longing_delta = (
        0.04 * isolation_pressure +
        0.02 * kin_pull +
        pred_longing
        - 0.03 * min(1.0, nearby_known * 0.5)
        - 0.01  # natural decay
    )
    ds.longing = float(np.clip(ds.longing + longing_delta, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # 3. Grief                                                             #
    # ------------------------------------------------------------------ #
    ds.grief = float(np.clip(ds.grief * 0.75, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # 4. Boredom                                                           #
    # ------------------------------------------------------------------ #
    streak_pressure = min(1.0, ds._same_action_streak / 100.0)
    if ds._same_action_streak > 10:
        ds.boredom = float(np.clip(ds.boredom + 0.05 * streak_pressure, 0.0, 1.0))
    else:
        ds.boredom = float(np.clip(ds.boredom * 0.80, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # 5. Fear                                                              #
    # ------------------------------------------------------------------ #
    is_disaster = (
        getattr(world, "global_temp_offset", 0.0) != 0.0 or
        getattr(world, "global_water_multiplier", 1.0) < 0.8 or
        getattr(world, "global_food_multiplier", 1.0) < 0.8
    )
    disaster_boost = 0.08 if is_disaster else 0.0
    isolation_fear_boost = 0.04 * isolation_pressure if ds.fear > 0.25 else 0.0

    rest_bonus = 0.06 if (agent.current_action == "Resting" and nearby_known > 0) else 0.0
    fear_delta = disaster_boost + isolation_fear_boost + pred_fear - rest_bonus
    ds.fear = float(np.clip(ds.fear + fear_delta, 0.0, 1.0))

    # 6. Record histories
    _append_history(ds.fear_history,        ds.fear)
    _append_history(ds.frustration_history, ds.frustration)
    _append_history(ds.longing_history,     ds.longing)
    _append_history(ds.grief_history,       ds.grief)
    _append_history(ds.boredom_history,     ds.boredom)


def trigger_grief(agent, deceased_agent_id: int, world) -> None:
    """
    External event: called from simulation.py when a known agent dies nearby.

    Spikes the agent's grief drive based on how well they knew the deceased
    and how close they were at the time of death.

    The spike magnitude reflects:
      - Social relationship strength (social modifier for that agent)
      - Proximity at time of death
      - Kinship (parent/child relationship)
    """
    if getattr(world, "ablation", {}).get("emotion", True) == False:
        return
    ds = agent.drives
    cy, cx = agent.location

    # Relationship strength via social modifier
    rel_strength = agent.get_social_modifier(deceased_agent_id, world.tick)
    # rel_strength ∈ [0, 2]; normalize to [0, 1]
    rel_norm = float(np.clip((rel_strength - 1.0) / 1.0, 0.0, 1.0))

    # Kinship bonus
    parents = getattr(agent, "parent_ids", None) or []
    children = getattr(agent, "children_ids", None) or []
    is_kin = (
        deceased_agent_id in parents or
        deceased_agent_id in children
    )
    kin_bonus = 0.35 if is_kin else 0.0

    # Compute grief spike: max 0.8 for very close kin, floor 0.1 for acquaintances
    grief_spike = float(np.clip(0.2 + rel_norm * 0.4 + kin_bonus, 0.1, 0.8))
    ds.grief = float(np.clip(ds.grief + grief_spike, 0.0, 1.0))

    # Grief also slightly amplifies longing (missing the person)
    ds.longing = float(np.clip(ds.longing + grief_spike * 0.3, 0.0, 1.0))


def update_relationships(agent, world) -> None:
    """
    Relationship Clock — runs every 30 ticks.

    Reviews episodic memories from the last 30 ticks to update continuous
    dyadic relationship dimensions with other agents.

    Dimensions updated:
      - trust      : positive on share/mate, negative on disputes
      - attachment : builds on co-location, sharing, mating
      - respect    : builds on sharing, drops on disputes
      - fear       : builds on disputes, drops on sharing/mating
      - dominance  : updates during disputes based on muscle mass & aggression differences
      - reciprocity: builds on sharing, drops on disputes
      - competition: builds on disputes, drops on sharing, colony difference baseline
    """
    if getattr(world, "ablation", {}).get("relationships", True) == False:
        return
    # Build a fast map of other active agents to check their attributes
    agent_map = getattr(world, "alive_agents_map", None)
    if agent_map is None:
        agent_map = {a.id: a for a in world.agents if not a.dead}

    # Filter episodic memories from the last 30 ticks that have associated_id
    recent_mems = [
        m for m in agent.episodic_memory
        if (world.tick - 30) < m.timestamp <= world.tick
        and getattr(m, "associated_id", -1) >= 0
    ]

    # Group recent memories by other agent ID
    mems_by_other = {}
    for m in recent_mems:
        other_id = m.associated_id
        mems_by_other.setdefault(other_id, []).append(m)

    # We update relationships for any agent we have previously interacted with
    # plus any new encounters in the last 30 ticks
    all_targets = set(agent.relationships.keys()) | set(mems_by_other.keys())

    for other_id in all_targets:
        # If the other agent is dead, we don't update their relationship further
        # (their record is frozen/archived)
        if other_id not in agent_map and other_id in agent.relationships:
            continue

        if other_id not in agent.relationships:
            agent.relationships[other_id] = Relationship()
        rel = agent.relationships[other_id]

        other_agent = agent_map.get(other_id)
        matching_mems = mems_by_other.get(other_id, [])

        if matching_mems:
            # 1. Update last seen tick
            newest_tick = max(m.timestamp for m in matching_mems)
            rel.last_seen_tick = max(rel.last_seen_tick, newest_tick)

            # Count interaction types in the last 30 ticks
            disputes = sum(1 for m in matching_mems if getattr(m, "outcome", "neutral") == "dispute" or m.type == "DANGER")
            shares   = sum(1 for m in matching_mems if getattr(m, "outcome", "neutral") == "share")
            mates    = sum(1 for m in matching_mems if getattr(m, "outcome", "neutral") == "mate")
            persons  = sum(1 for m in matching_mems if m.type == "PERSON")

            # 2. Trust update
            trust_delta = 0.10 * shares + 0.15 * mates - 0.20 * disputes
            if trust_delta > 0:
                rel.trust = rel.trust + trust_delta * (1.0 - rel.trust)
            else:
                rel.trust = rel.trust + trust_delta * (1.0 + rel.trust)

            # 3. Attachment update
            attach_rate = 0.01 * persons + 0.08 * shares + 0.18 * mates
            rel.attachment = rel.attachment + attach_rate * (1.0 - rel.attachment)

            # 4. Respect update
            respect_delta = 0.08 * shares - 0.06 * disputes
            if respect_delta > 0:
                rel.respect = rel.respect + respect_delta * (1.0 - rel.respect)
            else:
                rel.respect = rel.respect + respect_delta * rel.respect

            # 5. Fear update
            fear_delta = 0.15 * disputes - 0.10 * (shares + mates)
            if fear_delta > 0:
                rel.fear = rel.fear + fear_delta * (1.0 - rel.fear)
            else:
                rel.fear = rel.fear + fear_delta * rel.fear

            # 6. Dominance update (only if dispute occurred and other agent is alive to compare)
            if disputes > 0 and other_agent is not None:
                mass_diff = agent.muscle_mass - other_agent.muscle_mass
                my_agg = agent.genome.genes[1] if hasattr(agent, "genome") else 0.5
                other_agg = other_agent.genome.genes[1] if hasattr(other_agent, "genome") else 0.5
                agg_diff = my_agg - other_agg
                dom_shift = 0.10 * disputes * (mass_diff / 100.0 + agg_diff * 0.5)
                if dom_shift > 0:
                    rel.dominance = rel.dominance + dom_shift * (1.0 - rel.dominance)
                else:
                    rel.dominance = rel.dominance + dom_shift * (1.0 + rel.dominance)

            # 7. Reciprocity update
            recip_delta = 0.08 * shares - 0.10 * disputes
            if recip_delta > 0:
                rel.reciprocity = rel.reciprocity + recip_delta * (1.0 - rel.reciprocity)
            else:
                rel.reciprocity = rel.reciprocity + recip_delta * rel.reciprocity

            # 8. Competition update
            comp_delta = 0.15 * disputes - 0.08 * (shares + mates)
            if comp_delta > 0:
                rel.competition = rel.competition + comp_delta * (1.0 - rel.competition)
            else:
                rel.competition = rel.competition + comp_delta * rel.competition
        else:
            # Decay relationships slowly if no interactions occurred in the last 30 ticks
            rel.trust       *= 0.98
            rel.attachment  *= 0.96
            rel.respect     *= 0.98
            rel.fear        *= 0.95
            rel.dominance   *= 0.98
            rel.reciprocity *= 0.96
            rel.competition *= 0.98

        # Enforce cross-colony competition baseline floor
        if other_agent is not None and getattr(agent, "colony_id", -1) != getattr(other_agent, "colony_id", -1):
            rel.competition = max(0.25, rel.competition)

        # Hard clamp all dimensions to their theoretical bounds
        rel.trust       = float(np.clip(rel.trust,       -1.0,  1.0))
        rel.attachment  = float(np.clip(rel.attachment,   0.0,  1.0))
        rel.respect     = float(np.clip(rel.respect,      0.0,  1.0))
        rel.fear        = float(np.clip(rel.fear,         0.0,  1.0))
        rel.dominance   = float(np.clip(rel.dominance,   -1.0,  1.0))
        rel.reciprocity = float(np.clip(rel.reciprocity,   0.0,  1.0))
        rel.competition = float(np.clip(rel.competition,   0.0,  1.0))


def update_adaptive_motivations(agent, world) -> None:
    """
    Personality/Motivation Clock — runs every 100 ticks.

    Drifts the agent's MotivationProfile priorities based on lifetime
    experiences, and updates the Welford running statistics (mean/variance).
    """
    if getattr(world, "ablation", {}).get("motivation", True) == False:
        return
    if not hasattr(agent, "motivation") or agent.motivation is None:
        return

    m = agent.motivation
    ds = agent.drives

    # 1. Safety drift
    safety_trigger = (agent.health < 40.0 or ds.fear > 0.3 or agent.injury_level > 20.0 or ds.thermal_stress > 0.5)
    safety_drift = 0.05 if safety_trigger else -0.02
    m.safety.update(m.safety.current + safety_drift)

    # 2. Family drift
    has_recent_mate = any(getattr(mem, "outcome", "neutral") == "mate" for mem in agent.episodic_memory if world.tick - 100 < mem.timestamp <= world.tick)
    has_high_attachment = any(r.attachment > 0.5 for r in agent.relationships.values())
    family_trigger = (has_recent_mate or has_high_attachment or ds.longing > 0.3)
    family_drift = 0.05 if family_trigger else -0.02
    m.family.update(m.family.current + family_drift)

    # 3. Exploration drift
    has_recent_landmark = any(mem.type == "LANDMARK" for mem in agent.episodic_memory if world.tick - 100 < mem.timestamp <= world.tick)
    exploration_trigger = (ds.boredom > 0.4 or has_recent_landmark or agent.max_radius > 120.0)
    exploration_drift = 0.04 if exploration_trigger else -0.02
    m.exploration.update(m.exploration.current + exploration_drift)

    # 4. Knowledge drift
    has_procedures = len(agent.procedures) > 0
    has_concepts = sum(len(lst) for lst in agent.concepts.values()) > 0
    knowledge_trigger = (has_procedures or has_concepts or agent.prediction_confidence > 0.6)
    knowledge_drift = 0.05 if knowledge_trigger else -0.02
    m.knowledge.update(m.knowledge.current + knowledge_drift)

    # 5. Comfort drift
    comfort_trigger = (agent.shelter_level >= 2 or ds.thermal_stress > 0.3 or agent.current_action == "Sheltering")
    comfort_drift = 0.04 if comfort_trigger else -0.02
    m.comfort.update(m.comfort.current + comfort_drift)

    # 6. Dominance drift
    has_recent_dispute = any(getattr(mem, "outcome", "neutral") == "dispute" for mem in agent.episodic_memory if world.tick - 100 < mem.timestamp <= world.tick)
    dominance_trigger = (has_recent_dispute or agent.muscle_mass > 105.0 or ds.frustration > 0.4)
    dominance_drift = 0.05 if dominance_trigger else -0.02
    m.dominance.update(m.dominance.current + dominance_drift)


# ---------------------------------------------------------------------------
# Phase 8.15: Memory Importance Scoring
# ---------------------------------------------------------------------------

def compute_memory_importance(
    agent,
    mem_type: str,
    outcome: str = "neutral",
    near_death: bool = False,
    is_first_encounter: bool = False,
    emotional_intensity: float = 0.0,
) -> float:
    """
    Computes the importance score for a memory event before it is stored.

    Formula
    -------
    importance = novelty × survival_impact × social_impact × (1 + emotional_intensity)

    Ranges
    ------
    All factors are in [0, 1].  Final importance is clamped to [0.05, 1.0].

    Examples
    --------
    Found water when thirsty          → 0.65–0.80
    Child died (grief spike)          → 0.90–0.98
    Nearly died of dehydration        → 0.85–0.95
    Walked 5 cells                    → 0.01–0.05
    Shared food with known companion  → 0.50–0.70
    Stranger sighted                  → 0.30–0.50

    Parameters
    ----------
    agent               : Agent instance
    mem_type            : WATER, FOOD, DANGER, PERSON, LANDMARK
    outcome             : "neutral", "dispute", "share", "mate", "death"
    near_death          : True if the agent's health was critically low
    is_first_encounter  : True if this location/agent has not been seen before
    emotional_intensity : current arousal [0, 1] from drives
    """
    ds = agent.drives

    # ------------------------------------------------------------------ #
    # Novelty: first encounters are more memorable                        #
    # ------------------------------------------------------------------ #
    novelty = 0.9 if is_first_encounter else 0.3

    # ------------------------------------------------------------------ #
    # Survival impact: how dangerous / urgent was this event?             #
    # ------------------------------------------------------------------ #
    if near_death:
        survival_impact = 1.0
    elif mem_type == "DANGER":
        survival_impact = 0.90
    elif mem_type == "WATER":
        survival_impact = 0.4 + ds.thirst_tension * 0.6    # more urgent when thirsty
    elif mem_type == "FOOD":
        survival_impact = 0.4 + ds.hunger_tension * 0.6    # more urgent when hungry
    elif mem_type == "PERSON":
        survival_impact = 0.2
    else:
        survival_impact = 0.1   # LANDMARK

    # ------------------------------------------------------------------ #
    # Social impact: social events carry emotional weight                 #
    # ------------------------------------------------------------------ #
    if outcome == "death":
        social_impact = 1.0
    elif outcome == "mate":
        social_impact = 0.80
    elif outcome == "share":
        social_impact = 0.55
    elif outcome == "dispute":
        social_impact = 0.70
    elif mem_type == "PERSON":
        social_impact = 0.35
    else:
        social_impact = 0.10

    # ------------------------------------------------------------------ #
    # Emotional intensity amplifier                                        #
    # ------------------------------------------------------------------ #
    emo_amp = 1.0 + float(np.clip(emotional_intensity, 0.0, 1.0)) * 0.5

    # ------------------------------------------------------------------ #
    # Combine: geometric blend to prevent any one factor dominating       #
    # ------------------------------------------------------------------ #
    raw = (novelty * 0.30 + survival_impact * 0.40 + social_impact * 0.30) * emo_amp
    return float(np.clip(raw, 0.05, 1.0))


# ---------------------------------------------------------------------------
# Drive modulation  — per-action utility multipliers  (Phase 8.1 + 8.2)
# ---------------------------------------------------------------------------

def compute_drive_modulation(agent) -> dict:
    """
    Returns a dict mapping action names to drive-derived utility multipliers.

    Phase 8.1: biological tension multipliers.
    Phase 8.2: emotional drives (fear, frustration, boredom, grief, longing).
    Phase 8.4: Conflict Resolver (Lateral Inhibition) & Motivation Profile.
    """
    ds = agent.drives

    ht  = ds.hunger_tension
    tt  = ds.thirst_tension
    ext = ds.exhaustion_tension
    pt  = ds.pain_tension
    ts  = ds.thermal_stress

    ablation = getattr(agent, "ablation", {})
    
    fear        = 0.0 if ablation.get("emotion", True) == False else ds.fear
    frustration = 0.0 if ablation.get("emotion", True) == False else ds.frustration
    boredom     = 0.0 if ablation.get("emotion", True) == False else ds.boredom
    grief       = 0.0 if ablation.get("emotion", True) == False else ds.grief
    longing     = 0.0 if ablation.get("emotion", True) == False else ds.longing

    # Chronic fear (sustained) suppresses exploration more than a brief spike
    effective_fear_suppress = 0.0 if ablation.get("emotion", True) == False else max(fear, ds.fear_mean * 0.6)

    # ------------------------------------------------------------------
    # Phase 8.4: Conflict Resolver (Lateral Inhibition)
    # ------------------------------------------------------------------
    if hasattr(agent, "motivation") and agent.motivation is not None and ablation.get("motivation", True):
        m = agent.motivation
        # Calculate raw active motivations (drive tension * priority weight)
        m_safety      = m.safety.current      * max(fear, pt, ts)
        m_family      = m.family.current      * max(longing, grief)
        m_exploration = m.exploration.current * boredom
        m_knowledge   = m.knowledge.current   * max(0.0, 1.0 - getattr(agent, "prediction_confidence", 1.0))
        m_comfort     = m.comfort.current     * max(ts, ext)
        m_dominance   = m.dominance.current   * frustration

        # Compile them into a list for lateral inhibition
        m_list = [m_safety, m_family, m_exploration, m_knowledge, m_comfort, m_dominance]
        m_sum = sum(m_list)

        # Apply Lateral Inhibition (beta = 0.15)
        # Inhibited strength: M_i = max(0.0, m_i - beta * sum(m_j for j != i))
        beta = 0.15
        M_safety      = max(0.0, m_safety      - beta * (m_sum - m_safety))
        M_family      = max(0.0, m_family      - beta * (m_sum - m_family))
        M_exploration = max(0.0, m_exploration - beta * (m_sum - m_exploration))
        M_knowledge   = max(0.0, m_knowledge   - beta * (m_sum - m_knowledge))
        M_comfort     = max(0.0, m_comfort     - beta * (m_sum - m_comfort))
        M_dominance   = max(0.0, m_dominance   - beta * (m_sum - m_dominance))
    else:
        # Backward compatibility / fallback: no inhibition, neutral weights (0.5)
        M_safety      = 0.5
        M_family      = 0.5
        M_exploration = 0.5
        M_knowledge   = 0.5
        M_comfort     = 0.5
        M_dominance   = 0.5

    # ------------------------------------------------------------------
    # Biological & Emotional drive multipliers (Phase 8.1 + 8.2)
    # ------------------------------------------------------------------

    # Drinking: thirst urgency is superlinear; frustration from unmet thirst
    # boosts it further.
    drink_mult = 1.0 + 2.0 * tt * tt + frustration * 0.4

    # Eating: hunger urgency; frustration from unmet hunger boosts it.
    eat_mult = 1.0 + 1.8 * ht * ht + frustration * 0.3

    # Resting / Sheltering:
    #   - exhaustion raises the drive to rest.
    #   - pain and grief both amplify rest drive.
    #   - fear suppresses rest.
    grief_rest_override = max(0.0, grief - fear * 0.5)
    rest_mult = (
        (1.0 + 1.5 * ext + 0.8 * pt + 1.0 * grief_rest_override)
        * max(0.3, 1.0 - fear * 0.6)
    )
    shelter_mult = (1.0 + 1.2 * ts + 0.6 * pt) * (1.0 + fear * 0.5)

    # Exploring: boredom boosts when fear is low; chronic fear suppresses hard.
    boredom_explore = boredom * 0.9 * max(0.0, 1.0 - fear * 1.2)
    explore_mult = (
        1.0 + boredom_explore + frustration * 0.2
    ) * (1.0 - effective_fear_suppress * 0.8)

    # Reproduce: fear and grief both strongly suppress reproduction.
    reproduce_mult = 1.0 - fear * 0.95 - grief * 0.6 + longing * 0.2

    # Shelter building: scared agents build more urgently.
    build_mult = 1.0 + fear * 0.6 + ts * 0.4

    # Sharing: longing with known agent amplifies generosity; fear suppresses it.
    longing_share_boost = longing * 0.3
    share_mult = 1.0 - (pt + fear) * 0.3 + longing_share_boost

    # Stored food/water consumption: direct biological pressure.
    drink_stored_mult = 1.0 + 1.5 * tt + frustration * 0.2
    eat_stored_mult   = 1.0 + 1.2 * ht + frustration * 0.15

    # Caching: suppressed by urgency AND frustration (urgent agents don't plan ahead).
    cache_mult = 1.0 - (ht + tt) * 0.35 - frustration * 0.2

    # ------------------------------------------------------------------
    # Phase 8.4: Apply resolved Motivation weights to final multipliers
    # ------------------------------------------------------------------
    if ablation.get("motivation", True):
        rest_mult      = rest_mult      * (0.5 + M_comfort + M_safety * 0.5)
        shelter_mult   = shelter_mult   * (0.5 + M_comfort + M_safety * 0.5)
        explore_mult   = explore_mult   * (0.5 + M_exploration + M_knowledge * 0.5) * max(0.1, 1.5 - M_safety)
        reproduce_mult = reproduce_mult * (0.5 + M_family) * max(0.1, 1.5 - M_safety - M_dominance * 0.5)
        build_mult     = build_mult     * (0.5 + M_safety + M_comfort * 0.5)
        share_mult     = share_mult     * (0.5 + M_family) * max(0.1, 1.5 - M_dominance - M_safety * 0.5)
        cache_mult     = cache_mult     * (0.5 + M_safety * 0.5 + M_comfort * 0.5) * max(0.1, 1.5 - M_dominance * 0.5)

    # Pure-Python clamp — avoids NumPy dispatch overhead on scalar values
    # (np.clip on a Python float is ~10x slower than max/min)
    _clamp = lambda v: max(0.10, min(3.0, float(v)))

    return {
        "Drinking":           _clamp(drink_mult),
        "Eating":             _clamp(eat_mult),
        "Resting":            _clamp(rest_mult),
        "Sheltering":         _clamp(shelter_mult),
        "Exploring":          _clamp(explore_mult),
        "Reproduce":          _clamp(reproduce_mult),
        "Building Shelter":   _clamp(build_mult),
        "Share Food":         _clamp(share_mult),
        "Share Water":        _clamp(share_mult),
        "Store Food":         _clamp(cache_mult),
        "Store Water":        _clamp(cache_mult),
        "Deposit Food":       _clamp(cache_mult),
        "Deposit Water":      _clamp(cache_mult),
        "Withdraw Food":      _clamp(drink_stored_mult),
        "Withdraw Water":     _clamp(drink_stored_mult),
        "Drink Stored Water": _clamp(drink_stored_mult),
        "Eat Stored Food":    _clamp(eat_stored_mult),
    }


# ---------------------------------------------------------------------------
# Emotional label  — human-readable telemetry string
# ---------------------------------------------------------------------------

def emotional_label(agent) -> str:
    """
    Returns a concise, human-readable emotional state label for telemetry
    and the Research Dashboard timeline.

    Priority order: most urgent / acute drive wins the label.
    Phase 8.2: emotional drives (fear, grief, frustration, boredom, longing)
               are now properly populated and included.
    Chronic states use history-based properties.
    """
    ds = agent.drives

    # Acute events take priority
    if ds.grief > 0.5:
        return "Grieving"
    if ds.fear > 0.65:
        return "Terrified"
    if ds.fear > 0.35:
        return "Anxious"

    # Chronic states (based on rolling history)
    if ds.chronic_fear:
        return "Chronically Anxious"
    if ds.chronic_frustration:
        return "Chronically Frustrated"

    # Emotional drives
    if ds.frustration > 0.55:
        return "Frustrated"
    if ds.boredom > 0.6:
        return "Bored"
    if ds.longing > 0.5:
        return "Longing"
    if ds.grief > 0.2:
        return "Mourning"

    # Biological tensions
    if ds.thirst_tension > 0.75:
        return "Critically Thirsty"
    if ds.hunger_tension > 0.75:
        return "Critically Hungry"
    if ds.exhaustion_tension > 0.75:
        return "Exhausted"
    if ds.pain_tension > 0.6:
        return "In Pain"
    if ds.thermal_stress > 0.65:
        return "Thermal Stress"
    if ds.thirst_tension > 0.45:
        return "Thirsty"
    if ds.hunger_tension > 0.45:
        return "Hungry"
    if ds.exhaustion_tension > 0.45:
        return "Fatigued"

    # Contentment
    if ds.contentment > 0.7:
        return "Content"

    return "Neutral"
