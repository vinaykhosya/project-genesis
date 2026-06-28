"""
checkpoint_io.py — Phase 9A: Full-State Simulation Checkpoint

PHILOSOPHY
----------
State is divided into two categories:

  Persistent state  — anything that affects future simulation behaviour.
                      This is the ONLY thing we save.
  Derived state     — anything that can be recomputed (statistics, caches,
                      visual overlays, numpy world arrays).
                      This is NEVER saved.

World numpy arrays (elevation, biome, climate, rivers, resources) are 100%
deterministic from world.seed and are NOT saved.  They are regenerated from the
seed in 2-3 seconds on resume.  This keeps checkpoints small (~10-20 MB),
human-readable, and backward-compatible.

CHECKPOINT FORMAT
-----------------
{
  "checkpoint_version": 1,        ← bump when format changes
  "genesis_phase":      "9A",
  "seed":  int,
  "tick":  int,
  ...world counters...
  "config": {...},               ← full experiment config at save time
  "colonies": [...],
  "shelters": {...},
  "population_history": [...],
  "genetic_history":    [...],
  "extinction_events":  [...],
  "events_timeline":    [...],   ← last MAX_EVENTS_ENTRIES entries
  "history":            [...],   ← last MAX_HISTORY_ENTRIES entries
  "death_density":      [...],   ← sparse list of [cx, cy, val]
  "agents":             [...]    ← full per-agent persistent state
}

Each agent entry serializes:
  - All physiological scalars
  - Genome (gene list)
  - Predictor neural-net weights  (w1 b1 w2 b2)
  - Feature weights (Hebbian)
  - Drive tensions + histories    (Phase 8.1 / 8.2)
  - Motivation profile + stats    (Phase 8.4 / 8.5)
  - Relationship graph            (Phase 8.3)
  - Episodic memory  (last 50)
  - Spatial knowledge
  - Concepts + Procedures
  - Action counts + telemetry counters
  - Social graph (known_agents)

Transient caches cleared on resume (rebuilt on first tick):
  training_buffer, procedural_buffer, last_prediction_input,
  plan_cache, utility_cache, memory_cache, action_queue,
  current_goal, current_target, _wants_to_reproduce_with
"""

from __future__ import annotations
import json
import random
import numpy as np
from typing import Tuple, Dict, Any

CHECKPOINT_VERSION = 1
GENESIS_PHASE = "9A"

# Bounds for pruning long lists (keeps file size manageable)
MAX_HISTORY_ENTRIES  = 2000
MAX_EVENTS_ENTRIES   = 5000
MAX_EPISODIC_MEMORY  = 50    # per agent
MAX_PATH_HISTORY     = 100   # sampled path points saved per agent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _loc_key(loc: tuple) -> str:
    """Encode (y, x) tuple as "y,x" string for JSON dict keys."""
    return f"{int(loc[0])},{int(loc[1])}"


def _parse_loc_key(s: str) -> tuple:
    """Decode "y,x" string back to (y, x) int tuple."""
    a, b = s.split(",")
    return (int(a), int(b))


# ---------------------------------------------------------------------------
# Serialise a single Agent → dict
# ---------------------------------------------------------------------------

def _ser_agent(agent) -> dict:
    """Return a JSON-serialisable dict of all persistent agent state."""
    ds = agent.drives
    m  = agent.motivation

    # ---- drives ----
    drives_d = {
        "hunger_tension":     float(ds.hunger_tension),
        "thirst_tension":     float(ds.thirst_tension),
        "exhaustion_tension": float(ds.exhaustion_tension),
        "pain_tension":       float(ds.pain_tension),
        "thermal_stress":     float(ds.thermal_stress),
        "fear":               float(ds.fear),
        "frustration":        float(ds.frustration),
        "longing":            float(ds.longing),
        "grief":              float(ds.grief),
        "boredom":            float(ds.boredom),
        "fear_history":        [float(v) for v in ds.fear_history],
        "frustration_history": [float(v) for v in ds.frustration_history],
        "longing_history":     [float(v) for v in ds.longing_history],
        "grief_history":       [float(v) for v in ds.grief_history],
        "boredom_history":     [float(v) for v in ds.boredom_history],
        "_same_action_streak":     int(ds._same_action_streak),
        "_unmet_hunger_ticks":     int(ds._unmet_hunger_ticks),
        "_unmet_thirst_ticks":     int(ds._unmet_thirst_ticks),
        "_ticks_since_known_agent": int(ds._ticks_since_known_agent),
        "_prev_pain":              float(ds._prev_pain),
    }

    # ---- motivation ----
    def _dim(name):
        d = getattr(m, name)
        return {"current": float(d.current), "mean": float(d.mean),
                "variance": float(d.variance), "update_count": int(d.update_count)}

    motiv_d = {nm: _dim(nm) for nm in
               ("safety", "family", "exploration", "knowledge", "comfort", "dominance")}

    # ---- relationships ----
    rels_d = {}
    for oid, rel in (agent.relationships or {}).items():
        rels_d[str(oid)] = {
            "trust":        float(rel.trust),
            "attachment":   float(rel.attachment),
            "respect":      float(rel.respect),
            "fear":         float(rel.fear),
            "dominance":    float(rel.dominance),
            "reciprocity":  float(rel.reciprocity),
            "competition":  float(rel.competition),
            "last_seen_tick": int(rel.last_seen_tick),
        }

    # ---- spatial knowledge ----
    water_k  = {_loc_key(k): v for k, v in agent.knowledge.water_sources.items()}
    food_k   = {_loc_key(k): v for k, v in agent.knowledge.food_sources.items()}
    danger_k = {_loc_key(k): float(v) for k, v in agent.knowledge.danger_locations.items()}

    # ---- episodic memory (pruned) ----
    ep_mem = [
        {
            "type":         mem.type,
            "location":     list(mem.location),
            "timestamp":    int(mem.timestamp),
            "importance":   float(mem.importance),
            "confidence":   float(mem.confidence),
            "associated_id": int(mem.associated_id),
            "outcome":      mem.outcome,
        }
        for mem in agent.episodic_memory[-MAX_EPISODIC_MEMORY:]
    ]

    # ---- concepts ----
    concepts_d = {}
    for rtype, clist in agent.concepts.items():
        concepts_d[rtype] = [
            {
                "resource_type": c.resource_type,
                "elevation_mean": c.elevation_mean,
                "elevation_std":  c.elevation_std,
                "temp_mean":      c.temp_mean,
                "temp_std":       c.temp_std,
                "rain_mean":      c.rain_mean,
                "rain_std":       c.rain_std,
                "support":        int(c.support),
                "confidence":     float(c.confidence),
            }
            for c in clist
        ]

    # ---- procedures ----
    procs_d = [
        {
            "trigger_season":  int(p.trigger_season),
            "action_sequence": list(p.action_sequence),
            "success_count":   int(p.success_count),
            "confidence":      float(p.confidence),
        }
        for p in agent.procedures
    ]

    # ---- predictor weights ----
    pred_d = {
        "w1": agent.predictor.w1.tolist(),
        "b1": agent.predictor.b1.tolist(),
        "w2": agent.predictor.w2.tolist(),
        "b2": agent.predictor.b2.tolist(),
    }

    return {
        # --- identity ---
        "id":               int(agent.id),
        "colony_id":        int(getattr(agent, "colony_id", 0)),
        "generation":       int(getattr(agent, "generation", 0)),
        "born_tick":        int(getattr(agent, "born_tick", 0)),
        "parent_ids":       list(agent.parent_ids) if agent.parent_ids else None,
        "children_ids":     [int(i) for i in agent.children_ids],
        "spawn_biome":      str(getattr(agent, "spawn_biome", "Unknown")),
        "archetype":        str(getattr(agent, "archetype", "Balanced")),
        "behavior_cluster": str(getattr(agent, "behavior_cluster", "C0")),
        # --- location ---
        "location":       list(agent.location),
        "home_location":  list(agent.home_location),
        "spawn_location": list(agent.spawn_location),
        # --- physiology ---
        "health":         float(agent.health),
        "dead":           bool(agent.dead),
        "cause_of_death": agent.cause_of_death,
        "primary_cause":  agent.primary_cause,
        "secondary_cause":agent.secondary_cause,
        "age":            int(agent.age),
        "max_age":        int(agent.max_age),
        "hunger":         float(agent.hunger),
        "thirst":         float(agent.thirst),
        "energy":         float(agent.energy),
        "curiosity_need": float(agent.curiosity_need),
        "fat_reserves":   float(agent.fat_reserves),
        "muscle_mass":    float(agent.muscle_mass),
        "injury_level":   float(agent.injury_level),
        # --- damage accumulators ---
        "starvation_damage_accumulated":  float(agent.starvation_damage_accumulated),
        "dehydration_damage_accumulated": float(agent.dehydration_damage_accumulated),
        "exposure_damage_accumulated":    float(agent.exposure_damage_accumulated),
        "injury_damage_accumulated":      float(agent.injury_damage_accumulated),
        "age_damage_accumulated":         float(agent.age_damage_accumulated),
        # --- shelter ---
        "shelter_location":   list(agent.shelter_location) if agent.shelter_location else None,
        "shelter_level":      int(agent.shelter_level),
        "shelter_durability": float(agent.shelter_durability),
        # --- colony resources ---
        "stored_food":  float(agent.stored_food),
        "stored_water": float(agent.stored_water),
        # --- reproduction ---
        "reproduction_cooldown": int(agent.reproduction_cooldown),
        # --- genome ---
        "genome_genes": [float(g) for g in agent.genome.genes],
        # --- traits ---
        "traits": {k: float(v) for k, v in agent.traits.items()},
        # --- neural predictor ---
        "predictor": pred_d,
        # --- feature weights ---
        "feature_weights": [float(v) for v in agent.feature_weights],
        # --- cognition ---
        "learning_rate":         float(agent.learning_rate),
        "prediction_confidence": float(agent.prediction_confidence),
        "plan_commitment":       float(agent.plan_commitment),
        "plan_age":              int(agent.plan_age),
        "planning_accuracy":     float(agent.planning_accuracy),
        "prediction_accuracy":   float(agent.prediction_accuracy),
        "memory_accuracy":       float(agent.memory_accuracy),
        # --- Phase 7 / 9 planning state ---
        "current_goal":          agent.current_goal,
        "current_target":        list(agent.current_target) if agent.current_target else None,
        "action_queue":          [[act, list(tgt) if tgt else None] for act, tgt in agent.action_queue],
        "wants_to_reproduce_with": list(agent._wants_to_reproduce_with) if agent._wants_to_reproduce_with else None,
        # --- Phase 8 ---
        "drives":        drives_d,
        "motivation":    motiv_d,
        "relationships": rels_d,
        # --- knowledge ---
        "knowledge_water":  water_k,
        "knowledge_food":   food_k,
        "knowledge_danger": danger_k,
        # --- memory ---
        "episodic_memory": ep_mem,
        "concepts":        concepts_d,
        "procedures":      procs_d,
        # --- action ---
        "action_counts":  dict(agent.action_counts),
        "action_history": list(agent.action_history),
        "current_action": str(getattr(agent, "current_action", "Idle")),
        # --- telemetry counters ---
        "drinks_count":         int(agent.drinks_count),
        "eats_count":           int(agent.eats_count),
        "resting_ticks":        int(agent.resting_ticks),
        "discoveries_count":    int(agent.discoveries_count),
        "failed_water_visits":  int(agent.failed_water_visits),
        "failed_food_visits":   int(agent.failed_food_visits),
        "nodes_added_count":    int(agent.nodes_added_count),
        "nodes_removed_count":  int(agent.nodes_removed_count),
        "ticks_survived":       int(agent.ticks_survived),
        "max_radius":           float(agent.max_radius),
        "prediction_attempts":  int(agent.prediction_attempts),
        "prediction_successes": int(agent.prediction_successes),
        "prediction_decisions": int(agent.prediction_decisions),
        "prediction_gains":     int(agent.prediction_gains),
        "rediscoveries":        int(agent.rediscoveries),
        "decision_evals":       int(getattr(agent, "decision_evals", 0)),
        "last_prediction_error":float(getattr(agent, "last_prediction_error", 0.0)),
        "training_buffer":       [[inp.tolist(), float(rew)] for inp, rew in getattr(agent, "training_buffer", []) if inp is not None],
        "procedural_buffer":     [[int(s), list(seq), float(rew)] for s, seq, rew in getattr(agent, "procedural_buffer", [])],
        "last_prediction_input": agent.last_prediction_input.tolist() if getattr(agent, "last_prediction_input", None) is not None else None,
        # --- social ---
        "known_agents":         sorted(list(agent.known_agents)),
        # --- season ---
        "season_observations":  {str(k): int(v) for k, v in agent.season_observations.items()},
        # --- exploration tracking ---
        "visited_chunks":       [list(c) for c in sorted(agent.visited_chunks)],
        "discovered_landmarks": sorted(list(getattr(agent, "discovered_landmarks", set()))),
        "discovered_water":     [list(loc) for loc in getattr(agent, "discovered_water", set())],
        "discovered_food":      [list(loc) for loc in getattr(agent, "discovered_food", set())],
        # --- path history (compact) ---
        "sampled_path_history": agent.sampled_path_history[-MAX_PATH_HISTORY:],
        # --- telemetry continuity ---
        "last_health":        float(getattr(agent, "last_health", agent.health)),
        "last_hunger":        float(getattr(agent, "last_hunger", agent.hunger)),
        "last_thirst":        float(getattr(agent, "last_thirst", agent.thirst)),
        "last_injury_level":  float(getattr(agent, "last_injury_level", agent.injury_level)),
        "last_children_count":int(getattr(agent, "last_children_count", len(agent.children_ids))),
    }


# ---------------------------------------------------------------------------
# Public: save
# ---------------------------------------------------------------------------

def save_full_checkpoint(world, config: dict, path: str) -> None:
    """
    Save a complete, resumable simulation checkpoint to *path* (JSON).

    Parameters
    ----------
    world  : WorldState   — current simulation state.
    config : dict         — the experiment configuration dictionary.
    path   : str          — output file path (must end in .json).
    """
    # Sparse death-density map
    sparse_density: list = []
    nz = np.argwhere(world.death_density_map > 0.01)
    for cy, cx in nz:
        sparse_density.append([int(cx), int(cy),
                                round(float(world.death_density_map[cy, cx]), 3)])

    # Shelter registry
    shelters_d: dict = {}
    for loc, sh in (getattr(world, "shelters", None) or {}).items():
        shelters_d[_loc_key(loc)] = {
            "durability": float(sh["durability"]),
            "level":      int(sh.get("level", 1)),
            "owner_id":   sh.get("owner_id"),
        }

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "genesis_phase":      GENESIS_PHASE,
        # world identity
        "seed":   int(world.seed),
        "width":  int(world.width),
        "height": int(world.height),
        "tick":   int(world.tick),
        # counters
        "total_births":      int(world.total_births),
        "total_deaths":      int(world.total_deaths),
        "generation_number": int(world.generation_number),
        "next_agent_id":     int(getattr(world, "next_agent_id", len(world.agents))),
        "max_population":    int(getattr(world, "max_population", 64)),
        # full experiment config (all run_test.py parameters)
        "config": config,
        # colony resource stores
        "colonies": [
            {
                "id":           int(c["id"]),
                "name":         str(c["name"]),
                "color":        str(c["color"]),
                "stored_food":  float(c.get("stored_food",  0.0)),
                "stored_water": float(c.get("stored_water", 0.0)),
                "founder_ids":  [int(i) for i in c.get("founder_ids", [])],
            }
            for c in getattr(world, "colonies", [])
        ],
        # shelter registry
        "shelters": shelters_d,
        # histories (pruned)
        "population_history": getattr(world, "population_history", []),
        "genetic_history":    getattr(world, "genetic_history",    []),
        "extinction_events":  getattr(world, "extinction_events",  []),
        "events_timeline":    getattr(world, "events_timeline",    [])[-MAX_EVENTS_ENTRIES:],
        "history":            getattr(world, "history",            [])[-MAX_HISTORY_ENTRIES:],
        # death heatmap (sparse)
        "death_density": sparse_density,
        "spawn_conditions": getattr(world, "spawn_conditions", {}),
        "spawn_mode":       getattr(world, "spawn_mode",       "fixed"),
        "climate_epoch_state": getattr(world, "climate_epoch_state", None),
        "evolution_journal_history": getattr(world, "evolution_journal_history", None),
        # RNG states for absolute determinism on resume
        "random_state": {
            "numpy": [
                np.random.get_state()[0],
                np.random.get_state()[1].tolist(),
                int(np.random.get_state()[2]),
                int(np.random.get_state()[3]),
                float(np.random.get_state()[4])
            ],
            "python": [
                random.getstate()[0],
                list(random.getstate()[1]),
                random.getstate()[2]
            ]
        },
        # all agent states
        "agents": [_ser_agent(a) for a in world.agents],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, separators=(",", ":"))   # compact — faster write

    size_mb = len(json.dumps(checkpoint, separators=(",", ":")).encode()) / (1024 * 1024)
    year = world.tick // 360
    print(f"  💾 Full checkpoint → {path}  ({size_mb:.1f} MB | tick {world.tick} | year {year})")


# ---------------------------------------------------------------------------
# Public: load
# ---------------------------------------------------------------------------

def load_full_checkpoint(path: str) -> Tuple[Any, dict]:
    """
    Load a full checkpoint and reconstruct the WorldState and config dict.

    Returns
    -------
    (world, config)
        world  — fully reconstructed WorldState, ready to pass to run_simulation().
        config — the config dict that was in effect at save time.
    """
    from world.generator import generate_world
    from world.agents.agent import Agent, Memory, Knowledge
    from world.agents.genetics import Genome
    from world.agents.cognitive import Predictor, Concept, Procedure
    from world.agents.drives import Relationship

    print(f"\n{'='*60}")
    print(f"  PROJECT GENESIS — Loading Checkpoint")
    print(f"  {path}")
    print(f"{'='*60}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ver = data.get("checkpoint_version", 0)
    if ver < CHECKPOINT_VERSION:
        print(f"  ⚠ Checkpoint version {ver} (current: {CHECKPOINT_VERSION}). "
              f"Proceeding with best-effort compatibility.")

    seed   = int(data["seed"])
    width  = int(data.get("width",  1024))
    height = int(data.get("height", 1024))
    tick   = int(data["tick"])
    config = data.get("config", {})
    n_agents = len(data["agents"])

    print(f"  Seed:    {seed}")
    print(f"  Tick:    {tick}  (Year {tick // 360})")
    print(f"  Agents:  {n_agents}")
    print(f"  Phase:   {data.get('genesis_phase', '?')}")
    print()
    print(f"  Regenerating world from seed {seed} (deterministic, ~2-3 s)...")

    world = generate_world(width=width, height=height, seed=seed)

    # ---- world counters ----
    world.tick              = tick
    world.total_births      = int(data["total_births"])
    world.total_deaths      = int(data["total_deaths"])
    world.generation_number = int(data["generation_number"])
    world.next_agent_id     = int(data.get("next_agent_id", 0))
    world.max_population    = int(data.get("max_population", 64))

    # ---- histories ----
    world.population_history = data.get("population_history", [])
    world.genetic_history    = data.get("genetic_history",    [])
    world.extinction_events  = data.get("extinction_events",  [])
    world.events_timeline    = data.get("events_timeline",    [])
    world.history            = data.get("history",            [])
    world.spawn_conditions   = data.get("spawn_conditions",   {})
    world.spawn_mode         = data.get("spawn_mode",         "fixed")
    world.climate_epoch_state = data.get("climate_epoch_state", None)
    world.evolution_journal_history = data.get("evolution_journal_history", None)

    # ---- Restore RNG states for absolute determinism ----
    if "random_state" in data:
        r_state = data["random_state"]
        try:
            py_state = r_state["python"]
            random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))
        except Exception as py_err:
            print(f"  Warning: Python RNG state restore failed: {py_err}")
            
        try:
            np_state_data = r_state["numpy"]
            np_state = (
                np_state_data[0],
                np.array(np_state_data[1], dtype=np.uint32),
                int(np_state_data[2]),
                int(np_state_data[3]),
                float(np_state_data[4])
            )
            np.random.set_state(np_state)
        except Exception as np_err:
            print(f"  Warning: NumPy RNG state restore failed: {np_err}")

    # ---- death density ----
    world.death_density_map[:] = 0.0
    for cx, cy, val in data.get("death_density", []):
        if 0 <= int(cy) < height and 0 <= int(cx) < width:
            world.death_density_map[int(cy), int(cx)] = float(val)

    # ---- shelters ----
    world.shelters = {}
    for key_str, sh in data.get("shelters", {}).items():
        loc = _parse_loc_key(key_str)
        world.shelters[loc] = {
            "durability": float(sh["durability"]),
            "level":      int(sh.get("level", 1)),
            "owner_id":   sh.get("owner_id"),
        }

    # ---- colonies ----
    world.colonies = [
        {
            "id":           int(c["id"]),
            "name":         str(c["name"]),
            "color":        str(c["color"]),
            "stored_food":  float(c.get("stored_food",  0.0)),
            "stored_water": float(c.get("stored_water", 0.0)),
            "founder_ids":  [int(i) for i in c.get("founder_ids", [])],
        }
        for c in data.get("colonies", [])
    ]

    # ---- reconstruct agents ----
    print(f"  Reconstructing {n_agents} agents (genomes, neural nets, drives, memories)...")
    world.agents = []

    for ad in data["agents"]:
        loc    = tuple(ad["location"])
        genome = Genome(np.array(ad["genome_genes"], dtype=np.float32))
        agent  = Agent(agent_id=int(ad["id"]), location=loc, genome=genome, reconstruct=True)
        agent.id             = int(ad["id"])
        agent.colony_id      = int(ad.get("colony_id", 0))
        agent.generation     = int(ad.get("generation", 0))
        agent.born_tick      = int(ad.get("born_tick", 0))
        pi = ad.get("parent_ids")
        agent.parent_ids     = tuple(pi) if pi else None
        agent.children_ids   = [int(i) for i in ad.get("children_ids", [])]
        agent.spawn_biome    = str(ad.get("spawn_biome", "Unknown"))
        agent.archetype      = str(ad.get("archetype", "Balanced"))
        agent.behavior_cluster = str(ad.get("behavior_cluster", "C0"))

        # location
        agent.location       = tuple(ad["location"])
        agent.home_location  = tuple(ad["home_location"])
        agent.spawn_location = tuple(ad["spawn_location"])

        # physiology
        agent.health          = float(ad["health"])
        agent.dead            = bool(ad["dead"])
        agent.cause_of_death  = ad.get("cause_of_death")
        agent.primary_cause   = ad.get("primary_cause")
        agent.secondary_cause = ad.get("secondary_cause")
        agent.age             = int(ad["age"])
        agent.max_age         = int(ad["max_age"])
        agent.hunger          = float(ad["hunger"])
        agent.thirst          = float(ad["thirst"])
        agent.energy          = float(ad["energy"])
        agent.curiosity_need  = float(ad.get("curiosity_need", 0.0))
        agent.fat_reserves    = float(ad["fat_reserves"])
        agent.muscle_mass     = float(ad["muscle_mass"])
        agent.injury_level    = float(ad["injury_level"])

        # damage accumulators
        agent.starvation_damage_accumulated  = float(ad.get("starvation_damage_accumulated",  0.0))
        agent.dehydration_damage_accumulated = float(ad.get("dehydration_damage_accumulated", 0.0))
        agent.exposure_damage_accumulated    = float(ad.get("exposure_damage_accumulated",    0.0))
        agent.injury_damage_accumulated      = float(ad.get("injury_damage_accumulated",      0.0))
        agent.age_damage_accumulated         = float(ad.get("age_damage_accumulated",         0.0))

        # shelter
        sl = ad.get("shelter_location")
        agent.shelter_location   = tuple(sl) if sl else None
        agent.shelter_level      = int(ad.get("shelter_level",      0))
        agent.shelter_durability = float(ad.get("shelter_durability", 0.0))

        # colony resources
        agent.stored_food  = float(ad.get("stored_food",  0.0))
        agent.stored_water = float(ad.get("stored_water", 0.0))

        # reproduction
        agent.reproduction_cooldown = int(ad.get("reproduction_cooldown", 0))

        # traits
        agent.traits      = {k: float(v) for k, v in ad.get("traits", {}).items()}
        agent.base_traits = agent.traits.copy()

        # neural predictor (weights)
        pd = ad.get("predictor", {})
        agent.predictor = Predictor(reconstruct=True)
        if pd:
            agent.predictor.w1 = np.array(pd["w1"], dtype=np.float32)
            agent.predictor.b1 = np.array(pd["b1"], dtype=np.float32)
            agent.predictor.w2 = np.array(pd["w2"], dtype=np.float32)
            agent.predictor.b2 = np.array(pd["b2"], dtype=np.float32)

        # feature weights
        agent.feature_weights = np.array(ad.get("feature_weights", [0.0]*5), dtype=np.float32)

        # cognition
        agent.learning_rate         = float(ad.get("learning_rate", 0.05))
        agent.prediction_confidence = float(ad.get("prediction_confidence", 1.0))
        agent.plan_commitment       = float(ad.get("plan_commitment", 0.5))
        agent.plan_age              = int(ad.get("plan_age", 0))
        agent.planning_accuracy     = float(ad.get("planning_accuracy", 1.0))
        agent.prediction_accuracy   = float(ad.get("prediction_accuracy", 1.0))
        agent.memory_accuracy       = float(ad.get("memory_accuracy", 1.0))

        # planning targets
        agent.current_goal   = ad.get("current_goal")
        ct = ad.get("current_target")
        agent.current_target = tuple(ct) if ct else None
        
        aq_data = ad.get("action_queue", [])
        agent.action_queue   = [(act, tuple(tgt) if tgt else None) for act, tgt in aq_data]
        
        wr = ad.get("wants_to_reproduce_with")
        agent._wants_to_reproduce_with = tuple(wr) if wr else None

        # Phase 8.1 / 8.2 drives
        dd = ad.get("drives", {})
        ds = agent.drives
        for field, default in [
            ("hunger_tension", 0.0), ("thirst_tension", 0.0),
            ("exhaustion_tension", 0.0), ("pain_tension", 0.0),
            ("thermal_stress", 0.0), ("fear", 0.0), ("frustration", 0.0),
            ("longing", 0.0), ("grief", 0.0), ("boredom", 0.0),
        ]:
            setattr(ds, field, float(dd.get(field, default)))
        for field, default in [
            ("_same_action_streak", 0), ("_unmet_hunger_ticks", 0),
            ("_unmet_thirst_ticks", 0), ("_ticks_since_known_agent", 0),
        ]:
            setattr(ds, field, int(dd.get(field, default)))
        ds._prev_pain           = float(dd.get("_prev_pain", 0.0))
        ds.fear_history         = list(dd.get("fear_history", []))
        ds.frustration_history  = list(dd.get("frustration_history", []))
        ds.longing_history      = list(dd.get("longing_history", []))
        ds.grief_history        = list(dd.get("grief_history", []))
        ds.boredom_history      = list(dd.get("boredom_history", []))

        # Phase 8.4 / 8.5 motivation
        md = ad.get("motivation", {})
        for dim_name in ("safety", "family", "exploration", "knowledge", "comfort", "dominance"):
            dd2  = md.get(dim_name, {})
            dim  = getattr(agent.motivation, dim_name)
            dim.current      = float(dd2.get("current",      0.5))
            dim.mean         = float(dd2.get("mean",         0.5))
            dim.variance     = float(dd2.get("variance",     0.0))
            dim.update_count = int(dd2.get("update_count",   0))

        # Phase 8.3 relationships
        agent.relationships = {}
        for oid_s, rd in ad.get("relationships", {}).items():
            rel = Relationship()
            rel.trust        = float(rd.get("trust",        0.0))
            rel.attachment   = float(rd.get("attachment",   0.0))
            rel.respect      = float(rd.get("respect",      0.0))
            rel.fear         = float(rd.get("fear",         0.0))
            rel.dominance    = float(rd.get("dominance",    0.0))
            rel.reciprocity  = float(rd.get("reciprocity",  0.0))
            rel.competition  = float(rd.get("competition",  0.0))
            rel.last_seen_tick = int(rd.get("last_seen_tick", -1))
            agent.relationships[int(oid_s)] = rel

        # Knowledge
        agent.knowledge = Knowledge()
        for ks, v in ad.get("knowledge_water",  {}).items():
            restored_node = dict(v)
            if "active_seasons" in restored_node:
                restored_node["active_seasons"] = {int(s_k): int(s_v) for s_k, s_v in restored_node["active_seasons"].items()}
            if "dry_seasons" in restored_node:
                restored_node["dry_seasons"] = {int(s_k): int(s_v) for s_k, s_v in restored_node["dry_seasons"].items()}
            agent.knowledge.water_sources[_parse_loc_key(ks)] = restored_node

        for ks, v in ad.get("knowledge_food",   {}).items():
            restored_node = dict(v)
            if "active_seasons" in restored_node:
                restored_node["active_seasons"] = {int(s_k): int(s_v) for s_k, s_v in restored_node["active_seasons"].items()}
            if "dry_seasons" in restored_node:
                restored_node["dry_seasons"] = {int(s_k): int(s_v) for s_k, s_v in restored_node["dry_seasons"].items()}
            agent.knowledge.food_sources[_parse_loc_key(ks)] = restored_node
        for ks, v in ad.get("knowledge_danger", {}).items():
            agent.knowledge.danger_locations[_parse_loc_key(ks)] = float(v)

        # Episodic memory
        agent.episodic_memory = []
        for me in ad.get("episodic_memory", []):
            agent.episodic_memory.append(Memory(
                type=me["type"],
                location=tuple(me["location"]),
                timestamp=int(me["timestamp"]),
                importance=float(me["importance"]),
                confidence=float(me["confidence"]),
                associated_id=int(me.get("associated_id", -1)),
                outcome=str(me.get("outcome", "neutral")),
            ))

        # Concepts
        agent.concepts = {"WATER": [], "FOOD": []}
        for rtype, clist in ad.get("concepts", {}).items():
            for cd in clist:
                c = Concept(resource_type=cd["resource_type"])
                c.elevation_mean = cd.get("elevation_mean")
                c.elevation_std  = cd.get("elevation_std")
                c.temp_mean      = cd.get("temp_mean")
                c.temp_std       = cd.get("temp_std")
                c.rain_mean      = cd.get("rain_mean")
                c.rain_std       = cd.get("rain_std")
                c.support        = int(cd.get("support",    0))
                c.confidence     = float(cd.get("confidence", 1.0))
                agent.concepts.setdefault(rtype, []).append(c)

        # Procedures
        agent.procedures = []
        for pe in ad.get("procedures", []):
            agent.procedures.append(Procedure(
                trigger_season=int(pe["trigger_season"]),
                action_sequence=tuple(pe["action_sequence"]),
                success_count=int(pe.get("success_count", 1)),
                confidence=float(pe.get("confidence", 1.0)),
            ))

        # Action tracking
        saved_ac = ad.get("action_counts", {})
        for k in list(agent.action_counts.keys()):
            agent.action_counts[k] = int(saved_ac.get(k, 0))
        agent.action_history = list(ad.get("action_history", []))
        agent.current_action = str(ad.get("current_action", "Idle"))

        # Telemetry counters
        agent.drinks_count          = int(ad.get("drinks_count", 0))
        agent.eats_count            = int(ad.get("eats_count", 0))
        agent.resting_ticks         = int(ad.get("resting_ticks", 0))
        agent.discoveries_count     = int(ad.get("discoveries_count", 0))
        agent.failed_water_visits   = int(ad.get("failed_water_visits", 0))
        agent.failed_food_visits    = int(ad.get("failed_food_visits", 0))
        agent.nodes_added_count     = int(ad.get("nodes_added_count", 0))
        agent.nodes_removed_count   = int(ad.get("nodes_removed_count", 0))
        agent.ticks_survived        = int(ad.get("ticks_survived", 0))
        agent.max_radius            = float(ad.get("max_radius", 0.0))
        agent.prediction_attempts   = int(ad.get("prediction_attempts", 0))
        agent.prediction_successes  = int(ad.get("prediction_successes", 0))
        agent.prediction_decisions  = int(ad.get("prediction_decisions", 0))
        agent.prediction_gains      = int(ad.get("prediction_gains", 0))
        agent.rediscoveries         = int(ad.get("rediscoveries", 0))
        agent.decision_evals        = int(ad.get("decision_evals", 0))
        agent.last_prediction_error = float(ad.get("last_prediction_error", 0.0))

        # Social
        agent.known_agents = set(int(i) for i in ad.get("known_agents", []))

        # Season observations
        agent.season_observations = {
            int(k): int(v) for k, v in ad.get("season_observations",
                                               {"0":0,"1":0,"2":0,"3":0}).items()
        }

        # Exploration tracking
        agent.visited_chunks       = set(tuple(c) for c in ad.get("visited_chunks", []))
        agent.discovered_landmarks = set(ad.get("discovered_landmarks", []))
        agent.discovered_water     = set(tuple(loc) for loc in ad.get("discovered_water", []))
        agent.discovered_food      = set(tuple(loc) for loc in ad.get("discovered_food", []))

        # Path history
        sph = ad.get("sampled_path_history", [])
        agent.sampled_path_history = sph if sph else [[
            int(agent.location[1]), int(agent.location[0]),
            0, float(agent.health), float(agent.hunger),
            float(agent.thirst), float(agent.energy), int(agent.generation)
        ]]
        agent.path_history = [agent.location]

        # Telemetry continuity
        agent.last_health         = float(ad.get("last_health",         agent.health))
        agent.last_hunger         = float(ad.get("last_hunger",         agent.hunger))
        agent.last_thirst         = float(ad.get("last_thirst",         agent.thirst))
        agent.last_injury_level   = float(ad.get("last_injury_level",   agent.injury_level))
        agent.last_children_count = int(ad.get("last_children_count",   len(agent.children_ids)))

        # ---- Restore learning buffers and predictor inputs ----
        tb_data = ad.get("training_buffer", [])
        agent.training_buffer = [(np.array(inp, dtype=np.float32), float(rew)) for inp, rew in tb_data]
        
        pb_data = ad.get("procedural_buffer", [])
        agent.procedural_buffer = [(int(s), tuple(seq), float(rew)) for s, seq, rew in pb_data]
        
        lpi = ad.get("last_prediction_input")
        agent.last_prediction_input = np.array(lpi, dtype=np.float32) if lpi is not None else None
        agent.plan_cache              = {}
        agent.utility_cache           = {}
        agent.memory_cache            = {
            "Best Water": None, "Best Food": None,
            "Best Shelter": None, "Best Mate": None, "last_cache_tick": -1
        }
        agent._wants_to_reproduce_with = None
        agent.action_queue             = []
        agent.current_goal             = None
        agent.current_target           = None
        agent.target_coordinate        = None
        agent._last_drive_action       = "Idle"
        # Event flags (reset each tick)
        agent._consumed_food      = False
        agent._consumed_water     = False
        agent._shelter_upgraded   = False
        agent._failed_visit       = False
        agent._stepped            = False

        world.agents.append(agent)

    alive = sum(1 for a in world.agents if not a.dead)
    print(f"  ✅ Checkpoint loaded: {len(world.agents)} agents restored "
          f"({alive} alive) | tick {tick} → resuming from year {tick // 360}")
    print(f"{'='*60}\n")
    return world, config
