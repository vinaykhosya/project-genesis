"""
Reproduction system for Project Genesis — Phase 5.

Implements sexual reproduction with genetic crossover and mutation.
Reproduction is gated on biological state alone — there is no 'reproduction
drive' gene. Whether an agent pursues reproduction emerges from the utility
evaluation in decision.py comparing it against all other needs.

Children start with zero memories. Cultural transmission must be earned
through active interaction — memory inheritance would conflate evolution
with learning, obscuring both signals.
"""

import numpy as np
from .agent import Agent
from .genetics import crossover, mutate, express_genome

# ---------------------------------------------------------------------------
# Biological reproduction requirements (hardcoded as physics / biology).
# These are NOT agent traits — evolution operates within these constraints.
# ---------------------------------------------------------------------------
MIN_REPRO_AGE_TICKS  = 14 * 360     # Must be an Adult (≥ 14 years)
MAX_REPRO_AGE_TICKS  = 60 * 360     # Elders (≥ 60 years) cannot reproduce
MIN_HEALTH           = 75.0
MIN_FAT_RESERVES     = 70.0
MAX_HUNGER_TO_REPRO  = 30.0
MAX_THIRST_TO_REPRO  = 30.0
MIN_SHELTER_DURA     = 60.0
MAX_INJURY_TO_REPRO  = 20.0
MATE_SEARCH_RADIUS   = 30            # Grid cells; mirrors Adult vision baseline

# Reproduction costs applied to both parents on a successful birth
PARENT_FAT_COST      = 20.0
PARENT_ENERGY_COST   = 10.0
RECOVERY_TICKS       = 360           # 1 year pregnancy/recovery cooldown


def is_eligible_to_reproduce(agent) -> bool:
    """
    Returns True if this agent satisfies all biological requirements for
    reproduction. Checked both in evaluate_utility (for action selection)
    and in attempt_reproduce (final safety gate before birth).
    """
    return (
        not agent.dead
        and MIN_REPRO_AGE_TICKS <= agent.age < MAX_REPRO_AGE_TICKS
        and agent.health       >= MIN_HEALTH
        and agent.fat_reserves >= MIN_FAT_RESERVES
        and agent.hunger       <= MAX_HUNGER_TO_REPRO
        and agent.thirst       <= MAX_THIRST_TO_REPRO
        and agent.shelter_durability >= MIN_SHELTER_DURA
        and agent.injury_level <= MAX_INJURY_TO_REPRO
        and agent.reproduction_cooldown == 0
    )


def find_eligible_mate(agent, world, context=None) -> "Agent | None":
    """
    Scans within MATE_SEARCH_RADIUS for the nearest eligible colony member.
    Returns None if no partner is available.
    Only same-colony partners qualify (colony_id matching).
    """
    cy, cx = agent.location
    best_dist_sq = MATE_SEARCH_RADIUS ** 2
    best_mate = None

    if context is not None and hasattr(context, "nearby_agents"):
        candidates = []
        for other in context.nearby_agents:
            oy, ox = other.location
            dist_sq = (oy - cy) ** 2 + (ox - cx) ** 2
            if dist_sq <= best_dist_sq:
                candidates.append(other)
    else:
        candidates = world.query_agents(agent.location, MATE_SEARCH_RADIUS, alive_only=True)

    for other in candidates:
        if other.id == agent.id:
            continue
        if getattr(other, "colony_id", -1) != getattr(agent, "colony_id", -1):
            continue
        if not is_eligible_to_reproduce(other):
            continue
        # Threat check: refuse to mate with agents who have broken trust
        if agent.get_social_modifier(other.id, world.tick) < 0.6:
            continue
        oy, ox = other.location
        dist_sq = (oy - cy) ** 2 + (ox - cx) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_mate = other

    return best_mate


def attempt_reproduce(
    parent_a: Agent,
    parent_b: Agent,
    world,
    next_agent_id: int,
    mutation_rate: float = 0.05,
) -> "Agent | None":
    """
    Attempts to produce a child via sexual reproduction.

    Returns a new Agent if successful, or None if:
    - The alive population has reached world.max_population
    - Biological eligibility conditions are no longer met (state can change
      between utility evaluation and the tick where the action fires)

    The child genome is the crossover of both parent genomes, then mutated
    with Gaussian noise at the fixed simulation mutation_rate. The child
    begins life with zero memories — any cultural knowledge must be acquired
    through its own experience.
    """
    # Population cap check (soft carrying-capacity pressure)
    alive_count = sum(1 for a in world.agents if not a.dead)
    max_pop = getattr(world, "max_population", 64)
    if alive_count >= max_pop:
        return None

    # Final eligibility re-check (conditions may have changed this tick)
    if not is_eligible_to_reproduce(parent_a) or not is_eligible_to_reproduce(parent_b):
        return None

    # --- Genetic operations ------------------------------------------------
    child_genome = crossover(parent_a.genome, parent_b.genome)
    child_genome = mutate(child_genome, mutation_rate=mutation_rate)

    # --- Create child Agent first to read cached brain ----------------------
    child = Agent(
        agent_id=next_agent_id,
        location=parent_a.location,   # Born at parent A's location
        genome=child_genome,
    )
    brain = child.brain

    # --- Lifespan from genome + base stochastic distribution ---------------
    base_max_age = int(max(20 * 360, int(np.random.normal(70, 10) * 360)))
    child_max_age = max(20 * 360, base_max_age + brain["max_age_offset"])
    print(f"[BIRTH DEBUG] Child {next_agent_id} genome: {[round(float(g), 4) for g in child.genome.genes]}")
    child.age       = 0              # Newborn Infant
    child.max_age   = child_max_age
    child.health    = 100.0          # Born healthy

    # --- Lineage fields -----------------------------------------------------
    child.colony_id   = parent_a.colony_id
    child.generation  = max(parent_a.generation, parent_b.generation) + 1
    child.parent_ids  = (parent_a.id, parent_b.id)
    child.born_tick   = world.tick
    child.sampled_path_history = [[
        int(child.location[1]), # x (col)
        int(child.location[0]), # y (row)
        0,                      # action_id: Idle (0)
        float(child.health),
        float(child.hunger),
        float(child.thirst),
        float(child.energy),
        int(child.generation)
    ]]

    # Children begin with empty memories — intentional design choice.
    # Cultural transmission must be an active learned interaction, not
    # passive inheritance. Do NOT add memory copying here.

    # --- Link parents to child --------------------------------------------
    parent_a.children_ids.append(next_agent_id)
    parent_b.children_ids.append(next_agent_id)

    # --- Apply reproduction costs to both parents -------------------------
    parent_a.fat_reserves         = max(0.0, parent_a.fat_reserves - PARENT_FAT_COST)
    parent_a.energy               = max(0.0, parent_a.energy - PARENT_ENERGY_COST)
    parent_a.reproduction_cooldown = RECOVERY_TICKS

    parent_b.fat_reserves         = max(0.0, parent_b.fat_reserves - PARENT_FAT_COST)
    parent_b.energy               = max(0.0, parent_b.energy - PARENT_ENERGY_COST)
    parent_b.reproduction_cooldown = RECOVERY_TICKS

    # --- World-level tracking ----------------------------------------------
    world.total_births     += 1
    world.generation_number = max(world.generation_number, child.generation)

    colony_name = "Unknown"
    if hasattr(world, "colonies") and 0 <= parent_a.colony_id < len(world.colonies):
        colony_name = world.colonies[parent_a.colony_id].get("name", colony_name)

    world.history.append(
        f"[BIRTH] Tick {world.tick} | Colony {colony_name} | "
        f"Parents #{parent_a.id} x #{parent_b.id} -> Child #{next_agent_id} "
        f"(Gen {child.generation})"
    )

    if not hasattr(world, "events_timeline") or world.events_timeline is None:
        world.events_timeline = []

    world.events_timeline.append({
        "tick": int(world.tick),
        "year": int(world.tick // 360),
        "day": int(world.tick % 360),
        "type": "Birth",
        "description": f"Child #{next_agent_id} (Gen {child.generation}) born in Colony {colony_name} to Parents #{parent_a.id} and #{parent_b.id}.",
        "metadata": {
            "child_id": int(next_agent_id),
            "parent_a_id": int(parent_a.id),
            "parent_b_id": int(parent_b.id),
            "generation": int(child.generation),
            "location": [int(child.location[1]), int(child.location[0])],  # [x, y]
            "colony_id": int(parent_a.colony_id),
            "colony_name": colony_name
        }
    })

    return child
