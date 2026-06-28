import numpy as np
from .agent import Agent, WATER, FOOD, DANGER, PERSON, LANDMARK
from .genetics import express_genome
from .drives import compute_drive_modulation

def sigmoid_utility(need: float, threshold: float, scale: float) -> float:
    """Computes a biological sigmoid curve for utility values to simulate soft thresholds."""
    z = (need - threshold) / scale
    z = max(-20.0, min(20.0, z)) # Prevent exp overflow/warnings
    return 100.0 / (1.0 + np.exp(-z))

def get_predictor_context(agent: Agent, action_name: str, target_loc: tuple, world, local_temp=None, scarcity_prediction=None) -> np.ndarray:
    """
    Constructs a 20-element context vector for the predictor network.
    """
    cy, cx = agent.location
    ty, tx = target_loc if target_loc is not None else (cy, cx)
    
    # Calculate seasonal target for prediction
    day = world.tick % 360
    season = day // 90
    is_transition = (day % 90) >= 45
    target_season = (season + 1) % 4 if is_transition else season
    
    # Local temp
    if local_temp is None:
        biome_id = int(world.biome[ty, tx])
        if biome_id == 6:
            amplitude = 22.0
        elif biome_id in (1, 2, 3):
            amplitude = 18.0
        elif biome_id in (4, 5):
            amplitude = 12.0
        elif biome_id in (0, 8):
            amplitude = 6.0
        elif biome_id == 7:
            amplitude = 4.0
        else:
            amplitude = 12.0
        base_temp = float(world.temperature[ty, tx])
        cos_factor = getattr(world, "temp_cos_factor", None)
        if cos_factor is None:
            cos_factor = np.cos(((day - 180.0) / 180.0) * np.pi)
        temp_seasonal_offset = amplitude * cos_factor
        global_offset = getattr(world, "global_temp_offset", 0.0)
        local_temp = base_temp + temp_seasonal_offset + global_offset
    
    # Scarcity prediction
    if scarcity_prediction is None:
        total_active = 0
        total_obs = 0
        for node in list(agent.knowledge.water_sources.values()) + list(agent.knowledge.food_sources.values()):
            active_count = node.get("active_seasons", {}).get(target_season, 0)
            dry_count = node.get("dry_seasons", {}).get(target_season, 0)
            total_active += active_count
            total_obs += (active_count + dry_count)
        abundance_ratio = total_active / total_obs if total_obs > 0 else 0.8
        scarcity_prediction = 1.0 - abundance_ratio
        scarcity_prediction = 0.2 + (scarcity_prediction - 0.2) * agent.prediction_confidence

    c_id = getattr(agent, "colony_id", -1)
    colony_food = 0.0
    colony_water = 0.0
    if c_id >= 0 and c_id < len(world.colonies):
        colony_food = world.colonies[c_id].get("stored_food", 0.0)
        colony_water = world.colonies[c_id].get("stored_water", 0.0)
        
    action_names = [
        "Resting", "Drinking", "Eating", "Exploring", "Building Shelter", 
        "Sheltering", "Reproduce", "Store Food", "Store Water", "Share Food", 
        "Share Water", "Drink Stored Water", "Eat Stored Food", "Deposit Food", 
        "Deposit Water", "Withdraw Food", "Withdraw Water"
    ]
    try:
        action_idx = action_names.index(action_name)
    except ValueError:
        action_idx = 0
        
    vec = np.array([
        agent.hunger / 100.0,
        agent.thirst / 100.0,
        (100.0 - agent.energy) / 100.0,
        agent.injury_level / 100.0,
        agent.stored_food / 100.0,
        agent.stored_water / 100.0,
        colony_food / 100.0,
        colony_water / 100.0,
        local_temp / 50.0,
        float(world.elevation[ty, tx]),
        float(world.rainfall[ty, tx] / 3000.0),
        1.0 if agent.shelter_location is not None else 0.0,
        agent.shelter_durability / 100.0,
        agent.reproduction_cooldown / 360.0,
        agent.senescence_factor,
        1.0 if is_transition else 0.0,
        target_season / 3.0,
        scarcity_prediction,
        1.0 if agent.life_stage == "Adult" else 0.0,
        action_idx / 16.0
    ], dtype=np.float32)
    
    return vec

def get_environmental_modulation(agent: Agent, y: int, x: int, world) -> float:
    """
    Extracts normalized environmental features at (y, x) and computes
    their dot product with the agent's feature weights to modulate utility.
    Features: [temperature, elevation, moisture, food_density, water_density]
    """
    # 1. Temperature: normalized to ~[-1.5, 1.5], centered at 20°C
    base_temp = float(world.temperature[y, x])
    day = world.tick % 360
    biome_id = int(world.biome[y, x])
    if biome_id == 6:
        amplitude = 22.0
    elif biome_id in (1, 2, 3):
        amplitude = 18.0
    elif biome_id in (4, 5):
        amplitude = 12.0
    elif biome_id in (0, 8):
        amplitude = 6.0
    elif biome_id == 7:
        amplitude = 4.0
    else:
        amplitude = 12.0
    cos_factor = getattr(world, "temp_cos_factor", None)
    if cos_factor is None:
        cos_factor = np.cos(((day - 180.0) / 180.0) * np.pi)
    temp_seasonal_offset = amplitude * cos_factor
    global_offset = getattr(world, "global_temp_offset", 0.0)
    local_temp = base_temp + temp_seasonal_offset + global_offset
    
    f_temp = (local_temp - 20.0) / 20.0
    
    # 2. Elevation: [0, 1]
    f_elev = float(world.elevation[y, x])
    
    # 3. Moisture (rainfall): [0, 1]
    f_moisture = float(world.rainfall[y, x] / 3000.0)
    
    # 4. Food Density: [0, 2]
    f_food = float(world.wildlife[y, x] + world.fertility[y, x])
    
    # 5. Water Density: 1.0 if lake/active river, else 0.0
    has_water = (world.biome[y, x] == 8) or (world.lake_map[y, x] > 0.0) or (world.river_map[y, x] > 1500.0)
    f_water = 1.0 if has_water else 0.0
    
    features = np.array([f_temp, f_elev, f_moisture, f_food, f_water], dtype=np.float32)
    
    # Dot product with agent's learned feature weights
    if not hasattr(agent, "feature_weights"):
        agent.feature_weights = np.zeros(5, dtype=np.float32)
    val = float(np.dot(agent.feature_weights, features))
    
    # Return exponential modulation multiplier, clamped to prevent extreme values
    return float(np.clip(np.exp(val), 0.1, 5.0))

# ==============================================================================
# COMBAT HELPER FUNCTIONS
# Shared by update_agent_needs for the two-stage territorial encounter system.
# Parameters drive the outcome; the existence of each stage is hardcoded biology.
# ==============================================================================

def _is_in_home_radius(agent) -> bool:
    """
    Returns True if the agent is within its own genome-derived home radius.
    Agents do not engage in territorial friction inside their own home territory
    (den/nest safety) — this models universal animal behaviour, not a game rule.
    The radius is set from aggression_mult at birth and can evolve.
    """
    hy, hx = agent.home_location
    ay, ax = agent.location
    return (ay - hy) ** 2 + (ax - hx) ** 2 <= agent.home_radius ** 2


def _encounter_reason(agent, world) -> str:
    """
    Infer the spatial context of an encounter from environmental signals.
    Logged in disputes_history to allow post-hoc analysis of conflict triggers.
    Returns one of: 'CHEST', 'SHELTER', 'RESOURCE', 'TERRITORIAL'.
    """
    loc = agent.location
    cy, cx = loc
    # Colony chest check
    for col in getattr(world, "colonies", []):
        if col.get("chest_location") == loc:
            return "CHEST"
    # Own shelter check
    if agent.shelter_location is not None and agent.shelter_location == loc:
        return "SHELTER"
    # Resource cell check (wildlife, fertility, water)
    if (world.wildlife[cy, cx] > 0.3 or
            world.fertility[cy, cx] > 0.3 or
            world.lake_map[cy, cx] > 0.1 or
            world.river_map[cy, cx] > 1500.0):
        return "RESOURCE"
    return "TERRITORIAL"


def _fight_confidence_components(agent, other) -> dict:
    """
    Decomposes fight confidence into named components.
    Returns a dict so callers and telemetry can inspect individual drivers.
    New terms (allies_nearby, energy, territory_bonus) can be added here
    later without changing any call site.

    confidence = my_strength / (my_strength + their_strength)
    where strength = health × injury_survival × (aggression + resilience_bonus)
    """
    h_score  = agent.health / 100.0
    inj_score = 1.0 - (agent.injury_level / 100.0)   # 1.0 = fully healthy
    aggr      = agent.brain["aggression_mult"] / 2.0  # normalise 0–2 → 0–1
    resil     = agent.brain["resilience"]

    oh_score  = other.health / 100.0
    oinj      = 1.0 - (other.injury_level / 100.0)
    oaggr     = other.brain["aggression_mult"] / 2.0
    oresil    = other.brain["resilience"]

    my_str    = h_score  * inj_score  * (aggr  + resil  * 0.3)
    their_str = oh_score * oinj       * (oaggr + oresil * 0.3)

    if their_str < 1e-6:
        confidence = 1.0
    else:
        confidence = min(1.0, my_str / (my_str + their_str))

    return {
        "health":     round(h_score,    3),
        "injury":     round(inj_score,  3),
        "aggression": round(aggr,       3),
        "resilience": round(resil,      3),
        "confidence": round(confidence, 3),
    }


def _record_combat_memory(agent, other, world, agent_won: bool) -> None:
    """
    Stores a win or loss in both agents' episodic memories and updates
    lifetime counters. A future encounter with the same associated_id
    will influence confidence through the existing relationship system.
    Sets a genome+injury proportional fear cooldown on the loser.
    """
    cy, cx = agent.location

    if agent_won:
        agent.fight_wins   += 1
        other.fight_losses += 1
        agent.add_memory(DANGER,    (cy, cx), world.tick, importance=0.45,
                         associated_id=other.id,  outcome="combat_win")
        other.add_memory(DANGER,    (cy, cx), world.tick, importance=0.80,
                         associated_id=agent.id,  outcome="combat_loss")
        # Fear cooldown on the loser: base 20 ticks + injury pressure + genome risk sensitivity
        # risk_mult 0.5 (bold) → shorter recovery; 2.0 (cautious) → longer recovery
        cooldown = int(20 + other.injury_level * 0.8 * other.brain["risk_mult"] / 1.25)
        other.combat_fear_timer  = cooldown
        other.combat_fear_target = agent.id
    else:
        agent.fight_losses += 1
        other.fight_wins   += 1
        other.add_memory(DANGER,    (cy, cx), world.tick, importance=0.45,
                         associated_id=agent.id,  outcome="combat_win")
        agent.add_memory(DANGER,    (cy, cx), world.tick, importance=0.80,
                         associated_id=other.id,  outcome="combat_loss")
        cooldown = int(20 + agent.injury_level * 0.8 * agent.brain["risk_mult"] / 1.25)
        agent.combat_fear_timer  = cooldown
        agent.combat_fear_target = other.id


def update_agent_needs(agent: Agent, world, moved: bool, context=None):
    """
    Increases need levels per tick. Need depletion rate is scaled 
    by the local movement cost if the agent moved.
    Decays spatial knowledge confidence. Handles health reduction and death.
    """
    if agent.dead:
        return
        
    prev_health = agent.health
    cy, cx = agent.location
    cost = float(world.movement_cost[cy, cx])
    
    # --- 1. Biome-Specific Seasonal Temperature Calculation ---
    if context is not None and hasattr(context, "local_temp"):
        local_temp = context.local_temp
    else:
        biome_id = int(world.biome[cy, cx])
        # Amplitude mapping based on biome climate characteristics
        if biome_id == 6:  # DESERT
            amplitude = 22.0
        elif biome_id in (1, 2, 3):  # GLACIER, TUNDRA, TAIGA
            amplitude = 18.0
        elif biome_id in (4, 5):  # TEMPERATE_FOREST, GRASSLAND
            amplitude = 12.0
        elif biome_id in (0, 8):  # OCEAN, LAKE
            amplitude = 6.0
        elif biome_id == 7:  # RAINFOREST
            amplitude = 4.0
        else:
            amplitude = 12.0
            
        base_temp = float(world.temperature[cy, cx])
        day = world.tick % 360
        temp_seasonal_offset = amplitude * np.cos(((day - 180.0) / 180.0) * np.pi)
        global_offset = getattr(world, "global_temp_offset", 0.0)
        local_temp = base_temp + temp_seasonal_offset + global_offset
    
    # --- 2. Shelter Protection Check ---
    is_sheltered = False
    protection = 0.0
    if agent.shelter_location is not None:
        # Check if agent is currently at shelter and active action is Sheltering/Resting
        if agent.location == agent.shelter_location and agent.current_action in ("Sheltering", "Resting"):
            is_sheltered = True
            max_protection = 0.0
            if agent.shelter_level == 1:
                max_protection = 0.60
            elif agent.shelter_level == 2:
                max_protection = 0.80
            elif agent.shelter_level == 3:
                max_protection = 0.95
            protection = max_protection * (agent.shelter_durability / 100.0)
            
    # --- 3. Layered Temperature Pressures (Metabolic Modifiers) ---
    hunger_modifier = 1.0
    thirst_modifier = 1.0
    energy_drain_modifier = 1.0
    
    # --- Genome brain parameters for physiology ---
    brain = agent.brain

    if local_temp < 12.0:  # Cold Pressure
        cold_severity = 12.0 - local_temp
        # g_thermoregulation scales the effective cold penalty: high gene → less stress
        cold_net = cold_severity * (1.0 - protection) * (1.0 - brain["cold_resistance"])
        if cold_net > 0:
            hunger_modifier += 0.04 * cold_net
            energy_drain_modifier += 0.05 * cold_net
            thirst_modifier = max(0.5, thirst_modifier - 0.01 * cold_net)
    elif local_temp > 28.0:  # Heat Pressure
        heat_severity = local_temp - 28.0
        # g_thermoregulation also scales heat penalty
        heat_net = heat_severity * (1.0 - protection) * (1.0 - brain["heat_resistance"])
        if heat_net > 0:
            thirst_modifier += 0.10 * heat_net
            energy_drain_modifier += 0.03 * heat_net
            hunger_modifier = max(0.8, hunger_modifier - 0.005 * heat_net)

    # Senescence Metabolic Strain (up to 50% increase in need growth)
    r = agent.senescence_factor
    age_multiplier = 1.0 + 0.5 * r
    hunger_modifier *= age_multiplier
    thirst_modifier *= age_multiplier
    energy_drain_modifier *= age_multiplier

    # g_metabolism scales base metabolic rates (0.5× to 1.5× of base)
    hunger_modifier  *= brain["hunger_rate_mult"]
    thirst_modifier  *= brain["thirst_rate_mult"]
    
    # --- 4. Update Needs ---
    if agent.current_action == "Resting":
        agent.resting_ticks += 1
        # Recover energy, scaled down by shivering/heat strain
        agent.energy = min(100.0, agent.energy + 5.0 / energy_drain_modifier)
        # Slowed need decay when asleep
        agent.hunger += 0.05 * hunger_modifier
        agent.thirst += 0.10 * thirst_modifier
        agent.curiosity_need += 0.01
    else:
        # Awake rates
        if moved:
            # Traversal cost multiplies metabolic depletion (moving is hard work)
            # g_mobility (move_energy_mult) scales how costly movement is
            agent.hunger += 0.15 * cost * hunger_modifier
            agent.thirst += 0.30 * cost * thirst_modifier
            agent.energy = max(0.0, agent.energy - 0.12 * cost * energy_drain_modifier * brain["move_energy_mult"])
            agent.curiosity_need += 0.05
        else:
            # Stationary rates
            if agent.current_action == "Building Shelter":
                agent.hunger += 0.20 * hunger_modifier
                agent.thirst += 0.20 * thirst_modifier
                agent.energy = max(0.0, agent.energy - 0.30 * energy_drain_modifier)
            else:
                agent.energy = max(0.0, agent.energy - 0.04 * energy_drain_modifier)
            agent.hunger += 0.05 * hunger_modifier
            agent.thirst += 0.10 * thirst_modifier
            agent.curiosity_need += 0.02

    # Tick down reproduction cooldown
    if agent.reproduction_cooldown > 0:
        agent.reproduction_cooldown -= 1
            
    # Clamp needs
    agent.hunger = np.clip(agent.hunger, 0.0, 100.0)
    agent.thirst = np.clip(agent.thirst, 0.0, 100.0)
    agent.curiosity_need = np.clip(agent.curiosity_need, 0.0, 100.0)
    
    # --- 5. Extreme Exposure Health Damage ---
    if local_temp < -5.0 or local_temp > 45.0:
        cold_dmg = max(0.0, 0.05 * (-5.0 - local_temp))
        heat_dmg = max(0.0, 0.08 * (local_temp - 45.0))
        exp_dmg = cold_dmg + heat_dmg
        exp_dmg_net = exp_dmg * (1.0 - protection)
        
        # Shelter absorbs wear-and-tear
        if is_sheltered and protection > 0.0:
            durability_loss = 0.05 * (exp_dmg - exp_dmg_net)
            agent.shelter_durability = max(0.0, agent.shelter_durability - durability_loss)
            
        if exp_dmg_net > 0.0:
            agent.health -= exp_dmg_net
            agent.exposure_damage_accumulated += exp_dmg_net
            
    # --- 6. Starvation & Dehydration Damage (Reserves Buffer) ---
    starving = agent.hunger >= 100.0
    dehydrated = agent.thirst >= 100.0
    
    if dehydrated:
        if getattr(world, "ecology_ablation", {}).get("dehydration_ramp", True):
            # Ramp damage from 0.4 up to 2.0 over 120 ticks of unmet thirst to give a buffer window
            unmet = getattr(agent.drives, "_unmet_thirst_ticks", 0)
            dehydration_dmg = 0.4 + 1.6 * min(1.0, unmet / 120.0)
        else:
            # Fallback to legacy flat 2.0 cliff damage
            dehydration_dmg = 2.0
            
        agent.health -= dehydration_dmg
        agent.dehydration_damage_accumulated += dehydration_dmg
        agent.add_memory(DANGER, (cy, cx), world.tick, importance=0.95)
        
    if starving:
        # Starving first burns fat reserves
        if agent.fat_reserves > 0.0:
            agent.fat_reserves = max(0.0, agent.fat_reserves - 3.0)
        else:
            # Burning muscle mass degrades movement speed
            agent.muscle_mass = max(0.0, agent.muscle_mass - 1.5)
            if agent.muscle_mass <= 0.0:
                agent.health -= 10.0
                agent.starvation_damage_accumulated += 10.0
            agent.add_memory(DANGER, (cy, cx), world.tick, importance=0.90)
    else:
        # Recover reserves slowly if hunger is satisfied
        if agent.hunger < 50.0:
            if agent.muscle_mass < 100.0:
                agent.muscle_mass = min(100.0, agent.muscle_mass + 0.5)
            elif agent.fat_reserves < 100.0:
                agent.fat_reserves = min(100.0, agent.fat_reserves + 1.0)
                
    # --- 7. Senescence Health Decline ---
    if r > 0.8:
        sen_dmg = 0.5 * (r - 0.8) / 0.2
        agent.health -= sen_dmg
        agent.age_damage_accumulated += sen_dmg
        
    # --- 8. Injury System Trauma & Healing ---
    # Trauma Source 1: Steep terrain falls
    if moved and cost > 1.8:
        if np.random.uniform() < 0.002 * cost:
            injury_gain = float(np.random.uniform(5.0, 15.0))
            agent.injury_level = min(100.0, agent.injury_level + injury_gain)
            if injury_gain > 10.0:
                world.history.append(
                    f"Year {world.tick // 360}, Day {world.tick % 360}: Agent {agent.id} "
                    f"sprained an ankle on steep slope at coordinate ({cx}, {cy})."
                )
                
    # Trauma Source 2: Extreme weather frostbite/heatstroke
    if (local_temp < -5.0 or local_temp > 45.0) and not is_sheltered:
        if np.random.uniform() < 0.05:
            agent.injury_level = min(100.0, agent.injury_level + 8.0)

    # Trauma Source 3: Territorial friction / encounters
    # Two-stage encounter: Assess → Display → Fight (only if both stand ground).
    # Existence of each stage is hardcoded biology; all parameters are genome-derived.
    if getattr(world, "disputes_enabled", True):
        # Decrement fear cooldown timer each tick (biology: stress hormone fading)
        if agent.combat_fear_timer > 0:
            agent.combat_fear_timer -= 1

        # Initialise per-tick pair deduplication set on the world (reset by simulation.py)
        if not hasattr(world, "_disputes_this_tick"):
            world._disputes_this_tick = set()

        # Initialise history list once
        if not hasattr(world, "disputes_history"):
            world.disputes_history = []

        # g_aggression scales encounter probability: high gene → more frequent disputes
        friction_prob = 0.01 * brain["aggression_mult"]
        reason = _encounter_reason(agent, world)

        for other in world.agents:
            if other.id == agent.id or other.dead:
                continue
            if other.location != agent.location:
                continue

            # ── Safe Zone: No friction inside either agent's home radius (den safety) ──
            if _is_in_home_radius(agent) or _is_in_home_radius(other):
                continue

            # ── Fear cooldown: Skip re-engagement with the specific aggressor ──
            if agent.combat_fear_timer > 0 and agent.combat_fear_target == other.id:
                continue

            # Colony members have greatly reduced friction
            same_colony = getattr(other, "colony_id", -1) == getattr(agent, "colony_id", -1)
            prob = friction_prob * (0.2 if same_colony else 1.0)

            # Duplicate evaluation tracking (same ordered pair already fought this tick?)
            pair_key = (min(agent.id, other.id), max(agent.id, other.id))
            is_duplicate = pair_key in world._disputes_this_tick

            # Distance metrics for telemetry
            hay, hax = agent.home_location
            dist_from_home   = float(np.sqrt((cy - hay)**2 + (cx - hax)**2))
            oay, oax = other.location
            dist_between     = float(np.sqrt((cy - oay)**2 + (cx - oax)**2))

            # Relationship trust lookup
            rel_trust = None
            if hasattr(agent, "relationships") and other.id in agent.relationships:
                rel_trust = round(float(agent.relationships[other.id].trust), 3)

            # ── Stage 1: Probability trigger ──
            encounter_triggered = np.random.uniform() < prob

            if not encounter_triggered:
                # Log the skipped encounter for completeness
                if len(world.disputes_history) < 5000:
                    world.disputes_history.append({
                        "tick": world.tick, "year": world.tick // 360, "day": world.tick % 360,
                        "initiator_id": agent.id, "target_id": other.id,
                        "duplicate_eval": is_duplicate,
                        "encounter_reason": reason,
                        "fight_occurred": False, "display_only": False,
                        "display_intensity": 0.0, "confidence_components": None,
                        "initiator_confidence": None, "target_confidence": None,
                        "distance_between": round(dist_between, 2),
                        "dist_from_home": round(dist_from_home, 2),
                        "initiator_health": round(agent.health, 1),
                        "target_health": round(other.health, 1),
                        "initiator_injury": round(agent.injury_level, 1),
                        "target_injury": round(other.injury_level, 1),
                        "initiator_action": agent.current_action,
                        "target_action": other.current_action,
                        "same_colony": same_colony,
                        "relationship_trust": rel_trust,
                    })
                world._disputes_this_tick.add(pair_key)
                continue

            # ── Stage 2: Threat assessment — compute confidence for both agents ──
            my_comps    = _fight_confidence_components(agent, other)
            their_comps = _fight_confidence_components(other, agent)
            my_conf     = my_comps["confidence"]
            their_conf  = their_comps["confidence"]

            # Retreat threshold derived from genome risk sensitivity.
            # risk_mult 0.5 (bold) → threshold 0.33 (fights even at disadvantage)
            # risk_mult 2.0 (cautious) → threshold 0.40 (backs down more readily)
            my_thresh    = 1.0 - agent.brain["risk_mult"] / 3.0
            their_thresh = 1.0 - other.brain["risk_mult"] / 3.0
            both_fight   = (my_conf >= my_thresh) and (their_conf >= their_thresh)

            # Display intensity: magnitude of confidence gap (small diff = polite warning)
            display_intensity = round(abs(my_conf - their_conf), 3)

            if both_fight:
                # ── Stage 3: Combat — both hold their ground ──
                agent.injury_level = min(100.0, agent.injury_level + 5.0)
                other.injury_level = min(100.0, other.injury_level + 5.0)

                # Determine winner by higher confidence (loser gets memory and fear cooldown)
                _record_combat_memory(agent, other, world, agent_won=(my_conf >= their_conf))

                world.history.append(
                    f"Year {world.tick // 360}, Day {world.tick % 360}: Agent {agent.id} "
                    f"and Agent {other.id} had a territorial dispute at coordinate ({cx}, {cy})."
                )

                log_entry = {
                    "tick": world.tick, "year": world.tick // 360, "day": world.tick % 360,
                    "initiator_id": agent.id, "target_id": other.id,
                    "duplicate_eval": is_duplicate,
                    "encounter_reason": reason,
                    "fight_occurred": True, "display_only": False,
                    "display_intensity": display_intensity,
                    "confidence_components": my_comps,
                    "initiator_confidence": round(my_conf, 3),
                    "target_confidence": round(their_conf, 3),
                    "distance_between": round(dist_between, 2),
                    "dist_from_home": round(dist_from_home, 2),
                    "initiator_health": round(agent.health, 1),
                    "target_health": round(other.health, 1),
                    "initiator_injury": round(agent.injury_level, 1),
                    "target_injury": round(other.injury_level, 1),
                    "initiator_action": agent.current_action,
                    "target_action": other.current_action,
                    "same_colony": same_colony,
                    "relationship_trust": rel_trust,
                }
            else:
                # ── Display only — threat assessment caused one/both to back down ──
                # The weaker agent gets a fear/stress spike; no injury exchanged.
                loser = agent if my_conf < their_conf else other
                loser.drives.fear = min(1.0, loser.drives.fear + 0.08 * display_intensity)

                log_entry = {
                    "tick": world.tick, "year": world.tick // 360, "day": world.tick % 360,
                    "initiator_id": agent.id, "target_id": other.id,
                    "duplicate_eval": is_duplicate,
                    "encounter_reason": reason,
                    "fight_occurred": False, "display_only": True,
                    "display_intensity": display_intensity,
                    "confidence_components": my_comps,
                    "initiator_confidence": round(my_conf, 3),
                    "target_confidence": round(their_conf, 3),
                    "distance_between": round(dist_between, 2),
                    "dist_from_home": round(dist_from_home, 2),
                    "initiator_health": round(agent.health, 1),
                    "target_health": round(other.health, 1),
                    "initiator_injury": round(agent.injury_level, 1),
                    "target_injury": round(other.injury_level, 1),
                    "initiator_action": agent.current_action,
                    "target_action": other.current_action,
                    "same_colony": same_colony,
                    "relationship_trust": rel_trust,
                }

            if len(world.disputes_history) < 5000:
                world.disputes_history.append(log_entry)
            world._disputes_this_tick.add(pair_key)

    # Injury Trauma Health Drain
    if agent.injury_level >= 50.0:
        inj_drain = 0.5 * (agent.injury_level / 100.0)
        agent.health -= inj_drain
        agent.injury_damage_accumulated += inj_drain
        
    # Injury Healing
    if agent.injury_level > 0.0:
        if agent.hunger < 50.0 and agent.thirst < 50.0:
            healing_mult = getattr(world, "healing_speed_mult", 1.0)
            if agent.current_action == "Resting" and is_sheltered:
                agent.injury_level = max(0.0, agent.injury_level - 2.0 * healing_mult)
            elif agent.current_action == "Resting":
                agent.injury_level = max(0.0, agent.injury_level - 0.5 * healing_mult)
            elif np.random.uniform() < 0.05:
                agent.injury_level = min(100.0, agent.injury_level + 1.0)
                
    # --- 9. Health Restoration & Near-Miss Recording ---
    if not starving and not dehydrated and agent.injury_level < 50.0 and (local_temp >= -5.0 and local_temp <= 45.0):
        healing_mult = getattr(world, "healing_speed_mult", 1.0)
        agent.health = min(100.0, agent.health + 0.5 * healing_mult)
        
    if agent.health < 30.0:
        agent.add_memory(DANGER, (cy, cx), world.tick, importance=0.85)
        
    # --- 10. Death Handling & Cause Attribution ---
    if agent.health <= 0.0 or agent.age >= agent.max_age:
        agent.dead = True
        
        # Determine Primary and Secondary Causes of Death
        if agent.age >= agent.max_age:
            agent.primary_cause = "Old Age"
            dams = [
                ("Starvation", agent.starvation_damage_accumulated),
                ("Dehydration", agent.dehydration_damage_accumulated),
                ("Exposure", agent.exposure_damage_accumulated),
                ("Injury", agent.injury_damage_accumulated)
            ]
            dams.sort(key=lambda x: x[1], reverse=True)
            agent.secondary_cause = dams[0][0] if dams[0][1] > 0.0 else "Natural Decline"
        else:
            dams = [
                ("Starvation", agent.starvation_damage_accumulated),
                ("Dehydration", agent.dehydration_damage_accumulated),
                ("Exposure", agent.exposure_damage_accumulated),
                ("Injury", agent.injury_damage_accumulated)
            ]
            dams.sort(key=lambda x: x[1], reverse=True)
            agent.primary_cause = dams[0][0] if dams[0][1] > 0.0 else "Unknown"
            agent.secondary_cause = dams[1][0] if dams[1][1] > 0.0 else "None"
            
        agent.cause_of_death = agent.primary_cause
        
        # Increment total deaths
        world.total_deaths = getattr(world, "total_deaths", 0) + 1

        # Dehydration Deaths Telemetry Audit (Phase 10)
        if agent.primary_cause == "Dehydration" or agent.secondary_cause == "Dehydration":
            carrying = (agent.stored_water > 0.0)
            beside = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < world.height and 0 <= nx < world.width:
                        if (world.biome[ny, nx] == 8) or (world.lake_map[ny, nx] > 0.0) or (world.river_map[ny, nx] > 1500.0):
                            beside = True
                            break
                if beside:
                    break
            remembered = (len(agent.knowledge.water_sources) > 0)
            dry_visit = getattr(agent, "visited_dry_water_recently", False)
            world.telemetry.record_dehydration_death(carrying, beside, remembered, dry_visit)

            # Change 1c: Dump the per-tick death trace for post-mortem analysis.
            # Annotate the last trace entry with distance-to-target at death,
            # then store in world.dehydration_death_traces (capped at 20).
            death_trace = getattr(agent, "_death_trace", None)
            if death_trace:
                last_entry = death_trace[-1]
                if last_entry.get("target") is not None:
                    ty_dt, tx_dt = last_entry["target"]
                    last_entry["dist_to_target_at_death"] = round(
                        float(np.sqrt((ty_dt - cy)**2 + (tx_dt - cx)**2)), 1
                    )
                if not hasattr(world, "dehydration_death_traces"):
                    world.dehydration_death_traces = []
                world.dehydration_death_traces.append({
                    "agent_id":    int(agent.id),
                    "year":        int(world.tick // 360),
                    "tick":        int(world.tick),
                    "action_dist": {k: v for k, v in agent.action_counts.items() if v > 0},
                    "trace":       death_trace,
                })
                # Keep only the 20 most recent traces to bound memory use
                if len(world.dehydration_death_traces) > 20:
                    world.dehydration_death_traces.pop(0)

        
        world.history.append(
            f"Year {world.tick // 360}, Day {world.tick % 360}: Agent {agent.id} "
            f"died at coordinate ({cx}, {cy}). Primary: {agent.primary_cause}, Secondary: {agent.secondary_cause}."
        )

        if not hasattr(world, "events_timeline") or world.events_timeline is None:
            world.events_timeline = []

        colony_name = "Unknown"
        if hasattr(world, "colonies") and 0 <= agent.colony_id < len(world.colonies):
            colony_name = world.colonies[agent.colony_id].get("name", colony_name)

        grandchildren = 0
        for c_id in agent.children_ids:
            for a in world.agents:
                if a.id == c_id:
                    grandchildren += len(a.children_ids)
                    break

        world.events_timeline.append({
            "tick": int(world.tick),
            "year": int(world.tick // 360),
            "day": int(world.tick % 360),
            "type": "Death",
            "description": f"Agent #{agent.id} died at coordinate ({cx}, {cy}). Primary: {agent.primary_cause}. Secondary: {agent.secondary_cause}.",
            "metadata": {
                "agent_id": int(agent.id),
                "age": int(agent.age),
                "generation": int(agent.generation),
                "primary_cause": agent.primary_cause,
                "secondary_cause": agent.secondary_cause,
                "location": [int(cx), int(cy)],  # [x, y]
                "colony_id": int(agent.colony_id),
                "colony_name": colony_name,
                "children_count": int(len(agent.children_ids)),
                "grandchildren_count": int(grandchildren)
            }
        })
        
        # Disperse death density in a 15-pixel radius in the global heatmap
        h, w = world.height, world.width
        for dy in range(-15, 16):
            for dx in range(-15, 16):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    dist = np.sqrt(dy**2 + dx**2)
                    if dist <= 15.0:
                        world.death_density_map[ny, nx] += float(1.0 / (1.0 + dist))
        return
        
    # --- 11. Decays Spatial Knowledge Confidence ---
    # g_memory_fidelity controls the decay rate: calibrated so neutral genome = 0.001/tick
    decay = brain["memory_decay_rate"]
    if not getattr(world, "ecology_ablation", {}).get("memory_fidelity", True):
        # Fallback to legacy unboosted decay formula
        decay = float(0.0002 + agent.genome.genes[4] * 0.0016)
    for loc in list(agent.knowledge.water_sources.keys()):
        agent.knowledge.water_sources[loc]["confidence"] -= decay
        if agent.knowledge.water_sources[loc]["confidence"] <= 0:
            del agent.knowledge.water_sources[loc]
            agent.nodes_removed_count += 1

    for loc in list(agent.knowledge.food_sources.keys()):
        agent.knowledge.food_sources[loc]["confidence"] -= decay
        if agent.knowledge.food_sources[loc]["confidence"] <= 0:
            del agent.knowledge.food_sources[loc]
            agent.nodes_removed_count += 1

    for loc in list(agent.knowledge.danger_locations.keys()):
        agent.knowledge.danger_locations[loc] -= decay
        if agent.knowledge.danger_locations[loc] <= 0:
            del agent.knowledge.danger_locations[loc]

    # --- Lifetime Learning: Hebbian/Perceptron updates on environmental features ---
    if hasattr(agent, "feature_weights") and not agent.dead:
        health_lost = max(0.0, prev_health - agent.health)
        
        f_temp = (local_temp - 20.0) / 20.0
        f_elev = float(world.elevation[cy, cx])
        f_moisture = float(world.rainfall[cy, cx] / 3000.0)
        f_food = float(world.wildlife[cy, cx] + world.fertility[cy, cx])
        has_water = (world.biome[cy, cx] == 8) or (world.lake_map[cy, cx] > 0.0) or (world.river_map[cy, cx] > 1500.0)
        f_water = 1.0 if has_water else 0.0
        
        features = np.array([f_temp, f_elev, f_moisture, f_food, f_water], dtype=np.float32)
        lr = agent.learning_rate
        
        # 1. Negative outcome: Health loss (starvation, dehydration, exposure, injury)
        if health_lost > 0.0:
            agent.feature_weights -= lr * features * (health_lost / 10.0)
            
        # 2. Positive outcome: Water consumed
        if getattr(agent, "_consumed_water", False):
            agent.feature_weights += lr * features * 0.3
            agent._consumed_water = False
            
        # 3. Positive outcome: Food consumed
        if getattr(agent, "_consumed_food", False):
            agent.feature_weights += lr * features * 0.3
            agent._consumed_food = False
            
        # 4. Positive outcome: Safe sheltering / rest comfort
        if agent.current_action in ("Resting", "Sheltering") and health_lost == 0.0 and agent.injury_level < 20.0:
            agent.feature_weights += lr * features * 0.1
            
        # Keep weights bounded to prevent numerical divergence
        agent.feature_weights = np.clip(agent.feature_weights, -2.0, 2.0)

def calculate_danger_penalty(agent: Agent, loc: tuple, risk_mult: float = 1.25) -> float:
    """
    Computes the utility penalty multiplier based on proximity to known danger spots.
    `risk_mult` is derived from the agent's g_risk_sensitivity gene (0.5 – 2.0).
    High risk_mult → stronger avoidance; low → bolder crossing of danger zones.
    """
    if not loc or not agent.knowledge.danger_locations:
        return 1.0

    penalty = 1.0
    cy, cx = loc
    # Scale the avoidance strength by risk_mult (clamped so it never fully block)
    avoidance = min(0.90, 0.55 * risk_mult)
    for d_loc, conf in agent.knowledge.danger_locations.items():
        dy = d_loc[0] - cy
        dx = d_loc[1] - cx
        dist = np.sqrt(dy**2 + dx**2)
        if dist < 50.0:
            penalty *= (1.0 - (conf * avoidance) / (1.0 + dist / 15.0))
    return float(np.clip(penalty, 0.05, 1.0))

def select_shelter_site(agent: Agent, world) -> tuple:
    """Finds the optimal coordinate to construct a shelter based on proximity to resources, elevation, and danger."""
    cy, cx = agent.location
    best_score = -float('inf')
    best_site = agent.home_location
    
    # Candidate sites: their home location, their current location, and all known food/water sources
    candidates = {agent.home_location, agent.location}
    for loc in agent.knowledge.water_sources:
        candidates.add(loc)
    for loc in agent.knowledge.food_sources:
        candidates.add(loc)
        
    h, w = world.height, world.width
    for site in candidates:
        sy, sx = site
        # Score this site
        # 1. Proximity to water sources
        water_dist_score = 0.0
        if agent.knowledge.water_sources:
            min_w_dist = min(np.sqrt((loc[0]-sy)**2 + (loc[1]-sx)**2) for loc in agent.knowledge.water_sources)
            water_dist_score = 100.0 / (1.0 + min_w_dist / 20.0)
        
        # 2. Proximity to food sources
        food_dist_score = 0.0
        if agent.knowledge.food_sources:
            min_f_dist = min(np.sqrt((loc[0]-sy)**2 + (loc[1]-sx)**2) for loc in agent.knowledge.food_sources)
            food_dist_score = 100.0 / (1.0 + min_f_dist / 20.0)
            
        # 3. Elevation/Temperature factor (prefer moderate elevations, avoid freezing peaks >= 0.7)
        elev = float(world.elevation[sy, sx])
        elev_score = 100.0 * (1.0 - abs(elev - 0.4) / 0.6)  # peak at 0.4 elevation, penalize high peaks
        
        # 4. Danger avoidance (huge penalty if near known danger spots)
        danger_penalty = 1.0
        for d_loc, conf in agent.knowledge.danger_locations.items():
            d_dist = np.sqrt((d_loc[0]-sy)**2 + (d_loc[1]-sx)**2)
            if d_dist < 30.0:
                danger_penalty *= (1.0 - conf * 0.9 / (1.0 + d_dist / 10.0))
                
        # 5. Local resource density for building (wood/stone density)
        wood_dens = float(world.wood[sy, sx])
        stone_dens = float(world.stone[sy, sx])
        resource_score = (wood_dens + stone_dens) * 50.0
        
        # 6. Distance from spawn/activity center
        spawn_dist = np.sqrt((agent.spawn_location[0]-sy)**2 + (agent.spawn_location[1]-sx)**2)
        dist_penalty = 50.0 / (1.0 + spawn_dist / 100.0)
        
        score = (water_dist_score * 1.5 + food_dist_score * 1.2 + elev_score * 1.0 + resource_score * 0.8 + dist_penalty) * danger_penalty
        
        if score > best_score:
            best_score = score
            best_site = site
            
    return best_site

def evaluate_utility(agent: Agent, world, chunk_size: int = 32, context=None) -> tuple:
    """
    Evaluates utility values for Drink, Eat, Rest, Explore, Build Shelter, Shelter,
    and all Phase 5 resource-sharing, caching, and pouch/chest actions.
    Returns (action_name, target_coordinate).
    """
    if agent.dead:
        return "Dead", None
        
    agent.decision_evals += 1
    cy, cx = agent.location
    
    if context is not None:
        assert context.context_tick == world.tick
        day = context.day
        season = context.season
        cos_factor = context.cos_factor
        local_temp = context.local_temp
    else:
        # Get seasonal target for prediction
        day = world.tick % 360
        season = day // 90
        cos_factor = getattr(world, "temp_cos_factor", None)
        if cos_factor is None:
            cos_factor = np.cos(((day - 180.0) / 180.0) * np.pi)
        
        # Calculate current local temperature to assess environmental hazards
        biome_id = int(world.biome[cy, cx])
        amplitude = {6: 22.0, 1: 18.0, 2: 18.0, 3: 18.0,
                     4: 12.0, 5: 12.0, 0: 6.0, 8: 6.0, 7: 4.0}.get(biome_id, 12.0)
        base_temp = float(world.temperature[cy, cx])
        temp_seasonal_offset = amplitude * cos_factor
        global_offset = getattr(world, "global_temp_offset", 0.0)
        local_temp = base_temp + temp_seasonal_offset + global_offset
        
    is_transition = (day % 90) >= 45
    target_season = (season + 1) % 4 if is_transition else season
    
    r = agent.senescence_factor
    brain = agent.brain
    risk_mult = brain["risk_mult"]
    cache = agent.memory_cache

    # Calculate scarcity prediction based on seasonal wet/dry forecasts in memory
    total_active = 0
    total_obs = 0
    agent.memories_searched += len(agent.knowledge.water_sources) + len(agent.knowledge.food_sources)
    for node in list(agent.knowledge.water_sources.values()) + list(agent.knowledge.food_sources.values()):
        active_count = node.get("active_seasons", {}).get(target_season, 0)
        dry_count = node.get("dry_seasons", {}).get(target_season, 0)
        total_active += active_count
        total_obs += (active_count + dry_count)
    
    if total_obs > 0:
        abundance_ratio = total_active / total_obs
    else:
        abundance_ratio = 0.8
        
    scarcity_prediction = 1.0 - abundance_ratio
    # Modulate forecast trust by prediction confidence (skepticism)
    scarcity_prediction = 0.2 + (scarcity_prediction - 0.2) * agent.prediction_confidence
    
    # Retrieve colony info
    c_id = getattr(agent, "colony_id", -1)
    colony_food = 0.0
    colony_water = 0.0
    if c_id >= 0 and c_id < len(world.colonies):
        colony_food = world.colonies[c_id].get("stored_food", 0.0)
        colony_water = world.colonies[c_id].get("stored_water", 0.0)

    # Common home/shelter targets
    shelter_target = agent.shelter_location if agent.shelter_location is not None else agent.home_location

    # Pre-calculate DecisionContext sigmoids & variables
    from .cognitive import DecisionContext
    d_ctx = DecisionContext(
        risk_mult=risk_mult,
        scarcity_prediction=scarcity_prediction,
        colony_food=colony_food,
        colony_water=colony_water,
        shelter_target=shelter_target,
        rest_sig=sigmoid_utility(100.0 - agent.energy, 55.0, 12.0),
        drink_sig=sigmoid_utility(agent.thirst, 50.0, 10.0),
        eat_sig=sigmoid_utility(agent.hunger, 50.0, 10.0),
        drink_stored_sig=sigmoid_utility(agent.thirst, 40.0, 10.0),
        eat_stored_sig=sigmoid_utility(agent.hunger, 40.0, 10.0),
        withdraw_food_sig=sigmoid_utility(agent.hunger, 45.0, 10.0),
        withdraw_water_sig=sigmoid_utility(agent.thirst, 45.0, 10.0)
    )

    # --- 1. Rest Utility ---
    # High fatigue creates a strong urge to return home to sleep
    rest_utility = d_ctx.rest_sig * 1.3
    if agent.energy < 15.0:
        rest_target = (cy, cx)
    else:
        rest_target = d_ctx.shelter_target
            
    # --- 2. Drink Utility (Predictive Selection & Danger Avoidance) ---
    drink_utility = 0.0
    drink_target = None
    reactive_drink_target = None
    if len(agent.knowledge.water_sources) > 0:
        base_drink = d_ctx.drink_sig * 1.6
        
        best_reactive_dist_sq = float('inf')
        agent.memories_searched += len(agent.knowledge.water_sources)
        for loc in agent.knowledge.water_sources:
            dist_sq = (loc[0] - cy)**2 + (loc[1] - cx)**2
            if dist_sq < best_reactive_dist_sq:
                best_reactive_dist_sq = dist_sq
                reactive_drink_target = loc
        best_reactive_dist = float(np.sqrt(best_reactive_dist_sq)) if reactive_drink_target is not None else float('inf')
                 
        cache_water_valid = (
            cache.get("last_update_tick", -1) != -1 and
            agent.discoveries_count == cache.get("last_discoveries", 0) and
            target_season == cache.get("last_season", -1) and
            "water_spots" in cache and len(cache["water_spots"]) > 0
        )
        if cache_water_valid:
            agent.memory_cache_hits += 1
        else:
            agent.memory_cache_misses += 1
        
        if not cache_water_valid:
            water_candidates = []
            for loc, node in agent.knowledge.water_sources.items():
                dist = np.sqrt((loc[0] - cy)**2 + (loc[1] - cx)**2)
                dist_mult = 1.0 / (1.0 + dist / 200.0)
                
                active_count = node.get("active_seasons", {}).get(target_season, 0)
                dry_count = node.get("dry_seasons", {}).get(target_season, 0)
                total_obs_node = active_count + dry_count
                seasonal_mult = active_count / total_obs_node if total_obs_node > 0 else 0.8
                eff_seasonal_mult = 1.0 + (seasonal_mult - 1.0) * agent.prediction_confidence
                
                raw_conf = node["confidence"]
                eff_conf = (raw_conf ** (1.0 + 2.0 * r)) * eff_seasonal_mult
                
                score = eff_conf * dist_mult
                water_candidates.append((score, loc))
                
            water_candidates.sort(key=lambda x: x[0], reverse=True)
            cache["water_spots"] = [x[1] for x in water_candidates[:3]]
            cache["last_update_tick"] = world.tick
            cache["last_discoveries"] = agent.discoveries_count
            cache["last_season"] = target_season
            
        best_drink_util = -1.0
        # Evaluate only the cached top water spots
        for loc in cache["water_spots"]:
            if loc not in agent.knowledge.water_sources:
                continue
            node = agent.knowledge.water_sources[loc]
            agent.targets_evaluated += 1
            agent.memories_searched += 1
            dist = np.sqrt((loc[0] - cy)**2 + (loc[1] - cx)**2)
            dist_mult = 1.0 / (1.0 + dist / 200.0)
            
            if agent.prediction_enabled:
                active_count = node.get("active_seasons", {}).get(target_season, 0)
                dry_count = node.get("dry_seasons", {}).get(target_season, 0)
                total_obs_node = active_count + dry_count
                seasonal_mult = active_count / total_obs_node if total_obs_node > 0 else 0.8
            else:
                seasonal_mult = 1.0
                
            eff_seasonal_mult = 1.0 + (seasonal_mult - 1.0) * agent.prediction_confidence
            raw_conf = node["confidence"]
            if raw_conf < 1.0 and hasattr(agent, "matches_concept") and agent.matches_concept(WATER, loc[0], loc[1], world):
                raw_conf = raw_conf + 0.25 * (1.0 - raw_conf)
            eff_conf = (raw_conf ** (1.0 + 2.0 * r)) * eff_seasonal_mult
            danger_penalty = calculate_danger_penalty(agent, loc, risk_mult)
            env_mod = get_environmental_modulation(agent, loc[0], loc[1], world)

            node_util = base_drink * eff_conf * dist_mult * danger_penalty * env_mod
            if node_util > best_drink_util:
                best_drink_util = node_util
                drink_target = loc
        drink_utility = best_drink_util
                
    # --- 3. Eat Utility (Predictive Selection & Danger Avoidance) ---
    eat_utility = 0.0
    eat_target = None
    reactive_eat_target = None
    if len(agent.knowledge.food_sources) > 0:
        base_eat = d_ctx.eat_sig * 1.2
        
        best_reactive_dist_sq = float('inf')
        agent.memories_searched += len(agent.knowledge.food_sources)
        for loc in agent.knowledge.food_sources:
            dist_sq = (loc[0] - cy)**2 + (loc[1] - cx)**2
            if dist_sq < best_reactive_dist_sq:
                best_reactive_dist_sq = dist_sq
                reactive_eat_target = loc
        best_reactive_dist = float(np.sqrt(best_reactive_dist_sq)) if reactive_eat_target is not None else float('inf')
                
        cache_food_valid = (
            cache.get("last_update_tick", -1) != -1 and
            agent.discoveries_count == cache.get("last_discoveries", 0) and
            target_season == cache.get("last_season", -1) and
            "food_spots" in cache and len(cache["food_spots"]) > 0
        )
        if cache_food_valid:
            agent.memory_cache_hits += 1
        else:
            agent.memory_cache_misses += 1
        
        if not cache_food_valid:
            food_candidates = []
            for loc, node in agent.knowledge.food_sources.items():
                dist = np.sqrt((loc[0] - cy)**2 + (loc[1] - cx)**2)
                dist_mult = 1.0 / (1.0 + dist / 200.0)
                
                active_count = node.get("active_seasons", {}).get(target_season, 0)
                dry_count = node.get("dry_seasons", {}).get(target_season, 0)
                total_obs_node = active_count + dry_count
                seasonal_mult = active_count / total_obs_node if total_obs_node > 0 else 0.8
                eff_seasonal_mult = 1.0 + (seasonal_mult - 1.0) * agent.prediction_confidence
                
                raw_conf = node["confidence"]
                eff_conf = (raw_conf ** (1.0 + 2.0 * r)) * eff_seasonal_mult
                
                score = eff_conf * dist_mult
                food_candidates.append((score, loc))
                
            food_candidates.sort(key=lambda x: x[0], reverse=True)
            cache["food_spots"] = [x[1] for x in food_candidates[:3]]
            cache["last_update_tick"] = world.tick
            cache["last_discoveries"] = agent.discoveries_count
            cache["last_season"] = target_season
            
        best_eat_util = -1.0
        # Evaluate only the cached top food spots
        for loc in cache["food_spots"]:
            if loc not in agent.knowledge.food_sources:
                continue
            node = agent.knowledge.food_sources[loc]
            agent.targets_evaluated += 1
            agent.memories_searched += 1
            dist = np.sqrt((loc[0] - cy)**2 + (loc[1] - cx)**2)
            dist_mult = 1.0 / (1.0 + dist / 200.0)
            
            if agent.prediction_enabled:
                active_count = node.get("active_seasons", {}).get(target_season, 0)
                dry_count = node.get("dry_seasons", {}).get(target_season, 0)
                total_obs_node = active_count + dry_count
                seasonal_mult = active_count / total_obs_node if total_obs_node > 0 else 0.8
            else:
                seasonal_mult = 1.0
                
            eff_seasonal_mult = 1.0 + (seasonal_mult - 1.0) * agent.prediction_confidence
            raw_conf = node["confidence"]
            if raw_conf < 1.0 and hasattr(agent, "matches_concept") and agent.matches_concept(FOOD, loc[0], loc[1], world):
                raw_conf = raw_conf + 0.25 * (1.0 - raw_conf)
            eff_conf = (raw_conf ** (1.0 + 2.0 * r)) * eff_seasonal_mult
            danger_penalty = calculate_danger_penalty(agent, loc, risk_mult)
            env_mod = get_environmental_modulation(agent, loc[0], loc[1], world)

            node_util = base_eat * eff_conf * dist_mult * danger_penalty * env_mod
            if node_util > best_eat_util:
                best_eat_util = node_util
                eat_target = loc
        eat_utility = best_eat_util

    # --- 4. Explore Utility (Danger Avoided) ---
    chunk_y, chunk_x = cy // chunk_size, cx // chunk_size
    unexplored_chunks = []
    for dy_c in (-1, 0, 1):
        for dx_c in (-1, 0, 1):
            ny_chunk, nx_chunk = chunk_y + dy_c, chunk_x + dx_c
            if 0 <= ny_chunk < (world.height // chunk_size) and 0 <= nx_chunk < (world.width // chunk_size):
                if (ny_chunk, nx_chunk) not in agent.visited_chunks:
                    unexplored_chunks.append((ny_chunk, nx_chunk))

    unexplored_ratio = len(unexplored_chunks) / 9.0
    explore_utility = (
        agent.curiosity_need
        * brain["novelty_weight"]
        * (1.0 / max(risk_mult, 0.5))
        * (unexplored_ratio * 4.0 + 0.1)
    )

    if len(unexplored_chunks) > 0:
        target_chunk = unexplored_chunks[np.random.randint(len(unexplored_chunks))]
        pref_type = WATER if agent.thirst > agent.hunger else FOOD
        chunk_y_start = target_chunk[0] * chunk_size
        chunk_x_start = target_chunk[1] * chunk_size
        chosen_target = None
        for _ in range(10):
            ry = int(np.random.randint(0, chunk_size))
            rx = int(np.random.randint(0, chunk_size))
            cand_y = int(np.clip(chunk_y_start + ry, 0, world.height - 1))
            cand_x = int(np.clip(chunk_x_start + rx, 0, world.width - 1))
            if hasattr(agent, "matches_concept") and agent.matches_concept(pref_type, cand_y, cand_x, world):
                chosen_target = (cand_y, cand_x)
                break
        if chosen_target is not None:
            explore_target = chosen_target
        else:
            explore_target = (
                target_chunk[0] * chunk_size + chunk_size // 2,
                target_chunk[1] * chunk_size + chunk_size // 2,
            )
    else:
        pref_type = WATER if agent.thirst > agent.hunger else FOOD
        chosen_target = None
        for _ in range(15):
            ry = int(np.random.randint(-30, 31))
            rx = int(np.random.randint(-30, 31))
            cand_y = int(np.clip(cy + ry, 0, world.height - 1))
            cand_x = int(np.clip(cx + rx, 0, world.width - 1))
            if hasattr(agent, "matches_concept") and agent.matches_concept(pref_type, cand_y, cand_x, world):
                chosen_target = (cand_y, cand_x)
                break
        if chosen_target is not None:
            explore_target = chosen_target
        else:
            ry = int(np.random.randint(-30, 31))
            rx = int(np.random.randint(-30, 31))
            explore_target = (
                int(np.clip(cy + ry, 0, world.height - 1)),
                int(np.clip(cx + rx, 0, world.width - 1)),
            )
    explore_utility *= calculate_danger_penalty(agent, explore_target, risk_mult)
    explore_utility *= get_environmental_modulation(agent, explore_target[0], explore_target[1], world)
    
    # --- 5. Build Shelter Utility (Optimization & Urgency) ---
    build_utility = 0.0
    build_target = None
    if agent.shelter_location is None:
        # Scan registry for unoccupied shelters
        unoccupied = []
        if hasattr(world, "shelters"):
            for loc, sh in world.shelters.items():
                if sh["owner_id"] is None:
                    dist = np.sqrt((loc[0] - cy)**2 + (loc[1] - cx)**2)
                    if dist < getattr(world, "shelter_search_dist", 100.0):
                        unoccupied.append((dist, loc, sh))
        if unoccupied:
            unoccupied.sort(key=lambda x: x[0])
            dist, loc, sh = unoccupied[0]
            sh["owner_id"] = agent.id
            agent.shelter_location = loc
            agent.shelter_level = sh["level"]
            agent.shelter_durability = sh["durability"]
            world.history.append(
                f"Year {world.tick // 360}, Day {world.tick % 360}: Agent {agent.id} "
                f"discovered and claimed an abandoned shelter at coordinate ({loc[1]}, {loc[0]})."
            )
        else:
            # Build a new one
            loc = select_shelter_site(agent, world)
            agent.shelter_location = loc
            agent.shelter_level = 1
            agent.shelter_durability = 30.0
            if not hasattr(world, "shelters"):
                world.shelters = {}
            world.shelters[loc] = {
                "level": 1,
                "durability": 30.0,
                "owner_id": agent.id
            }
        
    build_target = agent.shelter_location
    build_utility = (100.0 - agent.shelter_durability) * 0.6
    if local_temp < 8.0 or local_temp > 32.0:
        build_utility *= 1.5
    if agent.shelter_durability < 30.0:
        build_utility += 15.0
    build_utility = np.clip(build_utility, 0.0, 90.0)
    
    # --- 6. Sheltering Utility (Extreme Weather Safety) ---
    shelter_utility = 0.0
    if agent.shelter_location is not None:
        if local_temp < 8.0 or local_temp > 32.0:
            severity = max(8.0 - local_temp, local_temp - 32.0)
            shelter_utility = severity * 2.5 * (agent.shelter_durability / 100.0)
            shelter_utility += agent.injury_level * 0.4
        else:
            shelter_utility = 0.0
        shelter_utility *= get_environmental_modulation(agent, shelter_target[0], shelter_target[1], world)
            
    # --- 7. Phase 5 Capability Actions ---
    life_stage = agent.life_stage

    # --- 7a. Reproduce (biological gating only — no genome weight emergence check) ---
    reproduce_utility = 0.0
    reproduce_target  = None
    
    from .reproduction import MIN_REPRO_AGE_TICKS, MAX_REPRO_AGE_TICKS, MIN_HEALTH, MIN_FAT_RESERVES, MAX_HUNGER_TO_REPRO, MAX_THIRST_TO_REPRO, MIN_SHELTER_DURA, MAX_INJURY_TO_REPRO, MATE_SEARCH_RADIUS, is_eligible_to_reproduce
    
    is_fertile_age = (MIN_REPRO_AGE_TICKS <= agent.age < MAX_REPRO_AGE_TICKS)
    if is_fertile_age:
        world.telemetry.repro_potential_opportunities += 1
        
        # Check all biological constraints in parallel
        bio_failed = False
        terminal_reason = None
        
        if agent.reproduction_cooldown > 0:
            bio_failed = True
            terminal_reason = "cooldown"
            world.telemetry.repro_fail_cooldown += 1
            
        if agent.health < MIN_HEALTH:
            bio_failed = True
            if terminal_reason is None: terminal_reason = "health"
            world.telemetry.repro_fail_health += 1
            
        if agent.fat_reserves < MIN_FAT_RESERVES:
            bio_failed = True
            if terminal_reason is None: terminal_reason = "fat"
            world.telemetry.repro_fail_fat += 1
            
        if agent.hunger > MAX_HUNGER_TO_REPRO:
            bio_failed = True
            if terminal_reason is None: terminal_reason = "hunger"
            world.telemetry.repro_fail_hunger += 1
            
        if agent.thirst > MAX_THIRST_TO_REPRO:
            bio_failed = True
            if terminal_reason is None: terminal_reason = "thirst"
            world.telemetry.repro_fail_thirst += 1
            
        if agent.shelter_durability < MIN_SHELTER_DURA:
            bio_failed = True
            if terminal_reason is None: terminal_reason = "shelter"
            world.telemetry.repro_fail_shelter += 1
            
        if agent.injury_level > MAX_INJURY_TO_REPRO:
            bio_failed = True
            if terminal_reason is None: terminal_reason = "injury"
            world.telemetry.repro_fail_injury += 1
            
        if bio_failed:
            agent.repro_terminal_outcome = terminal_reason
        else:
            # Stage 2: Biologically Eligible
            world.telemetry.repro_actual_opportunities += 1
            
            # Check if a mate exists in the world
            mate_exists_world = any(a.id != agent.id and not a.dead and getattr(a, "colony_id", -1) == getattr(agent, "colony_id", -1) for a in world.agents)
            if mate_exists_world:
                world.telemetry.repro_mate_exists_world_ticks += 1
                
            # Query candidates in search radius (30)
            if context is not None and hasattr(context, "nearby_agents"):
                candidates = []
                for other in context.nearby_agents:
                    oy, ox = other.location
                    dist_sq = (oy - cy)**2 + (ox - cx)**2
                    if dist_sq <= MATE_SEARCH_RADIUS ** 2:
                        candidates.append(other)
            else:
                candidates = world.query_agents(agent.location, MATE_SEARCH_RADIUS, alive_only=True)
                
            partners_in_radius = [other for other in candidates if other.id != agent.id and getattr(other, "colony_id", -1) == getattr(agent, "colony_id", -1)]
            
            if not mate_exists_world:
                agent.repro_terminal_outcome = "no_mate"
                world.telemetry.repro_fail_no_mate += 1
            elif not partners_in_radius:
                agent.repro_terminal_outcome = "distance"
                world.telemetry.repro_fail_distance += 1
            else:
                # Stage 4: Mate in Search Radius
                world.telemetry.repro_mate_exists_radius_ticks += 1
                
                # Check relationship trust
                rel_ok = any(agent.get_social_modifier(other.id, world.tick) >= 0.6 for other in partners_in_radius)
                if rel_ok:
                    world.telemetry.repro_relationship_ok_ticks += 1
                else:
                    world.telemetry.repro_fail_relationship += 1
                    
                # Check mate eligibility
                mate_elig = any(is_eligible_to_reproduce(other) for other in partners_in_radius)
                if mate_elig:
                    world.telemetry.repro_mate_eligible_ticks += 1
                else:
                    world.telemetry.repro_fail_mate_ineligible += 1
                    
                # Find if we have any matching eligible mate with acceptable relationship
                best_mate = None
                for other in partners_in_radius:
                    social_mod = agent.get_social_modifier(other.id, world.tick)
                    if social_mod >= 0.6:
                        if is_eligible_to_reproduce(other):
                            best_mate = other
                            break
                            
                if not rel_ok:
                    agent.repro_terminal_outcome = "relationship"
                elif not mate_elig:
                    agent.repro_terminal_outcome = "mate_ineligible"
                elif best_mate is None:
                    agent.repro_terminal_outcome = "relationship"
                else:
                    # Eligible partner found with good relationship!
                    social_mod = agent.get_social_modifier(best_mate.id, world.tick)
                    if life_stage == "Adult" and getattr(world, "reproduction_enabled", True):
                        reproduce_utility = 30.0 * social_mod
                        reproduce_target = best_mate.location

    # --- 7b. Store Food (g_planning drives food-caching behaviour) ---
    store_food_utility = 0.0
    store_food_target  = None
    if life_stage not in ("Infant", "Juvenile") and agent.hunger < 30.0 and agent.stored_food < 100.0:
        if agent.knowledge.food_sources:
            best_fd = min(agent.knowledge.food_sources,
                          key=lambda loc: (loc[0]-cy)**2 + (loc[1]-cx)**2)
            store_food_utility = brain["planning_horizon"] * (100.0 - agent.stored_food) * 0.4
            store_food_target  = best_fd

    # --- 7c. Store Water (g_planning drives water-caching behaviour) ---
    store_water_utility = 0.0
    store_water_target  = None
    if life_stage not in ("Infant", "Juvenile") and agent.thirst < 30.0 and agent.stored_water < 100.0:
        if agent.knowledge.water_sources:
            best_ws = min(agent.knowledge.water_sources,
                          key=lambda loc: (loc[0]-cy)**2 + (loc[1]-cx)**2)
            caching_mult = 0.45 if getattr(world, "ecology_ablation", {}).get("water_caching", True) else 0.3
            store_water_utility = brain["planning_horizon"] * (100.0 - agent.stored_water) * caching_mult
            store_water_target  = best_ws

    # Pre-query candidates for sharing (to avoid redundant filtering/querying if both share loops execute)
    share_candidates = None
    if agent.stored_food > 20.0 or agent.stored_water > 20.0:
        if brain["sharing_weight"] > 0.05:
            if context is not None and hasattr(context, "nearby_agents"):
                share_candidates = []
                vision_radius_sq = agent.vision_radius ** 2
                for other in context.nearby_agents:
                    oy, ox = other.location
                    dist_sq = (oy - cy)**2 + (ox - cx)**2
                    if dist_sq <= vision_radius_sq:
                        share_candidates.append(other)
            else:
                share_candidates = world.query_agents(agent.location, agent.vision_radius, alive_only=True)

    # --- 7d. Share Food (g_resource_sharing; requires stored food, hungry colony member, social memory modulated) ---
    share_food_utility = 0.0
    share_food_target  = None
    if agent.stored_food > 20.0 and brain["sharing_weight"] > 0.05 and share_candidates is not None:
        for other in share_candidates:
            if other.id != agent.id and getattr(other, "colony_id", -1) == c_id:
                if other.hunger > 70.0:
                    social_mod = agent.get_social_modifier(other.id, world.tick)
                    util = brain["sharing_weight"] * (other.hunger - 70.0) * 0.3 * social_mod
                    if util > share_food_utility:
                        share_food_utility = util
                        share_food_target  = other.location

    # --- 7e. Share Water (g_resource_sharing; requires stored water, thirsty colony member, social memory modulated) ---
    share_water_utility = 0.0
    share_water_target  = None
    if agent.stored_water > 20.0 and brain["sharing_weight"] > 0.05 and share_candidates is not None:
        for other in share_candidates:
            if other.id != agent.id and getattr(other, "colony_id", -1) == c_id:
                if other.thirst > 70.0:
                    social_mod = agent.get_social_modifier(other.id, world.tick)
                    util = brain["sharing_weight"] * (other.thirst - 70.0) * 0.3 * social_mod
                    if util > share_water_utility:
                        share_water_utility = util
                        share_water_target  = other.location

    # --- 8. Decision-Driven Pouch & Chest Actions ---
    # 8a. Drink Stored Water
    drink_stored_utility = d_ctx.drink_stored_sig * (1.5 if agent.stored_water > 0.0 else 0.0)
    drink_stored_target = (cy, cx)

    # 8b. Eat Stored Food
    eat_stored_utility = d_ctx.eat_stored_sig * (1.5 if agent.stored_food > 0.0 else 0.0)
    eat_stored_target = (cy, cx)

    # 8c. Deposit Food (Put food in colony chest)
    deposit_food_utility = (brain["planning_horizon"] * agent.stored_food * 0.5 * (1.0 - d_ctx.scarcity_prediction)) if (d_ctx.shelter_target is not None and agent.stored_food > 0.0) else 0.0
    deposit_food_target = d_ctx.shelter_target

    # 8d. Deposit Water (Put water in colony chest)
    if getattr(world, "ecology_ablation", {}).get("deposit_utility_fix", True):
        deposit_water_utility = (brain["planning_horizon"] * agent.stored_water * 0.7) if (d_ctx.shelter_target is not None and agent.stored_water > 0.0) else 0.0
    else:
        # Fallback to legacy un-fixed formula containing the drought scarcity penalty
        deposit_water_utility = (brain["planning_horizon"] * agent.stored_water * 0.5 * (1.0 - d_ctx.scarcity_prediction)) if (d_ctx.shelter_target is not None and agent.stored_water > 0.0) else 0.0
    deposit_water_target = d_ctx.shelter_target

    # 8e. Withdraw Food (Eat from colony chest)
    withdraw_food_utility = 0.0
    withdraw_food_target = d_ctx.shelter_target
    if d_ctx.shelter_target is not None and d_ctx.colony_food > 0.0:
        dist = np.sqrt((withdraw_food_target[0] - cy)**2 + (withdraw_food_target[1] - cx)**2)
        dist_mult = 1.0 / (1.0 + dist / 200.0)
        withdraw_food_utility = d_ctx.withdraw_food_sig * 1.8 * dist_mult

    # 8f. Withdraw Water (Drink from colony chest)
    withdraw_water_utility = 0.0
    withdraw_water_target = d_ctx.shelter_target
    if d_ctx.shelter_target is not None and d_ctx.colony_water > 0.0:
        dist = np.sqrt((withdraw_water_target[0] - cy)**2 + (withdraw_water_target[1] - cx)**2)
        dist_mult = 1.0 / (1.0 + dist / 200.0)
        withdraw_water_utility = d_ctx.withdraw_water_sig * 1.8 * dist_mult

    # --- 9. Final Action Selection ---
    # Phase 8.1: Compute drive-based utility multipliers.  These are pure
    # multiplicative coefficients [0.10, 3.0] derived from biological drive
    # tensions.  They shift action priorities without overriding the utility
    # logic.  Each action's base_util is multiplied before the predictor step.
    drive_mods = compute_drive_modulation(agent)

    actions = [
        ("Resting",            rest_utility,          rest_target),
        ("Drinking",           drink_utility,         drink_target),
        ("Eating",             eat_utility,           eat_target),
        ("Exploring",          explore_utility,       explore_target),
        ("Building Shelter",   build_utility,         build_target),
        ("Sheltering",         shelter_utility,       shelter_target),
        ("Reproduce",          reproduce_utility,     reproduce_target),
        ("Store Food",         store_food_utility,    store_food_target),
        ("Store Water",        store_water_utility,   store_water_target),
        ("Share Food",         share_food_utility,    share_food_target),
        ("Share Water",        share_water_utility,   share_water_target),
        ("Drink Stored Water", drink_stored_utility,  drink_stored_target),
        ("Eat Stored Food",    eat_stored_utility,     eat_stored_target),
        ("Deposit Food",       deposit_food_utility,  deposit_food_target),
        ("Deposit Water",      deposit_water_utility, deposit_water_target),
        ("Withdraw Food",      withdraw_food_utility, withdraw_food_target),
        ("Withdraw Water",     withdraw_water_utility, withdraw_water_target),
    ]

    # Apply procedural memory biases based on recent action history
    proc_biases = {a[0]: 0.0 for a in actions}
    if hasattr(agent, "action_history") and agent.action_history:
        for p in getattr(agent, "procedures", []):
            if p.trigger_season == season:
                if len(agent.action_history) == 1:
                    if p.action_sequence[0] == agent.action_history[0] and len(p.action_sequence) > 1:
                        next_act = p.action_sequence[1]
                        if next_act in proc_biases:
                            proc_biases[next_act] += 2.0 * p.confidence
                elif len(agent.action_history) >= 2:
                    last_two = tuple(agent.action_history[-2:])
                    if len(p.action_sequence) > 2 and p.action_sequence[:2] == last_two:
                        next_act = p.action_sequence[2]
                        if next_act in proc_biases:
                            proc_biases[next_act] += 3.0 * p.confidence
                    if p.action_sequence[0] == agent.action_history[-1] and len(p.action_sequence) > 1:
                        next_act = p.action_sequence[1]
                        if next_act in proc_biases:
                            proc_biases[next_act] += 1.0 * p.confidence

    # Apply drive multipliers, predictor, and procedural memory biases
    # Phase 8.1: Each base utility is multiplied by its biological drive
    # coefficient FIRST.  This shifts priorities without overriding them.
    # Then procedural and predictor biases are added additively on top.
    biased_actions = []
    for action_name, base_util, target_coord in actions:
        p_bias = proc_biases.get(action_name, 0.0)

        # Phase 8.1 drive modulation (multiplier on base utility)
        drive_mult = drive_mods.get(action_name, 1.0)
        driven_util = base_util * drive_mult

        pred_val = 0.0
        agent.targets_evaluated += 1
        if target_coord is not None and hasattr(agent, "predictor"):
            agent.predictor_calls += 1
            eff_temp = local_temp if target_coord == (cy, cx) else None
            context_vec = get_predictor_context(agent, action_name, target_coord, world, local_temp=eff_temp, scarcity_prediction=scarcity_prediction)
            pred_val = float(np.clip(agent.predictor.predict(context_vec), -3.0, 3.0))
            
        # Metacognitive self-trust modulation: scale predictor bias by historical accuracy
        pred_trust = agent.prediction_successes / agent.prediction_attempts if agent.prediction_attempts > 0 else 1.0
        final_util = driven_util + p_bias + pred_val * pred_trust
        biased_actions.append((action_name, final_util, target_coord, base_util))

    biased_actions.sort(key=lambda a: a[1], reverse=True)

    # --- Change 1b: Capture utility winner and runner-up for death trace telemetry ---
    if len(biased_actions) >= 1:
        agent._last_utility_winner   = (biased_actions[0][0], round(biased_actions[0][1], 1))
    if len(biased_actions) >= 2:
        agent._last_utility_runnerup = (biased_actions[1][0], round(biased_actions[1][1], 1))

    # --- Change 2: Resting lock — interruptible by survival drive tensions ---
    # Uses smoothed drive tensions rather than raw physiology values to stay
    # consistent with the drive architecture throughout the rest of the system.
    # Injury does NOT break rest; only thirst/hunger survival emergencies do.
    if agent.current_action == "Resting" and agent.energy < 100.0:
        ds = agent.drives
        survival_emergency = (
            ds.thirst_tension > 0.9 or
            ds.hunger_tension > 0.9
        )
        if not survival_emergency:
            if not hasattr(agent, "action_queue"):
                agent.action_queue = []
            agent.action_queue.clear()
            agent.action_queue.append(("Resting", rest_target))
            return "Resting", rest_target
        # Fall through to normal utility selection for survival emergencies
        
    chosen = biased_actions[0]
    chosen_action = chosen[0]
    chosen_target = chosen[2]
    
    # Save the chosen action's context vector for learning in next ticks
    if chosen_target is not None and hasattr(agent, "predictor"):
        eff_temp = local_temp if chosen_target == (cy, cx) else None
        agent.last_prediction_input = get_predictor_context(agent, chosen_action, chosen_target, world, local_temp=eff_temp, scarcity_prediction=scarcity_prediction)
        agent.last_prediction_value = chosen[1] - chosen[3]  # store bias
    
    if agent.prediction_enabled and is_transition and chosen_action in ("Drinking", "Eating") and chosen_target is not None:
        reactive_target = reactive_drink_target if chosen_action == "Drinking" else reactive_eat_target
        
        if chosen_target != reactive_target:
            agent.prediction_decisions += 1
            
        has_history = False
        if chosen_action == "Drinking" and chosen_target in agent.knowledge.water_sources:
            node = agent.knowledge.water_sources[chosen_target]
            has_history = (node.get("active_seasons", {}).get(target_season, 0) > 0 or 
                           node.get("dry_seasons", {}).get(target_season, 0) > 0)
        elif chosen_action == "Eating" and chosen_target in agent.knowledge.food_sources:
            node = agent.knowledge.food_sources[chosen_target]
            has_history = (node.get("active_seasons", {}).get(target_season, 0) > 0 or 
                           node.get("dry_seasons", {}).get(target_season, 0) > 0)
                           
        if has_history:
            agent.prediction_attempts += 1
            agent.predicted_destination = (chosen_target, target_season, reactive_target)
            
    if chosen_action != agent.current_action or chosen_target != agent.target_coordinate:
        agent.meaningful_decisions += 1
    if chosen_action != agent.current_action:
        agent.action_changes += 1
        
    # Phase 7: Clear and populate hierarchical action queue
    if not hasattr(agent, "action_queue"):
        agent.action_queue = []
    agent.action_queue.clear()
    agent.action_queue.append((chosen_action, chosen_target))
    
    # Store plan commitment based on chosen action's utility
    agent.plan_commitment = float(np.clip(chosen[1], 15.0, 95.0))
    
    # Add secondary/tertiary high-utility distinct actions to queue
    for act_name, act_util, act_targ, _ in biased_actions:
        if act_name == chosen_action:
            continue
        if act_util > 35.0 and act_targ is not None:
            if not any(q[0] == act_name for q in agent.action_queue):
                agent.action_queue.append((act_name, act_targ))
                
    # Record utility decision in telemetry (Phase 10)
    world.telemetry.record_utility_decision(
        agent.age,
        reproduce_utility,
        chosen_action,
        chosen[1]  # winning utility
    )
    
    if is_fertile_age:
        if getattr(agent, "repro_terminal_outcome", None) is None:
            if chosen_action == "Reproduce":
                world.telemetry.repro_wanted_reproduction_ticks += 1
                agent.repro_terminal_outcome = "mate_unwilling"
            else:
                agent.repro_terminal_outcome = "low_utility"

    # Water searches count start
    if chosen_action == "Drinking" and agent.current_action != "Drinking":
        agent.water_search_start_tick = world.tick
        agent.water_search_start_location = agent.location
        agent.water_search_actual_distance = 0.0
        
        # Record search start
        if chosen_target is not None:
            ty, tx = chosen_target
            straight_dist = float(np.sqrt((ty - cy)**2 + (tx - cx)**2))
            world.telemetry.record_water_search_start(straight_dist)
                
    return chosen_action, chosen_target

def step_toward(agent: Agent, target: tuple, world) -> bool:
    """
    Takes one step toward the target coordinate.
    If the target is reached, performs the associated action.
    Returns True if agent moved, False if stationary.
    """
    if agent.dead or target is None:
        return False
        
    # Probabilistic movement delay based on age-scaled speed
    if np.random.uniform() > agent.effective_speed:
        return False
        
    cy, cx = agent.location
    ty, tx = target    # Already at target
    if cy == ty and cx == tx:
        # Perform action
        if agent.current_action == "Drinking":
            # Verify fresh water presence (Lakes, active River flow)
            has_water = (world.biome[ty, tx] == 8) or (world.lake_map[ty, tx] > 0.0) or (world.river_map[ty, tx] > 1500.0)
            if has_water:
                agent.thirst = 0.0
                agent.stored_water = 100.0  # Auto-fill pouch to capacity (canteen) when drinking at source
                agent._consumed_water = True
                agent.drinks_count += 1
                season_id = (world.tick % 360) // 90
                
                # Water telemetry outcomes (Phase 10)
                agent.visited_dry_water_recently = False
                if getattr(agent, "water_search_start_location", None) is not None:
                    sy, sx = agent.water_search_start_location
                    ty, tx = target
                    straight_dist = float(np.sqrt((ty - sy)**2 + (tx - sx)**2))
                    actual_dist = getattr(agent, "water_search_actual_distance", 0.0)
                    efficiency = straight_dist / actual_dist if actual_dist > 0.0 else 1.0
                    ticks_taken = world.tick - agent.water_search_start_tick
                    world.telemetry.record_water_search_outcome(True, ticks_taken, efficiency)
                    agent.water_search_start_location = None
                
                if target in agent.knowledge.water_sources:
                    node = agent.knowledge.water_sources[target]
                    node["confidence"] = 1.0
                    node["last_seen_tick"] = world.tick
                    node["season_seen"] = season_id
                    if "active_seasons" not in node:
                        node["active_seasons"] = {0: 0, 1: 0, 2: 0, 3: 0}
                    if "dry_seasons" not in node:
                        node["dry_seasons"] = {0: 0, 1: 0, 2: 0, 3: 0}
                    node["active_seasons"][season_id] += 1
                    node["dry_seasons"][season_id] = 0
                else:
                    agent.knowledge.water_sources[target] = {
                        "confidence": 1.0,
                        "last_seen_tick": world.tick,
                        "season_seen": season_id,
                        "active_seasons": {0: 0, 1: 0, 2: 0, 3: 0, season_id: 1},
                        "dry_seasons": {0: 0, 1: 0, 2: 0, 3: 0}
                    }
                    
                # Check for prediction success and gain
                if agent.predicted_destination is not None:
                    pred_loc, pred_season, bypassed_target = agent.predicted_destination
                    day = world.tick % 360
                    current_season = day // 90
                    is_transition = (day % 90) >= 45
                    
                    if target == pred_loc and (current_season == pred_season or (is_transition and (current_season + 1) % 4 == pred_season)):
                        agent.prediction_successes += 1
                        agent.prediction_confidence = min(1.0, agent.prediction_confidence + 0.05)
                        
                        # Verify if the bypassed closest target was actually dry (Prediction Gain)
                        if bypassed_target is not None:
                            by_y, by_x = bypassed_target
                            by_water = (world.biome[by_y, by_x] == 8) or (world.lake_map[by_y, by_x] > 0.0) or (world.river_map[by_y, by_x] > 1500.0)
                            if not by_water:
                                agent.prediction_gains += 1
                                
                    agent.predicted_destination = None
            else:
                # Dried up! Apply uncertainty decay (x0.2)
                agent.failed_water_visits += 1
                agent._failed_visit = True
                agent.prediction_confidence = max(0.1, agent.prediction_confidence - 0.15)
                season_id = (world.tick % 360) // 90
                
                # Water telemetry outcomes (Phase 10)
                agent.visited_dry_water_recently = True
                if getattr(agent, "water_search_start_location", None) is not None:
                    ticks_taken = world.tick - agent.water_search_start_tick
                    world.telemetry.record_water_search_outcome(False, ticks_taken, 0.0)
                    agent.water_search_start_location = None
                if target in agent.knowledge.water_sources:
                    node = agent.knowledge.water_sources[target]
                    node["confidence"] *= 0.2
                    if "active_seasons" not in node:
                        node["active_seasons"] = {0: 0, 1: 0, 2: 0, 3: 0}
                    if "dry_seasons" not in node:
                        node["dry_seasons"] = {0: 0, 1: 0, 2: 0, 3: 0}
                    node["dry_seasons"][season_id] += 1
                    node["active_seasons"][season_id] = 0
                    
                    # Change 4: Ecological failure tracking (enables seasonal learning)
                    node["failure_count"]   = node.get("failure_count", 0) + 1
                    node["last_failed_tick"] = world.tick
                    
                    if node["confidence"] < 0.05:
                        del agent.knowledge.water_sources[target]
                        agent.nodes_removed_count += 1
                
                # Change 4: Per-entry cache invalidation — remove only the failed spot
                # so the planner immediately re-ranks the remaining known sources.
                if hasattr(agent, "memory_cache") and "water_spots" in agent.memory_cache:
                    spots = agent.memory_cache["water_spots"]
                    if target in spots:
                        spots.remove(target)
                        # If no spots remain, force a full cache rebuild next cognition tick
                        if not spots:
                            agent.memory_cache["last_update_tick"] = -1
                            
                agent.predicted_destination = None
            agent.current_action = "Idle"
            
        elif agent.current_action == "Eating":
            # Verify food presence (Wildlife or Fertility on land tiles)
            has_food = (world.elevation[ty, tx] >= 0.3) and ((world.wildlife[ty, tx] > 0.15) or (world.fertility[ty, tx] > 0.4)) and (world.biome[ty, tx] != 8)
            if has_food:
                agent.hunger = max(0.0, agent.hunger - 60.0)
                agent.stored_food = 100.0   # Auto-fill pouch to capacity when eating at source
                agent._consumed_food = True
                agent.eats_count += 1
                season_id = (world.tick % 360) // 90
                
                if target in agent.knowledge.food_sources:
                    node = agent.knowledge.food_sources[target]
                    node["confidence"] = 1.0
                    node["last_seen_tick"] = world.tick
                    node["season_seen"] = season_id
                    if "active_seasons" not in node:
                        node["active_seasons"] = {0: 0, 1: 0, 2: 0, 3: 0}
                    if "dry_seasons" not in node:
                        node["dry_seasons"] = {0: 0, 1: 0, 2: 0, 3: 0}
                    node["active_seasons"][season_id] += 1
                    node["dry_seasons"][season_id] = 0
                else:
                    agent.knowledge.food_sources[target] = {
                        "confidence": 1.0,
                        "last_seen_tick": world.tick,
                        "season_seen": season_id,
                        "active_seasons": {0: 0, 1: 0, 2: 0, 3: 0, season_id: 1},
                        "dry_seasons": {0: 0, 1: 0, 2: 0, 3: 0}
                    }
                
                # Consume resource from world (deplete wildlife/fertility locally by 5%)
                world.wildlife[ty, tx] = max(0.0, world.wildlife[ty, tx] - 0.05)
                world.fertility[ty, tx] = max(0.0, world.fertility[ty, tx] - 0.02)
                
                # Check for prediction success and gain
                if agent.predicted_destination is not None:
                    pred_loc, pred_season, bypassed_target = agent.predicted_destination
                    day = world.tick % 360
                    current_season = day // 90
                    is_transition = (day % 90) >= 45
                    
                    if target == pred_loc and (current_season == pred_season or (is_transition and (current_season + 1) % 4 == pred_season)):
                        agent.prediction_successes += 1
                        agent.prediction_confidence = min(1.0, agent.prediction_confidence + 0.05)
                        
                        # Verify if the bypassed closest target was actually dry (Prediction Gain)
                        if bypassed_target is not None:
                            by_y, by_x = bypassed_target
                            by_food = (world.elevation[by_y, by_x] >= 0.3) and ((world.wildlife[by_y, by_x] > 0.15) or (world.fertility[by_y, by_x] > 0.4)) and (world.biome[by_y, by_x] != 8)
                            if not by_food:
                                agent.prediction_gains += 1
                                
                    agent.predicted_destination = None
            else:
                # Migrated/Dried up! Apply uncertainty decay (x0.2)
                agent.failed_food_visits += 1
                agent._failed_visit = True
                agent.prediction_confidence = max(0.1, agent.prediction_confidence - 0.15)
                season_id = (world.tick % 360) // 90
                if target in agent.knowledge.food_sources:
                    node = agent.knowledge.food_sources[target]
                    node["confidence"] *= 0.2
                    if "active_seasons" not in node:
                        node["active_seasons"] = {0: 0, 1: 0, 2: 0, 3: 0}
                    if "dry_seasons" not in node:
                        node["dry_seasons"] = {0: 0, 1: 0, 2: 0, 3: 0}
                    node["dry_seasons"][season_id] += 1
                    node["active_seasons"][season_id] = 0
                    
                    if node["confidence"] < 0.05:
                        del agent.knowledge.food_sources[target]
                        agent.nodes_removed_count += 1
                        
                agent.predicted_destination = None
            agent.current_action = "Idle"
            
        elif agent.current_action == "Exploring":
            agent.curiosity_need = 0.0
            agent.current_action = "Idle"
            
        elif agent.current_action == "Building Shelter":
            # Repair durability
            shelter_build_speed_mult = getattr(world, "shelter_build_speed_mult", 1.0)
            agent.shelter_durability = min(100.0, agent.shelter_durability + 10.0 * shelter_build_speed_mult)
            if hasattr(world, "shelters") and agent.shelter_location in world.shelters:
                world.shelters[agent.shelter_location]["durability"] = agent.shelter_durability
            
            # Check for upgrades if durability is fully restored
            if agent.shelter_durability >= 100.0:
                sy, sx = agent.shelter_location
                # Upgrade Level 1 (Primitive) -> Level 2 (Wood Cabin) if wood is available
                if agent.shelter_level == 1:
                    if world.wood[sy, sx] > 0.2:
                        agent.shelter_level = 2
                        agent._shelter_upgraded = True
                        world.wood[sy, sx] = max(0.0, world.wood[sy, sx] - 0.15)
                        world.history.append(
                            f"Year {world.tick // 360}, Day {world.tick % 360}: Agent {agent.id} "
                            f"upgraded shelter to Wood Cabin at coordinate ({sx}, {sy})."
                        )
                # Upgrade Level 2 (Wood Cabin) -> Level 3 (Stone Shelter) if stone is available
                elif agent.shelter_level == 2:
                    if world.stone[sy, sx] > 0.2:
                        agent.shelter_level = 3
                        agent._shelter_upgraded = True
                        world.stone[sy, sx] = max(0.0, world.stone[sy, sx] - 0.15)
                        world.history.append(
                            f"Year {world.tick // 360}, Day {world.tick % 360}: Agent {agent.id} "
                            f"upgraded shelter to Stone Shelter at coordinate ({sx}, {sy})."
                        )
                if hasattr(world, "shelters") and agent.shelter_location in world.shelters:
                    world.shelters[agent.shelter_location]["level"] = agent.shelter_level
            agent.current_action = "Idle"
            
        elif agent.current_action == "Sheltering":
            # Remain in shelter
            pass
 
        elif agent.current_action == "Reproduce":
            agent._wants_to_reproduce_with = target
 
        elif agent.current_action == "Store Food":
            has_food = (
                world.elevation[ty, tx] >= 0.3
                and (world.wildlife[ty, tx] > 0.15 or world.fertility[ty, tx] > 0.4)
                and world.biome[ty, tx] != 8
            )
            if has_food:
                amount = min(30.0, 100.0 - agent.stored_food)
                agent.stored_food += amount
                world.wildlife[ty, tx] = max(0.0, world.wildlife[ty, tx] - 0.02)
            agent.current_action = "Idle"
 
        elif agent.current_action == "Store Water":
            has_water = (
                world.biome[ty, tx] == 8
                or world.lake_map[ty, tx] > 0.0
                or world.river_map[ty, tx] > 1500.0
            )
            if has_water:
                amount = min(30.0, 100.0 - agent.stored_water)
                agent.stored_water += amount
            agent.current_action = "Idle"
 
        elif agent.current_action == "Share Food":
            for other in world.agents:
                if (not other.dead and other.id != agent.id
                        and other.location == (ty, tx)
                        and getattr(other, "colony_id", -1) == getattr(agent, "colony_id", -1)):
                    amount = min(20.0, agent.stored_food)
                    if amount > 0 and other.hunger > 0:
                        give = min(amount, max(0.0, 80.0 - other.hunger) * 0.5)
                        agent.stored_food  -= give
                        other.hunger       = max(0.0, other.hunger - give * 2.0)
                        other._consumed_food = True
                        # Record cooperational memory for both
                        agent.add_memory(PERSON, other.location, world.tick, importance=0.4, associated_id=other.id, outcome="share")
                        other.add_memory(PERSON, agent.location, world.tick, importance=0.4, associated_id=agent.id, outcome="share")
            agent.current_action = "Idle"
 
        elif agent.current_action == "Share Water":
            for other in world.agents:
                if (not other.dead and other.id != agent.id
                        and other.location == (ty, tx)
                        and getattr(other, "colony_id", -1) == getattr(agent, "colony_id", -1)):
                    amount = min(20.0, agent.stored_water)
                    if amount > 0 and other.thirst > 0:
                        give = min(amount, max(0.0, 80.0 - other.thirst) * 0.5)
                        agent.stored_water -= give
                        other.thirst       = max(0.0, other.thirst - give * 2.0)
                        other._consumed_water = True
                        agent.add_memory(PERSON, other.location, world.tick, importance=0.4, associated_id=other.id, outcome="share")
                        other.add_memory(PERSON, agent.location, world.tick, importance=0.4, associated_id=agent.id, outcome="share")
            agent.current_action = "Idle"

        elif agent.current_action == "Drink Stored Water":
            consumed = min(agent.stored_water, agent.thirst)
            if consumed > 0.0:
                agent.stored_water -= consumed
                agent.thirst -= consumed
                agent._consumed_water = True
                agent.drinks_count += 1
            agent.current_action = "Idle"

        elif agent.current_action == "Eat Stored Food":
            consumed = min(agent.stored_food, agent.hunger)
            if consumed > 0.0:
                agent.stored_food -= consumed
                agent.hunger -= consumed
                agent._consumed_food = True
                agent.eats_count += 1
            agent.current_action = "Idle"

        elif agent.current_action == "Deposit Food":
            c_id = getattr(agent, "colony_id", -1)
            if c_id >= 0 and c_id < len(world.colonies):
                # Keep a personal reserve of 30.0 food in the pouch for survival, deposit only the excess
                deposit_amount = max(0.0, agent.stored_food - 30.0)
                if deposit_amount > 0.0:
                    world.colonies[c_id]["stored_food"] += deposit_amount
                    agent.stored_food -= deposit_amount
            agent.current_action = "Idle"

        elif agent.current_action == "Deposit Water":
            c_id = getattr(agent, "colony_id", -1)
            if c_id >= 0 and c_id < len(world.colonies):
                # Keep a personal reserve of 30.0 water in the pouch for survival, deposit only the excess
                deposit_amount = max(0.0, agent.stored_water - 30.0)
                if deposit_amount > 0.0:
                    world.colonies[c_id]["stored_water"] += deposit_amount
                    agent.stored_water -= deposit_amount
            agent.current_action = "Idle"

        elif agent.current_action == "Withdraw Food":
            c_id = getattr(agent, "colony_id", -1)
            if c_id >= 0 and c_id < len(world.colonies):
                colony = world.colonies[c_id]
                # First satisfy immediate hunger need
                withdraw_amount = min(colony["stored_food"], agent.hunger)
                if withdraw_amount > 0.0:
                    colony["stored_food"] -= withdraw_amount
                    agent.hunger -= withdraw_amount
                    agent._consumed_food = True
                    agent.eats_count += 1
                # Then top off pouch if chest has excess
                needed = 100.0 - agent.stored_food
                if needed > 0.0 and colony["stored_food"] > 0.0:
                    fill = min(colony["stored_food"], needed)
                    colony["stored_food"] -= fill
                    agent.stored_food += fill
            agent.current_action = "Idle"

        elif agent.current_action == "Withdraw Water":
            c_id = getattr(agent, "colony_id", -1)
            if c_id >= 0 and c_id < len(world.colonies):
                colony = world.colonies[c_id]
                # First satisfy immediate thirst need
                withdraw_amount = min(colony["stored_water"], agent.thirst)
                if withdraw_amount > 0.0:
                    colony["stored_water"] -= withdraw_amount
                    agent.thirst -= withdraw_amount
                    agent._consumed_water = True
                    agent.drinks_count += 1
                # Then top off pouch if chest has excess
                needed = 100.0 - agent.stored_water
                if needed > 0.0 and colony["stored_water"] > 0.0:
                    fill = min(colony["stored_water"], needed)
                    colony["stored_water"] -= fill
                    agent.stored_water += fill
            agent.current_action = "Idle"
 
        return False
        
    # Move one step closer
    dy = np.sign(ty - cy)
    dx = np.sign(tx - cx)
    
    ny, nx = cy, cx
    dist_y = abs(ty - cy)
    dist_x = abs(tx - cx)
    
    # Step along coordinate axis of larger distance
    if dist_y >= dist_x and dist_y > 0:
        ny += dy
    elif dist_x > 0:
        nx += dx
        
    # Check boundaries
    ny = np.clip(ny, 0, world.height - 1)
    nx = np.clip(nx, 0, world.width - 1)
    
    agent.location = (int(ny), int(nx))
    agent.path_history.append((int(ny), int(nx)))
    
    # Water telemetry path actual distance tracking (Phase 10)
    if agent.current_action == "Drinking":
        agent.water_search_actual_distance = getattr(agent, "water_search_actual_distance", 0.0) + 1.0
    
    # Update maximum radius ever reached from spawn location
    sy, sx = agent.spawn_location
    dist = float(np.sqrt((ny - sy)**2 + (nx - sx)**2))
    agent.max_radius = max(agent.max_radius, dist)
    
    # Prune path history to prevent memory leaks
    if len(agent.path_history) > 1000:
        agent.path_history.pop(0)
        
    # Keep track of home range drift (slowly pull home towards frequented spots)
    # Home shifts by 0.1% towards current location to reflect territory centers
    hy, hx = agent.home_location
    agent.home_location = (
        int(hy * 0.999 + ny * 0.001),
        int(hx * 0.999 + nx * 0.001)
    )
    
    agent._stepped = True

    # Change 5: Environmental cue — set standing_on_water flag for the Reflex Layer.
    # Separates perception from action: the flag says "water is immediately available here".
    # The Reflex Layer in simulation.py decides whether to act on it (thirst threshold, etc.).
    agent.standing_on_water = (
        world.biome[int(ny), int(nx)] == 8
        or world.lake_map[int(ny), int(nx)] > 0.0
        or world.river_map[int(ny), int(nx)] > 1500.0
    )
    return True
