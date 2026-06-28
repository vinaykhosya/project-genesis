import sys
import time
import numpy as np
from .agent import Agent, WATER, FOOD, PERSON
from .perception import perceive
from .decision import evaluate_utility, step_toward, update_agent_needs
from .genetics import create_genome, express_genome, population_diversity
from .reproduction import attempt_reproduce
from .drives import update_biological_drives, update_emotional_drives, update_relationships, trigger_grief, emotional_label, update_adaptive_motivations
from ..predictor import predict_settlements
from ..state import WorldState, BIOME_NAMES

# Ensure stdout handles Unicode characters (e.g. history events with arrows)
# on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
ACTION_TO_ID = {
    "Idle": 0,
    "Exploring": 1,
    "Drinking": 2,
    "Eating": 3,
    "Resting": 4,
    "Building Shelter": 5,
    "Sheltering": 6,
    "Reproduce": 7,
    "Store Food": 8,
    "Store Water": 9,
    "Share Food": 10,
    "Share Water": 11,
    "Drink Stored Water": 12,
    "Eat Stored Food": 13,
    "Deposit Food": 14,
    "Deposit Water": 15,
    "Withdraw Food": 16,
    "Withdraw Water": 17,
    "Dead": 18
}

# ==============================================================================
# REFLEX LAYER
# Architectural position: after Physiology, before Cognition.
# Contains hard-wired survival responses that bypass deliberation when a drive
# tension crosses a critical threshold.  All future survival reflexes should be
# added here so they remain a coherent, inspectable layer of the nervous system.
#
# Reflexes implemented:
#   1. Emergency Drinking  — if thirst_tension > 0.95 and stored_water >= 5
#   2. Emergency Eating    — if hunger_tension > 0.95 and stored_food  >= 5
#   3. Opportunistic Sip   — if standing_on_water after a step and thirst >= 60
#
# Design rules:
#   - Reflexes do NOT change the agent's current_action or target_coordinate.
#     They fire physiologically in the background; the planner is unaware.
#   - Each reflex is guarded by a tension threshold so it is genuinely a
#     last-resort response, not a routine behaviour.
#   - The standing_on_water flag is set by step_toward() (decision.py) as a
#     perception cue; the reflex decides whether to act on it.
# ==============================================================================

# Reflex thresholds (defined here so they are easy to find and tune)
_REFLEX_EMERGENCY_THIRST_TENSION = 0.90   # smoothed drive tension [0–1]
_REFLEX_EMERGENCY_HUNGER_TENSION = 0.90
_REFLEX_EMERGENCY_WATER_RESERVE  = 5.0    # minimum stored water to trigger
_REFLEX_EMERGENCY_FOOD_RESERVE   = 5.0    # minimum stored food to trigger
_REFLEX_EMERGENCY_DRINK_AMOUNT   = 30.0   # maximum units consumed per reflex tick
_REFLEX_EMERGENCY_EAT_AMOUNT     = 30.0
_REFLEX_OPPORTUNISTIC_THRESHOLD  = 60.0   # raw thirst to trigger an opportunistic sip
_REFLEX_OPPORTUNISTIC_AMOUNT     = 12.0   # sip size (partial, not a full quench)


def _run_survival_reflexes(agent: Agent, world) -> None:
    """
    Reflex Layer — runs every tick between Physiology and Cognition.

    Executes hard-wired survival responses when biological drive tensions
    cross critical thresholds.  Each reflex modifies physiology directly
    without touching the agent's action queue or current_action.

    Parameters
    ----------
    agent : Agent
    world : WorldState
    """
    if agent.dead:
        return

    ds = agent.drives

    # Initialize reflex debug telemetry for post-mortem diagnostics (Change 1d)
    agent._reflex_debug = {
        "thirst_tension": round(float(ds.thirst_tension), 3),
        "hunger_tension": round(float(ds.hunger_tension), 3),
        "drink_reflex":   "inactive",
        "eat_reflex":     "inactive",
        "sip_reflex":     "inactive"
    }

    # ------------------------------------------------------------------
    # Reflex 1: Emergency Drinking
    # Fires when thirst_tension is critical and the agent has stored water.
    # The agent sips from their pouch immediately, then continues their
    # current planned action (e.g., walking to the river).
    # ------------------------------------------------------------------
    if ds.thirst_tension >= _REFLEX_EMERGENCY_THIRST_TENSION:
        if agent.stored_water >= _REFLEX_EMERGENCY_WATER_RESERVE:
            consumed = min(_REFLEX_EMERGENCY_DRINK_AMOUNT,
                           agent.stored_water, agent.thirst)
            if consumed > 0.0:
                agent.stored_water -= consumed
                agent.thirst        = max(0.0, agent.thirst - consumed)
                agent._consumed_water = True
                agent.drinks_count  += 1
                agent._reflex_debug["drink_reflex"] = f"fired (consumed={round(consumed,1)})"
            else:
                agent._reflex_debug["drink_reflex"] = "zero_need"
        else:
            agent._reflex_debug["drink_reflex"] = f"no_water_reserve (stored={round(agent.stored_water,1)})"
    elif ds.thirst_tension > 0.8:
        agent._reflex_debug["drink_reflex"] = f"tension_below_threshold (tension={round(ds.thirst_tension,3)})"

    # ------------------------------------------------------------------
    # Reflex 2: Emergency Eating
    # Fires when hunger_tension is critical and the agent has stored food.
    # ------------------------------------------------------------------
    if ds.hunger_tension >= _REFLEX_EMERGENCY_HUNGER_TENSION:
        if agent.stored_food >= _REFLEX_EMERGENCY_FOOD_RESERVE:
            consumed = min(_REFLEX_EMERGENCY_EAT_AMOUNT,
                           agent.stored_food, agent.hunger)
            if consumed > 0.0:
                agent.stored_food -= consumed
                agent.hunger       = max(0.0, agent.hunger - consumed)
                agent._consumed_food = True
                agent.eats_count   += 1
                agent._reflex_debug["eat_reflex"] = f"fired (consumed={round(consumed,1)})"
            else:
                agent._reflex_debug["eat_reflex"] = "zero_need"
        else:
            agent._reflex_debug["eat_reflex"] = f"no_food_reserve (stored={round(agent.stored_food,1)})"
    elif ds.hunger_tension > 0.8:
        agent._reflex_debug["eat_reflex"] = f"tension_below_threshold (tension={round(ds.hunger_tension,3)})"

    # ------------------------------------------------------------------
    # Reflex 3: Opportunistic Sipping
    # Fires when the agent steps onto a water tile and is sufficiently
    # thirsty.  The flag is set by step_toward() (decision.py) after each
    # successful move.  This separates perception (am I on water?) from
    # the decision (should I drink?).
    # ------------------------------------------------------------------
    if getattr(agent, "standing_on_water", False):
        if agent.thirst >= _REFLEX_OPPORTUNISTIC_THRESHOLD:
            sip = min(_REFLEX_OPPORTUNISTIC_AMOUNT, agent.thirst)
            agent.thirst        = max(0.0, agent.thirst - sip)
            agent._consumed_water = True
            agent.drinks_count  += 1
            agent._reflex_debug["sip_reflex"] = f"fired_opportunistic (sip={round(sip,1)})"
            # Update memory for this tile so the planner benefits from the discovery
            cy, cx = agent.location
            loc = (cy, cx)
            season_id = (world.tick % 360) // 90
            if loc not in agent.knowledge.water_sources:
                agent.knowledge.water_sources[loc] = {
                    "confidence":    1.0,
                    "last_seen_tick": world.tick,
                    "season_seen":   season_id,
                    "active_seasons": {0: 0, 1: 0, 2: 0, 3: 0, season_id: 1},
                    "dry_seasons":   {0: 0, 1: 0, 2: 0, 3: 0}
                }
            else:
                agent.knowledge.water_sources[loc]["confidence"]    = 1.0
                agent.knowledge.water_sources[loc]["last_seen_tick"] = world.tick
        else:
            agent._reflex_debug["sip_reflex"] = f"thirst_below_threshold (thirst={round(agent.thirst,1)})"
        # Clear flag so it doesn't fire again next tick without a new step
        agent.standing_on_water = False
    else:
        agent._reflex_debug["sip_reflex"] = "not_standing_on_water"



def simulate_agent_tick(agent: Agent, world: WorldState):
    """
    Executes the perceive -> evaluate -> step -> update needs cycle for a single agent.
    Also handles predictor training, concept confirmation/violation feedback, and procedural sequencing.
    """
    if agent.dead:
        return
        
    # Reset prediction registers for the current tick
    agent.last_prediction_error = 0.0
    agent.last_prediction_category = "Idle"
    
    # Construct ephemeral TickContext
    from .cognitive import TickContext
    cy, cx = agent.location
    biome_id = int(world.biome[cy, cx])
    amplitude = {6: 22.0, 1: 18.0, 2: 18.0, 3: 18.0,
                 4: 12.0, 5: 12.0, 0: 6.0, 8: 6.0, 7: 4.0}.get(biome_id, 12.0)
    day = world.tick % 360
    season = day // 90
    cos_factor = getattr(world, "temp_cos_factor", None)
    if cos_factor is None:
        cos_factor = float(np.cos(((day - 180.0) / 180.0) * np.pi))
    temp_seasonal_offset = amplitude * cos_factor
    global_offset = getattr(world, "global_temp_offset", 0.0)
    local_temp = float(world.temperature[cy, cx]) + temp_seasonal_offset + global_offset

    max_search_radius = max(agent.vision_radius, 30)
    nearby_agents = world.query_agents((cy, cx), max_search_radius, alive_only=True)

    tick_context = TickContext(
        context_tick=world.tick,
        brain=agent.brain,
        drives=agent.drives,
        local_temp=local_temp,
        day=day,
        season=season,
        cos_factor=cos_factor,
        nearby_agents=nearby_agents
    )
        
    # Predictor learning based on previous tick outcome
    if getattr(agent, "last_prediction_input", None) is not None:
        meaningful_event = (
            agent._consumed_food or
            agent._consumed_water or
            agent._failed_visit or
            agent._shelter_upgraded or
            (agent.injury_level > agent.last_injury_level) or
            (agent.health < agent.last_health) or
            (len(agent.children_ids) > agent.last_children_count) or
            agent._stepped
        )

        if meaningful_event:
            # Rich reward calculation
            reward = 1.0  # survival baseline
            if agent._consumed_food:
                reward += 3.0
            if agent._consumed_water:
                reward += 3.0
            if agent.shelter_location is not None:
                reward += 0.5 * (agent.shelter_durability / 100.0)
            if agent._shelter_upgraded:
                reward += 2.0
            if len(agent.children_ids) > agent.last_children_count:
                reward += 5.0
            
            # Penalties
            if agent.injury_level > agent.last_injury_level:
                reward -= 2.0 * (agent.injury_level - agent.last_injury_level)
            if agent.health < agent.last_health:
                reward -= 4.0 * (agent.last_health - agent.health)
            if agent.hunger > 80.0:
                reward -= 0.5 * (agent.hunger - 80.0) / 20.0
            if agent.thirst > 80.0:
                reward -= 0.5 * (agent.thirst - 80.0) / 20.0
            if agent._stepped:
                reward -= 0.05
            if agent._failed_visit:
                reward -= 1.5

            t0_nn = time.perf_counter()
            try:
                pred_val = agent.predictor.predict(agent.last_prediction_input)
                error = reward - pred_val
                agent.total_prediction_error += error * error
                agent.prediction_count += 1
                
                # Capture prediction error and category for emotional routing
                if getattr(world, "ablation", {}).get("prediction_error", True):
                    agent.last_prediction_error = float(error)
                    agent.last_prediction_category = agent.current_action
                else:
                    agent.last_prediction_error = 0.0
                    agent.last_prediction_category = "Idle"
            except Exception:
                pass
            t_nn = (time.perf_counter() - t0_nn) * 1000.0
            if hasattr(world, "profiler") and world.profiler is not None:
                p = world.profiler["neural_predictor"]
                p["calls"] += 1
                p["time"] += t_nn

            # Phase 7.5 Sleep Consolidation: Buffer training sample instead of training immediately!
            if not hasattr(agent, "training_buffer"):
                agent.training_buffer = []
            agent.training_buffer.append((agent.last_prediction_input, reward))

            # Phase 7.5 Sleep Consolidation: Buffer procedural memory sequence caching
            if len(agent.action_history) == 3:
                current_season = (world.tick % 360) // 90
                action_seq = tuple(agent.action_history)
                if not hasattr(agent, "procedural_buffer"):
                    agent.procedural_buffer = []
                agent.procedural_buffer.append((current_season, action_seq, reward))

        # Concept confirmation and violation feedback
        last_target = agent.target_coordinate
        if last_target is not None:
            if agent._consumed_water:
                for c in agent.concepts.get("WATER", []):
                    if agent.matches_concept("WATER", last_target[0], last_target[1], world):
                        c.confidence = min(1.0, c.confidence + 0.1)
            elif agent._consumed_food:
                for c in agent.concepts.get("FOOD", []):
                    if agent.matches_concept("FOOD", last_target[0], last_target[1], world):
                        c.confidence = min(1.0, c.confidence + 0.1)
            elif agent._failed_visit:
                failed_res = "WATER" if agent.current_action == "Drinking" else "FOOD"
                for c in agent.concepts.get(failed_res, []):
                    if agent.matches_concept(failed_res, last_target[0], last_target[1], world):
                        c.confidence = max(0.0, c.confidence - 0.3)
                agent.concepts[failed_res] = [c for c in agent.concepts[failed_res] if c.confidence >= 0.1]

        # Metacognitive self-trust update
        if agent._consumed_food or agent._consumed_water or agent._shelter_upgraded:
            agent.planning_accuracy = min(1.0, agent.planning_accuracy + 0.02)
        elif agent._failed_visit:
            agent.planning_accuracy = max(0.1, agent.planning_accuracy - 0.05)

        # Reset trackers
        agent._consumed_food = False
        agent._consumed_water = False
        agent._shelter_upgraded = False
        agent._failed_visit = False
        agent._stepped = False
        agent.last_prediction_input = None

    # Update tracking registers for the current tick
    agent.last_health = agent.health
    agent.last_hunger = agent.hunger
    agent.last_thirst = agent.thirst
    agent.last_injury_level = agent.injury_level
    agent.last_children_count = len(agent.children_ids)

    # Update individual age, ticks survived, and seasonal observations
    agent.ticks_survived += 1
    agent.age += 1
    agent.season_observations[(world.tick % 360) // 90] += 1
        
    
    # ==================================================================
    # CLOCK 1: PHYSIOLOGY CLOCK (Runs every tick)
    # ==================================================================
    # Cheap continuous perception scan (situational awareness)
    t0 = time.perf_counter()
    perceive(agent, world, vision_radius=agent.vision_radius)
    t_perc = (time.perf_counter() - t0) * 1000.0  # Convert to ms

    t0_dr = time.perf_counter()
    update_biological_drives(agent, tick_context.local_temp)
    t_bio = (time.perf_counter() - t0_dr) * 1000.0

    # Phase 8.2: Emotional Clock — biologically appropriate rate: every 30 ticks.
    # Emotions integrate slowly from sustained drives; 30-tick updates are sufficient
    # and reduce per-agent cost by 3× vs the original 10-tick rate.
    if world.tick % 30 == 0:
        t0_emo = time.perf_counter()
        update_emotional_drives(agent, world)
        t_emo = (time.perf_counter() - t0_emo) * 1000.0
        if hasattr(world, "profiler") and world.profiler is not None:
            p = world.profiler["emotion_clock"]
            p["calls"] += 1
            p["time"] += t_emo
            if t_emo > p["max"]:
                p["max"] = t_emo
                p["worst_tick"] = world.tick

    # Phase 8.3: Relationship Clock — every 120 ticks.
    # Relationship trust shifts on the timescale of days/weeks in agent-time;
    # 120 ticks (≈ 1 simulation day) is physiologically appropriate and 4×
    # cheaper than the previous 30-tick rate.
    if world.tick % 120 == 0:
        t0_rel = time.perf_counter()
        update_relationships(agent, world)
        t_rel = (time.perf_counter() - t0_rel) * 1000.0
        if hasattr(world, "profiler") and world.profiler is not None:
            p = world.profiler["relationship_clock"]
            p["calls"] += 1
            p["time"] += t_rel
            if t_rel > p["max"]:
                p["max"] = t_rel
                p["worst_tick"] = world.tick

    # Phase 8.4: Personality/Motivation Clock — every 500 ticks.
    # Motivation profile drifts over seasons; 500 ticks (≈ 1.4 sim-days) is
    # biologically appropriate and 5× cheaper than the previous 100-tick rate.
    if world.tick % 500 == 0:
        t0_mot = time.perf_counter()
        update_adaptive_motivations(agent, world)
        t_mot = (time.perf_counter() - t0_mot) * 1000.0
        if hasattr(world, "profiler") and world.profiler is not None:
            p = world.profiler["motivation_clock"]
            p["calls"] += 1
            p["time"] += t_mot
            if t_mot > p["max"]:
                p["max"] = t_mot
                p["worst_tick"] = world.tick

    t_dr = t_bio  # biological drive time for the drives_relationships bucket
    if hasattr(world, "profiler") and world.profiler is not None:
        p = world.profiler["drives_relationships"]
        p["calls"] += 1
        p["time"] += t_dr
        if t_dr > p["max"]:
            p["max"] = t_dr
            p["worst_tick"] = world.tick
    
    if hasattr(world, "profiler") and world.profiler is not None:
        p = world.profiler["perception"]
        p["calls"] += 1
        p["time"] += t_perc
        if t_perc > p["max"]:
            p["max"] = t_perc
            p["worst_tick"] = world.tick
            
    # Senses & Interrupt calculations
    cy, cx = agent.location
    interrupt_triggered = False
    max_attention = 0.0
    
    # 1. Internal Need Interrupts (only if not already addressing them)
    if agent.current_action != "Drinking":
        if agent.thirst > max_attention:
            max_attention = agent.thirst
    if agent.current_action != "Eating":
        if agent.hunger > max_attention:
            max_attention = agent.hunger
    if agent.current_action not in ("Resting", "Sheltering"):
        energy_need = 100.0 - agent.energy
        if energy_need > max_attention:
            max_attention = energy_need
            
    # 2. External Sensory Interrupts (newly perceived resources and neighbors)
    for mem in reversed(agent.episodic_memory):
        if mem.timestamp != world.tick:
            break
        m_type, m_loc = mem.type, mem.location
        dy, dx = m_loc[0] - cy, m_loc[1] - cx
        dist = np.sqrt(dy*dy + dx*dx)
        urgency = max(0.0, 1.0 - dist / agent.vision_radius)
        
        if m_type == WATER and agent.current_action != "Drinking":
            novelty = 1.0 if m_loc not in agent.knowledge.water_sources else 0.2
            attn = novelty * urgency * (agent.thirst / 100.0) * 100.0
            if attn > max_attention:
                max_attention = attn
        elif m_type == FOOD and agent.current_action != "Eating":
            novelty = 1.0 if m_loc not in agent.knowledge.food_sources else 0.2
            attn = novelty * urgency * (agent.hunger / 100.0) * 100.0
            if attn > max_attention:
                max_attention = attn
        elif m_type == PERSON and dist < 8.0:
            # Saliency of nearby agents
            attn = 0.8 * urgency * 100.0
            if attn > max_attention:
                max_attention = attn

    # 3. Environmental / Disaster Interrupts
    is_disaster = getattr(world, "global_temp_offset", 0.0) != 0.0 or getattr(world, "global_water_multiplier", 1.0) < 1.0 or getattr(world, "global_food_multiplier", 1.0) < 1.0
    if is_disaster and agent.current_action not in ("Sheltering", "Resting") and agent.shelter_location is not None:
        attn = 75.0
        if attn > max_attention:
            max_attention = attn
            
    # Compare max attention to plan commitment
    commitment = getattr(agent, "plan_commitment", 50.0)
    if max_attention > commitment:
        interrupt_triggered = True
        
    # ==================================================================
    # CLOCK 4: LEARNING CLOCK (Sleep Consolidation - Runs during rest/sleep)
    # ==================================================================
    sleep_consolidation_on = getattr(world, "sleep_consolidation_enabled", True)

    def _flush_learning_buffers(agent, world):
        """Shared helper: flush training and procedural buffers immediately."""
        if hasattr(agent, "training_buffer") and agent.training_buffer:
            t0_nn = time.perf_counter()
            for inp, rew in agent.training_buffer:
                agent.predictor.train(inp, rew, agent.learning_rate)
            agent.training_buffer.clear()
            t_nn = (time.perf_counter() - t0_nn) * 1000.0
            if hasattr(world, "profiler") and world.profiler is not None:
                p = world.profiler["neural_predictor"]
                p["calls"] += 1
                p["time"] += t_nn
        if hasattr(agent, "procedural_buffer") and agent.procedural_buffer:
            for current_season, action_seq, reward in agent.procedural_buffer:
                if reward > 4.0:
                    from .cognitive import Procedure
                    similar = None
                    for p in agent.procedures:
                        if p.trigger_season == current_season and p.action_sequence == action_seq:
                            similar = p
                            break
                    if similar is not None:
                        similar.confidence = min(1.0, similar.confidence + 0.1)
                        similar.success_count += 1
                    else:
                        agent.procedures.append(Procedure(
                            trigger_season=current_season,
                            action_sequence=action_seq,
                            confidence=1.0
                        ))
                elif reward < -2.0:
                    for p in agent.procedures:
                        if p.trigger_season == current_season and p.action_sequence == action_seq:
                            p.confidence = max(0.0, p.confidence - 0.2)
                    agent.procedures = [p for p in agent.procedures if p.confidence >= 0.1]
            agent.procedural_buffer.clear()

    if sleep_consolidation_on:
        # ON: consolidation deferred to rest ticks only
        if agent.current_action == "Resting":
            _flush_learning_buffers(agent, world)
            # Concept updates consolidation (runs only during rest, every 100 ticks)
            if world.tick % 100 == 0:
                agent.update_concepts(world)
    else:
        # OFF: flush learning buffers every single tick (continuous learning baseline)
        _flush_learning_buffers(agent, world)
        if world.tick % 100 == 0:
            agent.update_concepts(world)

    # Wake up check for Resting state
    if agent.current_action == "Resting" and agent.energy >= 100.0:
        agent.current_action = "Idle"
        agent.target_coordinate = None
        
    # Check if a failed visit occurred or if a sensory interrupt triggered, forcing immediate queue clearing
    if getattr(agent, "_failed_visit", False) or interrupt_triggered:
        agent.action_queue.clear()
        agent.current_action = "Idle"
        agent.target_coordinate = None

    # ==================================================================
    # CLOCK 1b: REFLEX LAYER (Runs every tick, between Physiology & Cognition)
    # Hard-wired survival responses that bypass deliberation.
    # See _run_survival_reflexes() for reflex definitions and thresholds.
    # ==================================================================
    _run_survival_reflexes(agent, world)

    # ==================================================================
    # CLOCK 2: COGNITION CLOCK (Event-driven / Queue exhaustion)
    # ==================================================================
    # We trigger the cognition clock if the current action is Idle or None, OR
    # if the action queue is completely empty.
    planner_on = getattr(world, "planner_enabled", True)
    is_idle = (agent.current_action == "Idle" or agent.current_action is None)
    
    if is_idle:
        if planner_on and hasattr(agent, "action_queue") and len(agent.action_queue) > 0:
            # CHEAP MOTOR CLOCK STEP: Pop next action from the queue without evaluating utility
            action_name, target = agent.action_queue.pop(0)
            agent.current_action = action_name
            agent.target_coordinate = target
        else:
            # COGNITION CLOCK STEP: Deliberate and formulate a new plan!
            t0_eval = time.perf_counter()
            
            # Planning metabolic cost: deduct 0.2 energy
            agent.energy = max(0.0, agent.energy - 0.2)
            
            # Strategic & Tactical choice (utility & predictor passes)
            action_name, target = evaluate_utility(agent, world, context=tick_context)
            
            # Pop the first action from the newly generated queue (only if planner is on)
            if planner_on and hasattr(agent, "action_queue") and len(agent.action_queue) > 0:
                action_name, target = agent.action_queue.pop(0)
                
            agent.current_action = action_name
            agent.target_coordinate = target
            
            t_eval = (time.perf_counter() - t0_eval) * 1000.0
            if hasattr(world, "profiler") and world.profiler is not None:
                p = world.profiler["decision"]
                p["calls"] += 1
                p["time"] += t_eval
                if t_eval > p["max"]:
                    p["max"] = t_eval
                    p["worst_tick"] = world.tick
    else:
        # Continue executing current action/target
        action_name = agent.current_action
        target = agent.target_coordinate

    # Record action counts for behavioral classification
    agent.action_counts[action_name] = agent.action_counts.get(action_name, 0) + 1

    # ==================================================================
    # CLOCK 1c: DEATH TRACE BUFFER (Change 1a)
    # When thirst enters the doomed zone (> 90), buffer a lightweight
    # per-tick trace snapshot for post-mortem analysis.
    # Captures utility competition (winner vs runner-up) and action
    # distribution so that the exact failure path is reconstructable.
    # ==================================================================
    if agent.thirst > 90.0:
        if not hasattr(agent, "_death_trace"):
            agent._death_trace = []
        if len(agent._death_trace) < 50:
            cy_t, cx_t = agent.location
            agent._death_trace.append({
                "tick":           world.tick,
                "thirst":         round(agent.thirst, 1),
                "stored_water":   round(agent.stored_water, 1),
                "energy":         round(agent.energy, 1),
                "health":         round(agent.health, 1),
                "action":         action_name,
                "target":         agent.target_coordinate,
                "util_winner":    getattr(agent, "_last_utility_winner",   None),
                "util_runnerup":  getattr(agent, "_last_utility_runnerup", None),
                "reflex_debug":   dict(getattr(agent, "_reflex_debug", {})),
                # action_dist is NOT copied here (expensive dict copy every tick).
                # It is captured once at death in update_agent_needs() instead.
            })

    # ==================================================================
    # CLOCK 3: MOTOR CLOCK (Runs every tick)
    # ==================================================================
    t0_step = time.perf_counter()
    moved = step_toward(agent, target, world)
    t_step = (time.perf_counter() - t0_step) * 1000.0
    
    if hasattr(world, "profiler") and world.profiler is not None:
        p = world.profiler["movement"]
        p["calls"] += 1
        p["time"] += t_step
        if t_step > p["max"]:
            p["max"] = t_step
            p["worst_tick"] = world.tick
            
    # 4. Updates: need depletion, health updates
    update_agent_needs(agent, world, moved, context=tick_context)

    # Record current action in history
    agent.action_history.append(action_name)
    if len(agent.action_history) > 3:
        agent.action_history.pop(0)

    # (Concept updates offloaded to Sleep Consolidation during resting)

    # Run daily decays every 360 ticks
    if world.tick % 360 == 0:
        t0_mem = time.perf_counter()
        for r_type in ("WATER", "FOOD"):
            for c in agent.concepts.get(r_type, []):
                c.confidence *= 0.999
            agent.concepts[r_type] = [c for c in agent.concepts[r_type] if c.confidence >= 0.1]
        for p in agent.procedures:
            p.confidence *= 0.99
        agent.procedures = [p for p in agent.procedures if p.confidence >= 0.1]

        # Exponential memory decay and pruning (Phase 8.4)
        decayed_memories = []
        for mem in agent.episodic_memory:
            importance_weight = mem.importance if getattr(world, "ablation", {}).get("memory_importance", True) else 0.5
            mem.confidence = float(mem.confidence * np.exp(-0.2 * (1.0 - importance_weight)))
            if mem.confidence >= 0.10:
                decayed_memories.append(mem)
        agent.episodic_memory = decayed_memories
        t_mem = (time.perf_counter() - t0_mem) * 1000.0
        if hasattr(world, "profiler") and world.profiler is not None:
            p = world.profiler["memory_pruning"]
            p["calls"] += 1
            p["time"] += t_mem
            if t_mem > p["max"]:
                p["max"] = t_mem
                p["worst_tick"] = world.tick

    # Clean up ephemeral TickContext
    del tick_context

def find_biome_spawn(coords, world):
    """Safely finds a coordinate from candidate list or falls back to standard habitable cell."""
    if len(coords) > 0:
        idx = np.random.randint(len(coords))
        return (int(coords[idx][0]), int(coords[idx][1]))
    # Fallback to random habitable coordinate
    hab_coords = np.argwhere((world.habitability > 20) & (world.elevation >= 0.3))
    if len(hab_coords) > 0:
        idx = np.random.randint(len(hab_coords))
        return (int(hab_coords[idx][0]), int(hab_coords[idx][1]))
    return (world.height // 2, world.width // 2)

def generate_baseline_cognitive_report(world):
    """
    Aggregates cognitive profiling metrics from all agents and writes a
    Baseline Cognitive Profile report to the active experiment directory.
    """
    import os
    import numpy as np

    total_decisions = sum(a.decision_evals for a in world.agents)
    total_meaningful = sum(a.meaningful_decisions for a in world.agents)
    total_predictor_calls = sum(a.predictor_calls for a in world.agents)
    total_perception_calls = sum(a.perception_calls for a in world.agents)
    total_memories_searched = sum(a.memories_searched for a in world.agents)
    total_targets_evaluated = sum(a.targets_evaluated for a in world.agents)
    total_action_changes = sum(a.action_changes for a in world.agents)

    # Cognitive Efficiency
    cognitive_efficiency = (total_meaningful / total_decisions) * 100.0 if total_decisions > 0 else 0.0

    # Averages
    ticks = max(1, world.tick)
    n_agents = max(1, len(world.agents))
    avg_decisions_per_tick_agent = total_decisions / (ticks * n_agents)
    avg_predictor_calls_per_tick_agent = total_predictor_calls / (ticks * n_agents)
    avg_perception_calls_per_tick_agent = total_perception_calls / (ticks * n_agents)
    avg_memories_searched_per_tick_agent = total_memories_searched / (ticks * n_agents)
    avg_targets_evaluated_per_tick_agent = total_targets_evaluated / (ticks * n_agents)

    report = f"""# Project Genesis - Baseline Cognitive Profile Report

## Run Metadata
*   **Timestamp / Seed**: {getattr(world, "seed", "Unknown")}
*   **Active Tick**: {world.tick}
*   **Total Agents Spawned**: {len(world.agents)}
*   **Living Agents**: {sum(1 for a in world.agents if not a.dead)}

---

## 📊 Core Cognitive Metrics (Baseline)

| Metric | Total Count | Average per Agent per Tick | Description |
| :--- | :--- | :--- | :--- |
| **Decision Evaluations** | {total_decisions:,} | {avg_decisions_per_tick_agent:.4f} | Total times `evaluate_utility` was executed. |
| **Meaningful Decisions** | {total_meaningful:,} | - | Decisions resulting in a target coordinate or action change. |
| **Cognitive Efficiency** | **{cognitive_efficiency:.2f}%** | - | Ratio of Meaningful Decisions / Total Decisions. |
| **Predictor Calls** | {total_predictor_calls:,} | {avg_predictor_calls_per_tick_agent:.4f} | Total forward passes run in the neural predictor. |
| **Perception Calls** | {total_perception_calls:,} | {avg_perception_calls_per_tick_agent:.4f} | Total visual perception bounding box scans. |
| **Memories Searched** | {total_memories_searched:,} | {avg_memories_searched_per_tick_agent:.4f} | Total episodic and spatial knowledge nodes scanned. |
| **Targets Evaluated** | {total_targets_evaluated:,} | {avg_targets_evaluated_per_tick_agent:.4f} | Total candidate coordinates evaluated in utility loops. |
| **Action Changes** | {total_action_changes:,} | - | Frequency of agents shifting action types. |

---

## 🔍 Key Insights & Analysis
1.  **Redundant Decisions**: At a baseline cognitive efficiency of **{cognitive_efficiency:.2f}%**, approx. **{100.0 - cognitive_efficiency:.2f}%** of all decision evaluations are redundant, meaning agents re-evaluate their entire 18-action utility model and neural predictor even when their action and target remain unchanged.
2.  **Predictor Overhead**: The neural predictor network is called **{total_predictor_calls:,}** times (averaging **{avg_predictor_calls_per_tick_agent:.4f}** calls per agent per tick), representing a major source of computational load during continuous walking.
3.  **Search Overhead**: Agents scanned **{total_memories_searched:,}** knowledge nodes and evaluated **{total_targets_evaluated:,}** candidate targets, demonstrating the performance impact of unbounded spatial knowledge searches.
"""
    # Print to console
    print("\n" + "="*80)
    print("                PROJECT GENESIS - COGNITIVE PERFORMANCE PROFILE REPORT          ")
    print("="*80)
    print(f"  Total Ticks Profiled       : {world.tick}")
    print(f"  Total Decision Evaluations : {total_decisions:,}")
    print(f"  Meaningful Decisions       : {total_meaningful:,}")
    print(f"  Cognitive Efficiency       : {cognitive_efficiency:.2f}%")
    print(f"  Predictor Network Calls    : {total_predictor_calls:,} ({avg_predictor_calls_per_tick_agent:.4f}/agent/tick)")
    print(f"  Perception Calls           : {total_perception_calls:,} ({avg_perception_calls_per_tick_agent:.4f}/agent/tick)")
    print(f"  Memories Searched          : {total_memories_searched:,} ({avg_memories_searched_per_tick_agent:.4f}/agent/tick)")
    print(f"  Targets Evaluated          : {total_targets_evaluated:,} ({avg_targets_evaluated_per_tick_agent:.4f}/agent/tick)")
    print("="*80 + "\n")

    # Write to file if exp_folder exists
    exp_folder = getattr(world, "exp_folder", None)
    if exp_folder and os.path.exists(exp_folder):
        report_path = os.path.join(exp_folder, "baseline_cognitive_profile.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Baseline Cognitive Profile report successfully written to: {report_path}")

def compute_spawn_conditions(world, x: int, y: int) -> dict:
    """Computes geographic and ecological conditions around a spawn location (x, y)."""
    from world.state import BIOME_NAMES
    h_y, h_x = world.biome.shape
    
    # 1. Local values
    biome_id = int(world.biome[y, x])
    biome_name = BIOME_NAMES.get(biome_id, "Unknown")
    temp = float(world.temperature[y, x])
    elev = float(world.elevation[y, x])
    
    # 2. Nearest water
    r = 200
    y0, y1 = max(0, y-r), min(h_y, y+r)
    x0, x1 = max(0, x-r), min(h_x, x+r)
    
    water_mask = (world.lake_map[y0:y1, x0:x1] > 0.1) | \
                 (world.river_map[y0:y1, x0:x1] > 0.1) | \
                 (world.biome[y0:y1, x0:x1] == 0) | \
                 (world.biome[y0:y1, x0:x1] == 8)
                 
    water_coords = np.argwhere(water_mask)
    if len(water_coords) > 0:
        dists = np.sqrt((water_coords[:, 0] + y0 - y)**2 + (water_coords[:, 1] + x0 - x)**2)
        dist_water = float(np.min(dists))
    else:
        dist_water = float(r)
        
    # 3. Nearest food
    food_mask = (world.wildlife[y0:y1, x0:x1] > 0.1)
    food_coords = np.argwhere(food_mask)
    if len(food_coords) > 0:
        dists = np.sqrt((food_coords[:, 0] + y0 - y)**2 + (food_coords[:, 1] + x0 - x)**2)
        dist_food = float(np.min(dists))
    else:
        dist_food = float(r)
        
    # 4. Resource density in 50 radius
    r_d = 50
    y_d0, y_d1 = max(0, y-r_d), min(h_y, y+r_d)
    x_d0, x_d1 = max(0, x-r_d), min(h_x, x+r_d)
    
    avg_wildlife = float(np.mean(world.wildlife[y_d0:y_d1, x_d0:x_d1]))
    avg_fertility = float(np.mean(world.fertility[y_d0:y_d1, x_d0:x_d1]))
    avg_water = float(np.mean(world.lake_map[y_d0:y_d1, x_d0:x_d1] + world.river_map[y_d0:y_d1, x_d0:x_d1]))
    
    # 5. Biome mix in 50 radius
    slice_biomes = world.biome[y_d0:y_d1, x_d0:x_d1]
    unique_b, counts_b = np.unique(slice_biomes, return_counts=True)
    total_b = len(slice_biomes.flatten())
    biome_mix = {}
    for b_val, count in zip(unique_b, counts_b):
        b_name = BIOME_NAMES.get(int(b_val), "Unknown")
        biome_mix[b_name] = round(float(count / total_b) * 100.0, 1)
        
    return {
        "coords": [int(x), int(y)],
        "biome": biome_name,
        "dist_to_water": round(dist_water, 1),
        "dist_to_food": round(dist_food, 1),
        "temperature": round(temp, 2),
        "elevation": round(elev, 2),
        "avg_wildlife_density": round(avg_wildlife, 3),
        "avg_fertility_density": round(avg_fertility, 3),
        "avg_water_density": round(avg_water, 3),
        "biome_mix": biome_mix
    }

def run_simulation(world: WorldState, ticks: int = 10000, experiment_type: str = "default", scarcity_level: float = 1.0, save_paths: bool = True, save_epochs: bool = False, sample_interval: int = 1000, live_callback=None, callback=None, prediction_enabled: bool = True, planner_enabled: bool = True, sleep_consolidation_enabled: bool = True, long_run: bool = False, pacing_delay: float = 0.0):
    """
    Spawns 16 independent agents based on experiment rules, applies resource scarcity,
    and runs the simulation loop. Logs telemetry every 100 ticks and records sampled
    paths every 10 ticks (if save_paths is True). Compiles epoch statistics if save_epochs is True.
    """
    # 1. Apply Scarcity Level Scaling
    if scarcity_level != 1.0:
        world.wildlife = np.clip(world.wildlife * scarcity_level, 0.0, 1.0)
        world.fertility = np.clip(world.fertility * scarcity_level, 0.0, 1.0)
        world.lake_map = np.clip(world.lake_map * scarcity_level, 0.0, float('inf'))
        world.river_map = np.clip(world.river_map * scarcity_level, 0.0, float('inf'))

    # 2. Back up base grids for seasonal fluctuations and generate persistent Perlin seasonal noise
    from world.noise import fbm_noise_2d
    if not hasattr(world, "base_wildlife"):
        world.base_wildlife = world.wildlife.copy()
        world.base_fertility = world.fertility.copy()
        world.base_river_map = world.river_map.copy()
        world.base_lake_map = world.lake_map.copy()
        # Low-frequency noise mask generated once deterministically from seed + 1010
        world.seasonal_noise = fbm_noise_2d((world.height, world.width), seed=world.seed + 1010, octaves=2, base_res=(2, 2))

    # Ensure shelters registry exists
    if not hasattr(world, "shelters") or world.shelters is None:
        world.shelters = {}

    # 3. Spawning Logic — Phase 5: Colony-based with random genomes
    if not world.agents:
        if experiment_type == "environment":
            # --- Legacy environment experiment: 4 biome groups, no colonies ---
            forest_coords   = np.argwhere((world.biome == 4) | (world.biome == 7) | (world.biome == 3))
            desert_coords   = np.argwhere(world.biome == 6)
            mountain_coords = np.argwhere((world.elevation > 0.65) & (world.biome != 0) & (world.biome != 8))
            plains_coords   = np.argwhere((world.biome == 5) | (world.biome == 2))
            biomes_to_spawn = [
                ("Forest",   forest_coords),
                ("Desert",   desert_coords),
                ("Mountain", mountain_coords),
                ("Plains",   plains_coords),
            ]
            agents     = []
            agent_id   = 0
            for name, coords in biomes_to_spawn:
                spawn_loc = find_biome_spawn(coords, world)
                for _ in range(4):
                    genome = create_genome()
                    agent  = Agent(agent_id=agent_id, location=spawn_loc, genome=genome)
                    agent.spawn_biome  = name
                    agent.colony_id    = agent_id // 4
                    agent.generation   = 0
                    agent.born_tick    = world.tick
                    agent.sampled_path_history = [[
                        int(agent.location[1]), # x
                        int(agent.location[0]), # y
                        0,                      # action_id
                        float(agent.health),
                        float(agent.hunger),
                        float(agent.thirst),
                        float(agent.energy),
                        int(agent.generation)
                    ]]
                    agents.append(agent)
                    agent_id += 1
            world.agents = agents
            world.next_agent_id = agent_id
        else:
            # --- Phase 5 Colony Spawning: 4 independent colonies ---
            COLONY_NAMES   = ["Alpha", "Beta", "Gamma", "Delta"]
            COLONY_COLORS  = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
            AGENTS_PER_COL = 4

            # Spawning Mode resolver (default is "fixed")
            spawn_mode = getattr(world, "spawn_mode", "fixed")
            w_scale = world.width / 1024.0
            
            if spawn_mode == "random_anywhere":
                # True chaos: find land cells and pick 4 at random
                land_coords = np.argwhere((world.biome != 0) & (world.biome != 8))
                spots = []
                if len(land_coords) >= 4:
                    indices = np.random.choice(len(land_coords), 4, replace=False)
                    for idx in indices:
                        y, x = land_coords[idx]
                        spots.append({"x": int(x), "y": int(y)})
                else:
                    spots = [{"x": 512, "y": 512}] * 4
                    
            elif spawn_mode == "random_valid":
                # Choice land cells shuffled
                land_coords = np.argwhere((world.biome != 0) & (world.biome != 8))
                np.random.shuffle(land_coords)
                
                spots = []
                min_dist = 150.0 * w_scale
                
                for y, x in land_coords:
                    if len(spots) >= 4:
                        break
                        
                    too_close = False
                    for s in spots:
                        d = np.sqrt((s["x"] - x)**2 + (s["y"] - y)**2)
                        if d < min_dist:
                            too_close = True
                            break
                    if too_close:
                        continue
                        
                    # Check resources in 50 cell radius
                    half = 50
                    y0, y1 = max(0, y-half), min(world.height, y+half)
                    x0, x1 = max(0, x-half), min(world.width, x+half)
                    
                    has_water = (np.any(world.lake_map[y0:y1, x0:x1] > 0.1) or 
                                 np.any(world.river_map[y0:y1, x0:x1] > 0.1) or 
                                 np.any(world.biome[y0:y1, x0:x1] == 0))
                    has_food = np.any(world.wildlife[y0:y1, x0:x1] > 0.1)
                    
                    if has_water and has_food:
                        spots.append({"x": int(x), "y": int(y)})
                        
                if len(spots) < 4:
                    # Fallback to NMS fixed spots if needed
                    spots_fixed = predict_settlements(world, count=4, exclusion_radius=100.0 * w_scale)
                    for sf in spots_fixed:
                        if len(spots) >= 4:
                            break
                        spots.append(sf)
                        
            elif spawn_mode == "targeted_biome":
                # Spawn in specific biomes
                def find_biome_coords(biome_name):
                    mapping = {
                        "Desert": 6,
                        "Forest": 4,
                        "Grassland": 5,
                        "Tundra": 2,
                        "Rainforest": 7,
                        "Taiga": 3,
                    }
                    if biome_name == "Coast":
                        coords = np.argwhere((world.biome != 0) & (world.biome != 8) & (world.elevation < 0.35))
                        if len(coords) > 0:
                            return coords
                    b_id = mapping.get(biome_name, 4)
                    coords = np.argwhere(world.biome == b_id)
                    if len(coords) > 0:
                        return coords
                    return np.argwhere((world.biome != 0) & (world.biome != 8))
                
                spots = []
                targeted_list = getattr(world, "targeted_biomes", ["Desert", "Forest", "Tundra", "Grassland"])
                for b_name in targeted_list:
                    coords = find_biome_coords(b_name)
                    idx = np.random.randint(len(coords))
                    y, x = coords[idx]
                    spots.append({"x": int(x), "y": int(y)})
                    
            else: # "fixed"
                spots = predict_settlements(world, count=4, exclusion_radius=250.0 * w_scale)
                if len(spots) < 4:
                    spots = predict_settlements(world, count=4, exclusion_radius=100.0 * w_scale)
                if len(spots) < 4:
                    spots = predict_settlements(world, count=4, exclusion_radius=20.0 * w_scale)

            # Get custom colony mapping (1-based spots mapping)
            colony_mapping = getattr(world, "colony_spawn_locations", {
                "Alpha": 1, "Beta": 2, "Gamma": 3, "Delta": 4
            })
            
            world.spawn_conditions = {}
            world.colonies = []
            all_agents     = []
            agent_id       = 0

            for i in range(min(4, len(spots))):
                colony_name = COLONY_NAMES[i]
                
                # Retrieve spot based on custom mapping
                spot_num = colony_mapping.get(colony_name, i + 1)
                spot_idx = np.clip(spot_num - 1, 0, len(spots) - 1)
                spot = spots[spot_idx]
                
                # Record geographical spawn conditions for the metadata
                world.spawn_conditions[colony_name] = compute_spawn_conditions(world, spot["x"], spot["y"])
                
                colony = {
                    "id":           i,
                    "name":         colony_name,
                    "color":        COLONY_COLORS[i],
                    "founder_ids": [],
                    "stored_food":  0.0,
                    "stored_water": 0.0,
                }

                for _ in range(AGENTS_PER_COL):
                    genome = create_genome()
                    brain  = express_genome(genome)
                    agent  = Agent(agent_id=agent_id, location=(spot["y"], spot["x"]), genome=genome)

                    # All founding agents are adults (random age 20–40 years)
                    start_age     = float(np.random.uniform(20, 40))
                    agent.age     = int(start_age * 360)
                    base_max_age  = int(max(start_age + 15.0, np.random.normal(70, 10)) * 360)
                    agent.max_age = max(20 * 360, base_max_age + brain["max_age_offset"])

                    agent.colony_id    = i
                    agent.generation   = 0
                    agent.parent_ids   = None
                    agent.born_tick    = world.tick
                    agent.spawn_biome  = BIOME_NAMES.get(int(world.biome[spot["y"], spot["x"]]), "Unknown")
                    agent.sampled_path_history = [[
                        int(agent.location[1]), # x
                        int(agent.location[0]), # y
                        0,                      # action_id
                        float(agent.health),
                        float(agent.hunger),
                        float(agent.thirst),
                        float(agent.energy),
                        int(agent.generation)
                    ]]

                    colony["founder_ids"].append(agent_id)
                    all_agents.append(agent)
                    agent_id += 1

                world.colonies.append(colony)

            world.agents        = all_agents
            world.next_agent_id = agent_id

        # Archetype setup (legacy path for environment experiment only)
        for agent in world.agents:
            if agent.spawn_biome == "Unknown":
                biome_id = world.biome[agent.location[0], agent.location[1]]
                agent.spawn_biome = BIOME_NAMES.get(biome_id, "Unknown")
            # Default archetype (will be overridden by K-Means cluster at export time)
            agent.archetype = "Balanced"
            agent.base_traits = agent.traits.copy()
            if not hasattr(agent, "sampled_path_history") or not agent.sampled_path_history:
                agent.sampled_path_history = [[
                    int(agent.location[1]), # x
                    int(agent.location[0]), # y
                    0,                      # action_id
                    float(agent.health),
                    float(agent.hunger),
                    float(agent.thirst),
                    float(agent.energy),
                    int(agent.generation)
                ]]
    else:
        # Fallback Colony Setup for pre-populated agents (e.g. in unit tests)
        if not getattr(world, "colonies", None):
            COLONY_NAMES   = ["Alpha", "Beta", "Gamma", "Delta"]
            COLONY_COLORS  = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
            world.colonies = []
            for i in range(4):
                colony = {
                    "id":           i,
                    "name":         COLONY_NAMES[i],
                    "color":        COLONY_COLORS[i],
                    "founder_ids": [],
                    "stored_food":  0.0,
                    "stored_water": 0.0,
                }
                world.colonies.append(colony)
            for idx, agent in enumerate(world.agents):
                c_id = idx % 4
                agent.colony_id = c_id
                world.colonies[c_id]["founder_ids"].append(agent.id)

    # Propagate prediction flag and ablation settings to all agents
    for agent in world.agents:
        agent.prediction_enabled = prediction_enabled
        agent.ablation = getattr(world, "ablation", {})

    # Propagate Phase 7 cognition architecture flags to world (read by simulate_agent_tick)
    world.planner_enabled = planner_enabled
    world.sleep_consolidation_enabled = sleep_consolidation_enabled

    # Initialize Climate Epoch Engine
    epoch_mode = getattr(world, "climate_epoch_mode", "legacy")
    from world.climate_epochs import ClimateEpochEngine
    engine = ClimateEpochEngine(mode=epoch_mode, seed=world.seed)
    if hasattr(world, "climate_epoch_state") and world.climate_epoch_state is not None:
        engine.load_state(world.climate_epoch_state)
    else:
        world.climate_epoch_state = engine.save_state()

    # Initialize Evolution Journal
    from world.evolution_journal import EvolutionJournal
    journal = None
    if getattr(world, "exp_folder", None):
        journal = EvolutionJournal(world.exp_folder)
        if hasattr(world, "evolution_journal_history") and world.evolution_journal_history is not None:
            journal.history_records = world.evolution_journal_history
            if journal.history_records:
                journal.baseline_genes = np.array(journal.history_records[0]["gene_means"])

    print(f"Spawning complete. Total agents: {len(world.agents)}")
    
    # Print initial census
    alive_count = sum(1 for a in world.agents if not a.dead)
    print(f"Tick {world.tick}: {alive_count}/{len(world.agents)} agents alive.")
    
    last_printed_history_idx = len(world.history)
     # Initialize high-precision profiler structure
    world.profiler = {
        "perception":           {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "decision":             {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "movement":             {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "drives_relationships": {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "emotion_clock":        {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "relationship_clock":   {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "motivation_clock":     {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "neural_predictor":     {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "memory_pruning":       {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "reproduction":         {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "ecology":              {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "logging":              {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1},
        "callback":             {"calls": 0, "time": 0.0, "max": 0.0, "worst_tick": -1}
    }

    epoch_stats = []
    prev_discoveries_sum = sum(a.discoveries_count for a in world.agents) if world.agents else 0
    prev_nodes_added_sum = sum(a.nodes_added_count for a in world.agents) if world.agents else 0
    prev_nodes_removed_sum = sum(a.nodes_removed_count for a in world.agents) if world.agents else 0
    prev_failed_water = sum(a.failed_water_visits for a in world.agents) if world.agents else 0
    prev_failed_food = sum(a.failed_food_visits for a in world.agents) if world.agents else 0

    # ==================================================================
    # SCIENTIFIC INSTRUMENTATION: Pre-loop state initialization
    # ==================================================================
    import os as _os

    # --- 1. Civilization Milestone tracking state ---
    milestone_state = {
        "generations":          set(),    # which gen milestones (3,5,10,20,50) already logged
        "first_reproduction":   False,
        "grandchild":           False,    # first Gen >= 2 born
        "stone_shelter":        False,    # first shelter_level == 3
        "complete_shelter_net": False,    # all living agents have shelter >= 1
        "pop_100":              False,
        "pop_below_50":         False,
        "pop_recovered_80":     False,
        "pop_peak":             0,        # highest alive count ever reached
        "pop_peak_tick":        0,
        "oldest_logged_age":    0,        # highest age_ticks logged for oldest living
        "max_lineage_logged":   0,        # highest generation milestone logged for lineage
    }

    # --- 2. Plateau detection rolling window ---
    plateau_samples = []   # list of {tick, alive, births, deaths, avg_hunger, avg_thirst}
    plateau_prev_births = int(world.total_births)
    plateau_prev_deaths = int(getattr(world, "total_deaths", 0))
    plateau_status = "Initializing"

    # --- 3. Performance log CSV ---
    exp_folder = getattr(world, "exp_folder", None)
    perf_log_path = _os.path.join(exp_folder, "performance_log.csv") if exp_folder else None
    if perf_log_path and not _os.path.exists(perf_log_path):
        with open(perf_log_path, "w", newline="", encoding="utf-8") as _f:
            _f.write(
                "tick,year,elapsed_s,avg_ms_per_tick,population,alive,births,deaths,"
                "avg_age_ticks,max_generation,avg_memories,avg_concepts,avg_procedures,"
                "avg_pred_confidence,avg_shelter_level,avg_queue_length,memory_mb,"
                "perc_ms_avg,decision_ms_avg,movement_ms_avg,ecology_ms_avg,repro_ms_avg\n"
            )

    # --- 4. Agent census CSV header ---
    census_log_path = _os.path.join(exp_folder, "agent_census.csv") if exp_folder else None
    if census_log_path and not _os.path.exists(census_log_path):
        with open(census_log_path, "w", newline="", encoding="utf-8") as _f:
            _f.write(
                "snapshot_tick,id,generation,colony,age_ticks,health,children,"
                "concept_count,procedure_count,pred_accuracy,max_radius,"
                "shelter_level,alive,cause_of_death\n"
            )

    # Cache alive agents to avoid looping over dead ones or repeatedly filtering world.agents (O(N) -> O(1))
    world.alive_agents = [a for a in world.agents if not a.dead]
    world.alive_agents_map = {a.id: a for a in world.alive_agents}

    perf_run_start = time.perf_counter()

    ticks_run = 0
    for _ in range(ticks):
        # Check emergency stop request
        if getattr(world, "stop_requested", False):
            world.history.append(f"[EMERGENCY STOP] Simulation terminated by user at tick {world.tick}.")
            print(f"\n⚠️ [EMERGENCY STOP] Terminating simulation loop gracefully at tick {world.tick} and saving data...")
            # Save a full resumable checkpoint automatically on stop
            _full_cp_config = getattr(world, "_full_checkpoint_config", None)
            _full_cp_folder = getattr(world, "exp_folder", ".")
            if _full_cp_config is not None:
                try:
                    from world.checkpoint_io import save_full_checkpoint as _sfc
                    _cp_path = _os.path.join(_full_cp_folder, f"full_checkpoint_{world.tick}.json")
                    _sfc(world, _full_cp_config, _cp_path)
                except Exception as _cpe:
                    print(f"  Warning: full checkpoint save on stop failed: {_cpe}")
            break

        world.tick += 1
        ticks_run += 1
        world.temp_cos_factor = float(np.cos((((world.tick % 360) - 180.0) / 180.0) * np.pi))

        # Reset per-tick combat pair-dedup set (used by territorial encounter logging)
        world._disputes_this_tick = set()


        # --- ECOLOGY PASS (Timed) ---
        t_eco_start = time.perf_counter()
        
        # Build O(n) binned spatial grid for this tick
        bin_size = 32
        grid_w = world.width // bin_size
        grid_h = world.height // bin_size
        spatial_grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
        for agent in world.alive_agents:
            bx = min(grid_w - 1, max(0, agent.location[1] // bin_size))
            by = min(grid_h - 1, max(0, agent.location[0] // bin_size))
            spatial_grid[by][bx].append(agent)
        world.spatial_grid = spatial_grid
        
        # Weathering decay of shelters: all shelters in the registry decay by 0.1 per tick
        if hasattr(world, "shelters") and world.shelters:
            to_remove = []
            for loc, sh in list(world.shelters.items()):
                sh["durability"] -= 0.1
                owner_id = sh["owner_id"]
                owner_agent = None
                if owner_id is not None:
                    owner_agent = world.alive_agents_map.get(owner_id)
                if owner_agent is not None:
                    owner_agent.shelter_durability = max(0.0, sh["durability"])
                else:
                    sh["owner_id"] = None
                
                if sh["durability"] <= 0.0:
                    to_remove.append(loc)
                    if owner_agent is not None:
                        owner_agent.shelter_location = None
                        owner_agent.shelter_level = 0
                        owner_agent.shelter_durability = 0.0
            for loc in to_remove:
                if loc in world.shelters:
                    del world.shelters[loc]
        
        # --- 1. Climate Epochs and Global Event Modifiers ---
        year = world.tick // 360
        day = world.tick % 360
        
        if getattr(world, "disasters_enabled", True):
            def history_callback(msg):
                world.history.append(msg)
            def timeline_callback(etype, msg):
                world.events_timeline.append({
                    "tick": int(world.tick),
                    "year": int(year),
                    "event": etype,
                    "description": msg
                })
            
            temp_off, water_m, food_m, seasonal_m = engine.tick(
                world.tick,
                history_callback=history_callback,
                timeline_callback=timeline_callback
            )
            world.global_temp_offset = temp_off
            world.global_water_multiplier = water_m
            world.global_food_multiplier = food_m
            world.global_seasonal_multiplier = seasonal_m
            world.climate_epoch_state = engine.save_state()
        else:
            world.global_temp_offset = 0.0
            world.global_water_multiplier = 1.0
            world.global_food_multiplier = 1.0
            world.global_seasonal_multiplier = 1.0
            engine.current_epoch_name = "Temperate"
            world.climate_epoch_state = engine.save_state()

        # Apply seasonal changes vectorially (yearly cycle = 360 ticks)
        # Oscillates globally between 0.3 (Peak Winter) and 1.0 (Peak Summer)
        seasonal_mult = getattr(world, "global_seasonal_multiplier", 1.0)
        global_seasonal_factor = 0.65 + (0.35 * seasonal_mult) * np.cos(((day - 180.0) / 180.0) * np.pi)
        
        # Performance Optimization: Only perform heavy 1024x1024 array multiplications once every 30 ticks
        if world.tick % 30 == 1 or _ == 0:
            # Compute local seasonal factors using persistent Perlin noise
            local_factor = global_seasonal_factor * (0.7 + 0.6 * world.seasonal_noise)
            
            food_mult = getattr(world, "global_food_multiplier", 1.0)
            water_mult = getattr(world, "global_water_multiplier", 1.0)
            
            # Modulate dynamic resources
            world.wildlife = np.clip(world.base_wildlife * local_factor * food_mult, 0.0, 1.0)
            world.fertility = np.clip(world.base_fertility * local_factor * food_mult, 0.0, 1.0)
            world.river_map = world.base_river_map * local_factor * water_mult
            # Lakes remain stable permanent anchors but are scaled during drought
            world.lake_map = world.base_lake_map * water_mult
            
        t_eco = (time.perf_counter() - t_eco_start) * 1000.0
        if hasattr(world, "profiler") and world.profiler is not None:
            p = world.profiler["ecology"]
            p["calls"] += 1
            p["time"] += t_eco
            if t_eco > p["max"]:
                p["max"] = t_eco
                p["worst_tick"] = world.tick
        
        # Simulate each agent
        for agent in list(world.alive_agents):
            if not agent.dead:
                simulate_agent_tick(agent, world)
                if agent.dead:
                    # Remove from alive caches
                    if agent in world.alive_agents:
                        world.alive_agents.remove(agent)
                    if agent.id in world.alive_agents_map:
                        del world.alive_agents_map[agent.id]
                    # Trigger grief on nearby living agents who knew the deceased
                    cy, cx = agent.location
                    candidates = world.query_agents(agent.location, 30.0, alive_only=True)
                    for other in candidates:
                        if other.id != agent.id:
                            if agent.id in other.known_agents:
                                oy, ox = other.location
                                dist = float(np.sqrt((oy - cy)**2 + (ox - cx)**2))
                                if dist <= other.vision_radius:
                                    trigger_grief(other, agent.id, world)

        # --- REPRODUCTION PASS (Timed, Spatial Hashing Optimized) ---
        t_rep_start = time.perf_counter()
        reproduced_ids = set()
        new_agents     = []
        if getattr(world, "reproduction_enabled", True):
            for agent in world.alive_agents:
                if agent.id in reproduced_ids:
                    continue
                target_loc = getattr(agent, "_wants_to_reproduce_with", None)
                if target_loc is None:
                    continue
                agent._wants_to_reproduce_with = None
                
                # Find matching willing mate at the target location in the same colony
                # Optimize by querying candidates at the target coordinate
                candidates = world.query_agents(target_loc, 1.0, alive_only=True)
                
                mate = next(
                    (a for a in candidates
                     if a.id != agent.id
                     and a.id not in reproduced_ids
                     and getattr(a, "colony_id", -1) == getattr(agent, "colony_id", -1)
                     and a.location == target_loc
                     and getattr(a, "_wants_to_reproduce_with", None) is not None),
                    None,
                )
                if mate is not None:
                    mate._wants_to_reproduce_with = None
                    mut_rate = getattr(world, "mutation_rate", 0.05)
                    child = attempt_reproduce(agent, mate, world, world.next_agent_id, mutation_rate=mut_rate)
                    if child is not None:
                        new_agents.append(child)
                        world.next_agent_id += 1
                        
                        # Terminal funnel outcomes updates (Phase 10)
                        agent.repro_terminal_outcome = "birth"
                        mate.repro_terminal_outcome = "birth"
                        world.telemetry.repro_mutual_will_ticks += 2
                        world.telemetry.record_birth(child.generation)
                    reproduced_ids.add(agent.id)
                    reproduced_ids.add(mate.id)
                else:
                    # Coordination failure logic (Phase 10)
                    partners_at_loc = [a for a in candidates if a.id != agent.id and getattr(a, "colony_id", -1) == getattr(agent, "colony_id", -1) and a.location == target_loc]
                    if not partners_at_loc:
                        agent.repro_terminal_outcome = "no_mate_nearby"
                    else:
                        agent.repro_terminal_outcome = "mate_unwilling"

            for child in new_agents:
                world.agents.append(child)
                world.alive_agents.append(child)
                world.alive_agents_map[child.id] = child
                
        # 3. Telemetry commit for this tick (Phase 10)
        for agent in world.alive_agents:
            if getattr(agent, "repro_terminal_outcome", None) is not None:
                world.telemetry.record_repro_terminal(agent.repro_terminal_outcome)
                agent.repro_terminal_outcome = None

        # Collect Resource Timeline statistics every 500 ticks or at tick 1 (Phase 10)
        if world.tick % 500 == 0 or world.tick == 1:
            living = [a for a in world.agents if not a.dead]
            pop = len(world.agents)
            alive = len(living)
            births = int(world.total_births)
            deaths = int(getattr(world, "total_deaths", 0))
            
            stored_food = 0.0
            stored_water = 0.0
            if hasattr(world, "colonies") and world.colonies:
                for col in world.colonies:
                    stored_food += col.get("stored_food", 0.0)
                    stored_water += col.get("stored_water", 0.0)
            
            mean_hunger = float(np.mean([a.hunger for a in living])) if alive > 0 else 0.0
            mean_thirst = float(np.mean([a.thirst for a in living])) if alive > 0 else 0.0
            mean_shelter = float(np.mean([a.shelter_durability for a in living])) if alive > 0 else 0.0
            
            # Dominant emotion mode
            emotions = [emotional_label(a) for a in living if hasattr(a, "drives")]
            if emotions:
                from collections import Counter
                dom_emotion = Counter(emotions).most_common(1)[0][0]
            else:
                dom_emotion = "Neutral"
                
            # Novelty: Mean prediction confidence
            avg_novelty = float(np.mean([a.prediction_confidence for a in living])) if alive > 0 else 0.0
            
            # Average trust: Mean pairwise relationship modifier
            trust_sum = 0.0
            trust_count = 0
            for i, a in enumerate(living):
                for b in living[i+1:]:
                    trust_sum += a.get_social_modifier(b.id, world.tick)
                    trust_count += 1
            avg_trust = trust_sum / max(1, trust_count) if trust_count > 0 else 1.0
            
            # Climate: Current active epoch
            climate = engine.current_epoch_name
            
            # Major event: Latest milestone/disaster logged in the last 500 ticks
            recent_events = [ev for ev in getattr(world, "events_timeline", []) if world.tick - 500 < ev["tick"] <= world.tick]
            if recent_events:
                ev_type = recent_events[-1].get("type") or recent_events[-1].get("event") or "Event"
                major_event = ev_type + ": " + recent_events[-1].get("description", "")
                if len(major_event) > 60:
                    major_event = major_event[:57] + "..."
            else:
                major_event = "None"
                
            # Gene averages
            from world.agents.genetics import GENE_NAMES
            gene_averages = {}
            if living:
                genes_matrix = np.array([a.genome.genes for a in living])
                for idx, name in enumerate(GENE_NAMES):
                    gene_averages[name] = float(np.mean(genes_matrix[:, idx]))
            else:
                for name in GENE_NAMES:
                    gene_averages[name] = 0.5
            
            world.telemetry.resource_timeline.append({
                "tick": world.tick,
                "population": pop,
                "alive": alive,
                "births": births,
                "deaths": deaths,
                "stored_food": stored_food,
                "stored_water": stored_water,
                "mean_hunger": mean_hunger,
                "mean_thirst": mean_thirst,
                "mean_shelter": mean_shelter,
                "dominant_emotion": dom_emotion,
                "novelty": avg_novelty,
                "avg_trust": avg_trust,
                "climate": climate,
                "major_event": major_event,
                "gene_averages": gene_averages
            })
                
        t_rep = (time.perf_counter() - t_rep_start) * 1000.0
        if hasattr(world, "profiler") and world.profiler is not None:
            p = world.profiler["reproduction"]
            p["calls"] += 1
            p["time"] += t_rep
            if t_rep > p["max"]:
                p["max"] = t_rep
                p["worst_tick"] = world.tick

        # --- Extinction Early Exit check ---
        alive_count = sum(1 for a in world.agents if not a.dead)
        if alive_count == 0:
            world.history.append(f"[GLOBAL EXTINCTION] Tick {world.tick}: All agents have died!")
            print(f"\n*** [GLOBAL EXTINCTION] Tick {world.tick}: All agents have died! ***\n")
            
            extinction_year = world.tick // 360
            generations_survived = world.generation_number
            total_births = world.total_births
            
            longest_lineage = 0
            for a in world.agents:
                if a.generation > longest_lineage:
                    longest_lineage = a.generation
                    
            last_agent = None
            if world.agents:
                sorted_agents = sorted(world.agents, key=lambda a: a.ticks_survived, reverse=True)
                last_agent = sorted_agents[0]
                
            dominant_genome_str = "None"
            if last_agent is not None and hasattr(last_agent, "genome"):
                dominant_genome_str = repr(last_agent.genome)
                
            report_msg = (
                f"[EXTINCTION REPORT] Extinction Year: {extinction_year}, "
                f"Generations Survived: {generations_survived}, "
                f"Total Births: {total_births}, "
                f"Longest Lineage: Gen {longest_lineage}, "
                f"Dominant Genome: {dominant_genome_str}"
            )
            world.history.append(report_msg)
            print(f"*** {report_msg} ***\n")
            
            if save_epochs:
                per_colony = [0] * len(getattr(world, "colonies", []))
                world.population_history.append({
                    "tick": int(world.tick),
                    "total": 0,
                    "per_colony": per_colony,
                })
                epoch_num = world.tick // sample_interval
                epoch_stats.append({
                    "epoch": int(epoch_num),
                    "alive": 0,
                    "avg_discoveries": 0.0,
                    "novelty_rate": 0,
                    "avg_radius": 0.0,
                    "avg_knowledge": 0.0,
                    "avg_accuracy": 0.0,
                    "knowledge_churn": 0,
                    "failed_water": 0,
                    "failed_food": 0,
                    "avg_hunger": 0.0,
                    "avg_thirst": 0.0,
                    "seasonal_factor": 0.0,
                    "avg_pred_accuracy": 0.0,
                    "avg_pred_gain": 0.0,
                    "rediscoveries": 0,
                    "avg_fat_reserves": 0.0,
                    "avg_muscle_mass": 0.0,
                    "avg_injury_level": 0.0,
                    "avg_starvation_dmg": 0.0,
                    "avg_dehydration_dmg": 0.0,
                    "avg_exposure_dmg": 0.0,
                    "avg_injury_dmg": 0.0,
                    "avg_age_dmg": 0.0,
                    "deaths_starvation": 0,
                    "deaths_dehydration": 0,
                    "deaths_exposure": 0,
                    "deaths_old_age": 0,
                    "deaths_injury": 0,
                    "deaths_unknown": 0,
                    "total_births": int(world.total_births),
                    "total_deaths": int(world.total_deaths),
                    "avg_generation": 0.0,
                    "max_generation": longest_lineage,
                    "genetic_diversity": 0.0,
                    "per_colony_alive": per_colony,
                    "_births_prev": int(world.total_births),
                    "_deaths_prev": int(world.total_deaths),
                })
            break

        # --- Extinction Tracking: check if any colony just went to 0 alive adults ---
        if new_agents or world.tick % 100 == 0:
            colony_alive = {c["id"]: 0 for c in getattr(world, "colonies", [])}
            for a in world.agents:
                if not a.dead and a.life_stage in ("Adult", "Elder"):
                    cid = getattr(a, "colony_id", -1)
                    if cid in colony_alive:
                        colony_alive[cid] += 1
            for colony in getattr(world, "colonies", []):
                cid = colony["id"]
                if colony_alive.get(cid, 0) == 0:
                    # Check that we haven't already logged this extinction
                    already_logged = colony.get("extinction_tick") is not None
                    if not already_logged:
                        colony["extinction_tick"] = int(world.tick)
                        world.extinction_events.append({
                            "tick":        world.tick,
                            "colony_id":   cid,
                            "colony_name": colony["name"],
                        })
                        world.history.append(
                            f"[EXTINCTION] Tick {world.tick}: Colony {colony['name']} has gone extinct!"
                        )
                        if not hasattr(world, "events_timeline") or world.events_timeline is None:
                            world.events_timeline = []
                        world.events_timeline.append({
                            "tick": int(world.tick),
                            "year": int(world.tick // 360),
                            "day": int(world.tick % 360),
                            "type": "Extinction",
                            "description": f"Colony {colony['name']} has gone extinct (all members have died).",
                            "metadata": {
                                "colony_id": int(cid),
                                "colony_name": colony["name"]
                            }
                        })
            

        # ==================================================================
        # SCIENTIFIC INSTRUMENTATION BLOCK
        # Runs every tick (milestones) or at fixed intervals (plateau/perf/census)
        # No agent behavior is modified here — pure observation.
        # ==================================================================

        # ── A. Civilization Milestones (checked every tick, cheap) ──────────
        def _log_milestone(tag, desc, metadata=None):
            """Helper: log a milestone to world history, events_timeline, and stdout."""
            msg = f"[MILESTONE] Tick {world.tick} (Year {world.tick // 360}, Day {world.tick % 360}): {desc}"
            world.history.append(msg)
            entry = {
                "tick":        int(world.tick),
                "year":        int(world.tick // 360),
                "day":         int(world.tick % 360),
                "type":        "Milestone",
                "description": desc,
            }
            if metadata:
                entry["metadata"] = metadata
            world.events_timeline.append(entry)
            print(f"  ⭐ {msg}")

        alive_now = sum(1 for a in world.agents if not a.dead)

        # Population peak tracking
        if alive_now > milestone_state["pop_peak"]:
            milestone_state["pop_peak"] = alive_now
            milestone_state["pop_peak_tick"] = world.tick

        # Generation milestones
        current_max_gen = world.generation_number
        for gen_thresh in (3, 5, 10, 20, 50):
            if current_max_gen >= gen_thresh and gen_thresh not in milestone_state["generations"]:
                milestone_state["generations"].add(gen_thresh)
                _log_milestone(f"gen_{gen_thresh}", f"Generation {gen_thresh} reached! Max generation in history: {current_max_gen}.",
                               {"generation": gen_thresh})

        # First reproduction
        if not milestone_state["first_reproduction"] and world.total_births > 0:
            milestone_state["first_reproduction"] = True
            first_born = next((a for a in world.agents if getattr(a, "generation", 0) == 1), None)
            parent_str = f"(parents #{first_born.parent_ids[0]} x #{first_born.parent_ids[1]})" if first_born and first_born.parent_ids else ""
            _log_milestone("first_reproduction", f"First successful reproduction! {parent_str} Birth #{world.total_births}.",
                           {"total_births": int(world.total_births)})

        # First grandchild (Gen >= 2)
        if not milestone_state["grandchild"] and current_max_gen >= 2:
            milestone_state["grandchild"] = True
            _log_milestone("first_grandchild", "First grandchild born — the civilization now spans three generations!",
                           {"generation": 2})

        # First Stone shelter (level == 3)
        if not milestone_state["stone_shelter"]:
            stone_builder = next((a for a in world.agents if not a.dead and a.shelter_level >= 3), None)
            if stone_builder is not None:
                milestone_state["stone_shelter"] = True
                col_name = "Unknown"
                if hasattr(world, "colonies") and 0 <= stone_builder.colony_id < len(world.colonies):
                    col_name = world.colonies[stone_builder.colony_id].get("name", col_name)
                _log_milestone("first_stone_shelter",
                               f"First Stone Shelter constructed by Agent #{stone_builder.id} (Colony {col_name})!",
                               {"agent_id": int(stone_builder.id), "colony": col_name})

        # Complete shelter network (every living agent has shelter >= 1)
        if not milestone_state["complete_shelter_net"] and alive_now >= 4:
            all_sheltered = all(a.shelter_level >= 1 for a in world.agents if not a.dead)
            if all_sheltered:
                milestone_state["complete_shelter_net"] = True
                _log_milestone("complete_shelter_network",
                               f"Complete shelter network achieved — all {alive_now} living agents have shelter!",
                               {"population": alive_now})

        # Population milestones
        if not milestone_state["pop_100"] and alive_now >= 100:
            milestone_state["pop_100"] = True
            _log_milestone("population_100", f"Population reached 100! ({alive_now} alive)", {"alive": alive_now})

        if not milestone_state["pop_below_50"] and alive_now < 50 and world.tick > 500:
            milestone_state["pop_below_50"] = True
            milestone_state["pop_recovered_80"] = False  # re-arm recovery trigger
            _log_milestone("population_below_50",
                           f"Population crisis — dropped below 50 ({alive_now} alive). Survival at risk.",
                           {"alive": alive_now})
        elif milestone_state["pop_below_50"] and not milestone_state["pop_recovered_80"] and alive_now >= 80:
            milestone_state["pop_recovered_80"] = True
            milestone_state["pop_below_50"] = False  # re-arm crisis trigger
            _log_milestone("population_recovered_80",
                           f"Population recovery — bounced back above 80 ({alive_now} alive)!",
                           {"alive": alive_now})

        # Population peak milestone (log whenever a new peak is set past 20)
        if milestone_state["pop_peak"] >= 20 and alive_now == milestone_state["pop_peak"] and world.tick == milestone_state["pop_peak_tick"]:
            # Only log distinct peaks divisible by 10 to avoid spam
            if milestone_state["pop_peak"] % 10 == 0:
                _log_milestone("population_peak",
                               f"New population peak reached: {milestone_state['pop_peak']} alive at tick {world.tick}!",
                               {"peak": milestone_state["pop_peak"]})

        # Oldest living agent (every 50,000 ticks)
        if world.tick % 50000 == 0 and world.agents:
            oldest = max((a for a in world.agents if not a.dead), key=lambda a: a.age, default=None)
            if oldest and oldest.age > milestone_state["oldest_logged_age"]:
                milestone_state["oldest_logged_age"] = oldest.age
                years = oldest.age // 360
                _log_milestone("oldest_living",
                               f"Oldest living agent: #{oldest.id} aged {years} years ({oldest.age} ticks), Gen {oldest.generation}.",
                               {"agent_id": int(oldest.id), "age_years": years})

        # Longest lineage (track max generation alive)
        if world.tick % 10000 == 0:
            max_gen_alive = max((a.generation for a in world.agents if not a.dead), default=0)
            if max_gen_alive > milestone_state["max_lineage_logged"]:
                milestone_state["max_lineage_logged"] = max_gen_alive
                _log_milestone("longest_lineage",
                               f"New longest living lineage: Generation {max_gen_alive} agents are alive!",
                               {"generation": max_gen_alive})

        # ── B. Plateau Detection (every 5,000 ticks) ───────────────────────
        if world.tick % 5000 == 0 and world.tick >= 5000:
            cur_births = int(world.total_births)
            cur_deaths = int(getattr(world, "total_deaths", 0))
            births_delta = cur_births - plateau_prev_births
            deaths_delta = cur_deaths - plateau_prev_deaths
            plateau_prev_births = cur_births
            plateau_prev_deaths = cur_deaths

            alive_agents_plateau = [a for a in world.agents if not a.dead]
            avg_hunger_p = float(np.mean([a.hunger for a in alive_agents_plateau])) if alive_agents_plateau else 0.0
            avg_thirst_p = float(np.mean([a.thirst for a in alive_agents_plateau])) if alive_agents_plateau else 0.0

            plateau_samples.append({
                "tick": int(world.tick),
                "alive": int(alive_now),
                "births": births_delta,
                "deaths": deaths_delta,
                "avg_hunger": round(avg_hunger_p, 2),
                "avg_thirst": round(avg_thirst_p, 2),
            })

            # Keep only last 4 samples (= 20,000 tick window)
            if len(plateau_samples) > 4:
                plateau_samples.pop(0)

            if len(plateau_samples) >= 4:
                pop_vals    = [s["alive"]   for s in plateau_samples]
                birth_vals  = [s["births"]  for s in plateau_samples]
                death_vals  = [s["deaths"]  for s in plateau_samples]
                hunger_vals = [s["avg_hunger"] for s in plateau_samples]
                thirst_vals = [s["avg_thirst"] for s in plateau_samples]

                pop_mean  = np.mean(pop_vals)
                pop_std   = np.std(pop_vals)
                b_mean    = np.mean(birth_vals)
                d_mean    = np.mean(death_vals)

                # Stability scores (0–100%)
                pop_stability = max(0.0, 100.0 * (1.0 - pop_std / (pop_mean + 1.0)))
                bd_balance    = max(0.0, 100.0 * (1.0 - abs(b_mean - d_mean) / max(b_mean + d_mean, 1.0)))
                food_stab     = max(0.0, 100.0 * (1.0 - float(np.std(hunger_vals)) / (float(np.mean(hunger_vals)) + 1.0)))
                water_stab    = max(0.0, 100.0 * (1.0 - float(np.std(thirst_vals)) / (float(np.mean(thirst_vals)) + 1.0)))

                # Overall plateau classification
                if pop_stability >= 90 and bd_balance >= 85:
                    plateau_status = "Stable"
                elif np.polyfit(range(4), pop_vals, 1)[0] > 0.5:
                    plateau_status = "Growing"
                elif np.polyfit(range(4), pop_vals, 1)[0] < -0.5:
                    plateau_status = "Declining"
                else:
                    plateau_status = "Oscillating"

                plateau_desc = (
                    f"Ecosystem Status: {plateau_status} | "
                    f"Pop Stability: {pop_stability:.0f}% | B/D Balance: {bd_balance:.0f}% | "
                    f"Food Stability: {food_stab:.0f}% | Water Stability: {water_stab:.0f}%"
                )
                msg_p = f"[PLATEAU] Tick {world.tick}: {plateau_desc}"
                world.history.append(msg_p)
                world.events_timeline.append({
                    "tick": int(world.tick),
                    "year": int(world.tick // 360),
                    "day":  int(world.tick % 360),
                    "type": "PlateauStatus",
                    "description": plateau_desc,
                    "metadata": {
                        "status":          plateau_status,
                        "pop_stability":   round(pop_stability, 1),
                        "bd_balance":      round(bd_balance, 1),
                        "food_stability":  round(food_stab, 1),
                        "water_stability": round(water_stab, 1),
                        "pop_mean":        round(float(pop_mean), 1),
                        "births_mean":     round(float(b_mean), 1),
                        "deaths_mean":     round(float(d_mean), 1),
                    }
                })
                print(f"  📊 {msg_p}")

        # ── C. Performance Log + Agent Census (every 10,000 ticks) ──────────
        if world.tick % 10000 == 0 and world.tick >= 10000:
            elapsed_s   = time.perf_counter() - perf_run_start
            ticks_done  = world.tick
            avg_ms_tick = (elapsed_s * 1000.0) / max(ticks_done, 1)

            alive_agents_perf = [a for a in world.agents if not a.dead]
            n_alive = len(alive_agents_perf)

            avg_age    = float(np.mean([a.age for a in alive_agents_perf])) if alive_agents_perf else 0.0
            max_gen_p  = int(max((a.generation for a in world.agents), default=0))
            avg_mem    = float(np.mean([len(a.episodic_memory) for a in alive_agents_perf])) if alive_agents_perf else 0.0
            avg_conc   = float(np.mean([
                sum(len(lst) for lst in a.concepts.values()) if hasattr(a, "concepts") else 0
                for a in alive_agents_perf
            ])) if alive_agents_perf else 0.0
            avg_proc   = float(np.mean([len(a.procedures) if hasattr(a, "procedures") else 0
                                        for a in alive_agents_perf])) if alive_agents_perf else 0.0
            avg_pred_conf = float(np.mean([
                a.prediction_successes / a.prediction_attempts if a.prediction_attempts > 0 else 1.0
                for a in alive_agents_perf
            ])) if alive_agents_perf else 1.0
            avg_shelt  = float(np.mean([a.shelter_level for a in alive_agents_perf])) if alive_agents_perf else 0.0
            avg_queue  = float(np.mean([len(a.action_queue) if hasattr(a, "action_queue") else 0
                                        for a in alive_agents_perf])) if alive_agents_perf else 0.0

            # Memory usage (psutil optional)
            mem_mb_str = "N/A"
            try:
                import psutil
                mem_mb_str = f"{psutil.Process().memory_info().rss / 1e6:.1f}"
            except ImportError:
                pass

            # Profiler averages
            def _pavg(key):
                p = world.profiler.get(key, {})
                c = p.get("calls", 0)
                return round(p.get("time", 0.0) / c, 4) if c > 0 else 0.0

            if perf_log_path:
                with open(perf_log_path, "a", newline="", encoding="utf-8") as _f:
                    _f.write(
                        f"{world.tick},{world.tick // 360},{elapsed_s:.1f},{avg_ms_tick:.3f},"
                        f"{len(world.agents)},{n_alive},{world.total_births},{getattr(world, 'total_deaths', 0)},"
                        f"{avg_age:.0f},{max_gen_p},{avg_mem:.1f},{avg_conc:.1f},{avg_proc:.1f},"
                        f"{avg_pred_conf:.4f},{avg_shelt:.2f},{avg_queue:.2f},{mem_mb_str},"
                        f"{_pavg('perception')},{_pavg('decision')},{_pavg('movement')},{_pavg('ecology')},{_pavg('reproduction')}\n"
                    )

            # Agent census snapshot
            if census_log_path:
                with open(census_log_path, "a", newline="", encoding="utf-8") as _f:
                    for a in world.agents:
                        conc_cnt = sum(len(lst) for lst in a.concepts.values()) if hasattr(a, "concepts") else 0
                        proc_cnt = len(a.procedures) if hasattr(a, "procedures") else 0
                        pred_acc = a.prediction_successes / a.prediction_attempts if a.prediction_attempts > 0 else 1.0
                        col_name = "Unknown"
                        if hasattr(world, "colonies") and 0 <= getattr(a, "colony_id", -1) < len(world.colonies):
                            col_name = world.colonies[a.colony_id].get("name", "Unknown")
                        _f.write(
                            f"{world.tick},{a.id},{a.generation},{col_name},{a.age},"
                            f"{round(a.health, 1)},{len(a.children_ids)},"
                            f"{conc_cnt},{proc_cnt},{round(pred_acc, 4)},{round(float(getattr(a, 'max_radius', 0.0)), 1)},"
                            f"{a.shelter_level},{int(not a.dead)},{getattr(a, 'primary_cause', 'None') or 'None'}\n"
                        )

        # Sample coordinates for timeline playback every 10 ticks
        if save_paths and world.tick % 10 == 0:
            for agent in world.agents:
                if not agent.dead:
                    action_id = ACTION_TO_ID.get(getattr(agent, "current_action", "Idle"), 0)
                    agent.sampled_path_history.append([
                        int(agent.location[1]), # x (col)
                        int(agent.location[0]), # y (row)
                        int(action_id),
                        round(float(agent.health), 1),
                        round(float(agent.hunger), 1),
                        round(float(agent.thirst), 1),
                        round(float(agent.energy), 1),
                        int(agent.generation)
                    ])
                
        # Sample population history every sample_interval ticks
        if save_epochs and (world.tick % sample_interval == 0):
            per_colony = []
            for colony in getattr(world, "colonies", []):
                cnt = sum(1 for a in world.agents if not a.dead and getattr(a, "colony_id", -1) == colony["id"])
                per_colony.append(cnt)
            world.population_history.append({
                "tick":       int(world.tick),
                "total":      int(sum(1 for a in world.agents if not a.dead)),
                "per_colony": per_colony,
            })
            # Genetic history snapshot — expanded with median, min, max, colony diversity
            living_genomes = [a.genome for a in world.agents if not a.dead]
            alive_agents_for_gen = [a for a in world.agents if not a.dead]
            if len(living_genomes) >= 2:
                gene_matrix = np.array([g.genes for g in living_genomes])
                # Within-colony vs between-colony diversity
                colony_ids_alive = [getattr(a, "colony_id", 0) for a in alive_agents_for_gen]
                unique_cols = list(set(colony_ids_alive))
                within_diversities = []
                col_gene_means = {}
                for cid in unique_cols:
                    col_genomes = [g for a, g in zip(alive_agents_for_gen, living_genomes) if getattr(a, "colony_id", 0) == cid]
                    if len(col_genomes) >= 2:
                        within_diversities.append(population_diversity(col_genomes))
                    if col_genomes:
                        col_gene_means[cid] = np.array([g.genes for g in col_genomes]).mean(axis=0)
                within_diversity = float(np.mean(within_diversities)) if within_diversities else 0.0
                # Between-colony: variance of colony gene means
                if len(col_gene_means) >= 2:
                    means_matrix = np.array(list(col_gene_means.values()))
                    between_diversity = float(np.mean(np.var(means_matrix, axis=0)))
                else:
                    between_diversity = 0.0
                world.genetic_history.append({
                    "tick":              int(world.tick),
                    "gene_means":        [round(float(v), 4) for v in gene_matrix.mean(axis=0)],
                    "gene_medians":      [round(float(v), 4) for v in np.median(gene_matrix, axis=0)],
                    "gene_variances":    [round(float(v), 4) for v in gene_matrix.var(axis=0)],
                    "gene_mins":         [round(float(v), 4) for v in gene_matrix.min(axis=0)],
                    "gene_maxs":         [round(float(v), 4) for v in gene_matrix.max(axis=0)],
                    "diversity_score":   round(population_diversity(living_genomes), 4),
                    "within_colony_diversity":  round(within_diversity, 4),
                    "between_colony_diversity": round(between_diversity, 4),
                })

        # Invoke live streaming callback at the end of each tick
        t_call_start = time.perf_counter()
        # Support both `live_callback` (run_test.py) and `callback` (run_resume.py) params
        _effective_cb = callback if callback is not None else live_callback
        if _effective_cb is not None:
            _effective_cb(world.tick, epoch_stats)
        t_call = (time.perf_counter() - t_call_start) * 1000.0
        if hasattr(world, "profiler") and world.profiler is not None:
            p = world.profiler["callback"]
            p["calls"] += 1
            p["time"] += t_call
            if t_call > p["max"]:
                p["max"] = t_call
                p["worst_tick"] = world.tick

        # Evolution Journal & Replay Logging (every 100 years = 36,000 ticks, or at start of run)
        if journal is not None:
            if world.tick == 1 and not journal.history_records:
                journal.record_epoch(world)
                world.evolution_journal_history = journal.history_records
            elif world.tick % 36000 == 0:
                journal.record_epoch(world)
                world.evolution_journal_history = journal.history_records

        # Full resumable checkpoint save (periodic, independent of vis checkpoints)
        _fci = getattr(world, "_full_checkpoint_interval", 0)
        if _fci > 0 and world.tick % _fci == 0 and world.tick > 0:
            _full_cp_config = getattr(world, "_full_checkpoint_config", None)
            _full_cp_folder = getattr(world, "exp_folder", ".")
            if _full_cp_config is not None:
                try:
                    from world.checkpoint_io import save_full_checkpoint as _sfc2
                    _cp_path2 = _os.path.join(_full_cp_folder, f"full_checkpoint_{world.tick}.json")
                    _sfc2(world, _full_cp_config, _cp_path2)
                except Exception as _cpe2:
                    print(f"  Warning: periodic full checkpoint failed at tick {world.tick}: {_cpe2}")


        # Sample epoch stats every sample_interval ticks
        if save_epochs and (world.tick % sample_interval == 0):
            epoch_num = world.tick // sample_interval
            alive = sum(1 for a in world.agents if not a.dead)
            
            # Cumulative discoveries average
            current_discoveries_sum = sum(a.discoveries_count for a in world.agents)
            avg_discoveries = float(np.mean([a.discoveries_count for a in world.agents])) if world.agents else 0.0
            
            # Novelty rate (new discoveries in this epoch)
            novelty_rate = current_discoveries_sum - prev_discoveries_sum
            prev_discoveries_sum = current_discoveries_sum
            
            # Exploration radius (average max radius)
            avg_radius = float(np.mean([a.max_radius for a in world.agents])) if world.agents else 0.0
            
            # Knowledge size (average known water & food locations)
            avg_knowledge = float(np.mean([len(a.knowledge.water_sources) + len(a.knowledge.food_sources) for a in world.agents])) if world.agents else 0.0
            
            # Knowledge Accuracy
            accuracy_list = []
            for a in world.agents:
                if a.dead:
                    continue
                total_nodes = len(a.knowledge.water_sources) + len(a.knowledge.food_sources)
                if total_nodes == 0:
                    accuracy_list.append(1.0)
                    continue
                accurate_nodes = 0
                for (y, x) in a.knowledge.water_sources:
                    has_water = (world.biome[y, x] == 8) or (world.lake_map[y, x] > 0.0) or (world.river_map[y, x] > 1500.0)
                    if has_water:
                        accurate_nodes += 1
                for (y, x) in a.knowledge.food_sources:
                    has_food = (world.elevation[y, x] >= 0.3) and ((world.wildlife[y, x] > 0.15) or (world.fertility[y, x] > 0.4)) and (world.biome[y, x] != 8)
                    if has_food:
                        accurate_nodes += 1
                accuracy_list.append(accurate_nodes / total_nodes)
            avg_accuracy = float(np.mean(accuracy_list)) if accuracy_list else 1.0
            
            # Knowledge Churn
            current_added = sum(a.nodes_added_count for a in world.agents)
            current_removed = sum(a.nodes_removed_count for a in world.agents)
            churn = (current_added - prev_nodes_added_sum) + (current_removed - prev_nodes_removed_sum)
            prev_nodes_added_sum = current_added
            prev_nodes_removed_sum = current_removed
            
            # Failed Visits (epoch deltas)
            current_failed_water = sum(a.failed_water_visits for a in world.agents)
            current_failed_food = sum(a.failed_food_visits for a in world.agents)
            failed_water = current_failed_water - prev_failed_water
            failed_food = current_failed_food - prev_failed_food
            prev_failed_water = current_failed_water
            prev_failed_food = current_failed_food
            
            # Metabolic need averages
            avg_hunger = float(np.mean([a.hunger for a in world.agents])) if world.agents else 0.0
            avg_thirst = float(np.mean([a.thirst for a in world.agents])) if world.agents else 0.0
            
            # Predictive Intelligence Metrics & Rediscoveries Churn
            pred_accuracy_list = []
            pred_gain_list = []
            for a in world.agents:
                if a.dead:
                    continue
                if a.prediction_attempts > 0:
                    pred_accuracy_list.append(a.prediction_successes / a.prediction_attempts)
                else:
                    pred_accuracy_list.append(1.0)
                pred_gain_list.append(a.prediction_gains)
            avg_pred_accuracy = float(np.mean(pred_accuracy_list)) if pred_accuracy_list else 1.0
            avg_pred_gain = float(np.mean(pred_gain_list)) if pred_gain_list else 0.0
            total_rediscoveries = sum(a.rediscoveries for a in world.agents)
            
            # Compile deaths by cause in this epoch
            deaths_in_epoch = [a for a in world.agents if a.dead and (world.tick - sample_interval < a.ticks_survived <= world.tick)]
            cause_counts = {"Starvation": 0, "Dehydration": 0, "Exposure": 0, "Old Age": 0, "Injury": 0, "Unknown": 0}
            for d in deaths_in_epoch:
                cause = d.primary_cause or "Unknown"
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
            
            # --- Phase 5 Epoch Stats ---
            alive_agents    = [a for a in world.agents if not a.dead]
            all_generations = [a.generation for a in world.agents] + [0]

            # Per-colony alive counts
            per_colony_alive = []
            for colony in getattr(world, "colonies", []):
                cnt = sum(1 for a in alive_agents if getattr(a, "colony_id", -1) == colony["id"])
                per_colony_alive.append(cnt)

            # Genetic diversity
            living_genomes   = [a.genome for a in alive_agents]
            diversity        = round(population_diversity(living_genomes), 4) if len(living_genomes) >= 2 else 0.0
            avg_gen          = round(float(np.mean([a.generation for a in alive_agents])), 2) if alive_agents else 0.0
            max_gen          = int(max((a.generation for a in world.agents), default=0))

            # Births/deaths in this epoch
            epoch_births = world.total_births - sum(e.get("_births_prev", 0) for e in epoch_stats[-1:] or [{}])
            epoch_deaths = world.total_deaths - sum(e.get("_deaths_prev", 0) for e in epoch_stats[-1:] or [{}])

            epoch_stats.append({
                "epoch":              int(epoch_num),
                "alive":              int(len(alive_agents)),
                "avg_discoveries":    round(avg_discoveries, 1),
                "novelty_rate":       int(novelty_rate),
                "avg_radius":         round(avg_radius, 1),
                "avg_knowledge":      round(avg_knowledge, 1),
                "avg_accuracy":       round(avg_accuracy * 100.0, 1),
                "knowledge_churn":    int(churn),
                "failed_water":       int(failed_water),
                "failed_food":        int(failed_food),
                "avg_hunger":         round(avg_hunger, 1),
                "avg_thirst":         round(avg_thirst, 1),
                "seasonal_factor":    round(float(global_seasonal_factor), 3),
                "avg_pred_accuracy":  round(avg_pred_accuracy * 100.0, 1),
                "avg_pred_gain":      round(avg_pred_gain, 1),
                "rediscoveries":      int(total_rediscoveries),
                # Phase 4
                "avg_fat_reserves":   round(float(np.mean([a.fat_reserves  for a in world.agents])), 1) if world.agents else 100.0,
                "avg_muscle_mass":    round(float(np.mean([a.muscle_mass   for a in world.agents])), 1) if world.agents else 100.0,
                "avg_injury_level":   round(float(np.mean([a.injury_level  for a in world.agents])), 1) if world.agents else 0.0,
                "avg_starvation_dmg": round(float(np.mean([a.starvation_damage_accumulated  for a in world.agents])), 1) if world.agents else 0.0,
                "avg_dehydration_dmg":round(float(np.mean([a.dehydration_damage_accumulated for a in world.agents])), 1) if world.agents else 0.0,
                "avg_exposure_dmg":   round(float(np.mean([a.exposure_damage_accumulated    for a in world.agents])), 1) if world.agents else 0.0,
                "avg_injury_dmg":     round(float(np.mean([a.injury_damage_accumulated      for a in world.agents])), 1) if world.agents else 0.0,
                "avg_age_dmg":        round(float(np.mean([a.age_damage_accumulated         for a in world.agents])), 1) if world.agents else 0.0,
                "deaths_starvation":  cause_counts.get("Starvation",  0),
                "deaths_dehydration": cause_counts.get("Dehydration", 0),
                "deaths_exposure":    cause_counts.get("Exposure",    0),
                "deaths_old_age":     cause_counts.get("Old Age",     0),
                "deaths_injury":      cause_counts.get("Injury",      0),
                "deaths_unknown":     cause_counts.get("Unknown",     0),
                # Phase 5
                "total_births":       int(world.total_births),
                "total_deaths":       int(world.total_deaths),
                "avg_generation":     avg_gen,
                "max_generation":     max_gen,
                "genetic_diversity":  diversity,
                "per_colony_alive":   per_colony_alive,
                # Internal tracking (used for epoch delta computation)
                "_births_prev":       int(world.total_births),
                "_deaths_prev":       int(world.total_deaths),
            })
            
        # Print status and events every 100 ticks (Timed)
        t_log_start = time.perf_counter()
        if world.tick % 100 == 0:
            alive_count = sum(1 for a in world.agents if not a.dead)
            print(f"Tick {world.tick}: {alive_count}/{len(world.agents)} agents alive. Season Factor: {global_seasonal_factor:.2f}")
            while last_printed_history_idx < len(world.history):
                print(f"  [EVENT] {world.history[last_printed_history_idx]}")
                last_printed_history_idx += 1
        t_log = (time.perf_counter() - t_log_start) * 1000.0
        if hasattr(world, "profiler") and world.profiler is not None:
            p = world.profiler["logging"]
            p["calls"] += 1
            p["time"] += t_log
            if t_log > p["max"]:
                p["max"] = t_log
                p["worst_tick"] = world.tick
                
    # Print final events if any remaining
    while last_printed_history_idx < len(world.history):
        print(f"  [EVENT] {world.history[last_printed_history_idx]}")
        last_printed_history_idx += 1
        
    final_alive = sum(1 for a in world.agents if not a.dead)
    print(f"Simulation finished. Final Population: {final_alive}/{len(world.agents)} alive after {world.tick} total ticks.")

    if hasattr(world, "profiler") and world.profiler:
        print("\n" + "=" * 60)
        print("                 SIMULATION PROFILE SUMMARY")
        print("=" * 60)
        total_time = sum(data["time"] for data in world.profiler.values())
        print(f"  Total Tracked Execution Time: {total_time:.2f} ms")
        print(f"  Subsystem        | Calls      | Total (ms)  | Avg (ms)   | Max (ms)   | % Time")
        print(f"  -----------------|------------|-------------|------------|------------|-------")
        for name, data in sorted(world.profiler.items(), key=lambda x: x[1]["time"], reverse=True):
            calls = data["calls"]
            t = data["time"]
            avg = t / calls if calls > 0 else 0.0
            pct = (t / total_time * 100.0) if total_time > 0 else 0.0
            print(f"  {name.replace('_', ' ').title():16} | {calls:10d} | {t:11.2f} | {avg:10.4f} | {data['max']:10.2f} | {pct:5.1f}%")
        
        # Aggregate Workload Counters & Cache Statistics
        total_decisions = sum(a.decision_evals for a in world.agents)
        total_predictor_calls = sum(a.predictor_calls for a in world.agents)
        total_perception_calls = sum(a.perception_calls for a in world.agents)
        total_memories_searched = sum(a.memories_searched for a in world.agents)
        total_targets_evaluated = sum(a.targets_evaluated for a in world.agents)
        total_relationship_lookups = sum(getattr(a, "relationship_lookups", 0) for a in world.agents)
        
        total_cache_hits = sum(getattr(a, "memory_cache_hits", 0) for a in world.agents)
        total_cache_misses = sum(getattr(a, "memory_cache_misses", 0) for a in world.agents)
        total_cache_calls = total_cache_hits + total_cache_misses
        cache_hit_rate = (total_cache_hits / total_cache_calls * 100.0) if total_cache_calls > 0 else 0.0
        
        q_calls = getattr(world, "query_agents_calls", 0)
        q_returned = getattr(world, "query_agents_returned_sum", 0)
        avg_density = (q_returned / q_calls) if q_calls > 0 else 0.0

        print("=" * 60)
        print("                 WORKLOAD & TELEMETRY SUMMARY")
        print("=" * 60)
        print(f"  Decision Evaluations:    {total_decisions:15d}")
        print(f"  Predictor Neural Calls:  {total_predictor_calls:15d}")
        print(f"  Perception Scans:        {total_perception_calls:15d}")
        print(f"  Memory Nodes Searched:   {total_memories_searched:15d}")
        print(f"  Path Targets Evaluated:  {total_targets_evaluated:15d}")
        print(f"  Relationship Lookups:    {total_relationship_lookups:15d}")
        print(f"  Memory Cache Hit Rate:   {cache_hit_rate:13.1f}% ({total_cache_hits} hits, {total_cache_misses} misses)")
        print(f"  Spatial Queries Density:  {avg_density:14.2f} average agents returned ({q_calls} calls)")
        print("=" * 60 + "\n")
        
        # Print Demographic Health Dashboard and Water Economy report (Phase 10)
        print(world.telemetry.generate_dashboard_report())
    
    return epoch_stats
