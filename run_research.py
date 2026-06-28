import os
import sys
import json
import random
import string
import datetime
import numpy as np
import scipy.stats as stats
import subprocess

# Add project root to python path to ensure imports work cleanly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from world.generator import generate_world
from world.predictor import predict_settlements
from world.agents.agent import Agent
from world.agents.simulation import run_simulation
from world.state import WorldState, BIOME_NAMES

# 📍 EXPERIMENTAL COLONY SPAWNING MODES FOR RUNS
SPAWN_MODE = "fixed"               # Available modes:
                                   #   "fixed"           = control; NMS spots, same every run
                                   #   "random_valid"    = random valid land locations with resources
                                   #   "random_anywhere" = true chaos; random land cells
                                   #   "targeted_biome"  = target specific biomes defined in TARGETED_BIOMES
                                   
# Mapped default locations for "fixed" spawn mode:
# Specify the preferred spot index (1, 2, 3, or 4) for each colony.
# E.g., if you map "Alpha": 4, Alpha will spawn at the 4th spot.
COLONY_SPAWN_LOCATIONS = {
    "Alpha": 1,
    "Beta": 2,
    "Gamma": 3,
    "Delta": 4
}

# Targeted biomes mapped to Colonies (Alpha, Beta, Gamma, Delta) for "targeted_biome" mode.
# Available biomes: "Desert", "Forest", "Tundra", "Coast", "Rainforest", "Taiga"
TARGETED_BIOMES = ["Desert", "Forest", "Tundra", "Coast"]

# ---------------------------------------------------------------------------
# UUID and Path Utilities
# ---------------------------------------------------------------------------

def generate_uuid() -> str:
    """Generates a research experiment ID of the format EXP-YYYYMMDD-XXXX."""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"EXP-{date_str}-{suffix}"

def get_git_commit() -> str:
    """Retrieves short git hash if inside a git repository."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "N/A"

# ---------------------------------------------------------------------------
# Metric Registry & Calculations
# ---------------------------------------------------------------------------

class MetricRegistry:
    """Centralized definition of extractable world and agent-level telemetry."""
    
    @staticmethod
    def extract_world_metrics(world: WorldState) -> dict:
        living = [a for a in world.agents if not a.dead]
        n_living = len(living)
        
        if n_living == 0:
            return {
                "tick": world.tick,
                "population": 0,
                "avg_hunger": 100.0,
                "avg_thirst": 100.0,
                "avg_energy": 0.0,
                "avg_health": 0.0,
                "avg_fear": 0.0,
                "avg_frustration": 0.0,
                "avg_boredom": 0.0,
                "avg_longing": 0.0,
                "avg_grief": 0.0,
                "avg_stored_food": 0.0,
                "avg_stored_water": 0.0,
                "avg_shelter_level": 0.0,
                "avg_shelter_durability": 0.0,
                "avg_curiosity": 0.0,
                "avg_risk_tolerance": 0.0,
                "avg_prediction_confidence": 0.0,
                "avg_prediction_accuracy": 0.0,
                "total_births": int(getattr(world, "total_births", 0)),
                "total_deaths": int(getattr(world, "total_deaths", 0)),
                "discoveries": int(sum(a.discoveries_count for a in world.agents) if world.agents else 0),
                "network_density": 0.0,
                "average_trust": 0.0,
                "average_attachment": 0.0,
                "network_centralization": 0.0,
                "connected_components": 0
            }
            
        # Extract base agent stats
        avg_hunger = float(np.mean([a.hunger for a in living]))
        avg_thirst = float(np.mean([a.thirst for a in living]))
        avg_energy = float(np.mean([a.energy for a in living]))
        avg_health = float(np.mean([a.health for a in living]))
        
        # Extract drive states
        avg_fear = float(np.mean([a.drives.fear for a in living]))
        avg_frustration = float(np.mean([a.drives.frustration for a in living]))
        avg_boredom = float(np.mean([a.drives.boredom for a in living]))
        avg_longing = float(np.mean([a.drives.longing for a in living]))
        avg_grief = float(np.mean([a.drives.grief for a in living]))
        
        # Shelter & Storage
        avg_stored_food = float(np.mean([a.stored_food for a in living]))
        avg_stored_water = float(np.mean([a.stored_water for a in living]))
        avg_shelter_level = float(np.mean([a.shelter_level for a in living]))
        avg_shelter_durability = float(np.mean([a.shelter_durability for a in living]))
        
        # Cognition / Traits
        avg_curiosity = float(np.mean([a.traits.get("curiosity", 0.5) for a in living]))
        avg_risk = float(np.mean([a.traits.get("risk_tolerance", 0.5) for a in living]))
        avg_pred_conf = float(np.mean([a.prediction_confidence for a in living]))
        avg_pred_acc = float(np.mean([a.prediction_accuracy if hasattr(a, "prediction_accuracy") else 1.0 for a in living]))
        
        # Graph Metrics
        graph_metrics = compute_network_metrics(living)
        
        return {
            "tick": world.tick,
            "population": n_living,
            "avg_hunger": round(avg_hunger, 2),
            "avg_thirst": round(avg_thirst, 2),
            "avg_energy": round(avg_energy, 2),
            "avg_health": round(avg_health, 2),
            "avg_fear": round(avg_fear, 3),
            "avg_frustration": round(avg_frustration, 3),
            "avg_boredom": round(avg_boredom, 3),
            "avg_longing": round(avg_longing, 3),
            "avg_grief": round(avg_grief, 3),
            "avg_stored_food": round(avg_stored_food, 2),
            "avg_stored_water": round(avg_stored_water, 2),
            "avg_shelter_level": round(avg_shelter_level, 2),
            "avg_shelter_durability": round(avg_shelter_durability, 2),
            "avg_curiosity": round(avg_curiosity, 3),
            "avg_risk_tolerance": round(avg_risk, 3),
            "avg_prediction_confidence": round(avg_pred_conf, 3),
            "avg_prediction_accuracy": round(avg_pred_acc, 3),
            "total_births": int(getattr(world, "total_births", 0)),
            "total_deaths": int(getattr(world, "total_deaths", 0)),
            "discoveries": int(sum(a.discoveries_count for a in world.agents) if world.agents else 0),
            **graph_metrics
        }

# ---------------------------------------------------------------------------
# Relationship Network Analysis
# ---------------------------------------------------------------------------

def compute_network_metrics(living_agents: list) -> dict:
    """Computes relationship trust graph centralities, densities, and components."""
    N = len(living_agents)
    if N < 2:
        return {
            "network_density": 0.0,
            "average_trust": 0.0,
            "average_attachment": 0.0,
            "network_centralization": 0.0,
            "connected_components": 1
        }
        
    id_to_idx = {agent.id: idx for idx, agent in enumerate(living_agents)}
    trust_mat = np.zeros((N, N), dtype=np.float32)
    attach_mat = np.zeros((N, N), dtype=np.float32)
    
    # Build matrices
    for i, agent in enumerate(living_agents):
        for other_id, rel in agent.relationships.items():
            if other_id in id_to_idx:
                j = id_to_idx[other_id]
                trust_mat[i, j] = max(0.0, rel.trust)
                attach_mat[i, j] = max(0.0, rel.attachment)
                
    # Densities and Averages
    network_density = float(np.count_nonzero(trust_mat > 0.1) / (N * (N - 1)))
    average_trust = float(np.mean(trust_mat[trust_mat > 0.0])) if np.any(trust_mat > 0.0) else 0.0
    average_attachment = float(np.mean(attach_mat[attach_mat > 0.0])) if np.any(attach_mat > 0.0) else 0.0
    
    # Out-degree trust centrality
    deg_centrality = np.sum(trust_mat, axis=1)
    max_deg = np.max(deg_centrality)
    
    # Network centralization (Freeman's Index)
    if N > 2:
        denom = (N - 1) * (N - 2)  # Max possible difference sum for directed graph normalized
        centralization = float(np.sum(max_deg - deg_centrality) / denom) if denom > 0 else 0.0
    else:
        centralization = 0.0
        
    # Find connected components (BFS/DFS on weak/undirected connections of trust > 0.1)
    adj = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(N):
            if trust_mat[i, j] > 0.1 or trust_mat[j, i] > 0.1:
                adj[i].append(j)
                
    visited = set()
    components = 0
    for node in range(N):
        if node not in visited:
            components += 1
            # BFS traverse
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        
    return {
        "network_density": round(network_density, 3),
        "average_trust": round(average_trust, 3),
        "average_attachment": round(average_attachment, 3),
        "network_centralization": round(centralization, 3),
        "connected_components": int(components)
    }

# ---------------------------------------------------------------------------
# Emergence Detector & Anomaly Monitor
# ---------------------------------------------------------------------------

class EmergenceDetector:
    """Monitors the simulation step-by-step to flag emergent behaviors and anomalies."""
    def __init__(self, founding_spots: list):
        self.founding_spots = founding_spots
        self.logged_behaviors = []
        self.permanent_settlement_detected = False
        self.satellite_settlement_detected = False
        self.colony_split_detected = False
        
        # Relocation tracking for migration detection: agent_id -> list of (tick, coord)
        self.shelter_history = {}
        # Population tracker for crash detection
        self.prev_population = 16

    def scan_tick(self, tick: int, world: WorldState):
        living = [a for a in world.agents if not a.dead]
        n_living = len(living)
        
        # 1. Anomaly Detection: Population Crash (>= 40% drop in last 100 ticks)
        if tick % 100 == 0:
            if self.prev_population > 0:
                drop_pct = (self.prev_population - n_living) / self.prev_population
                if drop_pct >= 0.40:
                    self._log(tick, "Anomaly", f"Population crash detected: dropped {round(drop_pct * 100.0, 1)}% (from {self.prev_population} to {n_living})")
            self.prev_population = n_living
            
        # 2. Emergence: Permanent Settlement
        if not self.permanent_settlement_detected:
            for agent in living:
                if agent.shelter_level >= 2 and agent.shelter_durability > 50.0:
                    self.permanent_settlement_detected = True
                    self._log(tick, "Emergence: Permanent Settlement", f"Agent {agent.id} established a durable permanent shelter (Level {agent.shelter_level})")
                    break
                    
        # 3. Emergence: Satellite Settlement (shelter built > 150 units from founder coordinates)
        if not self.satellite_settlement_detected and self.founding_spots:
            for agent in living:
                if agent.shelter_location is not None:
                    sy, sx = agent.shelter_location
                    # Check distance from all founding spots
                    min_dist = float('inf')
                    for spot in self.founding_spots:
                        dist = np.sqrt((sy - spot["y"])**2 + (sx - spot["x"])**2)
                        if dist < min_dist:
                            min_dist = dist
                    if min_dist >= 150.0:
                        self.satellite_settlement_detected = True
                        self._log(tick, "Emergence: Satellite Settlement", f"Agent {agent.id} built a satellite shelter at ({sx}, {sy}), {round(min_dist, 1)} units away from nearest founding settlement")
                        break
                        
        # 4. Emergence: Migration Wave (when >= 3 agents move shelter home locations within 200 ticks)
        for agent in living:
            if agent.shelter_location is not None:
                curr_loc = agent.shelter_location
                history = self.shelter_history.setdefault(agent.id, [])
                if not history or history[-1][1] != curr_loc:
                    history.append((tick, curr_loc))
                    
        # Detect group relocation
        if tick % 50 == 0:
            relocations = []
            for agent_id, hist in self.shelter_history.items():
                # Count relocations in the last 200 ticks
                recent_moves = [h for h in hist if tick - 200 < h[0] <= tick]
                if len(recent_moves) >= 2:
                    relocations.append(agent_id)
            if len(relocations) >= 3:
                self._log(tick, "Emergence: Migration Wave", f"Migration wave detected! Agents {relocations} relocated their shelters in close succession.")
                self.shelter_history.clear() # Reset triggers

        # 5. Emergence: Colony Split (Founders establishing two distant shelter clusters)
        if not self.colony_split_detected and tick % 200 == 0:
            # Map colony shelters
            shelters = [a.shelter_location for a in living if a.shelter_location is not None]
            if len(shelters) >= 4:
                # Run simple dispersion scan: max distance between any two shelters
                max_disp = 0.0
                for i in range(len(shelters)):
                    for j in range(i+1, len(shelters)):
                        s1, s2 = shelters[i], shelters[j]
                        d = np.sqrt((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)
                        if d > max_disp:
                            max_disp = d
                if max_disp >= 350.0:
                    self.colony_split_detected = True
                    self._log(tick, "Emergence: Colony Split", f"Colony split detected: shelters are highly dispersed across the land (Max distance: {round(max_disp, 1)} cells)")

    def _log(self, tick: int, event_type: str, details: str):
        print(f"  [Emergence Detector] Tick {tick} | {event_type}: {details}")
        self.logged_behaviors.append({
            "tick": int(tick),
            "type": event_type,
            "details": details
        })

# ---------------------------------------------------------------------------
# Causal Chain Recorder
# ---------------------------------------------------------------------------

class CausalChainRecorder:
    """Tracks chronological sequences of trigger events leading to cognitive changes and physical choices."""
    def __init__(self):
        self.tracked_chains = []
        self.active_traces = {}  # agent_id -> trace dict
        
    def log_trigger(self, agent_id: int, tick: int, trigger_type: str, world: WorldState):
        agent = next((a for a in world.agents if a.id == agent_id), None)
        if not agent:
            return
            
        # Do not overwrite an ongoing chain for this agent
        if agent_id in self.active_traces:
            return
            
        # Capture pre-trigger state
        self.active_traces[agent_id] = {
            "agent_id": agent_id,
            "trigger": trigger_type,
            "start_tick": tick,
            "pre_drives": {
                "fear": round(agent.drives.fear, 3),
                "frustration": round(agent.drives.frustration, 3),
                "boredom": round(agent.drives.boredom, 3),
                "longing": round(agent.drives.longing, 3),
                "grief": round(agent.drives.grief, 3),
                "hunger_tension": round(agent.drives.hunger_tension, 3),
                "thirst_tension": round(agent.drives.thirst_tension, 3)
            },
            "snapshots": [],
            "completed": False
        }
        
    def update_tick(self, tick: int, world: WorldState):
        for agent_id, trace in list(self.active_traces.items()):
            agent = next((a for a in world.agents if a.id == agent_id), None)
            if not agent or agent.dead:
                # Close chain with death outcome
                trace["completed"] = True
                trace["outcome"] = "Agent died"
                trace["end_tick"] = tick
                self.tracked_chains.append(trace)
                del self.active_traces[agent_id]
                continue
                
            elapsed = tick - trace["start_tick"]
            
            # Record snapshots of shifts
            if elapsed in (5, 15, 30):
                trace["snapshots"].append({
                    "elapsed_ticks": elapsed,
                    "action": agent.current_action,
                    "fear": round(agent.drives.fear, 3),
                    "frustration": round(agent.drives.frustration, 3),
                    "hunger_tension": round(agent.drives.hunger_tension, 3)
                })
                
            # Check for resolution/outcome
            outcome = None
            if trace["trigger"] == "Storm / Freezing Temp":
                if agent.current_action == "Sheltering" and agent.shelter_level >= 1:
                    outcome = f"Sheltered successfully (Level {agent.shelter_level})"
            elif trace["trigger"] == "Starvation Threat":
                if agent.hunger < 20.0:
                    outcome = "Ate food and resolved hunger"
            elif trace["trigger"] == "High Longing / Loneliness":
                if len(agent.children_ids) > len([cid for cid in agent.children_ids if cid < trace["start_tick"]]):
                    outcome = "Reproduced and resolved mating drive"
                    
            if outcome or elapsed >= 50:
                trace["completed"] = True
                trace["outcome"] = outcome or "Drive subsided or trace timed out"
                trace["end_tick"] = tick
                self.tracked_chains.append(trace)
                del self.active_traces[agent_id]

# ---------------------------------------------------------------------------
# Evolution Tracker
# ---------------------------------------------------------------------------

def calculate_genetic_evolution(agents: list) -> dict:
    """Aggregates trait genes across generations (Gen 0, Gen 1, Gen 2, etc.)."""
    gen_groups = {}
    for agent in agents:
        gen = agent.generation
        gen_groups.setdefault(gen, []).append(agent)
        
    evolution = {}
    for gen, group in sorted(gen_groups.items()):
        evolution[f"Gen {gen}"] = {
            "count": len(group),
            "curiosity": round(float(np.mean([a.genome.genes[6] for a in group])), 3),
            "risk_sensitivity": round(float(np.mean([a.genome.genes[10] for a in group])), 3),
            "learning_rate": round(float(np.mean([a.genome.genes[13] for a in group])), 3),
            "aggression": round(float(np.mean([a.genome.genes[8] for a in group])), 3),
            "sharing": round(float(np.mean([a.genome.genes[9] for a in group])), 3)
        }
    return evolution

# ---------------------------------------------------------------------------
# Statistical Calculations
# ---------------------------------------------------------------------------

def calculate_cohens_d(treatment_vals: list, control_vals: list) -> float:
    """Calculates Cohen's d effect size between treatment and control groups."""
    t_arr = np.array(treatment_vals, dtype=np.float32)
    c_arr = np.array(control_vals, dtype=np.float32)
    
    n_t, n_c = len(t_arr), len(c_arr)
    if n_t < 2 or n_c < 2:
        return 0.0
        
    m_t, m_c = np.mean(t_arr), np.mean(c_arr)
    v_t, v_c = np.var(t_arr, ddof=1), np.var(c_arr, ddof=1)
    
    pooled_sd = np.sqrt(((n_t - 1) * v_t + (n_c - 1) * v_c) / (n_t + n_c - 2))
    if pooled_sd == 0.0:
        return 0.0
    return float((m_t - m_c) / pooled_sd)

def calculate_correlation_matrix(agents: list) -> dict:
    """Computes Pearson and Spearman correlations across agent lifetime metrics."""
    if len(agents) < 3:
        return {}
        
    metrics = {
        "longevity": [a.ticks_survived for a in agents],
        "curiosity": [a.genome.genes[6] for a in agents],
        "aggression": [a.genome.genes[8] for a in agents],
        "avg_fear": [getattr(a.drives, "fear_mean", 0.0) for a in agents],
        "avg_frustration": [getattr(a.drives, "frustration_mean", 0.0) for a in agents],
        "discoveries": [a.discoveries_count for a in agents],
        "shelter_level": [a.shelter_level for a in agents],
        "shares": [a.action_counts.get("Share Food", 0) + a.action_counts.get("Share Water", 0) for a in agents]
    }
    
    keys = list(metrics.keys())
    pearson_mat = {}
    spearman_mat = {}
    
    for i, k1 in enumerate(keys):
        pearson_mat[k1] = {}
        spearman_mat[k1] = {}
        for j, k2 in enumerate(keys):
            x, y = metrics[k1], metrics[k2]
            # Pearson
            r_val, _ = stats.pearsonr(x, y)
            r_val = 0.0 if np.isnan(r_val) else r_val
            pearson_mat[k1][k2] = round(float(r_val), 3)
            
            # Spearman
            rho_val, _ = stats.spearmanr(x, y)
            rho_val = 0.0 if np.isnan(rho_val) else rho_val
            spearman_mat[k1][k2] = round(float(rho_val), 3)
            
    return {"pearson": pearson_mat, "spearman": spearman_mat}

# ---------------------------------------------------------------------------
# Headless Simulation Runner
# ---------------------------------------------------------------------------

def run_headless_simulation(exp_id: str, seed: int, ticks: int, scarcity: float, ablation_dict: dict, notes: str, level_name: str) -> tuple:
    """Runs a single simulation headlessly and returns metrics log and summary data."""
    print(f"\n[Simulation Executor] Launching Seed = {seed} | Ticks = {ticks} | Scarcity = {scarcity}")
    
    # Create clean folder for logs
    exp_folder = os.path.join("research", exp_id, f"run_seed_{seed}")
    os.makedirs(exp_folder, exist_ok=True)
    
    world = generate_world(width=1024, height=1024, seed=seed)
    world.ablation = ablation_dict
    world.exp_folder = exp_folder
    world.spawn_mode = SPAWN_MODE
    world.colony_spawn_locations = COLONY_SPAWN_LOCATIONS
    world.targeted_biomes = TARGETED_BIOMES
    
    # Find founder locations for emergence detection
    w_scale = world.width / 1024.0
    founding_spots = predict_settlements(world, count=4, exclusion_radius=250.0 * w_scale)
    
    # Initialize trackers
    detector = EmergenceDetector(founding_spots)
    causal_recorder = CausalChainRecorder()
    
    temporal_logs = []
    
    def live_callback(tick, epoch_stats):
        # Tick scans for emergence and metrics
        detector.scan_tick(tick, world)
        
        # Trigger logging scans
        for agent in world.agents:
            if not agent.dead:
                # Starvation threat trigger
                if agent.hunger > 60.0:
                    causal_recorder.log_trigger(agent.id, tick, "Starvation Threat", world)
                # Storm trigger (temperature drops)
                _biome_id = int(world.biome[agent.location[0], agent.location[1]])
                _day = tick % 360
                _amp = {6: 22.0, 1: 18.0, 2: 18.0, 3: 18.0, 4: 12.0}.get(_biome_id, 12.0)
                _local_temp = float(world.temperature[agent.location[0], agent.location[1]]) + _amp * np.cos(((_day - 180.0)/180.0)*np.pi)
                if _local_temp < -2.0:
                    causal_recorder.log_trigger(agent.id, tick, "Storm / Freezing Temp", world)
                # Longing trigger
                if agent.drives.longing > 0.6:
                    causal_recorder.log_trigger(agent.id, tick, "High Longing / Loneliness", world)
                    
        causal_recorder.update_tick(tick, world)
        
        # Pull Registry Metrics every 50 ticks
        if tick % 50 == 0:
            temporal_logs.append(MetricRegistry.extract_world_metrics(world))
            
    try:
        run_simulation(
            world,
            ticks=ticks,
            scarcity_level=scarcity,
            save_paths=False,
            save_epochs=False,
            live_callback=live_callback
        )
    except KeyboardInterrupt:
        print("\n⚠️ [EMERGENCY STOP] KeyboardInterrupt detected! Exiting simulation tick loop and compiling current sweep telemetry immediately...")
    
    # End-of-run data compilation
    living = [a for a in world.agents if not a.dead]
    survival_ages = [a.ticks_survived for a in world.agents]
    
    summary = {
        "seed": seed,
        "ticks_completed": world.tick,
        "survivors": len(living),
        "total_population": len(world.agents),
        "average_lifespan": round(float(np.mean(survival_ages)), 1),
        "max_lifespan": int(np.max(survival_ages)) if survival_ages else 0,
        "genetic_evolution": calculate_genetic_evolution(world.agents),
        "network_centralities": compute_network_metrics(living),
        "correlations": calculate_correlation_matrix(world.agents),
        "spawn_mode": getattr(world, "spawn_mode", "fixed"),
        "spawn_conditions": getattr(world, "spawn_conditions", {})
    }
    
    # Dump run files
    with open(os.path.join(exp_folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    if temporal_logs:
        with open(os.path.join(exp_folder, "metrics.csv"), "w", encoding="utf-8") as f:
            f.write(",".join(temporal_logs[0].keys()) + "\n")
            for log in temporal_logs:
                f.write(",".join(str(v) for v in log.values()) + "\n")
            
    return summary, temporal_logs, detector.logged_behaviors, causal_recorder.tracked_chains

# ---------------------------------------------------------------------------
# Scientific Report Formatter
# ---------------------------------------------------------------------------

def generate_experiment_report(exp_id: str, manifest: dict, run_results: list, all_behaviors: list, all_chains: list) -> str:
    """Assembles a publication-ready scientific report for an experiment."""
    n_runs = len(run_results)
    avg_lifespan = float(np.mean([r["average_lifespan"] for r in run_results]))
    avg_survivors = float(np.mean([r["survivors"] for r in run_results]))
    
    # Calculate Hypothesis Rankings
    hypotheses = [
        {"name": "Altruism / resource sharing predicts longevity", "v1": "shares", "v2": "longevity"},
        {"name": "Curiosity selects for longevity", "v1": "curiosity", "v2": "longevity"},
        {"name": "Aggression reduces longevity (Inverted)", "v1": "aggression", "v2": "longevity", "invert": True},
        {"name": "Fear predicts technology (shelter levels)", "v1": "avg_fear", "v2": "shelter_level"},
        {"name": "Curiosity predicts discoveries count", "v1": "curiosity", "v2": "discoveries"},
        {"name": "Technology (shelter quality) predicts longevity", "v1": "shelter_level", "v2": "longevity"},
        {"name": "Frustration suppresses cooperation (Inverted)", "v1": "avg_frustration", "v2": "shares", "invert": True}
    ]
    
    ranked_hyps = []
    for hyp in hypotheses:
        r_vals = []
        for r in run_results:
            corr = r.get("correlations", {})
            p_matrix = corr.get("pearson", {})
            v1, v2 = hyp["v1"], hyp["v2"]
            r_val = 0.0
            if v1 in p_matrix and v2 in p_matrix[v1]:
                r_val = p_matrix[v1][v2]
            elif v2 in p_matrix and v1 in p_matrix[v2]:
                r_val = p_matrix[v2][v1]
            r_vals.append(r_val)
            
        r_avg = float(np.mean(r_vals)) if r_vals else 0.0
        abs_r = abs(r_avg)
        if abs_r >= 0.7:
            strength = "Very Strong Evidence"
        elif abs_r >= 0.5:
            strength = "Strong Evidence"
        elif abs_r >= 0.3:
            strength = "Moderate Evidence"
        elif abs_r >= 0.1:
            strength = "Weak Evidence"
        else:
            strength = "Negligible / No Evidence"
            
        ranked_hyps.append({
            "name": hyp["name"],
            "r_avg": r_avg,
            "strength": strength,
            "abs_r": abs_r
        })
    ranked_hyps.sort(key=lambda x: x["abs_r"], reverse=True)

    # Format report string
    report = f"""# Scientific Research Report: {manifest["name"]}

**Experiment ID:** `{exp_id}`  
**Level:** {manifest["level"]}  
**Genesis Core Version:** {manifest["genesis_version"]}  
**Git Commit:** `{manifest["git_commit"]}`  
**Date:** {manifest["date"]}  
**Seed Count / Scarcity:** {n_runs} seeds / Scarcity {manifest["scarcity"]}  
**Ablations Applied:** `{json.dumps(manifest["ablations"])}`  

---

## 1. Abstract
This scientific report documents the behavior, survival, and dynamics of the Project Genesis agent populations under a controlled experiment setup. In this run (`{exp_id}`), we simulated {n_runs} independent populations across unique topographic seeds. The population evolved under a scarcity setting of {manifest["scarcity"]} and completed a runtime of {manifest["ticks"]} ticks.

---

## 2. Experimental Setup & Methods
Agents are spawned in 4 distinct colonies of 4 founders each, derived from random Perlin genetic priors.
*   **Spawn Mode:** `{manifest.get("spawn_mode", "fixed").upper()}`
*   **Ablation Parameters:**
    *   **Planner Subsystem:** {"ENABLED" if manifest["ablations"]["planner"] else "ABLATED (Bypassed queue pop and commitment)"}
    *   **Emotion Subsystem:** {"ENABLED" if manifest["ablations"]["emotion"] else "ABLATED (Continuous drives clamped to 0.0)"}
    *   **Relationships Subsystem:** {"ENABLED" if manifest["ablations"]["relationships"] else "ABLATED (Graph disabled, social weights neutral)"}
    *   **Memory Importance:** {"ENABLED" if manifest["ablations"]["memory_importance"] else "ABLATED (Uniform exponential decay)"}
    *   **Motivation Drift:** {"ENABLED" if manifest["ablations"]["motivation"] else "ABLATED (Lateral inhibition and profile drifts disabled)"}
    *   **Prediction Error Feedback:** {"ENABLED" if manifest["ablations"]["prediction_error"] else "ABLATED (expectation feedback zeroed)"}

---

## 3. Results & Lifespan Summary
The average lifespan across all seeds was **{round(avg_lifespan, 1)}** ticks (approx. {round(avg_lifespan / 360, 1)} years). The average number of surviving agents at tick {manifest["ticks"]} was **{round(avg_survivors, 2)}** founders/descendants per seed.

| Seed | Ticks Completed | Survivors | Avg Lifespan (Ticks) | Max Lifespan (Ticks) | Trust Centralization |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in run_results:
        cent = r["network_centralities"]["network_centralization"]
        report += f"| {r['seed']} | {r['ticks_completed']} | {r['survivors']}/{r['total_population']} | {r['average_lifespan']} | {r['max_lifespan']} | {cent} |\n"
        
    report += """
---

## 4. Automatic Hypothesis Ranking

The framework automatically maps expressed behaviors to lifetime outcomes to rank which psychological, genetic, and social factors most strongly predict civilization survival:

| Rank | Hypothesis | Avg Correlation (r) | Evidence Strength |
| :---: | :--- | :---: | :--- |
"""
    for rank, h in enumerate(ranked_hyps):
        report += f"| {rank+1} | {h['name']} | `{h['r_avg']:.3f}` | **{h['strength']}** |\n"

    report += """
---

## 5. Emergent Behaviors & Discoveries
The Emergence Detector flagged the following civilization-level behaviors during execution:
"""
    if all_behaviors:
        for idx, bh in enumerate(all_behaviors[:10]):
            report += f"*   **Tick {bh['tick']}** | *{bh['type']}*: {bh['details']}\n"
        if len(all_behaviors) > 10:
            report += f"*   *(and {len(all_behaviors) - 10} other emergent events recorded in the database)*\n"
    else:
        report += "*   No novel emergent structures were detected in this run.\n"
        
    report += """
---

## 6. Causal Chain Traces
The following causal chains document trigger-to-outcome sequences recorded step-by-step:
"""
    if all_chains:
        for idx, ch in enumerate(all_chains[:4]):
            pre = ", ".join(f"{k}: {v}" for k, v in list(ch['pre_drives'].items())[:3])
            report += f"""
### Chain {idx+1}: Agent {ch['agent_id']} ({ch['trigger']})
*   **Trigger Tick:** {ch['start_tick']}
*   **Pre-Trigger State:** `{pre}`
*   **Timeline:**
"""
            for snap in ch['snapshots']:
                report += f"    - +{snap['elapsed_ticks']} ticks: action='{snap['action']}', fear={snap['fear']}, frustration={snap['frustration']}\n"
            report += f"*   **Resolution:** {ch['outcome']} (at tick {ch.get('end_tick', 'N/A')})\n"
    else:
        report += "*   No complete causal chains were logged during simulation execution.\n"
        
    report += """
---

## 7. Genetic Evolution Trajectory
The drift of expressed genotypic parameters from Generation 0 founders to Generation 1+ descendants highlights adaptive traits selecting over time:

| Generation | Population | Avg Curiosity | Risk Sensitivity | Learning Rate | Aggression |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    # Aggregate gen drift from first run
    first_evo = run_results[0]["genetic_evolution"]
    for gen, metrics in sorted(first_evo.items()):
        report += f"| {gen} | {metrics['count']} | {metrics['curiosity']} | {metrics['risk_sensitivity']} | {metrics['learning_rate']} | {metrics['aggression']} |\n"
        
    report += """
---

## 8. Discussion & Limitations
These results reflect how individual agent cognition constraints shape collective colony structures. Ablating subsystems changes how rapidly agents organize shelters or share food stockpiles, impacting survival. Limitations of this trial include deterministic water/food placements and isolated colony networks.

---

## 9. Meta-Analysis Data Reference
To conduct comparative meta-analyses or construct Cohen's d effect size tables, refer to `research/meta_analysis.md` which compiles summaries across all experiments.
"""
    return report

# ---------------------------------------------------------------------------
# Meta-Analysis Aggregator
# ---------------------------------------------------------------------------

def generate_global_meta_analysis():
    """Aggregates all experiment manifests and runs to compile a meta-analysis file."""
    research_dir = "research"
    if not os.path.exists(research_dir):
        return
        
    exps = []
    for d in os.listdir(research_dir):
        manifest_path = os.path.join(research_dir, d, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                # Scan summaries
                summaries = []
                for rd in os.listdir(os.path.join(research_dir, d)):
                    summary_path = os.path.join(research_dir, d, rd, "summary.json")
                    if os.path.exists(summary_path):
                        with open(summary_path, "r") as sf:
                            summaries.append(json.load(sf))
                exps.append((manifest, summaries))
            except Exception:
                pass
                
    if not exps:
        return
        
    report = """# Project Genesis Meta-Analysis & Ablation Table

This document aggregates overall metrics across all reproducible experiments inside the `research/` database.

## 📊 Global Experiment Registry

| Experiment ID | Level | Ticks | Scarcity | Planner | Emotion | Rel. | Mem. Imp. | Motivation | Avg Lifespan | Avg Survivors |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for manifest, summaries in exps:
        avg_life = float(np.mean([s["average_lifespan"] for s in summaries])) if summaries else 0.0
        avg_surv = float(np.mean([s["survivors"] for s in summaries])) if summaries else 0.0
        ab = manifest["ablations"]
        report += (
            f"| `{manifest['experiment_id']}` | {manifest['level']} | {manifest['ticks']} | {manifest['scarcity']} | "
            f"{'On' if ab.get('planner', True) else 'Off'} | {'On' if ab.get('emotion', True) else 'Off'} | "
            f"{'On' if ab.get('relationships', True) else 'Off'} | {'On' if ab.get('memory_importance', True) else 'Off'} | "
            f"{'On' if ab.get('motivation', True) else 'Off'} | {round(avg_life, 1)} | {round(avg_surv, 2)} |\n"
        )
        
    # Effect Size Analysis (Cohen's d comparison if both treatment and control exist)
    treatment_summaries = None
    control_summaries = None
    
    for manifest, summaries in exps:
        ab = manifest["ablations"]
        is_full = all(ab.values())
        is_control = not any(ab.values())
        if is_full:
            treatment_summaries = summaries
        elif is_control:
            control_summaries = summaries
            
    if treatment_summaries and control_summaries:
        t_lifespans = [s["average_lifespan"] for s in treatment_summaries]
        c_lifespans = [s["average_lifespan"] for s in control_summaries]
        t_survivors = [s["survivors"] for s in treatment_summaries]
        c_survivors = [s["survivors"] for s in control_summaries]
        
        d_life = calculate_cohens_d(t_lifespans, c_lifespans)
        d_surv = calculate_cohens_d(t_survivors, c_survivors)
        
        report += f"""
---

## 🔬 Subsystem Treatment Impact (Control vs Treatment Effect Size)

Comparing the baseline Control Group (all cognitive systems ablated) against the Treatment Group (full Phase 8.4 psychological architecture):

*   **Average Lifespan Effect Size (Cohen's d):** **{round(d_life, 2)}**
*   **Average Survivors Count Effect Size (Cohen's d):** **{round(d_surv, 2)}**

*Interpretation:* Cohen's d > 0.8 represents a large, statistically significant effect on colony survival indicators.
"""

    # Global Hypothesis Discovery Rankings
    hypotheses = [
        {"name": "Altruism / resource sharing predicts longevity", "v1": "shares", "v2": "longevity"},
        {"name": "Curiosity selects for longevity", "v1": "curiosity", "v2": "longevity"},
        {"name": "Aggression reduces longevity (Inverted)", "v1": "aggression", "v2": "longevity", "invert": True},
        {"name": "Fear predicts technology (shelter levels)", "v1": "avg_fear", "v2": "shelter_level"},
        {"name": "Curiosity predicts discoveries count", "v1": "curiosity", "v2": "discoveries"},
        {"name": "Technology (shelter quality) predicts longevity", "v1": "shelter_level", "v2": "longevity"},
        {"name": "Frustration suppresses cooperation (Inverted)", "v1": "avg_frustration", "v2": "shares", "invert": True}
    ]

    all_summaries = []
    for manifest, summaries in exps:
        all_summaries.extend(summaries)

    global_hyps = []
    for hyp in hypotheses:
        r_vals = []
        for s in all_summaries:
            corr = s.get("correlations", {})
            p_matrix = corr.get("pearson", {})
            v1, v2 = hyp["v1"], hyp["v2"]
            r_val = 0.0
            if v1 in p_matrix and v2 in p_matrix[v1]:
                r_val = p_matrix[v1][v2]
            elif v2 in p_matrix and v1 in p_matrix[v2]:
                r_val = p_matrix[v2][v1]
            r_vals.append(r_val)

        r_avg = float(np.mean(r_vals)) if r_vals else 0.0
        abs_r = abs(r_avg)
        if abs_r >= 0.7:
            strength = "Very Strong Evidence"
        elif abs_r >= 0.5:
            strength = "Strong Evidence"
        elif abs_r >= 0.3:
            strength = "Moderate Evidence"
        elif abs_r >= 0.1:
            strength = "Weak Evidence"
        else:
            strength = "Negligible / No Evidence"

        global_hyps.append({
            "name": hyp["name"],
            "r_avg": r_avg,
            "strength": strength,
            "abs_r": abs_r
        })
    global_hyps.sort(key=lambda x: x["abs_r"], reverse=True)

    report += """
---

## 🔬 Global Hypothesis Discovery Rankings

Across all compiled experiments and unique topographic seeds, the engine has aggregated correlation coefficients to rank the overall evolutionary predictors of colony survival:

| Rank | Hypothesis | Global Avg Correlation (r) | Aggregated Evidence Strength |
| :---: | :--- | :---: | :--- |
"""
    for rank, h in enumerate(global_hyps):
        report += f"| {rank+1} | {h['name']} | `{h['r_avg']:.3f}` | **{h['strength']}** |\n"

    with open(os.path.join(research_dir, "meta_analysis.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n[Meta-Analysis Engine] Global meta-analysis file updated in research/meta_analysis.md")

# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------

def run_experiment_suite(level: int, ticks: int = 1000, scarcity: float = 1.0, ablation_name: str = "none"):
    """Configures and runs an experiment batch matching the selected Level."""
    print("="*80)
    print("                  GENESIS COGNITIVE RESEARCH EXPERIMENT FRAMEWORK              ")
    print("="*80)
    
    exp_id = generate_uuid()
    git_commit = get_git_commit()
    date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Define Ablations dictionary
    default_ablations = {
        "planner": True,
        "emotion": True,
        "relationships": True,
        "memory_importance": True,
        "motivation": True,
        "prediction_error": True
    }
    
    if ablation_name.lower() == "all":
        # Control group (all disabled)
        ablations = {k: False for k in default_ablations}
    elif ablation_name.lower() in default_ablations:
        ablations = default_ablations.copy()
        ablations[ablation_name.lower()] = False
    else:
        ablations = default_ablations.copy()
        
    # 2. Determine configuration based on level
    if level == 1:
        level_name = "Level 1 (Verification)"
        seeds = [1010, 1020, 1030]
        run_ticks = ticks if ticks != 1000 else 1000
    elif level == 2:
        level_name = "Level 2 (Validation)"
        seeds = list(range(1200, 1220)) # 20 runs
        run_ticks = ticks if ticks != 1000 else 5000
    elif level == 3:
        level_name = "Level 3 (Discovery)"
        seeds = list(range(1500, 1550)) # 50 runs
        run_ticks = ticks if ticks != 1000 else 10000
    else:
        print(f"Error: Unknown research level: {level}")
        return
        
    print(f"Experiment ID   : {exp_id}")
    print(f"Spawn Mode      : {SPAWN_MODE.upper()}")
    print(f"Research Level  : {level_name}")
    print(f"Run Length      : {run_ticks} ticks")
    print(f"Seeds Sweep     : {seeds}")
    print(f"Scarcity Level  : {scarcity}")
    print(f"Ablation sweep  : {ablation_name}")
    
    # 3. Write manifest
    manifest = {
        "experiment_id": exp_id,
        "genesis_version": "Phase 8.4",
        "git_commit": git_commit,
        "date": date_now,
        "ticks": run_ticks,
        "scarcity": scarcity,
        "level": level_name,
        "ablations": ablations,
        "spawn_mode": SPAWN_MODE,
        "name": f"Ablation sweep: {ablation_name} under {level_name}"
    }
    
    exp_dir = os.path.join("research", exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    # 4. Batch simulation execution
    run_summaries = []
    all_behaviors = []
    all_chains = []
    
    for s in seeds:
        summary, log, behaviors, chains = run_headless_simulation(
            exp_id=exp_id,
            seed=s,
            ticks=run_ticks,
            scarcity=scarcity,
            ablation_dict=ablations,
            notes=manifest["name"],
            level_name=level_name
        )
        run_summaries.append(summary)
        all_behaviors.extend(behaviors)
        all_chains.extend(chains)
        
    # Write aggregated metrics
    with open(os.path.join(exp_dir, "emergent_behaviors.json"), "w", encoding="utf-8") as f:
        json.dump(all_behaviors, f, indent=2)
        
    with open(os.path.join(exp_dir, "events.json"), "w", encoding="utf-8") as f:
        json.dump(all_chains, f, indent=2)
        
    # 5. Format and save scientific report
    report = generate_experiment_report(exp_id, manifest, run_summaries, all_behaviors, all_chains)
    with open(os.path.join(exp_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\n[Experiment Manager] Scientific report written to research/{exp_id}/report.md")
    
    # 6. Re-generate global meta-analysis
    generate_global_meta_analysis()
    
    print("\nExperiment Suite Execution complete!")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genesis Research and Hypothesis Testing Suite.")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3], help="Research Level (1=Verification, 2=Validation, 3=Discovery).")
    parser.add_argument("--ticks", type=int, default=1000, help="Number of ticks per simulation run.")
    parser.add_argument("--scarcity", type=float, default=1.0, help="Wildlife and fertility scarcity multiplier.")
    parser.add_argument("--ablate", type=str, default="none", help="Ablate single subsystem: emotion, relationships, planner, memory_importance, motivation, prediction_error, or 'all' for complete control baseline.")
    
    args = parser.parse_args()
    
    run_experiment_suite(
        level=args.level,
        ticks=args.ticks,
        scarcity=args.scarcity,
        ablation_name=args.ablate
    )
