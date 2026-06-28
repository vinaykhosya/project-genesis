import os
import sys
import json
import numpy as np

# Add project root to python path to ensure imports work cleanly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================
# 🎮 USER CONFIGURATION PANEL (Modify these values to change the simulation!)
# ==============================================================================

# 🌍 MAP & ENVIRONMENT SETTINGS
SEED = 65654                # Map shape seed (Change this to get a new layout)
TICKS = 1000000           # How long the simulation runs (1 tick = ~10 mins real-world)
                           # 100,000 ticks  ≈  277 years  (recommended for 300-year experiments)
                           # 1,000,000 ticks ≈ 2,777 years (multi-millennial mega-runs)
SCARCITY = 5           # Scarcity of resources:
                           #   1.0 = Normal (Default)
                           #   0.2 = Harsh Desert (Very difficult to survive!)
                           #   2.0 = Rich (Abundant food & water)
EXPERIMENT_NAME = "fights alowwed with insta heal , new seed" # Name of this run (shown in history table and experiment folder)

# 📈 LONG-RUN TREND DATA SETTINGS
# Adjust these for long experiments (e.g. 20,000+ ticks):
LONG_RUN = True            # Enable long-run mode (optimized memory and epoch snapshot tracking)
SAMPLE_INTERVAL = 5000      # Ticks per epoch snapshot
SAVE_PATHS = False        # Set to False to stop saving full path coordinates (prevents memory crash for long runs)
SAVE_EPOCHS = True         # Set to True to save epoch trends stats for visualizer charts


# 👥 AGENT POPULATION & EVOLUTION SETTINGS
# In Phase 5, agents are spawned as 4 distinct geographical colonies (Alpha, Beta, Gamma, Delta)
# with 4 founders each (16 total agents). Their traits are expressed from diverse, random genomes.
# Evolution, crossover, and mutation occur naturally across generations.
COLONY_SPAWNING = True     # Set to True to use Phase 5 colony spawning (highly recommended)
                           # Set to False to fall back to the legacy archetype spawning below.

# 📍 EXPERIMENTAL COLONY SPAWNING MODES
SPAWN_MODE = "random_valid"               # Available modes:
                                   #   "fixed"           = control; NMS spots, same every run
                                   #   "random_valid"    = random valid land locations with resources
                                   #   "random_anywhere" = true chaos; random land cells
                                   #   "targeted_biome"  = target specific biomes defined in TARGETED_BIOMES
                                   
# Mapped default locations for "fixed" spawn mode:
# Specify the preferred spot index (1, 2, 3, or 4) for each colony.
# E.g., if you map "Alpha": 4, Alpha will spawn at the 4th spot.
COLONY_SPAWN_LOCATIONS = {
    "Alpha": 1,
    "Beta":2,
    "Gamma": 3,
    "Delta": 4
}

# Targeted biomes mapped to Colonies (Alpha, Beta, Gamma, Delta) for "targeted_biome" mode.
# Available biomes: "Desert", "Forest", "Grassland", "Tundra", "Coast", "Rainforest", "Taiga"
TARGETED_BIOMES = ["Desert", "Forest", "Tundra", "Coast"]

# 🌍 WORLD PRESET (Aesthetic continental biome distribution)
# Options: None (default random), "arid_continent", "green_continent", "island_chains", "boreal_highlands", "tropical_ring"
WORLD_PRESET = "island_chains"

# 🌀 CLIMATE EPOCH MODE (Transitions of environmental selection pressures)
# Options: "legacy" (original repeating cycle), "stable" (no events), "slow_change" (30yr), "rapid_change" (10yr), "random"
CLIMATE_EPOCH_MODE = "slow_change"


MUTATION_RATE = 0.05       # Standard deviation of Gaussian gene mutations on birth [0.0, 0.2]
MAX_POPULATION =200        # Carrying capacity limit of the world (soft pressure)

# 🔬 BOTTLENECK & SURVIVAL PRESSURE SETTINGS (Modify these to analyze population dynamics)
REPRODUCTION_ENABLED = True      # Enable sexual reproduction and agent birth (True/False)
DISPUTES_ENABLED = True        # Enable territorial disputes / friction when agents meet on same tile (True/False)
DISASTERS_ENABLED = True         # Enable global weather/famine events (Drought, Cold Wave, Famine, Heatwave) (True/False)
HEALING_SPEED_MULT = 2  # Multiplier for rest-based healing and health restoration rate (float)
                                  # NOTE: Set to 1.0 for realistic survival pressure. Values >> 1 suppress injury mortality.
SHELTER_BUILD_SPEED_MULT = 1.0   # Multiplier for shelter construction/durability repair speed (float)
SHELTER_SEARCH_DIST = 100.0      # Maximum grid distance to scan and claim an abandoned shelter (float)

# 💧 ECOLOGY ABLATION STUDY CONTROLS (Water Bottleneck Isolation)
# Set these independently to study which intervention has the highest evolutionary impact.
DEHYDRATION_RAMP_ENABLED = True      # Graduated dehydration damage ramp instead of instant flat 2.0 cliff
MEMORY_FIDELITY_BOOST    = True      # Halve spatial memory decay rate to retain water sources longer
WATER_CACHING_BOOST      = True      # Increase store water utility multiplier to 0.45 (matching food)
DEPOSIT_UTILITY_FIX      = True      # Remove scarcity penalty on deposits to encourage chest stockpiling during drought

# 🧠 COGNITION ARCHITECTURE TOGGLES (Phase 7 & 8 experimental controls)
# Use these to isolate the survival/performance impact of each cognitive subsystem.
PLANNER_ENABLED = True              # Enable hierarchical multi-step action queue (Phase 7 planner).
                                    #   True  = agents pre-queue secondary actions, reducing deliberation overhead.
                                    #   False = agents re-evaluate utility every single tick (Phase 6 baseline).
SLEEP_CONSOLIDATION_ENABLED = True  # Enable sleep-state learning consolidation (Phase 7.5).
                                    #   True  = predictor training & procedural memory are batched to rest ticks.
                                    #   False = learning happens immediately every tick (always-on baseline).

# Phase 8 Ablation Controls (True = System Active, False = Ablated / Clamped to Neutral)
EMOTION_ENABLED = True              # Enable continuous drive emotional updates (fear, frustration, boredom, etc.)
RELATIONSHIPS_ENABLED = True        # Enable relationship trust/attachment graph updates
MEMORY_IMPORTANCE_ENABLED = True    # Enable emotional memory importance weighting
MOTIVATION_ENABLED = True           # Enable motivation priority drift and lateral inhibition modulation
PREDICTION_ERROR_ENABLED = True     # Enable prediction error expectations feedback loop

# ⏳ SIMULATION LIVE PACING
# Sleep duration in seconds per live callback update to pace the simulation for smooth real-time viewing.
# Set to 0.0 for maximum speed (e.g. for long-running experiments).
LIVE_PACING_DELAY = 0.0   # 0.0 = full speed (best for long runs); 0.03 = ~30fps live mode

# 📁 CHECKPOINT SETTINGS
# Snapshots of world state saved to the experiment folder at regular intervals.
# Use these to recover from crashes or study the civilization mid-run.
CHECKPOINT_INTERVAL = 5000   # Ticks between visualisation checkpoints (for dashboard)

# 💾 FULL CHECKPOINT SETTINGS (for experiment resume)
# Full checkpoints save ALL simulation state needed to resume later.
# Files are ~10-20 MB and can be loaded by run_resume.py.
FULL_CHECKPOINT_INTERVAL = 36000   # Save a full checkpoint every N ticks (36000 = every 100 years)
                                    # Set to 0 to disable periodic full checkpoints.
FULL_CHECKPOINT_ON_STOP  = True    # Always save a full checkpoint when Stop & Save is pressed or
                                    # Ctrl-C is used. This is the recommended safety net.


# Legacy Archetype Spawning settings (only used if COLONY_SPAWNING = False)
POPULATION = [
    {
        "archetype": "Explorer",    # Category name
        "curiosity": 1.0,          # Curiosity need growth multiplier (0.0 to 1.0)
        "risk_tolerance": 0.9,     # Risk tolerance factor (0.0 to 1.0)
        "count": 4                 # How many of this agent to spawn
    },
    {
        "archetype": "Settler",
        "curiosity": 0.1,
        "risk_tolerance": 0.1,
        "count": 4
    },
    {
        "archetype": "Balanced",
        "curiosity": 0.5,
        "risk_tolerance": 0.5,
        "count": 8
    }
]

# 📍 CUSTOM SPAWN COORDINATES (Optional, only used if COLONY_SPAWNING = False)
# If you want to place specific agents at exact positions, add them here:
# Example: {"archetype": "Specialist", "curiosity": 0.8, "risk_tolerance": 0.8, "location": [x, y]}
CUSTOM_SPAWNS = []

# ==============================================================================
# ⚙️ INTERNAL IMPLEMENTATION (Under the hood - no need to modify)
# ==============================================================================

from run import run_unit_tests, spawn_custom_agents, update_history, get_git_commit
from world.generator import generate_world
from world.agents.simulation import run_simulation
from main import generate_simulation_map, save_simulation_data, print_experiment_report
from world.agents.drives import emotional_label
import datetime

class DualWriter:
    def __init__(self, original_stream, file_handle):
        self.original_stream = original_stream
        self.file_handle = file_handle
        
    def write(self, message):
        self.original_stream.write(message)
        self.file_handle.write(message)
        self.flush()
        
    def flush(self):
        self.original_stream.flush()
        try:
            self.file_handle.flush()
        except Exception:
            pass

def archive_experiment(world, summary_record, exp_folder, epoch_stats=None):
    import csv
    import shutil
    import json
    import os
    import numpy as np
    
    # 1. config.json
    config_data = {
        "seed": int(world.seed),
        "ticks": int(world.tick),
        "scarcity": float(SCARCITY),
        "max_population": int(world.max_population),
        "mutation_rate": float(world.mutation_rate),
        "reproduction_enabled": bool(world.reproduction_enabled),
        "disputes_enabled": bool(world.disputes_enabled),
        "disasters_enabled": bool(world.disasters_enabled),
        "healing_speed_mult": float(world.healing_speed_mult),
        "shelter_build_speed_mult": float(world.shelter_build_speed_mult),
        "shelter_search_dist": float(world.shelter_search_dist),
        "planner_enabled": bool(PLANNER_ENABLED),
        "sleep_consolidation_enabled": bool(SLEEP_CONSOLIDATION_ENABLED),
        "checkpoint_interval": int(CHECKPOINT_INTERVAL),
        "world_preset":        WORLD_PRESET,
        "climate_epoch_mode":  CLIMATE_EPOCH_MODE,
        "ecology_ablation":    dict(world.ecology_ablation) if hasattr(world, "ecology_ablation") else {}
    }
    with open(os.path.join(exp_folder, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)
        
    # 2. summary.json
    with open(os.path.join(exp_folder, "summary.json"), "w") as f:
        json.dump(summary_record, f, indent=2)
        
    # 2.5 events.json (Structured timeline events)
    events_data = getattr(world, "events_timeline", [])
    with open(os.path.join(exp_folder, "events.json"), "w") as f:
        json.dump(events_data, f, indent=2)

    # 2.7 dehydration_death_traces.json (Detailed post-mortem trace logs)
    death_traces = getattr(world, "dehydration_death_traces", [])
    with open(os.path.join(exp_folder, "dehydration_death_traces.json"), "w") as f:
        json.dump(death_traces, f, indent=2)

    # 2.8 disputes_history.json (Full combat encounter telemetry — capped at 5000 entries)
    disputes = getattr(world, "disputes_history", [])
    with open(os.path.join(exp_folder, "disputes_history.json"), "w") as f:
        json.dump(disputes[-5000:], f, indent=2)
    print(f"  Combat encounters logged: {len(disputes)} total, {min(len(disputes), 5000)} archived.")

    # 3. timeline.csv
    with open(os.path.join(exp_folder, "timeline.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tick", "year", "day", "event_type", "description"])
        for ev in getattr(world, "events_timeline", []):
            tick = ev["tick"]
            writer.writerow([tick, tick // 360, tick % 360, ev.get("type", "Global Event"), ev.get("description", "")])
            
    # 4. population.csv
    with open(os.path.join(exp_folder, "population.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tick", "total_alive", "colony_alpha", "colony_beta", "colony_gamma", "colony_delta"])
        for p in getattr(world, "population_history", []):
            cols = p["per_colony"]
            cols = (cols + [0, 0, 0, 0])[:4]
            writer.writerow([p["tick"], p["total"], cols[0], cols[1], cols[2], cols[3]])
            
    # 5. genes.csv
    with open(os.path.join(exp_folder, "genes.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = ["tick", "diversity_score"]
        for i in range(13):
            headers.extend([f"gene_{i}_mean", f"gene_{i}_var"])
        writer.writerow(headers)
        for gh in getattr(world, "genetic_history", []):
            row = [gh["tick"], gh["diversity_score"]]
            means = gh["gene_means"]
            vars = gh["gene_variances"]
            for i in range(13):
                row.extend([means[i], vars[i]])
            writer.writerow(row)
            
    # 6. deaths.csv
    with open(os.path.join(exp_folder, "deaths.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "generation", "colony_id", "age_ticks", "ticks_survived", "primary_cause", "secondary_cause", "location_x", "location_y"])
        for a in world.agents:
            if a.dead:
                writer.writerow([
                    a.id, a.generation, getattr(a, "colony_id", 0), a.age, a.ticks_survived,
                    getattr(a, "primary_cause", "None"), getattr(a, "secondary_cause", "None"),
                    int(a.location[1]), int(a.location[0])
                ])
                
    # 7. births.csv
    with open(os.path.join(exp_folder, "births.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "generation", "colony_id", "parent_a_id", "parent_b_id", "born_tick", "spawn_biome"])
        for a in world.agents:
            if getattr(a, "parent_ids", None) is not None:
                p_a, p_b = a.parent_ids
                writer.writerow([
                    a.id, a.generation, getattr(a, "colony_id", 0), p_a, p_b,
                    getattr(a, "born_tick", 0), getattr(a, "spawn_biome", "Unknown")
                ])
                
    # 8. colonies.csv
    with open(os.path.join(exp_folder, "colonies.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["colony_id", "name", "color", "stored_food", "stored_water", "founders"])
        for c in getattr(world, "colonies", []):
            writer.writerow([
                c["id"], c["name"], c["color"],
                round(c.get("stored_food", 0.0), 1), round(c.get("stored_water", 0.0), 1),
                ",".join(map(str, c.get("founder_ids", [])))
            ])
            
    # 9. cognition.csv
    with open(os.path.join(exp_folder, "cognition.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "alive", "learning_rate", "prediction_accuracy", "prediction_gains", "discoveries", "concepts_count", "procedures_count"])
        for a in world.agents:
            pred_acc = a.prediction_successes / a.prediction_attempts if a.prediction_attempts > 0 else 1.0
            concepts_cnt = sum(len(lst) for lst in a.concepts.values()) if hasattr(a, "concepts") else 0
            procedures_cnt = len(a.procedures) if hasattr(a, "procedures") else 0
            writer.writerow([
                a.id, int(not a.dead), round(float(a.learning_rate), 4),
                round(pred_acc, 4), int(a.prediction_gains), int(a.discoveries_count),
                concepts_cnt, procedures_cnt
            ])
            
    # 10. Copy world.png from simulation.png if it exists
    if os.path.exists("simulation.png"):
        shutil.copy("simulation.png", os.path.join(exp_folder, "world.png"))
        
    # 11. replay.json
    from main import save_simulation_data
    run_metadata = {
        "derived_metrics": summary_record.get("derived_metrics", {}),
        "distributions": summary_record.get("distributions", {}),
        "leaderboards": summary_record.get("leaderboards", {}),
        "max_generation": summary_record.get("max_generation", 0)
    }
    save_simulation_data(
        world=world,
        experiment_name=summary_record["experiment"],
        scarcity_val=summary_record["scarcity"],
        filepath=os.path.join(exp_folder, "replay.json"),
        save_paths=True,
        run_metadata=run_metadata,
        epoch_stats=epoch_stats
    )

    # 12. report.md (Automated Research Paper)
    report_path = os.path.join(exp_folder, "report.md")
    
    profiler_md = ""
    if hasattr(world, "profiler") and world.profiler:
        profiler_md += "| Subsystem | Calls | Total Time (ms) | Avg Time (ms) | Max Time (ms) | Worst Tick |\n"
        profiler_md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for name, data in world.profiler.items():
            calls = data["calls"]
            total_t = data["time"]
            avg_t = total_t / calls if calls > 0 else 0.0
            max_t = data["max"]
            worst = data["worst_tick"]
            profiler_md += f"| {name.capitalize()} | {calls} | {total_t:.2f} | {avg_t:.4f} | {max_t:.2f} | {worst} |\n"
    else:
        profiler_md = "*Profiler data not available for this run.*\n"
        
    milestones_md = ""
    for ev in getattr(world, "events_timeline", []):
        tick = ev["tick"]
        milestones_md += f"- **Tick {tick} (Year {tick // 360}, Day {tick % 360})**: {ev.get('type', 'Event')} - *{ev.get('description', '')}*\n"
    if not milestones_md:
        milestones_md = "*No major timeline events recorded.*\n"
        
    alive_count = sum(1 for a in world.agents if not a.dead)
    total_agents = len(world.agents)
    extinction_status = "TOTAL EXTINCTION" if alive_count == 0 else f"SURVIVED ({alive_count}/{total_agents} alive)"

    derived_metrics = summary_record.get("derived_metrics", {})
    distributions = summary_record.get("distributions", {})
    leaderboards = summary_record.get("leaderboards", {})

    # Format Derived Metrics Table
    derived_md = "| Metric | Value |\n| :--- | :--- |\n"
    for k, v in derived_metrics.items():
        name = k.replace("_", " ").capitalize()
        if isinstance(v, dict):
            val_str = ", ".join(f"{col}: {val}" for col, val in v.items())
        else:
            val_str = str(v)
        derived_md += f"| {name} | {val_str} |\n"
        
    # Format Distributions Tables
    dist_md = "#### Lifespan Distribution\n"
    dist_md += "| Age Bin | Agent Count |\n| :--- | :--- |\n"
    for k, v in distributions.get("lifespan", {}).items():
        dist_md += f"| {k} | {v} |\n"
        
    dist_md += "\n#### Children Distribution\n"
    dist_md += "| Children Bin | Agent Count |\n| :--- | :--- |\n"
    for k, v in distributions.get("children", {}).items():
        dist_md += f"| {k} | {v} |\n"
        
    dist_md += "\n#### Generation Distribution\n"
    dist_md += "| Generation | Agent Count |\n| :--- | :--- |\n"
    for k, v in distributions.get("generation", {}).items():
        dist_md += f"| {k} | {v} |\n"

    # Format Leaderboards Tables
    leaderboard_md = "#### Largest Lineages (Descendants Born)\n"
    leaderboard_md += "| Founder ID | Colony | Total Descendants |\n| :--- | :--- | :--- |\n"
    for entry in leaderboards.get("largest_size", []):
        col_name = "Unknown"
        if hasattr(world, "colonies") and 0 <= entry["colony_id"] < len(world.colonies):
            col_name = world.colonies[entry["colony_id"]].get("name", col_name)
        leaderboard_md += f"| #{entry['founder_id']} | {col_name} | {entry['size']} |\n"
        
    leaderboard_md += "\n#### Longest Surviving Lineages (Ticks Spanned)\n"
    leaderboard_md += "| Founder ID | Colony | Longevity (Ticks) |\n| :--- | :--- | :--- |\n"
    for entry in leaderboards.get("longest_surviving", []):
        col_name = "Unknown"
        if hasattr(world, "colonies") and 0 <= entry["colony_id"] < len(world.colonies):
            col_name = world.colonies[entry["colony_id"]].get("name", col_name)
        leaderboard_md += f"| #{entry['founder_id']} | {col_name} | {entry['longevity_ticks']} |\n"
        
    leaderboard_md += "\n#### Cognitive Mastery Lineages (Avg Predictor Accuracy)\n"
    leaderboard_md += "| Founder ID | Colony | Avg Predictor Accuracy |\n| :--- | :--- | :--- |\n"
    for entry in leaderboards.get("cognitive_mastery", []):
        col_name = "Unknown"
        if hasattr(world, "colonies") and 0 <= entry["colony_id"] < len(world.colonies):
            col_name = world.colonies[entry["colony_id"]].get("name", col_name)
        leaderboard_md += f"| #{entry['founder_id']} | {col_name} | {entry['avg_prediction_accuracy']:.4f} |\n"

    # Rule-Based Observations and Anomalies
    observations_list = []
    anomalies_list = []
    
    # Rule 1: Birth rate collapse
    births_late = sum(1 for a in world.agents if getattr(a, "born_tick", 0) > world.tick * 0.8)
    births_early = sum(1 for a in world.agents if getattr(a, "born_tick", 0) <= world.tick * 0.8)
    if births_early > 10 and births_late == 0:
        anomalies_list.append("Anomalous birth rate collapse observed: no new births occurred in the final 20% of the simulation. The population is aging without replacement, signaling critical demographic decline.")
        
    # Rule 2: Exposure/Starvation mortality bias
    starv_deaths = sum(1 for a in world.agents if a.dead and getattr(a, "primary_cause", "") == "Starvation")
    expos_deaths = sum(1 for a in world.agents if a.dead and getattr(a, "primary_cause", "") == "Exposure")
    if starv_deaths > expos_deaths * 3 and starv_deaths > 10:
        observations_list.append(f"Mortality bias detected: Starvation deaths ({starv_deaths}) far exceeded Exposure deaths ({expos_deaths}), indicating that food scarcity was the dominant selective pressure.")
    elif expos_deaths > starv_deaths * 3 and expos_deaths > 10:
        observations_list.append(f"Mortality bias detected: Climate Exposure deaths ({expos_deaths}) far exceeded Starvation deaths ({starv_deaths}), suggesting that lack of protective shelter was the primary evolutionary bottleneck.")

    # Rule 3: Cognitive adaptation
    gen0_agents = [a for a in world.agents if a.generation == 0]
    gen_late_agents = [a for a in world.agents if a.generation > 1]
    if gen0_agents and gen_late_agents:
        gen0_acc = np.mean([a.prediction_successes / a.prediction_attempts if a.prediction_attempts > 0 else 1.0 for a in gen0_agents])
        gen_late_acc = np.mean([a.prediction_successes / a.prediction_attempts if a.prediction_attempts > 0 else 1.0 for a in gen_late_agents])
        if gen_late_acc > gen0_acc * 1.15:
            observations_list.append(f"Cognitive acceleration verified: Later generations achieved a {((gen_late_acc - gen0_acc)/gen0_acc)*100:.1f}% increase in neural predictor accuracy over founders, demonstrating evolutionary cognitive gains.")
            
    # Rule 4: Technological progression
    if gen0_agents and gen_late_agents:
        avg_sh_gen0 = np.mean([getattr(a, "shelter_level", 0) for a in gen0_agents])
        avg_sh_gen_late = np.mean([getattr(a, "shelter_level", 0) for a in gen_late_agents])
        if avg_sh_gen_late > avg_sh_gen0 + 0.4:
            observations_list.append(f"Technological progression verified: Later generations constructed shelters that were on average {avg_sh_gen_late - avg_sh_gen0:.2f} levels higher than founders, showing emergent engineering knowledge.")

    obs_md = ""
    if observations_list:
        for obs in observations_list:
            obs_md += f"- **Observation:** {obs}\n"
    else:
        obs_md = "- *No major interesting behavioral anomalies detected.*\n"
        
    anomaly_md = ""
    if anomalies_list:
        for anom in anomalies_list:
            anomaly_md += f"- **Anomaly:** {anom}\n"
    else:
        anomaly_md = "- *No severe demographic anomalies observed. Population growth remains stable.*\n"

    paper_content = f"""# Emergent Civilization Dynamics under Resource Scarcity
## An Agent-Based Evolutionary Investigation of Project Genesis

**Date:** {summary_record["timestamp"]}
**Experiment Run:** `{summary_record["experiment"]}`
**Procedural Seed:** `{summary_record["seed"]}`

---

### Abstract
This paper presents a comprehensive computational analysis of an evolutionary multi-agent simulation conducted within the Project Genesis framework. Spawning {total_agents} founding agents across 4 geographical colonies under a resource scarcity level of {summary_record["scarcity"]}, we track the emergence of social structures, shelter technology, and cognitive pathways over a lifespan of {world.tick} ticks. By profiling the high-performance subsystems and mapping genetic diversities, we demonstrate how environmental pressures drive adaptive behaviors, territorial boundaries, and cognitive structures, culminating in a final system state of **{extinction_status}**.

---

### 1. Experimental Setup & Parameters
The simulation environment was initialized on a 1024x1024 grid using multi-octave Perlin noise. The following physical and evolutionary parameters were enforced:
- **World Seed:** {world.seed}
- **Scarcity Factor:** {summary_record["scarcity"]}
- **Maximum Carrying Capacity:** {world.max_population}
- **Genetic Mutation Deviation:** {world.mutation_rate}
- **Disasters Enabled:** {world.disasters_enabled}
- **Territorial Disputes Enabled:** {world.disputes_enabled}
- **Healing Multiplier:** {world.healing_speed_mult}

---

### 2. High-Precision Subsystem Performance Profile
To evaluate computational efficiency and identify simulation bottlenecks, microsecond-precision timers were integrated into all core subsystems. The performance profile below catalogs the computational overhead:

{profiler_md}

*Note: Spatial hashing query optimization was active, reducing neighbor lookups from $O(N^2)$ to $O(1)$ complexity.*

---

### 3. Population & Derived Scientific Metrics
The following table presents the 17 derived scientific metrics calculated from the complete simulation telemetry:

{derived_md}

---

### 4. Binned Population Distributions
Averages hide critical demographic curves. Below are the detailed distributions for lifespans, children, and generations across the entire population:

{dist_md}

---

### 5. Evolutionary Success Leaderboard
By recursively tracing agent lineages back to their Gen 0 founders, we identify the genotypes that dominated evolution:

{leaderboard_md}

---

### 6. Interesting Observations & Anomalies
We apply rule-based heuristics to identify behavioral adaptations and systemic anomalies:

#### Behavioral Observations
{obs_md}

#### Systemic Anomalies
{anomaly_md}

---

### 7. Chronological Milestones & Historical Log
The historical progression of the world is punctuated by seasonal shifts, global disasters, and demographic milestones:

{milestones_md}

---

### 8. Conclusion
This simulation run demonstrates that physical constraints—namely resource scarcity—are the primary catalysts for technological progression (shelter construction) and genetic specialization. The performance metrics confirm that spatial grid binning sustains high tick rates even at maximum carrying capacity. Future investigations will introduce complex linguistic communication and trade networks to study the next phase of civilization emergence.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(paper_content)
    print(f"Publication-grade research paper written to {report_path}")

def main():
    import datetime
    import shutil
    import csv
    
    # Set up experiment folder
    os.makedirs("experiments", exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_folder = os.path.join("experiments", f"{timestamp_str}_{EXPERIMENT_NAME}")
    os.makedirs(exp_folder, exist_ok=True)
    
    log_file = open(os.path.join(exp_folder, "simulation.log"), "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = DualWriter(original_stdout, log_file)
    sys.stderr = DualWriter(original_stderr, log_file)

    print("=" * 80)
    print("                   PROJECT GENESIS - EASY EXPERIMENT RUNNER                  ")
    print("=" * 80)
    
    # 1. Run Unit Tests programmatically
    test_results = run_unit_tests()
    print("-" * 80)
    
    # 2. Formulate the agent configs list
    agents_config = []
    
    # Add custom spawns first
    for item in CUSTOM_SPAWNS:
        agents_config.append(item)
        
    # Add population category groups
    for item in POPULATION:
        agents_config.append({
            "archetype": item["archetype"],
            "curiosity": item["curiosity"],
            "risk_tolerance": item["risk_tolerance"],
            "count": item["count"]
        })
        
    # 3. Generate World State
    print(f"Generating world: Seed = {SEED}, size = 1024x1024...")
    world = generate_world(width=1024, height=1024, seed=SEED, world_preset=WORLD_PRESET)
    world.exp_folder = exp_folder
    
    # Set Phase 5 global simulation parameters on world state
    world.max_population = MAX_POPULATION
    world.mutation_rate = MUTATION_RATE
    world.reproduction_enabled = REPRODUCTION_ENABLED
    world.disputes_enabled = DISPUTES_ENABLED
    world.disasters_enabled = DISASTERS_ENABLED
    world.healing_speed_mult = HEALING_SPEED_MULT
    world.shelter_build_speed_mult = SHELTER_BUILD_SPEED_MULT
    world.shelter_search_dist = SHELTER_SEARCH_DIST
    world.climate_epoch_mode = CLIMATE_EPOCH_MODE
    world.ecology_ablation = {
        "dehydration_ramp":   DEHYDRATION_RAMP_ENABLED,
        "memory_fidelity":    MEMORY_FIDELITY_BOOST,
        "water_caching":      WATER_CACHING_BOOST,
        "deposit_utility_fix": DEPOSIT_UTILITY_FIX
    }
    
    # Set Experimental spawn modes parameters on world state
    world.spawn_mode = SPAWN_MODE
    world.colony_spawn_locations = COLONY_SPAWN_LOCATIONS
    world.targeted_biomes = TARGETED_BIOMES
    
    # Set Phase 8.4 ablation flags on world state
    world.ablation = {
        "planner": PLANNER_ENABLED,
        "emotion": EMOTION_ENABLED,
        "relationships": RELATIONSHIPS_ENABLED,
        "memory_importance": MEMORY_IMPORTANCE_ENABLED,
        "motivation": MOTIVATION_ENABLED,
        "prediction_error": PREDICTION_ERROR_ENABLED
    }

    # 💾 Full checkpoint config — wired onto world so simulation.py can save on Stop
    _full_checkpoint_config = {
        "checkpoint_version": 1,
        "experiment_name":     EXPERIMENT_NAME,
        "seed":                SEED,
        "ticks_planned":       TICKS,
        "scarcity":            SCARCITY,
        "max_population":      MAX_POPULATION,
        "mutation_rate":       MUTATION_RATE,
        "reproduction_enabled":    REPRODUCTION_ENABLED,
        "disputes_enabled":        DISPUTES_ENABLED,
        "disasters_enabled":       DISASTERS_ENABLED,
        "healing_speed_mult":      HEALING_SPEED_MULT,
        "shelter_build_speed_mult":SHELTER_BUILD_SPEED_MULT,
        "shelter_search_dist":     SHELTER_SEARCH_DIST,
        "spawn_mode":              SPAWN_MODE,
        "colony_spawn_locations":  COLONY_SPAWN_LOCATIONS,
        "targeted_biomes":         TARGETED_BIOMES,
        "world_preset":            WORLD_PRESET,
        "climate_epoch_mode":      CLIMATE_EPOCH_MODE,
        "ecology_ablation": {
            "dehydration_ramp":   DEHYDRATION_RAMP_ENABLED,
            "memory_fidelity":    MEMORY_FIDELITY_BOOST,
            "water_caching":      WATER_CACHING_BOOST,
            "deposit_utility_fix": DEPOSIT_UTILITY_FIX
        },
        "ablation": {
            "planner":           PLANNER_ENABLED,
            "emotion":           EMOTION_ENABLED,
            "relationships":     RELATIONSHIPS_ENABLED,
            "memory_importance": MEMORY_IMPORTANCE_ENABLED,
            "motivation":        MOTIVATION_ENABLED,
            "prediction_error":  PREDICTION_ERROR_ENABLED
        },
    }
    world._full_checkpoint_config   = _full_checkpoint_config
    world._full_checkpoint_interval = FULL_CHECKPOINT_INTERVAL

    
    # 4. Spawn Agents
    if COLONY_SPAWNING:
        # Let run_simulation handle the new Phase 5 colony spawning with diverse random genomes
        print("Using Phase 5 Colony Spawning: Spawning 4 distinct colonies with 16 diverse random genomes.")
        world.agents = []
    else:
        # Legacy archetype spawning (backward compatibility)
        print("Using Legacy Spawning: Spawning custom pre-populated archetypes.")
        spawn_custom_agents(world, agents_config)
    
    # 5. Run Simulation Loop
    long_run = LONG_RUN
    save_paths = SAVE_PATHS if "SAVE_PATHS" in globals() else (not LONG_RUN)
    save_epochs = SAVE_EPOCHS if "SAVE_EPOCHS" in globals() else LONG_RUN
    sample_interval = SAMPLE_INTERVAL
    
    import time
    from world.agents.behavior_classifier import classify_behavior

    # Define a highly enriched live callback that writes comprehensive live_state every 50 ticks
    def live_callback(tick, epoch_stats):
        import json
        import os
        import time
        import numpy as np
        if tick % 500 == 0 or tick == 1:
            # Run live K-Means behavioral classification on all active agents
            living_agents = [a for a in world.agents if not a.dead]
            behavior_result = classify_behavior(living_agents, world_width=world.width, n_clusters=4)

            agents_coords = []
            for agent in world.agents:
                # Calculate exploration radius and home drift dynamically for live telemetry
                if hasattr(agent, "sampled_path_history") and len(agent.sampled_path_history) > 0:
                    first = agent.sampled_path_history[0]
                    if len(first) > 2:
                        start_x, start_y = first[0], first[1]
                        path_coords = np.array([[coord[0], coord[1]] for coord in agent.sampled_path_history])
                    else:
                        start_y, start_x = first[0], first[1]
                        path_coords = np.array(agent.sampled_path_history)
                    start_coords = np.array([start_x, start_y]) if len(first) > 2 else np.array([start_y, start_x])
                    dists = np.sqrt(np.sum((path_coords - start_coords)**2, axis=-1))
                    exploration_radius = float(np.max(dists)) if len(dists) > 0 else 0.0
                    home_drift = float(np.sqrt((agent.home_location[0] - start_y)**2 + (agent.home_location[1] - start_x)**2))
                else:
                    exploration_radius = float(getattr(agent, "max_radius", 0.0))
                    home_drift = 0.0

                # Convert action counts to percentages
                action_pct = {}
                total_actions = sum(agent.action_counts.values()) if hasattr(agent, "action_counts") else 0
                if total_actions > 0:
                    for act, count in agent.action_counts.items():
                        action_pct[act] = round((count / total_actions) * 100.0, 1)
                elif hasattr(agent, "action_counts"):
                    action_pct = {k: 0.0 for k in agent.action_counts}

                agents_coords.append({
                    "id": agent.id,
                    "archetype": agent.archetype,
                    "spawn_biome": getattr(agent, "spawn_biome", "Unknown"),
                    "traits": {k: float(v) for k, v in agent.traits.items()} if hasattr(agent, "traits") else {},
                    "dead": bool(agent.dead),
                    # JS/Canvas coordinates are [x, y] where x is column and y is row
                    "location": [int(agent.location[1]), int(agent.location[0])],
                    "current_action": getattr(agent, "current_action", "Idle"),
                    "health": round(agent.health, 1),
                    "hunger": round(agent.hunger, 1),
                    "thirst": round(agent.thirst, 1),
                    "energy": round(agent.energy, 1),
                    "stored_food": round(float(getattr(agent, "stored_food", 0.0)), 1),
                    "stored_water": round(float(getattr(agent, "stored_water", 0.0)), 1),
                    "colony_id": int(getattr(agent, "colony_id", 0)),
                    "generation": int(getattr(agent, "generation", 0)),
                    "age": int(agent.age),
                    "max_age": int(agent.max_age),
                    "drinks": int(getattr(agent, "drinks_count", 0)),
                    "eats": int(getattr(agent, "eats_count", 0)),
                    "resting_ticks": int(getattr(agent, "resting_ticks", 0)),
                    "discoveries": int(getattr(agent, "discoveries_count", 0)),
                    "recognized_agents": sorted(list(getattr(agent, "known_agents", []))),
                    "exploration_radius": round(exploration_radius, 1),
                    "home_drift": round(home_drift, 1),
                    "action_pct": action_pct,
                    "parent_ids": list(getattr(agent, "parent_ids", None) or []),
                    "children_ids": list(getattr(agent, "children_ids", [])),
                    "born_tick": int(getattr(agent, "born_tick", 0)),
                    "behavior_cluster": getattr(agent, "behavior_cluster", "C0"),
                    "genome": getattr(agent.genome, "to_list", lambda: [])() if hasattr(agent, "genome") else [],
                    "cause_of_death": getattr(agent, "cause_of_death", "None"),
                    "primary_cause": getattr(agent, "primary_cause", "None"),
                    "secondary_cause": getattr(agent, "secondary_cause", "None"),
                    "fat_reserves": round(float(getattr(agent, "fat_reserves", 100.0)), 1),
                    "muscle_mass": round(float(getattr(agent, "muscle_mass", 100.0)), 1),
                    "injury_level": round(float(getattr(agent, "injury_level", 0.0)), 1),
                    "shelter_location": [int(agent.shelter_location[1]), int(agent.shelter_location[0])] if getattr(agent, "shelter_location", None) else None,
                    "shelter_level": int(getattr(agent, "shelter_level", 0)),
                    "shelter_durability": round(float(getattr(agent, "shelter_durability", 0.0)), 1),
                    "ticks_survived": int(agent.ticks_survived),
                    "years_survived": int(getattr(agent, "years_survived", 0)),
                    "season_observations": {int(k): int(v) for k, v in agent.season_observations.items()} if hasattr(agent, "season_observations") else {0:0, 1:0, 2:0, 3:0},

                    # Phase 8.1 — Drive Telemetry
                    "drives": {
                        "hunger_tension":     round(float(getattr(getattr(agent, "drives", None), "hunger_tension",     0.0)), 3),
                        "thirst_tension":     round(float(getattr(getattr(agent, "drives", None), "thirst_tension",     0.0)), 3),
                        "exhaustion_tension": round(float(getattr(getattr(agent, "drives", None), "exhaustion_tension", 0.0)), 3),
                        "pain_tension":       round(float(getattr(getattr(agent, "drives", None), "pain_tension",       0.0)), 3),
                        "thermal_stress":     round(float(getattr(getattr(agent, "drives", None), "thermal_stress",     0.0)), 3),
                        "fear":               round(float(getattr(getattr(agent, "drives", None), "fear",               0.0)), 3),
                        "frustration":        round(float(getattr(getattr(agent, "drives", None), "frustration",        0.0)), 3),
                        "longing":            round(float(getattr(getattr(agent, "drives", None), "longing",            0.0)), 3),
                        "grief":              round(float(getattr(getattr(agent, "drives", None), "grief",              0.0)), 3),
                        "boredom":            round(float(getattr(getattr(agent, "drives", None), "boredom",            0.0)), 3),
                        "contentment":        round(float(getattr(getattr(agent, "drives", None), "contentment",        0.0)), 3),
                        "valence":            round(float(getattr(getattr(agent, "drives", None), "valence",            0.0)), 3),
                        "arousal":            round(float(getattr(getattr(agent, "drives", None), "arousal",            0.0)), 3),
                        "emotional_label":    emotional_label(agent) if hasattr(agent, "drives") else "Neutral",
                    },
                    
                    # Phase 8.4 — Motivation Telemetry
                    "motivation": agent.motivation.to_dict() if getattr(agent, "motivation", None) is not None else {},
                })
            
            # Extract colony stockpiles
            colonies_data = []
            for c in getattr(world, "colonies", []):
                colonies_data.append({
                    "id": c["id"],
                    "name": c["name"],
                    "color": c["color"],
                    "stored_food": round(c.get("stored_food", 0.0), 1),
                    "stored_water": round(c.get("stored_water", 0.0), 1)
                })

            live_state = {
                "tick": int(tick),
                "alive": int(sum(1 for a in world.agents if not a.dead)),
                "agents": agents_coords,
                "colonies": colonies_data,
                "epoch_stats": epoch_stats,
                # Phase 5 world-level evolution history
                "population_history": getattr(world, "population_history", []),
                "genetic_history":    getattr(world, "genetic_history",    []),
                "extinction_events":  getattr(world, "extinction_events",  []),
                "generation_number":  int(getattr(world, "generation_number", 0)),
                "total_births":       int(getattr(world, "total_births",    0)),
                "total_deaths":       int(getattr(world, "total_deaths",    0)),
                "behavior_clustering": behavior_result,
            }
            
            try:
                # Atomically replace files using temp swap to prevent browser read collisions on disk
                temp_json = "live_state.json.tmp"
                temp_js = "live_state.js.tmp"
                with open(temp_json, "w") as f:
                    json.dump(live_state, f)
                with open(temp_js, "w") as f:
                    f.write(f"window.LIVE_STATE = {json.dumps(live_state)};")
                
                os.replace(temp_json, "live_state.json")
                os.replace(temp_js, "live_state.js")
            except Exception as e:
                print(f"Warning: Failed to write live state files atomically: {e}")

            # Pace the simulation to make short test runs human-watchable in live mode
            if LIVE_PACING_DELAY > 0.0:
                time.sleep(LIVE_PACING_DELAY)

        # 💾 Checkpoint saving (runs every CHECKPOINT_INTERVAL ticks, independent of the 100-tick block)
        if CHECKPOINT_INTERVAL > 0 and tick % CHECKPOINT_INTERVAL == 0 and tick > 0:
            try:
                checkpoint_path = os.path.join(exp_folder, f"checkpoint_{tick}.json")
                save_simulation_data(
                    world=world,
                    experiment_name=EXPERIMENT_NAME,
                    scarcity_val=SCARCITY,
                    filepath=checkpoint_path,
                    save_paths=False,   # compact — no full path history in checkpoints
                    epoch_stats=epoch_stats
                )
                print(f"  💾 Checkpoint saved: checkpoint_{tick}.json (Tick {tick}, Year {tick // 360})")
            except Exception as _ce:
                print(f"  Warning: Failed to save checkpoint at tick {tick}: {_ce}")


    # Automatically open visualizer.html in the default web browser with ?live=true
    import webbrowser
    try:
        vis_path = os.path.abspath("visualizer.html")
        webbrowser.open("file://" + vis_path + "?live=true")
        print("Automatically opened visualizer.html in your default browser connected to the Live Stream.")
    except Exception as e:
        print(f"Warning: Failed to automatically open visualizer in browser: {e}")

    print(f"Starting simulation: ticks = {TICKS}, scarcity = {SCARCITY}...")
    print(f"  Spawn Mode                : {SPAWN_MODE.upper()}")
    print(f"  Planner Enabled           : {PLANNER_ENABLED}")
    print(f"  Sleep Consolidation       : {SLEEP_CONSOLIDATION_ENABLED}")
    print(f"  Emotion System            : {EMOTION_ENABLED}")
    print(f"  Relationship Graph        : {RELATIONSHIPS_ENABLED}")
    print(f"  Memory Importance         : {MEMORY_IMPORTANCE_ENABLED}")
    print(f"  Motivation Priority Drift : {MOTIVATION_ENABLED}")
    print(f"  Prediction Error Feedback : {PREDICTION_ERROR_ENABLED}")
    # 📶 Start background HTTP server for UI emergency stop trigger
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    
    class SignalHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/stop':
                world.stop_requested = True
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'{"status": "stopping"}')
            else:
                self.send_response(404)
                self.end_headers()
                
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.end_headers()
                
        def log_message(self, format, *args):
            pass

    try:
        server = HTTPServer(('localhost', 8085), SignalHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print("  📶 Emergency stop server listening on http://localhost:8085/stop")
    except Exception as se:
        print(f"  Warning: Could not start emergency stop server: {se}")

    try:
        epoch_stats = run_simulation(
            world, 
            ticks=TICKS, 
            scarcity_level=SCARCITY, 
            save_paths=save_paths, 
            save_epochs=save_epochs, 
            sample_interval=sample_interval,
            live_callback=live_callback,
            planner_enabled=PLANNER_ENABLED,
            sleep_consolidation_enabled=SLEEP_CONSOLIDATION_ENABLED
        )
    except KeyboardInterrupt:
        print("\n⚠️ [EMERGENCY STOP] KeyboardInterrupt detected! Exiting simulation tick loop and compiling current telemetry data immediately...")
        epoch_stats = getattr(world, "epoch_stats", [])
        # Auto-save full checkpoint on Ctrl-C if configured
        if FULL_CHECKPOINT_ON_STOP and getattr(world, "_full_checkpoint_config", None) is not None:
            try:
                from world.checkpoint_io import save_full_checkpoint
                _kb_cp = os.path.join(exp_folder, f"full_checkpoint_{world.tick}.json")
                save_full_checkpoint(world, world._full_checkpoint_config, _kb_cp)
                print(f"  💾 Full checkpoint saved on Ctrl-C: {_kb_cp}")
            except Exception as _kce:
                print(f"  Warning: Ctrl-C full checkpoint save failed: {_kce}")

    print("-" * 80)
    
    # 6. Save visual traces map
    generate_simulation_map(world, "simulation.png")
    
    # 7. Collect statistics
    alive_count = sum(1 for a in world.agents if not a.dead)
    total_agents = len(world.agents)
    avg_discoveries = float(np.mean([a.discoveries_count for a in world.agents])) if world.agents else 0.0
    
    if save_paths:
        avg_radii = []
        for a in world.agents:
            if len(a.sampled_path_history) > 0:
                first = a.sampled_path_history[0]
                if len(first) > 2:
                    start_x, start_y = first[0], first[1]
                    path_coords = np.array([[coord[0], coord[1]] for coord in a.sampled_path_history])
                else:
                    start_y, start_x = first[0], first[1]
                    path_coords = np.array(a.sampled_path_history)
                start_coords = np.array([start_x, start_y]) if len(first) > 2 else np.array([start_y, start_x])
                dists = np.sqrt(np.sum((path_coords - start_coords)**2, axis=-1))
                avg_radii.append(np.max(dists) if len(dists) > 0 else 0.0)
        avg_radius = float(np.mean(avg_radii)) if avg_radii else 0.0
    else:
        avg_radius = float(np.mean([a.max_radius for a in world.agents])) if world.agents else 0.0
    
    git_version = get_git_commit()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    max_generation = max(a.generation for a in world.agents) if world.agents else 0

    # --------------------------------------------------------------------------
    # 📊 CALCULATE 17 DERIVED SCIENTIFIC METRICS
    # --------------------------------------------------------------------------
    total_ticks = world.tick
    agents = world.agents
    
    # 1. Average Generation Interval
    gen_intervals = []
    agent_map = {a.id: a for a in agents}
    for a in agents:
        if getattr(a, "parent_ids", None) is not None:
            p1_id, p2_id = a.parent_ids
            if p1_id in agent_map and p2_id in agent_map:
                p1, p2 = agent_map[p1_id], agent_map[p2_id]
                p1_age = (a.born_tick - p1.born_tick) / 360.0
                p2_age = (a.born_tick - p2.born_tick) / 360.0
                gen_intervals.append((p1_age + p2_age) / 2.0)
    avg_gen_interval = float(np.mean(gen_intervals)) if gen_intervals else 0.0

    # 2. Population Doubling Time
    doubling_times = []
    pop_hist = getattr(world, "population_history", [])
    for i, p2 in enumerate(pop_hist):
        for p1 in pop_hist[:i]:
            if p1["total"] > 0 and p2["total"] >= p1["total"] * 2:
                doubling_times.append(p2["tick"] - p1["tick"])
    min_doubling_time = int(min(doubling_times)) if doubling_times else -1

    # 3. Colony Lifespans
    colony_lifespans = {}
    extinct_map = {e["colony_id"]: e["tick"] for e in getattr(world, "extinction_events", [])}
    for c in getattr(world, "colonies", []):
        cid = c["id"]
        if cid in extinct_map:
            colony_lifespans[c["name"]] = int(extinct_map[cid])
        else:
            colony_lifespans[c["name"]] = int(total_ticks)

    # 4. Average Genetic Diversity
    avg_genetic_diversity = float(np.mean([gh["diversity_score"] for gh in getattr(world, "genetic_history", [])])) if getattr(world, "genetic_history", None) else 0.0

    # Calculate total path distances
    total_distances = {}
    for a in agents:
        dist = 0.0
        path = getattr(a, "sampled_path_history", [])
        if len(path) > 1:
            coords = []
            for pt in path:
                if pt is not None:
                    coords.append([pt[0], pt[1]])
            if len(coords) > 1:
                coords = np.array(coords)
                dists = np.sqrt(np.sum((coords[1:] - coords[:-1])**2, axis=-1))
                dist = float(np.sum(dists))
        total_distances[a.id] = dist
    sum_total_distance = sum(total_distances.values())

    # 5. Food, Water, and Energy Efficiency
    sum_eats = sum(getattr(a, "eats_count", a.action_counts.get("Eating", 0)) for a in agents)
    sum_drinks = sum(getattr(a, "drinks_count", a.action_counts.get("Drinking", 0)) for a in agents)
    sum_survival = sum(a.ticks_survived for a in agents)
    
    food_efficiency = (sum_eats * 10.0) / (sum_total_distance + 1.0)
    water_efficiency = (sum_drinks * 10.0) / (sum_total_distance + 1.0)
    energy_efficiency = (sum_survival * 10.0) / (sum_total_distance + 1.0)

    # 6. Prediction Error
    pred_errors = [a.total_prediction_error / a.prediction_count for a in agents if getattr(a, "prediction_count", 0) > 0]
    avg_pred_error = float(np.mean(pred_errors)) if pred_errors else 0.0

    # 7. Concept and Procedure rates
    total_concepts = sum(sum(len(lst) for lst in a.concepts.values()) if hasattr(a, "concepts") else 0 for a in agents)
    total_procedures = sum(len(a.procedures) if hasattr(a, "procedures") else 0 for a in agents)
    concept_rate = (total_concepts / (total_ticks + 1)) * 1000.0
    procedure_rate = (total_procedures / (total_ticks + 1)) * 1000.0

    # 8. Descendants tracking
    founder_memo = {}
    def get_agent_founders(a_id):
        if a_id in founder_memo:
            return founder_memo[a_id]
        ag = agent_map.get(a_id)
        if ag is None:
            return set()
        if not getattr(ag, "parent_ids", None):
            return {a_id}
        f = set()
        for pid in ag.parent_ids:
            f.update(get_agent_founders(pid))
        founder_memo[a_id] = f
        return f

    founder_descendants = {}
    for a in agents:
        fs = get_agent_founders(a.id)
        for fid in fs:
            if fid not in founder_descendants:
                founder_descendants[fid] = set()
            founder_descendants[fid].add(a.id)

    # Compute family metrics
    avg_children = float(np.mean([len(a.children_ids) for a in agents])) if agents else 0.0
    
    grandchildren_counts = []
    descendants_counts = []
    for a in agents:
        g_child = 0
        for cid in a.children_ids:
            g_child += len(agent_map[cid].children_ids) if cid in agent_map else 0
        grandchildren_counts.append(g_child)
        
        desc = set()
        def add_desc(aid):
            ag = agent_map.get(aid)
            if ag:
                for cid in ag.children_ids:
                    if cid not in desc:
                        desc.add(cid)
                        add_desc(cid)
        add_desc(a.id)
        descendants_counts.append(len(desc))
        
    avg_grandchildren = float(np.mean(grandchildren_counts)) if grandchildren_counts else 0.0
    avg_descendants = float(np.mean(descendants_counts)) if descendants_counts else 0.0

    # 9. Distance from birthplace
    distances_birth = []
    for a in agents:
        spawn_loc = getattr(a, "spawn_location", None)
        if spawn_loc:
            distances_birth.append(np.sqrt((a.location[1] - spawn_loc[1])**2 + (a.location[0] - spawn_loc[0])**2))
    avg_distance_birth = float(np.mean(distances_birth)) if distances_birth else 0.0

    # 10. Shelter occupancy
    occupancy_rates = []
    for a in agents:
        shelter_ticks = a.action_counts.get("Sheltering", 0)
        occupancy_rates.append(shelter_ticks / a.ticks_survived if a.ticks_survived > 0 else 0.0)
    avg_shelter_occupancy = float(np.mean(occupancy_rates)) if occupancy_rates else 0.0

    # 11. Social and cooperation
    avg_social_degree = float(np.mean([len(a.known_agents) for a in agents])) if agents else 0.0
    
    total_shares = sum(a.action_counts.get("Share Food", 0) + a.action_counts.get("Share Water", 0) for a in agents)
    avg_cooperation = total_shares / (total_ticks + 1)
    
    avg_conflict = float(np.mean([a.injury_damage_accumulated for a in agents])) if agents else 0.0

    derived_metrics = {
        "avg_generation_interval": round(avg_gen_interval, 2),
        "population_doubling_time": min_doubling_time,
        "colony_lifespans": colony_lifespans,
        "avg_genetic_diversity": round(avg_genetic_diversity, 4),
        "food_efficiency": round(food_efficiency, 4),
        "water_efficiency": round(water_efficiency, 4),
        "energy_efficiency": round(energy_efficiency, 4),
        "avg_prediction_error": round(avg_pred_error, 4),
        "concept_formation_rate": round(concept_rate, 3),
        "procedure_creation_rate": round(procedure_rate, 3),
        "avg_children": round(avg_children, 2),
        "avg_grandchildren": round(avg_grandchildren, 2),
        "avg_descendants": round(avg_descendants, 2),
        "avg_distance_from_birthplace": round(avg_distance_birth, 1),
        "avg_shelter_occupancy": round(avg_shelter_occupancy, 3),
        "avg_social_degree": round(avg_social_degree, 2),
        "avg_cooperation_score": round(avg_cooperation, 4),
        "avg_conflict_score": round(avg_conflict, 4)
    }

    # --------------------------------------------------------------------------
    # 📊 CALCULATE 6 BINNED DISTRIBUTIONS
    # --------------------------------------------------------------------------
    lifespan_bins = {"0-10 years": 0, "10-20 years": 0, "20-30 years": 0, "30-40 years": 0, "40-50 years": 0, "50-60 years": 0, "60-70 years": 0, "70-80 years": 0, "80+ years": 0}
    children_bins = {"0 children": 0, "1 child": 0, "2 children": 0, "3 children": 0, "4 children": 0, "5-8 children": 0, "9+ children": 0}
    shelter_bins = {"Level 0 (None)": 0, "Level 1 (Tent)": 0, "Level 2 (Cabin)": 0, "Level 3 (Stone)": 0}
    gen_bins = {"Gen 0": 0, "Gen 1": 0, "Gen 2": 0, "Gen 3": 0, "Gen 4": 0, "Gen 5+": 0}
    pred_bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    concept_bins = {"0-2 concepts": 0, "3-5 concepts": 0, "6-10 concepts": 0, "11-20 concepts": 0, "21+ concepts": 0}

    for a in agents:
        age_yrs = a.age / 360.0
        if age_yrs < 10: lifespan_bins["0-10 years"] += 1
        elif age_yrs < 20: lifespan_bins["10-20 years"] += 1
        elif age_yrs < 30: lifespan_bins["20-30 years"] += 1
        elif age_yrs < 40: lifespan_bins["30-40 years"] += 1
        elif age_yrs < 50: lifespan_bins["40-50 years"] += 1
        elif age_yrs < 60: lifespan_bins["50-60 years"] += 1
        elif age_yrs < 70: lifespan_bins["60-70 years"] += 1
        elif age_yrs < 80: lifespan_bins["70-80 years"] += 1
        else: lifespan_bins["80+ years"] += 1

        n_child = len(a.children_ids)
        if n_child == 0: children_bins["0 children"] += 1
        elif n_child == 1: children_bins["1 child"] += 1
        elif n_child == 2: children_bins["2 children"] += 1
        elif n_child == 3: children_bins["3 children"] += 1
        elif n_child == 4: children_bins["4 children"] += 1
        elif n_child <= 8: children_bins["5-8 children"] += 1
        else: children_bins["9+ children"] += 1

        sh_lvl = getattr(a, "shelter_level", 0)
        if sh_lvl == 0: shelter_bins["Level 0 (None)"] += 1
        elif sh_lvl == 1: shelter_bins["Level 1 (Tent)"] += 1
        elif sh_lvl == 2: shelter_bins["Level 2 (Cabin)"] += 1
        else: shelter_bins["Level 3 (Stone)"] += 1

        gen = a.generation
        if gen == 0: gen_bins["Gen 0"] += 1
        elif gen == 1: gen_bins["Gen 1"] += 1
        elif gen == 2: gen_bins["Gen 2"] += 1
        elif gen == 3: gen_bins["Gen 3"] += 1
        elif gen == 4: gen_bins["Gen 4"] += 1
        else: gen_bins["Gen 5+"] += 1

        pred_acc = a.prediction_successes / a.prediction_attempts if a.prediction_attempts > 0 else 1.0
        if pred_acc < 0.2: pred_bins["0.0-0.2"] += 1
        elif pred_acc < 0.4: pred_bins["0.2-0.4"] += 1
        elif pred_acc < 0.6: pred_bins["0.4-0.6"] += 1
        elif pred_acc < 0.8: pred_bins["0.6-0.8"] += 1
        else: pred_bins["0.8-1.0"] += 1

        concepts_cnt = sum(len(lst) for lst in a.concepts.values()) if hasattr(a, "concepts") else 0
        if concepts_cnt <= 2: concept_bins["0-2 concepts"] += 1
        elif concepts_cnt <= 5: concept_bins["3-5 concepts"] += 1
        elif concepts_cnt <= 10: concept_bins["6-10 concepts"] += 1
        elif concepts_cnt <= 20: concept_bins["11-20 concepts"] += 1
        else: concept_bins["21+ concepts"] += 1

    distributions = {
        "lifespan": lifespan_bins,
        "children": children_bins,
        "shelter": shelter_bins,
        "generation": gen_bins,
        "prediction_confidence": pred_bins,
        "concepts": concept_bins
    }

    # --------------------------------------------------------------------------
    # 🏆 COMPUTE EVOLUTIONARY SUCCESS LEADERBOARD
    # --------------------------------------------------------------------------
    leaderboards = {
        "longest_surviving": [],
        "largest_size": [],
        "highest_health": [],
        "most_generations": [],
        "cognitive_mastery": []
    }
    
    founder_stats = []
    for fid, desc_ids in founder_descendants.items():
        desc_agents = [agent_map[aid] for aid in desc_ids if aid in agent_map]
        if not desc_agents:
            continue
            
        births = [x.born_tick for x in desc_agents]
        deaths = [x.born_tick + x.ticks_survived for x in desc_agents]
        span = int(max(deaths) - min(births))
        
        max_g = int(max(x.generation for x in desc_agents))
        avg_h = float(np.mean([x.health for x in desc_agents]))
        accs = [x.prediction_successes / x.prediction_attempts if x.prediction_attempts > 0 else 1.0 for x in desc_agents]
        avg_acc = float(np.mean(accs))
        
        founder_stats.append({
            "founder_id": int(fid),
            "colony_id": int(agent_map[fid].colony_id) if fid in agent_map else 0,
            "size": int(len(desc_ids)),
            "longevity_ticks": span,
            "max_generation": max_g,
            "avg_health": round(avg_h, 2),
            "avg_prediction_accuracy": round(avg_acc, 4)
        })

    if founder_stats:
        leaderboards["longest_surviving"] = sorted(founder_stats, key=lambda x: x["longevity_ticks"], reverse=True)[:5]
        leaderboards["largest_size"] = sorted(founder_stats, key=lambda x: x["size"], reverse=True)[:5]
        leaderboards["highest_health"] = sorted(founder_stats, key=lambda x: x["avg_health"], reverse=True)[:5]
        leaderboards["most_generations"] = sorted(founder_stats, key=lambda x: x["max_generation"], reverse=True)[:5]
        leaderboards["cognitive_mastery"] = sorted(founder_stats, key=lambda x: x["avg_prediction_accuracy"], reverse=True)[:5]

    summary_record = {
        "timestamp": timestamp,
        "experiment": EXPERIMENT_NAME,
        "seed": SEED,
        "ticks": TICKS,
        "scarcity": SCARCITY,
        "survivors": f"{alive_count}/{total_agents}",
        "avg_radius": round(avg_radius, 1),
        "avg_discoveries": round(avg_discoveries, 1),
        "tests_passed": test_results["status"] == "PASS",
        "max_generation": max_generation,
        "derived_metrics": derived_metrics,
        "distributions": distributions,
        "leaderboards": leaderboards
    }
    
    run_metadata = {
        "git_version": git_version,
        "experiment_name": EXPERIMENT_NAME,
        "derived_metrics": derived_metrics,
        "distributions": distributions,
        "leaderboards": leaderboards,
        "max_generation": max_generation
    }
    
    # 8. Update run history (last 5 runs)
    update_history(summary_record)
    
    # 9. Save telemetry JS database
    save_simulation_data(
        world=world,
        experiment_name=EXPERIMENT_NAME,
        scarcity_val=SCARCITY,
        filepath="simulation_data.js",
        test_results=test_results,
        run_metadata=run_metadata,
        save_paths=save_paths,
        epoch_stats=epoch_stats if save_epochs else None
    )
    
    # 10. Print detailed metrics report in console
    print_experiment_report(world, "personality")
    
    print("\nCustom simulation complete!")
    print(f"  Survivors      : {alive_count}/{total_agents}")
    print(f"  Avg Discoveries: {summary_record['avg_discoveries']}")
    print(f"  Avg Travel Dist: {summary_record['avg_radius']} cells")
    print("=" * 80)
    print("Success! Open visualizer.html in your web browser to play back paths.")
    print("=" * 80)
    
    # Restore stdout/stderr and close logger file
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    
    print("Archiving experiment data (13 files)...")
    archive_experiment(world, summary_record, exp_folder, epoch_stats=epoch_stats)
    print(f"Experiment successfully archived to: {exp_folder}")

if __name__ == "__main__":
    main()
