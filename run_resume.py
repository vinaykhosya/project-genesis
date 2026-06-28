"""
run_resume.py — Project Genesis: Resume a Paused Experiment

USAGE
-----
1. Set RESUME_FROM to the full checkpoint path saved by a previous run.
2. Set ADDITIONAL_TICKS to how many more ticks to run in this session.
3. Optionally override specific simulation parameters below.
4. Run:  python run_resume.py

WORKFLOW
--------
Session 1  (e.g. run_test.py):   Ticks 0 → 36,000   → Stop & Save → full_checkpoint_36000.json
Session 2  (run_resume.py):       Ticks 36,001 → 72,000  → Stop & Save → full_checkpoint_72000.json
Session 3  (run_resume.py):       Ticks 72,001 → 108,000 → ...
...continuing until you reach 1,000 years, 5,000 years, or beyond.

NOTES
-----
- World numpy arrays are NOT loaded from the checkpoint — they are regenerated
  from the seed in ~2-3 seconds.  This is fast and keeps checkpoints small.
- All agent neural net weights, drive states, memories, genomes, relationships,
  and colony data are fully restored.
- The Live Dashboard (visualizer.html) connects automatically and works exactly
  as during a normal run.
"""

import os
import sys
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==============================================================================
# 📂 RESUME CONFIGURATION — Edit these values
# ==============================================================================

# Path to the full checkpoint file you want to resume from.
# This is saved automatically by run_test.py or run_resume.py on Stop & Save
# or at every FULL_CHECKPOINT_INTERVAL ticks.
RESUME_FROM = r"experiments/2026-06-27_21-00-59_emotion with no fights and same spawn/full_checkpoint_11416.json"   # e.g. r"experiments\2026-06-27_my_run\full_checkpoint_36000.json"

# How many additional ticks to run in this session.
ADDITIONAL_TICKS = 36000   # 36,000 ticks ≈ 100 simulated years

# ==============================================================================
# ✏️ OPTIONAL OVERRIDES
# Set to None to keep the saved config value.
# Use these to change conditions in the resumed session — e.g. switch climate,
# stress-test with a harder environment, or increase population cap.
# ==============================================================================

OVERRIDE_MAX_POPULATION      = None   # int or None
OVERRIDE_DISASTERS_ENABLED   = None   # True / False / None
OVERRIDE_DISPUTES_ENABLED    = None   # True / False / None
OVERRIDE_HEALING_SPEED_MULT  = None   # float or None
OVERRIDE_MUTATION_RATE       = None   # float or None

# Full checkpoint save settings for this resume session
FULL_CHECKPOINT_INTERVAL = 36000    # Save every N ticks (36000 = every 100 years)
FULL_CHECKPOINT_ON_STOP  = True     # Save on Stop & Save button / Ctrl-C

# Live pacing (0.0 = max speed, recommended for long runs)
LIVE_PACING_DELAY = 0.0

# Checkpoint interval for visualization checkpoints (not full state)
VIS_CHECKPOINT_INTERVAL = 50000

# ==============================================================================
# ⚙️  IMPLEMENTATION — no need to modify below
# ==============================================================================

from world.checkpoint_io import load_full_checkpoint, save_full_checkpoint
from world.agents.simulation import run_simulation
from main import save_simulation_data, print_experiment_report
from world.agents.drives import emotional_label
import datetime
import time


class DualWriter:
    """Writes to both stdout and a log file simultaneously."""
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


def main():
    if not RESUME_FROM:
        print("ERROR: RESUME_FROM is not set.")
        print("  Edit run_resume.py and set RESUME_FROM to the path of your full checkpoint file.")
        print("  Example: RESUME_FROM = r'experiments\\2026-06-27_my_run\\full_checkpoint_36000.json'")
        sys.exit(1)

    if not os.path.exists(RESUME_FROM):
        print(f"ERROR: Checkpoint file not found:\n  {RESUME_FROM}")
        sys.exit(1)

    # --- Load the checkpoint ---
    world, saved_config = load_full_checkpoint(RESUME_FROM)

    # --- Apply overrides on top of saved config ---
    if OVERRIDE_MAX_POPULATION is not None:
        world.max_population = int(OVERRIDE_MAX_POPULATION)
        saved_config["max_population"] = world.max_population
        print(f"  Override: max_population → {world.max_population}")

    if OVERRIDE_DISASTERS_ENABLED is not None:
        world.disasters_enabled = bool(OVERRIDE_DISASTERS_ENABLED)
        saved_config["disasters_enabled"] = world.disasters_enabled
        print(f"  Override: disasters_enabled → {world.disasters_enabled}")

    if OVERRIDE_DISPUTES_ENABLED is not None:
        world.disputes_enabled = bool(OVERRIDE_DISPUTES_ENABLED)
        saved_config["disputes_enabled"] = world.disputes_enabled
        print(f"  Override: disputes_enabled → {world.disputes_enabled}")

    if OVERRIDE_HEALING_SPEED_MULT is not None:
        world.healing_speed_mult = float(OVERRIDE_HEALING_SPEED_MULT)
        saved_config["healing_speed_mult"] = world.healing_speed_mult
        print(f"  Override: healing_speed_mult → {world.healing_speed_mult}")

    if OVERRIDE_MUTATION_RATE is not None:
        world.mutation_rate = float(OVERRIDE_MUTATION_RATE)
        saved_config["mutation_rate"] = world.mutation_rate
        print(f"  Override: mutation_rate → {world.mutation_rate}")

    # Restore world config attributes (in case they're missing from older checkpoints)
    world.max_population         = int(saved_config.get("max_population",         200))
    world.mutation_rate          = float(saved_config.get("mutation_rate",         0.05))
    world.reproduction_enabled   = bool(saved_config.get("reproduction_enabled",   True))
    world.disputes_enabled       = bool(saved_config.get("disputes_enabled",       False))
    world.disasters_enabled      = bool(saved_config.get("disasters_enabled",      True))
    world.healing_speed_mult     = float(saved_config.get("healing_speed_mult",    1.0))
    world.shelter_build_speed_mult = float(saved_config.get("shelter_build_speed_mult", 1.0))
    world.shelter_search_dist    = float(saved_config.get("shelter_search_dist",   100.0))
    world.spawn_mode             = str(saved_config.get("spawn_mode",              "fixed"))
    world.colony_spawn_locations = saved_config.get("colony_spawn_locations",      {})
    world.targeted_biomes        = saved_config.get("targeted_biomes",             [])

    # Phase 8 ablation flags
    world.ablation = saved_config.get("ablation", {
        "planner":           True,
        "emotion":           True,
        "relationships":     True,
        "memory_importance": True,
        "motivation":        True,
        "prediction_error":  True,
    })

    # Experiment identity
    base_name   = saved_config.get("experiment_name", "resumed_experiment")
    scarcity    = float(saved_config.get("scarcity", 1.0))
    resume_tick = int(world.tick)
    resume_year = resume_tick // 360

    # --- Set up experiment folder for this session ---
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_name  = f"{base_name}_resumed_y{resume_year}"
    os.makedirs("experiments", exist_ok=True)
    exp_folder    = os.path.join("experiments", f"{timestamp_str}_{session_name}")
    os.makedirs(exp_folder, exist_ok=True)
    world.exp_folder = exp_folder

    # Save config with resume metadata
    resume_config = dict(saved_config)
    resume_config["resumed_from"]   = RESUME_FROM
    resume_config["resume_tick"]    = resume_tick
    resume_config["resume_year"]    = resume_year
    resume_config["additional_ticks"] = ADDITIONAL_TICKS
    resume_config["experiment_name"] = session_name

    # --- Set up logging ---
    log_file         = open(os.path.join(exp_folder, "simulation.log"), "w", encoding="utf-8")
    original_stdout  = sys.stdout
    original_stderr  = sys.stderr
    sys.stdout = DualWriter(original_stdout, log_file)
    sys.stderr = DualWriter(original_stderr, log_file)

    print("=" * 70)
    print("          PROJECT GENESIS — RESUME SESSION")
    print("=" * 70)
    print(f"  Resumed from  : {RESUME_FROM}")
    print(f"  Resume tick   : {resume_tick}  (Year {resume_year})")
    print(f"  Running for   : {ADDITIONAL_TICKS} more ticks  "
          f"({ADDITIONAL_TICKS // 360} more years)")
    print(f"  Will reach    : tick {resume_tick + ADDITIONAL_TICKS}  "
          f"(Year {(resume_tick + ADDITIONAL_TICKS) // 360})")
    print(f"  Alive agents  : {sum(1 for a in world.agents if not a.dead)}")
    print(f"  Total agents  : {len(world.agents)}")
    print(f"  Max pop cap   : {world.max_population}")
    print(f"  Mutation rate : {world.mutation_rate}")
    print(f"  Scarcity      : {scarcity}")
    print("=" * 70)
    print()

    # --- Background HTTP server (Stop & Save button) ---
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class SignalHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/stop":
                world.stop_requested = True
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status": "stopping"}')
            else:
                self.send_response(404)
                self.end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

        def log_message(self, fmt, *args):
            pass   # suppress request logs

    signal_server = HTTPServer(("localhost", 8765), SignalHandler)
    server_thread = threading.Thread(target=signal_server.serve_forever, daemon=True)
    server_thread.start()

    # --- Auto-open visualizer ---
    import webbrowser
    try:
        vis_path = os.path.abspath("visualizer.html")
        webbrowser.open("file://" + vis_path + "?live=true")
        print("Opened visualizer.html in browser (live mode).")
    except Exception as e:
        print(f"Warning: could not open browser: {e}")

    # --- Live callback (same as run_test.py) ---
    from world.agents.behavior_classifier import classify_behavior

    def live_callback(tick, epoch_stats):
        if tick % 100 == 0 or tick == resume_tick + 1:
            # Build agents_coords payload
            agents_coords = []
            for agent in world.agents:
                action_pct = {}
                total_actions = sum(agent.action_counts.values()) if hasattr(agent, "action_counts") else 0
                if total_actions > 0:
                    for act, count in agent.action_counts.items():
                        action_pct[act] = round((count / total_actions) * 100.0, 1)

                agents_coords.append({
                    "id":             agent.id,
                    "archetype":      agent.archetype,
                    "spawn_biome":    getattr(agent, "spawn_biome", "Unknown"),
                    "traits":         {k: float(v) for k, v in agent.traits.items()} if hasattr(agent, "traits") else {},
                    "dead":           bool(agent.dead),
                    "location":       [int(agent.location[1]), int(agent.location[0])],
                    "current_action": getattr(agent, "current_action", "Idle"),
                    "health":         round(agent.health, 1),
                    "hunger":         round(agent.hunger, 1),
                    "thirst":         round(agent.thirst, 1),
                    "energy":         round(agent.energy, 1),
                    "stored_food":    round(float(getattr(agent, "stored_food", 0.0)), 1),
                    "stored_water":   round(float(getattr(agent, "stored_water", 0.0)), 1),
                    "colony_id":      int(getattr(agent, "colony_id", 0)),
                    "generation":     int(getattr(agent, "generation", 0)),
                    "age":            int(agent.age),
                    "max_age":        int(agent.max_age),
                    "drinks":         int(getattr(agent, "drinks_count", 0)),
                    "eats":           int(getattr(agent, "eats_count", 0)),
                    "resting_ticks":  int(getattr(agent, "resting_ticks", 0)),
                    "discoveries":    int(getattr(agent, "discoveries_count", 0)),
                    "recognized_agents": sorted(list(getattr(agent, "known_agents", []))),
                    "exploration_radius": round(float(getattr(agent, "max_radius", 0.0)), 1),
                    "action_pct":     action_pct,
                    "parent_ids":     list(getattr(agent, "parent_ids", None) or []),
                    "children_ids":   list(getattr(agent, "children_ids", [])),
                    "born_tick":      int(getattr(agent, "born_tick", 0)),
                    "behavior_cluster": getattr(agent, "behavior_cluster", "C0"),
                    "genome":         getattr(agent.genome, "to_list", lambda: [])() if hasattr(agent, "genome") else [],
                    "cause_of_death": getattr(agent, "cause_of_death", "None"),
                    "primary_cause":  getattr(agent, "primary_cause", "None"),
                    "secondary_cause":getattr(agent, "secondary_cause", "None"),
                    "fat_reserves":   round(float(getattr(agent, "fat_reserves", 100.0)), 1),
                    "muscle_mass":    round(float(getattr(agent, "muscle_mass", 100.0)), 1),
                    "injury_level":   round(float(getattr(agent, "injury_level", 0.0)), 1),
                    "shelter_location": [int(agent.shelter_location[1]), int(agent.shelter_location[0])] if getattr(agent, "shelter_location", None) else None,
                    "shelter_level":  int(getattr(agent, "shelter_level", 0)),
                    "shelter_durability": round(float(getattr(agent, "shelter_durability", 0.0)), 1),
                    "ticks_survived": int(agent.ticks_survived),
                    "years_survived": int(getattr(agent, "years_survived", 0)),
                    "season_observations": {int(k): int(v) for k, v in agent.season_observations.items()} if hasattr(agent, "season_observations") else {},
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
                    "motivation": agent.motivation.to_dict() if getattr(agent, "motivation", None) is not None else {},
                })

            colonies_data = [
                {"id": c["id"], "name": c["name"], "color": c["color"],
                 "stored_food": round(c.get("stored_food", 0.0), 1),
                 "stored_water": round(c.get("stored_water", 0.0), 1)}
                for c in getattr(world, "colonies", [])
            ]

            behavior_result = classify_behavior(
                [a for a in world.agents if not a.dead],
                world_width=world.width, n_clusters=4
            )

            live_state = {
                "tick":               int(tick),
                "alive":              int(sum(1 for a in world.agents if not a.dead)),
                "agents":             agents_coords,
                "colonies":           colonies_data,
                "epoch_stats":        epoch_stats,
                "population_history": getattr(world, "population_history", []),
                "genetic_history":    getattr(world, "genetic_history",    []),
                "extinction_events":  getattr(world, "extinction_events",  []),
                "generation_number":  int(getattr(world, "generation_number", 0)),
                "total_births":       int(getattr(world, "total_births",    0)),
                "total_deaths":       int(getattr(world, "total_deaths",    0)),
                "behavior_clustering": behavior_result,
            }

            try:
                import json as _json
                temp_json = "live_state.json.tmp"
                temp_js   = "live_state.js.tmp"
                with open(temp_json, "w") as f:
                    _json.dump(live_state, f)
                with open(temp_js, "w") as f:
                    f.write(f"window.LIVE_STATE = {_json.dumps(live_state)};")
                os.replace(temp_json, "live_state.json")
                os.replace(temp_js,   "live_state.js")
            except Exception as e:
                print(f"Warning: live state write failed: {e}")

            if LIVE_PACING_DELAY > 0.0:
                time.sleep(LIVE_PACING_DELAY)

        # --- Full checkpoint save (every FULL_CHECKPOINT_INTERVAL ticks) ---
        if FULL_CHECKPOINT_INTERVAL > 0 and tick % FULL_CHECKPOINT_INTERVAL == 0 and tick > resume_tick:
            try:
                cp_path = os.path.join(exp_folder, f"full_checkpoint_{tick}.json")
                save_full_checkpoint(world, resume_config, cp_path)
            except Exception as ce:
                print(f"  Warning: full checkpoint save failed at tick {tick}: {ce}")

        # --- Visualisation checkpoint ---
        if VIS_CHECKPOINT_INTERVAL > 0 and tick % VIS_CHECKPOINT_INTERVAL == 0 and tick > resume_tick:
            try:
                vis_cp_path = os.path.join(exp_folder, f"checkpoint_{tick}.json")
                save_simulation_data(
                    world=world,
                    experiment_name=session_name,
                    scarcity_val=scarcity,
                    filepath=vis_cp_path,
                    save_paths=False,
                    epoch_stats=epoch_stats,
                )
                print(f"  📊 Visualisation checkpoint saved: checkpoint_{tick}.json")
            except Exception as _ce:
                print(f"  Warning: vis checkpoint failed at tick {tick}: {_ce}")

    # --- Run the simulation (agents already in world, skip spawning) ---
    print(f"▶ Resuming simulation at tick {resume_tick}...")
    epoch_stats = run_simulation(
        world,
        ADDITIONAL_TICKS,
        scarcity_level=scarcity,
        save_epochs=True,
        sample_interval=5000,
        long_run=True,
        save_paths=False,
        prediction_enabled=True,
        planner_enabled=world.ablation.get("planner", True),
        sleep_consolidation_enabled=True,
        callback=live_callback,
        pacing_delay=LIVE_PACING_DELAY,
    )

    end_tick = world.tick
    end_year = end_tick // 360
    alive    = sum(1 for a in world.agents if not a.dead)
    print(f"\n{'='*70}")
    print(f"  Session complete.")
    print(f"  Ran: {resume_tick} → {end_tick}  (Year {resume_year} → {end_year})")
    print(f"  Alive: {alive} / {len(world.agents)}")
    print(f"{'='*70}\n")

    # --- Save final full checkpoint ---
    if FULL_CHECKPOINT_ON_STOP:
        final_cp = os.path.join(exp_folder, f"full_checkpoint_{end_tick}.json")
        try:
            save_full_checkpoint(world, resume_config, final_cp)
            print(f"  💾 Final full checkpoint: {final_cp}")
        except Exception as e:
            print(f"  Warning: final full checkpoint failed: {e}")

    # --- Save final visualisation data ---
    try:
        final_vis = os.path.join(exp_folder, "replay.json")
        save_simulation_data(
            world=world,
            experiment_name=session_name,
            scarcity_val=scarcity,
            filepath=final_vis,
            save_paths=True,
            epoch_stats=epoch_stats,
        )
        print(f"  📊 Final replay data: {final_vis}")
    except Exception as e:
        print(f"  Warning: final vis data failed: {e}")

    # --- Write summary config ---
    try:
        with open(os.path.join(exp_folder, "config.json"), "w") as f:
            json.dump(resume_config, f, indent=2)
    except Exception:
        pass

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    print(f"Experiment folder: {exp_folder}")


if __name__ == "__main__":
    main()
