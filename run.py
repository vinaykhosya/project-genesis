import os
import sys
import json
import unittest
import io
import datetime
import subprocess
import numpy as np

# Add project root to python path to ensure imports work cleanly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from world.generator import generate_world
from world.predictor import predict_settlements
from world.agents.agent import Agent
from world.agents.simulation import run_simulation
from main import generate_simulation_map, save_simulation_data

def run_unit_tests():
    """Runs the unit tests and returns a dict with status and log details."""
    print("Running test suite validation...")
    suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=1)
    result = runner.run(suite)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    
    status = "PASS" if result.wasSuccessful() else "FAIL"
    print(f"Tests execution complete: {status} ({passed}/{total} passed)")
    
    return {
        "status": status,
        "total": total,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "log": stream.getvalue()
    }

def get_git_commit():
    """Retrieves short git hash if inside a git repository."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "N/A"

def load_config(config_path):
    """Loads configuration JSON file."""
    if not os.path.exists(config_path):
        print(f"Config path {config_path} not found. Falling back to default_experiment.json")
        config_path = os.path.join("configs", "default_experiment.json")
        
    with open(config_path, "r") as f:
        return json.load(f), config_path

def spawn_custom_agents(world, agents_config):
    """Spawns agents dynamically based on configurations."""
    spots = predict_settlements(world, count=10, exclusion_radius=45.0)
    agents = []
    agent_id = 0
    
    for item in agents_config:
        archetype = item.get("archetype", "Balanced")
        curiosity = item.get("curiosity", 0.5)
        risk = item.get("risk_tolerance", 0.5)
        traits = {"curiosity": float(curiosity), "risk_tolerance": float(risk)}
        
        # Check if coordinates location is hardcoded
        if "location" in item:
            # JSON format is [x, y], python location tuple is (y, x)
            loc_x, loc_y = item["location"]
            # Clamp coordinates to within world borders
            loc_y = np.clip(int(loc_y), 0, world.height - 1)
            loc_x = np.clip(int(loc_x), 0, world.width - 1)
            
            agent = Agent(agent_id=agent_id, location=(loc_y, loc_x), traits=traits)
            agent.archetype = archetype
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
        else:
            # Count-based spawning at NMS locations
            count = item.get("count", 1)
            for _ in range(count):
                # Cycle through predicted spots
                spot_idx = agent_id % len(spots)
                spot = spots[spot_idx]
                
                agent = Agent(agent_id=agent_id, location=(spot['y'], spot['x']), traits=traits)
                agent.archetype = archetype
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
    
    # Initialize biomes for each agent
    from world.state import BIOME_NAMES
    for agent in world.agents:
        biome_id = world.biome[agent.location[0], agent.location[1]]
        agent.spawn_biome = BIOME_NAMES.get(biome_id, "Unknown")
        
    print(f"Spawned {len(world.agents)} custom agents successfully.")

def update_history(summary_record):
    """Updates the run_history.json and dumps run_history.js."""
    history_json_path = "run_history.json"
    history_js_path = "run_history.js"
    
    history = []
    if os.path.exists(history_json_path):
        try:
            with open(history_json_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    # Append the new run summary
    history.append(summary_record)
    
    # Keep only the last 5 runs to prevent memory bloat
    history = history[-5:]
    
    with open(history_json_path, "w") as f:
        json.dump(history, f, indent=2)
        
    with open(history_js_path, "w") as f:
        f.write(f"const RUN_HISTORY = {json.dumps(history, indent=2)};")
    print(f"Run history database updated in {history_js_path}")

def main():
    # 1. Run Unit Tests programmatically
    test_results = run_unit_tests()
    
    # 2. Parse config file path from CLI or default
    config_path = "configs/default_experiment.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    # 3. Load configuration
    config, resolved_path = load_config(config_path)
    config_name = os.path.splitext(os.path.basename(resolved_path))[0]
    print(f"Loaded config: {resolved_path}")
    
    seed = config.get("seed", 1729)
    ticks = config.get("ticks", 10000)
    scarcity = config.get("scarcity", 1.0)
    
    long_run = config.get("long_run", False)
    save_paths = config.get("save_paths", not long_run)
    save_epochs = config.get("save_epochs", long_run)
    sample_interval = config.get("sample_interval", 1000)
    live_pacing_delay = config.get("live_pacing_delay", 0.03)

    
    # 4. Generate World State
    print(f"Generating world: Seed = {seed}, size = 1024x1024...")
    world = generate_world(width=1024, height=1024, seed=seed)
    
    # 5. Spawn Custom Config-driven Agents
    spawn_custom_agents(world, config.get("agents", []))
    
    prediction_enabled = config.get("prediction_enabled", True)
    
    import time
    from world.agents.behavior_classifier import classify_behavior

    # Define a highly enriched live callback that writes comprehensive live_state every 100 ticks
    def live_callback(tick, epoch_stats):
        import json
        import os
        import time
        import numpy as np
        if tick % 100 == 0 or tick == 1:
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
                    "season_observations": {int(k): int(v) for k, v in agent.season_observations.items()} if hasattr(agent, "season_observations") else {0:0, 1:0, 2:0, 3:0}
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
            if live_pacing_delay > 0.0:
                time.sleep(live_pacing_delay)

                
    # Automatically open visualizer.html in the default web browser with ?live=true
    import webbrowser
    try:
        vis_path = os.path.abspath("visualizer.html")
        webbrowser.open("file://" + vis_path + "?live=true")
        print("Automatically opened visualizer.html in your default browser connected to the Live Stream.")
    except Exception as e:
        print(f"Warning: Failed to automatically open visualizer in browser: {e}")

    # 6. Execute Simulation Loop
    print(f"Starting custom simulation: ticks = {ticks}, scarcity = {scarcity}...")
    epoch_stats = run_simulation(
        world, 
        ticks=ticks, 
        scarcity_level=scarcity, 
        save_paths=save_paths, 
        save_epochs=save_epochs, 
        sample_interval=sample_interval,
        live_callback=live_callback,
        prediction_enabled=prediction_enabled
    )
    
    # 7. Generate diagnostic visual traces map
    generate_simulation_map(world, "simulation.png")
    
    # 8. Collect telemetry metadata summaries
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
    
    summary_record = {
        "timestamp": timestamp,
        "experiment": config_name,
        "seed": seed,
        "ticks": ticks,
        "scarcity": scarcity,
        "survivors": f"{alive_count}/{total_agents}",
        "avg_radius": round(avg_radius, 1),
        "avg_discoveries": round(avg_discoveries, 1),
        "tests_passed": test_results["status"] == "PASS"
    }
    
    run_metadata = {
        "git_version": git_version,
        "experiment_name": config_name
    }
    
    # 9. Update histories database
    update_history(summary_record)
    
    # 10. Dump simulation JS data
    save_simulation_data(
        world=world,
        experiment_name=config_name,
        scarcity_val=scarcity,
        filepath="simulation_data.js",
        test_results=test_results,
        run_metadata=run_metadata,
        save_paths=save_paths,
        epoch_stats=epoch_stats if save_epochs else None
    )
    
    print("\nCustom run execution complete!")
    print(f"  Survivors: {alive_count}/{total_agents}")
    print(f"  Avg discoveries: {summary_record['avg_discoveries']}")
    print(f"  Avg radius: {summary_record['avg_radius']} cells")
    print("Open visualizer.html in your browser to interactively scrub paths and compare run logs.")

if __name__ == "__main__":
    main()
