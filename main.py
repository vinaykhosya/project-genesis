import os
import argparse
import numpy as np
import json
import datetime
from PIL import Image
from world.generator import generate_world
from world.agents.behavior_classifier import classify_behavior, BEHAVIOR_FEATURE_NAMES
from world.biomes import biomes_to_rgb, BIOME_COLORS
from world.explainer import explain_tile
from world.predictor import predict_settlements
from world.state import LAKE, WorldState, BIOME_NAMES
from world.agents.genetics import GENE_NAMES
from world.agents.drives import emotional_label
import colorsys
from world.agents.simulation import run_simulation


def interpolate_colormap(data: np.ndarray, stops: list) -> np.ndarray:
    """
    Linearly interpolates 2D float data values across a list of RGB color stops.
    """
    h, w = data.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    stops = sorted(stops, key=lambda x: x[0])
    
    for i in range(len(stops) - 1):
        v0, c0 = stops[i]
        v1, c1 = stops[i+1]
        
        mask = (data >= v0) & (data <= v1)
        if np.any(mask):
            t = (data[mask] - v0) / (v1 - v0)
            t = np.expand_dims(t, axis=-1)
            
            c0_arr = np.array(c0, dtype=np.float32)
            c1_arr = np.array(c1, dtype=np.float32)
            
            rgb[mask] = c0_arr * (1.0 - t) + c1_arr * t
            
    # Clamp elements falling out of range bounds
    below_mask = data < stops[0][0]
    if np.any(below_mask):
        rgb[below_mask] = np.array(stops[0][1], dtype=np.float32)
        
    above_mask = data > stops[-1][0]
    if np.any(above_mask):
        rgb[above_mask] = np.array(stops[-1][1], dtype=np.float32)
        
    return np.clip(rgb, 0, 255).astype(np.uint8)

def render_rivers_map(world, sea_level=0.3):
    """Generates a high-quality visualization of oceans, lakes, and river networks."""
    h, w = world.elevation.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Base land color: soft, clean parchment tan
    rgb[:, :] = (235, 230, 220)
    
    # 2. Ocean: dark navy blue
    is_ocean = world.elevation < sea_level
    rgb[is_ocean] = (15, 37, 65)
    
    # 3. Lakes: fresh sky blue
    is_lake = (world.biome == LAKE) | (world.lake_map > 0.0)
    rgb[is_lake] = (115, 195, 235)
    
    # 4. Rivers: draw channels where accumulation is high
    is_land = ~is_ocean & ~is_lake
    is_river = is_land & (world.river_map > 1500.0)
    rgb[is_river] = (50, 120, 200)
    
    return rgb

def draw_settlement_markers(img, settlements):
    """Draws target indicators (white circles with black crosshairs) at settlement coordinates."""
    h, w, c = img.shape
    for s in settlements:
        cx, cy = s['x'], s['y']
        
        # Draw concentric white circle border of radius 6
        for r in (5, 6):
            for angle in np.linspace(0, 2*np.pi, 24):
                px = int(cx + r * np.cos(angle))
                py = int(cy + r * np.sin(angle))
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = (255, 255, 255)
                    
        # Draw a small black crosshair in the center
        for d in range(-2, 3):
            if 0 <= cx + d < w and 0 <= cy < h:
                img[cy, cx + d] = (0, 0, 0)
            if 0 <= cx < w and 0 <= cy + d < h:
                img[cy + d, cx] = (0, 0, 0)

def run_query(world, x, y, settlements_list):
    """Executes an advanced query, printing all resource, passability, and trade data."""
    analysis = explain_tile(world, x, y)
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
        
    # Find if coordinate is a predicted settlement
    settlement_rank = "N/A"
    for s in settlements_list:
        if s['x'] == x and s['y'] == y:
            settlement_rank = f"Rank #{s['rank']} Prime Location"
            break
            
    print("\n" + "="*65)
    print(f"GEOGRAPHIC & RESOURCE ANALYSIS FOR COORDINATE ({x}, {y})")
    print("="*65)
    print(f"Biome:          {analysis['biome_name']} ({settlement_rank})")
    print(f"Elevation:      {analysis['elevation_m']}m")
    print(f"Temperature:    {analysis['temperature_c']:.1f}°C")
    print(f"Rainfall:       {analysis['rainfall_mm']:.0f}mm")
    
    print("\n--- Resources (Richness 0-100) ---")
    print(f"  Wood:      {analysis['wood']:.1f}%")
    print(f"  Stone:     {analysis['stone']:.1f}%")
    print(f"  Iron:      {analysis['iron']:.1f}%")
    print(f"  Copper:    {analysis['copper']:.1f}%")
    print(f"  Wildlife:  {analysis['wildlife']:.1f}%")
    print(f"  Fertility: {analysis['fertility']:.1f}%")
    
    print("\n--- Habitability Analysis ---")
    print(f"  Water Score:   {analysis['water_score']:.0f}/100")
    print(f"  Food Score:    {analysis['food_score']:.0f}/100")
    print(f"  Resource:      {analysis['resource_score']:.0f}/100")
    print(f"  Climate:       {analysis['climate_score']:.0f}/100")
    print(f"  Terrain:       {analysis['terrain_score']:.0f}/100")
    print(f"  Composite:     {analysis['habitability']:.0f}/100")
    
    print("\n--- Travel & Economics ---")
    print(f"  Traversal Cost:  {analysis['movement_cost']:.2f} (Grassland = 1.0, Ocean = 10.0)")
    print(f"  Trade Potential: {analysis['trade_potential']:.0f}/100")
    
    print(f"\nExplanation:")
    import textwrap
    for line in textwrap.wrap(analysis["explanation"], width=62):
        print(f"  {line}")
    print("="*65 + "\n")

def get_agent_colors(num_agents):
    """Generates a list of distinct colors for drawing agent traces."""
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def generate_simulation_map(world, filepath="simulation.png"):
    """
    Renders the biomes map and overlays the color-coded movement path traces of all agents.
    Saves the output to simulation.png.
    """
    # 1. Start with the base biomes map
    bg_rgb = biomes_to_rgb(world.biome)
    
    # Soften the background so paths stand out
    bg_rgb = (bg_rgb.astype(np.float32) * 0.7 + 76.5).astype(np.uint8)
    
    h, w, c = bg_rgb.shape
    
    # 2. Get distinct colors for agents
    agent_colors = get_agent_colors(len(world.agents))
    
    for idx, agent in enumerate(world.agents):
        color = agent_colors[idx]
        
        # Draw path history
        for coord in agent.path_history:
            cy, cx = coord
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        bg_rgb[ny, nx] = color
                        
        # Draw spawn location
        sy, sx = agent.path_history[0]
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if abs(dy) == 2 or abs(dx) == 2:
                        bg_rgb[ny, nx] = (0, 0, 0)
                        
        # Draw final location
        fy, fx = agent.location
        if agent.dead:
            for d in (-3, -2, -1, 0, 1, 2, 3):
                if 0 <= fy + d < h and 0 <= fx + d < w:
                    bg_rgb[fy + d, fx + d] = (255, 0, 0)
                if 0 <= fy + d < h and 0 <= fx - d < w:
                    bg_rgb[fy + d, fx - d] = (255, 0, 0)
        else:
            for dy in (-3, -2, -1, 0, 1, 2, 3):
                for dx in (-3, -2, -1, 0, 1, 2, 3):
                    ny, nx = fy + dy, fx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        dist_sq = dy**2 + dx**2
                        if dist_sq <= 9:
                            bg_rgb[ny, nx] = (255, 255, 255)
                        if dist_sq <= 4:
                            bg_rgb[ny, nx] = color

    Image.fromarray(bg_rgb).save(filepath)
    print(f"Simulation traces map successfully saved to {filepath}")

def get_path_metrics(sampled_path_history, home_location):
    """
    Returns (exploration_radius, home_drift, sampled_path) from sampled_path_history.
    Handles both legacy (y, x) and enriched 8-element formats.
    """
    if not sampled_path_history:
        return 0.0, 0.0, []
        
    first = sampled_path_history[0]
    if len(first) > 2:
        # Enriched format: [x, y, action_id, health, hunger, thirst, energy, generation]
        start_x, start_y = first[0], first[1]
        path_coords = np.array([[coord[0], coord[1]] for coord in sampled_path_history])
        sampled_path = [
            [
                int(coord[0]), # x
                int(coord[1]), # y
                int(coord[2]), # action_id
                round(float(coord[3]), 1), # health
                round(float(coord[4]), 1), # hunger
                round(float(coord[5]), 1), # thirst
                round(float(coord[6]), 1), # energy
                int(coord[7]) # generation
            ] for coord in sampled_path_history
        ]
    else:
        # Legacy format: (y, x)
        start_y, start_x = first[0], first[1]
        path_coords = np.array(sampled_path_history)
        sampled_path = [[int(coord[1]), int(coord[0])] for coord in sampled_path_history]
        
    start_coords = np.array([start_x, start_y]) if len(first) > 2 else np.array([start_y, start_x])
    dists = np.sqrt(np.sum((path_coords - start_coords)**2, axis=-1))
    exploration_radius = float(np.max(dists)) if len(dists) > 0 else 0.0
    home_drift = float(np.sqrt((home_location[0] - start_y)**2 + (home_location[1] - start_x)**2))
    
    return exploration_radius, home_drift, sampled_path

def save_simulation_data(world, experiment_name, scarcity_val, filepath="simulation_data.js", test_results=None, run_metadata=None, save_paths=True, epoch_stats=None):
    import json
    import datetime
    
    # Calculate sparse death density map to minimize JSON payload size
    sparse_density = []
    non_zero_coords = np.argwhere(world.death_density_map > 0.01)
    for cy, cx in non_zero_coords:
        sparse_density.append([int(cx), int(cy), round(float(world.death_density_map[cy, cx]), 3)])
        
    agents_data = []
    for agent in world.agents:
        if save_paths:
            # Calculate radius and home drift from sampled path history
            exploration_radius, home_drift, sampled_path = get_path_metrics(agent.sampled_path_history, agent.home_location)
        else:
            # Low-memory fallback: use max_radius and spawn_location
            exploration_radius = float(agent.max_radius)
            home_drift = float(np.sqrt((agent.home_location[0] - agent.spawn_location[0])**2 + (agent.home_location[1] - agent.spawn_location[1])**2))
            sampled_path = [[int(agent.location[1]), int(agent.location[0])]]
        
        # Convert action counts
        action_pct = {}
        total_actions = sum(agent.action_counts.values())
        if total_actions > 0:
            for act, count in agent.action_counts.items():
                action_pct[act] = round((count / total_actions) * 100.0, 1)
        else:
            action_pct = {k: 0.0 for k in agent.action_counts}
            
        # Convert rich spatial knowledge dicts to JS-friendly lists of coordinates and confidences
        water_nodes = []
        for loc, data in agent.knowledge.water_sources.items():
            water_nodes.append([int(loc[1]), int(loc[0]), float(data["confidence"])])
            
        food_nodes = []
        for loc, data in agent.knowledge.food_sources.items():
            food_nodes.append([int(loc[1]), int(loc[0]), float(data["confidence"])])
            
        agents_data.append({
            "id": agent.id,
            "archetype": agent.archetype,
            "spawn_biome": agent.spawn_biome,
            "traits": {k: float(v) for k, v in agent.traits.items()},
            "dead": bool(agent.dead),
            "age": int(agent.age),
            "drinks": int(agent.drinks_count),
            "eats": int(agent.eats_count),
            "failed_water": int(agent.failed_water_visits),
            "failed_food": int(agent.failed_food_visits),
            "resting_ticks": int(agent.resting_ticks),
            "discoveries": int(agent.discoveries_count),
            "recognized_agents": sorted(list(agent.known_agents)),
            "exploration_radius": round(exploration_radius, 1),
            "home_drift": round(home_drift, 1),
            "action_pct": action_pct,
            "sampled_path": sampled_path,
            "known_water": water_nodes,
            "known_food": food_nodes,
            "prediction_attempts": int(agent.prediction_attempts),
            "prediction_successes": int(agent.prediction_successes),
            "prediction_decisions": int(agent.prediction_decisions),
            "prediction_gains": int(agent.prediction_gains),
            "rediscoveries": int(agent.rediscoveries),
            "years_survived": int(agent.years_survived),
            "ticks_survived": int(agent.ticks_survived),
            "season_observations": {int(k): int(v) for k, v in agent.season_observations.items()},
            
            # Phase 4 Telemetry
            "max_age": int(agent.max_age),
            "cause_of_death": agent.cause_of_death,
            "primary_cause": agent.primary_cause,
            "secondary_cause": agent.secondary_cause,
            "fat_reserves": round(float(agent.fat_reserves), 1),
            "muscle_mass": round(float(agent.muscle_mass), 1),
            "injury_level": round(float(agent.injury_level), 1),
            "shelter_location": [int(agent.shelter_location[1]), int(agent.shelter_location[0])] if agent.shelter_location else None,
            "shelter_level": int(agent.shelter_level),
            "shelter_durability": round(float(agent.shelter_durability), 1),

            # Phase 5 — Genetics & Evolution
            "life_stage":        agent.life_stage,
            "colony_id":         int(getattr(agent, "colony_id", 0)),
            "generation":        int(getattr(agent, "generation", 0)),
            "parent_ids":        list(getattr(agent, "parent_ids", None) or []),
            "children_ids":      list(getattr(agent, "children_ids", [])),
            "born_tick":         int(getattr(agent, "born_tick", 0)),
            "behavior_cluster":  getattr(agent, "behavior_cluster", "C0"),
            "genome":            getattr(agent.genome, "to_list", lambda: [])() if hasattr(agent, "genome") else [],
            "stored_food":       round(float(getattr(agent, "stored_food",  0.0)), 1),
            "stored_water":      round(float(getattr(agent, "stored_water", 0.0)), 1),
            "repro_cooldown":    int(getattr(agent, "reproduction_cooldown", 0)),

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
        
    # Calculate global aggregates
    avg_discoveries = float(np.mean([a.discoveries_count for a in world.agents])) if world.agents else 0.0
    avg_radius = 0.0
    if world.agents:
        if save_paths:
            radii = []
            for a in world.agents:
                if len(a.sampled_path_history) > 0:
                    r, _, _ = get_path_metrics(a.sampled_path_history, a.home_location)
                    radii.append(r)
            avg_radius = float(np.mean(radii)) if radii else 0.0
        else:
            avg_radius = float(np.mean([a.max_radius for a in world.agents]))
        
    alive_count = sum(1 for a in world.agents if not a.dead)
    total_agents = len(world.agents)
    
    metadata = {
        "seed": int(world.seed),
        "ticks": int(world.tick),
        "scarcity": float(scarcity_val),
        "experiment": experiment_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "survivors": f"{alive_count}/{total_agents}",
        "avg_discoveries": round(avg_discoveries, 1),
        "avg_radius": round(avg_radius, 1),
        "tests_passed": True
    }
    
    if run_metadata is not None:
        metadata.update(run_metadata)
        
    if test_results is not None:
        metadata["tests_passed"] = (test_results.get("status") == "PASS")
        metadata["test_results"] = test_results
        
    # --- Phase 5: Run K-Means behavioral clustering before export ---
    living_agents = [a for a in world.agents if not a.dead]
    behavior_result = classify_behavior(living_agents, world_width=world.width, n_clusters=4)
    # K-Means cluster labels are applied to agents in-place by classify_behavior.
    # Centroid info is included in the JSON for the visualizer's Clusters tab.

    data = {
        "metadata": metadata,
        "experiment": experiment_name,
        "scarcity": scarcity_val,
        "agents": agents_data,
        "events": world.history,
        "death_density": sparse_density,
        "events_timeline": getattr(world, "events_timeline", []),

        # Phase 5 world-level evolution data
        "colonies":           getattr(world, "colonies",           []),
        "population_history": getattr(world, "population_history", []),
        "genetic_history":    getattr(world, "genetic_history",    []),
        "extinction_events":  getattr(world, "extinction_events",  []),
        "generation_number":  int(getattr(world, "generation_number", 0)),
        "total_births":       int(getattr(world, "total_births",    0)),
        "total_deaths":       int(getattr(world, "total_deaths",    0)),

        # Behavior clustering (K-Means, no hardcoded labels)
        "behavior_clustering": behavior_result,
        "gene_names": GENE_NAMES,
        "epoch_stats": epoch_stats if epoch_stats is not None else [],
        
        # Experimental Spawn Conditions metadata
        "spawn_conditions":   getattr(world, "spawn_conditions",   {}),
        "spawn_mode":         getattr(world, "spawn_mode",         "fixed"),
        
        # Centralized Telemetry (Phase 10)
        "telemetry": {
            "repro_potential_opportunities":   int(world.telemetry.repro_potential_opportunities),
            "repro_actual_opportunities":      int(world.telemetry.repro_actual_opportunities),
            "repro_mate_exists_world_ticks":   int(world.telemetry.repro_mate_exists_world_ticks),
            "repro_mate_exists_radius_ticks":  int(world.telemetry.repro_mate_exists_radius_ticks),
            "repro_relationship_ok_ticks":     int(world.telemetry.repro_relationship_ok_ticks),
            "repro_mate_eligible_ticks":      int(world.telemetry.repro_mate_eligible_ticks),
            "repro_wanted_reproduction_ticks": int(world.telemetry.repro_wanted_reproduction_ticks),
            "repro_mutual_will_ticks":         int(world.telemetry.repro_mutual_will_ticks),
            "repro_successes":                 int(world.telemetry.repro_successes),

            # Detailed reproduction failures
            "repro_fail_cooldown":      int(world.telemetry.repro_fail_cooldown),
            "repro_fail_health":        int(world.telemetry.repro_fail_health),
            "repro_fail_fat":           int(world.telemetry.repro_fail_fat),
            "repro_fail_hunger":        int(world.telemetry.repro_fail_hunger),
            "repro_fail_thirst":        int(world.telemetry.repro_fail_thirst),
            "repro_fail_shelter":       int(world.telemetry.repro_fail_shelter),
            "repro_fail_injury":        int(world.telemetry.repro_fail_injury),
            "repro_fail_distance":      int(world.telemetry.repro_fail_distance),
            "repro_fail_no_mate":       int(world.telemetry.repro_fail_no_mate),
            "repro_fail_relationship":  int(world.telemetry.repro_fail_relationship),
            "repro_fail_mate_ineligible": int(world.telemetry.repro_fail_mate_ineligible),
            "repro_fail_low_utility":   int(world.telemetry.repro_fail_low_utility),
            "repro_fail_mate_unwilling": int(world.telemetry.repro_fail_mate_unwilling),
            "repro_fail_no_mate_nearby": int(world.telemetry.repro_fail_no_mate_nearby),

            # Repro lost to counts
            "repro_lost_to_counts":     world.telemetry.repro_lost_to_counts,
            "repro_lost_margin_avg":    float(world.telemetry.repro_lost_margin_sum / max(1, world.telemetry.repro_lost_margin_count)) if world.telemetry.repro_lost_margin_count > 0 else 0.0,

            # Online running stats by age bracket
            "utility_stats_by_bracket": world.telemetry.utility_stats_by_bracket,

            # Water economy audit metrics
            "water_searches":           int(world.telemetry.water_searches),
            "water_search_successes":   int(world.telemetry.water_search_successes),
            "water_search_failures":    int(world.telemetry.water_search_failures),
            "water_search_distance_avg": float(world.telemetry.water_search_distance_sum / max(1, world.telemetry.water_search_distance_count)) if world.telemetry.water_search_distance_count > 0 else 0.0,
            "water_time_to_first_avg":  float(world.telemetry.water_time_to_first_sum / max(1, world.telemetry.water_time_to_first_count)) if world.telemetry.water_time_to_first_count > 0 else 0.0,
            "water_path_efficiency_avg": float(world.telemetry.water_path_efficiency_sum / max(1, world.telemetry.water_path_efficiency_count)) if world.telemetry.water_path_efficiency_count > 0 else 0.0,

            # Dehydration deaths
            "dehydration_deaths":          int(world.telemetry.dehydration_deaths),
            "died_carrying_water":         int(world.telemetry.died_carrying_water),
            "died_beside_water":           int(world.telemetry.died_beside_water),
            "died_with_remembered_water":  int(world.telemetry.died_with_remembered_water),
            "died_after_dry_visit":        int(world.telemetry.died_after_dry_visit),

            # Births per gen
            "births_by_generation":        world.telemetry.births_by_generation,
            
            # Resource timeline
            "resource_timeline":           world.telemetry.resource_timeline,
            
            # Phase 10.2: Structured hypotheses from explanation engine
            "scientific_hypotheses":       world.telemetry.generate_hypotheses()
        }
    }
    
    with open(filepath, "w") as f:
        if filepath.endswith(".json"):
            json.dump(data, f, indent=2)
        else:
            f.write(f"const SIMULATION_DATA = {json.dumps(data, indent=2)};")
    print(f"Simulation JSON database written to {filepath}")
    
    # Save Epoch Stats if present
    if epoch_stats is not None:
        with open("epoch_stats.json", "w") as f:
            json.dump(epoch_stats, f, indent=2)
        with open("epoch_stats.js", "w") as f:
            f.write(f"const EPOCH_STATS = {json.dumps(epoch_stats, indent=2)};")
        print("Epoch stats database written to epoch_stats.js and epoch_stats.json")
    else:
        # Overwrite with empty array to make sure old runs don't interfere
        with open("epoch_stats.js", "w") as f:
            f.write("const EPOCH_STATS = [];")

def print_experiment_report(world, experiment_type):
    print("\n" + "="*80)
    print(f"EMERGENCE VALIDATION REPORT: {experiment_type.upper()} EXPERIMENT")
    print("="*80)
    
    if len(world.agents) == 0:
        print("No agents spawned.")
        print("="*80 + "\n")
        return
        
    if experiment_type == "personality" or experiment_type == "default":
        # Group by Archetype
        groups = {}
        for a in world.agents:
            groups.setdefault(a.archetype, []).append(a)
            
        print(f"{'Archetype':<12} | {'Count':<5} | {'Alive':<6} | {'Avg Discoveries':<15} | {'Avg Radius':<11} | {'Avg Home Drift':<14} | {'Avg Eats/Drinks':<15}")
        print("-" * 90)
        for arch, members in sorted(groups.items()):
            count = len(members)
            alive = sum(1 for m in members if not m.dead)
            avg_disc = np.mean([m.discoveries_count for m in members])
            
            avg_radii = []
            avg_drifts = []
            for m in members:
                r, d, _ = get_path_metrics(m.sampled_path_history, m.home_location)
                avg_radii.append(r)
                avg_drifts.append(d)
            avg_rad = np.mean(avg_radii) if avg_radii else 0.0
            avg_drift = np.mean(avg_drifts) if avg_drifts else 0.0
            
            avg_eats = np.mean([m.eats_count for m in members])
            avg_drinks = np.mean([m.drinks_count for m in members])
            
            print(f"{arch:<12} | {count:<5} | {alive:<6} | {avg_disc:<15.1f} | {avg_rad:<11.1f} | {avg_drift:<14.1f} | {avg_eats:.1f} / {avg_drinks:.1f}")
            
    elif experiment_type == "environment":
        # Group by Spawn Biome
        groups = {}
        for a in world.agents:
            groups.setdefault(a.spawn_biome, []).append(a)
            
        print(f"{'Spawn Biome':<12} | {'Count':<5} | {'Alive':<6} | {'Avg Discoveries':<15} | {'Avg Radius':<11} | {'Avg Eats/Drinks':<15}")
        print("-" * 75)
        for biome, members in sorted(groups.items()):
            count = len(members)
            alive = sum(1 for m in members if not m.dead)
            avg_disc = np.mean([m.discoveries_count for m in members])
            
            avg_radii = []
            for m in members:
                r, _, _ = get_path_metrics(m.sampled_path_history, m.home_location)
                avg_radii.append(r)
            avg_rad = np.mean(avg_radii) if avg_radii else 0.0
            
            avg_eats = np.mean([m.eats_count for m in members])
            avg_drinks = np.mean([m.drinks_count for m in members])
            
            print(f"{biome:<12} | {count:<5} | {alive:<6} | {avg_disc:<15.1f} | {avg_rad:<11.1f} | {avg_eats:.1f} / {avg_drinks:.1f}")
            
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Genesis World Engine (Layer 1.5)")
    parser.add_argument("--seed", type=int, default=1729, help="Procedural generation random seed")
    parser.add_argument("--size", type=int, default=1024, help="Grid size of the map (square)")
    parser.add_argument("--query", nargs=2, type=int, metavar=("X", "Y"), help="Directly query coordinates and exit")
    parser.add_argument("--sim", type=int, help="Number of ticks to run the agent simulation")
    parser.add_argument("--experiment", type=str, default="default", choices=["default", "personality", "environment", "scarcity"], help="Experiment mode to run")
    parser.add_argument("--scarcity", type=float, default=1.0, help="Scarcity factor for resources (1.0 = normal, 0.5 = reduced, 0.2 = harsh)")
    args = parser.parse_args()
    
    # 1. Generate World
    print(f"Initializing World Generation Pipeline...")
    print(f"  Seed: {args.seed} | Size: {args.size}x{args.size}")
    world = generate_world(width=args.size, height=args.size, seed=args.seed)
    print("World Generation Complete!")
    
    # 2. Predict Settlements
    print("Running Settlement Predictor (NMS)...")
    settlements = predict_settlements(world, count=10, exclusion_radius=45.0)
    print("\n" + "="*50)
    print("TOP 10 RECOMMENDED SETTLEMENT LOCATIONS")
    print("="*50)
    for s in settlements:
        print(f"  #{s['rank']}: Coordinate ({s['x']}, {s['y']}) | Habitability: {s['score']:.0f}% | Biome: {s['biome']}")
    print("="*50 + "\n")
    
    # 3. Run Simulation if requested
    run_sim = (args.sim is not None) or (args.experiment != "default")
    scarcity_val = args.scarcity
    
    if run_sim:
        ticks_to_run = args.sim if args.sim is not None else 10000
        
        if args.experiment == "scarcity":
            scarcity_levels = [1.0, 0.5, 0.2]
            results = []
            last_world = None
            for level in scarcity_levels:
                print(f"\nRunning Scarcity Run: Level = {level}...")
                world_run = generate_world(width=args.size, height=args.size, seed=args.seed)
                run_simulation(world_run, ticks=ticks_to_run, experiment_type="scarcity", scarcity_level=level)
                
                alive_count = sum(1 for a in world_run.agents if not a.dead)
                total_agents = len(world_run.agents)
                
                avg_discoveries = np.mean([a.discoveries_count for a in world_run.agents])
                avg_radii = []
                for a in world_run.agents:
                    r, _, _ = get_path_metrics(a.sampled_path_history, a.home_location)
                    avg_radii.append(r)
                avg_radius = np.mean(avg_radii) if avg_radii else 0.0
                
                results.append({
                    "level": level,
                    "alive": alive_count,
                    "total": total_agents,
                    "avg_discoveries": avg_discoveries,
                    "avg_radius": avg_radius
                })
                last_world = world_run
                
            # Print scarcity comparison table
            print("\n" + "="*80)
            print("EMERGENCE VALIDATION REPORT: SCARCITY LEVEL COMPARISON")
            print("="*80)
            print(f"{'Scarcity Level':<15} | {'Survival Rate':<13} | {'Avg Discoveries':<15} | {'Avg Exploration Radius':<22}")
            print("-" * 75)
            for r in results:
                pct = int((r["alive"] / r["total"]) * 100.0)
                rate_str = f"{r['alive']}/{r['total']} ({pct}%)"
                print(f"{r['level']:<15.1f} | {rate_str:<13} | {r['avg_discoveries']:<15.1f} | {r['avg_radius']:<22.1f}")
            print("="*80 + "\n")
            
            world = last_world
            scarcity_val = 0.2
        else:
            print(f"Running Agent Simulation ({args.experiment}) for {ticks_to_run} ticks...")
            run_simulation(world, ticks=ticks_to_run, experiment_type=args.experiment, scarcity_level=scarcity_val)
            print_experiment_report(world, args.experiment)
    
    # --- Export 7 PNG Maps ---
    # Elevation map
    elevation_stops = [
        (0.0, (10, 20, 45)),      # Deep Ocean
        (0.3, (30, 75, 125)),     # Shallow Ocean/Coast
        (0.31, (220, 205, 160)),  # Beach Sand
        (0.45, (45, 115, 60)),    # Valley / Deciduous Forest
        (0.65, (105, 135, 85)),   # High grassy foothills
        (0.8, (120, 100, 80)),    # Exposed mountain rock
        (1.0, (245, 245, 245))    # Glacier-snow peaks
    ]
    print("Generating elevation.png...")
    elev_rgb = interpolate_colormap(world.elevation, elevation_stops)
    Image.fromarray(elev_rgb).save("elevation.png")
    
    # Temperature map
    temp_stops = [
        (-15.0, (30, 60, 140)),   # Polar freeze (blue)
        (0.0, (100, 180, 220)),   # Cold tundra (cyan)
        (12.0, (230, 220, 120)),  # Temperate forest (yellow)
        (24.0, (220, 100, 40)),   # Subtropical savanna (orange)
        (35.0, (160, 20, 20))     # Equatorial hot (red)
    ]
    print("Generating temperature.png...")
    temp_rgb = interpolate_colormap(world.temperature, temp_stops)
    Image.fromarray(temp_rgb).save("temperature.png")
    
    # Rainfall map
    rain_stops = [
        (0.0, (225, 200, 150)),    # Dry sand
        (250.0, (200, 205, 180)),  # Steppe
        (750.0, (140, 190, 160)),  # Woodland
        (1500.0, (80, 150, 190)),  # Wet forest
        (3000.0, (15, 60, 130))    # Rainforest
    ]
    print("Generating rainfall.png...")
    rain_rgb = interpolate_colormap(world.rainfall, rain_stops)
    Image.fromarray(rain_rgb).save("rainfall.png")
    
    # Biomes map
    print("Generating biomes.png...")
    biomes_rgb = biomes_to_rgb(world.biome)
    Image.fromarray(biomes_rgb).save("biomes.png")
    
    # Rivers and Lakes map
    print("Generating rivers.png...")
    rivers_rgb = render_rivers_map(world, sea_level=0.3)
    Image.fromarray(rivers_rgb).save("rivers.png")
    
    # Habitability map with overlay markers
    habitability_stops = [
        (0.0, (160, 40, 40)),      # Uninhabitable Red
        (30.0, (220, 100, 50)),    # Orange
        (50.0, (230, 210, 110)),   # Yellow
        (75.0, (120, 190, 100)),   # Light Green
        (100.0, (30, 130, 50))     # Dark Green (Ideal)
    ]
    print("Generating habitability.png...")
    hab_rgb = interpolate_colormap(world.habitability, habitability_stops)
    draw_settlement_markers(hab_rgb, settlements)
    Image.fromarray(hab_rgb).save("habitability.png")
    
    # Trade potential map
    trade_stops = [
        (0.0, (40, 25, 45)),       # Inactive dark purple
        (25.0, (110, 60, 120)),    # Low purple
        (50.0, (190, 100, 130)),   # Moderate pink
        (75.0, (235, 170, 110)),   # High peach
        (100.0, (250, 230, 140))   # Prime trade gold
    ]
    print("Generating trade.png...")
    trade_rgb = interpolate_colormap(world.trade_potential, trade_stops)
    Image.fromarray(trade_rgb).save("trade.png")
    
    # Generate simulation paths if run
    if run_sim:
        generate_simulation_map(world, "simulation.png")
        save_simulation_data(world, args.experiment, scarcity_val, "simulation_data.js")
        
    print("\nDiagnostic maps successfully saved to the workspace.")
    
    # If direct query is requested, print and exit
    if args.query:
        run_query(world, args.query[0], args.query[1], settlements)
        exit(0)
        
    # Interactive loop
    print("\nEntering Interactive Geographic Explorer Mode.")
    print("Type 'exit' or 'q' to quit.")
    print(f"Coordinates boundaries: X [0..{args.size-1}], Y [0..{args.size-1}]")
    
    while True:
        try:
            user_input = input("\nEnter 'X Y' coordinate to query: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break
            parts = user_input.split()
            if len(parts) != 2:
                print("Invalid format. Use space-separated integers, e.g. '500 500'.")
                continue
            qx, qy = int(parts[0]), int(parts[1])
            run_query(world, qx, qy, settlements)
        except ValueError:
            print("Invalid input: Coordinate inputs must be valid integers.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
