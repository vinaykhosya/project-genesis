import numpy as np
from .agent import Agent, WATER, FOOD, PERSON, LANDMARK, DANGER
from ..state import WorldState, LAKE

def perceive(agent: Agent, world: WorldState, vision_radius: int = 20, chunk_size: int = 32):
    """
    Limits agent knowledge by slicing the environment within a local radius.
    Updates the agent's visited chunks set and catalogs local resources (water, food),
    other agents, and mountain peak landmarks into episodic memory and generalized knowledge.
    
    Parameters:
    - agent: The Agent performing perception.
    - world: The WorldState container.
    - vision_radius: Vision distance in grid cells.
    - chunk_size: Grid cell size of memory chunks.
    """
    if agent.dead:
        return
        
    agent.perception_calls += 1
    cy, cx = agent.location
    h, w = world.height, world.width
    
    # 1. Calculate bounding coordinates of the vision circle/box
    y_min = max(0, cy - vision_radius)
    y_max = min(h, cy + vision_radius + 1)
    x_min = max(0, cx - vision_radius)
    x_max = min(w, cx + vision_radius + 1)
    
    # 2. Record visited chunks in set (N * N grid chunk indices)
    chunk_y_min, chunk_y_max = y_min // chunk_size, (y_max - 1) // chunk_size
    chunk_x_min, chunk_x_max = x_min // chunk_size, (x_max - 1) // chunk_size
    
    for chunk_y in range(chunk_y_min, chunk_y_max + 1):
        for chunk_x in range(chunk_x_min, chunk_x_max + 1):
            agent.visited_chunks.add((chunk_y, chunk_x))
            
    # 3. Detect Fresh Water (Lakes, major Rivers; exclude saltwater Ocean)
    # Shape of sliced arrays is (y_max - y_min, x_max - x_min)
    sliced_biome = world.biome[y_min:y_max, x_min:x_max]
    sliced_lake = world.lake_map[y_min:y_max, x_min:x_max]
    sliced_river = world.river_map[y_min:y_max, x_min:x_max]
    
    is_water = (sliced_biome == LAKE) | (sliced_lake > 0.0) | (sliced_river > 1500.0)
    
    water_coords = np.argwhere(is_water)
    if len(water_coords) > 0:
        # Convert back to global grid coordinates
        global_water = water_coords + np.array([y_min, x_min])
        # Find the single closest water source coordinate
        dists = np.sum((global_water - np.array([cy, cx]))**2, axis=-1)
        closest_idx = np.argmin(dists)
        closest_water_coord = (int(global_water[closest_idx][0]), int(global_water[closest_idx][1]))
        
        if closest_water_coord not in agent.discovered_water:
            agent.discoveries_count += 1
            agent.discovered_water.add(closest_water_coord)
        elif closest_water_coord not in agent.knowledge.water_sources:
            agent.rediscoveries += 1
            
        # Record water memory and update confidence-weighted water knowledge
        agent.add_memory(WATER, closest_water_coord, world.tick, importance=0.8)
        
    # 4. Detect Food (Wildlife > 0.15 or Fertility > 0.4; exclude sub-sea oceans/lakes)
    sliced_elev = world.elevation[y_min:y_max, x_min:x_max]
    sliced_wildlife = world.wildlife[y_min:y_max, x_min:x_max]
    sliced_fertility = world.fertility[y_min:y_max, x_min:x_max]
    
    is_land = sliced_elev >= 0.3
    is_food = is_land & ((sliced_wildlife > 0.15) | (sliced_fertility > 0.4)) & (sliced_biome != LAKE)
    
    food_coords = np.argwhere(is_food)
    if len(food_coords) > 0:
        global_food = food_coords + np.array([y_min, x_min])
        dists = np.sum((global_food - np.array([cy, cx]))**2, axis=-1)
        closest_idx = np.argmin(dists)
        closest_food_coord = (int(global_food[closest_idx][0]), int(global_food[closest_idx][1]))
        
        if closest_food_coord not in agent.discovered_food:
            agent.discoveries_count += 1
            agent.discovered_food.add(closest_food_coord)
        elif closest_food_coord not in agent.knowledge.food_sources:
            agent.rediscoveries += 1
            
        agent.add_memory(FOOD, closest_food_coord, world.tick, importance=0.7)
        
    # 5. Detect Other Agents & Skeletons/Corpses
    # 5. Detect Other Agents & Skeletons/Corpses
    candidates = world.query_agents((cy, cx), vision_radius, alive_only=False)
    for other in candidates:
        if other.id != agent.id:
            oy, ox = other.location
            if not other.dead:
                if other.id not in agent.known_agents:
                    agent.known_agents.add(other.id)
                    agent.discoveries_count += 1
                agent.add_memory(PERSON, (int(oy), int(ox)), world.tick, importance=0.5)
            else:
                agent.add_memory(DANGER, (int(oy), int(ox)), world.tick, importance=0.90)

                
    # 6. Detect Landmarks (High elevation peak peaks > 0.8)
    is_peak = is_land & (sliced_elev > 0.8)
    peak_coords = np.argwhere(is_peak)
    if len(peak_coords) > 0:
        global_peaks = peak_coords + np.array([y_min, x_min])
        dists = np.sum((global_peaks - np.array([cy, cx]))**2, axis=-1)
        closest_idx = np.argmin(dists)
        closest_peak_coord = (int(global_peaks[closest_idx][0]), int(global_peaks[closest_idx][1]))
        
        if closest_peak_coord not in agent.discovered_landmarks:
            agent.discovered_landmarks.add(closest_peak_coord)
            agent.discoveries_count += 1
            
        agent.add_memory(LANDMARK, closest_peak_coord, world.tick, importance=0.4)
