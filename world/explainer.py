import numpy as np
from scipy.ndimage import distance_transform_edt
from .state import WorldState, BIOME_NAMES, OCEAN, GLACIER, TUNDRA, TAIGA, TEMPERATE_FOREST, GRASSLAND, DESERT, RAINFOREST, LAKE

def explain_tile(world: WorldState, x: int, y: int, sea_level: float = 0.3) -> dict:
    """
    Analyzes the physical and socioeconomic parameters of a coordinate (x, y) 
    and generates a detailed geographic explanation of its properties.
    
    Returns a dictionary of raw values and a natural language explanation.
    """
    if x < 0 or x >= world.width or y < 0 or y >= world.height:
        return {"error": "Coordinates out of bounds"}
        
    elev = world.elevation[y, x]
    temp = world.temperature[y, x]
    rain = world.rainfall[y, x]
    biome_id = world.biome[y, x]
    biome_name = BIOME_NAMES.get(biome_id, "Unknown")
    
    # Hydrology
    riv_acc = world.river_map[y, x]
    is_lake = world.lake_map[y, x] > 0.0
    
    # Calculate approximate physical elevation in meters (max peaks around 3000m)
    elev_m = int(elev * 3000)
    
    # Base explanation by biome
    base_explanation = ""
    
    if biome_id == OCEAN:
        depth_m = int((sea_level - elev) * 3000)
        base_explanation = (
            f"This tile is part of the ocean. Its elevation ({elev_m}m) is below the "
            f"sea level threshold, putting it under approximately {depth_m} meters of water."
        )
    elif biome_id == LAKE:
        base_explanation = (
            f"This tile lies in a fresh water body. It forms a natural lake/river basin, "
            f"pooling drainage from the surrounding slopes."
        )
    elif biome_id == GLACIER:
        base_explanation = (
            f"This tile is covered in glacial ice. It sits at a high elevation "
            f"({elev_m}m) in a high-latitude region, dropping its annual temperature "
            f"to {temp:.1f}°C, which prevents snow from melting."
        )
    elif biome_id == TUNDRA:
        base_explanation = (
            f"This is tundra. The climate is very cold ({temp:.1f}°C) with low rainfall "
            f"({rain:.0f}mm), supporting only mosses, lichens, and dwarf shrubs."
        )
    elif biome_id == TAIGA:
        base_explanation = (
            f"This belongs to a taiga forest. The cool climate ({temp:.1f}°C) and moderate "
            f"rainfall ({rain:.0f}mm) support dense evergreen pine forests."
        )
    elif biome_id == TEMPERATE_FOREST:
        base_explanation = (
            f"This is a temperate forest. It features a moderate climate ({temp:.1f}°C) and "
            f"abundant rainfall ({rain:.0f}mm), allowing deciduous broadleaf trees to thrive."
        )
    elif biome_id == GRASSLAND:
        base_explanation = (
            f"This is grassland/savanna. The climate is warm ({temp:.1f}°C) with moderate rainfall "
            f"({rain:.0f}mm), supporting rich grasses but too dry for dense forests."
        )
    elif biome_id == RAINFOREST:
        base_explanation = (
            f"This is a tropical rainforest. Strong solar heating (temperature: {temp:.1f}°C) and "
            f"moisture-laden wind advection produce high rainfall ({rain:.0f}mm), forming a dense canopy."
        )
    elif biome_id == DESERT:
        # Check for rain shadow effect: search for high mountains to the west (upwind) at the same latitude y
        search_dist = 180
        start_x = max(0, x - search_dist)
        west_elevs = world.elevation[y, start_x:x]
        
        if len(west_elevs) > 0 and np.max(west_elevs) > 0.50:
            max_west_elev = np.max(west_elevs)
            peak_x = start_x + np.argmax(west_elevs)
            peak_m = int(max_west_elev * 3000)
            base_explanation = (
                f"This is a desert because it lies in the rain shadow of the high mountain range to the "
                f"west (peaks up to {peak_m}m at coordinate x={peak_x}). Westerly winds dropped their "
                f"moisture on the western slopes before descending here as hot, dry air."
            )
        else:
            base_explanation = (
                f"This is a desert due to its high annual temperatures ({temp:.1f}°C) "
                f"and low average precipitation ({rain:.0f}mm)."
            )
            
    # Socioeconomic/Resource explanations
    res_explanation = []
    if world.wood[y, x] > 0.5:
        res_explanation.append("wood is dense and abundant, providing rich forestry opportunities")
    if world.fertility[y, x] > 0.6:
        res_explanation.append("the soil is highly fertile (river valley/floodplain) and ideal for farming")
    if world.iron[y, x] > 0.4:
        res_explanation.append("rich iron deposits are exposed here, forming part of a mountain mineral belt")
    elif world.copper[y, x] > 0.4:
        res_explanation.append("significant copper veins are accessible here, forming part of a mountain mineral belt")
        
    res_text = ""
    if res_explanation:
        res_text = " Resource-wise, " + ", while ".join(res_explanation) + "."
        
    # Passability and Trade explanations
    pass_text = ""
    if biome_id not in (OCEAN, LAKE):
        m_cost = world.movement_cost[y, x]
        t_pot = world.trade_potential[y, x]
        
        if m_cost > 3.5:
            pass_desc = "travel is extremely slow and hazardous due to steep slopes or rugged terrain"
        elif m_cost > 1.8:
            pass_desc = "travel is moderately slow due to forest cover or sandy terrain"
        else:
            pass_desc = "travel is fast and easy across the flat open plains"
            
        trade_desc = []
        if riv_acc > 5000.0:
            trade_desc.append("access to a major navigable river highway")
        # Check ocean distance for harbor potential
        is_ocean = world.elevation < sea_level
        dist_ocean = distance_transform_edt(~is_ocean)[y, x]
        if dist_ocean < 12.0:
            trade_desc.append("proximity to the coastline provides natural maritime harbor potential")
        if t_pot > 65.0:
            trade_desc.append("the intersection of diverse resources nearby attracts merchant travel")
            
        trade_text = ""
        if trade_desc:
            trade_text = f" Economically, it has high trade potential ({t_pot:.0f}) thanks to " + " and ".join(trade_desc) + "."
        else:
            trade_text = f" Economically, trade potential is low ({t_pot:.0f}) due to geographic isolation."
            
        pass_text = f" Regarding passability, {pass_desc}.{trade_text}"
        
    full_explanation = f"{base_explanation}{res_text}{pass_text}"
    
    return {
        "x": x,
        "y": y,
        "elevation_m": elev_m,
        "temperature_c": temp,
        "rainfall_mm": rain,
        "biome_id": int(biome_id),
        "biome_name": biome_name,
        
        # Hydrology details
        "river_flow": float(riv_acc),
        "is_lake": bool(is_lake),
        
        # Resource richness [0, 100]
        "wood": float(world.wood[y, x] * 100),
        "stone": float(world.stone[y, x] * 100),
        "copper": float(world.copper[y, x] * 100),
        "iron": float(world.iron[y, x] * 100),
        "wildlife": float(world.wildlife[y, x] * 100),
        "fertility": float(world.fertility[y, x] * 100),
        
        # Habitability details [0, 100]
        "water_score": float(world.water_score[y, x] * 100),
        "food_score": float(world.food_score[y, x] * 100),
        "resource_score": float(world.resource_score[y, x] * 100),
        "climate_score": float(world.climate_score[y, x] * 100),
        "terrain_score": float(world.terrain_score[y, x] * 100),
        "habitability": float(world.habitability[y, x]),
        
        # Passability
        "movement_cost": float(world.movement_cost[y, x]),
        "trade_potential": float(world.trade_potential[y, x]),
        
        "explanation": full_explanation
    }
