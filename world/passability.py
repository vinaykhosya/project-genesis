import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from .state import (
    WorldState, OCEAN, GLACIER, TUNDRA, TAIGA, 
    TEMPERATE_FOREST, GRASSLAND, DESERT, RAINFOREST, LAKE
)

def calculate_passability(world: WorldState, sea_level: float = 0.3) -> WorldState:
    """
    Simulates physical travel constraints and economic trade potential across the world.
    
    1. Movement Cost: Traversal difficulty based on biome baseline costs and steepness penalties.
    2. Trade Potential: High near coastlines, navigable river deltas, flat terrain corridors,
       and boundaries overlapping different resources (wood/metal/fertile soil).
       
    Parameters:
    - world: The WorldState container to modify.
    - sea_level: The sea level threshold.
    
    Returns:
    - The modified WorldState container.
    """
    h, w = world.height, world.width
    shape = (h, w)
    
    # --- 1. Movement Cost Calculation ---
    # Baseline traversal cost by biome (1.0 = easiest open grassland, higher = slower)
    base_cost = np.ones(shape, dtype=np.float32)
    base_cost[world.biome == OCEAN] = 10.0             # Impassable ocean barrier without ships
    base_cost[world.biome == LAKE] = 10.0              # Impassable deep lake water
    base_cost[world.biome == GLACIER] = 5.0            # Extremely hazardous ice crevasses
    base_cost[world.biome == RAINFOREST] = 3.0          # Dense, choked tropical undergrowth
    base_cost[world.biome == DESERT] = 2.5             # Soft shifting sand dunes
    base_cost[world.biome == TAIGA] = 2.0              # Pine forest and swampy muskeg
    base_cost[world.biome == TEMPERATE_FOREST] = 1.8   # Hilly deciduous woodland
    base_cost[world.biome == TUNDRA] = 1.5             # Frozen mossy plains
    base_cost[world.biome == GRASSLAND] = 1.0          # Flat, open grasslands (ideal path)
    
    # Calculate slope penalty
    dy, dx = np.gradient(world.elevation)
    slope = np.sqrt(dy**2 + dx**2)
    
    # Slope scaling: steep cliffs add up to +15.0 traversal penalty
    slope_penalty = slope * 250.0
    
    # Final movement cost
    world.movement_cost = base_cost + slope_penalty
    
    # --- 2. Trade Potential Calculation ---
    # A. Water Access (Rivers and Coasts)
    # Navigable river trade: high near massive rivers
    river_nav = np.clip(world.river_map / 10000.0, 0.0, 1.0)
    
    # Maritime coastal trade: high on land close to oceans
    is_ocean = world.elevation < sea_level
    dist_to_ocean = distance_transform_edt(~is_ocean)
    ocean_coast = np.exp(-dist_to_ocean / 15.0) * (world.elevation >= sea_level)
    
    water_trade_access = np.maximum(river_nav, ocean_coast)
    
    # B. Flat Terrain Passability (easier travel means easier trade)
    # Inverse of movement cost normalized (open plains = 1.0, mountains/oceans -> 0)
    flatness_factor = 1.0 / world.movement_cost
    
    # C. Resource Diversity Index
    # Trade occurs where resources overlap (e.g. metals near forests)
    # Blur resource maps to find general proximity intersection regions
    blur_wood = gaussian_filter(world.wood, 15.0)
    blur_stone = gaussian_filter(world.stone, 15.0)
    blur_iron = gaussian_filter(world.iron, 15.0)
    blur_copper = gaussian_filter(world.copper, 15.0)
    blur_fertility = gaussian_filter(world.fertility, 15.0)
    
    # Summing presence of wood, metal, stone, and agriculture
    resource_diversity = (
        blur_wood * 0.25 + 
        blur_fertility * 0.25 + 
        (blur_iron + blur_copper) * 0.35 + 
        blur_stone * 0.15
    )
    
    # Combine trade potential: 40% water trade lanes, 30% passability, 30% resource diversity
    raw_trade = 0.40 * water_trade_access + 0.30 * flatness_factor + 0.30 * resource_diversity
    
    # Normalize to [0, 100]
    world.trade_potential = (raw_trade * 100.0).astype(np.float32)
    
    # Zero out water tiles for city trade centers
    is_water = (world.elevation < sea_level) | (world.biome == LAKE) | (world.lake_map > 0.0)
    world.trade_potential[is_water] = 0.0
    
    return world
