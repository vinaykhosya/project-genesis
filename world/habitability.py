import numpy as np
from scipy.ndimage import distance_transform_edt
from .state import WorldState, LAKE

def calculate_habitability(world: WorldState, sea_level: float = 0.3) -> WorldState:
    """
    Computes a multi-factor habitability layer for the world.
    
    Generates five sub-scores:
    1. water_score: Proximity to fresh water (rivers and lakes)
    2. food_score: Blends fertility (soil) and wildlife
    3. resource_score: Blends wood, stone, copper, and iron
    4. climate_score: Temperature and rainfall comfort index
    5. terrain_score: Flatness index (slope penalty)
    
    Combines them into a weighted composite habitability score in range [0, 100].
    Terrestrial water bodies (oceans and lakes) are zeroed out.
    
    Parameters:
    - world: The WorldState container to modify.
    - sea_level: The sea level elevation threshold.
    
    Returns:
    - The modified WorldState container.
    """
    h, w = world.height, world.width
    shape = (h, w)
    
    # 1. Proximity to Fresh Water
    is_fresh_water = (world.biome == LAKE) | (world.lake_map > 0.0) | (world.river_map > 1500.0)
    dist_fresh = distance_transform_edt(~is_fresh_water)
    # Exponential decay over distance (decays to near 0 around 80 pixels / ~240km)
    world.water_score = np.exp(-dist_fresh / 25.0).astype(np.float32)
    
    # 2. Food Score (60% Fertility, 40% Wildlife)
    world.food_score = (world.fertility * 0.6 + world.wildlife * 0.4).astype(np.float32)
    
    # 3. Resource Score (40% Wood, 30% Stone, 15% Iron, 15% Copper)
    world.resource_score = (world.wood * 0.4 + world.stone * 0.3 + world.iron * 0.15 + world.copper * 0.15).astype(np.float32)
    
    # 4. Climate Score (Temperature and Rainfall comfort)
    # Optimum temperature is 18°C, falls off towards freezing (-10°C) and hot (40°C)
    temp_comfort = np.exp(-((world.temperature - 18.0)**2) / (2.0 * (10.0**2)))
    # Optimum rainfall is 1000mm, dry deserts and swampy regions fall off
    rain_comfort = np.exp(-((world.rainfall - 1000.0)**2) / (2.0 * (500.0**2)))
    world.climate_score = (temp_comfort * rain_comfort).astype(np.float32)
    
    # 5. Terrain Score (Slope penalty)
    dy, dx = np.gradient(world.elevation)
    slope = np.sqrt(dy**2 + dx**2)
    # Steeper slopes linearly drop building suitability (flat terrain = 1.0, slope >= 0.06 is penalized to 0.0)
    world.terrain_score = np.maximum(0.0, 1.0 - slope * 16.0).astype(np.float32)
    
    # 6. Combined Habitability (Weighted Sum)
    # Proximity to water and food are critical (30% each).
    # Climate comfort is very important (20%).
    # Local metals/stone and buildable flatness are secondary (10% each).
    weighted_hab = (
        0.30 * world.water_score +
        0.30 * world.food_score +
        0.20 * world.climate_score +
        0.10 * world.resource_score +
        0.10 * world.terrain_score
    )
    
    # Convert to standard range [0, 100]
    world.habitability = (weighted_hab * 100.0).astype(np.float32)
    
    # Zero out water tiles (humans can't settle on oceans or lakes)
    is_water_tile = (world.elevation < sea_level) | (world.biome == LAKE) | (world.lake_map > 0.0)
    world.habitability[is_water_tile] = 0.0
    
    return world
