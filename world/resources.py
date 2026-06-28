import numpy as np
from scipy.ndimage import distance_transform_edt
from .state import (
    WorldState, OCEAN, GLACIER, TUNDRA, TAIGA, 
    TEMPERATE_FOREST, GRASSLAND, DESERT, RAINFOREST, LAKE
)
from .noise import fbm_noise_2d

def generate_resources(world: WorldState, sea_level: float = 0.3) -> WorldState:
    """
    Spawns wood, stone, copper, iron, wildlife, and fertility across the world.
    Resources are distributed logically based on biomes, elevations, and 
    low-frequency noise masks to create localized resource belts.
    
    Parameters:
    - world: The WorldState container to modify.
    - sea_level: The sea level threshold.
    
    Returns:
    - The modified WorldState container.
    """
    h, w = world.height, world.width
    shape = (h, w)
    
    # 1. Proximity to Water
    # Fresh water (Lakes, major Rivers) and Ocean coasts
    is_water = (world.elevation < sea_level) | (world.biome == LAKE) | (world.lake_map > 0.0) | (world.river_map > 1500.0)
    dist_to_water = distance_transform_edt(~is_water)
    
    # Calculate local slopes (needed for fertility floodplains and stone availability)
    dy, dx = np.gradient(world.elevation)
    slope = np.sqrt(dy**2 + dx**2)
    
    # 2. Wood Resources
    # Direct function of tree-growing biomes + high-frequency density variation
    wood_base = np.zeros(shape, dtype=np.float32)
    wood_base[world.biome == RAINFOREST] = 0.95
    wood_base[world.biome == TEMPERATE_FOREST] = 0.85
    wood_base[world.biome == TAIGA] = 0.70
    wood_base[world.biome == GRASSLAND] = 0.20
    wood_base[world.biome == TUNDRA] = 0.05
    
    wood_noise = fbm_noise_2d(shape, seed=world.seed + 404, octaves=3, lacunarity=2.0, gain=0.5, base_res=(10, 10))
    world.wood = np.clip(wood_base * (0.6 + 0.4 * wood_noise), 0.0, 1.0)
    
    # 3. Wildlife Resources
    # Proportional to wood cover and access to fresh water
    wildlife_base = np.zeros(shape, dtype=np.float32)
    wildlife_base[world.biome == RAINFOREST] = 0.90
    wildlife_base[world.biome == TEMPERATE_FOREST] = 0.80
    wildlife_base[world.biome == TAIGA] = 0.60
    wildlife_base[world.biome == GRASSLAND] = 0.50
    wildlife_base[world.biome == TUNDRA] = 0.20
    wildlife_base[world.biome == DESERT] = 0.05
    
    # Water drinking factor: high near water, dropping over distance
    water_factor = np.exp(-dist_to_water / 40.0)
    wildlife_pot = wildlife_base * (0.7 + 0.3 * water_factor)
    
    wildlife_noise = fbm_noise_2d(shape, seed=world.seed + 505, octaves=3, lacunarity=2.0, gain=0.5, base_res=(8, 8))
    world.wildlife = np.clip(wildlife_pot * (0.6 + 0.4 * wildlife_noise), 0.0, 1.0)
    
    # 4. Stone Resources
    # Abundant on steep slopes and high rocky mountain elevations
    stone_pot = world.elevation * 0.5 + slope * 3.0
    stone_noise = fbm_noise_2d(shape, seed=world.seed + 606, octaves=3, lacunarity=2.0, gain=0.5, base_res=(12, 12))
    world.stone = np.clip(stone_pot * (0.7 + 0.3 * stone_noise), 0.0, 1.0)
    
    # 5. Fertility (Soil Richness)
    # High in temperate valleys, river basins, and floodplains; zero in deserts, glaciers, oceans
    fert_base = np.zeros(shape, dtype=np.float32)
    fert_base[world.biome == TEMPERATE_FOREST] = 0.75
    fert_base[world.biome == GRASSLAND] = 0.60
    fert_base[world.biome == RAINFOREST] = 0.50  # Tropical soils are often nutrient-leached, moderate fertility
    fert_base[world.biome == TAIGA] = 0.25
    fert_base[world.biome == TUNDRA] = 0.05
    
    # Floodplain bonus: flat land close to rivers/lakes
    # Within 25 pixels of fresh water and slope is low (< 0.02)
    is_fresh_water = (world.biome == LAKE) | (world.lake_map > 0.0) | (world.river_map > 1500.0)
    dist_to_fresh = distance_transform_edt(~is_fresh_water)
    
    floodplain_mask = (dist_to_fresh < 25.0) & (slope < 0.02) & (world.elevation >= sea_level)
    fert_base[floodplain_mask] = np.minimum(1.0, fert_base[floodplain_mask] + 0.25)
    
    # Soil fertility noise variation
    fert_noise = fbm_noise_2d(shape, seed=world.seed + 707, octaves=3, lacunarity=2.0, gain=0.5, base_res=(6, 6))
    world.fertility = np.clip(fert_base * (0.8 + 0.2 * fert_noise), 0.0, 1.0)
    
    # 6. Clustered Metallic Minerals (Iron and Copper Belts)
    # Metals only spawn in high hills or mountains (elevation >= 0.45)
    is_mountainous = world.elevation >= 0.45
    mountain_richness = np.maximum(0.0, world.elevation - 0.4) / 0.6
    
    # Low-frequency noise masks to cluster resources into large "mineral belts"
    iron_belt_mask = fbm_noise_2d(shape, seed=world.seed + 808, octaves=2, base_res=(2, 2))
    copper_belt_mask = fbm_noise_2d(shape, seed=world.seed + 909, octaves=2, base_res=(2, 2))
    
    # High-frequency noise to add localized vein structures
    iron_vein_noise = fbm_noise_2d(shape, seed=world.seed + 818, octaves=4, base_res=(10, 10))
    copper_vein_noise = fbm_noise_2d(shape, seed=world.seed + 919, octaves=4, base_res=(10, 10))
    
    # Iron Belt: high belt noise (> 0.5)
    iron_cells = is_mountainous & (iron_belt_mask > 0.5)
    world.iron[iron_cells] = mountain_richness[iron_cells] * (0.4 + 0.6 * iron_vein_noise[iron_cells])
    
    # Copper Belt: high copper noise (> 0.5) and no overlapping iron belt (exclusive areas)
    copper_cells = is_mountainous & (copper_belt_mask > 0.5) & (~iron_cells)
    world.copper[copper_cells] = mountain_richness[copper_cells] * (0.4 + 0.6 * copper_vein_noise[copper_cells])
    
    # Ensure water bodies have no terrestrial resources
    is_water_tile = world.elevation < sea_level
    world.wood[is_water_tile] = 0.0
    world.stone[is_water_tile] = 0.0
    world.copper[is_water_tile] = 0.0
    world.iron[is_water_tile] = 0.0
    world.wildlife[is_water_tile] = 0.0
    world.fertility[is_water_tile] = 0.0
    
    return world
