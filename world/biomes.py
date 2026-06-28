import numpy as np
from .state import (
    WorldState, OCEAN, GLACIER, TUNDRA, TAIGA, 
    TEMPERATE_FOREST, GRASSLAND, DESERT, RAINFOREST
)

# A curated aesthetic color palette (RGB) for each biome
BIOME_COLORS = {
    OCEAN: (15, 37, 65),             # Premium deep navy blue
    GLACIER: (225, 238, 245),        # Frozen ice white
    TUNDRA: (150, 155, 140),         # Cold muted olive gray
    TAIGA: (34, 73, 56),             # Rich boreal dark pine green
    TEMPERATE_FOREST: (46, 117, 72), # Warm vibrant deciduous green
    GRASSLAND: (172, 184, 106),      # Earthy golden grassland
    DESERT: (224, 187, 115),         # Warm desert sand yellow
    RAINFOREST: (12, 85, 45)         # Intense deep tropical jungle green
}

def generate_biomes(world: WorldState, sea_level: float = 0.3) -> WorldState:
    """
    Categorizes each grid cell into a specific biome based on its elevation, 
    temperature (Celsius), and annual rainfall (mm).
    
    Parameters:
    - world: The WorldState object to modify.
    - sea_level: The elevation threshold for the ocean.
    
    Returns:
    - The modified WorldState object.
    """
    elevation = world.elevation
    temperature = world.temperature
    rainfall = world.rainfall
    
    # Initialize the biome grid with a fallback (Grassland)
    biomes = np.full(elevation.shape, GRASSLAND, dtype=np.int32)
    
    # --- 1. Ocean Classification ---
    is_ocean = elevation < sea_level
    biomes[is_ocean] = OCEAN
    
    # Land mask for remaining steps
    is_land = ~is_ocean
    
    # --- 2. Glacier & Polar Regions (MAT < -8°C) ---
    is_glacier = is_land & (temperature < -8.0)
    biomes[is_glacier] = GLACIER
    
    # --- 3. Tundra (MAT -8°C to 0°C) ---
    is_tundra = is_land & (temperature >= -8.0) & (temperature < 0.0)
    biomes[is_tundra] = TUNDRA
    
    # --- 4. Subpolar Zone (MAT 0°C to 8°C) ---
    is_subpolar = is_land & (temperature >= 0.0) & (temperature < 8.0)
    # Wet subpolar areas are Taiga (Boreal Forest)
    is_taiga = is_subpolar & (rainfall >= 400.0)
    biomes[is_taiga] = TAIGA
    # Dry subpolar areas degrade to Tundra
    is_cold_dry = is_subpolar & (rainfall < 400.0)
    biomes[is_cold_dry] = TUNDRA
    
    # --- 5. Temperate Zone (MAT 8°C to 20°C) ---
    is_temperate = is_land & (temperature >= 8.0) & (temperature < 20.0)
    # Wet temperate areas are Temperate Deciduous Forests
    is_temp_forest = is_temperate & (rainfall >= 1000.0)
    biomes[is_temp_forest] = TEMPERATE_FOREST
    # Moderate rainfall temperate areas are Grasslands/Savannas
    is_temp_grass = is_temperate & (rainfall >= 250.0) & (rainfall < 1000.0)
    biomes[is_temp_grass] = GRASSLAND
    # Dry temperate areas are Temperate Deserts
    is_temp_desert = is_temperate & (rainfall < 250.0)
    biomes[is_temp_desert] = DESERT
    
    # --- 6. Tropical/Subtropical Zone (MAT >= 20°C) ---
    is_tropical = is_land & (temperature >= 20.0)
    # Wet tropical areas are Rainforests
    is_rainforest = is_tropical & (rainfall >= 1600.0)
    biomes[is_rainforest] = RAINFOREST
    # Dry tropical areas are Hot Deserts
    is_hot_desert = is_tropical & (rainfall < 450.0)
    biomes[is_hot_desert] = DESERT
    # Intermediate zones are Grasslands (Savannas)
    is_savanna = is_tropical & (rainfall >= 450.0) & (rainfall < 1600.0)
    biomes[is_savanna] = GRASSLAND
    
    world.biome = biomes
    return world

def biomes_to_rgb(biome_grid: np.ndarray) -> np.ndarray:
    """
    Converts a 2D grid of biome integer IDs into an RGB image grid.
    
    Parameters:
    - biome_grid: 2D NumPy array of biome IDs.
    
    Returns:
    - 3D NumPy uint8 array of shape (height, width, 3).
    """
    h, w = biome_grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for biome_id, color in BIOME_COLORS.items():
        mask = biome_grid == biome_id
        rgb[mask] = color
        
    return rgb
