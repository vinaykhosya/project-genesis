"""
world_presets.py — Phase 9B: Named World Presets

Defines climate, terrain, and hydrologic configuration parameters for named presets.
These parameters scale the physical parameters in world generation to dynamically
alter the resulting biome distribution.
"""

from typing import Dict, Any

WORLD_PRESETS: Dict[str, Dict[str, Any]] = {
    "arid_continent": {
        "name": "Arid Continent",
        "description": "A dry landmass dominated by vast deserts (40%), temperate forests (25%), mountains (15%), grasslands (10%), and fertile wetlands/rivers (10%).",
        "sea_level": 0.25,
        "base_moisture": 0.22,
        "temp_offset": 4.0,
        "mountain_scale": 1.2,
        "evaporation_rate_mult": 0.6,
        "uplift_factor_mult": 1.2,
        "radial_mask_power": 1.3,
    },
    "green_continent": {
        "name": "Green Continent",
        "description": "A lush, humid continent with broad deciduous forests (40%), grassy savannas (30%), boreal taiga (15%), cold tundra (10%), and small dry desert pockets (5%).",
        "sea_level": 0.28,
        "base_moisture": 0.75,
        "temp_offset": -1.0,
        "mountain_scale": 0.9,
        "evaporation_rate_mult": 1.2,
        "uplift_factor_mult": 0.8,
        "radial_mask_power": 1.5,
    },
    "island_chains": {
        "name": "Island Chains",
        "description": "Deep ocean covering 55% of the world, with scattered temperate forest (25%), grassland (15%), and dry sandy beach (5%) islands.",
        "sea_level": 0.48,
        "base_moisture": 0.70,
        "temp_offset": 2.0,
        "mountain_scale": 0.7,
        "evaporation_rate_mult": 1.3,
        "uplift_factor_mult": 0.7,
        "radial_mask_power": 1.8,
    },
    "boreal_highlands": {
        "name": "Boreal Highlands",
        "description": "A cold, high-altitude northern region covered in dark taiga pine (40%), cold tundra (25%), high rocky glacier ranges (20%), and scrubby grassland (15%).",
        "sea_level": 0.26,
        "base_moisture": 0.45,
        "temp_offset": -12.0,
        "mountain_scale": 1.6,
        "evaporation_rate_mult": 0.7,
        "uplift_factor_mult": 1.5,
        "radial_mask_power": 1.2,
    },
    "tropical_ring": {
        "name": "Tropical Ring",
        "description": "A hot tropical ring-shaped archipelago dominated by dense rainforests (50%), open savannas (25%), dry scrublands (15%), and coastal wetlands (10%).",
        "sea_level": 0.32,
        "base_moisture": 0.90,
        "temp_offset": 8.0,
        "mountain_scale": 0.7,
        "evaporation_rate_mult": 1.4,
        "uplift_factor_mult": 0.9,
        "radial_mask_power": 1.4,
        "use_ring_gradient": True,
    }
}
