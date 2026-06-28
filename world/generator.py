import numpy as np
from .state import WorldState
from .noise import fbm_noise_2d, generate_perlin_noise_2d
from .erosion import simulate_erosion
from .climate import simulate_climate
from .biomes import generate_biomes
from .hydrology import generate_hydrology
from .resources import generate_resources
from .habitability import calculate_habitability
from .passability import calculate_passability

from .world_presets import WORLD_PRESETS

def generate_continents(
    world: WorldState,
    radial_mask_power: float = 1.5,
    use_ring_gradient: bool = False
) -> WorldState:
    """
    Step 1: Generates base continental terrain using low-frequency fBm noise
    and applies a radial gradient mask to push the grid borders into the ocean.
    """
    shape = (world.height, world.width)
    
    # Base continental noise (broad shapes)
    continental_noise = fbm_noise_2d(
        shape, seed=world.seed, octaves=6, lacunarity=2.0, gain=0.5, base_res=(3, 3)
    )
    
    # Generate radial mask to center the landmass and create surrounding ocean
    y, x = np.ogrid[:world.height, :world.width]
    center_y, center_x = world.height / 2.0, world.width / 2.0
    
    # Normalized distance to center [0, 1]
    dist_to_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    norm_dist = dist_to_center / max_dist
    
    if use_ring_gradient:
        # A ring gradient: peaks at 0.5 distance from center, fading to 0 at center and edges
        radial_mask = np.sin(np.clip(norm_dist, 0.0, 1.0) * np.pi)
        radial_mask = radial_mask ** radial_mask_power
    else:
        # Cosine falloff mask: 1.0 at center, fading smoothly to 0.0 at borders
        radial_mask = np.cos(np.clip(norm_dist, 0.0, 1.0) * np.pi / 2.0)
        # Squaring the mask makes the ocean borders wider and coasts more interesting
        radial_mask = radial_mask ** radial_mask_power
    
    world.elevation = continental_noise * radial_mask
    return world

def generate_mountains(world: WorldState, mountain_scale: float = 1.0) -> WorldState:
    """
    Step 2: Generates linear mountain ranges using ridge noise restricted
    by a secondary mountain mask, then adds detail noise.
    """
    shape = (world.height, world.width)
    
    # Mountain mask: controls where mountain ranges can spawn (needs to feel linear/clustered)
    mountain_mask_noise = fbm_noise_2d(
        shape, seed=world.seed + 101, octaves=4, lacunarity=2.0, gain=0.4, base_res=(4, 4)
    )
    # Threshold the noise to select only high-value clusters as mountains
    mountain_mask = np.maximum(0.0, mountain_mask_noise - 0.45) / 0.55
    # Smooth the mask boundary
    mountain_mask = mountain_mask ** 2
    
    # Generate Ridge Noise (sharp linear peaks)
    # Ridge Noise formula: 1.0 - abs(2 * Perlin - 1)^2
    ridge_noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    max_amp = 0.0
    
    rng = np.random.default_rng(world.seed + 202)
    octave_seeds = rng.integers(0, 100000, size=4)
    
    for i in range(4):
        res_y = max(2, min(int(5 * frequency), world.height - 1))
        res_x = max(2, min(int(5 * frequency), world.width - 1))
        
        # Clamp resolutions to reasonable limits
        res_y = max(2, min(res_y, world.height - 1))
        res_x = max(2, min(res_x, world.width - 1))
        
        noise = generate_perlin_noise_2d(shape, (res_y, res_x), seed=octave_seeds[i])
        
        # Apply ridge formula
        r_octave = 1.0 - np.abs(2.0 * noise - 1.0)
        r_octave = r_octave ** 2  # Sharpen the ridges
        
        ridge_noise += r_octave * amplitude
        max_amp += amplitude
        
        amplitude *= 0.5
        frequency *= 2.0
        
    ridge_noise /= max_amp
    
    # Detail noise (high-frequency sand/rocks texture)
    detail_noise = fbm_noise_2d(
        shape, seed=world.seed + 303, octaves=3, lacunarity=2.0, gain=0.5, base_res=(16, 16)
    )
    
    # Blend them: continents + ridges * mask * scale + minor details
    mountain_contrib = ridge_noise * mountain_mask * 0.4 * mountain_scale
    world.elevation = world.elevation + mountain_contrib + (detail_noise * 0.02)
    
    # Clamp to physical range
    world.elevation = np.clip(world.elevation, 0.0, 1.0)
    
    return world

def generate_world(
    width: int,
    height: int,
    seed: int,
    sea_level: float = 0.3,
    world_preset: str = None
) -> WorldState:
    """
    Coordinates the Layer 1 & 1.5 World Generation pipeline:
    Continents -> Mountains -> Erosion -> Climate -> Biomes -> Hydrology -> Resources -> Habitability -> Passability
    """
    world = WorldState(width=width, height=height, seed=seed)
    
    # Defaults
    radial_mask_power = 1.5
    use_ring_gradient = False
    mountain_scale = 1.0
    base_moisture = 0.6
    temp_offset = 0.0
    evaporation_rate_mult = 1.0
    uplift_factor_mult = 1.0
    
    # Load preset parameters if specified
    if world_preset and world_preset in WORLD_PRESETS:
        preset = WORLD_PRESETS[world_preset]
        sea_level = preset.get("sea_level", sea_level)
        base_moisture = preset.get("base_moisture", base_moisture)
        temp_offset = preset.get("temp_offset", temp_offset)
        mountain_scale = preset.get("mountain_scale", mountain_scale)
        evaporation_rate_mult = preset.get("evaporation_rate_mult", evaporation_rate_mult)
        uplift_factor_mult = preset.get("uplift_factor_mult", uplift_factor_mult)
        radial_mask_power = preset.get("radial_mask_power", radial_mask_power)
        use_ring_gradient = preset.get("use_ring_gradient", use_ring_gradient)
        print(f"🌍 Using World Preset: '{preset['name']}' (Sea Level={sea_level}, Base Moisture={base_moisture}, Temp Offset={temp_offset}°C)")
        
    world.spawn_preset = world_preset
    
    # 1. Continents
    world = generate_continents(world, radial_mask_power=radial_mask_power, use_ring_gradient=use_ring_gradient)
    
    # 2. Mountains
    world = generate_mountains(world, mountain_scale=mountain_scale)
    
    # 3. Erosion
    world = simulate_erosion(world, iterations=30, talus_slope=0.005, rate=0.1)
    
    # 4. Climate (Temperature and Rainfall)
    world = simulate_climate(
        world,
        sea_level=sea_level,
        base_moisture=base_moisture,
        temp_offset=temp_offset,
        evaporation_rate_mult=evaporation_rate_mult,
        uplift_factor_mult=uplift_factor_mult
    )
    
    # 5. Biomes
    world = generate_biomes(world, sea_level=sea_level)
    
    # 6. Hydrology (Rivers & Lakes)
    world = generate_hydrology(world, sea_level=sea_level)
    
    # 7. Resources (Fertility, Wood, Stone, Metals, Wildlife)
    world = generate_resources(world, sea_level=sea_level)
    
    # 8. Habitability Index
    world = calculate_habitability(world, sea_level=sea_level)
    
    # 9. Movement Cost & Trade Potential
    world = calculate_passability(world, sea_level=sea_level)
    
    return world
