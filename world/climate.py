import numpy as np
from scipy.ndimage import gaussian_filter
from .state import WorldState

def simulate_climate(
    world: WorldState,
    sea_level: float = 0.3,
    base_moisture: float = 0.6,
    temp_offset: float = 0.0,
    evaporation_rate_mult: float = 1.0,
    uplift_factor_mult: float = 1.0
) -> WorldState:
    """
    Simulates climate layers (temperature and rainfall) on top of the elevation map.
    
    1. Temperature: Calculated using latitude-based solar radiation and elevation cooling.
    2. Rainfall: Simulated via moisture-laden wind advection (West-to-East trade winds) 
       creating windward rainfall and leeward rain shadows.
       
    Parameters:
    - world: The WorldState object to modify.
    - sea_level: Elevation threshold below which tiles are considered ocean.
    - base_moisture: Base air humidity along the West edge.
    - temp_offset: Global temperature adjustment shift.
    - evaporation_rate_mult: Multiplier for moisture evaporation over ocean.
    - uplift_factor_mult: Multiplier for windward precipitation slope trigger.
    
    Returns:
    - The modified WorldState object.
    """
    h, w = world.height, world.width
    
    # --- 1. Temperature Simulation ---
    # Latitude factor: 1.0 at the equator (center of the grid), 0.0 at the poles (top/bottom)
    y_coords = np.arange(h).reshape(-1, 1)
    latitude_factor = 1.0 - np.abs(y_coords - (h / 2.0)) / (h / 2.0)
    latitude_factor = np.tile(latitude_factor, (1, w))
    
    # Base temperature ranges from -15°C at the poles to 32°C at the equator
    base_temp = -15.0 + 47.0 * latitude_factor
    
    # Elevation cooling: lapse rate of 38°C drop from sea level to highest peaks (elevation = 1.0)
    elevation_cooling = 38.0 * np.maximum(0.0, world.elevation)
    
    # Final temperature in Celsius (with optional global offset)
    world.temperature = base_temp - elevation_cooling + temp_offset
    
    # --- 2. Rainfall Simulation (Wind Moisture Advection) ---
    # Wind moves from West to East (left to right across columns)
    # Initialize air moisture columns along the West edge
    moisture = np.ones(h, dtype=np.float32) * base_moisture  # Parametrized base moisture
    raw_rainfall = np.zeros((h, w), dtype=np.float32)
    
    # Moisture simulation constants scaled by grid width for resolution independence
    evaporation_rate = (50.0 / w) * evaporation_rate_mult
    land_evap_rate = 8.0 / w
    max_moisture = 1.8
    base_precipitation = 0.6 / w
    uplift_factor = 35.0 * uplift_factor_mult  # Precipitation multiplier on uphill slopes
    
    prev_elevation = world.elevation[:, 0]
    
    for x in range(w):
        curr_elevation = world.elevation[:, x]
        
        # Over oceans (elevation < sea_level), water evaporates and recharges the air moisture
        is_ocean = curr_elevation < sea_level
        ocean_depth = np.maximum(0.0, sea_level - curr_elevation)
        moisture[is_ocean] = np.minimum(
            max_moisture, 
            moisture[is_ocean] + evaporation_rate * ocean_depth[is_ocean]
        )
        
        # Over land, simulate evapotranspiration (moisture recycling from soil/forests)
        is_land = ~is_ocean
        if np.any(is_land):
            moisture[is_land] = np.minimum(
                max_moisture,
                moisture[is_land] + land_evap_rate * (1.0 - curr_elevation[is_land])
            )
        
        # Calculate effective slope: wind blows over flat ocean surface at sea_level
        eff_curr = np.maximum(sea_level, curr_elevation)
        eff_prev = np.maximum(sea_level, prev_elevation)
        slope = np.maximum(0.0, eff_curr - eff_prev)
        
        # Precipitation occurs on slope uplift or as base rainfall on humid columns
        rain = moisture * (base_precipitation + uplift_factor * slope)
        
        # Clamp rain to current moisture to preserve mass-balance
        rain = np.minimum(moisture, rain)
        
        # Deplete moisture
        moisture -= rain
        
        # Store raw rainfall
        raw_rainfall[:, x] = rain
        
        prev_elevation = curr_elevation
        
    # Scale rainfall to physical units dynamically based on grid resolution
    rainfall_mm = raw_rainfall * (3500.0 * (w / 128.0))
    
    # Smooth rainfall map using a light Gaussian blur to eliminate horizontal scanlines and create organic clouds
    world.rainfall = gaussian_filter(rainfall_mm, sigma=6.0)
    
    return world
