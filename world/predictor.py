import numpy as np
from .state import WorldState, BIOME_NAMES

def predict_settlements(world: WorldState, count: int = 10, exclusion_radius: float = 40.0) -> list:
    """
    Predicts the top N settlement locations on the map using Non-Maximum Suppression (NMS).
    Once a location of maximum habitability is selected, adjacent locations within 
    'exclusion_radius' are masked out to ensure settlements are spread logically.
    
    Parameters:
    - world: The WorldState container.
    - count: The number of top locations to return.
    - exclusion_radius: Radius in grid pixels to suppress around chosen centers.
    
    Returns:
    - A list of dicts, each representing a settlement:
      {"rank": int, "x": int, "y": int, "score": float, "biome": str}
    """
    h, w = world.height, world.width
    # Make a copy of the habitability array to mutate
    temp_hab = np.copy(world.habitability)
    
    predictions = []
    
    y_coords, x_coords = np.ogrid[:h, :w]
    
    for i in range(count):
        # Find global maximum in the remaining habitability map
        max_idx = np.argmax(temp_hab)
        max_y = max_idx // w
        max_x = max_idx % w
        max_score = temp_hab[max_y, max_x]
        
        # If the highest score is 0, no more habitable locations exist
        if max_score <= 0.0:
            break
            
        biome_id = world.biome[max_y, max_x]
        biome_name = BIOME_NAMES.get(biome_id, "Unknown")
        
        predictions.append({
            "rank": i + 1,
            "x": int(max_x),
            "y": int(max_y),
            "score": float(max_score),
            "biome": biome_name
        })
        
        # Suppress/mask out neighbors within circular exclusion radius
        dist_sq = (y_coords - max_y)**2 + (x_coords - max_x)**2
        suppress_mask = dist_sq < exclusion_radius**2
        temp_hab[suppress_mask] = 0.0
        
    return predictions
