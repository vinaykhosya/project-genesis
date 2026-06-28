import heapq
import numpy as np
from .state import WorldState, LAKE

def priority_flood_fill(elevation: np.ndarray) -> np.ndarray:
    """
    Applies the Priority-Flood algorithm to fill all terrain sinks/depressions.
    Ensures that every cell on the map has a continuous downhill path to the boundaries or oceans.
    
    Parameters:
    - elevation: 2D NumPy array of elevations.
    
    Returns:
    - A 2D NumPy array of filled elevations.
    """
    h, w = elevation.shape
    filled = np.copy(elevation)
    visited = np.zeros((h, w), dtype=bool)
    
    # Priority queue stores tuples of (elevation, flat_index)
    heap = []
    
    # Push all boundary pixels into the heap
    # Top & Bottom rows
    for x in range(w):
        for y in (0, h - 1):
            if not visited[y, x]:
                visited[y, x] = True
                heapq.heappush(heap, (elevation[y, x], y * w + x))
    # Left & Right columns
    for y in range(h):
        for x in (0, w - 1):
            if not visited[y, x]:
                visited[y, x] = True
                heapq.heappush(heap, (elevation[y, x], y * w + x))
                
    # Directions for 8-neighborhood
    dx = [-1, 1, 0, 0, -1, 1, -1, 1]
    dy = [0, 0, -1, 1, -1, -1, 1, 1]
    
    while heap:
        z, idx = heapq.heappop(heap)
        cy = idx // w
        cx = idx % w
        
        # Traverse 8 neighbors
        for i in range(8):
            nx = cx + dx[i]
            ny = cy + dy[i]
            
            if 0 <= nx < w and 0 <= ny < h:
                if not visited[ny, nx]:
                    visited[ny, nx] = True
                    # The fill level of the neighbor is the max of its own elevation and 
                    # the current cell's spillover level.
                    z_neigh = elevation[ny, nx]
                    z_filled = max(z_neigh, z)
                    filled[ny, nx] = z_filled
                    heapq.heappush(heap, (z_filled, ny * w + nx))
                    
    return filled

def generate_hydrology(world: WorldState, sea_level: float = 0.3) -> WorldState:
    """
    Simulates hydrology (rivers and lakes) by tracing rainfall accumulation downhill.
    Uses the Priority-Flood algorithm to resolve depressions and sinks.
    
    Parameters:
    - world: The WorldState container to modify.
    - sea_level: The sea level threshold.
    
    Returns:
    - The modified WorldState container.
    """
    h, w = world.height, world.width
    
    # 1. Fill terrain sinks
    filled = priority_flood_fill(world.elevation)
    
    # 2. Map flow directions (8-neighborhood)
    dy = [-1, 1, 0, 0, -1, -1, 1, 1]
    dx = [0, 0, -1, 1, -1, 1, -1, 1]
    dists = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
    
    # Calculate downhill slopes to neighbors
    slopes = []
    # Pad filled map to easily handle borders without bounds errors
    padded_filled = np.pad(filled, 1, mode='edge')
    
    for i in range(8):
        # Shifted neighbor heights
        neigh = padded_filled[1 + dy[i] : 1 + dy[i] + h, 1 + dx[i] : 1 + dx[i] + w]
        slope = (filled - neigh) / dists[i]
        slopes.append(slope)
        
    slopes_arr = np.stack(slopes, axis=0) # shape (8, h, w)
    flow_dir = np.argmax(slopes_arr, axis=0) # Index of neighbor of steepest descent
    
    # 3. Flow accumulation
    # Sort cells by filled elevation in descending order
    sorted_indices = np.argsort(filled.flat)[::-1]
    
    # Accumulation starts with rainfall weight
    # Base weight 1.0 + scaling rainfall in mm (e.g. 0 to 3000mm maps to 0 to 6.0)
    accumulation = 1.0 + world.rainfall * 0.002
    accumulation = accumulation.astype(np.float32)
    
    for idx in sorted_indices:
        cy = idx // w
        cx = idx % w
        
        # Oceans don't route flow downstream
        if filled.flat[idx] < sea_level:
            continue
            
        dir_idx = flow_dir.flat[idx]
        nx = cx + dx[dir_idx]
        ny = cy + dy[dir_idx]
        
        if 0 <= nx < w and 0 <= ny < h:
            next_idx = ny * w + nx
            accumulation.flat[next_idx] += accumulation.flat[idx]
            
    # 4. Save River Map
    # Store raw flow accumulation for sizing and render thickness
    world.river_map = accumulation
    
    # Binary masks for rivers and lakes
    is_land = world.elevation >= sea_level
    # River criteria: high flow accumulation on land
    is_river = is_land & (accumulation > 1500.0)
    
    # Lake criteria: depressions that were filled where significant water accumulates
    depression = filled - world.elevation
    is_lake = is_land & (depression > 0.008) & (accumulation > 300.0)
    
    # 5. Apply Lakes to Biome & Elevation Map
    # Lakes flood the terrain up to the filled elevation level (water level)
    world.elevation[is_lake] = filled[is_lake]
    world.biome[is_lake] = LAKE
    world.lake_map = is_lake.astype(np.float32)
    
    # Rivers are also marked on the biome map where they are very wide
    is_major_river = is_land & (accumulation > 4000.0)
    world.biome[is_major_river] = LAKE  # Mark large rivers as water bodies on the biome map
    
    return world
