import numpy as np
from .state import WorldState

def simulate_erosion(world: WorldState, iterations: int = 25, talus_slope: float = 0.005, rate: float = 0.1) -> WorldState:
    """
    Applies thermal erosion (soil creep) to the world's elevation map.
    Moves material from higher tiles to lower neighbors where the slope exceeds 
    the angle of repose (talus_slope).
    
    Parameters:
    - world: The WorldState object to modify.
    - iterations: Number of erosion simulation steps.
    - talus_slope: Minimum elevation difference threshold for soil movement.
    - rate: Percentage of excess material moved downhill per step.
    
    Returns:
    - The modified WorldState object.
    """
    E = world.elevation.copy()
    
    for _ in range(iterations):
        # Create copies of the grid shifted by 1 pixel in each direction.
        # Boundary pixels are clamped to themselves (non-wrapping border condition).
        E_up = np.zeros_like(E)
        E_up[1:, :] = E[:-1, :]
        E_up[0, :] = E[0, :]
        
        E_down = np.zeros_like(E)
        E_down[:-1, :] = E[1:, :]
        E_down[-1, :] = E[-1, :]
        
        E_left = np.zeros_like(E)
        E_left[:, 1:] = E[:, :-1]
        E_left[:, 0] = E[:, 0]
        
        E_right = np.zeros_like(E)
        E_right[:, :-1] = E[:, 1:]
        E_right[:, -1] = E[:, -1]
        
        # Calculate slope height differences
        d_up = E - E_up
        d_down = E - E_down
        d_left = E - E_left
        d_right = E - E_right
        
        # Compute how much height exceeds the talus slope (only move downhill: diff > talus_slope)
        m_up = np.maximum(0.0, d_up - talus_slope)
        m_down = np.maximum(0.0, d_down - talus_slope)
        m_left = np.maximum(0.0, d_left - talus_slope)
        m_right = np.maximum(0.0, d_right - talus_slope)
        
        # Total potential outflow of sediment from each cell
        total_move = m_up + m_down + m_left + m_right
        
        # Scale outflow by the rate
        out_up = m_up * rate
        out_down = m_down * rate
        out_left = m_left * rate
        out_right = m_right * rate
        
        # Compute inflow received from neighbors (opposite direction of their outflows)
        in_up = np.zeros_like(E)
        in_up[1:, :] = out_down[:-1, :]
        
        in_down = np.zeros_like(E)
        in_down[:-1, :] = out_up[1:, :]
        
        in_left = np.zeros_like(E)
        in_left[:, 1:] = out_right[:, :-1]
        
        in_right = np.zeros_like(E)
        in_right[:, :-1] = out_left[:, 1:]
        
        # Apply mass-conservation: new elevation = elevation - outflow + inflow
        E = E - (out_up + out_down + out_left + out_right) + (in_up + in_down + in_left + in_right)
        
    world.elevation = E
    return world
