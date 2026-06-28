import numpy as np

def generate_perlin_noise_2d(shape, res, seed=1729):
    """
    Generates a 2D Perlin noise grid.
    
    Parameters:
    - shape: Tuple of (height, width) for the output grid.
    - res: Tuple of (res_y, res_x) representing the frequency of the noise grid.
    - seed: Integer random seed for generating gradients.
    
    Returns:
    - A 2D NumPy array of shape 'shape' with values normalized between 0 and 1.
    """
    h, w = shape
    res_y, res_x = res
    
    # Coordinate grids normalized to the resolution grid
    y = np.arange(h) * (res_y / h)
    x = np.arange(w) * (res_x / w)
    
    # Integer and fractional parts
    iy = y.astype(int)
    ix = x.astype(int)
    
    fy = y - iy
    fx = x - ix
    
    # Create coordinate grid for bilinear interpolation
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing='ij')
    
    # Wrap integer coordinates to stay within resolution bounds
    iy = iy % res_y
    ix = ix % res_x
    
    # Generate deterministic random gradients for each grid cell corner
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, (res_y, res_x))
    grads = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    
    # Index the gradients for each corner of the grid cell
    g00 = grads[iy[:, None], ix]
    g10 = grads[(iy[:, None] + 1) % res_y, ix]
    g01 = grads[iy[:, None], (ix + 1) % res_x]
    g11 = grads[(iy[:, None] + 1) % res_y, (ix + 1) % res_x]
    
    # Compute relative vectors from cell corners to pixel coordinates
    d00 = np.stack([fy_grid - 0, fx_grid - 0], axis=-1)
    d10 = np.stack([fy_grid - 1, fx_grid - 0], axis=-1)
    d01 = np.stack([fy_grid - 0, fx_grid - 1], axis=-1)
    d11 = np.stack([fy_grid - 1, fx_grid - 1], axis=-1)
    
    # Dot products between gradients and distance vectors
    n00 = np.sum(d00 * g00, axis=-1)
    n10 = np.sum(d10 * g10, axis=-1)
    n01 = np.sum(d01 * g01, axis=-1)
    n11 = np.sum(d11 * g11, axis=-1)
    
    # Fade function: 6t^5 - 15t^4 + 10t^3 for smooth transitions
    def fade(t):
        return t**3 * (t * (t * 6 - 15) + 10)
        
    t_y = fade(fy_grid)
    t_x = fade(fx_grid)
    
    # Interpolation
    nx0 = n00 * (1 - t_x) + n01 * t_x
    nx1 = n10 * (1 - t_x) + n11 * t_x
    n = nx0 * (1 - t_y) + nx1 * t_y
    
    # Normalize to [0, 1]. Original Perlin range is roughly [-0.707, 0.707]
    return np.clip((n + 0.707) / 1.414, 0.0, 1.0)

def fbm_noise_2d(shape, seed=1729, octaves=6, lacunarity=2.0, gain=0.5, base_res=(4, 4)):
    """
    Generates fractional Brownian motion (fBm) noise.
    
    Parameters:
    - shape: Tuple of (height, width) for the output grid.
    - seed: Integer base seed.
    - octaves: Number of noise layers.
    - lacunarity: Frequency multiplier per octave.
    - gain: Amplitude multiplier per octave.
    - base_res: Tuple of starting (res_y, res_x) frequency.
    
    Returns:
    - A 2D NumPy array of shape 'shape' with values in [0, 1].
    """
    h, w = shape
    total_noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    # Derive deterministic seeds for each octave
    rng = np.random.default_rng(seed)
    octave_seeds = rng.integers(0, 100000, size=octaves)
    
    for i in range(octaves):
        res_y = int(base_res[0] * frequency)
        res_x = int(base_res[1] * frequency)
        
        # Ensure frequency fits map boundaries and is at least 2x2
        res_y = max(2, min(res_y, h - 1))
        res_x = max(2, min(res_x, w - 1))
        
        noise = generate_perlin_noise_2d(shape, (res_y, res_x), seed=octave_seeds[i])
        
        total_noise += noise * amplitude
        max_value += amplitude
        
        amplitude *= gain
        frequency *= lacunarity
        
    return total_noise / max_value
