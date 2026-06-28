import numpy as np

# Biome Constants
OCEAN = 0
GLACIER = 1
TUNDRA = 2
TAIGA = 3
TEMPERATE_FOREST = 4
GRASSLAND = 5
DESERT = 6
RAINFOREST = 7
LAKE = 8

BIOME_NAMES = {
    OCEAN: "Ocean",
    GLACIER: "Glacier/Ice",
    TUNDRA: "Tundra",
    TAIGA: "Taiga (Boreal Forest)",
    TEMPERATE_FOREST: "Temperate Forest",
    GRASSLAND: "Grassland/Savanna",
    DESERT: "Desert",
    RAINFOREST: "Tropical Rainforest",
    LAKE: "Lake"
}

class WorldState:
    """
    Unified state container representing all layers of the simulated world.
    Each physical layer is stored as a 2D NumPy array for fast vectorized operations.
    """
    def __init__(self, width: int, height: int, seed: int):
        self.width = width
        self.height = height
        self.seed = seed
        
        # Grid layers (y, x) shape
        self.elevation = np.zeros((height, width), dtype=np.float32)
        self.temperature = np.zeros((height, width), dtype=np.float32)
        self.rainfall = np.zeros((height, width), dtype=np.float32)
        self.biome = np.zeros((height, width), dtype=np.int32)
        
        # Hydrology Layers
        self.river_map = np.zeros((height, width), dtype=np.float32)
        self.lake_map = np.zeros((height, width), dtype=np.float32)
        
        # Resource Layers
        self.wood = np.zeros((height, width), dtype=np.float32)
        self.stone = np.zeros((height, width), dtype=np.float32)
        self.copper = np.zeros((height, width), dtype=np.float32)
        self.iron = np.zeros((height, width), dtype=np.float32)
        self.wildlife = np.zeros((height, width), dtype=np.float32)
        self.fertility = np.zeros((height, width), dtype=np.float32)
        
        # Habitability Split Scores
        self.water_score = np.zeros((height, width), dtype=np.float32)
        self.food_score = np.zeros((height, width), dtype=np.float32)
        self.resource_score = np.zeros((height, width), dtype=np.float32)
        self.climate_score = np.zeros((height, width), dtype=np.float32)
        self.terrain_score = np.zeros((height, width), dtype=np.float32)
        self.habitability = np.zeros((height, width), dtype=np.float32)
        
        # Passability & Trade Layers
        self.movement_cost = np.zeros((height, width), dtype=np.float32)
        self.trade_potential = np.zeros((height, width), dtype=np.float32)
        
        # Agent & Simulation State (Phase 2)
        self.agents = []
        self.history = []
        self.tick = 0
        self.death_density_map = np.zeros((height, width), dtype=np.float32)
        self.events_timeline = []

        # Evolution & Multi-Generation Tracking (Phase 5)
        # colonies: list of dicts — {id, name, color, founder_ids, stored_food, stored_water}
        self.colonies = []
        # generation_number: highest generation seen in the simulation so far
        self.generation_number = 0
        self.total_births = 0
        self.total_deaths = 0
        # extinction_events: list of {tick, colony_id, colony_name}
        self.extinction_events = []
        # population_history: list of {tick, total, per_colony: [n0, n1, n2, n3]}
        # Sampled every sample_interval ticks by simulation.py
        self.population_history = []
        # genetic_history: list of {tick, gene_means, gene_variances, diversity_score}
        self.genetic_history = []
        # generation_stats: list of {generation, births, deaths, avg_fitness, genetic_diversity}
        self.generation_stats = {}  # keyed by generation number
        # Soft carrying-capacity cap on alive population (tunable per experiment)
        self.max_population = 64
        # Monotonically increasing ID counter for newly spawned agents
        self.next_agent_id = 0
        self.query_agents_calls = 0
        self.query_agents_returned_sum = 0
        
        # Centralized Telemetry Collector (Phase 10)
        from .telemetry import TelemetryCollector
        self.telemetry = TelemetryCollector(self)

    def query_agents(self, center: tuple, radius: float, alive_only: bool = True) -> list:
        """
        Queries agents within a Euclidean distance from a center coordinate.
        Uses the spatial hash grid when available for O(1) performance in local regions.
        """
        self.query_agents_calls += 1
        cy, cx = center
        
        if hasattr(self, "spatial_grid") and self.spatial_grid is not None:
            bin_size = 32
            grid_h = len(self.spatial_grid)
            grid_w = len(self.spatial_grid[0])
            
            bx_min = max(0, int(cx - radius) // bin_size)
            bx_max = min(grid_w - 1, int(cx + radius) // bin_size)
            by_min = max(0, int(cy - radius) // bin_size)
            by_max = min(grid_h - 1, int(cy + radius) // bin_size)
            
            candidates = []
            for by in range(by_min, by_max + 1):
                for bx in range(bx_min, bx_max + 1):
                    candidates.extend(self.spatial_grid[by][bx])
        else:
            candidates = getattr(self, "alive_agents", self.agents) if alive_only else self.agents
            
        results = []
        radius_sq = radius ** 2
        for other in candidates:
            if alive_only and other.dead:
                continue
            oy, ox = other.location
            dist_sq = (oy - cy) ** 2 + (ox - cx) ** 2
            if dist_sq <= radius_sq:
                results.append(other)
        
        self.query_agents_returned_sum += len(results)
        return results

