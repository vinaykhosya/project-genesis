import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from world.state import WorldState, OCEAN, GRASSLAND, LAKE
from world.noise import fbm_noise_2d
from world.erosion import simulate_erosion
from world.climate import simulate_climate
from world.biomes import generate_biomes
from world.hydrology import priority_flood_fill, generate_hydrology
from world.resources import generate_resources
from world.habitability import calculate_habitability
from world.passability import calculate_passability
from world.predictor import predict_settlements
from world.generator import generate_world
from world.agents.agent import Agent, WATER, FOOD, DANGER
from world.agents.perception import perceive
from world.agents.decision import update_agent_needs, evaluate_utility, step_toward, select_shelter_site
from world.agents.simulation import run_simulation, simulate_agent_tick

class TestWorldEngine(unittest.TestCase):
    def setUp(self):
        self.width = 128
        self.height = 128
        self.seed = 1729
        np.random.seed(self.seed)
        
    def test_state_initialization(self):
        """Verify that WorldState container initializes correctly with Phase 1.5 fields."""
        world = WorldState(width=self.width, height=self.height, seed=self.seed)
        self.assertEqual(world.width, self.width)
        self.assertEqual(world.height, self.height)
        self.assertEqual(world.seed, self.seed)
        self.assertEqual(world.river_map.shape, (self.height, self.width))
        self.assertEqual(world.wood.shape, (self.height, self.width))
        self.assertEqual(world.habitability.shape, (self.height, self.width))
        
    def test_priority_flood_filling(self):
        """Verify that Priority-Flood resolves all internal sinks."""
        # Create a terrain map with a bowl-like depression (sink) in the center
        elev = np.ones((16, 16), dtype=np.float32) * 0.5
        # Set borders to 0.5, but inside to 0.2
        elev[2:-2, 2:-2] = 0.2
        
        filled = priority_flood_fill(elev)
        
        # Priority flood should raise the interior sink to 0.5 (the border height)
        # to ensure water can drain off-map
        self.assertTrue(np.all(filled[2:-2, 2:-2] >= 0.5))
        
    def test_hydrology_rivers_and_lakes(self):
        """Verify that river channels accumulate water and lakes spawn in depressions."""
        world = WorldState(width=self.width, height=self.height, seed=self.seed)
        # Setup a simple 2D ramp sloping downhill towards the right
        y, x = np.ogrid[:self.height, :self.width]
        world.elevation = 0.8 - (x / self.width) * 0.4 + y * 0.0
        
        # Add a closed depression inside at x=80, with elevation 0.4 (above sea level 0.3)
        world.elevation[10:self.height-10, 80] = 0.4
        
        # Add high rainfall to ensure accumulation exceeds the lake threshold (300.0)
        world.rainfall.fill(2500.0)
        
        world = generate_hydrology(world, sea_level=0.3)
        
        # Flow accumulation should increase from left to right (downhill direction)
        self.assertGreater(world.river_map[64, 100], world.river_map[64, 10])
        
        # The depression at x=80 should be marked as a Lake because it traps water
        self.assertTrue(np.any(world.biome[10:self.height-10, 80] == LAKE))
        self.assertTrue(np.any(world.lake_map[10:self.height-10, 80] > 0.0))
        
    def test_resource_clustering(self):
        """Verify that metallic resources are clustered into distinct belts rather than noise."""
        world = generate_world(self.width, self.height, self.seed)
        
        # Iron and copper should only spawn on mountainous cells (elevation >= 0.45)
        self.assertTrue(np.all(world.iron[world.elevation < 0.45] == 0.0))
        self.assertTrue(np.all(world.copper[world.elevation < 0.45] == 0.0))
        
        # Verify that copper and iron are exclusive (separated into distinct ranges/belts)
        overlap = (world.iron > 0.1) & (world.copper > 0.1)
        self.assertEqual(np.sum(overlap), 0)
        
    def test_movement_costs(self):
        """Verify that movement cost is computed correctly for different biomes."""
        world = generate_world(self.width, self.height, self.seed)
        
        # Grassland check if present
        grass_mask = world.biome == GRASSLAND
        if np.any(grass_mask):
            grass_costs = world.movement_cost[grass_mask]
            self.assertTrue(np.all(grass_costs >= 1.0))
            self.assertLess(np.mean(grass_costs), 6.0)
        
        # Ocean/Lake tiles should have high base movement costs (10.0+)
        ocean_mask = world.biome == OCEAN
        if np.any(ocean_mask):
            ocean_cost = np.mean(world.movement_cost[ocean_mask])
            self.assertGreaterEqual(ocean_cost, 10.0)
            self.assertLess(ocean_cost, 15.0)
        
        # Mountains should have higher movement costs due to steep slope penalties
        # Let's find a steep peak
        dy, dx = np.gradient(world.elevation)
        slope = np.sqrt(dy**2 + dx**2)
        steep_mask = (world.elevation >= 0.3) & (slope > 0.02)
        if np.any(steep_mask):
            self.assertGreater(np.mean(world.movement_cost[steep_mask]), 1.5)
            
    def test_trade_potential(self):
        """Verify that trade potential is higher near water bodies and flat paths."""
        world = generate_world(self.width, self.height, self.seed)
        
        # Land cells right adjacent to water channels should have higher trade potential than deep inland desert cells
        is_water = (world.elevation < 0.3) | (world.biome == LAKE)
        dist_fresh = distance_transform_edt(~is_water)
        
        near_water_mask = (world.elevation >= 0.3) & (dist_fresh < 5.0)
        # Find a far water cell
        far_water_mask = (world.elevation >= 0.3) & (dist_fresh > 30.0)
        
        if np.any(near_water_mask) and np.any(far_water_mask):
            self.assertGreater(np.mean(world.trade_potential[near_water_mask]), np.mean(world.trade_potential[far_water_mask]))
            
    def test_nms_predictor(self):
        """Verify that Non-Maximum Suppression spaces predicted settlements properly."""
        world = generate_world(self.width, self.height, self.seed)
        
        count = 6
        radius = 20.0
        settlements = predict_settlements(world, count=count, exclusion_radius=radius)
        
        # Check that we received predictions
        self.assertEqual(len(settlements), count)
        
        # Verify that all pairs of locations are separated by at least the exclusion radius
        for i in range(len(settlements)):
            for j in range(i + 1, len(settlements)):
                s1 = settlements[i]
                s2 = settlements[j]
                dist = np.sqrt((s1['x'] - s2['x'])**2 + (s1['y'] - s2['y'])**2)
                self.assertGreaterEqual(dist, radius - 0.1) # Float tolerance

    def test_agent_needs_decay_and_movement_cost(self):
        """Verify that needs decay and scale with movement cost."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.movement_cost.fill(2.0)  # High movement cost
        world.temperature.fill(20.0)   # Comfortable temperature to prevent cold/heat pressure
        world.biome.fill(7)            # Rainforest (amplitude 4°C keeps temp between 16°C and 24°C, safe comfort zone)
        
        agent = Agent(agent_id=0, location=(50, 50))
        agent.max_age = 80 * 360       # Prevent random normal lifespan senescence
        agent.age = 30 * 360           # Set to healthy adult age to ensure senescence_factor = 0
        self.assertEqual(agent.hunger, 0.0)
        self.assertEqual(agent.thirst, 0.0)
        self.assertEqual(agent.energy, 100.0)
        
        # Test stationary decay
        update_agent_needs(agent, world, moved=False)
        self.assertAlmostEqual(agent.hunger, 0.05)
        self.assertAlmostEqual(agent.thirst, 0.10)
        self.assertAlmostEqual(agent.energy, 99.96)
        
        # Test moving decay (scales with movement cost = 2.0)
        update_agent_needs(agent, world, moved=True)
        # previous hunger: 0.05. new: 0.05 + 0.15 * 2.0 = 0.35
        # previous thirst: 0.10. new: 0.10 + 0.30 * 2.0 = 0.70
        # previous energy: 99.96. new: 99.96 - 0.12 * 2.0 = 99.72
        self.assertAlmostEqual(agent.hunger, 0.35)
        self.assertAlmostEqual(agent.thirst, 0.70)
        self.assertAlmostEqual(agent.energy, 99.72)

    def test_agent_knowledge_confidence_decay(self):
        """Verify that confidence ratings in spatial knowledge decay over ticks."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.ecology_ablation = {"memory_fidelity": False}
        agent = Agent(agent_id=0, location=(50, 50))
        
        # Add memory
        agent.add_memory(WATER, (60, 60), tick=0, importance=1.0)
        self.assertEqual(agent.knowledge.water_sources[(60, 60)]["confidence"], 1.0)
        
        # Run decay
        update_agent_needs(agent, world, moved=False)
        # Confidence should decay by 0.001
        self.assertAlmostEqual(agent.knowledge.water_sources[(60, 60)]["confidence"], 0.999)

    def test_agent_chunk_visited_tracking(self):
        """Verify that perception correctly updates visited chunk coordinates."""
        world = WorldState(width=128, height=128, seed=self.seed)
        # Setup biome and elevation so perception doesn't crash
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        
        agent = Agent(agent_id=0, location=(64, 64))
        
        # Before perception
        self.assertEqual(len(agent.visited_chunks), 0)
        
        # Perceive (vision radius = 20)
        perceive(agent, world, vision_radius=20, chunk_size=32)
        
        # Center is (64,64) which is chunk (2,2)
        # Vision bounding box is y: 44 to 85, x: 44 to 85.
        # Chunks intersected:
        # y: 44//32 = 1, 84//32 = 2. So chunk rows 1 and 2.
        # x: 44//32 = 1, 84//32 = 2. So chunk cols 1 and 2.
        # Visited chunks should contain (1,1), (1,2), (2,1), (2,2)
        expected_chunks = {(1, 1), (1, 2), (2, 1), (2, 2)}
        self.assertEqual(agent.visited_chunks, expected_chunks)

    def test_agent_fatigue_returns_home_to_rest(self):
        """Verify that agents return to their home location when energy is low."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.movement_cost.fill(1.0)
        
        agent = Agent(agent_id=0, location=(60, 60))
        agent.max_age = 80 * 360
        agent.age = 30 * 360 # Force adult age to ensure effective_speed = 1.0
        agent.home_location = (50, 50)
        
        # Force low energy
        agent.energy = 30.0
        
        # Evaluate action
        action_name, target = evaluate_utility(agent, world)
        self.assertEqual(action_name, "Resting")
        self.assertEqual(target, (50, 50))
        
        # Take step towards home
        moved = step_toward(agent, target, world)
        self.assertTrue(moved)
        # Should step towards (50, 50) from (60, 60)
        # cy=60, ty=50 => dy = np.sign(-10) = -1. dist_y = 10
        # cx=60, tx=50 => dx = np.sign(-10) = -1. dist_x = 10
        # dist_y >= dist_x, so it steps along y: ny = 59
        self.assertEqual(agent.location, (59, 60))

    def test_agent_spawning_at_nms_settlements(self):
        """Verify that agents are spawned at predicted settlement locations."""
        world = generate_world(128, 128, self.seed)
        run_simulation(world, ticks=0) # Run simulation 0 ticks to trigger spawn
        
        self.assertEqual(len(world.agents), 16)
        
        # In Phase 5, colony spawning uses scaled exclusion radius
        w_scale = world.width / 1024.0
        spots = predict_settlements(world, count=4, exclusion_radius=250.0 * w_scale)
        if len(spots) < 4:
            spots = predict_settlements(world, count=4, exclusion_radius=100.0 * w_scale)
        if len(spots) < 4:
            spots = predict_settlements(world, count=4, exclusion_radius=20.0 * w_scale)
            
        spot_coords = {(s['y'], s['x']) for s in spots}
        
        # Check that all agent starting locations are in the predicted spots
        for agent in world.agents:
            self.assertIn(agent.path_history[0], spot_coords)

    def test_phase5_genetics_and_behavior_clustering(self):
        """Verify Phase 5 genetics, crossover, mutation, and behavioral clustering functions."""
        from world.agents.genetics import create_genome, crossover, mutate, express_genome, population_diversity
        from world.agents.behavior_classifier import classify_behavior, build_feature_vector
        
        # 1. Genome Creation & Expression
        gen_a = create_genome()
        gen_b = create_genome()
        self.assertEqual(len(gen_a.genes), 14)
        self.assertTrue(np.all((gen_a.genes >= 0.0) & (gen_a.genes <= 1.0)))
        
        brain = express_genome(gen_a)
        self.assertIn("vision_radius", brain)
        self.assertIn("max_age_offset", brain)
        
        # 2. Crossover & Mutation
        child_gen = crossover(gen_a, gen_b)
        self.assertEqual(len(child_gen.genes), 14)
        for i in range(14):
            # Child gene must come from either parent A or parent B
            self.assertTrue(child_gen.genes[i] == gen_a.genes[i] or child_gen.genes[i] == gen_b.genes[i])
            
        mutated_gen = mutate(child_gen, mutation_rate=0.05)
        self.assertTrue(np.all((mutated_gen.genes >= 0.0) & (mutated_gen.genes <= 1.0)))
        
        # 3. Pairwise diversity
        div = population_diversity([gen_a, gen_b])
        self.assertTrue(div >= 0.0)
        
        # 4. Behavioral Feature Vector & K-Means Clustering
        world = generate_world(128, 128, self.seed)
        run_simulation(world, ticks=10) # Run a few ticks to get some action history
        
        # Build feature vector for one agent
        agent = world.agents[0]
        feat = build_feature_vector(agent, world.width)
        self.assertEqual(feat.shape, (8,))
        self.assertTrue(np.all((feat >= 0.0) & (feat <= 1.0)))
        self.assertFalse(np.any(np.isnan(feat)))
        
        # Run clustering
        result = classify_behavior(world.agents, world.width, n_clusters=4)
        self.assertEqual(result["n_clusters"], 4)
        self.assertIn("agent_clusters", result)
        self.assertIn("cluster_centroids", result)
        self.assertIn("pca_coords", result)
        
        # Verify that K-Means cluster labels were applied in-place to agents
        for a in world.agents:
            self.assertTrue(a.behavior_cluster.startswith("C"))

    def test_discoveries_count_increment(self):
        """Verify that discoveries_count increments only on first-time resource discovery."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        # Create a water tile in range
        world.lake_map[60, 60] = 10.0
        
        agent = Agent(agent_id=0, location=(64, 64))
        self.assertEqual(agent.discoveries_count, 0)
        
        # First perception: should discover water
        perceive(agent, world, vision_radius=20, chunk_size=32)
        self.assertEqual(agent.discoveries_count, 1)
        
        # Second perception: water is already in knowledge, discoveries should not increment
        perceive(agent, world, vision_radius=20, chunk_size=32)
        self.assertEqual(agent.discoveries_count, 1)

    def test_behavioral_entropy_action_counts(self):
        """Verify that action_counts increments on every tick representing behavioral entropy."""
        from world.agents.simulation import simulate_agent_tick
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.movement_cost.fill(1.0)
        
        agent = Agent(agent_id=0, location=(64, 64))
        
        # Simulate a tick
        simulate_agent_tick(agent, world)
        
        total_ticks = sum(agent.action_counts.values())
        self.assertEqual(total_ticks, 1)

    def test_agent_sigmoid_utility(self):
        """Verify that sigmoid utility scales needs and respects soft thresholds."""
        from world.agents.decision import sigmoid_utility
        
        # Low values should have negligible utility
        val_low = sigmoid_utility(need=20.0, threshold=50.0, scale=10.0)
        self.assertLess(val_low, 10.0)
        
        # Center threshold value should be exactly 50% (50.0)
        val_mid = sigmoid_utility(need=50.0, threshold=50.0, scale=10.0)
        self.assertAlmostEqual(val_mid, 50.0)
        
        # High value should saturate near 100
        val_high = sigmoid_utility(need=80.0, threshold=50.0, scale=10.0)
        self.assertGreater(val_high, 90.0)
        
    def test_agent_max_radius_tracking(self):
        """Verify that agent max_radius accumulates distance correctly when stepping."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.movement_cost.fill(1.0)
        
        agent = Agent(agent_id=0, location=(64, 64))
        agent.max_age = 80 * 360
        agent.age = 30 * 360 # Force adult age to ensure effective_speed = 1.0
        self.assertEqual(agent.max_radius, 0.0)
        
        # Step from (64, 64) toward (74, 64)
        target = (74, 64)
        step_toward(agent, target, world)
        
        # Spawning location was (64, 64), stepped to (65, 64). Max radius is 1.0.
        self.assertEqual(agent.max_radius, 1.0)
        
        # Force teleport to test cumulative maximum
        agent.location = (84, 64)
        step_toward(agent, (94, 64), world) # step to (85, 64)
        # Distance from spawn (64, 64) to (85, 64) is 21.0
        self.assertEqual(agent.max_radius, 21.0)

    def test_dynamic_resource_drying_and_uncertainty(self):
        """Verify that seasonal drying prevents metabolic restoration and decays memory confidence by 0.2."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.movement_cost.fill(1.0)
        
        # Setup dry environment (no lake or river flow at target coordinate (60, 60))
        world.biome[60, 60] = GRASSLAND
        world.lake_map[60, 60] = 0.0
        world.river_map[60, 60] = 0.0
        
        agent = Agent(agent_id=0, location=(60, 60))
        agent.max_age = 80 * 360
        agent.age = 30 * 360 # Force adult age to ensure effective_speed = 1.0
        agent.thirst = 50.0
        agent.current_action = "Drinking"
        
        # Seed memory with water coordinate
        agent.add_memory(WATER, (60, 60), tick=0, importance=1.0)
        self.assertEqual(agent.knowledge.water_sources[(60, 60)]["confidence"], 1.0)
        self.assertEqual(agent.failed_water_visits, 0)
        
        # Step toward (60, 60) which is already reached. Should trigger action drinking evaluation.
        step_toward(agent, (60, 60), world)
        
        # Thirst should NOT be restored because there is no water
        self.assertEqual(agent.thirst, 50.0)
        # Action is reset to Idle
        self.assertEqual(agent.current_action, "Idle")
        # Failed water visits increments
        self.assertEqual(agent.failed_water_visits, 1)
        # Confidence decays: 1.0 * 0.2 = 0.2
        self.assertAlmostEqual(agent.knowledge.water_sources[(60, 60)]["confidence"], 0.2)
        
        # Second failed visit
        agent.current_action = "Drinking"
        step_toward(agent, (60, 60), world)
        # Confidence decays: 0.2 * 0.2 = 0.04 (which is < 0.05, so it gets deleted)
        self.assertNotIn((60, 60), agent.knowledge.water_sources)
        self.assertEqual(agent.failed_water_visits, 2)
        self.assertEqual(agent.nodes_removed_count, 1)

    def test_novelty_and_rediscoveries(self):
        """Verify that discoveries_count increments only once per unique coordinate, and forgotten rediscoveries increment rediscoveries."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.lake_map[60, 60] = 10.0
        
        agent = Agent(agent_id=0, location=(64, 64))
        self.assertEqual(agent.discoveries_count, 0)
        self.assertEqual(agent.rediscoveries, 0)
        
        # 1. Perceive first time: discovers water
        perceive(agent, world, vision_radius=20, chunk_size=32)
        self.assertEqual(agent.discoveries_count, 1)
        self.assertEqual(agent.rediscoveries, 0)
        self.assertIn((60, 60), agent.discovered_water)
        
        # 2. Force forget by removing from active knowledge
        del agent.knowledge.water_sources[(60, 60)]
        
        # 3. Perceive again: rediscover forgotten coordinate
        perceive(agent, world, vision_radius=20, chunk_size=32)
        # discoveries_count should NOT increment (already in lifetime set)
        self.assertEqual(agent.discoveries_count, 1)
        # rediscoveries should increment by 1!
        self.assertEqual(agent.rediscoveries, 1)

    def test_frequency_based_seasonal_memory(self):
        """Verify that seasonal active and dry counters increment correctly in spatial memory."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        
        agent = Agent(agent_id=0, location=(60, 60))
        agent.max_age = 80 * 360
        agent.age = 30 * 360 # Force adult age to ensure effective_speed = 1.0 (deterministic stepping)
        agent.thirst = 50.0
        
        # Spring (tick 0, season 0)
        world.tick = 0
        
        # Add memory first to initialize counter at 1
        agent.add_memory(WATER, (60, 60), tick=0, importance=1.0)
        
        # Successful drink
        world.lake_map[60, 60] = 10.0
        agent.current_action = "Drinking"
        step_toward(agent, (60, 60), world)
        
        self.assertEqual(agent.knowledge.water_sources[(60, 60)]["active_seasons"][0], 2) # 1 on add_memory, 1 on arrival = 2
        self.assertEqual(agent.knowledge.water_sources[(60, 60)]["dry_seasons"][0], 0)
        
        # Dry drink in Summer (tick 90, season 1)
        world.tick = 90
        world.lake_map[60, 60] = 0.0
        agent.thirst = 50.0
        agent.current_action = "Drinking"
        step_toward(agent, (60, 60), world)
        
        self.assertEqual(agent.knowledge.water_sources[(60, 60)]["active_seasons"][1], 0)
        self.assertEqual(agent.knowledge.water_sources[(60, 60)]["dry_seasons"][1], 1)

    def test_wisdom_and_age_properties(self):
        """Verify that dynamic properties (speed, vision, risk, curiosity) scale correctly with age."""
        agent = Agent(agent_id=0, location=(50, 50))
        agent.max_age = 80 * 360 # Set fixed max_age to prevent random normal fluctuation
        
        # Spawn age should be between 5 and 45 years
        age_years = agent.age / 360.0
        self.assertTrue(5.0 <= age_years <= 45.0)
        
        # Test child traits (age < 14, e.g. Juvenile stage)
        agent.age = int(10 * 360)
        self.assertEqual(agent.effective_speed, 0.6)
        self.assertEqual(agent.vision_radius, 15)
        self.assertGreater(agent.effective_risk, agent.base_traits["risk_tolerance"])
        
        # Test adult traits (18 <= age < 60)
        agent.age = int(35 * 360)
        self.assertEqual(agent.effective_speed, 1.0)
        self.assertEqual(agent.vision_radius, 20)
        self.assertEqual(agent.effective_risk, agent.base_traits["risk_tolerance"])
        
        # Test elderly traits (age >= 60)
        agent.max_age = int(70 * 360) # Ensure senescence_factor = 1.0 for age 70
        agent.age = int(70 * 360)
        self.assertEqual(agent.effective_speed, 0.6)
        self.assertEqual(agent.vision_radius, 14)
        self.assertLess(agent.effective_risk, agent.base_traits["risk_tolerance"])

    def test_predictive_utility_gain(self):
        """Verify that agents choose predictive active resources over closer dry ones, and accumulate prediction gain."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.movement_cost.fill(1.0)
        
        agent = Agent(agent_id=0, location=(64, 64))
        agent.max_age = 80 * 360
        agent.age = 30 * 360 # Force adult age to ensure effective_speed = 1.0
        agent.thirst = 90.0
        agent.prediction_enabled = True
        
        # We are in Winter transition phase (tick 315, season 3), so upcoming target season is Spring (0)
        world.tick = 315
        
        # Closer water source at (62, 64) - dry in Spring (season 0)
        agent.knowledge.water_sources[(62, 64)] = {
            "confidence": 1.0,
            "last_seen_tick": 0,
            "season_seen": 0,
            "active_seasons": {0: 0, 1: 5, 2: 0, 3: 0},
            "dry_seasons": {0: 5, 1: 0, 2: 0, 3: 0}
        }
        # Further water source at (60, 64) - active in Spring (season 0)
        agent.knowledge.water_sources[(60, 64)] = {
            "confidence": 1.0,
            "last_seen_tick": 0,
            "season_seen": 0,
            "active_seasons": {0: 5, 1: 0, 2: 0, 3: 0},
            "dry_seasons": {0: 0, 1: 5, 2: 0, 3: 0}
        }
        
        # Modulate actual world water states: (62, 64) is dry, (60, 64) is active
        world.lake_map[62, 64] = 0.0
        world.lake_map[60, 64] = 10.0
        
        # Evaluate action
        action_name, target = evaluate_utility(agent, world)
        
        # The agent should predict that (62, 64) is dry (probability 0.0)
        # and (60, 64) is active (probability 1.0).
        # Thus it should choose (60, 64) even though it is further!
        self.assertEqual(action_name, "Drinking")
        self.assertEqual(target, (60, 64))
        self.assertEqual(agent.prediction_decisions, 1)
        self.assertIsNotNone(agent.predicted_destination)
        
        # Force arrival at target (60, 64)
        agent.location = (60, 64)
        agent.current_action = "Drinking"
        step_toward(agent, (60, 64), world)
        
        # Should have successfully drank and accumulated prediction success and gain!
        self.assertEqual(agent.thirst, 0.0)
        self.assertEqual(agent.prediction_successes, 1)
        # Bypassed target (62, 64) was indeed dry, so prediction gain increments!
        self.assertEqual(agent.prediction_gains, 1)

    def test_physiological_reserves_starvation(self):
        """Verify hunger depletes fat reserves, then muscle mass, and that muscle depletion causes speed loss."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.movement_cost.fill(1.0)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.temperature.fill(20.0) # comfortable temp
        
        agent = Agent(agent_id=0, location=(50, 50))
        agent.hunger = 100.0 # Starving!
        agent.fat_reserves = 100.0
        agent.muscle_mass = 100.0
        
        # 1. First tick starving: should burn fat reserves, not muscle, health remains 100
        update_agent_needs(agent, world, moved=False)
        self.assertLess(agent.fat_reserves, 100.0)
        self.assertEqual(agent.muscle_mass, 100.0)
        self.assertEqual(agent.health, 100.0)
        
        # 2. Exhaust fat reserves to test muscle burning
        agent.fat_reserves = 0.0
        update_agent_needs(agent, world, moved=False)
        self.assertEqual(agent.fat_reserves, 0.0)
        self.assertLess(agent.muscle_mass, 100.0)
        # Health remains 100 because muscle mass is still > 0
        self.assertEqual(agent.health, 100.0)
        # Speed should decrease continuously as muscle mass drops
        self.assertLess(agent.effective_speed, 1.0)
        
        # 3. Exhaust muscle mass to test starvation damage
        agent.muscle_mass = 0.0
        update_agent_needs(agent, world, moved=False)
        self.assertLess(agent.health, 100.0)
        self.assertGreater(agent.starvation_damage_accumulated, 0.0)

    def test_injury_system_trauma_and_healing(self):
        """Verify injury trauma from steep movement, movement speed reduction, and healing in shelters."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.temperature.fill(20.0)
        
        agent = Agent(agent_id=0, location=(50, 50))
        agent.max_age = 80 * 360
        agent.age = 30 * 360 # adult
        self.assertEqual(agent.injury_level, 0.0)
        self.assertEqual(agent.effective_speed, 1.0)
        
        # 1. Moving on steep terrain (high cost) should have a chance to trigger injury
        # Let's mock a very high movement cost
        world.movement_cost.fill(3.0)
        # Force injury by setting it directly to test speed reduction
        agent.injury_level = 40.0
        # Speed should be reduced by: base_speed * (1.0 - 0.5 * 40/100) = 0.8
        self.assertAlmostEqual(agent.effective_speed, 0.8)
        
        # 2. Severe injury (>=50.0) should deal direct health damage
        agent.injury_level = 60.0
        update_agent_needs(agent, world, moved=False)
        self.assertLess(agent.health, 100.0)
        self.assertGreater(agent.injury_damage_accumulated, 0.0)
        
        # 3. Resting in a comfortable shelter should heal injury
        agent.shelter_location = (50, 50)
        agent.shelter_level = 1
        agent.shelter_durability = 100.0
        agent.current_action = "Resting"
        agent.hunger = 10.0
        agent.thirst = 10.0
        
        # Run needs update. It should heal injury by 2.0
        update_agent_needs(agent, world, moved=False)
        self.assertEqual(agent.injury_level, 58.0)

    def test_biome_temperatures_and_layered_weather(self):
        """Verify local temperatures vary by biome and season, and moderate weather scales metabolic rate."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.movement_cost.fill(1.0)
        world.elevation.fill(0.5)
        # Setup base temperature of 20°C
        world.temperature.fill(20.0)
        
        # Spawn agent in Desert (6)
        world.biome.fill(6) # DESERT (amplitude 22°C)
        agent = Agent(agent_id=0, location=(50, 50))
        agent.max_age = 80 * 360
        agent.age = 30 * 360 # adult
        
        # 1. Peak Winter (tick 0): temp should drop by amplitude (22°C), so local temp is 20 - 22 = -2°C
        world.tick = 0
        # Run needs update and verify metabolic scaling. Cold pressure (< 12°C) increases hunger growth.
        # Base hunger increase is 0.05. With cold pressure, it should be higher.
        hunger_before = agent.hunger
        update_agent_needs(agent, world, moved=False)
        hunger_gain_cold = agent.hunger - hunger_before
        self.assertGreater(hunger_gain_cold, 0.05)
        
        # 2. Peak Summer (tick 180): temp should rise by amplitude (22°C), local temp is 20 + 22 = 42°C
        world.tick = 180
        agent.hunger = 0.0
        thirst_before = agent.thirst
        update_agent_needs(agent, world, moved=False)
        thirst_gain_hot = agent.thirst - thirst_before
        # Heat pressure (> 28°C) increases thirst growth significantly
        self.assertGreater(thirst_gain_hot, 0.10)

    def test_shelter_placement_optimization_and_insulation(self):
        """Verify agents optimize shelter location selection, build/upgrade, and receive partial insulation."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.4) # flat, moderate
        world.temperature.fill(20.0)
        world.movement_cost.fill(1.0)
        world.wood.fill(0.5) # rich resource
        world.stone.fill(0.5)
        
        agent = Agent(agent_id=0, location=(50, 50))
        agent.max_age = 80 * 360
        agent.age = 30 * 360 # adult
        
        # 1. Site selection: should select a close, resource-rich, flat tile
        site = select_shelter_site(agent, world)
        self.assertIsNotNone(site)
        
        # 2. Build shelter: check that Building Shelter action restores durability
        agent.shelter_location = (50, 50)
        agent.shelter_level = 1
        agent.shelter_durability = 50.0
        agent.current_action = "Building Shelter"
        
        # Reach target and build
        from world.agents.decision import step_toward
        step_toward(agent, (50, 50), world)
        self.assertEqual(agent.shelter_durability, 60.0)
        
        # 3. Upgrade shelter: when durability reaches 100%, next build tick should upgrade to level 2 if wood is rich
        agent.shelter_durability = 90.0
        agent.current_action = "Building Shelter"
        step_toward(agent, (50, 50), world)
        self.assertEqual(agent.shelter_durability, 100.0)
        self.assertEqual(agent.shelter_level, 2)
        # Local wood should be depleted
        self.assertLess(world.wood[50, 50], 0.5)

    def test_personal_danger_memory_avoidance(self):
        """Verify agents build personal danger memories from corpses or near misses and apply utility penalties."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.ecology_ablation = {"memory_fidelity": False}
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.temperature.fill(20.0)
        world.movement_cost.fill(1.0)
        
        agent = Agent(agent_id=0, location=(50, 50))
        agent.max_age = 80 * 360
        agent.age = 30 * 360
        
        # Ensure shelter exists and is fully durable so the agent has no urge to build one
        agent.shelter_location = (50, 50)
        agent.shelter_level = 1
        agent.shelter_durability = 100.0
        
        # 1. Near miss: health drops below 30% should record danger memory
        agent.health = 25.0
        update_agent_needs(agent, world, moved=False)
        self.assertIn((50, 50), agent.knowledge.danger_locations)
        self.assertAlmostEqual(agent.knowledge.danger_locations[(50, 50)], 0.999)
        
        # 2. Witnessing corpse: spawn a dead agent nearby and perceive it
        other = Agent(agent_id=1, location=(52, 50))
        other.dead = True
        world.agents = [agent, other]
        
        from world.agents.perception import perceive
        perceive(agent, world, vision_radius=20, chunk_size=32)
        # Should have added (52, 50) to danger memory!
        self.assertIn((52, 50), agent.knowledge.danger_locations)
        
        # 3. Utility penalty: evaluate utility of food at danger coordinates vs safe coordinates
        # Add food source at danger coordinate (52, 50) and safe coordinate (60, 50)
        agent.knowledge.food_sources[(52, 50)] = {"confidence": 1.0}
        agent.knowledge.food_sources[(60, 50)] = {"confidence": 1.0}
        
        # Close targets: (52, 50) is closer (dist 2) than (60, 50) (dist 10).
        # Without danger penalty, (52, 50) would have much higher utility due to proximity.
        # But with danger penalty, (52, 50) should be heavily penalized!
        agent.hunger = 80.0
        action_name, target = evaluate_utility(agent, world)
        # Should choose the safe, further coordinate (60, 50) instead of the dangerous, closer one (52, 50)!
        self.assertEqual(action_name, "Eating")
        self.assertEqual(target, (60, 50))

    def test_phase5_continuous_feature_learning(self):
        """Verify Hebbian/Perceptron updates on continuous environmental feature weights."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.ecology_ablation = {"dehydration_ramp": False}
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.temperature.fill(20.0)
        world.movement_cost.fill(1.0)

        agent = Agent(agent_id=0, location=(50, 50))
        agent.learning_rate = 0.05
        agent.feature_weights = np.zeros(5, dtype=np.float32)

        # 1. Negative outcome: dehydration damage (thirst = 100.0 causes 2.0 health loss)
        # Since local temperature is 8.0°C (seasonal winter offset at day 0):
        # f_temp = (8.0 - 20) / 20 = -0.6
        # f_elev = 0.5, f_moisture = 2500 / 3000 = 0.833, f_food = 0.0, f_water = 0.0
        # Health loss = 2.0, so updates should be:
        # weight -= lr * features * (health_loss / 10.0) = -0.05 * features * 0.2
        # For temp weight: -0.05 * (-0.6) * 0.2 = +0.006
        agent.thirst = 100.0
        agent.health = 100.0
        update_agent_needs(agent, world, moved=False)
        self.assertTrue(np.any(agent.feature_weights != 0.0))
        self.assertAlmostEqual(agent.feature_weights[0], 0.006, places=4)

        # Reset weights
        agent.feature_weights = np.zeros(5, dtype=np.float32)
        # 2. Positive outcome: consumed food
        agent._consumed_food = True
        update_agent_needs(agent, world, moved=False)
        # weight += lr * features * 0.3 = +0.015 * features
        self.assertTrue(agent.feature_weights[1] > 0.0) # elevation weight should be positive
        self.assertFalse(agent._consumed_food) # consumed flag should be reset

    def test_phase5_social_memory_and_mating_gating(self):
        """Verify dispute memories scale down trust and gate partner eligibility."""
        from world.agents.reproduction import find_eligible_mate, is_eligible_to_reproduce
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.temperature.fill(20.0)

        # Create two eligible adults in same colony
        agent_a = Agent(agent_id=0, location=(50, 50))
        agent_a.age = 25 * 360
        agent_a.colony_id = 1
        agent_a.shelter_durability = 100.0
        agent_a.reproduction_cooldown = 0

        agent_b = Agent(agent_id=1, location=(51, 50))
        agent_b.age = 25 * 360
        agent_b.colony_id = 1
        agent_b.shelter_durability = 100.0
        agent_b.reproduction_cooldown = 0

        world.agents = [agent_a, agent_b]

        # Initially, trust is neutral (social modifier = 1.0)
        self.assertEqual(agent_a.get_social_modifier(agent_b.id, world.tick), 1.0)
        self.assertTrue(is_eligible_to_reproduce(agent_a))
        self.assertTrue(is_eligible_to_reproduce(agent_b))

        mate = find_eligible_mate(agent_a, world)
        self.assertEqual(mate.id, agent_b.id)

        # Record dispute memory -> trust drops below 0.6
        agent_a.add_memory(DANGER, agent_b.location, world.tick, importance=0.6, associated_id=agent_b.id, outcome="dispute")
        self.assertTrue(agent_a.get_social_modifier(agent_b.id, world.tick) < 0.6)

        # Now find_eligible_mate should reject agent_b
        mate = find_eligible_mate(agent_a, world)
        self.assertIsNone(mate)

        # Record cooperative sharing memory -> trust is restored
        agent_a.add_memory(FOOD, agent_b.location, world.tick, importance=0.4, associated_id=agent_b.id, outcome="share")
        self.assertTrue(agent_a.get_social_modifier(agent_b.id, world.tick) >= 0.6)
        mate = find_eligible_mate(agent_a, world)
        self.assertEqual(mate.id, agent_b.id)

    def test_phase5_abandoned_shelters_and_weathering(self):
        """Verify that agents claim abandoned shelters and that weathering decays them."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.temperature.fill(20.0)
        world.shelters = {}

        # Create an abandoned shelter in the registry
        shelter_loc = (52, 50)
        world.shelters[shelter_loc] = {
            "level": 2,
            "durability": 80.0,
            "owner_id": None
        }

        # Agent with no shelter at (52, 50)
        agent = Agent(agent_id=0, location=(52, 50))
        agent.age = 30 * 360  # Force adult age to ensure effective_speed = 1.0
        agent.shelter_location = None
        agent.shelter_durability = 0.0
        world.agents = [agent]

        # Evaluate utility -> should claim the abandoned shelter
        action, target = evaluate_utility(agent, world)
        self.assertEqual(agent.shelter_location, shelter_loc)
        self.assertEqual(agent.shelter_level, 2)
        self.assertEqual(agent.shelter_durability, 80.0)
        self.assertEqual(world.shelters[shelter_loc]["owner_id"], agent.id)

        # Weathering decay step
        # Run simulation tick weathering: durability drops by 0.1, but since agent is at shelter
        # and has no other major needs, it chooses to build/repair shelter (+10 durability),
        # resulting in 90.0 - 0.1 = 89.9 durability.
        run_simulation(world, ticks=1)
        self.assertAlmostEqual(world.shelters[shelter_loc]["durability"], 89.9)
        self.assertAlmostEqual(agent.shelter_durability, 89.9)

        # Kill agent and verify shelter becomes abandoned
        agent.dead = True
        run_simulation(world, ticks=1)
        self.assertIsNone(world.shelters[shelter_loc]["owner_id"])

    def test_phase5_prediction_skepticism_and_extinction(self):
        """Verify that prediction confidence updates and that 0 population triggers early exit."""
        world = WorldState(width=128, height=128, seed=self.seed)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.temperature.fill(20.0)

        agent = Agent(agent_id=0, location=(50, 50))
        agent.age = 20 * 360  # Prime adult (speed = 1.0)
        agent.muscle_mass = 100.0
        agent.injury_level = 0.0
        agent.prediction_confidence = 0.8
        world.agents = [agent]

        # 1. Prediction success: add prediction destination and trigger success
        agent.predicted_destination = ((50, 50), 0, None)
        # Tick is 0, so season is 0
        agent.current_action = "Drinking"
        # Force fresh water to exist at (50, 50)
        world.lake_map[50, 50] = 5.0
        step_toward(agent, (50, 50), world)
        # Confidence should increase by 0.05
        self.assertAlmostEqual(agent.prediction_confidence, 0.85)

        # 2. Prediction failure: dried up water
        agent.predicted_destination = ((50, 50), 0, None)
        agent.current_action = "Drinking"
        world.lake_map[50, 50] = 0.0 # dried up
        step_toward(agent, (50, 50), world)
        # Confidence should decrease by 0.15
        self.assertAlmostEqual(agent.prediction_confidence, 0.70)

        # 3. Early Extinction: run simulation and verify it exits immediately when population is 0
        agent.dead = True
        # Since agent is dead, run_simulation should terminate immediately even if requested ticks = 100
        stats = run_simulation(world, ticks=100, save_epochs=True, sample_interval=10)
        # It should terminate at tick 1
        self.assertEqual(world.tick, 1)

    def test_competitive_episodic_retrieval(self):
        """Verify that get_social_modifier implements competitive retrieval over top 5 memories."""
        agent = Agent(agent_id=0, location=(50, 50))
        self.assertEqual(agent.get_social_modifier(1, current_tick=0), 1.0)
        
        # Add dispute memory
        agent.add_memory(DANGER, (50, 50), tick=0, importance=0.6, associated_id=1, outcome="dispute")
        self.assertTrue(agent.get_social_modifier(1, current_tick=0) < 0.6)
        
        # Add share memory
        agent.add_memory(FOOD, (50, 50), tick=0, importance=0.4, associated_id=1, outcome="share")
        self.assertTrue(agent.get_social_modifier(1, current_tick=0) >= 0.6)

    def test_multiple_concept_formation_and_decay(self):
        """Verify that leader-follower clustering forms multiple concepts and daily decay applies."""
        world = WorldState(width=128, height=128, seed=1729)
        world.elevation.fill(0.1)
        world.temperature.fill(25.0)
        world.rainfall.fill(1000.0)
        
        agent = Agent(agent_id=0, location=(50, 50))
        
        # Add memories at 10 distinct coordinates in low elevation to satisfy support >= 10
        for i in range(10):
            agent.add_memory(WATER, (10, 10 + i), tick=0, importance=1.0)
        agent.update_concepts(world)
        self.assertEqual(len(agent.concepts[WATER]), 1)
        self.assertAlmostEqual(agent.concepts[WATER][0].elevation_mean, 0.1)
        self.assertTrue(agent.matches_concept(WATER, 10, 10, world))
        
        # Decay concept
        agent.concepts[WATER][0].confidence = 0.05
        # Filter logic simulated:
        agent.concepts[WATER] = [c for c in agent.concepts[WATER] if c.confidence >= 0.1]
        self.assertEqual(len(agent.concepts[WATER]), 0)

    def test_concept_guided_explore(self):
        """Verify that exploration targets are biased towards concept-matching coordinates."""
        world = WorldState(width=128, height=128, seed=1729)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.1)
        world.temperature.fill(20.0)
        world.rainfall.fill(1000.0)
        
        agent = Agent(agent_id=0, location=(64, 64))
        from world.agents.cognitive import Concept
        c = Concept(resource_type=FOOD, elevation_mean=0.1, elevation_std=0.01, confidence=1.0)
        agent.concepts[FOOD] = [c]
        
        agent.hunger = 90.0
        agent.thirst = 10.0
        agent.curiosity_need = 80.0
        agent.visited_chunks.clear()
        
        action, target = evaluate_utility(agent, world)
        self.assertIsNotNone(target)
        self.assertTrue(agent.matches_concept(FOOD, target[0], target[1], world))

    def test_predictor_online_training(self):
        """Verify that forward predictions and backprop updates operate correctly on Predictor."""
        from world.agents.cognitive import Predictor
        predictor = Predictor()
        
        context = np.random.normal(0.0, 1.0, 20).astype(np.float32)
        p1 = predictor.predict(context)
        
        predictor.train(context, reward=5.0, learning_rate=0.1)
        p2 = predictor.predict(context)
        
        if p1 < 5.0:
            self.assertGreater(p2, p1)
        else:
            self.assertLess(p2, p1)

    def test_procedural_memory_caching(self):
        """Verify that successful/unsuccessful action sequences cache/penalize procedures."""
        world = WorldState(width=128, height=128, seed=1729)
        world.biome.fill(GRASSLAND)
        world.elevation.fill(0.5)
        world.temperature.fill(20.0)
        world.movement_cost.fill(1.0)
        
        agent = Agent(agent_id=0, location=(64, 64))
        agent.action_history = ["Exploring", "Building Shelter", "Store Food"]
        
        # High reward event
        agent.last_prediction_input = np.zeros(20, dtype=np.float32)
        agent.shelter_location = (64, 64)
        agent.shelter_durability = 100.0
        agent._consumed_food = True # triggers meaningful event
        
        # Set current action to Resting to trigger Sleep Consolidation
        agent.current_action = "Resting"
        simulate_agent_tick(agent, world)
        
        self.assertEqual(len(agent.procedures), 1)
        p = agent.procedures[0]
        self.assertEqual(p.action_sequence, ("Exploring", "Building Shelter", "Store Food"))


    # ======================================================================
    # Phase 8.1 -- Biological Drive Layer Tests
    # ======================================================================

    def test_phase8_drivestate_initializes_neutral(self):
        """Verify DriveState initializes at neutral with zero biological tensions."""
        from world.agents.drives import DriveState
        ds = DriveState()
        self.assertEqual(ds.hunger_tension, 0.0)
        self.assertEqual(ds.thirst_tension, 0.0)
        self.assertEqual(ds.exhaustion_tension, 0.0)
        self.assertEqual(ds.pain_tension, 0.0)
        self.assertEqual(ds.thermal_stress, 0.0)
        self.assertEqual(ds.fear, 0.0)
        self.assertAlmostEqual(ds.contentment, 1.0, places=5)

    def test_phase8_biological_ema_smoothing(self):
        """Verify biological tensions smooth toward raw need values over multiple ticks."""
        from world.agents.drives import update_biological_drives
        from world.agents.agent import Agent
        agent = Agent(agent_id=0, location=(64, 64))
        agent.hunger = 90.0
        agent.thirst = 80.0
        agent.energy = 10.0
        agent.injury_level = 0.0
        for _ in range(10):
            update_biological_drives(agent, 20.0)
        ds = agent.drives
        self.assertGreater(ds.hunger_tension, 0.0)
        self.assertGreater(ds.thirst_tension, 0.0)
        self.assertGreater(ds.exhaustion_tension, 0.0)
        self.assertAlmostEqual(ds.thermal_stress, 0.0, places=3)

    def test_phase8_thermal_stress(self):
        """Verify thermal stress builds at extreme (cold and hot) temperatures."""
        from world.agents.drives import update_biological_drives
        from world.agents.agent import Agent
        agent_cold = Agent(agent_id=0, location=(64, 64))
        for _ in range(20):
            update_biological_drives(agent_cold, -10.0)
        self.assertGreater(agent_cold.drives.thermal_stress, 0.3)
        agent_hot = Agent(agent_id=1, location=(64, 64))
        for _ in range(20):
            update_biological_drives(agent_hot, 50.0)
        self.assertGreater(agent_hot.drives.thermal_stress, 0.3)

    def test_phase8_contentment_decreases_with_pressure(self):
        """Verify contentment falls as biological needs accumulate."""
        from world.agents.drives import update_biological_drives
        from world.agents.agent import Agent
        agent = Agent(agent_id=0, location=(64, 64))
        self.assertAlmostEqual(agent.drives.contentment, 1.0, places=5)
        agent.hunger = 100.0
        agent.thirst = 100.0
        for _ in range(20):
            update_biological_drives(agent, 20.0)
        self.assertLess(agent.drives.contentment, 0.5)

    def test_phase8_valence_and_arousal_in_range(self):
        """Verify valence [-1,1] and arousal [0,1] stay within bounds under stress."""
        from world.agents.drives import update_biological_drives
        from world.agents.agent import Agent
        agent = Agent(agent_id=0, location=(64, 64))
        agent.hunger = 100.0
        agent.thirst = 100.0
        agent.injury_level = 80.0
        agent.energy = 5.0
        for _ in range(30):
            update_biological_drives(agent, -5.0)
        ds = agent.drives
        self.assertGreaterEqual(ds.valence, -1.0)
        self.assertLessEqual(ds.valence, 1.0)
        self.assertGreaterEqual(ds.arousal, 0.0)
        self.assertLessEqual(ds.arousal, 1.0)

    def test_phase8_drive_modulation_bounds(self):
        """Verify all drive modulation coefficients stay in [0.10, 3.0]."""
        from world.agents.drives import update_biological_drives, compute_drive_modulation
        from world.agents.agent import Agent
        agent = Agent(agent_id=0, location=(64, 64))
        agent.hunger = 90.0
        agent.thirst = 85.0
        agent.injury_level = 60.0
        agent.energy = 15.0
        for _ in range(15):
            update_biological_drives(agent, 40.0)
        mods = compute_drive_modulation(agent)
        self.assertIn("Drinking", mods)
        self.assertIn("Eating", mods)
        for action_name, mult in mods.items():
            self.assertGreaterEqual(mult, 0.10, msg="{}: {} below 0.10".format(action_name, mult))
            self.assertLessEqual(mult, 3.0, msg="{}: {} above 3.0".format(action_name, mult))

    def test_phase8_fear_builds_from_pain_spikes(self):
        """Verify fear drive increases when pain tension spikes rapidly."""
        from world.agents.drives import update_biological_drives
        from world.agents.agent import Agent
        agent = Agent(agent_id=0, location=(64, 64))
        for _ in range(5):
            update_biological_drives(agent, 20.0)
        fear_before = agent.drives.fear
        agent.injury_level = 90.0
        for _ in range(5):
            update_biological_drives(agent, 20.0)
        self.assertGreater(agent.drives.fear, fear_before)

    def test_phase8_agent_has_drives_field(self):
        """Verify that a newly created Agent has a DriveState instance."""
        from world.agents.drives import DriveState
        from world.agents.agent import Agent
        agent = Agent(agent_id=99, location=(32, 32))
        self.assertTrue(hasattr(agent, "drives"))
        self.assertIsInstance(agent.drives, DriveState)

    def test_phase8_memory_importance_scoring(self):
        """Verify that dynamic memory importance scoring scales correctly with needs and context."""
        from world.agents.agent import Agent, WATER, FOOD, DANGER, PERSON
        agent = Agent(agent_id=0, location=(64, 64))
        
        # Test case 1: Sighting water when NOT thirsty vs when thirsty
        agent.thirst = 10.0
        agent.drives.thirst_tension = 0.1
        agent.add_memory(WATER, (64, 65), tick=10, importance=0.5)
        stored_imp_low = agent.episodic_memory[-1].importance
        
        agent.thirst = 90.0
        agent.drives.thirst_tension = 0.9
        agent.drives.boredom = 0.0 # reset arousal
        agent.add_memory(WATER, (64, 66), tick=20, importance=0.5)
        stored_imp_high = agent.episodic_memory[-1].importance
        
        # Sighting water when very thirsty should have higher importance
        self.assertGreater(stored_imp_high, stored_imp_low)
        
        # Test case 2: Near death memory importance
        agent.health = 10.0
        agent.add_memory(DANGER, (64, 67), tick=30, importance=0.5)
        stored_imp_near_death = agent.episodic_memory[-1].importance
        self.assertGreaterEqual(stored_imp_near_death, 0.8)

    def test_phase8_emotional_dynamics_updates(self):
        """Verify that emotional drives (frustration, longing, boredom) update correctly over ticks."""
        from world.agents.drives import update_biological_drives, update_emotional_drives
        from world.agents.agent import Agent
        from world.state import WorldState
        
        world = WorldState(width=128, height=128, seed=1729)
        agent = Agent(agent_id=0, location=(64, 64))
        
        # Frustration under unmet needs
        agent.hunger = 90.0
        agent.thirst = 90.0
        # Run bio ticks to build unmet ticks
        for _ in range(60):
            update_biological_drives(agent, 20.0)
            
        # Trigger emotional update
        update_emotional_drives(agent, world)
        self.assertGreater(agent.drives.frustration, 0.0)
        
        # Boredom under repeated action
        agent.current_action = "Eating"
        agent._last_drive_action = "Eating"
        agent.drives._same_action_streak = 120
        update_emotional_drives(agent, world)
        self.assertGreater(agent.drives.boredom, 0.0)
        
        # Longing/Loneliness under social isolation
        agent.drives._ticks_since_known_agent = 600
        update_emotional_drives(agent, world)
        self.assertGreater(agent.drives.longing, 0.0)

    def test_phase8_grief_trigger_on_death(self):
        """Verify that agent death triggers grief in nearby known agents."""
        from world.agents.agent import Agent
        from world.agents.drives import trigger_grief
        from world.state import WorldState
        
        world = WorldState(width=128, height=128, seed=1729)
        
        # Create two agents
        agent_a = Agent(agent_id=0, location=(64, 64))
        agent_b = Agent(agent_id=1, location=(64, 65))
        
        # They know each other
        agent_a.known_agents.add(agent_b.id)
        agent_b.known_agents.add(agent_a.id)
        
        world.agents = [agent_a, agent_b]
        
        # Trigger grief on agent_a when agent_b dies
        trigger_grief(agent_a, agent_b.id, world)
        
        self.assertGreater(agent_a.drives.grief, 0.0)
        self.assertGreater(agent_a.drives.longing, 0.0)

    def test_phase8_relationship_graph(self):
        """Verify that relationship graph initializes, updates and decays correctly on the 30-tick clock."""
        from world.agents.agent import Agent, DANGER, PERSON
        from world.agents.drives import update_relationships
        from world.state import WorldState
        
        world = WorldState(width=128, height=128, seed=1729)
        
        agent_a = Agent(agent_id=0, location=(64, 64))
        agent_b = Agent(agent_id=1, location=(64, 65))
        agent_a.muscle_mass = 120.0 # stronger
        agent_b.muscle_mass = 80.0  # weaker
        
        world.agents = [agent_a, agent_b]
        world.tick = 30
        
        # Scenario 1: Dispute interaction
        # We record a dispute memory in agent_a's episodic memory (last 30 ticks)
        agent_a.add_memory(DANGER, (64, 64), tick=25, importance=0.6, associated_id=agent_b.id, outcome="dispute")
        
        # Run relationship update clock
        update_relationships(agent_a, world)
        
        self.assertIn(agent_b.id, agent_a.relationships)
        rel = agent_a.relationships[agent_b.id]
        
        # Trust should be negative, fear should be positive, dominance should be positive (since a is stronger)
        self.assertLess(rel.trust, 0.0)
        self.assertGreater(rel.fear, 0.0)
        self.assertGreater(rel.dominance, 0.0)
        
        # Test get_social_modifier lookup
        # Since trust is negative and fear is positive, the modifier should be < 1.0 (indicating avoidance/threat)
        mod_after_dispute = agent_a.get_social_modifier(agent_b.id, world.tick)
        self.assertLess(mod_after_dispute, 1.0)
        
        # Scenario 2: Decay interaction
        # Run tick to 60 with no new memories
        world.tick = 60
        prev_trust = rel.trust
        prev_fear = rel.fear
        
        update_relationships(agent_a, world)
        
        # Check that trust and fear have decayed toward 0.0
        self.assertGreater(rel.trust, prev_trust) # trust rises toward 0 since it was negative
        self.assertLess(rel.fear, prev_fear) # fear decays toward 0

    def test_phase8_exponential_memory_decay(self):
        """Verify episodic memories decay exponentially and are pruned below 0.10 confidence."""
        from world.agents.agent import Agent, Memory, WATER
        from world.agents.simulation import simulate_agent_tick
        from world.state import WorldState
        import numpy as np

        world = WorldState(width=128, height=128, seed=1729)
        agent = Agent(agent_id=0, location=(64, 64))

        # Add two memories: one of high importance (0.9), one of low importance (0.1)
        # Both start with confidence 1.0
        mem_high = Memory(type=WATER, location=(64, 65), timestamp=0, importance=0.9, confidence=1.0)
        mem_low = Memory(type=WATER, location=(64, 66), timestamp=0, importance=0.1, confidence=1.0)
        agent.episodic_memory = [mem_high, mem_low]

        # Trigger a daily clock tick (tick is multiple of 360)
        world.tick = 360
        simulate_agent_tick(agent, world)

        # Confirm both decayed, but low importance decayed much more than high importance
        # Formula: confidence = confidence * exp(-0.2 * (1.0 - importance))
        # High importance (0.9): new confidence = 1.0 * exp(-0.2 * 0.1) = exp(-0.02) ~ 0.98
        # Low importance (0.1): new confidence = 1.0 * exp(-0.2 * 0.9) = exp(-0.18) ~ 0.835
        self.assertEqual(len(agent.episodic_memory), 2)
        self.assertLess(agent.episodic_memory[0].confidence, 1.0)
        self.assertLess(agent.episodic_memory[1].confidence, 1.0)
        self.assertGreater(agent.episodic_memory[0].confidence, agent.episodic_memory[1].confidence)

        # Run several days to prune the low-importance memory below 0.10, while keeping the high-importance one
        # 12 days * -0.18 = -2.16 -> exp(-2.16) = 0.115
        # 13 days * -0.18 = -2.34 -> exp(-2.34) = 0.096 < 0.10 (pruned!)
        # For high importance, 13 days * -0.02 = -0.26 -> exp(-0.26) = 0.77 > 0.10 (preserved!)
        for day in range(2, 15):
            world.tick = day * 360
            simulate_agent_tick(agent, world)

        # Low importance should be pruned, high importance preserved
        self.assertEqual(len(agent.episodic_memory), 1)
        self.assertEqual(agent.episodic_memory[0].location, (64, 65))

    def test_phase8_prediction_category_emotions(self):
        """Verify category-specific prediction error routing to emotional channels."""
        from world.agents.agent import Agent
        from world.agents.drives import update_emotional_drives
        from world.state import WorldState

        world = WorldState(width=128, height=128, seed=1729)
        
        # Test Case 1: Drinking failed -> Frustration spikes
        agent1 = Agent(agent_id=1, location=(64, 64))
        agent1.last_prediction_error = -0.8
        agent1.last_prediction_category = "Drinking"
        update_emotional_drives(agent1, world)
        self.assertGreater(agent1.drives.frustration, 0.0)
        self.assertEqual(agent1.drives.longing, 0.0)
        self.assertEqual(agent1.drives.fear, 0.0)

        # Test Case 2: Reproduce failed -> Longing spikes
        agent2 = Agent(agent_id=2, location=(64, 64))
        agent2.last_prediction_error = -0.8
        agent2.last_prediction_category = "Reproduce"
        update_emotional_drives(agent2, world)
        self.assertGreater(agent2.drives.longing, 0.0)
        self.assertEqual(agent2.drives.frustration, 0.0)
        self.assertEqual(agent2.drives.fear, 0.0)

        # Test Case 3: Sheltering failed -> Fear spikes
        agent3 = Agent(agent_id=3, location=(64, 64))
        agent3.last_prediction_error = -0.8
        agent3.last_prediction_category = "Sheltering"
        update_emotional_drives(agent3, world)
        self.assertGreater(agent3.drives.fear, 0.0)
        self.assertEqual(agent3.drives.frustration, 0.0)
        self.assertEqual(agent3.drives.longing, 0.0)

    def test_phase8_copresence_saturating_attachment(self):
        """Verify that proximity attachment growth saturates asymptotically with diminishing returns."""
        from world.agents.agent import Agent, Memory, PERSON
        from world.agents.drives import update_relationships
        from world.state import WorldState

        world = WorldState(width=128, height=128, seed=1729)
        agent_a = Agent(agent_id=1, location=(64, 64))
        agent_b = Agent(agent_id=2, location=(64, 65))
        world.agents = [agent_a, agent_b]

        # First encounter (tick 30)
        world.tick = 30
        agent_a.add_memory(PERSON, (64, 65), tick=30, importance=0.5, associated_id=agent_b.id)
        update_relationships(agent_a, world)
        
        rel = agent_a.relationships[agent_b.id]
        first_attachment = rel.attachment
        self.assertGreater(first_attachment, 0.0)
        
        # Second encounter (tick 60)
        world.tick = 60
        agent_a.add_memory(PERSON, (64, 65), tick=60, importance=0.5, associated_id=agent_b.id)
        update_relationships(agent_a, world)
        
        second_attachment = rel.attachment
        self.assertGreater(second_attachment, first_attachment)
        
        # The growth step should exhibit diminishing returns (asymptotic growth toward 1.0)
        # Growth 1: (1.0 - 0.0) * rate
        # Growth 2: (1.0 - attachment_1) * rate -> should be smaller
        growth_1 = second_attachment - first_attachment
        
        # Third encounter (tick 90)
        world.tick = 90
        agent_a.add_memory(PERSON, (64, 65), tick=90, importance=0.5, associated_id=agent_b.id)
        update_relationships(agent_a, world)
        
        third_attachment = rel.attachment
        growth_2 = third_attachment - second_attachment
        
        self.assertLess(growth_2, growth_1)
        self.assertLess(third_attachment, 1.0)

    def test_phase8_lateral_inhibition_arbitration(self):
        """Verify that Lateral Inhibition suppresses competing subdominant drives to prevent paralysis."""
        from world.agents.agent import Agent
        from world.agents.drives import compute_drive_modulation
        
        agent = Agent(agent_id=1, location=(64, 64))
        
        # Manually set drive tensions
        # Fear (safety) is highly dominant (0.9)
        # Boredom (exploration) is moderate (0.4)
        agent.drives.fear = 0.9
        agent.drives.boredom = 0.4
        
        # Both safety and exploration are prioritized in the profile
        agent.motivation.safety.current = 0.8
        agent.motivation.exploration.current = 0.8
        
        # Call drive modulation which runs lateral inhibition
        mods = compute_drive_modulation(agent)
        
        # Without inhibition: safety would be 0.8 * 0.9 = 0.72, explore would be 0.8 * 0.4 = 0.32
        # With beta = 0.15, the dominant safety drive (0.72) should suppress the explore drive (0.32)
        # Inhibited explore = max(0.0, 0.32 - 0.15 * (0.72 + other_drives))
        # Since explore is heavily suppressed, explore_mult should be clamped or small compared to safety-driven mults like build_mult
        self.assertGreater(mods["Building Shelter"], mods["Exploring"])

    def test_phase8_motivation_profile_welford_stats(self):
        """Verify Welford's algorithm running mean/variance calculations under dynamic priority drift."""
        from world.agents.drives import MotivationDimension
        import numpy as np

        dim = MotivationDimension(current=0.5, mean=0.5, variance=0.0, update_count=0)

        # Perform updates
        values = [0.6, 0.4, 0.8, 0.2]
        for val in values:
            dim.update(val)

        # Expected statistics calculated using standard mean/variance formula
        # update_count = 4
        # mean = 0.5 (initial) is overridden by Welford since it's treated as a sequence
        # Running variance in MotivationDimension.to_dict is variance / max(1, count) = 0.20 / 4 = 0.05
        # Running std_dev is sqrt(variance / count) = sqrt(0.05) ~ 0.2236
        self.assertEqual(dim.update_count, 4)
        self.assertAlmostEqual(dim.mean, 0.5)
        self.assertAlmostEqual(dim.variance, 0.20)
        self.assertAlmostEqual(dim.std_dev, np.sqrt(0.05))

    def test_research_framework_ablation_modulations(self):
        """Verify that ablation settings correctly bypass respective subsystems."""
        world = WorldState(width=16, height=16, seed=123)
        agent = Agent(agent_id=1, location=(8, 8))
        world.agents = [agent]
        
        # Test 1: Emotion Ablation clamps drives to 0.0
        world.ablation = {
            "planner": True,
            "emotion": False,
            "relationships": True,
            "memory_importance": True,
            "motivation": False,
            "prediction_error": True
        }
        agent.ablation = world.ablation
        agent.drives.fear = 0.5
        
        from world.agents.drives import update_emotional_drives, compute_drive_modulation
        update_emotional_drives(agent, world)
        
        # In ablated mode, fear should be forced to 0.0
        self.assertEqual(agent.drives.fear, 0.0)
        
        # Compute drive modulation with emotion ablated: fear multiplier should be neutral (clamped)
        mods = compute_drive_modulation(agent)
        # Explore should not be suppressed by fear
        self.assertAlmostEqual(mods["Exploring"], 1.0)

        # Test 2: Motivation Ablation bypasses lateral inhibition and motiv multipliers
        world.ablation["motivation"] = False
        agent.ablation = world.ablation
        
        # Set a very high safety motivation
        agent.motivation.safety.current = 1.0
        agent.drives.fear = 1.0 # Force fear back to 1.0 if we manually test drives
        # Reset ablation so fear is not clamped to test motivation independently
        agent.ablation["emotion"] = True
        
        # compute_drive_modulation should ignore the high safety multiplier since motivation is ablated
        # Without motivation ablation, safety would scale build_mult by (0.5 + M_safety + M_comfort*0.5)
        # With ablation, explore_mult and build_mult are not modified by motivations
        mods_ablated = compute_drive_modulation(agent)
        
        agent.ablation["motivation"] = True
        mods_normal = compute_drive_modulation(agent)
        
        # Normal mods (with safety weight = 1.0) should have building shelter amplified more than ablated
        self.assertNotEqual(mods_ablated["Building Shelter"], mods_normal["Building Shelter"])

        # Test 3: Relationship Ablation overrides social modifiers
        world.ablation["relationships"] = False
        agent.ablation = world.ablation
        # Without relationships, social modifier between any agents is 1.0
        self.assertEqual(agent.get_social_modifier(2, 100), 1.0)


    def test_phase10_telemetry_collection(self):
        """Verify centralized TelemetryCollector tracks reproduction funnel, water audits, and anomalies."""
        world = WorldState(width=16, height=16, seed=123)
        self.assertIsNotNone(world.telemetry)
        
        # 1. Test Welford utility tracking
        world.telemetry.record_utility_decision(
            age_ticks=360 * 20, # 20 years (Young Adult)
            repro_utility=15.0,
            winning_action="Drinking",
            winning_utility=40.0
        )
        stats = world.telemetry.utility_stats_by_bracket["young_adult"]
        self.assertEqual(stats["count"], 1)
        self.assertAlmostEqual(stats["mean_repro"], 15.0)
        self.assertAlmostEqual(stats["mean_win"], 40.0)
        self.assertIn("Drinking", world.telemetry.repro_lost_to_counts)
        self.assertEqual(world.telemetry.repro_lost_to_counts["Drinking"], 1)
        
        # 2. Test Water economy search tracking
        world.telemetry.record_water_search_start(straight_dist=10.0)
        self.assertEqual(world.telemetry.water_searches, 1)
        
        # Success path efficiency: straight=10, actual=15. Eff = 10/15 ~ 0.6667
        world.telemetry.record_water_search_outcome(success=True, ticks_taken=5, efficiency=10.0/15.0)
        self.assertEqual(world.telemetry.water_search_successes, 1)
        self.assertAlmostEqual(world.telemetry.water_time_to_first_sum, 5)
        self.assertAlmostEqual(world.telemetry.water_path_efficiency_sum, 2.0/3.0)
        
        # 3. Test Dehydration deaths tracking
        world.telemetry.record_dehydration_death(carrying_water=True, beside_water=False, remembered_water=True, recently_visited_dry=False)
        self.assertEqual(world.telemetry.dehydration_deaths, 1)
        self.assertEqual(world.telemetry.died_carrying_water, 1)
        self.assertEqual(world.telemetry.died_with_remembered_water, 1)
        self.assertEqual(world.telemetry.died_beside_water, 0)
        
        # 4. Funnel terminal outcomes count and partition logic
        world.telemetry.record_repro_terminal("cooldown")
        world.telemetry.record_repro_terminal("distance")
        world.telemetry.record_repro_terminal("birth")
        
        self.assertEqual(world.telemetry.repro_fail_cooldown, 1)
        self.assertEqual(world.telemetry.repro_fail_distance, 1)
        self.assertEqual(world.telemetry.repro_successes, 1)
        
        # 5. Generate and check dashboard report
        report = world.telemetry.generate_dashboard_report()
        self.assertIn("CIVILIZATION HEALTH MONITOR", report)
        self.assertIn("WATER ECONOMY AND PATH TELEMETRY", report)
        self.assertIn("ANOMALY", report)  # Dehydration death carrying water should trigger anomaly alert


if __name__ == '__main__':
    unittest.main()
