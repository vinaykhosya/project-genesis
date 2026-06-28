from dataclasses import dataclass
import numpy as np
from .genetics import Genome, neutral_genome, express_genome
from .cognitive import Predictor, Concept, Procedure
from .drives import DriveState, Relationship, MotivationProfile


# Memory / Knowledge type constants
WATER    = "WATER"
FOOD     = "FOOD"
DANGER   = "DANGER"
PERSON   = "PERSON"
LANDMARK = "LANDMARK"


@dataclass
class Memory:
    """A single episodic memory event experienced by an agent."""
    type:       str    # WATER, FOOD, DANGER, PERSON, LANDMARK
    location:   tuple  # (y, x)
    timestamp:  int    # World tick when recorded
    importance: float  # [0.0, 1.0]
    confidence: float  # [0.0, 1.0]
    associated_id: int = -1
    outcome:       str = "neutral"  # "neutral", "dispute", "share", "mate"


class Knowledge:
    """Generalised spatial knowledge derived from episodic memory."""
    def __init__(self):
        self.water_sources    = {}   # (y, x): confidence dict
        self.food_sources     = {}   # (y, x): confidence dict
        self.danger_locations = {}   # (y, x): confidence float


class Agent:
    """
    Simulated entity whose behaviour emerges from competing needs, restricted
    perception, episodic memory, and the brain parameters expressed by its genome.

    Phase 5 additions
    -----------------
    - genome: 13-gene Genome instance driving all brain parameters
    - 4 life stages: Infant (0–3 yr) / Juvenile (3–14 yr) / Adult (14–60 yr) / Elder (60+ yr)
    - Colony affiliation, generation lineage, reproduction cooldown
    - Stored food / water for colony resource pressure
    - behavior_cluster: assigned post-hoc by K-Means (not a spawning label)
    """

    def __init__(self, agent_id: int, location: tuple,
                 traits: dict = None, genome: Genome = None, reconstruct: bool = False):
        self.id       = agent_id
        self.location = location          # (y, x) coordinates
        self.home_location = location

        # ------------------------------------------------------------------ #
        # Genome / Brain parameters (Phase 5)                                 #
        # If genome is provided, brain params are derived from it.            #
        # If traits are provided (old API / tests), a neutral genome is used  #
        # so that memory_decay_rate, vision_radius etc. remain deterministic. #
        # ------------------------------------------------------------------ #
        if genome is not None:
            self.genome = genome
        elif traits is not None:
            # Backward-compat path: explicit traits → neutral genome
            self.genome = neutral_genome()
        else:
            # Default path: neutral genome to ensure deterministic baseline in unit tests
            self.genome = neutral_genome()

        self.brain = express_genome(self.genome)
        brain = self.brain

        # Traits dict kept for backward compatibility with older decision code.
        # Values are derived from genome expression; explicit traits dict overrides.
        if traits is not None:
            self.traits = traits
        else:
            self.traits = {
                "curiosity":      float(self.genome.genes[6]),          # g_novelty_seeking
                "risk_tolerance": float(1.0 - self.genome.genes[10]),   # inverse of g_risk_sensitivity
            }
        self.base_traits = self.traits.copy()
        # ------------------------------------------------------------------ #
        # Age & Lifespan                                                       #
        # ------------------------------------------------------------------ #
        if reconstruct:
            self.age = 0
            self.max_age = 0
        else:
            start_age_years = float(np.random.uniform(5, 45))
            self.age     = int(start_age_years * 360)
            base_max_age = int(max(start_age_years + 15.0, np.random.normal(80, 15)) * 360)
            self.max_age = max(20 * 360, base_max_age + brain["max_age_offset"])

        # ------------------------------------------------------------------ #
        # Physiological state                                                  #
        # ------------------------------------------------------------------ #
        self.health  = 100.0
        self.dead    = False
        self.cause_of_death   = None
        self.primary_cause    = None
        self.secondary_cause  = None

        self.fat_reserves  = 100.0
        self.muscle_mass   = 100.0
        self.injury_level  = 0.0

        # ------------------------------------------------------------------ #
        # Combat state                                                         #
        # ------------------------------------------------------------------ #
        # Ticks before this agent will re-engage with combat_fear_target.
        # Proportional to injury level at time of loss — hardcodes the existence
        # of a fear state, but the duration is injury+genome-driven.
        self.combat_fear_timer  = 0
        self.combat_fear_target = None   # agent_id of the specific aggressor being avoided

        # Genome-derived territory radius. High-aggression animals defend larger areas.
        # Neutral genome (aggression_mult=1.0): radius = 25 cells.
        self.home_radius = int(15 + brain["aggression_mult"] * 10)

        # Lifetime combat outcome counters (for telemetry and future memory modulation)
        self.fight_wins   = 0
        self.fight_losses = 0

        # Shelter
        self.shelter_location   = None
        self.shelter_level      = 0    # 0: None | 1: Primitive | 2: Wood | 3: Stone
        self.shelter_durability = 0.0

        # Damage accumulators for cause-of-death ranking
        self.starvation_damage_accumulated  = 0.0
        self.dehydration_damage_accumulated = 0.0
        self.exposure_damage_accumulated    = 0.0
        self.injury_damage_accumulated      = 0.0
        self.age_damage_accumulated         = 0.0

        # Needs (0.0 = fully satisfied, 100.0 = critical)
        self.hunger       = 0.0
        self.thirst       = 0.0
        self.energy       = 100.0    # 100 = fully rested, 0 = collapsed
        self.curiosity_need = 0.0

        # ------------------------------------------------------------------ #
        # Colony & Lineage (Phase 5)                                          #
        # ------------------------------------------------------------------ #
        self.colony_id    = 0         # Integer colony index (0–3)
        self.generation   = 0         # Birth generation (founders = 0)
        self.parent_ids   = None      # (parent_a_id, parent_b_id) or None for founders
        self.children_ids = []        # List of child agent IDs
        self.born_tick    = 0         # World tick at birth

        # Cognitive Memory & Learning
        self.learning_rate         = float(brain.get("learning_rate", 0.05))
        self.prediction_confidence = 1.0
        self.feature_weights       = np.zeros(5, dtype=np.float32)  # [temp, elev, rain, food, water]
        self.encounters            = []  # Required for backward-compat (unused but kept)

        # Reproduction cooldown (ticks until eligible again; 0 = ready)
        self.reproduction_cooldown = 0

        # Colony resource stores (personal carry capacity)
        self.stored_food  = 0.0      # Units [0, 100]
        self.stored_water = 0.0      # Units [0, 100]

        # Behavioral cluster label assigned post-hoc by K-Means (e.g. "C0")
        self.behavior_cluster = "C0"

        # Internal flag: set by step_toward when agent reaches a mate location
        self._wants_to_reproduce_with = None  # target location tuple or None

        # ------------------------------------------------------------------ #
        # Spatial bookkeeping                                                  #
        # ------------------------------------------------------------------ #
        self.spawn_location = location
        self.max_radius     = 0.0

        # ------------------------------------------------------------------ #
        # Emergence metrics and Recognition                                    #
        # ------------------------------------------------------------------ #
        self.archetype = "Balanced"  # Kept for export compat; no longer used for spawning
        self.known_agents       = set()
        self.drinks_count       = 0
        self.eats_count         = 0
        self.resting_ticks      = 0
        self.discoveries_count  = 0
        self.spawn_biome        = "Unknown"
        self.discovered_landmarks = set()
        self.action_counts = {
            "Eating": 0, "Drinking": 0, "Resting": 0, "Exploring": 0,
            "Idle": 0, "Building Shelter": 0, "Sheltering": 0,
            "Store Food": 0, "Store Water": 0,
            "Share Food": 0, "Share Water": 0, "Reproduce": 0,
        }

        # Phase 3 Adaptation Metrics
        self.failed_water_visits  = 0
        self.failed_food_visits   = 0
        self.nodes_added_count    = 0
        self.nodes_removed_count  = 0

        # Phase 7.0 Cognitive Profiling Counters
        self.decision_evals       = 0
        self.predictor_calls      = 0
        self.perception_calls     = 0
        self.action_changes       = 0
        self.memories_searched    = 0
        self.targets_evaluated    = 0
        self.meaningful_decisions = 0
        self.relationship_lookups = 0
        self.memory_cache_hits    = 0
        self.memory_cache_misses  = 0
        self.total_prediction_error = 0.0
        self.prediction_count = 0

        # Phase 3.5 Lifetime Discovery & Wisdom Telemetry
        self.discovered_water = set()
        self.discovered_food  = set()
        self.rediscoveries    = 0

        self.ticks_survived      = 0
        self.season_observations = {0: 0, 1: 0, 2: 0, 3: 0}

        # Predictive Telemetry
        self.prediction_attempts    = 0
        self.prediction_successes   = 0
        self.prediction_decisions   = 0
        self.prediction_gains       = 0
        self.predicted_destination  = None
        self.prediction_enabled     = True

        # Enrich initial coordinate for path history: [x, y, action_id, health, hunger, thirst, energy, generation]
        self.sampled_path_history = [[
            int(location[1]), # x (col)
            int(location[0]), # y (row)
            0,                # action_id: Idle (0)
            float(self.health),
            float(self.hunger),
            float(self.thirst),
            float(self.energy),
            int(self.generation)
        ]]


        # Memory & Knowledge
        self.knowledge      = Knowledge()
        self.episodic_memory = []
        self.visited_chunks = set()   # set of (chunk_y, chunk_x) where chunk_size = 32

        # Cognitive Fields
        self.predictor = Predictor(reconstruct=reconstruct)
        self.concepts = {WATER: [], FOOD: []}
        self.procedures = []
        self.action_history = []  # stores recent action names (last 3)

        # Phase 7: Event-Driven Cognition & Hierarchical Planning
        self.current_goal           = None  # Strategic Goal
        self.current_target         = None  # Tactical Target
        self.action_queue           = []    # Motor Action Queue: list of (action_name, target_coord)
        self.plan_commitment        = 0.5   # Commitment level (0.0 to 1.0)
        self.plan_age               = 0     # Number of ticks since last planning
        self.planning_accuracy      = 1.0   # Metacognitive planning self-trust
        self.prediction_accuracy    = 1.0   # Metacognitive prediction self-trust
        self.memory_accuracy        = 1.0   # Metacognitive memory self-trust
        self.plan_cache             = {}    # Plan caching dict
        self.utility_cache          = {}    # Utility values cache dict
        self.memory_cache           = {     # Memory retrieval cache for best targets
            "Best Water": None,
            "Best Food": None,
            "Best Shelter": None,
            "Best Mate": None,
            "last_cache_tick": -1
        }


        # Telemetry State Tracking Fields
        self.last_health = 100.0
        self.last_hunger = 0.0
        self.last_thirst = 0.0
        self.last_injury_level = 0.0
        self.last_children_count = 0

        # ---------------------------------------------------------------------- #
        # Phase 8: Unified Motivational Drive Architecture                        #
        # ---------------------------------------------------------------------- #

        # Biological drive tensions (Phase 8.1) — EMA-smoothed internal pressures
        self.drives = DriveState()

        # Per-dyad relationship records (Phase 8.3 will populate these)
        self.relationships: dict = {}   # int agent_id → Relationship

        # Supporting: last action seen by the drive system (for streak tracking)
        self._last_drive_action: str = "Idle"

        # Phase 8.4: Prediction error registers
        self.last_prediction_error = 0.0
        self.last_prediction_category = "Idle"

        # Motivation Profile
        self.motivation = MotivationProfile()
        self.ablation = {}

        # Initialize motivational dimensions from phenotypic genome mappings
        genes = self.genome.genes
        self.motivation.safety.current = float(genes[10])      # G_RISK_SENSITIVITY
        self.motivation.safety.mean = float(genes[10])
        self.motivation.family.current = float(0.5 * (genes[7] + genes[9]))  # Proximity + Sharing
        self.motivation.family.mean = float(0.5 * (genes[7] + genes[9]))
        self.motivation.exploration.current = float(genes[6]) # G_NOVELTY_SEEKING
        self.motivation.exploration.mean = float(genes[6])
        self.motivation.knowledge.current = float(genes[13])   # G_LEARNING_RATE
        self.motivation.knowledge.mean = float(genes[13])
        self.motivation.comfort.current = float(genes[1])     # G_THERMOREGULATION
        self.motivation.comfort.mean = float(genes[1])
        self.motivation.dominance.current = float(genes[8])   # G_AGGRESSION
        self.motivation.dominance.mean = float(genes[8])

        # Event Tracking Flags
        self._consumed_food = False
        self._consumed_water = False
        self._shelter_upgraded = False
        self._failed_visit = False
        self._stepped = False

        # Visual trace & status
        self.current_action    = "Idle"
        self.target_coordinate = None
        self.path_history      = [location]

        # Phase 10 Telemetry Tracking
        self.water_search_start_tick = None
        self.water_search_start_location = None
        self.water_search_actual_distance = 0.0
        self.visited_dry_water_recently = False
        self.repro_terminal_outcome = None

    def recompute_brain(self):
        """Re-expresses the genome to update the agent's brain parameters."""
        self.brain = express_genome(self.genome)

    # ---------------------------------------------------------------------- #
    # Life Stage (Phase 5)                                                    #
    # Four stages gate available actions and scale physiology.                #
    # ---------------------------------------------------------------------- #

    @property
    def life_stage(self) -> str:
        """
        Returns the current life stage based on age in years.
        Stages gate available actions and modify physiology:
          Infant   (0–3 yr)   — slow, limited vision, high growth
          Juvenile (3–14 yr)  — growing, no reproduction
          Adult    (14–60 yr) — full capability, can reproduce
          Elder    (60+ yr)   — senescence applies, cannot reproduce
        """
        age_years = self.age / 360.0
        if age_years < 3.0:
            return "Infant"
        elif age_years < 14.0:
            return "Juvenile"
        elif age_years < 60.0:
            return "Adult"
        else:
            return "Elder"

    # ---------------------------------------------------------------------- #
    # Derived physiological properties                                         #
    # ---------------------------------------------------------------------- #

    @property
    def senescence_factor(self) -> float:
        """Continuous aging factor in the last 30% of lifespan, [0.0, 1.0]."""
        threshold = 0.7 * self.max_age
        if self.age < threshold:
            return 0.0
        return float(np.clip((self.age - threshold) / (0.3 * self.max_age), 0.0, 1.0))

    @property
    def effective_risk(self) -> float:
        """
        Effective risk tolerance, modulated by life stage and senescence.
        Infants and Juveniles are more impulsive (higher risk, +30%).
        Elders become more cautious (risk drops with senescence).
        """
        base_risk = self.base_traits.get("risk_tolerance", 0.5)
        r = self.senescence_factor
        stage = self.life_stage
        if stage in ("Infant", "Juvenile"):
            risk = float(np.clip(base_risk * 1.3, 0.0, 1.0))
        else:
            risk = float(base_risk)
        if r > 0.0:
            risk *= (1.0 - 0.5 * r)   # Elders grow more cautious
        return float(np.clip(risk, 0.0, 1.0))

    @property
    def effective_curiosity(self) -> float:
        """
        Effective curiosity, amplified in Juveniles (+20%) and
        gradually reduced with senescence in Elders.
        """
        base_curio = self.base_traits.get("curiosity", 0.5)
        r = self.senescence_factor
        stage = self.life_stage
        if stage in ("Infant", "Juvenile"):
            curio = float(np.clip(base_curio * 1.2, 0.0, 1.0))
        else:
            curio = float(base_curio)
        if r > 0.0:
            curio *= (1.0 - 0.5 * r)
        return float(np.clip(curio, 0.0, 1.0))

    @property
    def effective_speed(self) -> float:
        """
        Movement speed multiplier, shaped by life stage, senescence,
        muscle mass, and injury level.

        Life stage base speeds
        ----------------------
        Infant   → 0.3×  (very small, dependent)
        Juvenile → 0.6×  (growing)
        Adult    → 1.0×  (full mobility)
        Elder    → 1.0×  base but reduced by senescence factor
        """
        stage = self.life_stage
        r = self.senescence_factor

        if stage == "Infant":
            speed = 0.3
        elif stage == "Juvenile":
            speed = 0.6
        else:
            speed = 1.0   # Adult or Elder (senescence applied below)

        # Aging penalty: up to 40% reduction in the Elder stage
        if r > 0.0:
            speed *= (1.0 - 0.4 * r)

        # Muscle mass penalty
        speed *= (self.muscle_mass / 100.0)

        # Injury penalty (up to 50% reduction)
        speed *= (1.0 - 0.5 * (self.injury_level / 100.0))

        return float(np.clip(speed, 0.05, 1.0))

    @property
    def vision_radius(self) -> int:
        """
        Effective vision radius in grid cells, derived from the g_vision gene
        and modulated by life stage and senescence.

        Life stage vision modifiers
        ---------------------------
        Infant   → 0.5× genome base vision
        Juvenile → 0.75× genome base vision
        Adult    → full genome base vision (10–30 cells)
        Elder    → genome base vision × (1 − 0.3 × senescence_factor)
        """
        base_vision = self.brain["vision_radius"]  # 10 – 30 cells
        r = self.senescence_factor
        stage = self.life_stage

        if stage == "Infant":
            return max(5, int(base_vision * 0.5))
        elif stage == "Juvenile":
            return max(8, int(base_vision * 0.75))
        elif stage == "Adult":
            return base_vision
        else:  # Elder
            return max(8, int(base_vision * (1.0 - 0.3 * r)))

    @property
    def years_survived(self) -> int:
        return self.ticks_survived // 360

    # ---------------------------------------------------------------------- #
    # Memory system                                                            #
    # ---------------------------------------------------------------------- #

    def add_memory(self, mem_type: str, location: tuple, tick: int, importance: float, associated_id: int = -1, outcome: str = "neutral"):
        """Appends an episodic memory and upserts the generalised knowledge pools."""
        # Phase 8.15 Dynamic Memory Importance Scoring
        if getattr(self, "ablation", {}).get("memory_importance", True):
            try:
                from .drives import compute_memory_importance
                
                # Check if this is the first encounter of this location/agent
                is_first = True
                if mem_type == WATER:
                    is_first = (location not in self.knowledge.water_sources)
                elif mem_type == FOOD:
                    is_first = (location not in self.knowledge.food_sources)
                elif mem_type == PERSON:
                    is_first = (associated_id >= 0 and associated_id not in self.known_agents)
                
                near_death = (self.health < 30.0)
                arousal = self.drives.arousal if hasattr(self, "drives") else 0.0
                
                dynamic_importance = compute_memory_importance(
                    agent=self,
                    mem_type=mem_type,
                    outcome=outcome,
                    near_death=near_death,
                    is_first_encounter=is_first,
                    emotional_intensity=arousal
                )
                importance = dynamic_importance
            except Exception:
                # Fallback to passed value if anything fails
                pass

        # Check if an episodic memory of this type and location already exists.
        # If so, update it and move it to the end to maintain chronological order,
        # avoiding duplicate entries and saving performance.
        existing_mem = None
        for idx_m, m in enumerate(self.episodic_memory):
            if m.type == mem_type and m.location == location and m.associated_id == associated_id:
                existing_mem = m
                self.episodic_memory.pop(idx_m)
                break

        if existing_mem is not None:
            existing_mem.timestamp = tick
            existing_mem.importance = max(existing_mem.importance, importance)
            existing_mem.confidence = 1.0
            existing_mem.outcome = outcome
            self.episodic_memory.append(existing_mem)
        else:
            mem = Memory(
                type=mem_type, location=location,
                timestamp=tick, importance=importance, confidence=1.0,
                associated_id=associated_id, outcome=outcome,
            )
            self.episodic_memory.append(mem)

            # Cap episodic memory at 200 entries based primarily on importance (lowest evicted first).
            # We use (importance, timestamp) to sort: absolute lowest importance is evicted first,
            # and age (timestamp) breaks ties. This prevents absolute tick growth from overwhelming
            # importance in the prune score.
            if len(self.episodic_memory) > 200:
                self.episodic_memory.sort(key=lambda m: (m.importance, m.timestamp))
                self.episodic_memory.pop(0)

        season_id = (tick % 360) // 90

        if mem_type == WATER:
            if location not in self.knowledge.water_sources:
                self.nodes_added_count += 1
                self.knowledge.water_sources[location] = {
                    "confidence": 1.0, "last_seen_tick": tick,
                    "season_seen": season_id,
                    "active_seasons": {0: 0, 1: 0, 2: 0, 3: 0},
                    "dry_seasons":    {0: 0, 1: 0, 2: 0, 3: 0},
                }
            node = self.knowledge.water_sources[location]
            node["confidence"]     = 1.0
            node["last_seen_tick"] = tick
            node["season_seen"]    = season_id
            node.setdefault("active_seasons", {0: 0, 1: 0, 2: 0, 3: 0})[season_id] += 1
            node.setdefault("dry_seasons",    {0: 0, 1: 0, 2: 0, 3: 0})

        elif mem_type == FOOD:
            if location not in self.knowledge.food_sources:
                self.nodes_added_count += 1
                self.knowledge.food_sources[location] = {
                    "confidence": 1.0, "last_seen_tick": tick,
                    "season_seen": season_id,
                    "active_seasons": {0: 0, 1: 0, 2: 0, 3: 0},
                    "dry_seasons":    {0: 0, 1: 0, 2: 0, 3: 0},
                }
            node = self.knowledge.food_sources[location]
            node["confidence"]     = 1.0
            node["last_seen_tick"] = tick
            node["season_seen"]    = season_id
            node.setdefault("active_seasons", {0: 0, 1: 0, 2: 0, 3: 0})[season_id] += 1
            node.setdefault("dry_seasons",    {0: 0, 1: 0, 2: 0, 3: 0})

        elif mem_type == DANGER:
            self.knowledge.danger_locations[location] = 1.0

    def get_social_modifier(self, other_id: int, current_tick: int) -> float:
        """
        Queries the pre-computed Relationship Graph (Phase 8.3) for blazing fast O(1)
        trust retrieval, falling back to legacy episodic memory scan if no record exists yet.
        """
        self.relationship_lookups += 1
        if getattr(self, "ablation", {}).get("relationships", True) == False:
            return 1.0

        # If a pre-computed relationship exists, use it immediately!
        if hasattr(self, "relationships") and other_id in self.relationships:
            rel = self.relationships[other_id]
            modifier = 1.0 + rel.trust - rel.fear * 0.4
            return float(np.clip(modifier, 0.0, 2.0))

        self.memories_searched += len(self.episodic_memory)
        mems = [m for m in self.episodic_memory if getattr(m, "associated_id", -1) == other_id]
        if not mems:
            return 1.0

        raw_acts = []
        for m in mems:
            age = max(0, current_tick - m.timestamp)
            act = m.importance * float(np.exp(-age / 2000.0))
            raw_acts.append((act, m))

        # Sort by activation descending and keep top 5
        raw_acts.sort(key=lambda x: x[0], reverse=True)
        top_5 = raw_acts[:5]

        total_raw = sum(act for act, _ in top_5)
        if total_raw == 0.0:
            return 1.0

        weighted_outcome = 0.0
        for act, m in top_5:
            norm_weight = act / total_raw
            outcome = getattr(m, "outcome", "neutral")
            if outcome == "dispute":
                val = -0.5
            elif outcome in ("share", "mate"):
                val = 0.3
            else:
                val = 0.0
            weighted_outcome += norm_weight * val

        # Scale final modifier by total activation strength
        modifier = 1.0 + weighted_outcome * min(2.0, total_raw * 1.5)
        return float(np.clip(modifier, 0.0, 2.0))

    def update_concepts(self, world):
        """
        Groups unique memory locations of WATER and FOOD into clusters
        in 3D feature space (elevation, temp, rain) using Leader-Follower clustering.
        If a cluster has support >= 10, forms or updates a Concept.
        """
        for r_type in (WATER, FOOD):
            # Gather unique locations from episodic memory
            mems = [m for m in self.episodic_memory if m.type == r_type]
            unique_locs = list({m.location for m in mems})
            if len(unique_locs) < 10:
                continue

            # Cluster unique locations in feature space (threshold = 0.25)
            groups = []
            for loc in unique_locs:
                y, x = loc
                feat = np.array([
                    world.elevation[y, x],
                    world.temperature[y, x] / 50.0,
                    world.rainfall[y, x] / 3000.0
                ], dtype=np.float32)

                placed = False
                for g in groups:
                    # centroid of group
                    centroid = np.mean([
                        np.array([world.elevation[ly, lx], world.temperature[ly, lx] / 50.0, world.rainfall[ly, lx] / 3000.0])
                        for ly, lx in g
                    ], axis=0)
                    if np.linalg.norm(feat - centroid) < 0.25:
                        g.append(loc)
                        placed = True
                        break
                if not placed:
                    groups.append([loc])

            # For each cluster with support >= 10, form/update concept
            for g in groups:
                if len(g) < 10:
                    continue

                elevs = np.array([world.elevation[ly, lx] for ly, lx in g], dtype=np.float32)
                temps = np.array([world.temperature[ly, lx] / 50.0 for ly, lx in g], dtype=np.float32)
                rains = np.array([world.rainfall[ly, lx] / 3000.0 for ly, lx in g], dtype=np.float32)

                e_mean, e_std = float(np.mean(elevs)), float(np.std(elevs))
                t_mean, t_std = float(np.mean(temps)), float(np.std(temps))
                r_mean, r_std = float(np.mean(rains)), float(np.std(rains))

                # Build candidate concept
                candidate = Concept(
                    resource_type=r_type,
                    elevation_mean=e_mean if e_std < 0.15 else None,
                    elevation_std=e_std if e_std < 0.15 else None,
                    temp_mean=t_mean if t_std < 0.15 else None,
                    temp_std=t_std if t_std < 0.15 else None,
                    rain_mean=r_mean if r_std < 0.15 else None,
                    rain_std=r_std if r_std < 0.15 else None,
                    support=len(g),
                    confidence=1.0
                )

                # Check if no features are active
                if candidate.elevation_mean is None and candidate.temp_mean is None and candidate.rain_mean is None:
                    continue

                # Find similar existing concept to update
                similar = None
                for ec in self.concepts[r_type]:
                    dist = 0.0
                    comparisons = 0
                    if candidate.elevation_mean is not None and ec.elevation_mean is not None:
                        dist += abs(candidate.elevation_mean - ec.elevation_mean)
                        comparisons += 1
                    if candidate.temp_mean is not None and ec.temp_mean is not None:
                        dist += abs(candidate.temp_mean - ec.temp_mean)
                        comparisons += 1
                    if candidate.rain_mean is not None and ec.rain_mean is not None:
                        dist += abs(candidate.rain_mean - ec.rain_mean)
                        comparisons += 1

                    if comparisons > 0 and dist / comparisons < 0.1:
                        similar = ec
                        break

                if similar is not None:
                    similar.elevation_mean = candidate.elevation_mean
                    similar.elevation_std = candidate.elevation_std
                    similar.temp_mean = candidate.temp_mean
                    similar.temp_std = candidate.temp_std
                    similar.rain_mean = candidate.rain_mean
                    similar.rain_std = candidate.rain_std
                    similar.support = candidate.support
                    similar.confidence = min(1.0, similar.confidence + 0.05)
                else:
                    self.concepts[r_type].append(candidate)

    def matches_concept(self, r_type: str, y: int, x: int, world) -> bool:
        """
        Checks if a coordinate (y, x) matches any of the agent's formed concepts for r_type.
        """
        concepts = self.concepts.get(r_type, [])
        if not concepts:
            return False

        for concept in concepts:
            if concept.confidence < 0.1:
                continue

            active_rules = 0
            matches = 0

            if concept.elevation_mean is not None:
                active_rules += 1
                val = world.elevation[y, x]
                if abs(val - concept.elevation_mean) <= 2.0 * concept.elevation_std:
                    matches += 1

            if concept.temp_mean is not None:
                active_rules += 1
                val = world.temperature[y, x] / 50.0
                if abs(val - concept.temp_mean) <= 2.0 * concept.temp_std:
                    matches += 1

            if concept.rain_mean is not None:
                active_rules += 1
                val = world.rainfall[y, x] / 3000.0
                if abs(val - concept.rain_mean) <= 2.0 * concept.rain_std:
                    matches += 1

            if active_rules > 0 and matches == active_rules:
                return True

        return False

    def record_visit(self, location: tuple, chunk_size: int = 32):
        """Marks a coordinate's parent chunk as visited."""
        cy, cx = location[0] // chunk_size, location[1] // chunk_size
        self.visited_chunks.add((cy, cx))
