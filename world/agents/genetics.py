"""
Genetics system for Project Genesis — Phase 5.

Implements a 13-gene genome with a separate gene-expression (development) layer that
converts raw gene values into brain parameters. The separation of genotype from phenotype
keeps the genome abstract: there is no 'generosity gene' or 'shelter preference gene'.
Instead, genes express into continuous parameters that influence outcomes when selection
pressure makes certain strategies advantageous.

Pipeline:
  Genotype  (raw float32 values in [0, 1])
      ↓  express_genome()
  Brain Parameters  (thresholds, weights, rates)
      ↓  decision.py evaluate_utility()
  Behavior  (emergent — detected statistically, not prescribed)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Gene index constants.
# Names describe what the gene influences physically or physiologically.
# The link between gene value and emergent behavior is indirect and arises
# solely through natural selection pressure on survival outcomes.
# ---------------------------------------------------------------------------
G_METABOLISM       = 0   # Scales hunger and thirst decay rates
G_THERMOREGULATION = 1   # Resistance to cold and heat metabolic stress
G_VISION           = 2   # Base vision radius in grid cells
G_MOBILITY         = 3   # Movement energy efficiency
G_MEMORY_FIDELITY  = 4   # Spatial knowledge confidence decay rate
G_PLANNING         = 5   # Weight on distant vs. near resource evaluation
G_NOVELTY_SEEKING  = 6   # Weight on unexplored territory in action utility
G_SOCIAL_PROXIMITY = 7   # Tendency to move toward / remain near colony members
G_AGGRESSION       = 8   # Territorial friction encounter probability modifier
G_RESOURCE_SHARING = 9   # Utility weight for donating stored resources to others
G_RISK_SENSITIVITY = 10  # Danger avoidance penalty multiplier
G_RESILIENCE       = 11  # Injury resistance and muscle mass preservation rate
G_LONGEVITY        = 12  # Maximum lifespan modifier
G_LEARNING_RATE    = 13  # Baseline learning rate for lifetime adjustments

GENE_COUNT = 14
GENE_NAMES = [
    "g_metabolism", "g_thermoregulation", "g_vision", "g_mobility",
    "g_memory_fidelity", "g_planning", "g_novelty_seeking", "g_social_proximity",
    "g_aggression", "g_resource_sharing", "g_risk_sensitivity",
    "g_resilience", "g_longevity", "g_learning_rate",
]


class Genome:
    """
    A fixed-length array of 13 gene values in [0.0, 1.0].
    Raw gene values carry no intrinsic behavioral meaning until expressed
    through the express_genome() development step.
    """
    __slots__ = ("genes",)

    def __init__(self, genes: np.ndarray):
        self.genes = np.clip(genes, 0.0, 1.0).astype(np.float32)

    def __repr__(self):
        return f"Genome([{', '.join(f'{g:.3f}' for g in self.genes)}])"

    def to_list(self) -> list:
        """Returns a JSON-serialisable list of gene values (4 decimal places)."""
        return [round(float(g), 4) for g in self.genes]


# ---------------------------------------------------------------------------
# Genome construction
# ---------------------------------------------------------------------------

def create_genome(rng=None) -> Genome:
    """Creates a new genome by sampling each gene uniformly from [0, 1]."""
    if rng is None:
        genes = np.random.uniform(0.0, 1.0, GENE_COUNT).astype(np.float32)
    else:
        genes = rng.uniform(0.0, 1.0, GENE_COUNT).astype(np.float32)
    return Genome(genes)


def neutral_genome() -> Genome:
    """
    Returns a genome with all genes set to 0.5.
    Used in unit tests and backward-compatibility paths where explicit
    trait values are supplied — ensures neutral expression of all parameters.
    """
    return Genome(np.full(GENE_COUNT, 0.5, dtype=np.float32))


# ---------------------------------------------------------------------------
# Reproduction operators
# ---------------------------------------------------------------------------

def crossover(genome_a: Genome, genome_b: Genome, rng=None) -> Genome:
    """
    Produces a child genome by independently drawing each gene from either
    parent A or parent B with equal probability (Mendelian inheritance).
    """
    if rng is None:
        mask = np.random.randint(0, 2, GENE_COUNT).astype(bool)
    else:
        mask = rng.integers(0, 2, GENE_COUNT, dtype=bool)
    child_genes = np.where(mask, genome_a.genes, genome_b.genes)
    return Genome(child_genes)


def mutate(genome: Genome, mutation_rate: float = 0.05, rng=None) -> Genome:
    """
    Applies independent Gaussian perturbation to each gene.
    `mutation_rate` is the standard deviation of the noise.
    All resulting values are clamped back to [0, 1].
    NOTE: mutation_rate is a fixed simulation parameter — it is NOT an
    evolvable gene, to maintain observational clarity during this phase.
    """
    if rng is None:
        noise = np.random.normal(0.0, mutation_rate, GENE_COUNT).astype(np.float32)
    else:
        noise = rng.normal(0.0, mutation_rate, GENE_COUNT).astype(np.float32)
    mutated = np.clip(genome.genes + noise, 0.0, 1.0)
    return Genome(mutated)


# ---------------------------------------------------------------------------
# Gene expression (genotype → phenotype / brain parameters)
# ---------------------------------------------------------------------------

def express_genome(genome: Genome) -> dict:
    """
    Development step: converts raw [0, 1] gene values into the brain
    parameters consumed by the decision system and physiology simulation.

    The expressions are calibrated so that a neutral genome (all genes = 0.5)
    produces parameter values identical to the pre-Phase-5 defaults, ensuring
    backward compatibility of unit tests and existing run configs.

    Returns a dict of named brain parameters.
    """
    g = genome.genes

    return {
        # ---- Metabolic rates -----------------------------------------------
        # g_metabolism = 0 → efficient (0.5× rates) | = 1 → high-burn (1.5×)
        # Neutral (0.5) → 1.0× (unchanged from Phase 4 baseline).
        "hunger_rate_mult":  float(0.5 + g[G_METABOLISM]),
        "thirst_rate_mult":  float(0.5 + g[G_METABOLISM]),

        # ---- Thermoregulation -----------------------------------------------
        # High value → less cold/heat metabolic penalty.
        # Neutral (0.5) → 0.4 resistance (mild protection; cold still hurts).
        "cold_resistance":   float(g[G_THERMOREGULATION] * 0.8),
        "heat_resistance":   float(g[G_THERMOREGULATION] * 0.8),

        # ---- Vision radius --------------------------------------------------
        # g_vision: 0 → 10 cells | 0.5 → 20 cells | 1.0 → 30 cells.
        # Neutral matches pre-Phase-5 default vision of 20.
        "vision_radius":     int(10 + g[G_VISION] * 20),

        # ---- Movement efficiency --------------------------------------------
        # High g_mobility → less energy drain per step.
        # Neutral (0.5) → 1.0× drain (unchanged baseline).
        "move_energy_mult":  float(1.5 - g[G_MOBILITY]),

        # ---- Spatial memory decay -------------------------------------------
        # Low value → knowledge retained longer; high → rapid forgetting.
        # Calibrated so neutral (0.5) → 0.00055 decay/tick.
        "memory_decay_rate": float(0.0001 + g[G_MEMORY_FIDELITY] * 0.0009),

        # ---- Planning horizon -----------------------------------------------
        # 0.0 = purely reactive (prefers closest resource)
        # 1.0 = strongly forward-planning (willing to travel far for quality)
        "planning_horizon":  float(g[G_PLANNING]),

        # ---- Novelty / exploration weight -----------------------------------
        # Scales the unexplored-territory term in explore utility.
        # Range 0.0 – 2.0; neutral (0.5) → 1.0.
        "novelty_weight":    float(g[G_NOVELTY_SEEKING] * 2.0),

        # ---- Social proximity weight ----------------------------------------
        # Scales utility of following / staying near colony members.
        # Range 0.0 – 1.5; neutral (0.5) → 0.75.
        "social_weight":     float(g[G_SOCIAL_PROXIMITY] * 1.5),

        # ---- Aggression multiplier ------------------------------------------
        # Scales territorial friction encounter probability.
        # Range 0.0 – 2.0; neutral (0.5) → 1.0 (baseline friction).
        "aggression_mult":   float(g[G_AGGRESSION] * 2.0),

        # ---- Resource-sharing weight ----------------------------------------
        # Scales utility of Share Food / Share Water actions.
        # Range 0.0 – 1.0; at 0.0, sharing actions are never chosen.
        "sharing_weight":    float(g[G_RESOURCE_SHARING]),

        # ---- Risk sensitivity multiplier ------------------------------------
        # Scales the danger-avoidance penalty in utility evaluation.
        # 0.5 (low) → bold; 2.0 (high) → very cautious.
        # Neutral (0.5) → 1.25× (slightly more cautious than baseline).
        "risk_mult":         float(0.5 + g[G_RISK_SENSITIVITY] * 1.5),

        # ---- Physical resilience --------------------------------------------
        # Scales injury resistance and rate of muscle-mass preservation.
        # Range 0.0 – 1.0.
        "resilience":        float(g[G_RESILIENCE]),

        # ---- Longevity offset -----------------------------------------------
        # Shifts the agent's max_age by ±20 years (in ticks).
        # g=0 → −7200 ticks (−20 yr) | g=0.5 → 0 | g=1.0 → +7200 ticks (+20 yr)
        "max_age_offset":    int((float(g[G_LONGEVITY]) - 0.5) * 2.0 * 20 * 360),

        # ---- Learning rate --------------------------------------------------
        # Baseline learning rate for lifetime feature-weight updates.
        # Neutral (0.5) → 0.05 step size.
        "learning_rate":     float(g[G_LEARNING_RATE] * 0.1),
    }


# ---------------------------------------------------------------------------
# Population-level diversity metrics
# ---------------------------------------------------------------------------

def genetic_distance(genome_a: Genome, genome_b: Genome) -> float:
    """
    Euclidean distance between two genomes in gene space,
    normalised to [0, 1] by dividing by sqrt(GENE_COUNT).
    """
    return float(
        np.linalg.norm(genome_a.genes - genome_b.genes) / np.sqrt(GENE_COUNT)
    )


def population_diversity(genomes: list) -> float:
    """
    Mean pairwise genetic distance across a list of Genome objects.
    Returns 0.0 for fewer than 2 genomes.
    A crash toward 0.0 indicates a dangerous genetic bottleneck.
    """
    n = len(genomes)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += genetic_distance(genomes[i], genomes[j])
            count += 1
    return total / count if count > 0 else 0.0
