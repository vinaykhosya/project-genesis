# Project Genesis: An Agentic Artificial Life & Evolutionary Biology Simulator

Project Genesis is a high-resolution simulation of artificial life, ecology, and natural selection. It models the survival, cognition, genetics, and emergent behavior of autonomous agents navigating a dynamic world.

---

## 🔬 Core System Architecture

The simulation is built as a layered pipeline that mirrors real-world biological and ecological structures:

```
Genetics (14-gene genotype to brain parameters)
   ↓
Physiology (Fat reserves, muscle mass, health)
   ↓
Innate Reflexes (Combat safe-zones, threat display, pain responses)
   ↓
Emotion & Hormones (Fear levels, stress cooldowns)
   ↓
Motivational Drives (Hunger, thirst, comfort, safety)
   ↓
Cognitive Planner (Sigmoid utility selector & neural prediction)
   ↓
Motor Actions (Pathfinding, resource extraction, sheltering)
   ↓
Episodic Memory (Spatial mapping, relationship trust, win/loss history)
```

---

## 🌍 Key Subsystems & Features

### 1. Unified State Container (`world/state.py`)
All simulation stages receive, mutate, and return a single, centralized `WorldState` object. To ensure maximum vectorization performance, all grids are stored as 2D NumPy arrays (`float32` or `int32`).

### 2. Whittaker Biome Matrix (`world/biomes.py`)
Dynamic continental temperature and rainfall maps translate to 9 distinct biomes via a resolution-independent Whittaker mapping:
* `OCEAN` (0) | `GLACIER` (1) | `TUNDRA` (2) | `TAIGA` (3) | `TEMPERATE_FOREST` (4) | `GRASSLAND` (5) | `DESERT` (6) | `RAINFOREST` (7) | `LAKE` (8)

### 3. Physical Hydrology & Climate
* **Resolution Independence**: Environmental rates scale dynamically with grid size to ensure map preset invariance.
* **Wind Advection Clamping**: Winds blow across flat water; height maps are clamped to sea level (`0.3`) for advection calculations.
* **Land Transpiration**: Moisture recyclers prevent dry continental centers.
* **Priority-Flood Hydrology**: Heap-queue Priority-Flood resolves sinks to trace realistic river drainage basins.
* **Mountainous Resource Belets**: Terrestrial mineral clusters (Iron, Copper) are generated using low-frequency noise masks on mountainous regions.

### 4. Innate Reflexes & Combat Layer v2 (`world/agents/decision.py`)
We hardcode biological primitives, while allowing strategies and thresholds to evolve via genetics:
* **Home-Radius Safe Zones**: Agents claim home coordinates with a genome-derived radius (`15 + aggression_mult * 10`). Territorial disputes are suppressed if either agent is inside their nesting grounds.
* **Threat Display Stage**: Fights enter a non-damaging warning phase. Agents exchange injury damage only if both hold their ground. If one backs down, they suffer a fear/stress spike instead.
* **Confidence-Based Retreat**: Relative strength components are calculated based on health, injury, and genetic aggression:
  $$\text{confidence} = \frac{\text{my strength}}{\text{my strength} + \text{their strength}}$$
  Agents retreat if confidence falls below their risk tolerance threshold.
* **Winner/Loser Memory**: Fights are registered as winning/losing outcomes in episodic memory, influencing future interaction confidence.
* **Fear Cooldown**: Deferring immediate re-engagement using a stress cooldown timer proportional to injury level.

---

## 📈 Running the Simulation

Initialize the standard validation suite:
```bash
python run_test.py
```

### Configuration Panel (`run_test.py`)
You can tweak map seeds, scarcity, disputes, and climate epoch modes directly from the top panel inside `run_test.py`.

### Interactive Dashboard
To view real-time population heatmaps, lineage trees, and genetic distributions:
1. Run the simulation.
2. Open `visualizer.html` in your browser.
