# Scientific Research Report: Ablation sweep: all under Level 1 (Verification)

**Experiment ID:** `EXP-20260627-XY81`  
**Level:** Level 1 (Verification)  
**Genesis Core Version:** Phase 8.4  
**Git Commit:** `N/A`  
**Date:** 2026-06-27 09:39:23  
**Seed Count / Scarcity:** 3 seeds / Scarcity 1.0  
**Ablations Applied:** `{"planner": false, "emotion": false, "relationships": false, "memory_importance": false, "motivation": false, "prediction_error": false}`  

---

## 1. Abstract
This scientific report documents the behavior, survival, and dynamics of the Project Genesis agent populations under a controlled experiment setup. In this run (`EXP-20260627-XY81`), we simulated 3 independent populations across unique topographic seeds. The population evolved under a scarcity setting of 1.0 and completed a runtime of 1000 ticks.

---

## 2. Experimental Setup & Methods
Agents are spawned in 4 distinct colonies of 4 founders each, derived from random Perlin genetic priors. We tested the target population using the following ablation parameters:
*   **Planner Subsystem:** ABLATED (Bypassed queue pop and commitment)
*   **Emotion Subsystem:** ABLATED (Continuous drives clamped to 0.0)
*   **Relationships Subsystem:** ABLATED (Graph disabled, social weights neutral)
*   **Memory Importance:** ABLATED (Uniform exponential decay)
*   **Motivation Drift:** ABLATED (Lateral inhibition and profile drifts disabled)
*   **Prediction Error Feedback:** ABLATED (expectation feedback zeroed)

---

## 3. Results & Lifespan Summary
The average lifespan across all seeds was **750.2** ticks (approx. 2.1 years). The average number of surviving agents at tick 1000 was **6.67** founders/descendants per seed.

| Seed | Ticks Completed | Survivors | Avg Lifespan (Ticks) | Max Lifespan (Ticks) | Trust Centralization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1010 | 1000 | 6/20 | 764.2 | 1000 | 0.0 |
| 1020 | 1000 | 12/21 | 844.5 | 1000 | 0.0 |
| 1030 | 1000 | 2/21 | 641.8 | 1000 | 0.0 |

---

## 4. Emergent Behaviors & Discoveries
The Emergence Detector flagged the following civilization-level behaviors during execution:
*   **Tick 200** | *Emergence: Colony Split*: Colony split detected: shelters are highly dispersed across the land (Max distance: 728.7 cells)
*   **Tick 815** | *Emergence: Permanent Settlement*: Agent 0 established a durable permanent shelter (Level 2)
*   **Tick 184** | *Emergence: Permanent Settlement*: Agent 2 established a durable permanent shelter (Level 2)
*   **Tick 200** | *Emergence: Colony Split*: Colony split detected: shelters are highly dispersed across the land (Max distance: 548.8 cells)
*   **Tick 200** | *Emergence: Colony Split*: Colony split detected: shelters are highly dispersed across the land (Max distance: 517.8 cells)
*   **Tick 600** | *Anomaly*: Population crash detected: dropped 47.4% (from 19 to 10)
*   **Tick 900** | *Anomaly*: Population crash detected: dropped 66.7% (from 6 to 2)

---

## 5. Causal Chain Traces
The following causal chains document trigger-to-outcome sequences recorded step-by-step:

### Chain 1: Agent 8 (Storm / Freezing Temp)
*   **Trigger Tick:** 1
*   **Pre-Trigger State:** `fear: 0.03, frustration: 0.0, boredom: 0.0`
*   **Timeline:**
*   **Resolution:** Sheltered successfully (Level 1) (at tick 4)

### Chain 2: Agent 9 (Storm / Freezing Temp)
*   **Trigger Tick:** 1
*   **Pre-Trigger State:** `fear: 0.03, frustration: 0.0, boredom: 0.0`
*   **Timeline:**
*   **Resolution:** Sheltered successfully (Level 1) (at tick 4)

### Chain 3: Agent 10 (Storm / Freezing Temp)
*   **Trigger Tick:** 1
*   **Pre-Trigger State:** `fear: 0.03, frustration: 0.0, boredom: 0.0`
*   **Timeline:**
*   **Resolution:** Sheltered successfully (Level 1) (at tick 4)

### Chain 4: Agent 11 (Storm / Freezing Temp)
*   **Trigger Tick:** 1
*   **Pre-Trigger State:** `fear: 0.03, frustration: 0.0, boredom: 0.0`
*   **Timeline:**
*   **Resolution:** Sheltered successfully (Level 1) (at tick 4)

---

## 6. Genetic Evolution Trajectory
The drift of expressed genotypic parameters from Generation 0 founders to Generation 1+ descendants highlights adaptive traits selecting over time:

| Generation | Population | Avg Curiosity | Risk Sensitivity | Learning Rate | Aggression |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gen 0 | 16 | 0.437 | 0.538 | 0.607 | 0.555 |
| Gen 1 | 4 | 0.667 | 0.403 | 0.637 | 0.713 |

---

## 7. Discussion & Limitations
These results reflect how individual agent cognition constraints shape collective colony structures. Ablating subsystems changes how rapidly agents organize shelters or share food stockpiles, impacting survival. Limitations of this trial include deterministic water/food placements and isolated colony networks.

---

## 8. Meta-Analysis Data Reference
To conduct comparative meta-analyses or construct Cohen's d effect size tables, refer to `research/meta_analysis.md` which compiles summaries across all experiments.
