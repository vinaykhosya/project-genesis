# Scientific Research Report: Ablation sweep: none under Level 1 (Verification)

**Experiment ID:** `EXP-20260627-41AP`  
**Level:** Level 1 (Verification)  
**Genesis Core Version:** Phase 8.4  
**Git Commit:** `N/A`  
**Date:** 2026-06-27 10:11:01  
**Seed Count / Scarcity:** 3 seeds / Scarcity 1.0  
**Ablations Applied:** `{"planner": true, "emotion": true, "relationships": true, "memory_importance": true, "motivation": true, "prediction_error": true}`  

---

## 1. Abstract
This scientific report documents the behavior, survival, and dynamics of the Project Genesis agent populations under a controlled experiment setup. In this run (`EXP-20260627-41AP`), we simulated 3 independent populations across unique topographic seeds. The population evolved under a scarcity setting of 1.0 and completed a runtime of 10 ticks.

---

## 2. Experimental Setup & Methods
Agents are spawned in 4 distinct colonies of 4 founders each, derived from random Perlin genetic priors. We tested the target population using the following ablation parameters:
*   **Planner Subsystem:** ENABLED
*   **Emotion Subsystem:** ENABLED
*   **Relationships Subsystem:** ENABLED
*   **Memory Importance:** ENABLED
*   **Motivation Drift:** ENABLED
*   **Prediction Error Feedback:** ENABLED

---

## 3. Results & Lifespan Summary
The average lifespan across all seeds was **9.6** ticks (approx. 0.0 years). The average number of surviving agents at tick 10 was **17.0** founders/descendants per seed.

| Seed | Ticks Completed | Survivors | Avg Lifespan (Ticks) | Max Lifespan (Ticks) | Trust Centralization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1010 | 10 | 16/16 | 10.0 | 10 | 0.0 |
| 1020 | 10 | 17/17 | 9.6 | 10 | 0.0 |
| 1030 | 10 | 18/18 | 9.2 | 10 | 0.0 |

---

## 4. Automatic Hypothesis Ranking

The framework automatically maps expressed behaviors to lifetime outcomes to rank which psychological, genetic, and social factors most strongly predict civilization survival:

| Rank | Hypothesis | Avg Correlation (r) | Evidence Strength |
| :---: | :--- | :---: | :--- |
| 1 | Curiosity selects for longevity | `-0.210` | **Weak Evidence** |
| 2 | Aggression reduces longevity (Inverted) | `-0.128` | **Weak Evidence** |
| 3 | Curiosity predicts discoveries count | `-0.021` | **Negligible / No Evidence** |
| 4 | Altruism / resource sharing predicts longevity | `0.000` | **Negligible / No Evidence** |
| 5 | Fear predicts technology (shelter levels) | `0.000` | **Negligible / No Evidence** |
| 6 | Technology (shelter quality) predicts longevity | `0.000` | **Negligible / No Evidence** |
| 7 | Frustration suppresses cooperation (Inverted) | `0.000` | **Negligible / No Evidence** |

---

## 5. Emergent Behaviors & Discoveries
The Emergence Detector flagged the following civilization-level behaviors during execution:
*   No novel emergent structures were detected in this run.

---

## 6. Causal Chain Traces
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

## 7. Genetic Evolution Trajectory
The drift of expressed genotypic parameters from Generation 0 founders to Generation 1+ descendants highlights adaptive traits selecting over time:

| Generation | Population | Avg Curiosity | Risk Sensitivity | Learning Rate | Aggression |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gen 0 | 16 | 0.717 | 0.525 | 0.529 | 0.511 |

---

## 8. Discussion & Limitations
These results reflect how individual agent cognition constraints shape collective colony structures. Ablating subsystems changes how rapidly agents organize shelters or share food stockpiles, impacting survival. Limitations of this trial include deterministic water/food placements and isolated colony networks.

---

## 9. Meta-Analysis Data Reference
To conduct comparative meta-analyses or construct Cohen's d effect size tables, refer to `research/meta_analysis.md` which compiles summaries across all experiments.
