## Phase 8.4: Adaptive Motivation & Cognitive-Emotional Feedback

- `[x]` Define `MotivationDimension` and `MotivationProfile` in `world/agents/drives.py`
- `[x]` Implement `update_adaptive_motivations` in `world/agents/drives.py` using Welford's algorithm
- `[x]` Update `update_emotional_drives` in `world/agents/drives.py` using category-specific prediction errors
- `[x]` Update `update_relationships` in `world/agents/drives.py` to use saturating co-presence attachment
- `[x]` Implement Lateral Inhibition Conflict Resolver in `compute_drive_modulation` in `world/agents/drives.py`
- `[x]` Initialize `MotivationProfile` in `Agent.__init__` in `world/agents/agent.py` using genome priors
- `[x]` Add `last_prediction_error` and `last_prediction_category` registers to `Agent` in `world/agents/agent.py`
- `[x]` Wire prediction error and category capture in `simulate_agent_tick` in `world/agents/simulation.py`
- `[x]` Wire 100-tick Motivation Clock in `simulate_agent_tick` in `world/agents/simulation.py`
- `[x]` Implement exponential memory decay in the 360-tick daily clock in `world/agents/simulation.py`
- `[x]` Export `motivation` telemetry in `simulation.py`, `main.py`, and `run_test.py`
- `[x]` Add 5 comprehensive Phase 8.4 unit tests to `tests/test_world.py`
- `[x]` Run all unit tests to verify 53/53 tests pass
