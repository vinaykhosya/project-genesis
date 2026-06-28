"""
climate_epochs.py — Phase 9C: Climate Epoch Transitions

Implements the ClimateEpochEngine for managing non-stationary environmental selection.
Replaces the fixed 10-year disaster cycle with named epochs and smooth transitions (ramps).
"""

import numpy as np
from typing import Dict, Any, List, Tuple

EPOCHS: Dict[str, Dict[str, float]] = {
    "Temperate": {
        "temp_offset": 0.0,
        "water_mult": 1.0,
        "food_mult": 1.0,
        "seasonal_mult": 1.0
    },
    "Wet / Flood": {
        "temp_offset": -2.0,
        "water_mult": 1.5,
        "food_mult": 1.2,
        "seasonal_mult": 0.8
    },
    "Drought": {
        "temp_offset": 3.0,
        "water_mult": 0.4,
        "food_mult": 0.7,
        "seasonal_mult": 1.2
    },
    "Cold Wave / Ice Age": {
        "temp_offset": -10.0,
        "water_mult": 0.7,
        "food_mult": 0.5,
        "seasonal_mult": 1.4
    },
    "Heatwave": {
        "temp_offset": 8.0,
        "water_mult": 0.6,
        "food_mult": 0.8,
        "seasonal_mult": 1.3
    },
    "Famine": {
        "temp_offset": 1.0,
        "water_mult": 0.9,
        "food_mult": 0.3,
        "seasonal_mult": 1.0
    }
}

SEQUENCE_PRESETS: Dict[str, List[Tuple[str, int]]] = {
    # Sequence of (epoch_name, duration_years)
    "stable": [
        ("Temperate", 1000)
    ],
    "slow_change": [
        ("Temperate", 25),
        ("Wet / Flood", 25),
        ("Temperate", 20),
        ("Drought", 30),
        ("Temperate", 20),
        ("Cold Wave / Ice Age", 25),
        ("Temperate", 20),
        ("Heatwave", 25),
        ("Famine", 30)
    ],
    "rapid_change": [
        ("Temperate", 10),
        ("Wet / Flood", 10),
        ("Drought", 10),
        ("Temperate", 8),
        ("Cold Wave / Ice Age", 10),
        ("Heatwave", 10),
        ("Famine", 10),
        ("Temperate", 8)
    ]
}

class ClimateEpochEngine:
    def __init__(self, mode: str = "legacy", seed: int = 42):
        self.mode = mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # State tracking (tick-based)
        self.current_epoch_name = "Temperate"
        self.prev_epoch_name = "Temperate"
        self.epoch_start_tick = 0
        self.epoch_duration = 9999999
        self.transition_ticks = 720 # 2 years transition ramp by default
        
        # Current blended values
        self.temp_offset = 0.0
        self.water_mult = 1.0
        self.food_mult = 1.0
        self.seasonal_mult = 1.0
        
        # Sequence list (for sequence-based modes)
        self.sequence: List[Tuple[str, int]] = []
        self.seq_index = 0
        
        self.initialize_engine()

    def initialize_engine(self):
        if self.mode == "legacy" or self.mode == "stable":
            self.sequence = [("Temperate", 9999999)]
            self.epoch_duration = 9999999 * 360
        elif self.mode in SEQUENCE_PRESETS:
            self.sequence = SEQUENCE_PRESETS[self.mode]
            self.seq_index = 0
            epoch_name, yrs = self.sequence[0]
            self.current_epoch_name = epoch_name
            self.prev_epoch_name = epoch_name
            self.epoch_duration = yrs * 360
            self.epoch_start_tick = 0
        elif self.mode == "random":
            # Generate initial random epoch
            self.current_epoch_name = "Temperate"
            self.prev_epoch_name = "Temperate"
            self.epoch_duration = self.rng.integers(8, 30) * 360
            self.epoch_start_tick = 0
        else:
            self.sequence = [("Temperate", 9999999)]
            self.epoch_duration = 9999999 * 360

    def load_state(self, state_dict: Dict[str, Any]):
        """Restores state from full checkpoint deserialization."""
        if not state_dict:
            return
        self.mode = state_dict.get("mode", "legacy")
        self.current_epoch_name = state_dict.get("current_epoch_name", "Temperate")
        self.prev_epoch_name = state_dict.get("prev_epoch_name", "Temperate")
        self.epoch_start_tick = state_dict.get("epoch_start_tick", 0)
        self.epoch_duration = state_dict.get("epoch_duration", 9999999)
        self.transition_ticks = state_dict.get("transition_ticks", 720)
        self.seq_index = state_dict.get("seq_index", 0)
        
        # Restore lists
        seq_data = state_dict.get("sequence", [])
        self.sequence = [(item[0], item[1]) for item in seq_data]
        
        # Current blended values
        self.temp_offset = state_dict.get("temp_offset", 0.0)
        self.water_mult = state_dict.get("water_mult", 1.0)
        self.food_mult = state_dict.get("food_mult", 1.0)
        self.seasonal_mult = state_dict.get("seasonal_mult", 1.0)

    def save_state(self) -> Dict[str, Any]:
        """Saves current state for checkpointing."""
        last_t = getattr(self, "last_tick", 0)
        elapsed = last_t - self.epoch_start_tick
        transition_pct = float(min(1.0, elapsed / self.transition_ticks) if elapsed < self.transition_ticks else 1.0)
        remaining_years = float(max(0.0, (self.epoch_duration - elapsed) / 360.0))
        return {
            "mode": self.mode,
            "current_epoch_name": self.current_epoch_name,
            "prev_epoch_name": self.prev_epoch_name,
            "epoch_start_tick": int(self.epoch_start_tick),
            "epoch_duration": int(self.epoch_duration),
            "transition_ticks": int(self.transition_ticks),
            "seq_index": int(self.seq_index),
            "sequence": [[item[0], int(item[1])] for item in self.sequence],
            "temp_offset": float(self.temp_offset),
            "water_mult": float(self.water_mult),
            "food_mult": float(self.food_mult),
            "seasonal_mult": float(self.seasonal_mult),
            "transition_pct": transition_pct,
            "remaining_years": remaining_years
        }

    def tick(self, world_tick: int, history_callback=None, timeline_callback=None) -> Tuple[float, float, float, float]:
        """
        Advances the climate epoch engine by one tick.
        Calculates smooth ramp transitions and decides when to trigger the next epoch.
        
        Returns:
        - (temp_offset, water_mult, food_mult, seasonal_mult)
        """
        self.last_tick = world_tick
        if self.mode == "legacy":
            # Backward compatibility with the original 10-year repeating disaster cycle
            year = world_tick // 360
            day = world_tick % 360
            cycle_year = year % 10
            
            temp_off = 0.0
            water_m = 1.0
            food_m = 1.0
            
            if cycle_year == 3:
                temp_off = -8.0
                name = "Cold Wave"
            elif cycle_year == 5:
                water_m = 0.4
                name = "Drought"
            elif cycle_year == 7:
                temp_off = 8.0
                name = "Heatwave"
            elif cycle_year == 9:
                food_m = 0.3
                name = "Famine"
            else:
                name = "Temperate"
                
            self.current_epoch_name = name
            self.temp_offset = temp_off
            self.water_mult = water_m
            self.food_mult = food_m
            self.seasonal_mult = 1.0
            
            return self.temp_offset, self.water_mult, self.food_mult, self.seasonal_mult

        # Check if the current epoch has ended
        elapsed = world_tick - self.epoch_start_tick
        if elapsed >= self.epoch_duration:
            # Advance to the next epoch
            self.prev_epoch_name = self.current_epoch_name
            self.epoch_start_tick = world_tick
            
            if self.mode in SEQUENCE_PRESETS:
                self.seq_index = (self.seq_index + 1) % len(self.sequence)
                epoch_name, yrs = self.sequence[self.seq_index]
                self.current_epoch_name = epoch_name
                self.epoch_duration = yrs * 360
            elif self.mode == "random":
                # Choose random new epoch name different from current
                options = [k for k in EPOCHS.keys() if k != self.current_epoch_name]
                self.current_epoch_name = self.rng.choice(options)
                self.epoch_duration = self.rng.integers(5, 25) * 360
                
            # Log transition events
            msg = f"[CLIMATE EPOCH] Transition started from {self.prev_epoch_name} to {self.current_epoch_name} (Duration: {self.epoch_duration // 360} years)."
            if history_callback:
                history_callback(msg)
            if timeline_callback:
                timeline_callback(self.current_epoch_name, msg)
            print(f"\n*** {msg} ***\n")

        # Calculate smooth blend factor using Cosine interpolation during transition window
        elapsed = world_tick - self.epoch_start_tick
        if elapsed < self.transition_ticks:
            t = elapsed / self.transition_ticks
            blend = (1.0 - np.cos(t * np.pi)) / 2.0  # Smooth S-curve [0.0, 1.0]
        else:
            blend = 1.0
            
        # Blend values between previous and current targets
        prev_vals = EPOCHS[self.prev_epoch_name]
        curr_vals = EPOCHS[self.current_epoch_name]
        
        self.temp_offset = prev_vals["temp_offset"] * (1.0 - blend) + curr_vals["temp_offset"] * blend
        self.water_mult  = prev_vals["water_mult"] * (1.0 - blend) + curr_vals["water_mult"] * blend
        self.food_mult   = prev_vals["food_mult"] * (1.0 - blend) + curr_vals["food_mult"] * blend
        self.seasonal_mult = prev_vals["seasonal_mult"] * (1.0 - blend) + curr_vals["seasonal_mult"] * blend
        
        return self.temp_offset, self.water_mult, self.food_mult, self.seasonal_mult
