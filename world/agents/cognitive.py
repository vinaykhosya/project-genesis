from dataclasses import dataclass
import numpy as np

@dataclass
class Concept:
    resource_type: str      # "WATER" or "FOOD"
    elevation_mean: float = None
    elevation_std: float = None
    temp_mean: float = None  # Normalized temp
    temp_std: float = None
    rain_mean: float = None  # Normalized rain
    rain_std: float = None
    support: int = 0
    confidence: float = 1.0

@dataclass
class Procedure:
    trigger_season: int
    action_sequence: tuple  # tuple of action names
    success_count: int = 1
    confidence: float = 1.0

class Predictor:
    """
    A lightweight neural network (20 inputs -> 3 hidden -> 1 reward)
    learned purely during the agent's lifetime via gradient descent.
    """
    def __init__(self, reconstruct: bool = False):
        # Initialize weights with small random values, or zeros if reconstructing
        if reconstruct:
            self.w1 = np.zeros((20, 3), dtype=np.float32)
            self.b1 = np.zeros(3, dtype=np.float32)
            self.w2 = np.zeros((3, 1), dtype=np.float32)
            self.b2 = np.zeros(1, dtype=np.float32)
        else:
            self.w1 = np.random.normal(0.0, 0.1, (20, 3)).astype(np.float32)
            self.b1 = np.zeros(3, dtype=np.float32)
            self.w2 = np.random.normal(0.0, 0.1, (3, 1)).astype(np.float32)
            self.b2 = np.zeros(1, dtype=np.float32)

    def predict(self, context_vec: np.ndarray) -> float:
        """Forward pass of the predictor network."""
        h = np.tanh(np.dot(context_vec, self.w1) + self.b1)
        out = np.dot(h, self.w2)[0] + self.b2[0]
        return float(out)

    def train(self, context_vec: np.ndarray, reward: float, learning_rate: float):
        """Backward pass to update the weights of the predictor network."""
        X = context_vec
        # Forward pass again to get intermediate values
        z1 = np.dot(X, self.w1) + self.b1
        h = np.tanh(z1)
        pred = np.dot(h, self.w2)[0] + self.b2[0]
        
        error = reward - pred
        
        # Scale learning rate to keep it stable
        lr = learning_rate * 0.1
        
        d_pred = -2.0 * error
        
        # Gradients
        d_w2 = d_pred * h.reshape(3, 1)
        d_b2 = d_pred
        
        d_h = d_pred * self.w2.reshape(3)
        d_z1 = d_h * (1.0 - h * h)
        
        d_w1 = np.outer(X, d_z1)
        d_b1 = d_z1
        
        # Update weights
        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2
        
        # Clip weights to prevent exploding values
        self.w1 = np.clip(self.w1, -2.0, 2.0)
        self.b1 = np.clip(self.b1, -2.0, 2.0)
        self.w2 = np.clip(self.w2, -2.0, 2.0)
        self.b2 = np.clip(self.b2, -2.0, 2.0)


@dataclass(slots=True)
class TickContext:
    context_tick: int
    brain: dict
    drives: object
    local_temp: float
    day: int
    season: int
    cos_factor: float
    nearby_agents: list


@dataclass(slots=True)
class DecisionContext:
    risk_mult: float
    scarcity_prediction: float
    colony_food: float
    colony_water: float
    shelter_target: tuple
    rest_sig: float
    drink_sig: float
    eat_sig: float
    drink_stored_sig: float
    eat_stored_sig: float
    withdraw_food_sig: float
    withdraw_water_sig: float

