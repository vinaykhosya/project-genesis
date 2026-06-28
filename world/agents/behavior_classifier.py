"""
Behavioral clustering system for Project Genesis — Phase 5.

Instead of assigning hardcoded behavioral labels (Explorer, Settler…), this
module computes an 8-dimensional behavioral feature vector for each agent and
applies K-Means clustering to discover natural groupings in the population.

Cluster labels are numbers: C0, C1, C2, C3. Genesis shows the clusters;
the scientist (user) interprets what each cluster represents by examining
the centroid feature values. No labels are prescribed by the simulator.

Implementation is pure NumPy — no scikit-learn dependency required.
"""

import numpy as np

# Feature names in display order — used by the visualizer's Clusters tab
BEHAVIOR_FEATURE_NAMES = [
    "Nomadism",            # max exploration radius relative to world size
    "Building",            # fraction of ticks spent building shelter
    "Sharing",             # fraction of ticks spent sharing food/water
    "Resting",             # fraction of ticks spent resting
    "Exploration Rate",    # new discoveries per tick (normalised)
    "Foraging Rate",       # eats + drinks per tick (normalised)
    "Planning Quality",    # prediction gains per attempt
    "Social Network",      # known colony members per year survived
]


def build_feature_vector(agent, world_width: int) -> np.ndarray:
    """
    Computes an 8-dimensional behavioural feature vector for one agent.
    All components are normalised to approximately [0, 1].
    Safe against division-by-zero and NaN/inf values.
    """
    ticks        = max(agent.ticks_survived, 1)
    total_acts   = max(sum(agent.action_counts.values()), 1)
    years        = max(ticks / 360.0, 0.01)

    nomadism    = float(agent.max_radius) / max(float(world_width), 1.0)

    building    = agent.action_counts.get("Building Shelter", 0) / total_acts

    sharing     = (
        agent.action_counts.get("Share Food",  0) +
        agent.action_counts.get("Share Water", 0)
    ) / total_acts

    resting     = agent.action_counts.get("Resting", 0) / total_acts

    explore_rate = min(1.0, agent.discoveries_count / ticks * 20.0)

    forage_rate  = min(1.0, (agent.eats_count + agent.drinks_count) / ticks * 5.0)

    pa = agent.prediction_attempts
    planning_q  = float(agent.prediction_gains / pa) if pa > 0 else 0.0

    social_net  = min(1.0, len(agent.known_agents) / years / 5.0)

    vec = np.array(
        [nomadism, building, sharing, resting,
         explore_rate, forage_rate, planning_q, social_net],
        dtype=np.float32,
    )
    return np.clip(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


# ---------------------------------------------------------------------------
# K-Means (pure NumPy) with K-Means++ initialisation
# ---------------------------------------------------------------------------

def _kmeans(
    X: np.ndarray,
    k: int = 4,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple:
    """
    K-Means clustering with K-Means++ initialisation.
    Returns (labels: ndarray[int], centroids: ndarray[float]).
    Falls back gracefully when n < k.
    """
    n = X.shape[0]
    if n < k:
        k = max(1, n)

    rng = np.random.default_rng(seed)

    # K-Means++ initialisation for faster convergence
    init_idx = [int(rng.integers(0, n))]
    for _ in range(k - 1):
        # Squared distance from each point to its nearest chosen centre
        sq_dists = np.min(
            np.sum((X[:, np.newaxis, :] - X[np.array(init_idx)][np.newaxis, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = sq_dists.sum()
        if total < 1e-9:
            probs = np.ones(n, dtype=np.float32) / n
        else:
            probs = sq_dists / total
            probs /= probs.sum()  # Ensure probabilities sum to exactly 1.0
        init_idx.append(int(rng.choice(n, p=probs)))

    centers = X[init_idx].copy()
    labels  = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assignment
        dists      = np.sum(
            (X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2
        )
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = X[mask].mean(axis=0)

    return labels, centers


# ---------------------------------------------------------------------------
# PCA projection for visualizer scatter plot (pure NumPy)
# ---------------------------------------------------------------------------

def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Projects a feature matrix to 2 principal components."""
    if X.shape[0] < 2 or X.shape[1] < 2:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    X_c  = X - X.mean(axis=0)
    cov  = np.cov(X_c.T)
    if cov.ndim < 2:
        return np.column_stack([X_c[:, 0], np.zeros(X.shape[0])]).astype(np.float32)
    vals, vecs = np.linalg.eigh(cov)
    top2 = vecs[:, np.argsort(vals)[::-1][:2]]
    return (X_c @ top2).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_behavior(
    agents: list,
    world_width: int,
    n_clusters: int = 4,
) -> dict:
    """
    Clusters all agents by behaviour using K-Means on an 8-feature vector.
    Assigns each agent's `behavior_cluster` attribute in place (e.g. "C0").

    Returns a dict:
    {
      "agent_clusters":    {agent_id: "C0" | "C1" | ...},
      "cluster_centroids": [{cluster_id, feature_means, count}, ...],
      "pca_coords":        {agent_id: [x, y]},
      "n_clusters":        int,
      "feature_names":     [str, ...],
    }
    Genesis shows the clusters. The scientist names them.
    """
    if len(agents) == 0:
        return {
            "agent_clusters":    {},
            "cluster_centroids": [],
            "pca_coords":        {},
            "n_clusters":        0,
            "feature_names":     BEHAVIOR_FEATURE_NAMES,
        }

    # Only include agents with enough history to compute meaningful features
    valid = [a for a in agents if a.ticks_survived >= 10]
    if len(valid) < 2:
        valid = agents

    features_raw = np.array(
        [build_feature_vector(a, world_width) for a in valid], dtype=np.float32
    )

    # Standardise per-feature (zero-mean, unit-std) for equal weighting
    stds = features_raw.std(axis=0)
    stds[stds < 1e-6] = 1.0
    features_norm = (features_raw - features_raw.mean(axis=0)) / stds

    k      = min(n_clusters, len(valid))
    labels, _ = _kmeans(features_norm, k=k)
    pca_coords = _pca_2d(features_norm)

    agent_clusters: dict  = {}
    pca_coord_map:  dict  = {}

    for i, agent in enumerate(valid):
        label                 = f"C{labels[i]}"
        agent.behavior_cluster = label
        agent_clusters[agent.id] = label
        pca_coord_map[agent.id]  = [
            round(float(pca_coords[i, 0]), 4),
            round(float(pca_coords[i, 1]), 4),
        ]

    # Fill agents that had fewer than 10 ticks with default cluster
    for a in agents:
        if a.id not in agent_clusters:
            agent.behavior_cluster = "C0"
            agent_clusters[a.id]   = "C0"
            pca_coord_map[a.id]    = [0.0, 0.0]

    # Build centroid descriptions in original (un-standardised) feature space
    centroids = []
    for j in range(k):
        mask = labels == j
        feat_means = features_raw[mask].mean(axis=0) if np.any(mask) else np.zeros(len(BEHAVIOR_FEATURE_NAMES))
        centroids.append({
            "cluster_id":    f"C{j}",
            "feature_means": {
                name: round(float(val), 3)
                for name, val in zip(BEHAVIOR_FEATURE_NAMES, feat_means)
            },
            "count": int(np.sum(mask)),
        })

    return {
        "agent_clusters":    agent_clusters,
        "cluster_centroids": centroids,
        "pca_coords":        pca_coord_map,
        "n_clusters":        k,
        "feature_names":     BEHAVIOR_FEATURE_NAMES,
    }
