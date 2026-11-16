"""AUC and recall at a fixed false-alarm rate on stroke cells."""

import numpy as np
from scipy.stats import rankdata

CELL_UM = 150.0


def rank_auc(a, b):
    ranks = rankdata(np.concatenate([a, b]))
    return float((ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def auc(score, positive, negative):
    """Rank AUC of a map over two masks."""
    return rank_auc(score[positive], score[negative])


def recall_at(a, b, rate):
    """Share of ink cells above the bare cells' (1 - rate) quantile."""
    return float((a > np.quantile(b, 1.0 - rate)).mean())
