import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import column_or_1d, check_array
from typing import cast, Callable

from mapie.metrics import (classification_coverage_score,
                           classification_mean_width_score)


def size_stratified_classification_coverage_score(
    y_true: ArrayLike,
    y_pred_set: ArrayLike
) -> float:
    """
    Implementation of the size-stratified classification coverage score as presented by Anastasious et al. (2021,2022)
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_pred_set = cast(
        NDArray,
        check_array(
            y_pred_set, force_all_finite=True, dtype=["bool"]
        )
    )
    y_pred_sizes = y_pred_set.sum(axis=1)
    cardinalities = np.arange(1, y_pred_sizes.max()+1)
    # one-hot encoding of membership
    members_oh = np.zeros((y_true.shape[0], cardinalities.shape[0]), dtype='bool')
    for g, c in enumerate(cardinalities):
        members_oh[:, g] = y_pred_sizes == c

    group_coverages = []
    for g, c in enumerate(cardinalities):
        covered_members = np.take_along_axis(
            y_pred_set[members_oh[:, g]], y_true.reshape(-1, 1)[members_oh[:, g]], axis=1
        )
        # if there are no covered members, do nothing
        if len(covered_members) == 0:
            pass
        else:
            group_coverages.append(covered_members.mean())
    return min(group_coverages)

def classification_x_width_score(
    y_pred_set: ArrayLike,
    agg: Callable[[ArrayLike], float]
) -> float:
    """
    Calculates a statistic over the coverage width for a score.
    For example ``classification_x_width_score(y_pred_set, np.mean)``
    returns the average coverage width or average coverage set size.
    """
    y_pred_set = cast(
        NDArray,
        check_array(
            y_pred_set, force_all_finite=True, dtype=["bool"]
        )
    )
    agg_width = agg(y_pred_set.sum(axis=1))
    return float(agg_width)

def adjusted_classification_x_width_score(
    y_pred_set: ArrayLike,
    agg: Callable[[ArrayLike], float],
    max_set_size: int=None,
) -> float:
    """
    Calculates a statistic over the coverage width for a score.
    For example ``classification_x_width_score(y_pred_set, np.mean)``
    returns the average coverage width or average coverage set size.
    
    Excludes any entries where set sizes > max_set_size.
    """
    if max_set_size is None:
        max_set_size = y_pred_set.shape[1]
    set_size = y_pred_set.sum(axis=1)
    masked = np.ma.masked_array(set_size, ~(set_size < max_set_size))
    agg_width = agg(masked)
    return float(agg_width)

def cq_stats(
    y_pred_set: ArrayLike,
    y_test: ArrayLike,
    max_set_size: int=None,
)-> float:
    """
    Calculates the average conformal question width for a score.
    """
    if max_set_size is None:
        max_set_size = y_pred_set.shape[1]    
    set_size = y_pred_set.sum(axis=1)
    single = (set_size == 1)
    rejected = set_size > max_set_size
    return classification_coverage_score(y_test, y_pred_set), single.sum() / y_pred_set.shape[0], (~rejected & ~single).sum() / y_pred_set.shape[0], rejected.sum() / y_pred_set.shape[0], cq_width_score(y_pred_set, np.mean, max_set_size)

def cq_width_score(
    y_pred_set: ArrayLike,
    agg: Callable[[ArrayLike], float]=np.mean,
    max_set_size: int=None,
)-> float:
    """
    Calculates the average conformal question width for a score.
    These are the prediction set widths, including only
    sets of 1 < width < max_set_size
    """
    if max_set_size is None:
        max_set_size = y_pred_set.shape[1]
    set_size = y_pred_set.sum(axis=1)
    masked = np.ma.masked_array(set_size, ~(set_size < max_set_size) | (set_size <= 1))
    agg_width = agg(masked)
    return float(agg_width)

def set_size_equals(
    y_pred_set: ArrayLike,
    n: int
) -> int:
    """
    Returns how often the set size of the prediction sets is equal to `n`.
    """
    return (y_pred_set.sum(axis=1) == n).sum()
