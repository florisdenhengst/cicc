from numpy.typing import ArrayLike
import pandas as pd
import numpy as np

def strip_label(name: str):
    """
    Removes the substring ``"label_"`` from the start of an input string ``name`` if it exists.
    """
    if name.startswith('label_'):
        return name[6:]
    else:
        return name
    
def index_for_arange(start: float, step: float, value: float):
    """
    Returns at which location/index a known ``value`` for an ``np.arange`` generated with ``start`` and ``step`` resides. 
    """
    return int((value - start) / step)

def write_results(row_id: str, dataset: str, model: str, approach: str, results: ArrayLike, filename: str='results/results.csv'):
    """
    Writes results to a results file. Overwrites existing results with same `row_id`
    * row_id (str): name of the resultset, to be written as single row
    * dataset (str): name of the dataset results were generated on
    * model (str): name of the intent classifier
    * results (ArrayLike): the results to be written
    * filename (str): the filename to write to. Defaults to 'results/results.csv'
    """
    current_results = pd.read_csv(filename, index_col='id')
    current_results.loc[row_id] = [dataset, model, approach, *results]
    current_results.to_csv(filename)


def cq_filter(y_pred_set: ArrayLike, max_ps_size: int):
    ambiguous = y_pred_set.sum(axis=1) > max_ps_size
    single = y_pred_set.sum(axis=1) <= 1
    return ~ambiguous & ~single

def cq_labels(y_pred_set: ArrayLike, max_ps_size: int, class_labels):
    """
    Returns 
    """
    cqs = cq_filter(y_pred_set, max_ps_size)
    cqs = y_pred_set[cqs]
    cq_labels = []
    for cq in cqs:
        alternatives = np.where(cq)[0]
        cq_labels.append([class_labels[l] for l in alternatives])
    return cq_labels