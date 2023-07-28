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
