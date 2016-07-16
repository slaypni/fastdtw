try:
    from ._fastdtw import fastdtw, dtw
except ImportError: 
    # user has been warned on installation
    from .fastdtw import fastdtw, dtw
