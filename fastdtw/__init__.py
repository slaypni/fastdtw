try:
    from ._fastdtw import fastdtw, dtw, fastdtw_parallel, get_path
except ImportError:
    from .fastdtw import fastdtw, dtw, fastdtw_parallel, get_path
