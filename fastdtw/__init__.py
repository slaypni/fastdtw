try:
    from ._fastdtw import fastdtw, dtw, dtw_parallel, fastdtw_parallel, get_path
except ImportError:
    from .fastdtw import fastdtw, dtw, dtw_parallel, fastdtw_parallel, get_path
