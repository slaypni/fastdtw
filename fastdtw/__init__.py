try:
    from ._fastdtw import fastdtw, dtw
    from .fastdtw import fastdtw_subsequence
except ImportError:
    from .fastdtw import fastdtw, dtw, fastdtw_subsequence
