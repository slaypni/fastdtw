fastdtw
-------

Python implementation of `FastDTW
<http://cs.fit.edu/~pkc/papers/tdm04.pdf>`_ [1]_, which is an approximate Dynamic Time Warping (DTW) algorithm that provides optimal or near-optimal alignments with an O(N) time and memory complexity.

Install
-------

::

  pip install fastdtw

Example
-------

::
  
  import numpy as np
  from scipy.spatial.distance import euclidean

  from fastdtw import fastdtw

  x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
  y = np.array([[2,2], [3,3], [4,4]])
  distance, path = fastdtw(x, y, dist=euclidean)
  print(distance)


Example parallel computing
-------

::

  import numpy as np
  from scipy.spatial.distance import euclidean

  from fastdtw import fastdtw_parallel, get_path

  X = np.random.randint(1, 40, size=(5, 5))
  distance_matrix, path_list = fastdtw_parallel(X, dist=euclidean, n_jobs=-1)
  print(distance_matrix)

  # To allow the straight forward access of warp paths to the respective distance matrix element
  # the get_path function can be used
  # Access the path for distance matrix element with index row=2 and column=3
  print(get_path(path_list, 2, 3)

References
----------

.. [1] Stan Salvador, and Philip Chan. "FastDTW: Toward accurate dynamic time warping in linear time and space." Intelligent Data Analysis 11.5 (2007): 561-580.
