#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division
import numbers
import numpy as np
from collections import defaultdict

try:
    range = xrange
except NameError:
    pass


def fastdtw(x, y, radius=1, dist=None):
    ''' return the approximate distance between 2 time series with O(N)
        time and memory complexity

        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        radius : int
            size of neighborhood when expanding the path. A higher value will
            increase the accuracy of the calculation but also increase time
            and memory consumption. A radius equal to the size of x and y will
            yield an exact dynamic time warping calculation.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.

        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y

        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.fastdtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __fastdtw(x, y, radius, dist)



def fastdtw_worker_(X, radius=1, dist=None, indices_list=None):
    ''' allow the partial calculation of pairwise comparisons in the parallel
        computing setting.

         Parameters
        ----------
        X : array
            matrix with all time series that should be compared.
        radius : int
            size of neighborhood when expanding the path. A higher value will
            increase the accuracy of the calculation but also increase time
            and memory consumption. A radius equal to the size of x and y will
            yield an exact dynamic time warping calculation.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
        indices_list : array, optional, default: None
            indices that have to be calculated by this worker

        Returns
        -------
        dist_mat : array
            matrix of all pairwise approximate distances
        path : list
            list of indexes for the inputs x and y for each pairwise comparison
    '''
    paths_list = []
    dist_mat = np.zeros((X.shape[0], X.shape[0]))
    for row, column in indices_list:
        cost, path_lst = fastdtw(X[row][~np.isnan(X[row])], X[column][~np.isnan(X[column])], radius, dist)
        dist_mat[row, column] = cost
        paths_list.append(path_lst)

    return dist_mat, paths_list


def fastdtw_parallel(X, radius=1, dist=None, n_jobs=-1):
    ''' return the approximate distance between N time series of the NxM matrix
        while using parallel computation

        Parameters
        ----------
        X : array
            NxM matrix with all N time series that should be compared, having
            M time points each. In case the time series have different lengths,
            the remaining values can be padded with np.nan values.
        radius : int
            size of neighborhood when expanding the path. A higher value will
            increase the accuracy of the calculation but also increase time
            and memory consumption. A radius equal to the size of x and y will
            yield an exact dynamic time warping calculation.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
        n_jobs : int, optional, default: -1
            The number of cores to use. -1 uses all cores.

        Returns
        -------
        dist_mat : array
            matrix of all pairwise approximate distances
        path : list
            list of indexes for the inputs x and y for each pairwise comparison,
            ordered by cumulative upper triangle count, i.e. having a 5 time series
            matrix gives a 5x5 distance matrix, which has 10 pairwise comparisons.
            The element distance_matrix[0, 4] is the 4th element
            in the path list (index 3 - Python indexing).
            The element distance_matrix[1, 4] is the 7th element
            in the path list (index 6 - Python indexing).
            For the ease of use the get_path function can be used to access the
            elements directly with the distance matrix indices.

        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw_parallel
        >>> X = np.random.randint(1, 40, size=(5, 5))
        >>> fastdtw.fastdtw_parallel(X)
        (array([[ 0., 49., 71., 52., 35.],
        [49.,  0., 69., 89., 39.],
        [71., 69.,  0., 64., 90.],
        [52., 89., 64.,  0., 76.],
        [35., 39., 90., 76.,  0.]]),
         [[[(0, 0), (0, 1), (1, 2), (2, 3), (3, 3), (4, 4)]],
          [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (4, 3), (4, 4)]],
          [[(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 4)]],
          [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (0, 1), (1, 2), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (0, 1), (1, 2), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (1, 1), (2, 2), (2, 3), (3, 4), (4, 4)]],
          [[(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (4, 4)]]])

        Optimization Considerations
        ---------------------------
        This function runs fastest if the following conditions are satisfied:
            1) x and y are either 1 or 2d numpy arrays whose dtype is a
               subtype of np.float
            2) The dist input is a positive integer or None
    '''
    from joblib import Parallel, delayed, cpu_count
    delay = delayed(fastdtw_worker_)
    dist_mat = np.zeros((X.shape[0], X.shape[0]))
    with Parallel(n_jobs=n_jobs) as parallel:
        if n_jobs == -1:
            n_jobs = cpu_count()
        # Check if there are not more processes than jobs available
        n_jobs = int(np.min([n_jobs, ((X.shape[0]**2 - X.shape[0])/2)]))
        # upper triangle indices
        triu_indices = np.vstack(np.triu_indices(X.shape[0], 1)).T
        # lower triangle indices
        tril_indices = np.vstack(np.tril_indices(X.shape[0], -1)).T
        indices = np.vstack((triu_indices, tril_indices))
        indices_lists = np.array_split(indices, n_jobs)
        result = parallel(delay(X, radius, dist, indices_lists[i]) for i in range(n_jobs))
    paths_list = []
    for item in result:
        dist_mat = dist_mat + item[0]
        paths_list.append(item[1])
    return dist_mat, paths_list


def get_path(path_list, row_index, column_index):
    ''' return the path respective to the position index of a pairwise comparison
        within the distance matrix.

        Parameters
        ----------
        path_list : list
            list containing lists of paths generated by the fastdtw_parallel
            method
        row_index : int
            row position of the pairwise comparison in the distance matrix
            generated by the fastdtw_parallel method
        column_index : int
            column position of the pairwise comparison in the distance matrix
            generated by the fastdtw_parallel method

        Returns
        -------
        path : list
            list of indices for the pairwise comparison
    '''
    if row_index == column_index:
        print('There is no warp path between the time series itself!')
        return []

    mat_size = 0.5 + np.sqrt(0.25 + len(path_list)*2)
    # upper triangle indices
    triu_indices = np.vstack(np.triu_indices(mat_size, 1)).T
    # lower triangle indices
    tril_indices = np.vstack(np.tril_indices(mat_size, -1)).T
    index = np.vstack((triu_indices, tril_indices))
    pos = np.argmax((index[:, 0] == row_index) & (index[:, 1] == column_index))
    path = path_list[pos]

    return path


def __difference(a, b):
    return abs(a - b)


def __norm(p):
    return lambda a, b: np.linalg.norm(a - b, p)


def __fastdtw(x, y, radius, dist):
    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y, dist=dist)

    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)
    distance, path = \
        __fastdtw(x_shrinked, y_shrinked, radius=radius, dist=dist)
    window = __expand_window(path, len(x), len(y), radius)
    return __dtw(x, y, window, dist=dist)


def __prep_inputs(x, y, dist):
    x = np.asanyarray(x, dtype='float')
    y = np.asanyarray(y, dtype='float')

    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('second dimension of x and y must be the same')
    if isinstance(dist, numbers.Number) and dist <= 0:
        raise ValueError('dist cannot be a negative integer')

    if dist is None:
        if x.ndim == 1:
            dist = __difference
        else: 
            dist = __norm(p=1)
    elif isinstance(dist, numbers.Number):
        dist = __norm(p=dist)

    return x, y, dist


def dtw(x, y, dist=None):
    ''' return the distance between 2 time series without approximation

        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.

        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y

        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.dtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __dtw(x, y, None, dist)


def dtw_worker_(X, dist=None, indices_list=None):
    ''' allow the partial calculation of pairwise comparisons in the parallel
        computing setting.

         Parameters
        ----------
        X : array
            matrix with all time series that should be compared.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
        indices_list : array, optional, default: None
            indices that have to be calculated by this worker

        Returns
        -------
        dist_mat : array
            matrix of all pairwise distances without approximation
        path : list
            list of indexes for the inputs x and y for each pairwise comparison
    '''
    paths_list = []
    dist_mat = np.zeros((X.shape[0], X.shape[0]))
    for row, column in indices_list:
        cost, path_lst = dtw(X[row][~np.isnan(X[row])], X[column][~np.isnan(X[column])], dist)
        dist_mat[row, column] = cost
        paths_list.append(path_lst)

    return dist_mat, paths_list


def dtw_parallel(X, dist=None, n_jobs=-1):
    ''' return the  distance between N time series of the NxM matrix
        while using parallel computation

        Parameters
        ----------
        X : array
            NxM matrix with all N time series that should be compared, having
            M time points each. In case the time series have different lengths,
            the remaining values can be padded with np.nan values.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
        n_jobs : int, optional, default: -1
            The number of cores to use. -1 uses all cores.

        Returns
        -------
        dist_mat : array
            matrix of all pairwise distances without approximation
        path : list
            list of indexes for the inputs x and y for each pairwise comparison,
            ordered by cumulative upper triangle count, i.e. having a 5 time series
            matrix gives a 5x5 distance matrix, which has 10 pairwise comparisons.
            The element distance_matrix[0, 4] is the 4th element
            in the path list (index 3 - Python indexing).
            The element distance_matrix[1, 4] is the 7th element
            in the path list (index 6 - Python indexing).
            For the ease of use the get_path function can be used to access the
            elements directly with the distance matrix indices.

        Examples
        --------
        >>> import numpy as np
        >>> import dtw_parallel
        >>> X = np.random.randint(1, 40, size=(5, 5))
        >>> fastdtw.dtw_parallel(X)
        (array([[ 0., 49., 71., 52., 35.],
        [49.,  0., 69., 89., 39.],
        [71., 69.,  0., 64., 90.],
        [52., 89., 64.,  0., 76.],
        [35., 39., 90., 76.,  0.]]),
         [[[(0, 0), (0, 1), (1, 2), (2, 3), (3, 3), (4, 4)]],
          [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (4, 3), (4, 4)]],
          [[(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 4)]],
          [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (0, 1), (1, 2), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (0, 1), (1, 2), (2, 2), (3, 3), (4, 4)]],
          [[(0, 0), (1, 1), (2, 2), (2, 3), (3, 4), (4, 4)]],
          [[(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (4, 4)]]])

        Optimization Considerations
        ---------------------------
        This function runs fastest if the following conditions are satisfied:
            1) x and y are either 1 or 2d numpy arrays whose dtype is a
               subtype of np.float
            2) The dist input is a positive integer or None
    '''
    from joblib import Parallel, delayed, cpu_count
    delay = delayed(dtw_worker_)
    dist_mat = np.zeros((X.shape[0], X.shape[0]))
    with Parallel(n_jobs=n_jobs) as parallel:
        if n_jobs == -1:
            n_jobs = cpu_count()
        # Check if there are not more processes than jobs available
        n_jobs = int(np.min([n_jobs, ((X.shape[0]**2 - X.shape[0])/2)]))
        # upper triangle indices
        triu_indices = np.vstack(np.triu_indices(X.shape[0], 1)).T
        # lower triangle indices
        tril_indices = np.vstack(np.tril_indices(X.shape[0], -1)).T
        indices = np.vstack((triu_indices, tril_indices))
        indices_lists = np.array_split(indices, n_jobs)
        result = parallel(delay(X, dist, indices_lists[i]) for i in range(n_jobs))
    paths_list = []
    for item in result:
        dist_mat = dist_mat + item[0]
        paths_list.append(item[1])
    return dist_mat, paths_list


def __dtw(x, y, window, dist):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)
    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)
    for i, j in window:
        dt = dist(x[i-1], y[j-1])
        D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
                      (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])
    path = []
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i-1, j-1))
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return (D[len_x, len_y][0], path)


def __reduce_by_half(x):
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]


def __expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius+1)
                     for b in range(-radius, radius+1)):
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window
