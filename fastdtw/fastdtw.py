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


def fastdtw(x, y, radius=1, dist=None,
            b_partial_start=False, b_partial_end=False, radius_x=4):
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
        b_partial_start: bool
            If True, calculate a partial match where the start of path does
            not point to the start of x. Otherwise, the start of path points
            to the start of x.
        b_partial_end: bool
            If True, calculate a partial match where the end of path does not
            point to the end of x. Otherwise, the end of path points to the
            end of x.
        radius_x: int
            When b_partial_{start|end} is True, radius_x is used for x-axis
            calculation instead of radius.

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
    x, y, dist, radius_x = __prep_inputs(x, y, radius, dist,
                                         b_partial_start, b_partial_end,
                                         radius_x)
    return __fastdtw(x, y, radius, dist,
                     b_partial_start, b_partial_end, radius_x)


def __difference(a, b):
    return abs(a - b)


def __norm(p):
    return lambda a, b: np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), p)


def __fastdtw(x, y, radius, dist,
              b_partial_start, b_partial_end, radius_x):
    min_time_size = radius + 2
    min_time_size_x = radius_x + 2 if b_partial_start or b_partial_end \
        else min_time_size

    if len(x) < min_time_size_x or len(y) < min_time_size:
        return dtw(x, y, dist=dist,
                   b_partial_start=b_partial_start,
                   b_partial_end=b_partial_end)

    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)
    distance, path = \
        __fastdtw(x_shrinked, y_shrinked, radius=radius, dist=dist,
                  b_partial_start=b_partial_start,
                  b_partial_end=b_partial_end,
                  radius_x=radius_x)
    window = __expand_window(path, len(x), len(y), radius, radius_x)
    return __dtw(x, y, window, dist=dist,
                 b_partial_start=b_partial_start,
                 b_partial_end=b_partial_end)


def __prep_inputs(x, y, radius, dist,
                  b_partial_start, b_partial_end, radius_x):
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

    if not (b_partial_start or b_partial_end):
        radius_x = radius

    return x, y, dist, radius_x


def dtw(x, y, dist=None, b_partial_start=False, b_partial_end=False):
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
        b_partial_start: bool
            If True, calculate a partial match where the start of path does
            not point to the start of x. Otherwise, the start of path points
            to the start of x.
        b_partial_end: bool
            If True, calculate a partial match where the end of path does not
            point to the end of x. Otherwise, the end of path points to the
            end of x.

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
    x, y, dist, _ = __prep_inputs(x, y, None, dist,
                                  b_partial_start, b_partial_end,
                                  None)
    return __dtw(x, y, None, dist, b_partial_start, b_partial_end)


def __dtw(x, y, window, dist, b_partial_start, b_partial_end):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)
    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)
    if b_partial_start:
        for i in range(1, len_x+1):
            D[i, 0] = (0, 0, 0)
    for i, j in window:
        dt = dist(x[i-1], y[j-1])
        D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
                      (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])
    path = []
    t_end = len_x
    if b_partial_end:
        distance_min = float('inf')
        for i in range(1, len_x+1):
            if distance_min > D[i, len_y][0]:
                distance_min = D[i, len_y][0]
                t_end = i
    i, j = t_end, len_y
    while not (i == j == 0):
        path.append((i-1, j-1))
        if b_partial_start and j == 1:
            break;
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return (D[t_end, len_y][0], path)


def __reduce_by_half(x):
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]


def __expand_window(path, len_x, len_y, radius, radius_x):
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius_x, radius_x+1)
                     for b in range(-radius, radius+1)):
            if a >= 0 and b >= 0:
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
        start_j = new_start_j if new_start_j is not None else 0

    return window
