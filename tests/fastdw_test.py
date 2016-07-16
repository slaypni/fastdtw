#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

import unittest

import numpy as np

from fastdtw._fastdtw import dtw as dtw_c
from fastdtw._fastdtw import fastdtw as fastdtw_c

from fastdtw.fastdtw import dtw as dtw_p
from fastdtw.fastdtw import fastdtw as fastdtw_p

class FastdtwTest(unittest.TestCase):

    def setUp(self):
        self.x_1d = [1, 2, 3, 4, 5]
        self.y_1d = [2, 3, 4]
        self.x_2d = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        self.y_2d = np.array([[2, 2], [3, 3], [4, 4]])
        self.dist_2d = lambda a, b: sum((a - b) ** 2) ** 0.5

    def test_1d_fastdtw(self):
        distance_c = fastdtw_c(self.x_1d, self.y_1d)[0]
        distance_p = fastdtw_p(self.x_1d, self.y_1d)[0]
        self.assertEqual(distance_c, 2)
        self.assertEqual(distance_c, distance_p)

    def test_1d_dtw(self):
        distance_c = dtw_c(self.x_1d, self.y_1d)[0]
        distance_p = dtw_p(self.x_1d, self.y_1d)[0]
        self.assertEqual(distance_c, 2)
        self.assertEqual(distance_c, distance_p)

    def test_2d_fastdtw(self):
        distance_c = fastdtw_c(self.x_2d, self.y_2d, dist=self.dist_2d)[0]
        distance_p = fastdtw_p(self.x_2d, self.y_2d, dist=self.dist_2d)[0]
        self.assertAlmostEqual(distance_c, ((1+1)**0.5)*2)
        self.assertEqual(distance_c, distance_p)


if __name__ == '__main__':
    unittest.main()

