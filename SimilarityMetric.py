#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:55:44 2020

@author: williamglazer
"""

from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import numpy as np

class SimilarityMetric:
    def __init__(self):
        pass

    def evaluate(self, r_u, r_v, item_avg):
        pass

    def item_intersection(self, r_u, r_v):
        i_u = np.nonzero(r_u)[0]
        i_v = np.nonzero(r_v)[0]
        i_uv = np.intersect1d(i_u, i_v)
        r_u_intersect = r_u[i_uv]
        r_v_intersect = r_v[i_uv]
        return i_uv, r_u_intersect, r_v_intersect



class PearsonUserSimilarity(SimilarityMetric):
    def __init__(self, is_scaling=False):
        self.is_scaling = is_scaling
        return

    def __repr__(self):
        return "Pearson (scaling {})".format(self.is_scaling)

    def evaluate(self, r_u, r_v, item_avg):
        # Get intersection and user rating mean
        r_uv, r_u, r_v = self.item_intersection(r_u, r_v)
        try:
            pearson_sim = pearsonr(r_u, r_v)[0]
        except ValueError:
            pearson_sim = 0
        except RuntimeWarning:
            pearson_sim = 0

        if self.is_scaling:
            scaling = min(len(r_uv)/50, 1)
            pearson_sim *= scaling

        return pearson_sim



class RawCosineUserSimilarity(SimilarityMetric):
    def __init__(self):
        return

    def __repr__(self):
        return "Raw Cosine"

    def evaluate(self, r_u, r_v, item_avg):

        try:
            cosine_similarity = 1 - cosine(r_u, r_v)
        except ValueError:
            cosine_similarity = 0
        except RuntimeWarning:
            cosine_similarity = 0

        return cosine_similarity



class JaccardUserSimilarity(SimilarityMetric):
    def __init__(self):
        return

    def __repr__(self):
        return "Jaccard"

    def evaluate(self, r_u, r_v, item_avg):
        # Get intersection and user rating mean
        r_uv, r_u, r_v =\
            self.item_intersection(r_u, r_v)

        try:
            jaccard_similarity = len(r_uv) / (len(r_u)+len(r_v))
        except ZeroDivisionError:
            jaccard_similarity = 0

        return jaccard_similarity



class PIPUserSimilarity(SimilarityMetric):
    def __init__(self, is_scaling=False, r_max=5, r_min=1):
        self.r_max = r_max
        self.r_min = r_min
        self.r_med = (self.r_max - r_min) / 2
        self.is_scaling = is_scaling
        return

    def __repr__(self):
        return "PIP"


    def agreement(self, r_u, r_v):
        is_smaller_median = r_u < self.r_med and r_v < self.r_med
        is_bigger_median = r_u > self.r_med and r_v > self.r_med
        is_agreeing = is_smaller_median or is_bigger_median
        return is_agreeing


    def distance(self, r_u, r_v):
        distance = abs(r_u - r_v)
        if not self.agreement(r_u, r_v):
            distance *= 2
        return distance


    def proximity(self, r_u, r_v):
        return ((2*(self.r_max - self.r_min) + 1) - self.distance(r_u, r_v))**2


    def impact(self, r_u, r_v):
        impact = (abs(r_u-self.r_med)+1)*(abs(r_v-self.r_med)+1)
        if not self.agreement(r_u, r_v):
            impact = impact**-1
        return impact

    def popularity(self, r_u, r_v, item_avg):
        is_smaller_item_rating = r_u < item_avg and r_v < item_avg
        is_bigger_item_rating = r_u > item_avg and r_v > item_avg
        if is_bigger_item_rating or is_smaller_item_rating:
            popularity = 1 + ((r_u+r_v)/2 - item_avg)
        else:
            popularity = 1
        return popularity

    def PIP(self, r_u, r_v, item_avg):
        return self.proximity(r_u, r_v) * self.impact(r_u, r_v) * self.popularity(r_u, r_v, item_avg)

    def evaluate(self, r_u, r_v, item_avg):
        i_uv, r_u_intersect, r_v_intersect =\
            self.item_intersection(r_u, r_v)
        sim = 0
        for r_u, r_v in zip(r_u_intersect, r_v_intersect):
            sim += self.PIP(r_u, r_v, item_avg)
        if self.is_scaling:
            scaling = min(len(i_uv)/50, 1)
            sim *= scaling
        return sim
