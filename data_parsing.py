#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:59:38 2020

@author: williamglazer
"""



import os
import numpy as np
from constants import DATA_DIMENSIONS
cwd = os.getcwd()

def parse(line):
    """


    Parameters
    ----------
    line : string
        string of: user_id, ittem_id, rating, timestamp
        split by tab with a trailing newline.

    Returns
    -------
    rating_link_array_no_timestamp : int[][3]
        array of [user_id, item_id, rating]
        where
            user_id in {1..943}
            item_id in {1..1682}
            rating  in {1..5}.

    """
    line_no_trailing_char = line.rstrip('\n')
    rating_link_array = line_no_trailing_char.split('\t')
    rating_link_array_no_timestamp = rating_link_array[:-1]
    int_link_array = [int(rating_link_array_no_timestamp[0])-1,
                      int(rating_link_array_no_timestamp[1])-1,
                      int(rating_link_array_no_timestamp[2])
                      ]
    return int_link_array



def create_matrix_from_ratings(links):
    """


    Parameters
    ----------
    links : int[][3]
        array of [user_id, item_id, rating]
        where
            user_id in {1..943}
            item_id in {1..1682}
            rating  in {1..5}.

    Returns
    -------
    R : np.matrix(int, int)
        matrix of [users, items].

    """

    R = np.zeros((
        DATA_DIMENSIONS['users'],
        DATA_DIMENSIONS['items']))
    for u, i, r_ui in links:
        R[u, i] = r_ui
    return R
