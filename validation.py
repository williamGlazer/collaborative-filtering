#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:23:17 2020

@author: williamglazer
"""
import numpy as np
from constants import K_FOLD

def get_random_matrix_index_partitions(R):
    """


    Parameters
    ----------
    R : np.ndarray(943, 1682)
        matrix with ratings in {1..5} 0 being absent rating.

    Returns
    -------
    partitions : np.ndarray[K_FOLD][10 000] of (user, item) positions
        returns K_FOLD partitions of mutually exclusive nonzero
        indices in the matrix.

    """
    u, i = R.nonzero()
    nonzero_indices = list(zip(u, i))

    np.random.shuffle(nonzero_indices)
    partitions = np.array_split(nonzero_indices, K_FOLD)
    return partitions


def crossfold_validation(R, prediction_algo, k_fold):
    """


    Parameters
    ----------
    R : np.ndarray(943, 1682)
        matrix with ratings in {1..5} 0 being absent rating.
    prediction_fct : function(np.ndarray(943, 1682))
        function to use for prediction by feeding ratings.

    Returns
    -------
    mse : float
        mean square error computed by
        AVG(1/n*(pred-truth)^2) over all iterations.
    mae : float
        mean average error computed by
        AVG(1/n*|pred-truth|) over all iterations..

    """
    partition_indices = get_random_matrix_index_partitions(R)
    fold_mae, fold_mse = [], []

    progress = 0
    missing = 0

    print("\n\nCurrently validating {}".format(prediction_algo))

    for partition in partition_indices[0:k_fold]:

        progress += 1
        print('\r Validation Fold {}/{}\n'.format(progress, k_fold), end='')

        R_train = R.copy()

        # Set partition to 0
        for u, i in partition:
            R_train[u, i] = 0
        R_hat = prediction_algo.predict(R_train, partition)

        # Evaluate metric for pred/truth difference
        mae, mse = [], []
        for u, i in partition:
            r = R[u, i]
            r_hat = R_hat[u, i]

            is_enough_information_for_prediction = \
                not np.isnan(r_hat) and not r_hat == 0
            if is_enough_information_for_prediction:
                mse.append((r_hat - r)**2)
                mae.append(abs(r_hat - r))
            else:
                missing += 1

        fold_mse.append(np.mean(mse))
        fold_mae.append(np.mean(mae))

    print("\nDue to random fold partition, unable to predict {} values".format(missing))

    total_mse = np.mean(fold_mse)
    total_mae = np.mean(fold_mae)
    return {"mse":total_mse, "mae":total_mae}
