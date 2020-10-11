#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:19:50 2020

@author: williamglazer
"""

import numpy as np
from SimilarityMetric import PearsonUserSimilarity
from constants import STANDARDIZATION


def rating_mean(r):
    """


    Parameters
    ----------
    r : np.array(int)
        Array of ratings.

    Returns
    -------
    mean : float
        Mean of all ratings.

    """
    return np.mean(r[r != 0])



def rating_std_deviation(r):
    """


    Parameters
    ----------
    r : np.array(int)
        Array of ratings.

    Returns
    -------
    std : float
        Standard deviation of all ratings.

    """
    return np.std(r[r != 0], ddof=1)



class PredictionAlgorithm:

    """
    Base abstract class for all prediction algorithms
    """

    def predict(self, R, wanted_prediction_indices):
        pass


class ItemMean(PredictionAlgorithm):
    def __repr__(self):
        return "Item Mean"

    def predict(self, R, wanted_prediction_indices):
        """


        Parameters
        ----------
        R : np.ndarray[user][item]
            Matrix of ratings for each user and item
            missing ratings are 0.
        wanted_prediction_indices : array(int)
            Indices to predict.

        Returns
        -------
        R_hat : np.ndarray[user][item]
            Matrix of ratings for each user and item
            with wanted_prediction_indices estimated with item mean
            missing ratings are 0..

        """
        R_hat = R.copy()

        for i, r_i in zip(range(R.T.shape[0]), R.T):

            try:
                i_mean = rating_mean(r_i)
            except RuntimeWarning:
                i_mean = 3

            r_i[r_i == 0] = i_mean
            R_hat.T[i] = r_i
        return R_hat




class UserBasedKNearestNeighbour(PredictionAlgorithm):

    def __init__(self, sim_metric=PearsonUserSimilarity(), k=10, std='normalization'):
        """


        Parameters
        ----------
        sim_metric : SimilarityMetric, optional
            Object to use for similarity metric.
            The default is PearsonUserSimilarity().
        k : int, optional
            Number of neighbours used for prediction.
            The default is 10.
        std : TYPE, STANDARDIZATION
            Type of standardization to use for prediction.
            The default is 'normalization'.

        Raises
        ------
        ValueError
            Detects error in std parameter specification.

        """
        if std not in STANDARDIZATION: raise ValueError
        self.sim_metric = sim_metric
        self.k = k
        self.std = std

    def __repr__(self):
        return "UBKNN with {} similarity".format(self.sim_metric)


    def ratings_statistics(self, r_u):
        """


        Parameters
        ----------
        r_u : np.array(int)
            Ratings for a given user.

        Returns
        -------
        u_mean : float
            Mean of the user ratings.
            Depends on specification of normalization.
        u_std : float
            Standard deviaiton of the user ratings.
            Depends on specification of normalization.

        """
        if self.std == 'normalization':
            u_mean = rating_mean(r_u)
            u_std = rating_std_deviation(r_u)

        elif self.std == 'mean':
            u_mean = rating_mean(r_u)
            u_std = 1

        elif self.std == 'none':
            u_mean = 0
            u_std = 1

        return u_mean, u_std


    def most_similar_users(self, R, r_u, r_i):
        """


        Parameters
        ----------
        R : np.ndarray[user][item]
            m.
        r_u : np.array[user]
            Ratings for a given user.
        r_i : np.array[item]
            Ratings for a given item.

        Returns
        -------
        most_similar_users : np.array[user]
            List of most similar user index.
        similarities : np.sarray[similarity]
            List of highest similarities.

        """
        I_u_intersect_I_v = np.nonzero(r_i)[0]
        n = len(I_u_intersect_I_v)

        sim = np.array([])
        for v in I_u_intersect_I_v:
            r_v = R[v]
            item_avg = np.average(r_i)
            sim_uv = self.sim_metric.evaluate(r_u, r_v, item_avg)
            sim = np.append(sim, sim_uv)

        k = min(n, self.k)

        knn = np.argpartition(sim, -k)[-k:]
        most_similar_users = I_u_intersect_I_v[knn]
        similarities = sim[knn]
        return most_similar_users, similarities


    def predict(self, R, wanted_prediction_indices):
        """


        Parameters
        ----------
        R : np.ndarray[user][item]
            Matrix of ratings for each user and item
            missing ratings are 0.
        wanted_prediction_indices : array(int)
            Indices to predict.

        Returns
        -------
        R_hat : np.ndarray[user][item]
            Matrix of ratings for each user and item
            with wanted_prediction_indices estimated with UBKNN
            missing ratings are 0..

        """
        progress = 0
        process_length = len(wanted_prediction_indices)
        missing = 0

        R_hat = R.copy()

        for u, i in wanted_prediction_indices:
            progress += 1 / process_length * 100
            print('\r UBKNN {:.2f}%'.format(progress), end='')

            r_u = R[u]
            r_i = R.T[i]
            u_mean, u_std = self.ratings_statistics(r_u)

            is_item_rated = R[u, i] != 0

            if not is_item_rated:
                most_similar_users, similarities = \
                    self.most_similar_users(R, r_u, r_i)

                nominator, denominator = 0, 0

                for v, sim_uv in zip(most_similar_users, similarities):

                    r_vi = R[v, i]
                    r_v = R[v]
                    v_mean, v_std = self.ratings_statistics(r_v)

                    z_vi = (r_vi - v_mean) / v_std

                    nominator += sim_uv * z_vi
                    denominator += abs(sim_uv)

                try:
                    r_hat = u_mean + u_std * (nominator / denominator)

                except ZeroDivisionError:
                    r_hat = u_mean + u_std
                    missing+=1
                except RuntimeWarning:
                    r_hat = u_mean + u_std
                    missing+=1

                if np.isnan(r_hat):
                    raise  ValueError("Predicting nan")
                if r_hat == 0 and most_similar_users.size != 0:
                    raise  RuntimeWarning("0 prediction")

                if r_hat < 1: r_hat = 1
                if r_hat > 5: r_hat = 5
                R_hat[u, i] = r_hat

        return R_hat


# Inspired by https://towardsdatascience.com/music-artist-recommender-system-using-stochastic-gradient-descent-machine-learning-from-scratch-5f2f1aae972c
class LatentMatrixFactorization(PredictionAlgorithm):

    def __init__(self, ground_truth, n_epochs=200, n_latent_features=3, lmbda=0.1, learning_rate=0.001):
        """


        Parameters
        ----------
        ground_truth : np.ndarray[int][int]
            True raating matrix with elements {1..5} and 0 being absent rating.
        n_epochs : int, optional
            Number of time the algorithm trains. The default is 200.
        n_latent_features : int, optional
            size of latent matrices. The default is 3.
        lmbda : float, optional
            Regularization parameter. The default is 0.1.
        learning_rate : float, optional
            Rate the algorithm will learn. The default is 0.001.

        Returns
        -------
        None.

        """
        self.n_epochs = n_epochs
        self.n_latent_features = n_latent_features
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.ground_truth = ground_truth

    def __repr__(self):
        return "Latent Matrix Factorization"

    def predictions(self, P, Q):
        """


        Parameters
        ----------
        P : np.ndarray[int][int]
            Latent user matrix.
        Q : np.ndarray[int][int]
            Latent item matrix.

        Returns
        -------
        preidction: float
            rating predicted by the matrix.

        """
        return np.dot(P.T, Q)

    def mse(self, pred, truth):
        """


        Parameters
        ----------
        pred : np.ndarray[int][int]
            all predictions made by the algorithm.
        truth : np.ndarray[int][int]
            all actual values of the matrix.

        Returns
        -------
        mse : float
            average mean squared error for the ratings.

        """
        pred = pred[truth.nonzero()].flatten()
        truth = truth[truth.nonzero()].flatten()
        return ((pred - truth)**2).mean(axis=None)

    def fit(self, R):
        """


        Parameters
        ----------
        R : np.ndarray[int][int]
            matrix to fit the algorithm to.

        Returns
        -------
        self : LatentMatrixFactorization
            returns a fitted version of itself.

        """
        m, n = R.shape

        self.P = 3 * np.random.rand(self.n_latent_features, m)
        self.Q = 3 * np.random.rand(self.n_latent_features, n)

        self.train_mse = []
        self.val_mse = []

        users, items = R.nonzero()

        for epoch in range(self.n_epochs):
            progress = epoch / self.n_epochs * 100
            print('\r SGD {:.2f}%'.format(progress), end='')

            for u, i in zip(users, items):
                error = R[u, i] - self.predictions(self.P[:, u], self.Q[:, i])
                self.P[:, u] += self.learning_rate * \
                    (error * self.Q[:, i] - self.lmbda * self.P[:, u])
                self.Q[:, i] += self.learning_rate * \
                    (error * self.P[:, u] - self.lmbda * self.Q[:, i])

            R_hat = self.predictions(self.P, self.Q)
            self.train_mse.append(self.mse(R_hat, R))
            self.val_mse.append(self.mse(R_hat, self.ground_truth - R))

        return self


    def predict(self, R, wanted_prediction_indices):
        """


        Parameters
        ----------
        R : np.ndarray[int][int]
            matrix to fit the algorithm.
        wanted_prediction_indices : np.ndarray[int]
            dummy parameter to fit the validation loop.

        Returns
        -------
        R_hat : np.ndarray[int][int]
            predictions.

        """
        self.fit(R)
        R_hat = self.predictions(self.P, self.Q)
        return R_hat
