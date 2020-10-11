#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:58:55 2020

@author: williamglazer
"""

from SimilarityMetric import PearsonUserSimilarity
from SimilarityMetric import RawCosineUserSimilarity
from SimilarityMetric import PIPUserSimilarity
from SimilarityMetric import JaccardUserSimilarity
from data_parsing import parse, create_matrix_from_ratings
from PredictionAlgorithm import ItemMean
from PredictionAlgorithm import UserBasedKNearestNeighbour
from PredictionAlgorithm import LatentMatrixFactorization
from validation import crossfold_validation
from constants import DATA_FILE
import warnings
warnings.filterwarnings("error")


# Function Requested
def predict(R):
    algo = LatentMatrixFactorization(
        ground_truth=rating_matrix,
        n_epochs=100,
        n_latent_features=3,
        lmbda=0.05,
        learning_rate=0.01
    )
    R_hat = algo.predict()
    return R_hat


# open files
f = open(DATA_FILE, "r")
data_array = [parse(line) for line in f.readlines()]
f.close()
rating_matrix = create_matrix_from_ratings(data_array)

# Baseline
algo = ItemMean()
accuracy = crossfold_validation(rating_matrix, algo, k_fold=10)
print(accuracy)

# Best Algo
# Matrix Fact
algo = LatentMatrixFactorization(
    ground_truth=rating_matrix,
    n_epochs=100,
    n_latent_features=3,
    lmbda=0.05,
    learning_rate=0.01
)
accuracy = crossfold_validation(rating_matrix, algo, k_fold=10)
print(accuracy)
R_hat = algo.predict(rating_matrix)


# UBKNN
metric = JaccardUserSimilarity() # can be any type of metric
metric = PearsonUserSimilarity() # can be any type of metric
metric = PIPUserSimilarity() # can be any type of metric
metric = RawCosineUserSimilarity() # can be any type of metric
algo = UserBasedKNearestNeighbour(metric=metric, k=15, std='normalization')
accuracy = crossfold_validation(rating_matrix, algo, k_fold=10)
print(accuracy)

R_hat = algo.predict(rating_matrix)
