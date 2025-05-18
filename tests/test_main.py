import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from functions import create_dataset, transform_PCA, model_kmeans, predict_clusters

def test_num_atributs_dades_originals():
    X, _ = create_dataset(4)
    assert X.shape[1] == 4, f"Esperat 4 atributs, però hi ha {X.shape[1]}"

def test_num_atributs_dades_pca():
    X, _ = create_dataset(4)
    X_PCA = transform_PCA(X, 2)
    assert X_PCA.shape[1] == 2, f"Esperat 2 atributs després de PCA, però hi ha {X_PCA.shape[1]}"

def test_assignacio_clusters_igual():
    X, _ = create_dataset(4)
    km = model_kmeans(3)
    km, y_km = predict_clusters(km, X)

    X_PCA = transform_PCA(X, 2)
    km_pca = model_kmeans(3)
    km_pca, y_km_pca = predict_clusters(km_pca, X_PCA)

    # Comprovem que les assignacions siguin iguals
    np.testing.assert_array_equal(y_km, y_km_pca)
