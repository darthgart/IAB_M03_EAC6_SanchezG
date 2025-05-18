"""
Script principal per crear un dataset sintètic, aplicar clustering KMeans i visualitzar-ne els resultats.

Aquest script:
- Crea un dataset amb un nombre configurable d'atributs.
- Aplica clustering KMeans per detectar grups.
- Visualitza les dades i els clusters en 2D i 3D, inclòs un gràfic HTML interactiu.
- Aplica una reducció dimensional amb PCA i revalida el clustering.
- Comprova la qualitat del clustering amb mètriques d'homogeneïtat i completesa.
- Utilitza el mètode del colze (Elbow Method) per ajudar a determinar el nombre òptim de clusters.
"""

import numpy as np
from sklearn.metrics import homogeneity_score, completeness_score
from functions import (
    create_dataset,
    plot_data_attributes,
    model_kmeans,
    predict_clusters,
    plot_clusters,
    plot_clusters3D,
    plot_clusters3D_HTML,
    transform_PCA,
    plot_elbow,
    plot_clusters_PCA,
)

# Creem el dataset amb 4 atributs
X, y = create_dataset(4)

# Mostrem el nombre d'atributs
num_atributs = X.shape[1]
print(f"Número d'atributs: {num_atributs}")

# Mostrem els 5 primers elements de l'atribut 1
print("5 primers elements de l'atribut 1:", X[:5, 0])

# Mostrem els atributs visualment
plot_data_attributes(X)

# Model de clustering amb KMeans
km = model_kmeans(3)
km, y_km = predict_clusters(km, X)

# Gràfiques dels clústers
plot_clusters(km, X, y_km)
plot_clusters3D(km, X, y_km)
plot_clusters3D_HTML(X, y_km)

# Mostrem les 5 primeres dades de cada clúster
for clust in range(3):
    dades_clust = X[y_km == clust]
    print(f"5 primeres dades del clúster {clust + 1}:\n{dades_clust[:5]}")

# PCA: transformació a 2 dimensions
X_PCA = transform_PCA(X, 2)

# Comprovació: haurien de quedar només 2 atributs
assert X_PCA.shape[1] == 2, (
    f"El número d'atributs després de PCA ha de ser 2, però és {X_PCA.shape[1]}"
)
print("Comprovació PCA OK: número d'atributs és 2")

# Mètode del colze per comprovar si 3 clústers és una bona elecció
plot_elbow(X_PCA)

# KMeans sobre les dades transformades amb PCA
km_PCA = model_kmeans(3)
km_PCA, y_km_PCA = predict_clusters(km_PCA, X_PCA)
print("Clusters amb PCA:", y_km_PCA[:10])

# Gràfic dels clústers amb PCA
plot_clusters_PCA(km_PCA, X_PCA, y_km_PCA, show=True, pca=True)

# Comparació d'assignacions amb i sense PCA
print("Clusters sense PCA:", y_km)
print("Clusters amb PCA:", y_km_PCA)

try:
    np.testing.assert_array_equal(y_km, y_km_PCA)
    print("Les assignacions dels clústers sense PCA i amb PCA són iguals")
except AssertionError:
    print("Les assignacions dels clústers sense PCA i amb PCA NO són iguals")

# Funció per calcular homogeneïtat i completesa
def calcular_scores(y_true, y_pred):
    """
    Calcula les puntuacions d'homogeneïtat i completesa entre etiquetes reals i predetes.

    Args:
        y_true (array-like): Etiquetes reals.
        y_pred (array-like): Etiquetes predetes pel model.

    Returns:
        tuple:
            - homogeneïtat (float): Mesura d'homogeneïtat del clustering.
            - completesa (float): Mesura de completesa del clustering.
    """
    homo = homogeneity_score(y_true, y_pred)
    compl = completeness_score(y_true, y_pred)
    return homo, compl

# Càlcul de mètriques per les dades originals i les transformades amb PCA
homo_orig, compl_orig = calcular_scores(y, y_km)
homo_pca, compl_pca = calcular_scores(y, y_km_PCA)

print(f"Homogeneïtat dades originals: {homo_orig:.4f}, Completesa: {compl_orig:.4f}")
print(f"Homogeneïtat dades PCA: {homo_pca:.4f}, Completesa: {compl_pca:.4f}")