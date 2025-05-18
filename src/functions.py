import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import plotly
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn import metrics
import os

# Funció per assegurar-nos que la carpeta captures existeix
def captures_dir_exists():
    os.makedirs('captures', exist_ok=True)

# P2. Create dataset (modificant la funcio)
def create_dataset(number_features):
    # 3 blobs, cada un amb 250 punts (750 total)
	X, y = make_blobs(
		 n_samples=250*3, n_features=number_features,
		 centers=3, cluster_std=0.75,
		 shuffle=True, random_state=100
	)
	return X, y

# P3.  Plot 2D data attributes - mostrem 4 gràfiques: 
# - attr1 vs attr2
# - attr1 vs attr3
# - attr1 vs attr4
# - attr2 vs attr3
def plot_data_attributes(X, show=True, nom_alumne="Edgar Sanchez"):
	captures_dir_exists()
	fig, axs = plt.subplots(2, 2)
	fig.suptitle(f'Data Attributes - {nom_alumne}', fontsize=12, y=1)
	fig.text(0.5, 0.95, 'title', horizontalalignment="center")
	fig.set_size_inches(6, 6)

	# attr1 vs attr2
	axs[0, 0].set_title('attr1 vs attr2')
	axs[0, 0].scatter(
		X[:, 0], X[:, 1],
		c='white', marker='o',
		edgecolor='black', s=50
	)

	# attr1 vs attr3
	axs[0, 1].set_title('attr1 vs attr3')
	axs[0, 1].scatter(
		X[:, 0], X[:, 2],
		c='white', marker='o',
		edgecolor='black', s=50
	)

	# attr1 vs attr4
	axs[1, 0].set_title('attr1 vs attr4')
	axs[1, 0].scatter(
		X[:, 0], X[:, 3],
		c='white', marker='o',
		edgecolor='black', s=50
	)

	# attr2 vs attr3
	axs[1, 1].set_title('attr2 vs attr3')
	axs[1, 1].scatter(
		X[:, 1], X[:, 2],
		c='white', marker='o',
		edgecolor='black', s=50
	)

	plt.tight_layout()
	nom_fitxer = f'captures/scatters_{nom_alumne.replace(" ", "_")}.png'
	plt.savefig(nom_fitxer)
	if show:
		plt.show()
	else:
		plt.close()
  
	return None

# Creem el model Kmeans
def model_kmeans(num_clusters):
	km = KMeans(
		n_clusters=num_clusters, init='random',
		n_init=10, max_iter=300, 
		tol=1e-04, random_state=0
	)
	
	return km

# Predim els clusters
def predict_clusters(model, X):
	y_km = model.fit_predict(X)
	return model, y_km

# P4. Plot clusters en 2D - mostrem 3 gràfiques:
# - attr1 vs attr2
# - attr1 vs attr3
# - attr1 vs attr4
def plot_clusters(km, X, y_km, show=True, nom_alumne="Edgar Sanchez", pca=True):
	# plot the 3 clusters
	captures_dir_exists()
	fig, axs = plt.subplots(1, 3)
	fig.suptitle(f'Clusters - {nom_alumne}', fontsize=12, y=1)
	fig.set_size_inches(12, 4)

	# attr1 vs attr2
	axs[0].set_title('attr1 vs attr2')
	axs[0].scatter(
		X[y_km == 0, 0], X[y_km == 0, 1],
		s=50, c='lightgreen',
		marker='s', edgecolor='black',
		label='cluster 1'
	)
	axs[0].scatter(
		X[y_km == 1, 0], X[y_km == 1, 1],
		s=50, c='orange',
		marker='o', edgecolor='black',
		label='cluster 2'
	)
	axs[0].scatter(
		X[y_km == 2, 0], X[y_km == 2, 1],
		s=50, c='lightblue',
		marker='v', edgecolor='black',
		label='cluster 3'
	)
	axs[0].scatter(
		km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
		s=250, marker='*',
		c='red', edgecolor='black',
		label='centroids'
	)
	axs[0].legend(scatterpoints=1)

	# attr1 vs attr3
	axs[1].set_title('attr1 vs attr3')
	axs[1].scatter(
		X[y_km == 0, 0], X[y_km == 0, 2],
		s=50, c='lightgreen',
		marker='s', edgecolor='black',
		label='cluster 1'
	)
	axs[1].scatter(
		X[y_km == 1, 0], X[y_km == 1, 2],
		s=50, c='orange',
		marker='o', edgecolor='black',
		label='cluster 2'
	)
	axs[1].scatter(
		X[y_km == 2, 0], X[y_km == 2, 2],
		s=50, c='lightblue',
		marker='v', edgecolor='black',
		label='cluster 3'
	)
	axs[1].scatter(
		km.cluster_centers_[:, 0], km.cluster_centers_[:, 2],
		s=250, marker='*',
		c='red', edgecolor='black',
		label='centroids'
	)
	axs[1].legend(scatterpoints=1)

	# attr1 vs attr4
	axs[2].set_title('attr1 vs attr4')
	axs[2].scatter(
		X[y_km == 0, 0], X[y_km == 0, 3],
		s=50, c='lightgreen',
		marker='s', edgecolor='black',
		label='cluster 1'
	)
	axs[2].scatter(
		X[y_km == 1, 0], X[y_km == 1, 3],
		s=50, c='orange',
		marker='o', edgecolor='black',
		label='cluster 2'
	)
	axs[2].scatter(
		X[y_km == 2, 0], X[y_km == 2, 3],
		s=50, c='lightblue',
		marker='v', edgecolor='black',
		label='cluster 3'
	)
	axs[2].scatter(
		km.cluster_centers_[:, 0], km.cluster_centers_[:, 3],
		s=250, marker='*',
		c='red', edgecolor='black',
		label='centroids'
	)
	axs[2].legend(scatterpoints=1)

	plt.tight_layout()

	nom_fitxer = f'captures/clusters_{nom_alumne.replace(" ", "_")}.png'
	plt.savefig(nom_fitxer)
	if show:
		plt.show()
	else:
		plt.close()

	return None

# P5. Plot clusters en 3D (amb matplotlib) - mostrem els clusters:
# - Cluster 1 (index 0)
# - Cluster 2 (index 1)
# - Cluster 3 (index 2)
# - Mostrem centroides dels clusters
def plot_clusters3D(km, X, y_km, show=True, nom_alumne="Edgar Sanchez"):
	captures_dir_exists()
	fig = plt.figure()
	fig.set_size_inches(6, 6)
	ax = fig.add_subplot(projection='3d')

	ax.set_title(f'Clusters 3D - {nom_alumne}')
	# Cluster 1 (índex 0)
	ax.scatter(
		X[y_km == 0, 0], X[y_km == 0, 1], X[y_km == 0, 2],
		s=50, c='lightgreen',
		marker='s', edgecolor='black',
		label='cluster 1'
	)

	# Cluster 2 (índex 1)
	ax.scatter(
		X[y_km == 1, 0], X[y_km == 1, 1], X[y_km == 1, 2],
		s=50, c='orange',
		marker='o', edgecolor='black',
		label='cluster 2'
	)

	# Cluster 3 (índex 2)
	ax.scatter(
		X[y_km == 2, 0], X[y_km == 2, 1], X[y_km == 2, 2],
		s=50, c='lightblue',
		marker='v', edgecolor='black',
		label='cluster 3'
	)

	# Centroids
	ax.scatter(
		km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[:, 2],
		s=250, marker='*',
		c='red', edgecolor='black',
		label='centroids'
	)

	ax.set_xlabel('attr1')
	ax.set_ylabel('attr2')
	ax.set_zlabel('attr3')

	ax.legend(scatterpoints=1)
	ax.grid()
	plt.title(f'Clusters 3D - {nom_alumne}')
	nom_fitxer = f'captures/clusters3D_{nom_alumne.replace(" ", "_")}.png'
	plt.savefig(nom_fitxer)
	if show:
		plt.show()
	else:
		plt.close()

	return None

# P6. Plot clusters en 3D (HTML amb Plotly) -  mostrem els clusters:
# - Cluster 1 (index 0)
# - Cluster 2 (index 1)
# - Cluster 3 (index 2)
def plot_clusters3D_HTML(X, y_km, nom_alumne="Edgar Sanchez", show=True):
	captures_dir_exists()
	# Configure Plotly to be rendered inline in the notebook.
	plotly.offline.init_notebook_mode()

	# Clúster 1
	cluster1 = go.Scatter3d(
		x=X[y_km == 0, 0],
		y=X[y_km == 0, 1],
		z=X[y_km == 0, 2],
		mode='markers',
		marker=dict(
			size=5,
			opacity=0.8,
			color='red'
		),
		name='Cluster 1'
	)

	# Clúster 2
	cluster2 = go.Scatter3d(
		x=X[y_km == 1, 0],
		y=X[y_km == 1, 1],
		z=X[y_km == 1, 2],
		mode='markers',
		marker=dict(
			size=5,
			opacity=0.8,
			color='orange'
		),
		name='Cluster 2'
	)

	# Clúster 3
	cluster3 = go.Scatter3d(
		x=X[y_km == 2, 0],
		y=X[y_km == 2, 1],
		z=X[y_km == 2, 2],
		mode='markers',
		marker=dict(
			size=5,
			opacity=0.8,
			color='lightblue'
		),
		name='Cluster 3'
	)

	layout = go.Layout(
		title=f'Clusters 3D HTML - {nom_alumne}',
		margin=dict(l=0, r=0, b=0, t=50),
		scene=dict(
			xaxis_title='attr1',
			yaxis_title='attr2',
			zaxis_title='attr3'
		)
	)

	data = [cluster1, cluster2, cluster3]

	fig = go.Figure(data=data, layout=layout)

	# Renderitzem la gràfica interactiva
	plotly.offline.iplot(fig)

	# Guardem el fitxer html a la carpeta captures amb nom personalitzat
	filename = f'captures/clusters3D_HTML_{nom_alumne}.html'
	plotly.offline.plot(fig, filename=filename, auto_open=show)

	return None

# Transformem usant PCA
def transform_PCA(X, num_components):
	X_PCA = PCA(n_components=num_components).fit_transform(X)
	return X_PCA

# P7. Implementem el plot Elbow Method
def plot_elbow(X_PCA, show=True, nom_alumne="Edgar Sanchez"):
	captures_dir_exists()
	inertias = []
	k_range = range(1, 10)

	for k in k_range:
		km = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, random_state=0)
		km.fit(X_PCA)
		inertias.append(km.inertia_)

	plt.figure(figsize=(6, 4))
	plt.plot(k_range, inertias, marker='o')
	plt.xlabel('Number of clusters')
	plt.ylabel('Inertia')
	plt.title(f'Elbow Method - {nom_alumne}')
	plt.grid(True)
	nom_fitxer = f'captures/elbow_{nom_alumne.replace(" ", "_")}.png'
	plt.savefig(nom_fitxer)
	if show:
		plt.show()
	else:
		plt.close()

	return None

# P8 - Creació i implementació de plot_clusters_PCA (mostra els grafics afegint dos arguments nous)
def plot_clusters_PCA(km, X, y_km, show, nom_alumne="Edgar Sanchez", pca=False):
    captures_dir_exists()
    fig, axs = plt.subplots(1, 3)
    
    # Definir títol segons si s'usa PCA o no
    if pca:
        titol = f'Clusters PCA - {nom_alumne}'
        fitxer = f'captures/clusters_PCA_{nom_alumne}.png'
    else:
        titol = f'Clusters - {nom_alumne}'
        fitxer = f'captures/clusters_{nom_alumne}.png'
        
    fig.suptitle(titol, fontsize=12, y=1)
    fig.set_size_inches(12, 4)

    axs[0].set_title('attr1 vs attr2')
    axs[0].scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )
    axs[0].scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )
    axs[0].scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )
    axs[0].scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    axs[0].legend(scatterpoints=1)
    
    if X.shape[1] > 2:
        axs[1].set_title('attr1 vs attr3')
        axs[1].scatter(
            X[y_km == 0, 0], X[y_km == 0, 2],
            s=50, c='lightgreen', marker='s', edgecolor='black'
        )
        axs[1].scatter(
            X[y_km == 1, 0], X[y_km == 1, 2],
            s=50, c='orange', marker='o', edgecolor='black'
        )
        axs[1].scatter(
            X[y_km == 2, 0], X[y_km == 2, 2],
            s=50, c='lightblue', marker='v', edgecolor='black'
        )
    
    if X.shape[1] > 3:
        axs[2].set_title('attr1 vs attr4')
        axs[2].scatter(
            X[y_km == 0, 0], X[y_km == 0, 3],
            s=50, c='lightgreen', marker='s', edgecolor='black'
        )
        axs[2].scatter(
            X[y_km == 1, 0], X[y_km == 1, 3],
            s=50, c='orange', marker='o', edgecolor='black'
        )
        axs[2].scatter(
            X[y_km == 2, 0], X[y_km == 2, 3],
            s=50, c='lightblue', marker='v', edgecolor='black'
        )
        
    plt.savefig(fitxer)
    if show:
        plt.show()
    else:
        plt.close()
        
    return None