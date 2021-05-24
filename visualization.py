import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random

plt.ion()


def visualize_pca_3d(arr, fraction=1, colors=None, other_arr=None):

	if not isinstance(arr, np.ndarray):
		arr = np.array(arr)		
	
	if fraction < 1:
		count = arr.shape[0]
		target_count = int(count * fraction)
		positional_indices = random.sample(range(count), target_count)
	else:
		positional_indices = slice(None, None)

	arr_subset = arr[positional_indices,:]
	colors_subset = np.array(colors)[positional_indices]

	pca = PCA(n_components=3)
	pca_result = pca.fit_transform(arr_subset)

	kwargs = {
	    "xs": pca_result[:, 0],
	    "ys": pca_result[:, 1],
	    "zs": pca_result[:, 2],
	}

	if colors is not None:
		kwargs["c"] = colors_subset
		kwargs["cmap"] = "tab20"

	ax = plt.figure(figsize=(16,10)).gca(projection='3d')
	ax.scatter(**kwargs)	
	ax.set_xlabel('pca-one')
	ax.set_ylabel('pca-two')
	ax.set_zlabel('pca-three')
	
	plt.show()



def visualize_tsne_3d(arr, fraction=1, colors=None):

	if not isinstance(arr, np.ndarray):
		arr = np.array(arr)		
	
	if fraction < 1:
		count = arr.shape[0]
		target_count = int(count * fraction)
		positional_indices = random.sample(range(count), target_count)
	else:
		positional_indices = slice(None, None)

	arr_subset = arr[positional_indices,:]
	colors_subset = np.array(colors)[positional_indices]

	tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(arr_subset)

	kwargs = {
	    "xs": tsne_results[:, 0],
	    "ys": tsne_results[:, 1],
	    "zs": tsne_results[:, 2],
	}

	if colors is not None:
		kwargs["c"] = colors_subset
		kwargs["cmap"] = "tab20"

	ax = plt.figure(figsize=(16,10)).gca(projection='3d')
	ax.scatter(**kwargs)	
	ax.set_xlabel('pca-one')
	ax.set_ylabel('pca-two')
	ax.set_zlabel('pca-three')
	
	plt.show()




