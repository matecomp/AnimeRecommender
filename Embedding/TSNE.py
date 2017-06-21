import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(24, 24))  #in inches

  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y, color="green")
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  # create database folder
  if not os.path.exists("Embedding/images/"):
    os.makedirs("Embedding/images/")
  plt.savefig("Embedding/images/"+filename)

def saveTSNE(final_embeddings, reverse_dictionary, n):
	try:
		from sklearn.manifold import TSNE

		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
		plot_only = n
		low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
		labels = [reverse_dictionary[i] for i in range(plot_only)]
		plot_with_labels(low_dim_embs, labels, filename="tsne.png")

	except ImportError:
		print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
