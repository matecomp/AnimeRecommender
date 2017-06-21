from RecommenderSystem.RecSys import *
from Datasets.loader import *
# from Embedding.TSNE import *

#Using model
def _start_shell(local_ns=None):
# An interactive shell is useful for debugging/development.
	import IPython
	user_ns = {}
	if local_ns:
		user_ns.update(local_ns)
	user_ns.update(globals())
	IPython.start_ipython(argv=[], user_ns=user_ns)

if __name__ == '__main__':
	myanimelist_anime = 'Datasets/MyAnimeList/anime.csv'
	myanimelist_rating = 'Datasets/MyAnimeList/rating.csv'
	feature_ignore = True
	size_ratings = 10000

	anime_generator, rating_generator = load(myanimelist_anime,
											myanimelist_rating,
										 	feature_ignore, size_ratings)

	itens = next(anime_generator)
	name2id, id2name, iditem2id = preprocessing(itens)

	#Config
	features = 100
	reg = 1e-06
	eta = 3e-01
	epochs = 50

	model = RecSys(name2id, id2name, iditem2id, rating_generator, features, reg, eta, epochs)
	_start_shell()

	# saveTSNE(model.W,model.id2name,50)
