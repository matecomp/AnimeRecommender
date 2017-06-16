from RecommenderSystem.RecSys import *
from Datasets.loader import *

if __name__ == '__main__':
	myanimelist_anime = 'Datasets/MyAnimeList/anime.csv'
	myanimelist_rating = 'Datasets/MyAnimeList/rating.csv'
	feature_ignore = True
	size_ratings = 10000
	
	anime_generator, rating_generator = load(myanimelist_anime,
											myanimelist_rating,
										 	feature_ignore, size_ratings)
	# anime_generator = read_csv(myanimelist_anime,True)
	print(next(rating_generator))
	# print(next(anime_generator))
	
	# #Config
	# features = 100
	# reg = 1e-06
	# eta = 3e-01
	# epochs = 50

	# model = RecSys(features, name2id, id2name, rating, reg, eta, epochs)
	# _start_shell()