import csv
import numpy as np
import sys

# My anime list directories (1 indexed)
myanimelist_anime = '../datasets/MyAnimeList/anime.csv'
myanimelist_rating = '../datasets/MyAnimeList/rating.csv'
len_user = 73517
len_anime = 34528

def get_structs(archive):
	with open(archive) as csvfile:
		reader = csv.reader(csvfile)
		id2name = [""]*(len_anime)
		name2id = dict()
		flag = False
		for row in reader:
			if flag is True:
				idx = int(row[0])
				name2id[row[1]] = idx
				id2name[idx] = row[1]
			else:
				flag = True
	return name2id, id2name

#This function read a CSV file and return a list with each row file per element
def read_csv(archive, feature_names=False):
	with open(archive) as csvfile:
		reader = csv.reader(csvfile)
		data = list()
		for row in reader:
			data.append(list(map(lambda word: word.lower(), row)))

	# Remove first row
	if feature_names is False:
		return np.array(data[1:]).astype(int)
	return np.array(data).astype(int)


def load(directory, tam="large", feature_names=False):
	name2id, id2name = get_structs(myanimelist_anime)
	rating = read_csv(directory, feature_names)
	return name2id, id2name, rating