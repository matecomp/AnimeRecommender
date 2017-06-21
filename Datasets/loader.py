import csv
import numpy as np
from copy import deepcopy
import sys

# My anime list directories (1 indexed)
myanimelist_anime = '../Datasets/MyAnimeList/anime.csv'
myanimelist_rating = '../Datasets/MyAnimeList/rating.csv'
len_user = 73517
len_anime = 12294
name = "MyAnimeList"
link = "https://storage.googleapis.com/kaggle-datasets/571/1094/anime-recommendations-database.zip?GoogleAccessId=datasets@kaggle-161607.iam.gserviceaccount.com&Expires=1497664649&Signature=E9VT09SgXL3CV%2Fhy2HZR%2BlaepIY6ZIEuTJL27rgUDimbMWjiK7mkMnyw%2FJEhe%2B88v%2FeEueiEUpf6RABmJURlMfUl07gpm4uBqrr6N4lPjNsvDwEvXWWIWXVN%2FP9Gg2f3WmrzgEkDBNva5MF%2BEMDYVzj62Hcwhw5VE8Q3lOz8uNOmXnBHvEnyCh7HVnXuIa2Q6ZBvWnug%2FaUWcAwr1p6uovzcmdT8NhOjPd%2FdoVp3pZbzsnXHMtI1ltawApnAID6bJtnJM5VdRhOBqn%2BpDJw8sRuhAnMNqHszuVNDJrEbuZNWm8ZXCLOtyqWi0DNsusnAPPtvhMANsg9IDT%2BeHjJmhw%3D%3D"


def preprocessing(itens):
	id2name = [""]*(len_anime)
	name2id = dict()
	iditem2id = dict()

	for i, item in enumerate(itens):
		idx = int(item[0])
		iditem2id[idx] = i
		name2id[item[1]] = i
		id2name[i] = item[1]

	return name2id, id2name, iditem2id

#This function read a CSV file and return a list with each row file per element
def read_csv(archive, feature_ignore=True, size=None):
	# when size is not defiened we use all data
	if size is None:
		with open(archive) as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			size = sum([1 for row in reader])-1
			print(size)

	with open(archive) as csvfile:
		reader = csv.reader(csvfile)
		data = list()
		copy = list()
		indexed = 1
		for idx, row in enumerate(reader):
			if feature_ignore is True:
				feature_ignore = False
				indexed = 0
				continue
			data.append(row)
			if (idx+indexed)%size == 0:
				copy = deepcopy(data)
				data = list()
				yield copy

import os
import urllib.request
import zipfile

def load(itemfile=None, ratingfile=None, feature_ignore=True, size=None, dataset_link=None):
	# set default values if user don't pass values
	if itemfile is None:
		itemfile = myanimelist_anime
	if ratingfile is None:
		ratingfile = myanimelist_rating
	if dataset_link is None:
		dataset_link = link

	# maybe download...
	directory = '/'.join(itemfile.split('/')[:-1])
	if not os.path.exists(directory):
		download(directory, dataset_link)

	itens_generator = read_csv(itemfile, feature_ignore)
	ratings_generator = read_csv(ratingfile, feature_ignore, size)

	return itens_generator, ratings_generator


def download(dataset_name=None, dataset_link=None, log=True):
	# set default values if user don't pass values
	if dataset_name is None:
		dataset_name = name
	if dataset_link is None:
		dataset_link = link

	dataset_directory = dataset_name + "/dataset.zip"
	# create database folder
	if not os.path.exists(dataset_name):
		os.makedirs(dataset_name)

	if log is True:
		print("Downloading...",dataset_name)
    # download dataset on new folder
	urllib.request.urlretrieve(dataset_link, dataset_directory)

	if log is True:
		print("Extract all files...")
	# extract all zipfile on new folder
	zip_ref = zipfile.ZipFile(dataset_directory, 'r')
	zip_ref.extractall(dataset_name)
	zip_ref.close()
	os.remove(dataset_directory)

	if log is True:
		print("Extract all files...")
