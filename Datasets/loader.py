import csv
import numpy as np
import sys

# My anime list directories (1 indexed)
myanimelist_anime = '../Datasets/MyAnimeList/anime.csv'
myanimelist_rating = '../Datasets/MyAnimeList/rating.csv'
len_user = 73517
len_anime = 34528
name = "MyAnimeList"
link = "https://storage.googleapis.com/kaggle-datasets/571/1094/anime-recommendations-database.zip?GoogleAccessId=datasets@kaggle-161607.iam.gserviceaccount.com&Expires=1497664649&Signature=E9VT09SgXL3CV%2Fhy2HZR%2BlaepIY6ZIEuTJL27rgUDimbMWjiK7mkMnyw%2FJEhe%2B88v%2FeEueiEUpf6RABmJURlMfUl07gpm4uBqrr6N4lPjNsvDwEvXWWIWXVN%2FP9Gg2f3WmrzgEkDBNva5MF%2BEMDYVzj62Hcwhw5VE8Q3lOz8uNOmXnBHvEnyCh7HVnXuIa2Q6ZBvWnug%2FaUWcAwr1p6uovzcmdT8NhOjPd%2FdoVp3pZbzsnXHMtI1ltawApnAID6bJtnJM5VdRhOBqn%2BpDJw8sRuhAnMNqHszuVNDJrEbuZNWm8ZXCLOtyqWi0DNsusnAPPtvhMANsg9IDT%2BeHjJmhw%3D%3D"

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


def load(itemfile=None, ratingfile=None, tam="large", feature_names=False):
	# set default values if user don't pass values
	if itemfile is None:
		itemfile = myanimelist_anime
	if ratingfile is None:
		ratingfile = myanimelist_rating

	name2id, id2name = get_structs(itemfile)
	rating = read_csv(ratingfile, feature_names)
	return name2id, id2name, rating

import os
import urllib.request
import zipfile

def download(dataset_name=None, dataset_link=None):
	# set default values if user don't pass values
	if dataset_name is None:
		dataset_name = name
	if dataset_link is None:
		dataset_link = link

	dataset_directory = dataset_name + "/dataset.zip"
	# create database folder
	if not os.path.exists(dataset_name):
		os.makedirs(dataset_name)

    # download dataset on new folder
	urllib.request.urlretrieve(dataset_link, dataset_directory)

	# extract all zipfile on new folder
	zip_ref = zipfile.ZipFile(dataset_directory, 'r')
	zip_ref.extractall(dataset_name)
	zip_ref.close()
	os.remove(dataset_directory)
