import fastai
import fastai.basics as fai
import fastai.collab as fc
import fastai.tabular as ft

import numpy as np
from fuzzywuzzy import fuzz
from pathlib import Path
from shutil import copyfile
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

class SimilarKeyFinder():
    def __init__(self, keyList: list):
        self.keyList = keyList

    def __call__(self, key, thresh = 60):
        similarity = list(map(lambda x: fuzz.ratio(x, key) if fuzz.ratio(x, key) > thresh else 0, self.keyList))
        best = np.max(similarity)
        if best != 0:
            return self.keyList[similarity.index(best)]
        else:
            return None

class ParameterCalculator():
    def __init__(self, learner:fc.collab_learner ,anime2id:dict, num_factors:int):
        self.anime2id = anime2id
        self.num_factors = num_factors
        self.learner = learner

    def __call__(self, entries:dict):
        self.entries = entries
        self.items = self.get_anime_ids()
        self.ratings = torch.FloatTensor(list(entries.values())).cuda()
        return self.optimizeParameters()

    def get_anime_ids(self):
        animes_watched = list(self.entries.keys())
        anime_ids = [self.anime2id[anime] for anime in animes_watched]
        return torch.tensor(anime_ids).cuda()

    def optimizeParameters(self, steps:int=100):
        new_user_weights, new_user_bias = self.initialize_new_user()
        optimizer = optim.SGD([new_user_weights,new_user_bias], lr=0.1)
        for it in tqdm(range(steps)):
            optimizer.zero_grad()
            result = self.calculate_result(new_user_weights, new_user_bias)
            loss = F.mse_loss(result,self.ratings)
            loss.backward()
            optimizer.step()
        self.error = loss.item()
        return new_user_weights, new_user_bias

    def calculate_result(self, new_user_weights:torch.tensor, new_user_bias:torch.tensor):
        weight = (self.learner.model.i_weight(self.items)*new_user_weights).sum(dim=1)
        bias = self.learner.model.i_bias(self.items).squeeze() + new_user_bias
        result = weight + bias
        normalized_result = torch.sigmoid(result)*11 - 0.5
        return normalized_result.cuda()

    def initialize_new_user(self):
        new_user_weights = torch.rand((self.num_factors,)).cuda().requires_grad_(True)
        new_user_bias = torch.zeros((1,)).cuda().requires_grad_(True)
        return new_user_weights, new_user_bias

    def getColumns(self, learner:fc.collab_learner, items: list):
        return learner.model.i_weight(torch.tensor(items).cuda())

class Recommendor():
    def __init__(self, learner:fc.collab_learner, data:fc.TabularDataBunch, anime2id:dict, entries:dict, num_factor:int=50):
        self.data = data
        self.entries = entries
        self.learner = learner
        self.anime2id = anime2id
        calculate_parameters = ParameterCalculator(learner, anime2id, num_factor)
        self.user_weights, self.user_bias = calculate_parameters(entries)
    
    def __call__(self, num:int=10, reverse:bool=False):
        animes = list(anime2id.keys())
        animes_watched = list(self.entries.keys())
        animes_rate_prediction = [(anime, self.predicted_score(anime2id[anime])) for anime in animes if anime not in animes_watched]
        animes_rate_prediction.sort(key = lambda x: x[1], reverse=True)
        if reverse:
            return animes_rate_prediction[num:]
        return animes_rate_prediction[:num]

    def predicted_score(self, anime_id:int):
        weight = (self.learner.model.i_weight(torch.tensor([anime_id]).cuda())*self.user_weights).sum(dim=1)
        bias = self.learner.model.i_bias(torch.tensor([anime_id]).cuda()).squeeze() + self.user_bias
        result = weight + bias
        normalized_result = torch.sigmoid(result)*11 - 0.5
        return normalized_result.item()

def get_input(anime2id:dict):
    closest_name = SimilarKeyFinder(list(anime2id.keys()))
    entries = {}
    print("To leave enter empty line")
    while True:
        regis = input("Input anime and rating in the following format. \nanime::rating\n")
        if regis == "":
            break
        try:
            anime_name, rating = regis.split("::")
            rating = int(rating.strip())
        except ValueError:
            print("Use '::' to seperate anime name and rating from 0 to 10")
            continue
        correct_anime = closest_name(anime_name)
        if correct_anime == None:
            print("Could not find that anime, try again")
            continue
        if anime_name != correct_anime:
            use_name = input(f"Did you mean '{correct_anime}'? [y/n]")
            while use_name.lower() not in ["y", "n", "yes", "no"]:
                print("Please input y or n")
                use_name = input(f"Did you mean '{correct_anime}'? [y/n]")
            if use_name.lower() in ["n","no"]:
                continue
        entries[correct_anime] = rating
    return entries

path = Path("data")
data = torch.load(path/"data.feather")
y_range = [-0.5,10.5]
learner = fc.collab_learner(data, n_factors = 50, use_nn = False, y_range=y_range)
learner.load('anime')

anime2id = torch.load(path/"anime2id.feather")

entries = get_input(anime2id)

recommend_anime = Recommendor(learner, data, anime2id, entries)
print(recommend_anime())