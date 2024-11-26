from re import S
import torch
import numpy as np
import json
import pickle
import tqdm
from torch.random import seed
from scripts.functions import generate_user_item_set, path_generate, get_attention_matrix, get_user_item_dict, sample_training_pairs
import sys 
from utils import *

class AmazonDataset():
    def __init__(self, preprocessing_args):
        super().__init__()
        
        self.args = preprocessing_args
        self.dataset = self.args.dataset
        self.feature_name_dict = {}
        self.features = []  # feature list
        self.users = []
        self.items = []

        self.user_hist_inter_dict = {}
        self.item_hist_inter_dict = {}  

        self.user_num = None
        self.item_num = None
        self.feature_num = None

        self.user_feature_matrix = None 
        self.item_feature_matrix = None 

        self.train_set = None
        self.test_set = None
        self.training_data = None
        self.test_data = None
        self.train_user_hist_inter_dict = {}
        self.test_user_hist_inter_dict = {}

        self.pre_processing()
        self.get_user_item_feature_matrix()

        self.sample_training() 
        self.sample_test() 

    def pre_processing(self,):
        kg = load_kg(self.dataset)

        features = get_all_relations()
        feature_name_dict = {}
        count = 0
        for feature in features:
            if feature not in feature_name_dict:
                feature_name_dict[feature] = count
                count += 1
        
        self.tarin_set = generate_user_item_set(self.dataset, 'train')
        self.test_set = generate_user_item_set(self.dataset, 'test')
        user_item_set = list(set(self.tarin_set + self.test_set)) # [(uid, pid), (uid, pid), ...]

         user_hist_inter_dict = {}  # {"u1": [i1, i2, i3, ...], "u2": [i1, i2, i3, ...]}
        item_hist_inter_dict = {}
        for i in user_item_set:
            user = i[0]
            item = i[1]
            if user not in user_hist_inter_dict:
                user_hist_inter_dict[user] = [item]
            else:
                user_hist_inter_dict[user].append(item)
            if item not in item_hist_inter_dict:
                item_hist_inter_dict[item] = [user]
            else:
                item_hist_inter_dict[item].append(user)

        user_hist_inter_dict = dict(sorted(user_hist_inter_dict.items()))
        item_hist_inter_dict = dict(sorted(item_hist_inter_dict.items()))
        
        users = list(user_hist_inter_dict.keys())
        items = list(item_hist_inter_dict.keys())

        self.train_user_hist_inter_dict = get_user_item_dict(self.tarin_set)
        self.test_user_hist_inter_dict = get_user_item_dict(self.test_set)

        self.kg = kg
        self.user_hist_inter_dict = user_hist_inter_dict
        self.item_hist_inter_dict = item_hist_inter_dict
        self.users = users
        self.items = items
        self.features = features
        self.user_num = len(users)
        self.item_num = len(items)
        self.feature_num = len(features)
        self.feature_name_dict = feature_name_dict
        return True

    def random_walk_paths(self,):
        path_user = path_generate(self.dataset, self.kg, "user", "product", self.user_hist_inter_dict, self.args.walk_length, self.args.num_walks, self.args.path_num)
        path_item = path_generate(self.dataset, self.kg, "product", "user", self.item_hist_inter_dict, self.args.walk_length, self.args.num_walks, self.args.path_num)
        print("generate paths end..")
        return True

    def get_user_item_feature_matrix(self,):
        path_user = load_paths(self.dataset, mode='user')
        path_item = load_paths(self.dataset, mode='product')
        
        self.user_feature_matrix = get_attention_matrix(
            path_user, 
            self.user_num, 
            self.features, 
            self.feature_name_dict, 
            max_range=5)
        self.item_feature_matrix = get_attention_matrix(
            path_item, 
            self.item_num, 
            self.features, 
            self.feature_name_dict, 
            max_range=5)
        return True
    
    def sample_training(self):
        print('======================= sample training data =======================')
        print(self.user_feature_matrix.shape, self.item_feature_matrix.shape)
        training_data = []
        item_set = set(self.items)
        for user, items in self.train_user_hist_inter_dict.items():
            training_pairs = sample_training_pairs(
                user, 
                items, 
                item_set, 
                self.args.sample_ratio)
            for pair in training_pairs:
                training_data.append(pair)
        print('# training samples :', len(training_data))
        self.training_data = np.array(training_data)
        return True
    
    def sample_test(self):
        print('======================= sample test data =======================')
        user_item_label_list = []  # [[u, [item1, item2, ...], [l1, l2, ...]], ...]
        for user, items in self.test_user_hist_inter_dict.items():
            user_item_label_list.append([user, items, np.ones(len(items))])  # add the test items
            negative_items = [item for item in self.items if 
                item not in self.test_user_hist_inter_dict[user]]  # the not interacted items
            negative_items = np.random.choice(np.array(negative_items), self.args.neg_length, replace=False)
            user_item_label_list[-1][1] = np.concatenate((user_item_label_list[-1][1], negative_items), axis=0)
            user_item_label_list[-1][2] = np.concatenate((user_item_label_list[-1][2], np.zeros(self.args.neg_length)), axis=0)
        print('# test samples :', len(user_item_label_list))
        self.test_data = np.array(user_item_label_list, dtype=object)
        return True

    def save(self, save_path):
        return True
    
    def load(self):
        return False


def preprocessing(pre_processing_args):
    rec_dataset = AmazonDataset(pre_processing_args)
    return rec_dataset
