import numpy as np
import pandas as pd
import gzip
import math
import matplotlib.pyplot as plt
from utils import *

np.random.seed(0)

def generate_user_item_set(dataset, mode='train'):
    review_file = '{}/{}.txt.gz'.format(DATASET_DIR[dataset], mode)
    #user_products = {}  # {uid: [pid,...], ...}
    user_products = []  # [(uid, pid), (uid, pid), ...]
    with gzip.open(review_file, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            u_p = (user_idx, product_idx)
            user_products.append(u_p)
    return user_products

def path_generate(dataset, kg, start_type, end_type, hist_inter_dict, walk_length, num_walks, path_num):
    paths = {} 
    print("start generate paths...", start_type)
    for start in hist_inter_dict:
        for end in hist_inter_dict[start]:
            paths[(start, end)]  = []
            times = 0
            while(len(paths[(start, end)]) < path_num):
                if times > num_walks:
                    break
                times = times + 1
                path = [start]
                current_node = start
                current_type = start_type

                for _ in range(walk_length):
                    neighbors_dict = kg(current_type, current_node)
                    if not neighbors_dict:
                        break
                    rel_list = [key for key in neighbors_dict if len(neighbors_dict[key])!= 0]
                    if len(rel_list) == 0:
                        break
                    rel = random.choice(rel_list)
                    neighbors = neighbors_dict[rel]
                    next_node = random.choice(neighbors)
                    next_type = KG_RELATION[current_type][rel]
                    current_node = next_node
                    current_type = next_type
                    path.append((rel, next_node))

                    # If we reached the end_node, stop the walk
                    if current_node == end and current_type == end_type:
                        if path not in paths[(start, end)]:
                            paths[(start, end)].append(path)
                        break
                # If the walk is interrupted and the end_node is reached
                if path[-1] == end :
                    paths.append(path)
    print("save paths in file...", start_type)
    save_paths(dataset, paths, mode=start_type)
    return paths

def get_attention_matrix(paths, user_num, feature_list, feature_name_dict, max_range=5):
    user_counting_matrix = np.zeros((user_num, len(feature_list)))  
    for key in paths: 
        user = key[0]
        for path in paths[key]: 
            for node in path[1:]:
                feature = feature_name_dict[node[0]]
                user_counting_matrix[user, feature] += 1
    user_attention_matrix = np.zeros((user_num, len(feature_list)))  
    for i in range(len(user_counting_matrix)):
        for j in range(len(user_counting_matrix[i])):
            if user_counting_matrix[i, j] == 0:
                norm_v = 0  
            else:
                norm_v = 1 + (max_range - 1) * ((2 / (1 + np.exp(-user_counting_matrix[i, j]))) - 1)  
            user_attention_matrix[i, j] = norm_v
    user_attention_matrix = np.array(user_attention_matrix, dtype='float32')
    return user_attention_matrix

def get_user_item_dict(dataset):
    """
    build user & item dictionary
    :param dataset: [(user, item), (user, item) ...]
    :return: user dictionary {u1:[i, i, i...], u2:[i, i, i...]}
    """
    user_dict = {}
    for row in dataset:
        user = row[0]
        item = row[1]
        if user not in user_dict:
            user_dict[user] = [item]
        else:
            user_dict[user].append(item)
    return user_dict

def sample_training_pairs(user, training_items, item_set, sample_ratio=10):
    positive_items = set(training_items)
    negative_items = set()
    for item in item_set:
        if item not in positive_items:
            negative_items.add(item)
    neg_length = len(positive_items) * sample_ratio
    negative_items = np.random.choice(np.array(list(negative_items)), neg_length, replace=False)
    train_pairs = []
    for p_item in positive_items:
        train_pairs.append([user, p_item, 1])
    for n_item in negative_items:
        train_pairs.append([user, n_item, 0])
    return train_pairs

def negative_sampling(df_dataset, item_count):
    # negative sampling
    full_item_set = set(range(item_count))
    user_list = []
    item_list = []
    label_list = []
    for user, group in df_dataset.groupby(['userID']):
        item_set = set(group['itemID'])
        negative_set = full_item_set - item_set
        negative_sampled = random.sample(negative_set, len(item_set))
        user_list.extend([user] * len(negative_sampled))
        item_list.extend(negative_sampled)
        label_list.extend([0] * len(negative_sampled))
    negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})
    df_dataset = pd.concat([df_dataset, negative])

    df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)
    df_dataset.reset_index(inplace=True, drop=True)
    return df_dataset