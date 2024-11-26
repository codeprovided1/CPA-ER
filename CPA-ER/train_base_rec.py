import torch
import numpy as np
import os
import tqdm
import pickle
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from cnt_model import BaseRecModel
from scripts.evaluate_functions import compute_ndcg
from utils import *
from torch.utils.data import Dataset

from scripts.dataset_preprocessing import preprocessing

logger = None


class UserItemInterDataset(Dataset):
    def __init__(self, data, user_feature_matrix, item_feature_matrix):
        self.data = data
        self.user_feature_matrix = user_feature_matrix
        self.item_feature_matrix = item_feature_matrix

    def __getitem__(self, index):
        user = self.data[index][0]
        item = self.data[index][1]
        label = self.data[index][2]
        user_feature = self.user_feature_matrix[user]
        item_feature = self.item_feature_matrix[item]
        return user_feature, item_feature, label
    def __len__(self):
        return len(self.data)

def train_base_recommendation(train_args, pre_processing_args):
    if train_args.gpu:
        device = torch.device('cuda:%s' % train_args.cuda)
    else:
        device = 'cpu'

    rec_dataset = preprocessing(pre_processing_args)
    
    data_path = '{}/dataset_obj.pickle'.format(pre_processing_args.log_dir)
    with open(data_path, 'wb') as outp:
        pickle.dump(rec_dataset, outp, pickle.HIGHEST_PROTOCOL)
    logger.info("Save dataset_obj to.." + data_path)

    train_loader = DataLoader(dataset=UserItemInterDataset(rec_dataset.training_data, 
                                rec_dataset.user_feature_matrix, 
                                rec_dataset.item_feature_matrix),
                          batch_size=train_args.batch_size,
                          shuffle=True)

    model = BaseRecModel(rec_dataset.feature_num).to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay)

    ndcg = compute_ndcg(rec_dataset.test_data, 
            rec_dataset.user_feature_matrix, 
            rec_dataset.item_feature_matrix, 
            train_args.rec_k, 
            model, 
            device)
    logger.info('init ndcg={:.5f}'.format(ndcg))

    for epoch in tqdm.trange(train_args.epoch):
        model.train()
        optimizer.zero_grad()
        losses = []
        for user_behaviour_feature, item_aspect_feature, label in train_loader:
            user_behaviour_feature = user_behaviour_feature.to(device)
            item_aspect_feature = item_aspect_feature.to(device)
            label = label.float().to(device)
            out = model(user_behaviour_feature, item_aspect_feature).squeeze()
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.to('cpu').detach().numpy())
            ave_train = np.mean(np.array(losses))
        
        if epoch % 10 == 0:
            ndcg = compute_ndcg(rec_dataset.test_data, 
            rec_dataset.user_feature_matrix, 
            rec_dataset.item_feature_matrix, 
            train_args.rec_k, 
            model, 
            device)

            logger.info('epoch={:d}'.format(epoch) + ' | training loss={:.5f}'.format(ave_train) + ' | NDCG={:.5f}'.format(ndcg))

    out_path = '{}/base_model.model'.format(train_args.log_dir)
    logger.info("Save model to " + out_path)
    torch.save(model.state_dict(), out_path)

    return 0

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='cnt_base_model', help='directory name.')
    parser.add_argument("--gpu", dest="gpu", action="store_false", help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default='1e-5', help="L2 norm to the wights")
    parser.add_argument("--lr", dest="lr", type=float, default=0.0005, help="learning rate for training")
    parser.add_argument("--epoch", dest="epoch", type=int, default=100, help="training epoch")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=128, help="batch size for training base rec model")
    parser.add_argument("--rec_k", dest="rec_k", type=int, default=5, help="length of rec list")
    t_args =  parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, 
    help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='cnt_base_model', help='directory name.')
    parser.add_argument("--sample_ratio", dest="sample_ratio", type=int, default=2, 
    help="the (negative: positive sample) ratio for training BPR loss")
    parser.add_argument("--neg_length", dest="neg_length", type=int, default=100, 
    help="# of negative samples in evaluation")
    parser.add_argument("--save_path", dest="save_path", type=str, default="./dataset_objs/", 
    help="The path to save the preprocessed dataset object")
    parser.add_argument("--walk_length", dest="walk_length", type=int, default="10", 
    help="The random walk length of path generating method")#15
    parser.add_argument("--path_num", dest="path_num", type=int, default="10", 
    help="The random walk path numbers")
    parser.add_argument("--num_walks", dest="num_walks", type=int, default="9999", 
    help="The random walk times of path generating method")#500
    p_args =  parser.parse_args()

    if t_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = t_args.cuda
        print("Using CUDA", t_args.cuda)
    else:
        print("Using CPU")

    p_args.log_dir = '{}/{}'.format(TMP_DIR[p_args.dataset], p_args.name)
    if not os.path.isdir(p_args.log_dir):
        os.makedirs(p_args.log_dir)
    t_args.log_dir = '{}/{}'.format(TMP_DIR[t_args.dataset], t_args.name)
    if not os.path.isdir(t_args.log_dir):
        os.makedirs(t_args.log_dir)

    logger = get_logger(t_args.log_dir + '/train_log.txt')
    logger.info(t_args)
    logger.info(p_args)

    train_base_recommendation(t_args, p_args)