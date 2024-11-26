from numpy import core
import torch
import tqdm
import numpy as np
import torch.nn.functional as F

class BaseRecModel(torch.nn.Module):
    def __init__(self, feature_length):
        super(BaseRecModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(feature_length * 2, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, user_feature, item_feature):
        fusion = torch.cat((user_feature, item_feature), 1)
        out = self.fc(fusion)
        return out

class ExpOptimizationModel(torch.nn.Module):
    def __init__(self, base_model, rec_dataset, device, cnt_args):
        super(ExpOptimizationModel, self).__init__()
        self.base_model = base_model
        self.rec_dataset = rec_dataset
        self.device = device
        self.cnt_args = cnt_args
        
        self.user_feature_matrix = torch.from_numpy(self.rec_dataset.user_feature_matrix).to(self.device)
        self.item_feature_matrix = torch.from_numpy(self.rec_dataset.item_feature_matrix).to(self.device)

    def generate_user_rel_pre(self, user, item):
        init_score = self.base_model(self.user_feature_matrix[user].unsqueeze(0), 
                            self.item_feature_matrix[item].unsqueeze(0)).squeeze()
        mask_vec = torch.ones(self.rec_dataset.feature_num, device=self.device).unsqueeze(0)
        optimize_delta = self.explain(
                    self.user_feature_matrix[user], 
                    self.item_feature_matrix[item], 
                    init_score,
                    mask_vec)
        
        user_rel_pre = F.softmax(optimize_delta, dim=0)

        user_rel_pre = user_rel_pre.tolist()
        user_rel_pre = [round(item, 10) for item in user_rel_pre]
        user_rel_weight = self.user_feature_matrix[user].tolist()
        user_rel_weight = [round(item, 10) for item in user_rel_weight]
        return user_rel_pre, user_rel_weight

    def explain(self, user_feature, item_feature, init_score, mask_vec):
        exp_generator = EXPGenerator(
            self.rec_dataset, 
            self.base_model, 
            user_feature, 
            item_feature, 
            init_score, 
            mask_vec,
            self.device, 
            self.cnt_args).to(self.device)

        # optimization
        optimizer = torch.optim.SGD(exp_generator.parameters(), lr=self.cnt_args.lr, weight_decay=0)
        exp_generator.train()
        lowest_loss = None
        lowest_bpr = None
        lowest_l2 = 0
        optimize_delta = None

        score = exp_generator() 
        bpr, l2, l1, loss = exp_generator.loss(score) 
        lowest_loss = loss
        lowest_l2 = l2
        
        optimize_delta = exp_generator.delta.detach().to('cpu')

        for epoch in range(self.cnt_args.step):
            exp_generator.zero_grad()
            score = exp_generator()
            bpr, l2, l1, loss = exp_generator.loss(score)
            loss.backward()
            optimizer.step()
            if loss < lowest_loss: 
                lowest_loss = loss
                lowest_l2 = l2
                lowest_bpr = bpr
                optimize_delta = exp_generator.delta.detach().to('cpu')
       
        return optimize_delta

class EXPGenerator(torch.nn.Module):
    def __init__(self, rec_dataset, base_model, user_feature, item_feature, init_score, mask_vec, device, cnt_args):
        super(EXPGenerator, self).__init__()
        self.rec_dataset = rec_dataset
        self.base_model = base_model
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.init_score = init_score
        self.mask_vec = mask_vec
        self.device = device
        self.cnt_args = cnt_args

        self.feature_range = [0, 5]  
        self.delta_range = self.feature_range[1] - self.feature_range[0] 
        self.delta = torch.nn.Parameter(
            torch.FloatTensor(len(self.user_feature)).uniform_(-self.delta_range, 0))

    def get_masked_item_aspect(self):
        item_feature_star = torch.clamp(
            (self.item_feature + torch.clamp((self.delta * self.mask_vec), -self.delta_range, 0)),
            self.feature_range[0], self.feature_range[1])
        return item_feature_star
    
    def forward(self):
        item_feature_star = self.get_masked_item_aspect()
        score = self.base_model(self.user_feature.unsqueeze(0), item_feature_star)
        return score
    
    def loss(self, score):
        bpr = torch.nn.functional.relu(self.cnt_args.alp + score - self.init_score) * self.cnt_args.lam
        l2 = torch.linalg.norm(self.delta)
        l1 = torch.linalg.norm(self.delta, ord=1) * self.cnt_args.gam
        loss = l2 + bpr + l1
        return bpr, l2, l1, loss