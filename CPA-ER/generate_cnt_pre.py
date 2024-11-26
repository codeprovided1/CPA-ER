import torch
import pickle
import os
from pathlib import Path
from cnt_model import BaseRecModel, ExpOptimizationModel


def generate_explanation(exp_args):
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    
    # import dataset
    data_path = '{}/dataset_obj.pickle'.format(exp_args.log_dir)
    with open(data_path, 'rb') as inp:
        rec_dataset = pickle.load(inp)
    # import base model
    base_model = BaseRecModel(rec_dataset.feature_num).to(device)
    model_path = '{}/base_model.model'.format(exp_args.log_dir)
    base_model.load_state_dict(torch.load(model_path))
    base_model.eval()
    #  fix the rec model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Create optimization model
    opt_model = ExpOptimizationModel(
        base_model=base_model,
        rec_dataset=rec_dataset,
        device = device,
        exp_args=exp_args,
    )

    opt_model.generate_user_rel_pre(self, user, item)
    # save model
    Path(exp_args.save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(exp_args.save_path, exp_args.dataset + "_explanation_obj.pickle"), 'wb') as outp:
        pickle.dump(opt_model, outp, pickle.HIGHEST_PROTOCOL)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='cnt_base_model', help='directory name.')
    parser.add_argument("--base_model_path", dest="base_model_path", type=str, default="./logs/")
    parser.add_argument("--gpu", dest="gpu", action="store_false", help="whether to use gpu")
    parser.add_argument("--cuda", dest="cuda", type=str, default='0', help="which cuda")
    parser.add_argument("--data_obj_path", dest="data_obj_path", type=str, 
        default="./dataset_objs/", help="the path to the saved dataset object in the training phase")
    parser.add_argument("--rec_k", dest="rec_k", type=int, default=5, help="length of rec list")
    parser.add_argument("--lam", dest="lam", type=float, default=200, help="the hyper-param for pairwise loss")
    parser.add_argument("--gam", dest="gam", type=float, default=1, help="the hyper-param for L1 reg")
    parser.add_argument("--alp", dest="alp", type=float, default=0.05, help="margin value for pairwise loss")
    parser.add_argument("--user_mask", dest="user_mask", action="store_false", help="whether to use the user mask.")
    parser.add_argument("--lr", dest="lr", type=float, default=0.01, help="learning rate in optimization")
    parser.add_argument("--step", dest="step", type=int, default=1000, help="# of steps in optimization")
    parser.add_argument("--mask_thresh", dest="mask_thresh", type=float, default=0.1, help="threshold for choosing explanations")
    parser.add_argument("--test_num", dest="test_num", type=int, default=-1, help="the # of users to generate explanation")
    parser.add_argument("--save_path", dest="save_path", type=str, default="./explanation_objs/", help="save the conterfactual explanation results")
    exp_args = parser.parse_args()

    exp_args.log_dir = '{}/{}'.format(TMP_DIR[exp_args.dataset], exp_args.name)
    if not os.path.isdir(exp_args.log_dir):
        os.makedirs(exp_args.log_dir)

    generate_explanation(exp_args)