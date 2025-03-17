import os
import argparse
from network.get_network import GetNetwork, feats_extractor
from torch.utils.tensorboard.writer import SummaryWriter
from data.officehome_dataset import OfficeHome_FedDG
from utils.classification_metric import Classification 
from utils.log_utils import *
from utils.fed_merge import Cal_Weight_Dict, FedAvg, FedUpdate
from utils.trainval_func import site_evaluation, feddgmoe_site_train, feddgmoe_testsite_eval, GetFedModel, SaveCheckPoint
import torch.nn.functional as F
from tqdm import tqdm
import torch

from utils.domain_stats.online.online_cosine import OnlineCosineTracker
from utils.domain_stats.online.online_gmm import OnlineGMMTracker
from utils.domain_stats.online.online_gaussian import OnlineGaussianTracker

from utils.domain_stats.offline.offline_gmm import OfflineGMMTracker
from utils.domain_stats.offline.offline_mahalanobis import OfflineMahalanobisTracker
from utils.domain_stats.offline.offline_cosine import OfflineCosineTracker, OfflineCosineMuVarTracker, OfflineCosineConcat

def collect_online_stats(model, dataloader, domain_stats, domain_id, args):
    """
    Collect online statistics for a domain using the provided dataloader.
    Note: online collection saves memory (no need to store all features at once).
    """
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            features = feats_extractor(inputs.cuda(), model, avg_tokens=args.avg_tokens, num_layers=args.num_layers)
            domain_stats.update(features, domain_id)

def collect_offline_stats(model, dataloader, domain_stats, domain_id, args):
    """
    Collect offline statistics for a domain. Refits from scratch.
    """
    all_features = []
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            features = feats_extractor(inputs.cuda(), model, avg_tokens=args.avg_tokens, num_layers=args.num_layers)
            all_features.append(features)
    all_features = torch.cat(all_features, dim=0)
    domain_stats.refit(all_features, domain_id)  # Call the refit method

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='officehome', choices=['officehome'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='clip_moe',
                        choices=['resnet18', 'resnet50', 'clip', 'clip_moe', 'clip_moe_layers'], help='model name')
    parser.add_argument("--test_domain", type=str, default='p',
                        choices=['p', 'a', 'c', 'r'], help='the domain name for testing')
    parser.add_argument('--num_classes', help='number of classes default 65', type=int, default=65)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', help='test_batch_size', type=int, default=32)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=10)
    parser.add_argument('--comm', help='epochs number', type=int, default=5)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    ## tracker args
    parser.add_argument('--domain_tracker', help='Which domain stats tracker to use', type=str, default='offline_cosine_muvar',
                       choices=['offline_cosine_muvar', 'offline_cosine', 'offline_mahalanobis', 
                               'offline_gmm', 'offline_cosine_concat',
                               'online_cosine', 'online_gaussian', 'online_gmm'])
    parser.add_argument('--avg_tokens', help='Average token features instead of leaving them flat (i.e token_dim * n_tokens)', action='store_true')
    parser.add_argument('--num_layers', help='Number of layers to extract features from', type=int, default=4)
    parser.add_argument('--mul_layers', help='does nothing just for backward comp', action='store_true')
    ### GMM args
    parser.add_argument('--gmm_num_components', help='Number of components for GMM tracker', type=int, default=3)
    parser.add_argument('--gmm_covariance_type', help='Covariance type for GMM', type=str, default='diag', 
                        choices=['full', 'tied', 'diag', 'spherical'])
    ### test time eval args
    parser.add_argument('--batch_agg_type', type=str, default='pre_similarity', 
                      choices=['pre_similarity', 'post_similarity'], 
                      help='How to aggregate batch features: pre_similarity averages features before computing similarities, post_similarity computes similarities per sample then averages')
    parser.add_argument('--log_similarities', help='Log similarity weights during evaluation', action='store_true')
    parser.add_argument('--inv_temp', help='1 / T for offline cos mu var', type=float, default=2.0)
    parser.add_argument('--use_prior', help='Use weight_dict as prior distribution for similarities', action='store_true')
    parser.add_argument('--prior_strength', help='Strength of the prior distribution (higher values give more weight to prior)', type=float, default=1.0)

    # misc args
    return parser.parse_args()

def main():
    '''log part'''
    file_name = 'fedavg_'+os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)
    
    '''dataset and dataloader'''
    dataobj = OfficeHome_FedDG(test_domain=args.test_domain, batch_size=args.batch_size, test_batch_size=args.test_batch_size)
    dataloader_dict, dataset_dict = dataobj.GetData()
    #manually add a new named arguments to args
    args.site_list = dataobj.train_domain_list
        
    '''model'''
    metric = Classification()
    global_model, model_dict, optimizer_dict, scheduler_dict = GetFedModel(args, args.num_classes)
    weight_dict = Cal_Weight_Dict(dataset_dict, site_list=dataobj.train_domain_list)

    ''' Domain Statistics Tracking'''
    # todo get feature level from the GetFedModel
    if args.domain_tracker == 'online_gaussian':
        domain_stats = OnlineGaussianTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Online Gaussian Tracker')
    elif args.domain_tracker == 'offline_cosine_muvar':
        domain_stats = OfflineCosineMuVarTracker(args)
        log_file.info('Using Offline Cosine MuVar Tracker')
    elif args.domain_tracker == 'offline_cosine_concat':
        domain_stats = OfflineCosineConcat(args)
        log_file.info('Using Offline Cosine Concat Tracker')
    elif args.domain_tracker == 'online_cosine':
        domain_stats = OnlineCosineTracker(num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Online Cosine Tracker')
    elif args.domain_tracker == 'online_gmm':
        domain_stats = OnlineGMMTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Online GMM Tracker')
    elif args.domain_tracker == 'offline_gmm':
        domain_stats = OfflineGMMTracker(args)
        log_file.info(f'Using Offline GMM Tracker with {args.gmm_num_components} components and {args.gmm_covariance_type} covariance')
    elif args.domain_tracker == 'offline_mahalanobis':
        domain_stats = OfflineMahalanobisTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Offline Mahalanobis Tracker')
    elif args.domain_tracker == 'offline_cosine':
        domain_stats = OfflineCosineTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Offline Cosine Tracker')
    else:
        raise ValueError(f"Unknown domain tracker: {args.domain_tracker}")


    # Simplified stats collection function selection
    collect_stats = collect_offline_stats if 'offline_' in args.domain_tracker else collect_online_stats


    FedUpdate(model_dict, global_model)
    best_val = 0.
    for i in range(args.comm+1):
        FedUpdate(model_dict, global_model)
        for domain_id, domain_name in enumerate(dataobj.train_domain_list):
            # Train Domain[i]
            feddgmoe_site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                      scheduler_dict[domain_name], dataloader_dict[domain_name]['train'], log_ten, metric)
            
            # Update Domain [i] Statistics
            collect_stats(model_dict[domain_name][0], dataloader_dict[domain_name]['train'], domain_stats, domain_id, args)

            # Val Domain[i]
            site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric, note='before_fed')

        FedAvg(model_dict, weight_dict, global_model)
        
        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            results_dict = site_evaluation(i, domain_name, args, global_model, dataloader_dict[domain_name]['val'], log_file, log_ten, metric)
            fed_val+= results_dict['acc']*weight_dict[domain_name]
        # val
        if fed_val >= best_val:
            best_val = fed_val
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            for domain_name in dataobj.train_domain_list: 
                SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'best_val_{domain_name}_model')
                
            log_file.info(f'Model saved! Best Val Acc: {best_val*100:.2f}%')
    
        feddgmoe_testsite_eval(i, args.test_domain, args, global_model, dataloader_dict[args.test_domain]['test'], 
                         log_file, log_ten, metric, note='test_domain', model_dict=model_dict, domain_stats=domain_stats,
                         weight_dict=weight_dict)
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    for domain_name in dataobj.train_domain_list: 
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'last_{domain_name}_model')
    
if __name__ == '__main__':
    main()