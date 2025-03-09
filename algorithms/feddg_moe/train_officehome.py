import os
import argparse
from network.get_network import GetNetwork
from torch.utils.tensorboard.writer import SummaryWriter
from data.officehome_dataset import OfficeHome_FedDG
from utils.classification_metric import Classification 
from utils.log_utils import *
from utils.fed_merge import Cal_Weight_Dict, FedAvg, FedUpdate
from utils.trainval_func import site_evaluation, feddgmoe_site_train, feddgmoe_testsite_eval_sample, feddgmoe_testsite_eval_batch, feddgmoe_testsite_eval_batch_v2 ,GetFedModel, SaveCheckPoint
import torch.nn.functional as F
from tqdm import tqdm
import torch

from utils.domain_stats.online.online_cosine import OnlineCosineTracker
from utils.domain_stats.online.online_gmm import OnlineGMMTracker
from utils.domain_stats.online.online_gaussian import OnlineGaussianTracker

from utils.domain_stats.offline.offline_gmm import OfflineGMMTracker
from utils.domain_stats.offline.offline_mahalanobis import OfflineMahalanobisTracker
from utils.domain_stats.offline.offline_cosine import OfflineCosineTracker, OfflineCosineMuVarTracker

def collect_online_stats(model, dataloader, domain_stats, domain_id):
    """
    Collect online statistics for a domain using the provided dataloader.
    Note: online collection saves memory (no need to store all features at once).
    """
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            features = model(inputs.cuda())
            domain_stats.update(features, domain_id)

def collect_offline_stats(model, dataloader, domain_stats, domain_id):
    """
    Collect offline statistics for a domain. Refits from scratch.
    """
    all_features = []
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            features = model(inputs.cuda())
            all_features.append(features)
    all_features = torch.cat(all_features, dim=0)
    domain_stats.refit(all_features, domain_id)  # Call the refit method

def collect_offline_stats_mul_layers(model, dataloader, domain_stats, domain_id):
    """
    Collect offline statistics for a domain. Refits from scratch.
    """
    all_features = []
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            intermediate_features = model.get_intermediate_layers(inputs.cuda(), n=4,
            return_prefix_tokens=True, norm=True)        
            averaged_layers = []
            for spatial_tokens, prefix_tokens in intermediate_features:
                all_tokens = torch.cat([prefix_tokens, spatial_tokens], dim=1)
                #averaged_layers.append(all_tokens.mean(dim=1))
                averaged_layers.append(all_tokens)       
            intermediate_features = torch.stack(averaged_layers).mean(dim=0) #[batch_size, embed_dim]
            all_features.append(intermediate_features)

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
    parser.add_argument('--batch_size', help='batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', help='test_batch_size', type=int, default=16)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=5)
    parser.add_argument('--comm', help='epochs number', type=int, default=40)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--domain_tracker', help='Which domain stats tracker to use', type=str, default='gmm')
    parser.add_argument('--flatten_tokens', help='Flatten token features instead of averaging', action='store_true')
    parser.add_argument('--note', help='note of experimental settings', type=str, default='feddg_moe')
    parser.add_argument('--display', help='display in controller', action='store_true')
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
        domain_stats = OfflineCosineMuVarTracker(num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Offline Cosine MuVar Tracker')
    elif args.domain_tracker == 'online_cosine':
        domain_stats = OnlineCosineTracker(num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Online Cosine Tracker')
    elif args.domain_tracker == 'online_gmm':
        domain_stats = OnlineGMMTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Online GMM Tracker')
    elif args.domain_tracker == 'offline_gmm':
        domain_stats = OfflineGMMTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Offline GMM Tracker')
    elif args.domain_tracker == 'offline_mahalanobis':
        domain_stats = OfflineMahalanobisTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Offline Mahalanobis Tracker')
    elif args.domain_tracker == 'offline_cosine':
        domain_stats = OfflineCosineTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        log_file.info('Using Offline Cosine Tracker')
    else:
        raise ValueError(f"Unknown domain tracker: {args.domain_tracker}")


    if 'offline_' in args.domain_tracker:
        #collect_stats = collect_offline_stats
        collect_stats = collect_offline_stats_mul_layers
    elif 'online_' in args.domain_tracker: 
        collect_stats = collect_online_stats

    FedUpdate(model_dict, global_model)
    best_val = 0.
    for i in range(args.comm+1):
        #FedUpdate(model_dict, global_model)
        for domain_id, domain_name in enumerate(dataobj.train_domain_list):
            # Train Domain[i]
            feddgmoe_site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                      scheduler_dict[domain_name], dataloader_dict[domain_name]['train'], log_ten, metric)
            
            # Update Domain [i] Statistics
            collect_stats(model_dict[domain_name][0], dataloader_dict[domain_name]['train'], domain_stats, domain_id)

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
        feddgmoe_testsite_eval_batch_v2(i, args.test_domain, args, global_model, dataloader_dict[args.test_domain]['test'], log_file, log_ten, metric, note='test_domain', model_dict=model_dict, domain_stats= domain_stats)
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    for domain_name in dataobj.train_domain_list: 
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'last_{domain_name}_model')
    
if __name__ == '__main__':
    main()