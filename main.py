import d4rl
from ast import parse
import numpy as np
import torch
import os
import time
import pathlib
from prompt_dt.fl_utils import * 
import argparse
import pickle
import random
import sys
from tqdm import trange

import itertools
import copy

from prompt_dt.prompt_decision_transformer import PromptDecisionTransformer
from prompt_dt.prompt_seq_trainer import PromptSequenceTrainer
from prompt_dt.prompt_utils import get_env_list, report_parameters
from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch
from prompt_dt.prompt_utils import process_info, load_meta_data_prompt
from prompt_dt.prompt_utils import eval_episodes
from prompt_dt.fl_utils import ERK_maskinit

from collections import namedtuple
import json, pickle, os
from logger import setup_logger, logger

@torch.no_grad()
def cosine_annealing(alpha_t,eta_max=30,eta_min=0):    
    return int(eta_max+0.5*(eta_min-eta_max)*(1+np.cos(2*np.pi*alpha_t)))

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def experiment_mix_env(
        variant,
):
    device = variant['device']
    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')
    seed = variant['seed']
    set_seed(variant['seed'])
    env_name_ = variant['env']
    
    ######
    # construct train and test environments
    ######
    eta_min=variant['eta_min']
    eta_max=variant['eta_max']
    data_save_path = variant['data_path']
    save_path = variant['save_path']
    timestr = time.strftime("%y%m%d-%H%M%S")

    task_config = os.path.join(variant['config_path'], "MetaWorld/task.json")
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    
    if '50' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt50_task_list, task_config.mt50_task_list
    elif '30' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt30_task_list, task_config.mt30_task_list
    elif '5' in variant['prefix_name']:
        train_env_name_list, test_env_name_list = task_config.mt5_task_list, task_config.mt5_task_list
    else:
        raise NotImplementedError("please use the 50, 30, 5")

    # training envs
    info, env_list = get_env_list(train_env_name_list, device, total_env='metaworld', seed=seed)
    # testing envs
    test_info, test_env_list = get_env_list(test_env_name_list, device, total_env='metaworld_test', seed=seed)

    num_env = len(train_env_name_list)
    group_name = variant['prefix_name']
    exp_prefix = f'{env_name_}-{seed}-{timestr}'
    if variant['no_prompt']:
        exp_prefix += '_NO_PROMPT'
    if variant['no_r']:
        exp_prefix += '_NO_R'
    save_path = os.path.join(save_path, group_name+'-'+exp_prefix)
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    setup_logger(exp_prefix, variant=variant, log_dir=save_path)

    logger.log(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    
    ######
    # process train and test datasets
    ######
    # load training dataset 
    optimal = not variant['suboptimal']
    trajectories_list, prompt_trajectories_list = load_meta_data_prompt(
        train_env_name_list, 
        data_save_path, 
        optimal
    )
    test_trajectories_list, test_prompt_trajectories_list = load_meta_data_prompt(
        test_env_name_list, 
        data_save_path, 
        optimal
    )

    # process train info
    prompt_info = copy.deepcopy(info)
    prompt_info = process_info(train_env_name_list, prompt_trajectories_list, prompt_info, mode, pct_traj, variant, logger)
    info = process_info(train_env_name_list, trajectories_list, info, mode, pct_traj, variant, logger)
    # process test info
    prompt_test_info = copy.deepcopy(test_info)
    prompt_test_info = process_info(test_env_name_list, test_prompt_trajectories_list, prompt_test_info, mode, pct_traj, variant, logger)
    test_info = process_info(test_env_name_list, test_trajectories_list, test_info, mode, pct_traj, variant, logger)

    ######
    # construct dt model and trainer
    ######
    model = PromptDecisionTransformer(
        env_name_list=train_env_name_list,
        info=info,
        special_embedding=variant['special_embedding'],
        max_length=K,
        max_ep_len=1000,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    model = model.to(device=device)
    report_parameters(model, logger)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    env_name = train_env_name_list[0]
    trainer = PromptSequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch(trajectories_list[0], info[env_name], variant),
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=None,
        get_prompt=get_prompt(prompt_trajectories_list[0], prompt_info[env_name], variant),
        get_prompt_batch=get_prompt_batch(trajectories_list, prompt_trajectories_list, info, prompt_info, variant, train_env_name_list),
        logger=logger,
        variant=variant
    )

    ######
    # start training
    ######
    
    best_suc = -10000
    best_iter = 0
    
    env_masks = {}
    mode="random"
    if mode =="random_same":
        mask=ERK_maskinit(model, 1 - variant['sparsity'])
        for env_name in train_env_name_list:
            env_masks[env_name] = mask
    elif mode =="random":
        for env_name in train_env_name_list:
            mask=ERK_maskinit(model, 1 - variant['sparsity'])
            env_masks[env_name] = mask
    harmo_gradient=None

    for iter in trange(variant['max_iters']):
        env_id = iter % num_env
        env_name = train_env_name_list[env_id]

        logs, _ = trainer.pure_train_iteration_mix(
            num_steps=1, 
            no_prompt=args.no_prompt,
            masks = env_masks[env_name],
            env_name = env_name,
        )
        
        if (iter+1) % args.test_eval_interval == 0:   
            
            # log training information
            logger.record_tabular('Env Name', env_name) 
            for key, value in logs.items():
                logger.record_tabular(key, value)
            logger.dump_tabular() 
            group='test-seperate'
            
            # evaluate test
            test_eval_logs = trainer.eval_iteration_metaworld(
                env_masks, get_prompt, test_prompt_trajectories_list,
                eval_episodes, test_env_name_list, test_info, prompt_test_info, variant, test_env_list, iter_num=iter + 1, 
                no_prompt=args.no_prompt, group=group)

            total_success_mean = test_eval_logs[f'{group}-Total-Success-Mean']
            if total_success_mean > best_suc:
                best_suc = total_success_mean
                best_iter = iter + 1
            
            logger.log('Current Best success: {}, Iteration {}'.format(best_suc, best_iter))

        # update the mask
        if (iter+1) % args.mask_interval == 0:
            alpha_t=iter/variant['max_iters']
            change_num=cosine_annealing(alpha_t,eta_min=eta_min,eta_max=eta_max)
            gradient_set={}
            
            for env_name in train_env_name_list:
                gradient = trainer.update_gradient(
                    no_prompt=args.no_prompt,
                    masks = env_masks[env_name],
                    env_name = env_name,
                )
                
                gradient_set[env_name] = gradient
            
            harmo_gradient=get_harmo_gradient(gradient_set)
    
            env_masks_vectors={name:dict_to_vector(env_masks[name]) for name in env_masks}
            model_vec = parameters_to_vector(trainer.model.parameters())            

            dead_masks_vectors=mask_dead_harmo(harmo_gradient, gradient_set, env_masks_vectors, change_num)

            generate_masks_vectors=mask_generate_harmo(harmo_gradient, gradient_set, env_masks_vectors, \
                model_vec, gamma=variant["gamma"], change_num=change_num, mode=variant["select"])
            
            for name in env_masks_vectors:
                env_masks_vectors[name]=env_masks_vectors[name]-generate_masks_vectors[name]+dead_masks_vectors[name]
                vector_to_dict(env_masks[name], env_masks_vectors[name])
    
    trainer.save_model(env_name=args.env,  postfix='iter_'+str(iter + 1),  folder=save_path, env_masks=env_masks)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['MetaWorld'], default='MetaWorld') 
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--config_path', type=str, default='./config')
    parser.add_argument('--prefix_name', type=str, default='MT50')
    parser.add_argument('--suboptimal', action='store_true', default=False)

    parser.add_argument('--special_embedding', action='store_true', default=False)
    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--stochastic-prompt', action='store_true', default=False)
    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--no-prompt', action='store_true', default=False)
    parser.add_argument('--no_state_normalize', action='store_true', default=False) 
    parser.add_argument('--select', type=str, default='fisher')
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=10) 
    parser.add_argument('--max_iters', type=int, default=1000000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_eval_interval', type=int, default=5000)
    parser.add_argument('--test_eval_seperate_interval', type=int, default=10000)

    parser.add_argument('--mask_interval', type=int, default=500)
    parser.add_argument('--eta_min', type=int, default=0)
    parser.add_argument('--eta_max', type=int, default=100)
    parser.add_argument('--sparsity', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--merge_thres', type=float, default=25.)

    parser.add_argument('--render', action='store_true', default=False) 
    parser.add_argument('--load_path', type=str, nargs='+', default=None) # choose a model when in evaluation mode
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--data_path', type=str, default='./MT50/dataset')
    
    args = parser.parse_args()
    
    experiment_mix_env(variant=vars(args))