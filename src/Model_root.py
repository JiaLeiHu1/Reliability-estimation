import copy
import os, pickle
import numpy as np
import torch as t
import torch.cuda
import torch.nn as nn
import pandas as pd
import math, datetime, csv, yaml
import queue, threading, psutil
from joblib import Parallel, delayed
from torch.nn.functional import softmax
from torch.nn import functional as F
from torch.autograd import Variable

from Dataloader.Dataloader import data_root
def minMDE(traj, gt):
    '''
    traj: [bs, random_query, T, 2]
    gt: [bs, T, 2]

    return: float
    '''
    return (t.norm(traj - gt[:,None],dim=-1).mean([0,-1]).min()).item()

def minFDE(traj, gt):
    '''
    traj: [bs, random_query, T, 2]
    gt: [bs, T, 2]

    return: float
    '''
    return t.norm(traj[:,:,-1] - gt[:,None,-1],dim=-1).mean(0).min().item()

def CE(traj, gt):
    '''
    traj: [nums, T, 2]
    gt: [T, 2]

    return: float
    '''
    assert isinstance(gt, t.Tensor) and gt.ndim == 2
    assert isinstance(traj, t.Tensor) and traj.ndim == 3

    query_num = traj.shape[0]
    gt = gt[None].repeat(query_num,1,1)
    ce = 1/(t.norm(traj-gt,dim=-1).mean())

    return ce.item()

# 预处理轨迹数据
def preprocess(traj, rotate=None, diff=False):
    '''
    :param traj: [bs, obs_T, 2]
    :param rotate: [bs, 2, 2]

    :return: traj_norm: [bs, obs_T, 2] or [bs, obs_T-1, 2]
    '''
    # Norm
    last_obs_pos = t.mean(traj, dim=1).unsqueeze(1)  # [bs,1,2]
    traj_norm = traj - last_obs_pos

    if diff:
        traj_norm = t.cat([t.zeros(traj_norm.shape[0], 1, 2).to(device),
                           (traj_norm[:, 1:] - traj_norm[:, :-1])], dim=1)

    if rotate!=None:
        traj_norm = traj_norm.double() @ rotate.double()  # [bs, obs_T, 2]

    return traj_norm, last_obs_pos

scenarioToShortcut_dict = {'DR_DEU_Roundabout_OF': 'OF',
                           'DR_USA_Intersection_EP0': 'EP0',
                           'DR_USA_Intersection_EP1': 'EP1',
                           'DR_USA_Intersection_GL': 'GL',
                           'DR_USA_Intersection_MA': 'MA',
                           'DR_USA_Roundabout_EP': 'EP',
                           'DR_USA_Roundabout_FT': 'FT',
                           'DR_CHN_Merging_ZS':'ZS',
                           'DR_DEU_Merging_MT':'MT'}
shortcutToScenario_dict = dict(zip(scenarioToShortcut_dict.values(),
                                   scenarioToShortcut_dict.keys()))
device = 'cuda' if t.cuda.is_available() else 'cpu'

class Model_root(nn.Module):
    def __init__(self, args, model_name, create_File=True):
        super().__init__()
        self.args = args

        # Record
        self.model_root = 'result'

        self.model_name = model_name
        from tool import create_file
        create_file(self.model_root)
        self.dir_name = self.getDir_name()
        self.model_save_dir = os.path.join(data_root, self.model_root, f'trained_models_seed{args.seed}', args.source)
        self.result_save_dir = os.path.join(data_root, self.model_root, self.dir_name, self.model_name, 'results')
        self.source_csv_save_dir = os.path.join(data_root, self.model_root, f'source_csv_seed{args.seed}', self.args.source, model_name)
        self.preprocessed_data = os.path.join(data_root,'Preprocessed_data')
        self.eval_data = os.path.join(data_root,'Eval_data')
        self.model_eval_model_path = os.path.join(self.eval_data,f'{args.source}_{args.target}_{self.model_name}.pkl')

        self.visualization_save_dir = os.path.join(data_root, self.model_root, self.dir_name, self.model_name, 'visualization')
        self.train_log_header = ['Model', 'Epoches', 'Train Loss']
        self.test_input_log_header = ['Model', 'batch_index', 'Step', 'x', 'y', 'mode', 'agent_type', 'manu']
        self.test_output_log_header = ['Model', 'epoch', 'batch_index', 'Step', 'MDE', 'FDE', 'agent_type', 'manu']
        self.input_loss_source_log_header = ['Model', 'epoch', 'non_zero_global_index', 'MDE','weight_ratio', 
                                             'location', 
                                             'agent_type', 'manu', 'eval_future_length']
        self.input_loss_target_log_header = ['Model', 'epoch', 'non_zero_global_index', 'MDE','weight_ratio'
                                             , 'location', 'agent_type', 'manu', 'eval_future_length']


        self.previous_length = args.previous_length // args.downsample_rate // args.TIMESTEP
        self.future_length = args.future_length // args.downsample_rate // args.TIMESTEP
        self.full_length = self.previous_length + self.future_length
        self.eval_length = args.eval_future_length // args.downsample_rate // args.TIMESTEP

        self.train_logs_save_path = os.path.join(self.result_save_dir,'train_logs.csv')
        
        self.train_loss_save_path = os.path.join(self.source_csv_save_dir,
                                                                   'train_loss.csv')
        self.trajectory_test_input_source_save_path = os.path.join(self.source_csv_save_dir,
                                                                   'trajectory_test_input_source.csv')
        self.trajectory_test_output_source_save_path = os.path.join(self.source_csv_save_dir,'trajectory_test_output_source.csv')
        self.trajectory_test_input_target_save_path = os.path.join(self.result_save_dir,'trajectory_test_input_target.csv')
        self.trajectory_test_output_target_save_path = os.path.join(self.result_save_dir, 'trajectory_test_output_target.csv')
        self.input_loss_source_save_path = os.path.join(self.result_save_dir, 'input_loss_source.csv')
        self.input_loss_target_save_path =os.path.join(self.result_save_dir, 'input_loss_target.csv')
        self.final_result_save_path =os.path.join(self.result_save_dir, 'final_result.csv')

        self.error_decomposition_save_path = os.path.join(self.result_save_dir,'error_decomposition.csv')
        self.best_model_save_path = os.path.join(self.model_save_dir,\
                                                f'best_epoch_{self.model_name}.pt')
        
        self.visualization_diagnose_data_bar_save_path = os.path.join(self.visualization_save_dir,
                                                                      f'diagnose_data_bar_{args.source}_{args.target}.svg')

        if create_File:
            create_file(self.model_save_dir)
            create_file(self.result_save_dir)
            create_file(self.source_csv_save_dir)
            create_file(self.visualization_save_dir)
            create_file(self.eval_data)
            create_file(os.path.join(self.preprocessed_data,'ind'))
            create_file(os.path.join(self.preprocessed_data,'highd'))
            create_file(os.path.join(self.preprocessed_data,'ngsim'))
            create_file(os.path.join(self.preprocessed_data,'interaction'))

            self.check_csv_exist(self.train_logs_save_path,
                                 self.train_log_header)
            self.check_csv_exist(self.trajectory_test_input_source_save_path,
                                 self.test_input_log_header)
            self.check_csv_exist(self.trajectory_test_output_source_save_path,
                                 self.test_output_log_header)
            self.check_csv_exist(self.trajectory_test_input_target_save_path,
                                 self.test_input_log_header)
            self.check_csv_exist(self.trajectory_test_output_target_save_path,
                                 self.test_output_log_header)
            self.check_csv_exist(self.input_loss_source_save_path,
                                 self.input_loss_source_log_header)
            self.check_csv_exist(self.input_loss_target_save_path,
                                 self.input_loss_target_log_header)

            # with open(os.path.join(self.model_root,
            #                     self.dir_name,
            #                     self.model_name) + '\\Params.yaml', 'w', encoding='utf-8') as f:
            #     yaml.dump(data=args, stream=f, allow_unicode=True)

        self.loss = 0

    def check_csv_exist(self, csv_path, csv_log_header):
        if not os.path.exists(csv_path):
            with open(csv_path, 
                    'a+', newline='') as write_obj:
                csv_writer = csv.writer(write_obj)
                csv_writer.writerow(csv_log_header)
        else:
            df = pd.read_csv(csv_path)
            if df.shape[0] ==0 and (len(df.columns)!=len(csv_log_header)\
                or (df.columns != csv_log_header).any()):
                os.remove(csv_path)
                with open(csv_path, 
                    'a+', newline='') as write_obj:
                    csv_writer = csv.writer(write_obj)
                    csv_writer.writerow(csv_log_header)

    def get_epoch_model_path(self, epoch):
        return os.path.join(self.model_save_dir,
                            f'epoch_{epoch}_{self.model_name}.pt')

    def cal_write_final_result(self):
        '''
        cal first, second, thrid term
        
        return:
            write a csv
        '''
        input_loss_source = pd.read_csv(self.input_loss_source_save_path)
        input_loss_target = pd.read_csv(self.input_loss_target_save_path)
        trajectory_test_output_source = pd.read_csv(self.trajectory_test_output_source_save_path)
        trajectory_test_output_target = pd.read_csv(self.trajectory_test_output_target_save_path)

        assert input_loss_source['eval_future_length'].unique().shape[0] ==\
            input_loss_target['eval_future_length'].unique().shape[0] == 1
        
        eval_future_length = (input_loss_source['eval_future_length'].unique() // self.args.downsample_rate \
                        // self.args.TIMESTEP)[0] - 1

        trajectory_test_output_source = trajectory_test_output_source[trajectory_test_output_source['Step']==eval_future_length]
        trajectory_test_output_target = trajectory_test_output_target[trajectory_test_output_target['Step']==eval_future_length]
        source_loss = trajectory_test_output_source['MDE'].mean()
        target_loss = trajectory_test_output_target['MDE'].mean()

        first_term = pd.DataFrame((input_loss_source[['E_s[R_P (x)]_IS','E_s[R_P (x)]_middle_dis',
                                        'E_s[R_P (x)]_simple_norm']] - source_loss).mean()).T
        second_term = pd.DataFrame((input_loss_target[['E_s[R_Q (x)]_IS','E_s[R_Q (x)]_middle_dis',
                                        'E_s[R_Q (x)]_simple_norm']]).mean().to_numpy() - \
                      (input_loss_source[['E_s[R_P (x)]_IS','E_s[R_P (x)]_middle_dis',
                                        'E_s[R_P (x)]_simple_norm']]).mean().to_numpy()).T
        third_term = pd.DataFrame((target_loss - input_loss_source[['E_s[R_P (x)]_IS','E_s[R_P (x)]_middle_dis',
                                        'E_s[R_P (x)]_simple_norm']]).mean()).T
        
        sample_method = list(map(lambda x:x[-1],first_term.columns.str.split('_')))
        first_term.columns = list(map(lambda x:f'first_term_{x}', sample_method))
        second_term.columns = list(map(lambda x:f'second_term_{x}', sample_method))
        third_term.columns = list(map(lambda x:f'third_term_{x}', sample_method))

        pd.concat([first_term, second_term, third_term], axis=1).to_csv(self.final_result_save_path)

    def get_preprocessed_data_root(self, isSource, mode):
        global data_root
        if isSource:
            preprocessed_data_root = os.path.join(data_root,f'Preprocessed_data',self.args.source,mode)
        else:
            preprocessed_data_root = os.path.join(data_root,f'Preprocessed_data',self.args.target,mode)
        
        return preprocessed_data_root
    



    def eval_final_result(self):
        source_df = pd.read_csv(self.input_loss_source_save_path)
        target_df = pd.read_csv(self.input_loss_target_save_path)
        source_test_df = pd.read_csv(self.trajectory_test_output_source_save_path)
        source_test_df = source_test_df.loc[(~source_test_df['MDE'].isnull())]
        source_test_df = source_test_df.loc[source_test_df['MDE']!='MDE']
        source_mu = source_test_df['MDE'].astype(float).mean()

        target_test_df = pd.read_csv(self.trajectory_test_output_target_save_path)
        target_test_df = target_test_df.loc[(~target_test_df['MDE'].isnull())]
        target_test_df = target_test_df.loc[target_test_df['MDE']!='MDE']
        target_mu = target_test_df['MDE'].astype(float).mean()

        # loss_P_over_S
        loss_P_over_S = source_df[~source_df['E_s[R_P (x)]'].isnull()].iloc[:,2].mean()
        # loss_P_over_P
        loss_P_over_P = source_mu

        # loss_Q-P_over_S
        loss_QP_over_S = target_df[~target_df['E_s[R_Q (x)]'].isnull()].iloc[:,2].mean()- source_df[~source_df['E_s[R_P (x)]'].isnull()].iloc[:,2].mean()

        # loss_Q_over_Q
        loss_Q_over_Q = target_mu
        # loss_Q_over_S
        loss_Q_over_S = target_df[~target_df['E_s[R_Q (x)]'].isnull()].iloc[:,2].mean()

        ## first term: X shift (X to S)
        first_term = source_df[~source_df['E_s[R_P (x)]'].isnull()].iloc[:,2].mean() - source_mu
        ## second term: Y|X shift (X to S)
        second_term = target_df[~target_df['E_s[R_Q (x)]'].isnull()].iloc[:,2].mean()- source_df[~source_df['E_s[R_P (x)]'].isnull()].iloc[:,2].mean()
        ## third term: X shift (S to Q)
        third_term = target_mu - target_df[~target_df['E_s[R_Q (x)]'].isnull()].iloc[:,2].mean()

        with open(self.error_decomposition_save_path, 
                        'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(['loss_P_over_S', 'loss_P_over_P', 'loss_Q-P_over_S',
                                 'loss_Q_over_Q', 'loss_Q_over_S', 'first_term', 
                                 'second_term', 'third_term', 'predictability shift'])
            csv_writer.writerow([loss_P_over_S, loss_P_over_P, loss_QP_over_S,
                                 loss_Q_over_Q, loss_Q_over_S, first_term,
                                 second_term, third_term])

    def print_log(self, args, epoch, predict_len,
                  loss, epoch_mde_loss, epoch_fde_loss,
                  predict_tensor, future_data_gt,
                  isTrain=True):
        if isTrain:
            print(r'*' * 20)
            print(f'模型是{args.model}')
            print(f'场景是{args.scenario}')
            print(rf'当前epoch{epoch}预测长度{predict_len}损失为{loss.item()},\
                                        mde为{epoch_mde_loss},\
                                                fde为{epoch_fde_loss}')
            print(rf'部分预测值为{predict_tensor[-1, -5:, :]}')
            print(f'其真实值为{future_data_gt[-1, -5:, :]}')
            print(r'*' * 20)

        else:
            # print(r'*' * 20)
            # print(f'场景是{args.scenario}')
            # print(rf'当前预测长度{predict_len}的,\
            #                             mde为{epoch_mde_loss},\
            #                                     fde为{epoch_fde_loss}')
            # print(rf'部分预测值为{predict_tensor[-1, -5:, :]}')
            # print(f'其真实值为{future_data_gt[-1, -5:, :]}')
            # print(r'*' * 20)
            pass

    def write_train_log(self, model, epoch, train_loss):
        with open(self.result_save_dir + '\\train_logs.csv', 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow([model,epoch,train_loss])

    def pushData(self, traj):
        self.data_pools = t.cat([self.data_pools, traj])

    def getDir_name(self):
        return self.args.dir_name+f'_{self.args.source}_{self.args.target}_seed_{self.args.seed}'
    
    def print_training_logs(self, args, epoch, batch_index,
                              loss,
                              mde,
                              fde,
                              now_lr=None):
        print(f'Training model:{self.model_name}, seed:{args.seed}')
        print(f'epoch:{epoch},batch_index:{batch_index}')
        print(f'loss:{loss}, mde:{mde}, fde:{fde}')
        print(f'Source:{args.source},Target:{args.target}')
        self.print_memory()
        print('-'*50)
        
    def print_testing_logs(self, args, epoch, batch_index,
                              mde,
                              fde,
                              isSource):
        print(f'Testing model:{self.model_name} at epoch {epoch} in Source') \
            if isSource else print(f'Testing model:{self.model_name} at epoch {epoch} in Target')
        print(f'batch_index:{batch_index}, seed:{args.seed}')
        print(f'mde:{mde}, fde:{fde}')
        print(f'Source:{args.source},Target:{args.target}')
        self.print_memory()

    def print_eval_logs(self, eval_epoch, 
                        weight_ratio, dataset_index, sample_type):
        print(f'model:{self.model_name}, eval_epoch:{eval_epoch}')
        print(f'Source:{self.args.source},Target:{self.args.target}')
    
        if dataset_index == 0:
            print(f'eval {self.args.source} dataset, sample_type:{sample_type}, weight_ratio:{weight_ratio}, seed:{self.args.seed}')
        else:
            print(f'eval {self.args.target} dataset, sample_type:{sample_type}, weight_ratio:{weight_ratio}, seed:{self.args.seed}')
        self.print_memory()

    def print_memory(self):
        mem = psutil.virtual_memory()
        print(f'Memory available: {mem.available/1024/1024/1024:.2f} G')

    def write_test_trajectory_output_logs(self, epoch, batch_index, mde, fde,
                                   isSource, agent_type=t.nan, manu=t.nan):
        '''
        write_test_trajectory_output_logs 的 Docstring
        
        :param self: 说明
        :param batch_index: ['Model', 'epoch', 'batch_index', 'Step', 'MDE', 'FDE',\
            'agent_type', 'manu']
        :param mde: 说明
        :param fde: 说明
        :param isSource: 说明
        '''
        T = mde.shape[0]
        if isSource:
            path = self.trajectory_test_output_source_save_path
        else:
            path = self.trajectory_test_output_target_save_path

        data = np.concatenate([np.array([self.model_name]*T)[:,None],
                            np.array([epoch]*T)[:,None],   
                        np.array([batch_index]*T)[:,None],
                        np.arange(T)[:,None],
                        mde.cpu().numpy()[:,None],
                        fde.cpu().numpy()[:,None],
                        np.array([agent_type]*T)[:,None],
                        np.array([manu]*T)[:,None],
                        ],axis=1)

        with open(path, 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerows(data)

    def write_train_loss_logs(self, epoch, loss):
        path = self.train_loss_save_path
        T = 1
        data = np.concatenate([np.array([self.model_name])[:,None],
                        np.array([epoch])[:,None],
                        np.array([loss])[:,None]
                        ],axis=1)

        with open(path, 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerows(data)

    def write_test_trajectory_input_logs(self, batch_index, input_data,
                                   isSource, mode, agent_type=t.nan, manu=t.nan):
        '''
        write_test_trajectory_input_logs 的 Docstring
        ['Model', 'batch_index', 'Step', 'x', 'y', 'mode', 'agent_type', 'manu']

        :param self: 说明
        :param batch_index: 说明
        :param input_data: [1,input_T,2]
        :param isSource: 说明
        '''
        assert input_data.shape[0] == 1
        _, T, _ = input_data.shape
        if isSource:
            path = self.trajectory_test_input_source_save_path
        else:
            path = self.trajectory_test_input_target_save_path

        data = np.concatenate([np.array([self.model_name]*T)[:,None],
                        np.array([batch_index]*T)[:,None],
                        np.arange(T)[:,None],
                        input_data.cpu().numpy()[0],
                        np.array([mode]*T)[:,None],                      
                        np.array([agent_type]*T)[:,None],
                        np.array([manu]*T)[:,None],
                        ],axis=1)

        with open(path, 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerows(data)

    def write_input_loss(self, epoch, mde, weight_ratio, non_zero_global_index,
                         dataset_index, location, 
                         agent_type, manu, eval_future_length):
        '''
        write_input_loss 的 Docstring
        
        :param self: 说明
        :param importance_sampling_loss, middle_distribution_loss,\
            simple_norm_loss: [non_agent_num] tensor
        :param dataset_index: 说明
        :param location, agent_type, manu: [non_agent_num, 1]
        :non_zero_global_index: 只包含不全为0的global index, tensor

        ['Model', 'epoch', 'batch_index', 'MDE', 'weight_ratio',
        'location', 'agent_type', 'manu', 'eval_future_length']
        '''
        # 如果是空, 直接返回
        if non_zero_global_index.shape[0] == 0:
            return

        if dataset_index == 0:
            path = self.input_loss_source_save_path

        else:
            path = self.input_loss_target_save_path

        if isinstance(location,t.Tensor) and location.ndim == 1:
            location = location[:,None]
        if isinstance(agent_type,t.Tensor) and agent_type.ndim == 1:
            agent_type = agent_type[:,None]
        if isinstance(manu,t.Tensor) and manu.ndim == 1:
            manu = manu[:,None]

        if isinstance(location,str):
            location = np.array([location])[:,None]
        if isinstance(agent_type,str):
            agent_type = np.array([agent_type])[:,None]
        if isinstance(manu,str):
            manu = np.array([manu])[:,None]

        data = np.concatenate([np.array([self.model_name]*non_zero_global_index.shape[0])[:,None],\
                               np.array([epoch]*non_zero_global_index.shape[0])[:,None],\
                        non_zero_global_index[:,None],\
                        mde[:,None].numpy(),\
                        weight_ratio[:,None].numpy(),\
                        location,\
                        agent_type,\
                        manu,\
                        np.array([eval_future_length]*non_zero_global_index.shape[0])[:,None]],
                        axis=1)
        
        with open(path, 'a+', newline='') as write_obj:
            np.savetxt(write_obj, data, fmt='%s', 
                       delimiter=',', comments='')
            write_obj.close()


class classical_encoder_decoder(nn.Module):        

    def __init__(self):
        super(classical_encoder_decoder, self).__init__()
        self.step = 0
        pass

    def get_linear(self):
        '''
        Args:
            linear_params_dict: {'w':w,     (2*hidden_size)
                                 'b':b}:    (2)

        Returns:
        '''
        return classical_linear(self.sample_params_dict['output'])

    def getMCMCLoss(self, inputs, gt, args):
        '''
        :param inputs: [bs, T1, 2]
        :param gt: [bs, T2, 2]

        分子: 拥有的所有总数
        分母: 每个batch包含的元素总数
        Reference:
        '''
        self.step += 1
        episode = args.MCMC_a * ((args.MCMC_b + self.step) ** args.MCMC_gamma)    # args.MCMC_gamma : (0.5,1]
        # 预处理
        from tool import preprocess, MDE
        # inputs_norm, _ = preprocess(inputs)
        # gt_norm, _ = preprocess(gt)
        input_data_norm, _ = preprocess(inputs, diff=True)
        last_pos = inputs[:,[-1]]
        # forwards
        pred_norm, posterior_loss = self(input_data_norm)
        MDE_loss = MDE(gt, pred_norm.cumsum(1)+last_pos)
        # print(f'MDE_loss:{MDE_loss}')
        # loss
        return -(episode/2*(args.MCMC_posterior_loss_weight*posterior_loss +
                            args.MCMC_trajectory_augment_nums / args.MCMC_iterations_batchsize * MDE_loss))

    # 初始化存放采样参数的列表
    def init_sample_params_dict(self):
        self.sample_params_dict = {}

# class classical_linear(nn.Module):
#     def __init__(self, linear_params_dict):
#         super(classical_linear, self).__init__()
#
#         output_size, input_size = linear_params_dict['w'].shape
#         self.linear = nn.Linear(input_size, output_size).to(device)
#         self.w_sampler = linear_params_dict['w_sampler']
#         self.b_sampler = linear_params_dict['b_sampler']
#
#         with torch.no_grad():  # 禁止自动求梯度
#             self.linear.weight.copy_(linear_params_dict['w'])  # 初始化权重
#             self.linear.bias.copy_(linear_params_dict['b'])  # 初始化偏置
#
#     def forward(self, x):
#         return self.linear(x)
#
#     def get_posterior_loss(self):
#         return (self.w_sampler.log_posterior(self.linear.weight) +
#                 self.b_sampler.log_posterior(self.linear.bias))

class classical_linear(nn.Module):
    def __init__(self, linear_params_dict):
        super(classical_linear, self).__init__()

        self.w = linear_params_dict['w']  # 初始化权重
        self.b = linear_params_dict['b']  # 初始化权重
        self.w_sampler = linear_params_dict['w_sampler']
        self.b_sampler = linear_params_dict['b_sampler']

    def forward(self, x):
        return F.linear(x, self.w, self.b)

    def get_posterior_loss(self):
        return (self.w_sampler.log_posterior(self.w) +
                self.b_sampler.log_posterior(self.b))














