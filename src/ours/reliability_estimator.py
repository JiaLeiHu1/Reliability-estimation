import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Standard imports
import math
import json
import torch as t
import pandas as pd
import copy
import time
import logging
from datetime import datetime
import argparse
import random
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np

from Model_root import Model_root

# heatmap
class heatmap:
    def __init__(self, args, x_grid_min, x_grid_max,
                 y_grid_min, y_grid_max, prediction_loss_source,
                 prediction_loss_target):
        self.args = args
        self.prediction_loss_source = prediction_loss_source
        self.prediction_loss_target = prediction_loss_target
        x_grid_resolution = (x_grid_max-x_grid_min)/args.x_grid_num
        y_grid_resolution = (y_grid_max-y_grid_min)/args.y_grid_num
        
        if (x_grid_resolution == 0 and y_grid_resolution == 0) or \
            (x_grid_max==0 and x_grid_min==0 and args.x_grid_num==0) or \
                (y_grid_max-y_grid_min)==0 or \
                    (x_grid_max-x_grid_min)==0:

            self.x_grid_num = 1
            self.y_grid_num = 1
            self.x_grid_min = 0
            self.x_grid_max = 1
            self.y_grid_min = 0
            self.y_grid_max = 1
            self.x_grid_resolution = (self.x_grid_max-self.x_grid_min)/self.x_grid_num
            self.y_grid_resolution = (self.y_grid_max-self.y_grid_min)/self.y_grid_num
            self.grids = np.ones((1,1))

        else:
            self.x_grid_num = args.x_grid_num
            self.y_grid_num = args.y_grid_num
            self.x_grid_min = x_grid_min
            self.x_grid_max = x_grid_max
            self.y_grid_min = y_grid_min
            self.y_grid_max = y_grid_max
            self.x_grid_resolution = (self.x_grid_max-self.x_grid_min)/self.x_grid_num
            self.y_grid_resolution = (self.y_grid_max-self.y_grid_min)/self.y_grid_num
            self.grids = np.zeros((math.ceil((x_grid_max-x_grid_min)/self.x_grid_resolution)+1, 
                                math.ceil((y_grid_max-y_grid_min)/self.y_grid_resolution)+1))

        self.x_range = [self.x_grid_min, self.x_grid_max] 
        self.y_range = [self.y_grid_min, self.y_grid_max] 

    def get_grid_index(self, sample_x, sample_y):
        '''
        sample_x, sample_y: [n]
        '''
        if isinstance(sample_x, np.ndarray):
            sample_x = t.from_numpy(sample_x)

        if isinstance(sample_y, np.ndarray):
            sample_y = t.from_numpy(sample_y)

        x_index = (((sample_x - self.x_grid_min) // self.x_grid_resolution)).int()
        y_index = (((sample_y - self.y_grid_min) // self.y_grid_resolution)).int()

        return x_index.cpu(), y_index.cpu()
    
    def push(self, sample):
        '''
        forward 的 Docstring
        
        :param sample: [n,1,2]
        '''
        assert sample.shape[1] == 1
        if self.grids.shape[0] == 1 and self.grids.shape[1] == 1:
            return 
        else:
            for bs in range(sample.shape[0]):
                x_index, y_index = self.get_grid_index(sample[[bs],0,0],sample[[bs],0,1])
                self.grids[x_index,y_index] += 1
    
    def cal_prob(self):
        '''
        min_value = self.grids[self.grids!=0].min()
        max_value = self.grids[self.grids!=0].max()
        self.grids = self.grids/self.grids.sum()
        '''
        self.grids = self.grids/(self.grids.sum()*self.x_grid_resolution*self.y_grid_resolution)
    
    def query_prob(self, input_data, is_mean=True):
        '''
        query_prob 的 Docstring
        
        :param self: 说明
        :param input_data: [n,2]
        :is_mean: return a fiture; else, return tensor, dim=[n]

        '''
        assert input_data.shape[1] == 2
        x_mask = t.isnan(input_data[:,0])
        y_mask = t.isnan(input_data[:,1])
        assert (x_mask == y_mask).all().item()
        
        x_index, y_index = self.get_grid_index(input_data[:,0], input_data[:,1])

        # bounding index
        x_index = t.clip(x_index, 0, self.x_grid_num-1)
        y_index = t.clip(y_index, 0, self.y_grid_num-1)
        
        x_index[x_mask] = -1
        y_index[y_mask] = -1

        query_prob = self.grids[x_index, y_index]

        if isinstance(query_prob, float):
            query_prob = np.array([query_prob])

        query_prob = t.masked_fill(t.from_numpy(query_prob).to(self.args.device),
                                   x_mask,t.nan)

        if (query_prob == 0).any().item():
            query_prob[query_prob==0] = self.grids[self.grids!=0].min()/2

        if is_mean:
            query_prob[x_mask] = query_prob.nanmean()
            return query_prob.mean().item()
        else:
            return query_prob
        
class reliability_estimator:
    def __init__(self, args, model, isSource, if_specify_horizon=False):
        self.args = args
        self.isSource = isSource
        
        self.previous_length = args.previous_length // args.downsample_rate // args.TIMESTEP
        self.future_length = args.future_length // args.downsample_rate // args.TIMESTEP
        self.full_length = self.previous_length + self.future_length
        self.eval_future_length = args.eval_future_length // args.downsample_rate // args.TIMESTEP-1

        if not if_specify_horizon:
            self.source_horizon = self.previous_length
            self.target_horizon = self.previous_length
            
        # load input data
        input_data_source_path = model.trajectory_test_input_source_save_path
        output_data_source_path = model.trajectory_test_output_source_save_path
        input_data_target_path = model.trajectory_test_input_target_save_path
        output_data_target_path = model.trajectory_test_output_target_save_path
            
        self.input_data_source = pd.read_csv(input_data_source_path)
        self.output_data_source = pd.read_csv(output_data_source_path)
        self.input_data_target = pd.read_csv(input_data_target_path)
        self.output_data_target = pd.read_csv(output_data_target_path)

        # 去除掉一些行
        ## source
        self.input_data_source.columns = model.test_input_log_header
        self.input_data_source = self.input_data_source[self.input_data_source['Step']!='Step']
        self.input_data_source = self.input_data_source[~self.input_data_source['x'].isnull()]
        self.input_data_source[['Step','x','y']] = self.input_data_source[['Step','x','y']].astype(float)

        if 'mode' not in self.input_data_target.columns and \
            len(self.input_data_target.columns) == 7:
            self.input_data_target['mode'] = t.nan
            self.input_data_target = self.input_data_target[model.test_input_log_header]

        self.input_data_target.columns = model.test_input_log_header
        self.input_data_target = self.input_data_target[self.input_data_target['Step']!='Step']
        self.input_data_target = self.input_data_target[~self.input_data_target['x'].isnull()]
        self.input_data_target[['Step','x','y']] = self.input_data_target[['Step','x','y']].astype(float)

        ## target
        self.output_data_source.columns = model.test_output_log_header
        self.output_data_source = self.output_data_source[self.output_data_source['Step']!='Step']
        self.output_data_source = self.output_data_source[~self.output_data_source['MDE'].isnull()]
        self.output_data_source = self.output_data_source[(self.output_data_source['Step']==self.eval_future_length)]
        self.output_data_source[['Step','MDE','FDE']] = self.output_data_source[['Step','MDE','FDE']].astype(float)
        self.output_data_source = self.output_data_source.drop_duplicates('batch_index')
        
        self.output_data_target.columns = model.test_output_log_header
        self.output_data_target = self.output_data_target[self.output_data_target['Step']!='Step']
        self.output_data_target = self.output_data_target[~self.output_data_target['MDE'].isnull()]
        self.output_data_target = self.output_data_target[(self.output_data_target['Step']==self.eval_future_length)]
        self.output_data_target[['Step','MDE','FDE']] = self.output_data_target[['Step','MDE','FDE']].astype(float)
        self.output_data_target = self.output_data_target.drop_duplicates('batch_index')

        self.prediction_loss_source = self.output_data_source.loc[self.output_data_source['Step']==self.eval_future_length,'MDE'].mean()
        self.prediction_loss_target = self.output_data_target.loc[self.output_data_target['Step']==self.eval_future_length,'MDE'].mean()

        self.x_user_source_heatmap, self.x_user_target_heatmap, \
            self.x_lane_source_heatmap, self.x_lane_target_heatmap = self.construct_x_heatmap()

        self.load_data()

        # self.sx_source_heatmap, self.sx_target_heatmap = [self.x_source_heatmap[0]], [self.x_target_heatmap[0]]
        # for index, time_step in enumerate(range(1,self.previous_length),start=1):
        #     timestep_sx_source_heatmap, timestep_sx_target_heatmap = self.get_s_heatmap(self.x_source_heatmap[index], \
        #                                                                                 self.x_target_heatmap[index])
        #     self.sx_source_heatmap.append(timestep_sx_source_heatmap)
        #     self.sx_target_heatmap.append(timestep_sx_target_heatmap)
    
    def load_data(self):
        # load data
        for index, time_step in enumerate(range(self.source_horizon)):
            # pushing agent data
            source_timestep_df = self.input_data_source[(self.input_data_source['Step']==time_step)&\
                                                        (self.input_data_source['agent_type']=='agent')]
            self.x_user_source_heatmap[index].push(source_timestep_df[['x','y']].to_numpy()[:,None])

        for index, time_step in enumerate(range(self.target_horizon)):
            target_timestep_df = self.input_data_target[(self.input_data_target['Step']==time_step)&\
                                                        (self.input_data_source['agent_type']=='agent')]
            self.x_user_target_heatmap[index].push(target_timestep_df[['x','y']].to_numpy()[:,None])

        # pushing lane data
        if 'm' in self.args.dataset_type and not isinstance(self.x_lane_source_heatmap,type(None)):
            source_timestep_df = self.input_data_source[(self.input_data_source['agent_type']=='lane')]
            self.x_lane_source_heatmap[0].push(source_timestep_df[['x','y']].to_numpy()[:,None])
            target_timestep_df = self.input_data_target[(self.input_data_source['agent_type']=='lane')]
            self.x_lane_target_heatmap[0].push(target_timestep_df[['x','y']].to_numpy()[:,None])

        # calculate probability
        for index, time_step in enumerate(range(self.source_horizon)):
            self.x_user_source_heatmap[index].cal_prob()

        for index, time_step in enumerate(range(self.target_horizon)):
            self.x_user_target_heatmap[index].cal_prob()

        if not isinstance(self.x_lane_source_heatmap,type(None)) and \
            not isinstance(self.x_lane_target_heatmap,type(None)):
            self.x_lane_source_heatmap[0].cal_prob()
            self.x_lane_target_heatmap[0].cal_prob()

    def construct_x_heatmap(self):

        x_source_heatmap = [heatmap(self.args,
                            min(self.input_data_source.loc[self.input_data_source['Step']==time_step,'x'].min(),
                                self.input_data_target.loc[self.input_data_target['Step']==time_step,'x'].min()),
                            max(self.input_data_source.loc[self.input_data_source['Step']==time_step,'x'].max(),
                                self.input_data_target.loc[self.input_data_target['Step']==time_step,'x'].max()),
                            min(self.input_data_source.loc[self.input_data_source['Step']==time_step,'y'].min(),
                            self.input_data_target.loc[self.input_data_target['Step']==time_step,'y'].min()),
                            max(self.input_data_source.loc[self.input_data_source['Step']==time_step,'y'].max(),
                                self.input_data_target.loc[self.input_data_target['Step']==time_step,'y'].max()),
                            self.prediction_loss_source,
                            self.prediction_loss_target) \
                                for time_step in range(self.previous_length)]
        x_target_heatmap = [heatmap(self.args,
                            min(self.input_data_source.loc[self.input_data_source['Step']==time_step,'x'].min(),
                                self.input_data_target.loc[self.input_data_target['Step']==time_step,'x'].min()),
                            max(self.input_data_source.loc[self.input_data_source['Step']==time_step,'x'].max(),
                                self.input_data_target.loc[self.input_data_target['Step']==time_step,'x'].max()),
                            min(self.input_data_source.loc[self.input_data_source['Step']==time_step,'y'].min(),
                            self.input_data_target.loc[self.input_data_target['Step']==time_step,'y'].min()),
                            max(self.input_data_source.loc[self.input_data_source['Step']==time_step,'y'].max(),
                                self.input_data_target.loc[self.input_data_target['Step']==time_step,'y'].max()),
                            self.prediction_loss_source,
                            self.prediction_loss_target) \
                                for time_step in range(self.previous_length)]
        
        return x_source_heatmap, x_target_heatmap, None, None

    # timestep-wise cal
    def get_s_heatmap(self, x_heatmap_source, x_heatmap_target):
        print()
        pass

    def push_sample(self, sample):
        '''
        push_sample 的 Docstring
        
        :param self: 说明
        :param sample: [bs,T,2]
        '''
        for bs in range(sample.shape[0]):
            for time_step in range(self.full_length):
                if time_step < self.previous_length:
                    self.x_heatmap[time_step].push(sample[bs,time_step][None,None])
                else:
                    self.y_heatmap[time_step].push(sample[bs,time_step][None,None])


class reliability_estimator_es(reliability_estimator):
    '''
    1.适用于spatial-temporal(无车道的DL模型)
    '''
    def __init__(self, args, model, isSource):
        super(reliability_estimator_es, self).__init__(args, model, isSource)

    def load_data(self):
        self.input_data_source = self.input_data_source.groupby('batch_index').filter(lambda x:(x['x']!=0).any() \
                                                                                      or (x['y']!=0).any())
        self.input_data_target = self.input_data_target.groupby('batch_index').filter(lambda x:(x['x']!=0).any() \
                                                                                      or (x['y']!=0).any())
        # load data
        for index, time_step in enumerate(range(self.previous_length)):
            # pushing agent data
            source_timestep_df = self.input_data_source[(self.input_data_source['Step']==time_step)]
            self.x_user_source_heatmap[index].push(source_timestep_df[['x','y']].to_numpy()[:,None])
            target_timestep_df = self.input_data_target[(self.input_data_target['Step']==time_step)]
            self.x_user_target_heatmap[index].push(target_timestep_df[['x','y']].to_numpy()[:,None])

        # calculate probability
        for index, time_step in enumerate(range(self.previous_length)):
            self.x_user_source_heatmap[index].cal_prob()
            self.x_user_target_heatmap[index].cal_prob()

    def query(self, x, dataset_index, agent_num):
        '''
        x: tensor [agent_num, previous_len, 2], agent_num != 1
        agent_num: tensor [agent_num]
        '''
        # query user
        source_user_prob = np.cumprod([x_source_heatmap_timestep.query_prob(x[:,timestep]) 
                       for timestep, x_source_heatmap_timestep in enumerate(self.x_user_source_heatmap[1:], start=1)])[-1]             # query_prob from heatmap
        target_user_prob = np.cumprod([x_target_heatmap.query_prob(x[:,timestep]) 
                       for timestep, x_target_heatmap in enumerate(self.x_user_target_heatmap[1:], start=1)])[-1]              # query_prob from heatmap
        
        # prob
        source_prob = source_user_prob 
        target_prob = target_user_prob

        # agent_num to list
        agent_num = agent_num.tolist()

        weight_ratio = target_prob/source_prob
        
        if np.isnan(weight_ratio):
            return 1
        else:
            return weight_ratio

class reliability_estimator_SGAN_DADA(reliability_estimator_es):
    def __init__(self, args, model, isSource):
        super(reliability_estimator_SGAN_DADA, self).__init__(args, model, isSource)

    def query(self, x, seq_start_end):
        '''
        x: tensor, [agent_num, input_T, 2]
        seq_start_end: tensor, [bs, 2]
        '''
        # query user
        bs_source_user_prob = t.stack([x_source_heatmap_timestep.query_prob(x[:,timestep], 
                                                                    is_mean=False) 
                                                            for timestep, x_source_heatmap_timestep 
                                                            in enumerate(self.x_user_source_heatmap[1:], start=1)]) # [future_T-1, agent_num]
        bs_target_user_prob = t.stack([x_target_heatmap.query_prob(x[:,timestep], 
                                                           is_mean=False) 
                                                        for timestep, x_target_heatmap 
                                                        in enumerate(self.x_user_target_heatmap[1:], start=1)])     # [future_T-1, agent_num]
        batch_index = np.concatenate([np.array([batch_index]*(e-s)) for batch_index, 
                                      (s, e) in enumerate(seq_start_end.cpu())])[:,None]    # [agent_num,1]
        source_df = pd.concat([pd.DataFrame(bs_source_user_prob.cpu().numpy().T),
                               pd.DataFrame(batch_index)],axis=1)
        source_df.columns = ['1step','2step','3step','4step','batch_index']
        source_df_cumprod = source_df.groupby('batch_index').apply(lambda x:x.
                                                                   mean(0)).iloc[:,:4].cumprod(1).iloc[:,-1]
        source_prob = source_df['batch_index'].map(source_df_cumprod)

        target_df = pd.concat([pd.DataFrame(bs_target_user_prob.cpu().numpy().T),
                               pd.DataFrame(batch_index)],axis=1)
        target_df.columns = ['1step','2step','3step','4step','batch_index']
        target_df_cumprod = target_df.groupby('batch_index').apply(lambda x:x.\
                                                                   mean(0)).iloc[:,:4].cumprod(1).iloc[:,-1]
        target_prob = target_df['batch_index'].map(target_df_cumprod)

        assert not source_prob.isnull().any()
        assert not target_prob.isnull().any()
    
        weight_ratio = (target_prob/source_prob).values
        weight_ratio = np.clip(weight_ratio, 0.2, 1.3)
        weight_ratio = np.nan_to_num(weight_ratio, nan=1)

        return weight_ratio, batch_index
    
class reliability_estimator_s(reliability_estimator_es):
    def __init__(self, args, model):
        super(reliability_estimator_s, self).__init__(args, model)

    # def load_data(self):
    #     self.input_data_source = self.input_data_source.groupby('batch_index').filter(lambda x:(x['x']!=0).any() \
    #                                                                                   or (x['y']!=0).any())
    #     self.input_data_target = self.input_data_target.groupby('batch_index').filter(lambda x:(x['x']!=0).any() \
    #                                                                                   or (x['y']!=0).any())
    #     # load data
    #     for index, time_step in enumerate(range(self.previous_length)):
    #         # pushing agent data
    #         source_timestep_df = self.input_data_source[(self.input_data_source['Step']==time_step)]
    #         self.x_user_source_heatmap[index].push(source_timestep_df[['x','y']].to_numpy()[:,None])
    #         target_timestep_df = self.input_data_target[(self.input_data_target['Step']==time_step)]
    #         self.x_user_target_heatmap[index].push(target_timestep_df[['x','y']].to_numpy()[:,None])

    #     # calculate probability
    #     for index, time_step in enumerate(range(self.previous_length)):
    #         self.x_user_source_heatmap[index].cal_prob()
    #         self.x_user_target_heatmap[index].cal_prob()

    # def query(self, x, dataset_index, agent_num):
    #     '''
    #     x: tensor [agent_num, previous_len, 2], agent_num != 1
    #     agent_num: tensor [agent_num]
    #     '''
    #     # query user
    #     source_user_prob = np.cumprod([x_source_heatmap_timestep.query_prob(x[:,timestep]) 
    #                    for timestep, x_source_heatmap_timestep in enumerate(self.x_user_source_heatmap[1:], start=1)])[-1]             # query_prob from heatmap
    #     target_user_prob = np.cumprod([x_target_heatmap.query_prob(x[:,timestep]) 
    #                    for timestep, x_target_heatmap in enumerate(self.x_user_target_heatmap[1:], start=1)])[-1]              # query_prob from heatmap
        
    #     # prob
    #     source_prob = source_user_prob 
    #     target_prob = target_user_prob

    #     # agent_num to list
    #     agent_num = agent_num.tolist()

    #     if dataset_index == 0:

    #         MDE_loss = self.output_data_source.loc[self.output_data_source['batch_index'].isin(agent_num),'MDE']
            
    #         # loss_P_over_S
    #         importance_sampling_loss = source_prob/self.importance_sampling_S(MDE_loss, source_prob, 
    #                                                                           target_prob)*MDE_loss
    #         middle_distribution_loss = source_prob/self.middle_distribution_S(source_prob, 
    #                                                                           target_prob)*MDE_loss
    #         simple_norm_loss = source_prob/self.simple_norm_S(source_prob, target_prob)*MDE_loss
            
    #     else:
    #         MDE_loss = self.output_data_target.loc[self.output_data_target['batch_index'].isin(agent_num),'MDE']
    #         # loss_Q_over_S
    #         importance_sampling_loss = target_prob/self.importance_sampling_S(MDE_loss, source_prob, 
    #                                                                           target_prob)*MDE_loss
    #         middle_distribution_loss = target_prob/self.middle_distribution_S(source_prob, 
    #                                                                           target_prob)*MDE_loss
    #         simple_norm_loss = target_prob/self.simple_norm_S(source_prob, target_prob)*MDE_loss

    #     # print(dataset_index, MDE_loss, agent_num)
    #     # assert MDE_loss.shape[0] == len(agent_num)

    #     # changing dtype
    #     importance_sampling_loss = t.from_numpy(importance_sampling_loss.to_numpy())
    #     middle_distribution_loss = t.from_numpy(middle_distribution_loss.to_numpy())
    #     simple_norm_loss = t.from_numpy(simple_norm_loss.to_numpy())

    #     if importance_sampling_loss.shape[0] == 0:
    #         importance_sampling_loss = t.tensor([t.nan])
        
    #     if middle_distribution_loss.shape[0] == 0:
    #         middle_distribution_loss = t.tensor([t.nan])
        
    #     if simple_norm_loss.shape[0] == 0:
    #         simple_norm_loss = t.tensor([t.nan])

    #     return importance_sampling_loss, middle_distribution_loss,\
    #             simple_norm_loss

class reliability_estimator_vectornet(reliability_estimator):
    def __init__(self, args, model, isSource):
        self.source_horizon = 4
        self.target_horizon = 3
        super(reliability_estimator_vectornet,self).__init__(args, model,isSource,True)

        

    def construct_x_heatmap(self):
        input_user_data_source = self.input_data_source.loc[self.input_data_source['agent_type']=='agent']
        input_user_data_target = self.input_data_target.loc[self.input_data_target['agent_type']=='agent']
        input_lane_data_source = self.input_data_source.loc[self.input_data_source['agent_type']=='lane']
        input_lane_data_target = self.input_data_target.loc[self.input_data_target['agent_type']=='lane']

        x_user_source_heatmap = [heatmap(self.args,
                            min(input_user_data_source.loc[input_user_data_source['Step']==time_step,'x'].min(),
                                input_user_data_target.loc[input_user_data_target['Step']==time_step,'x'].min()),
                            max(input_user_data_source.loc[input_user_data_source['Step']==time_step,'x'].max(),
                                input_user_data_target.loc[input_user_data_target['Step']==time_step,'x'].max()),
                            min(input_user_data_source.loc[input_user_data_source['Step']==time_step,'y'].min(),
                            input_user_data_target.loc[input_user_data_target['Step']==time_step,'y'].min()),
                            max(input_user_data_source.loc[input_user_data_source['Step']==time_step,'y'].max(),
                                input_user_data_target.loc[input_user_data_target['Step']==time_step,'y'].max()),
                            self.prediction_loss_source,
                            self.prediction_loss_target) \
                                # for time_step in range(self.previous_length)
                                for time_step in range(self.source_horizon)
                                ]
        x_user_target_heatmap = [heatmap(self.args,
                            min(input_user_data_source.loc[input_user_data_source['Step']==time_step,'x'].min(),
                                input_user_data_target.loc[input_user_data_target['Step']==time_step,'x'].min()),
                            max(input_user_data_source.loc[input_user_data_source['Step']==time_step,'x'].max(),
                                input_user_data_target.loc[input_user_data_target['Step']==time_step,'x'].max()),
                            min(input_user_data_source.loc[input_user_data_source['Step']==time_step,'y'].min(),
                            input_user_data_target.loc[input_user_data_target['Step']==time_step,'y'].min()),
                            max(input_user_data_source.loc[input_user_data_source['Step']==time_step,'y'].max(),
                                input_user_data_target.loc[input_user_data_target['Step']==time_step,'y'].max()),
                            self.prediction_loss_source,
                            self.prediction_loss_target) \
                                # for time_step in range(self.previous_length)
                                for time_step in range(self.target_horizon)
                                ]
        
        if 'm' in self.args.dataset_type:
            x_lane_source_heatmap = [heatmap(self.args,
                                min(input_lane_data_source['x'].min(),
                                    input_lane_data_target['x'].min()),
                                max(input_lane_data_source['x'].max(),
                                    input_lane_data_target['x'].max()),
                                min(input_lane_data_source['y'].min(),
                                input_lane_data_target['y'].min()),
                                max(input_lane_data_source['y'].max(),
                                    input_lane_data_target['y'].max()),
                                self.prediction_loss_source,
                                self.prediction_loss_target)]
            x_lane_target_heatmap = [heatmap(self.args,
                                min(input_lane_data_source['x'].min(),
                                    input_lane_data_target['x'].min()),
                                max(input_lane_data_source['x'].max(),
                                    input_lane_data_target['x'].max()),
                                min(input_lane_data_source['y'].min(),
                                input_lane_data_target['y'].min()),
                                max(input_lane_data_source['y'].max(),
                                    input_lane_data_target['y'].max()),
                                self.prediction_loss_source,
                                self.prediction_loss_target)]
        else:
            x_lane_source_heatmap, x_lane_target_heatmap = None, None
        
        return x_user_source_heatmap, x_user_target_heatmap, x_lane_source_heatmap, x_lane_target_heatmap
    
    def query(self, user_data_with_nan, lane_data, dataset_type, iteration):
        '''
        query 的 Docstring
        
        :param self: 说明
        :param user_data_with_nan: [agent_num,input_T,2]?
        :param lane_data: [lane,2]
        :param time_step_list:list, 对应的长度, user list
        :param dataset_index: 0=source, 1=target
        :param iteation: 说明

        计算原则: 同一类个体之间取平均, 不同个体之间相乘
        '''
        # query user
        source_user_prob = [x_source_heatmap_timestep.query_prob(user_data_with_nan[:,timestep]) 
                       for timestep, x_source_heatmap_timestep in enumerate(self.x_user_source_heatmap[:-1])]             # query_prob from heatmap
        target_user_prob = [x_target_heatmap.query_prob(user_data_with_nan[:,timestep]) 
                       for timestep, x_target_heatmap in enumerate(self.x_user_target_heatmap[:-1])]              # query_prob from heatmap
        source_user_prob = np.array(source_user_prob)
        target_user_prob = np.array(target_user_prob)
        source_user_prob[np.isnan(source_user_prob)] = 1
        target_user_prob[np.isnan(target_user_prob)] = 1    
        source_user_prob = np.cumprod(source_user_prob)[-1]
        target_user_prob = np.cumprod(target_user_prob)[-1]

        # prob
        if 'm' in dataset_type:
            source_lane_prob = self.x_lane_source_heatmap[0].query_prob(lane_data)
            target_lane_prob = self.x_lane_target_heatmap[0].query_prob(lane_data)
            source_prob = np.mean([source_user_prob, source_lane_prob])
            target_prob = np.mean([target_user_prob, target_lane_prob])
        else:
            source_prob = source_user_prob
            target_prob = target_user_prob

        # calculate prob
        weight_ratio = target_prob/source_prob
        if np.isnan(weight_ratio):
            return 1
        else:
            return weight_ratio
        


parser = argparse.ArgumentParser(description='param for LSTM model')
# scenario
parser.add_argument('--previous_length', type=int, default=1000)
parser.add_argument('--future_length', type=int, default=2000)
parser.add_argument('--downsample_rate', type=int, default=2)
parser.add_argument('--x_grid_num', type=int, default=1000)
parser.add_argument('--y_grid_num', type=int, default=1000)
parser.add_argument('--x_grid_min', type=int, default=-10)
parser.add_argument('--y_grid_min', type=int, default=-10)
parser.add_argument('--x_grid_max', type=int, default=10)
parser.add_argument('--y_grid_max', type=int, default=10)
parser.add_argument('--dir_name', type=str, default='2025-00-00-00')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--grid_resolution', type=float, default=0.5)
parser.add_argument('--TIMESTEP', type=int, default=100)

if t.cuda.is_available():
    parser.add_argument('--device', type=str, default='cuda:0')
else:
    parser.add_argument('--device', type=str, default='cpu')



# args = parser.parse_args()

# model = reliability_estimator(args,'test',True)
# x_sample = np.random.uniform(-10,10,(100000,15,2))
# model.push_sample(x_sample)
# print()
