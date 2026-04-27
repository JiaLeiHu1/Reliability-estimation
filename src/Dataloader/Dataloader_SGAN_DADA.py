import torch as t
import numpy as np
import pandas as pd
import os
import datetime
import csv
import yaml
import pickle, math
import random

from tqdm import tqdm
from scipy import spatial
from torch.utils.data import DataLoader
# from grip_train_Interaction import preprocess_data
from math import *
from joblib import Parallel, delayed

from Dataloader.Dataloader import Dataloader_root, MANU2ONEHOT, ONEHOT2MANU
from Dataloader.Dataloader_DADA import Dataloader_DADA
from tool import *

# 定义常数
total_feature_dimension = len(['frame_id','track_id','agent_type','x','y']) + 1 # we add mark "1" to the end of each row to indicate that this row exists

ind_dict = {0: list(range(1,7)),
            1: list(range(7,18)),
            2: list(range(18,30)),
            3: list(range(30,33))}

device = 'cuda' if t.cuda.is_available() else 'cpu'
S2MS = 1000

def seq_collate(data):
    (
        obs_seq_list,  # [bs, num, 2, 8]
        pred_seq_list,
        whole_seq_list,

        obs_seq_rel_list,
        pred_seq_rel_list,
        whole_seq_rel_list,

        # non_linear_ped_list,
        loss_mask_list,
        location_list,
        agent_type_list,
        manu_list,

    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]  # [num0, num1, ...]
    cum_start_idx = [0] + np.cumsum(_len).tolist()  # [num0, num0+num1, ...]
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]  # [bs, 2]
    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = t.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = t.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    whole_traj = t.cat(whole_seq_list, dim=0).permute(2, 0, 1)

    obs_traj_rel = t.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = t.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    whole_traj_rel = t.cat(whole_seq_rel_list, dim=0).permute(2, 0, 1)

    # non_linear_ped = t.cat(non_linear_ped_list)
    loss_mask = t.cat(loss_mask_list, dim=0)
    seq_start_end = t.LongTensor(seq_start_end)
    out = [
        obs_traj,  # [8, bs_num, 2]
        pred_traj,  # [12, bs_num, 2]
        whole_traj,  # [20, bs_num, 2]

        obs_traj_rel,  # [8, bs_num, 2]
        pred_traj_rel,  # [12, bs_num, 2]
        whole_traj_rel,  # [20, bs_num, 2]

        # non_linear_ped,
        loss_mask,
        seq_start_end,  # [bs, 2]
        location_list,
        agent_type_list,
        manu_list
    ]

    return tuple(out)

# 编码车辆为0，行人为1
class Dataloader_SGAN_DADA(Dataloader_root):
    def __init__(self, args,
                mode='train',
                 isSource=True,
                 is_DA=False,
                 ):
        super(Dataloader_SGAN_DADA, self).__init__(args, mode, isSource, is_DA)

        # 保存在Data/cleared_data里
        '''
        1.整理每个场景train test的行人,车辆数据,主要是修改编号 (用之前的代码应该就可以), 最后可以整理出n个文件(n为场景)
        2.对每个文件直接做降采样,不用留到后面了.然后用这n个场景来做预处理
        '''
        
        # 这里需要将每个list文件堆叠成grip官方的样子
        self.reset_track_id = 0
        if isSource:
            path = os.path.join(self.preprocessed_data_root, f'preprocessed_sgan_dada_source_{is_DA}.pkl')
            self.scenario = args.source
        else:
            path = os.path.join(self.preprocessed_data_root, f'preprocessed_sgan_dada_target_{is_DA}.pkl')
            self.scenario = args.target
            
        self.val_fraction = 0
        self.skip = 1

        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.raw_data = pickle.load(f)
        else:
            self.raw_data = self.getFullData()
            
            with open(path, 'wb') as f:
                pickle.dump(self.raw_data, f)

        self.num_seq, self.seq_start_end,\
        self.obs_traj,\
        self.pred_traj,\
        self.whole_traj,\
        self.obs_traj_rel,\
        self.pred_traj_rel,\
        self.whole_traj_rel,\
        self.loss_mask,\
        self.location_list,\
        self.agent_type_list,\
        self.manu_list = self.raw_data

        del self.raw_data
        
    def __len__(self):
        return self.num_seq


    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],  # [nums, 2, 8]
            self.pred_traj[start:end, :],  # [nums, 2, 12]
            self.whole_traj[start:end, :],  # [nums, 2, 20]

            self.obs_traj_rel[start:end, :],  # [nums, 2, 8]
            self.pred_traj_rel[start:end, :],  # [nums, 2, 12]
            self.whole_traj_rel[start:end, :],  # [nums, 2, 20]

            # self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            self.location_list[start:end],
            self.agent_type_list[start:end],
            self.manu_list[start:end],
        ]
        return out


    def process_origin_csv(self, csv_path,
                           unified_column=["frame", "ped", "x", "y"]):
        '''
        process_origin_csv 的 Docstring
        
        :param self: 说明
        :param csv_path: 说明
        :param unified_column: 说明

        车辆统称:0
        行人统称:1
        自行车统称:2
        '''
        if self.scenario == 'interaction':
            veh_data = pd.read_csv(csv_path[0])

            # 需要把agent_type改成0 1编码，车辆为0，行人为1
            veh_data['agent_type'] = veh_data['agent_type'].str.replace('car', '0')
            veh_data['agent_type'] = veh_data['agent_type'].astype(int)
            veh_data['track_id'] = veh_data['track_id'].astype(int)
            veh_data['psi_rad'] = veh_data['psi_rad']*180/t.pi

            if os.path.exists(csv_path[1]):
                ped_data = pd.read_csv(csv_path[1])
                ped_data['track_id'] = ped_data['track_id'].str.lstrip('P')
                ped_data['length'] = 0
                ped_data['width'] = 0
                ped_data['psi_rad'] = 0
                ped_data['agent_type'] = ped_data['agent_type'].str.replace('pedestrian/bicycle', '1')
                ped_data['agent_type'] = ped_data['agent_type'].astype(int)
                ped_data['track_id'] = ped_data['track_id'].astype(int)
                # 重新打编号
                max_veh_track_id = veh_data['track_id'].max()
                ped_data['track_id'] += max_veh_track_id
                record_data = pd.concat([veh_data, ped_data], axis=0)
            else:
                record_data = veh_data
            
            # 在这里就进行降采样了
            max_frame = record_data['frame_id'].max()
            min_frame = record_data['frame_id'].min()
            target_indices = list(range(min_frame, max_frame, self.args.downsample_rate))

            record_data = record_data.set_index(['frame_id'])
            valid_indices = record_data.index.intersection(target_indices)
            # 筛选数据
            record_data = record_data.loc[valid_indices, :].reset_index()

            # 调整顺序
            record_data = record_data[['frame_id', 'track_id', 'agent_type', 'x',
                                                'y','psi_rad']].sort_values(['frame_id', 'track_id'])

            df = record_data[['frame_id','track_id','x','y','agent_type','psi_rad']]          

        elif self.scenario == 'highd':
            track_df = pd.read_csv(csv_path[0])
            # track_df = track_df[track_df['frame']%4==1]     # 和interaction数据集作统一
            track_meta_df = pd.read_csv(os.path.join(csv_path[1]))
            # df = pd.merge(track_df, track_meta_df, on=['id'], how='left')
            # del track_df, track_meta_df
            # # 需要把agent_type改成0 1编码，车辆为0，行人为1
            # # df['timestamp_ms'] = df['frame']*100
            # # df = df[['id','frame','timestamp_ms','class','x','y',
            # #         'xVelocity','yVelocity','drivingDirection','height_y','width_x']]

            # # 在这里就进行降采样了
            # max_frame = df['frame'].max()
            # min_frame = df['frame'].min()
            # target_indices = list(range(min_frame, max_frame, self.args.downsample_rate))

            # df = df.set_index(['frame'])
            # valid_indices = df.index.intersection(target_indices)
            # # 筛选数据
            # df = df.loc[valid_indices, :].reset_index()

            df, track_meta_df = self.downsample_highd(track_df, track_meta_df)

            df['class'] = 0

            # 调整顺序
            df = df[['frame','id','x','y','class','laneId']]

        elif self.scenario == 'ind':
            '''
            ['recordingId', 'trackId', 'frame', 'trackLifetime', 'xCenter',
            'yCenter', 'heading', 'width', 'length', 'xVelocity', 'yVelocity',
            'xAcceleration', 'yAcceleration', 'lonVelocity', 'latVelocity',
            'lonAcceleration', 'latAcceleration']
            '''
            df = pd.read_csv(csv_path[0])
            track_meta_df = pd.read_csv(csv_path[1])
            # df = df[df['frame']%4==1]     # 和interaction数据集作统一
            # # map, such thta column 'frame' could start from 1
            # map_dict = dict(zip(df[df['frame']%4==1]['frame'],  
            #                     df[df['frame']%4==1]['frame']//4))
            # df['frame'] = df['frame'].map(map_dict)
            # assert not df['frame'].isnull().any()
            
            # df = pd.merge(df, track_meta_df, on=['trackId','recordingId'], how='left')

            # # 在这里就进行降采样了
            # max_frame = df['frame'].max()
            # min_frame = df['frame'].min()
            # target_indices = list(range(min_frame, max_frame, self.args.downsample_rate))

            # df = df.set_index(['frame'])
            # valid_indices = df.index.intersection(target_indices)
            # # 筛选数据
            # df = df.loc[valid_indices, :].reset_index()
            df, track_meta_df = self.downsample_ind(df, track_meta_df)


            # 调整顺序
            df = df[['frame','trackId','class','xCenter','yCenter','heading']]
            df['class'] = df['class'].str.replace('pedestrian', '1')
            df['class'] = df['class'].str.replace('bicycle', '2')
            df['class'] = df['class'].str.replace('car', '0')
            df['class'] = df['class'].str.replace('truck_bus', '0')
            df['class'] = df['class'].astype(int)
            df = df[['frame','trackId','xCenter','yCenter','class','heading']]
            
            del track_meta_df

        elif self.scenario == 'ngsim':
            df = pd.read_csv(csv_path[0])
            # df['v_Class'] = 0
            # df = df[['Frame_ID','Vehicle_ID','Global_X','Global_Y','v_Class','Lane_ID']]
            
            # # 在这里就进行降采样了
            # max_frame = df['Frame_ID'].max()
            # min_frame = df['Frame_ID'].min()
            # target_indices = list(range(min_frame, max_frame, self.args.downsample_rate))

            # df = df.set_index(['Frame_ID'])
            # valid_indices = df.index.intersection(target_indices)
            # # 筛选数据
            # df = df.loc[valid_indices, :].reset_index()

            df = self.downsample_ngsim(df)

            # 调整顺序
            df = df.sort_values(['Frame_ID', 'Vehicle_ID'])

        else:
            assert False
        
        if self.scenario != 'ngsim':
            df.columns = unified_column
        else:
            df.columns = unified_column[:4] # 只index ['frame', 'ped', 'x', 'y'], 不要agent_type和psi_degree

        df = df.sort_values([unified_column[0],
                             unified_column[1]])
        df[unified_column[1]] += self.reset_track_id
        self.reset_track_id = df[unified_column[1]].max()

        return df


    def getFullData(self):
        file_list = self.getFullData_list(is_DA=self.is_DA) 
        return_feature_list = []
        index = 0

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        location_list = []
        agent_type_list = []
        manu_list = []

        min_ped = 1

        for dataset_index, csv_path in enumerate(tqdm(file_list)):
            if '\\' in csv_path[0]:
                split_symbol = '\\'
            elif '/' in csv_path[0]:
                split_symbol = '/'

            if self.scenario == 'ind':
                location_name = self.ind_location2index_dict[int(csv_path[0].split(split_symbol)[-1][:2])]
            elif self.scenario == 'interaction':
                location_name = None
                for location_name in scenarioToShortcut_dict.keys():
                    if location_name in csv_path[0]:
                        break
            elif self.scenario == 'highd':
                location_name = csv_path[0].split('/')[-1][:2]
            elif self.scenario == 'ngsim':
                location_name = csv_path[0].split('/')[-1][:-4]
            else:
                location_name = csv_path.split('/')[-2]
            
            data = self.process_origin_csv(csv_path,
                                            unified_column=["frame", "ped", "x", "y", 'agent_type',
                                                            'psi_degree']).to_numpy()    # TIMESTAMP == frame_id
            print(f'Dataloader sgan_dada preprocessing {self.scenario}/{self.args.target}: {dataset_index}/{len(file_list)}')
            
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0],:])
            num_sequences = int(math.ceil((len(frames) - self.full_len + 1) / self.skip))
            
            # 对每帧
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                # curr_seq_data is a 20 length sequence
                curr_seq_data = np.concatenate(
                    frame_data[idx: idx + self.full_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.full_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.full_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.full_len))
                location = [0]*len(peds_in_curr_seq)
                agent_type = np.zeros(len(peds_in_curr_seq))
                manu = np.zeros(len(peds_in_curr_seq))

                num_peds_considered = 0
                _non_linear_ped = []
                
                # 对每帧的每个行人
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id]
                    if self.scenario != 'ngsim':
                        curr_agent_type_seq = np.unique(curr_ped_seq[:,[4]])
                        assert np.unique(curr_agent_type_seq).shape[0] == 1
                    
                    curr_ped_seq = curr_ped_seq[:,:4]  # 这个行人出现的全部数据, [frame_id, agent_id, x, y]

                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx      # 这个行人出现的第一帧在frames的位置
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1   # 这个行人出现的最后一帧在frames的位置
                    if pad_end - pad_front != self.full_len or curr_ped_seq.shape[0] != self.full_len:
                        continue
                    
                    if self.scenario != 'ngsim':
                        curr_menu_seq = self.get_manu(curr_seq_data[curr_seq_data[:, 1] == ped_id][:,[2,3,5]])

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    # _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                    location[_idx] = location_name
                    if self.scenario != 'ngsim':
                        agent_type[_idx] = curr_agent_type_seq
                        manu[_idx] = curr_menu_seq

                assert len(location) == agent_type.shape[0] == manu.shape[0]

                if num_peds_considered > min_ped:
                    # non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)  # [n1, n2, n3,...]
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    location_list.append(location)
                    if self.scenario != 'ngsim':
                        agent_type_list.append(agent_type)
                        manu_list.append(manu)

        if self.scenario != 'ngsim':
            assert len(agent_type_list) == len(location_list) == len(seq_list) \
                == len(seq_list_rel) == len(loss_mask_list) == len(manu_list)
            location_list = np.concatenate(location_list)
            agent_type_list = np.concatenate(agent_type_list)
            manu_list = np.concatenate(manu_list)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        # non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = t.from_numpy(seq_list[:, :, :self.previous_len]).type(
            t.float
        )
        self.pred_traj = t.from_numpy(seq_list[:, :, self.previous_len:]).type(
            t.float
        )
        self.whole_traj = t.from_numpy(seq_list[:, :, :]).type(
            t.float
        )
        self.obs_traj_rel = t.from_numpy(seq_list_rel[:, :, :self.previous_len]).type(
            t.float
        )
        self.pred_traj_rel = t.from_numpy(seq_list_rel[:, :, self.previous_len:]).type(
            t.float
        )
        self.whole_traj_rel = t.from_numpy(seq_list_rel[:, :, :]).type(
            t.float
        )
        self.loss_mask = t.from_numpy(loss_mask_list).type(t.float)
        # self.non_linear_ped = t.from_numpy(non_linear_ped).type(t.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        return_feature_list = [self.num_seq,
                               self.seq_start_end,
                               self.obs_traj,
                               self.pred_traj,
                               self.whole_traj,
                               self.obs_traj_rel,
                               self.pred_traj_rel,
                               self.whole_traj_rel,
                               # self.non_linear_ped,
                               self.loss_mask,
                               location_list,
                               agent_type_list,
                               manu_list]

        return return_feature_list

    def get_frame_instance_dict(self,record_data):
        '''
        Read raw data from files and return a dictionary:
            {frame_id:
                {object_id:
                    # 10 features
                    [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading]
                }
            }
        '''
        # print(train_file_path)
        content = record_data.float()
        now_dict = {}
        for row in content:
            # instance = {row[1]:row[2:]}
            n_dict = now_dict.get(row[0], {})
            n_dict[row[1].item()] = row  # [2:]
            # n_dict.append(instance)
            # now_dict[]
            now_dict[row[0].item()] = n_dict

        return now_dict


    def process_data(self, pra_now_dict, pra_start_ind,
                     pra_end_ind, pra_observed_last):
        val_list = []   # 在时间窗口中出现过的track_id
        for x in range(pra_start_ind, pra_end_ind, self.args.downsample_rate):
            if x in pra_now_dict:
                for val in pra_now_dict[x].keys():
                    val_list.append(val)
            else:
                return None, False
        if pra_observed_last not in pra_now_dict:
            return None, False

        visible_object_id_list = list(
            pra_now_dict[pra_observed_last].keys())  # object_id appears at the last observed frame
        num_visible_object = len(visible_object_id_list)  # number of current observed objects

        # compute the mean values of x and y for zero-centralization.
        visible_object_value = t.stack(list(pra_now_dict[pra_observed_last].values()))
        mean_xy = t.zeros(visible_object_value[0].shape, dtype=float)
        # xy = visible_object_value[:, 3:5].float()
        # m_xy = t.mean(xy, dim=0)
        # mean_xy[3:5] = m_xy

        now_all_object_id = set(val_list)   # 考虑所有时间步下的track_id
        non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))  # 在当下时间步看不到的id
        num_non_visible_object = len(non_visible_object_id_list)

        # for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
        object_feature_list = t.zeros((0,num_non_visible_object + num_visible_object,
                                       total_feature_dimension))

        # 针对每帧
        for frame_ind in range(pra_start_ind, pra_end_ind+1, self.args.downsample_rate):
            if frame_ind in pra_now_dict:
                # we add mark "1" to the end of each row to indicate that this row exists,
                # using list(pra_now_dict[frame_ind][obj_id])+[1]
                # -mean_xy is used to zero_centralize data
                # now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1]
                # for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
                now_frame_feature_dict = {obj_id: t.tensor((pra_now_dict[frame_ind][obj_id] - mean_xy).tolist() + [1])
                                                   if obj_id in visible_object_id_list
                                                   else t.tensor((pra_now_dict[frame_ind][obj_id] - mean_xy).tolist() + [0])
                                          for obj_id in pra_now_dict[frame_ind]}
            # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
                now_frame_feature = t.stack([now_frame_feature_dict.get(vis_id,
                                                                        t.zeros(total_feature_dimension))
                                              for vis_id in visible_object_id_list + non_visible_object_id_list])
            else:
                now_frame_feature = t.zeros((num_non_visible_object + num_visible_object,total_feature_dimension))

            # assert (t.logical_and((now_frame_feature[:,3]!=0),(now_frame_feature[:,4]==0)).sum()==0).sum().item()
            # assert (t.logical_and((now_frame_feature[:,3]==0),(now_frame_feature[:,4]!=0)).sum()==0).sum().item()

            object_feature_list = t.cat([object_feature_list,
                                         now_frame_feature[None]], dim=0)

        # object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
        object_frame_feature = t.zeros((num_visible_object + num_non_visible_object,
                                        (pra_end_ind - pra_start_ind) // self.args.downsample_rate + 1,
                                        total_feature_dimension))

        # np.transpose(object_feature_list, (1,0,2))
        object_frame_feature[:num_visible_object +
                              num_non_visible_object] = object_feature_list.permute(1, 0, 2)      # [max_agent_num,T,特征数]

        # assert t.logical_and((object_frame_feature[...,3]==0),(object_frame_feature[...,4]!=0)).sum().item()==0
        # assert t.logical_and((object_frame_feature[...,4]==0),(object_frame_feature[...,3]!=0)).sum().item()==0 
        return object_frame_feature[...,[3,4,5]], True


    def resort_frameId_and_trackId(self, item_full_data_df):
        '''
        item_full_data_df:这个是在原始dataset中用来枚举的对象,在这里我们只需要根据agent_id和record对track_id进行修改即可
        '''

        item_full_data_df['track_id'] = item_full_data_df['track_id'].astype(np.int64)
        full_data = pd.DataFrame(columns=item_full_data_df.columns)
        # track_id不管是否连续,只需要管frame_id即可
        last_frame, last_track = 0, 0
        for s in self.scenario:
            for record in range(1, item_full_data_df.loc[item_full_data_df['scenario'] == s,'record'].max()+1):

                record_veh_data = item_full_data_df[(item_full_data_df['scenario'] == s) &
                                                    (item_full_data_df['record'] == record) &
                                                    (item_full_data_df['agent_type'] == 0)]
                record_ped_data = item_full_data_df[(item_full_data_df['scenario'] == s) &
                                                    (item_full_data_df['record'] == record) &
                                                    (item_full_data_df['agent_type'] == 1)]
                record_ped_data['track_id'] += (record_veh_data['track_id'].max() + last_track)
                per_record_data = pd.concat([record_veh_data, record_ped_data], axis=0)

                per_record_data['frame_id'] += last_frame
                assert per_record_data.shape[0] == item_full_data_df[(item_full_data_df['scenario'] == s) &
                                                    (item_full_data_df['record'] == record)].shape[0]

                full_data = pd.concat([full_data, per_record_data], axis=0)
                last_track = full_data['track_id'].max()
                last_frame = full_data['frame_id'].max()

        # 保证输出输出相同
        assert item_full_data_df.shape[0] == full_data.shape[0]
        return full_data



# if __name__ == '__main__':
    # parser = getParser('DR_DEU_Roundabout_OF', start_with='', Time='2025-03-01-15-15-01')
    # # multimodal trajectory
    # parser.add_argument('--random_query_nums', type=int, default=1,
    #                     help='写1,不要写0，1和0都是表示只输出单条轨迹')
    # parser.add_argument('--Model_name_list', type=list, default=['transformer'],
    #                     help="'lstm','transformer','rnn','gru'/'")
    # args = parser.parse_args()
    # Grip_dataset = Dataloader_SGAN_DADA(args, mode='train')
    # dataset = DataLoader(Grip_dataset,
    #                      shuffle=True, batch_size=args.batch_size, drop_last=True,
    #
    #                      num_workers=args.num_workers)
    # rescale_xy = torch.ones((1, 2, 1, 1)).to(self.args.device)
    # rescale_xy[:, 0] = 1.
    # rescale_xy[:, 1] = 1.
    # for (ori_data, A, _) in dataset:
    #     data, no_norm_loc_data, object_type = preprocess_data(ori_data, rescale_xy)
    #     pass












