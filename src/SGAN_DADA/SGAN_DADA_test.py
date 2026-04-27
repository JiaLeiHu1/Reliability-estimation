import os

from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from Dataloader.Dataloader_SGAN_DADA import Dataloader_SGAN_DADA as Dataloader
from Dataloader.Dataloader_SGAN_DADA import seq_collate
from tool import *
from .SGAN_DADA_train import batch_size, epochs
from .utils import bool_flag, relative_to_abs

def main(args, model, isSource):
    # 数据需要体现的是预测的数据、行人还是车、以及对应的场景
    model.eval()
    
    target_dset = Dataloader(args, mode='test', isSource=isSource,
                             is_DA=False)
    target_loader = DataLoader(
        target_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=model.DADA_config.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    
    ########## Train DLA ##########
    len_target = len(target_loader)
    len_max = len_target

    # while step < model.DADA_config.num_iterations:
    with t.no_grad():
        for epoch in tqdm(range(epochs,-2,-1)):
            # load model
            try:
                model_dict = t.load(model.get_epoch_model_path(epoch))
                model.generator.load_state_dict(model_dict['g_state'])
                model.discriminator.load_state_dict(model_dict['d_state'])

            except:
                print(f'failed to find model at {model.get_epoch_model_path(epoch)}')
                continue

            # logger.info('Starting epoch {}:'.format(epoch))
            target_iter = iter(target_loader)
            index_batch = 0
            write_index = 0
            index = 0

            for index_batch in tqdm(range(len_max)):
                # logger.info('** Source iter is reset.')
                if index_batch % len_target == 0:
                    del target_iter
                    target_iter = iter(target_loader)
                    # logger.info('** Target iter is reset.')

                batch_t = next(target_iter)

                obs_traj, pred_traj_gt, whole_traj_s,\
                obs_traj_rel, pred_traj_gt_rel, whole_traj_rel_s,\
                loss_mask, seq_start_end, location_list_s,\
                    agent_type_list_s, manu_list_s = batch_t
                
                obs_traj = obs_traj.to(args.device)
                pred_traj_gt = pred_traj_gt.to(args.device)
                obs_traj_rel = obs_traj_rel.to(args.device)
                seq_start_end = seq_start_end.to(args.device)

                pred_traj_fake_rel = model.generator(
                        obs_traj, obs_traj_rel, seq_start_end
                    )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                mde = (t.norm(pred_traj_fake - pred_traj_gt,dim=2).cumsum(0)/\
                       t.arange(1,pred_traj_fake.shape[0]+1).to(args.device)[:,None]).transpose(0,1)    # [agent_num,future_T]
                fde = t.norm(pred_traj_fake - pred_traj_gt,dim=2).transpose(0,1)   # [agent_num,future_T]
                
                if epoch == epochs-1:
                    for start, end in seq_start_end:
                        input_trajectory_to_write_bs = obs_traj[:,start:end].transpose(0,1) # [agent_num, input_T, 2]
                        for input_trajectory_to_write in input_trajectory_to_write_bs:
                            model.write_test_trajectory_input_logs(write_index,
                                                                diff(input_trajectory_to_write[None]),
                                                                isSource,
                                                                'test')
                        write_index += 1
                
                for bs_agent_index in range(mde.shape[0]):
                    model.write_test_trajectory_output_logs(epoch, index_batch*batch_size+index,
                                                mde[bs_agent_index], fde[bs_agent_index],
                                                isSource=isSource
                                                )
                    index += 1

                model.print_testing_logs(args,epoch, 
                        index_batch, mde.nanmean(0)[-1].item(),
                        fde.nanmean(0)[-1].item(),
                        isSource
                        )
                    