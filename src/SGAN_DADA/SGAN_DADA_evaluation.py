import os, pickle

from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from Dataloader.Dataloader_SGAN_DADA import Dataloader_SGAN_DADA as Dataloader
from Dataloader.Dataloader_SGAN_DADA import seq_collate
from tool import *
from .SGAN_DADA_train import batch_size, epochs
from .utils import bool_flag, relative_to_abs
from ours.reliability_estimator import reliability_estimator_SGAN_DADA

def main(args, model, isSource):
    model.eval()
    
    target_dset = Dataloader(args, mode='train', isSource=isSource,
                             is_DA=False)
    target_loader = DataLoader(
        target_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=model.DADA_config.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)

    len_target = len(target_loader)
    len_max = len_target

    if not os.path.exists(model.model_eval_model_path):
        eval_model = reliability_estimator_SGAN_DADA(args, model, isSource)
        
        with open(model.model_eval_model_path, 'wb') as f:
            pickle.dump(eval_model, f)

    else:
        with open(model.model_eval_model_path, 'rb') as f:
            eval_model = pickle.load(f)

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

            target_iter = iter(target_loader)
            index_batch = 0
            write_index = 0
            index = 0
            query_index = 0

            for index_batch in tqdm(range(len_max)):
                if index_batch % len_target == 0:
                    del target_iter
                    target_iter = iter(target_loader)

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
                
                # query
                weight_ratio, batch_index = eval_model.query(diff(obs_traj.transpose(1,0)),
                                 seq_start_end)
                mde_to_write = mde
                fde_to_write = fde
                
                model.write_input_loss(epoch,
                                        mde_to_write[:,model.eval_length-1],
                                        weight_ratio, 
                                        batch_index.squeeze(1),
                                        0, 
                                        'nan',
                                        'nan', 'nan',
                                        eval_model.eval_future_length,
                                        )
                    
                model.print_eval_logs(epoch, weight_ratio.mean(),
                                        0,
                                        'nan')
                query_index += len(seq_start_end)

                
                    
                    