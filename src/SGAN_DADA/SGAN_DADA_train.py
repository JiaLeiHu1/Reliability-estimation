import os

from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from Dataloader.Dataloader_SGAN_DADA import Dataloader_SGAN_DADA as Dataloader
from Dataloader.Dataloader_SGAN_DADA import seq_collate
from tool import *
batch_size = 128
epochs = 80

def main(args, model):
    # 数据需要体现的是预测的数据、行人还是车、以及对应的场景
    model.train()
    source_dset = Dataloader(args, mode='train', isSource=True)
    source_loader = DataLoader(
        source_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=model.DADA_config.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    
    target_dset = Dataloader(args, mode='train', isSource=False,
                             is_DA=True)
    target_loader = DataLoader(
        target_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=model.DADA_config.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    
    iterations_per_epoch = max(len(source_dset), \
                               len(target_dset)) / batch_size

    timestep, epoch = 0, 0
    checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    
    ########## Train DLA ##########
    len_source, len_target = len(source_loader), len(target_loader)
    len_max = max(len_source, len_target)
    best_epoch_loss = t.inf
    train_epoch_loss = 0
    train_step = 0
    epoch = 0
    last_best_epoch = t.nan

    # while step < model.DADA_config.num_iterations:
    for epoch in tqdm(range(epochs)):

        # logger.info('Starting epoch {}:'.format(epoch))
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        index_batch = 0
        train_epoch_loss = 0

        d_steps_left = model.DADA_config.d_steps
        g_steps_left = model.DADA_config.g_steps
        write_index = 0

        for index_batch in tqdm(range(len_max)):
            if index_batch % len_source == 0:
                del source_iter
                source_iter = iter(source_loader)
                # logger.info('** Source iter is reset.')
            if index_batch % len_target == 0:
                del target_iter
                target_iter = iter(target_loader)
                # logger.info('** Target iter is reset.')

            batch_s, batch_t = next(source_iter), next(target_iter)

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d, mde, fde = model.discriminator_step(batch_s)
                d_steps_left -= 1
                train_epoch_loss += losses_d['D_total_loss']
                loss = losses_d['D_total_loss']
                mde = mde.mean().item()
                fde = fde.mean().item()
                print('discriminator_step')

            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = model.generator_step(batch_s, batch_t)
                mde, fde = t.nan, t.nan

                g_steps_left -= 1
                train_epoch_loss += losses_g['G_total_loss']
                loss = losses_g['G_total_loss']
                print('generator_step')

            print(f'last_best_epoch:{last_best_epoch}')
            model.print_training_logs(args, epoch,
                                      index_batch, loss,
                                      mde, fde)

            if epoch == 0:
                # 根据seq_start_end来写入
                obs_traj, pred_traj_gt, whole_traj_s,\
                    obs_traj_rel, pred_traj_gt_rel, whole_traj_rel_s,\
                    loss_mask, seq_start_end, location_list_s,\
                        agent_type_list_s, manu_list_s = batch_s
                
                for start, end in seq_start_end:
                    input_trajectory_to_write_bs = obs_traj[:,start:end].transpose(0,1) # [agent_num, input_T, 2]
                    for input_trajectory_to_write in input_trajectory_to_write_bs:
                        model.write_test_trajectory_input_logs(write_index,
                                                            diff(input_trajectory_to_write[None]),
                                                            True,
                                                            'train')
                    write_index += 1


            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            timestep += 1
            d_steps_left = model.DADA_config.d_steps
            g_steps_left = model.DADA_config.g_steps

        checkpoint['g_state'] = model.generator.state_dict()
        checkpoint['d_state'] = model.discriminator.state_dict()
            
        if best_epoch_loss >= train_epoch_loss:
            # 保存最优模型
            best_epoch_loss = train_epoch_loss
            last_best_epoch = epoch
            save_model(model=checkpoint,
                    model_path=model.best_model_save_path)

        # 保存每个epoch的模型
        if epoch % args.save_per_epoch == 0:
            save_model(model=checkpoint,
                    model_path=os.path.join(model.model_save_dir,
                                            f'epoch_{epoch}_{model.model_name}.pt'))
            