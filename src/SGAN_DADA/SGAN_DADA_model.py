import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import itertools
import numpy as np

from .modules import *
from Model_root import Model_root
from .losses import gan_g_loss, gan_d_loss, l2_loss
from .utils import bool_flag, relative_to_abs


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(
        self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
        self, args, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.args = args
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        # 直接norm
        obs_traj = obs_traj - obs_traj.mean(0,keepdim=True)

        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)  # Hidden state

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel

    def get_hidden_state(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)  # Hidden state

        return decoder_h


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, args, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()
        
        self.args = args
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """

        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores


class Discriminator_inlay(nn.Module):
    def __init__(self,
                 input_dim=32,
                 mlp_dim=64,
                 ):
        super(Discriminator_inlay, self).__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self,
                feature_in,  # fake or real[nums, feat_len]
                ):
        feature_out = self.mlp_layers(feature_in)  # [nums, 1]
        return feature_out  # fake_score, [nums, 1]


class dada_config:
    def __init__(self):
        self.dataset_name='A2B'
        self.delim='\t'
        self.loader_num_workers=4
        self.skip=1

        # Optimization
        self.batch_size=16
        self.num_iterations=20000
        self.num_epochs=500

        # Model Options
        self.embedding_dim=16
        self.num_layers=1
        self.dropout=0
        self.batch_norm=0
        self.mlp_dim=64

        # Generator Options
        self.encoder_h_dim_g=32
        self.decoder_h_dim_g=32
        self.noise_dim=(8,)
        self.noise_type='gaussian'
        self.noise_mix_type='global'
        self.clipping_threshold_g=1.5
        self.g_learning_rate=1e-3
        self.g_steps=1

        # Pooling Options
        self.pooling_type='pool_net'
        self.pool_every_timestep=0

        # Pool Net Option
        self.bottleneck_dim=32

        # Social Pooling Options
        self.neighborhood_size=2.0
        self.grid_size=8

        # Discriminator Options
        self.d_type='local'
        self.encoder_h_dim_d=64
        self.d_learning_rate=1e-3
        self.d_steps=2
        self.clipping_threshold_d=0

        # Loss Options
        self.l2_loss_weight=1
        self.best_k=1

        # Output
        self.output_dir=os.getcwd()  # need to modify
        self.print_every=50
        self.checkpoint_every=10
        self.checkpoint_name='checkpoint'
        self.checkpoint_start_from=None
        self.restore_from_checkpoint=0
        self.num_samples_check=5000

        # Misc
        self.use_gpu=1
        self.timing=0
        self.gpu_num="0"

class SGAN_DADA(Model_root):
    def __init__(self, args):
        super(SGAN_DADA, self).__init__(args, 'SGAN_DADA', True)
        
        DADA_config = dada_config()
        self.DADA_config = DADA_config
        long_dtype, float_dtype = get_dtypes(self.DADA_config)

        self.generator = TrajectoryGenerator(
                            args,
                            obs_len=self.previous_length,
                            pred_len=self.future_length,
                            embedding_dim=self.DADA_config.embedding_dim,
                            encoder_h_dim=self.DADA_config.encoder_h_dim_g,
                            decoder_h_dim=self.DADA_config.decoder_h_dim_g,
                            mlp_dim=self.DADA_config.mlp_dim,
                            num_layers=self.DADA_config.num_layers,
                            noise_dim=self.DADA_config.noise_dim,
                            noise_type=self.DADA_config.noise_type,
                            noise_mix_type=self.DADA_config.noise_mix_type,
                            pooling_type=self.DADA_config.pooling_type,
                            pool_every_timestep=self.DADA_config.pool_every_timestep,
                            dropout=self.DADA_config.dropout,
                            bottleneck_dim=self.DADA_config.bottleneck_dim,
                            neighborhood_size=self.DADA_config.neighborhood_size,
                            grid_size=self.DADA_config.grid_size,
                            batch_norm=self.DADA_config.batch_norm)

        self.generator.apply(init_weights)
        self.generator.type(float_dtype).train()
        
        self.discriminator = TrajectoryDiscriminator(
            args,
            obs_len=self.previous_length,
            pred_len=self.future_length,
            embedding_dim=self.DADA_config.embedding_dim,
            h_dim=self.DADA_config.encoder_h_dim_d,
            mlp_dim=self.DADA_config.mlp_dim,
            num_layers=self.DADA_config.num_layers,
            dropout=self.DADA_config.dropout,
            batch_norm=self.DADA_config.batch_norm,
            d_type=self.DADA_config.d_type).cuda()

        self.discriminator.apply(init_weights)
        self.discriminator.type(float_dtype).train()

        self.d_inlay = Discriminator_inlay(self.DADA_config.decoder_h_dim_g, 2*self.DADA_config.decoder_h_dim_g).cuda()

        self.g_loss_fn = gan_g_loss
        self.d_loss_fn = gan_d_loss

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.DADA_config.g_learning_rate)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.DADA_config.d_learning_rate)
        self.optimizer_d_inlay = optim.Adam(self.d_inlay.parameters(), lr=self.DADA_config.g_learning_rate)

    def discriminator_step(self, batch):
        # batch = [tensor.cuda() for tensor in batch]
        # (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
        # loss_mask, seq_start_end) = batch

        obs_traj, pred_traj_gt, whole_traj_s,\
             obs_traj_rel, pred_traj_gt_rel, whole_traj_rel_s,\
              loss_mask, seq_start_end, location_list_s,\
                agent_type_list_s, manu_list_s = batch
        
        obs_traj = obs_traj.to(self.args.device)
        pred_traj_gt = pred_traj_gt.to(self.args.device)
        obs_traj_rel = obs_traj_rel.to(self.args.device)
        pred_traj_gt_rel = pred_traj_gt_rel.to(self.args.device)
        seq_start_end = seq_start_end.to(self.args.device)

        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt)

        generator_out = self.generator(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

        scores_fake = self.discriminator(traj_fake, traj_fake_rel, seq_start_end)
        scores_real = self.discriminator(traj_real, traj_real_rel, seq_start_end)

        # Compute loss with optional gradient penalty
        data_loss = self.d_loss_fn(scores_real, scores_fake)
        losses['D_data_loss'] = data_loss.item()
        loss += data_loss
        losses['D_total_loss'] = loss.item()

        self.optimizer_d.zero_grad()
        loss.backward()
        if self.DADA_config.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                    self.DADA_config.clipping_threshold_d)
        self.optimizer_d.step()

        # MDE, FDE
        with torch.no_grad():
            pred_traj_fake_rel = self.generator(
                        obs_traj, obs_traj_rel, seq_start_end
                    )
            pred_traj_fake = relative_to_abs(
                        pred_traj_fake_rel, obs_traj[-1]
                    )
            mde = self.displacement_error(
                        pred_traj_fake, pred_traj_gt, mode='raw'
                    )
            fde = self.final_displacement_error(
                        pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                    )
        
        return losses, mde, fde
    
    def write_input_loss(self, epoch, mde, weight_ratio, non_zero_global_index,
                         dataset_index, location, 
                         agent_type, manu, eval_future_length):
        '''
        write_input_loss 的 Docstring
        
        :param self: 说明

        ['Model', 'epoch', 'batch_index', 'MDE', 'weight_ratio',
        'location', 'agent_type', 'manu', 'eval_future_length']

        epoch, dataset_index, eval_future_length: figure
        mde: torch.tensor, [agent_num]
        weight_ratio: torch.tensor, [1]
        non_zero_global_index: torch.tensor, [agent_num]
        location, agent_type, manu: str

        '''
        # 如果是空, 直接返回
        if non_zero_global_index.shape[0] == 0:
            return

        if dataset_index == 0:
            path = self.input_loss_source_save_path

        else:
            path = self.input_loss_target_save_path

        agent_num = len(non_zero_global_index)
        data = np.concatenate([np.array([self.model_name]*agent_num)[:,None],\
                               np.array([epoch]*agent_num)[:,None],\
                        non_zero_global_index[:,None],\
                        mde[:,None].cpu().numpy(),\
                        weight_ratio[:,None],\
                        np.array([location]*agent_num)[:,None],\
                        np.array([agent_type]*agent_num)[:,None],\
                        np.array([manu]*agent_num)[:,None],\
                        np.array([eval_future_length]*agent_num)[:,None]],
                        axis=1)
        
        with open(path, 'a+', newline='') as write_obj:
            np.savetxt(write_obj, data, fmt='%s', 
                       delimiter=',', comments='')
            write_obj.close()

    def final_displacement_error(self, pred_pos, pred_pos_gt, 
                                 consider_ped=None, mode='sum'):
        """
        Input:
        - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
        - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
        last pos
        - consider_ped: Tensor of shape (batch)
        Output:
        - loss: gives the eculidian displacement error
        """
        loss = pred_pos_gt - pred_pos
        loss = loss**2
        if consider_ped is not None:
            loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
        else:
            loss = torch.sqrt(loss.sum(dim=1))
        if mode == 'raw':
            return loss
        else:
            return torch.sum(loss)


    def displacement_error(self, pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
        """
        Input:
        - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
        - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
        predictions.
        - consider_ped: Tensor of shape (batch)
        - mode: Can be one of sum, raw
        Output:
        - loss: gives the eculidian displacement error
        """
        seq_len, _, _ = pred_traj.size()
        loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
        # loss = loss**2
        if consider_ped is not None:
            loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
        else:
            # loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
            loss = torch.norm(loss, dim=2).nanmean()
            
        if mode == 'sum':
            return torch.sum(loss)
        elif mode == 'raw':
            return loss

    def generator_step(self, batch_s, batch_t):
        # batch_s = [tensor.cuda() for tensor in batch_s]
        
        # (obs_traj_s, pred_traj_gt_s, obs_traj_rel_s, pred_traj_gt_rel_s, non_linear_ped_s,
        # loss_mask_s, seq_start_end_s) = batch_s

        obs_traj_s, pred_traj_gt_s, whole_traj_s,\
             obs_traj_rel_s, pred_traj_gt_rel_s, whole_traj_rel_s,\
              loss_mask_s, seq_start_end_s, location_list_s,\
                agent_type_list_s, manu_list_s = batch_s
        
        obs_traj_s = obs_traj_s.to(self.args.device)
        pred_traj_gt_s = pred_traj_gt_s.to(self.args.device)
        obs_traj_rel_s = obs_traj_rel_s.to(self.args.device)
        pred_traj_gt_rel_s = pred_traj_gt_rel_s.to(self.args.device)
        loss_mask_s = loss_mask_s.to(self.args.device)
        seq_start_end_s = seq_start_end_s.to(self.args.device)

        # batch_t = [tensor.cuda() for tensor in batch_t]
        # (obs_traj_t, pred_traj_gt_t, obs_traj_rel_t, pred_traj_gt_rel_t, non_linear_ped_t,
        # loss_mask_t, seq_start_end_t) = batch_t

        obs_traj_t, pred_traj_gt_t, whole_traj_t,\
             obs_traj_rel_t, pred_traj_gt_rel_t, whole_traj_rel_t,\
              loss_mask_t, seq_start_end_t, location_list_t,\
                agent_type_list_t, manu_list_t = batch_t

        obs_traj_t = obs_traj_t.to(self.args.device)
        pred_traj_gt_t = pred_traj_gt_t.to(self.args.device)
        obs_traj_rel_t = obs_traj_rel_t.to(self.args.device)
        pred_traj_gt_rel_t = pred_traj_gt_rel_t.to(self.args.device)
        loss_mask_t = loss_mask_t.to(self.args.device)
        seq_start_end_t = seq_start_end_t.to(self.args.device)
        
        # 1. Predictor
        losses = {}
        loss = torch.zeros(1).to(pred_traj_gt_s)
        g_l2_loss_rel = []

        loss_mask_s = loss_mask_s[:, self.previous_length:]

        for _ in range(self.DADA_config.best_k):
            generator_out = self.generator(obs_traj_s, obs_traj_rel_s, seq_start_end_s)

            pred_traj_fake_rel = generator_out
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj_s[-1])

            if self.DADA_config.l2_loss_weight > 0:
                g_l2_loss_rel.append(self.DADA_config.l2_loss_weight * l2_loss(
                    pred_traj_fake_rel,
                    pred_traj_gt_rel_s,
                    loss_mask_s,
                    mode='raw'))

        g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt_s)
        if self.DADA_config.l2_loss_weight > 0:
            g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
            for start, end in seq_start_end_s.data:
                _g_l2_loss_rel = g_l2_loss_rel[start:end]
                _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
                _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                    loss_mask_s[start:end])
                g_l2_loss_sum_rel += _g_l2_loss_rel
            losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
            loss += g_l2_loss_sum_rel

        traj_fake = torch.cat([obs_traj_s, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel_s, pred_traj_fake_rel], dim=0)

        scores_fake = self.discriminator(traj_fake, traj_fake_rel, seq_start_end_s)
        discriminator_loss = self.g_loss_fn(scores_fake)

        loss += discriminator_loss
        losses['G_discriminator_loss'] = discriminator_loss.item()
        losses['G_total_loss'] = loss.item()

        self.optimizer_g.zero_grad()
        loss.backward()
        if self.DADA_config.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.DADA_config.clipping_threshold_g
            )
        self.optimizer_g.step()

        # 2.Discriminator_inlay
        hidden_s = self.generator.get_hidden_state(obs_traj_s, obs_traj_rel_s, seq_start_end_s)
        score_fake = self.d_inlay(hidden_s)
        hidden_t = self.generator.get_hidden_state(obs_traj_t, obs_traj_rel_t, seq_start_end_t)
        score_real = self.d_inlay(hidden_t)
        d_inlay_loss = gan_d_loss(score_real, score_fake) * 10.
        self.optimizer_d_inlay.zero_grad()
        d_inlay_loss.backward()
        self.optimizer_d_inlay.step()

        # 3. Generator_inlay
        hidden_s = self.generator.get_hidden_state(obs_traj_s, obs_traj_rel_s, seq_start_end_s)
        score_fake = self.d_inlay(hidden_s)
        g_inlay_loss = gan_g_loss(score_fake) * 10.
        self.optimizer_g.zero_grad()
        g_inlay_loss.backward()
        if self.DADA_config.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.DADA_config.clipping_threshold_g
            )
        self.optimizer_g.step()

        return losses
        

    def test_step(self, obs_traj_t, pred_traj_gt_t, whole_traj_t,
             obs_traj_rel_t, pred_traj_gt_rel_t, whole_traj_rel_t,
              loss_mask_t, seq_start_end_t, location_list_t,
                agent_type_list_t, manu_list_t,):
        traj_rel_fake = self.generator_s2t(obs_traj_t, whole_traj_t,
                                        seq_start_end_t)   # G_S2T(S)=T', [20, nums, 2]
        traj_rel_fake = self.generator_s2t(whole_traj_t, whole_traj_rel_t,
                                        seq_start_end_t)
        
        traj_fake = relative_to_abs(traj_rel_fake,
                                    whole_traj_t[0]) 
        self.get_MDE(reconst_traj_rel, whole_traj_t,
                                    loss_mask_t)

        pass

    def train_step(self, obs_traj_s, pred_traj_gt_s, whole_traj_s,
             obs_traj_rel_s, pred_traj_gt_rel_s, whole_traj_rel_s,
              loss_mask_s, seq_start_end_s, location_list_s,
                agent_type_list_s, manu_list_s, 
             
             obs_traj_t, pred_traj_gt_t, whole_traj_t,
             obs_traj_rel_t, pred_traj_gt_rel_t, whole_traj_rel_t,
              loss_mask_t, seq_start_end_t, location_list_t,
                agent_type_list_t, manu_list_t,
             ):
        
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        
        # Train D_S, whole_traj_t + obs_traj_s
        traj_rel_fake = self.generator_t2s(whole_traj_t, whole_traj_rel_t,
                                        seq_start_end_t)  # G_T2S(T)=S', [20, nums, 2]
        traj_fake = relative_to_abs(traj_rel_fake,
                                    whole_traj_t[0])  # S'_abs, [20, nums, 2], X_TS
        scores_real = self.discriminator_s(obs_traj_s, obs_traj_rel_s,
                                        seq_start_end_s)  # D_S(S), [nums, 8]
        scores_fake = self.discriminator_s(traj_fake, traj_rel_fake,
                                        seq_start_end_t)  # D_S(S'), [nums, 20]
        d_s_loss_real, d_s_loss_fake = self.self.d_loss_fn(scores_real, scores_fake,
                                                     mode=self.DADA_config.gan_loss_mode)
        d_s_loss = d_s_loss_real + d_s_loss_fake
        d_s_loss.backward()
        # print('* D_S process has done.')

        # Train D_T, whole_traj_s + obs_traj_t
        traj_rel_fake = self.generator_s2t(whole_traj_s, whole_traj_rel_s,
                                        seq_start_end_s)   # G_S2T(S)=T', [20, nums, 2]
        traj_fake = relative_to_abs(traj_rel_fake,
                                    whole_traj_s[0])  # T'_abs, [20, nums, 2]
        scores_real = self.discriminator_t(obs_traj_t, obs_traj_rel_t,
                                        seq_start_end_t)  # D_T(T), [nums, 8]
        scores_fake = self.discriminator_t(traj_fake, traj_rel_fake,
                                        seq_start_end_s)  # D_T(T'), [nums, 20]
        
        d_t_loss_real, d_t_loss_fake = self.self.d_loss_fn(scores_real, scores_fake,
                                                    mode=self.DADA_config.gan_loss_mode)
        d_t_loss = d_t_loss_real + d_t_loss_fake
        d_t_loss.backward()
        # print('* D_T process has done.')

        # d_real_loss = d_s_loss_real + d_t_loss_real
        # d_fake_loss = d_s_loss_fake + d_t_loss_fake
        # d_loss = d_s_loss + d_t_loss

        if self.DADA_config.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(self.discriminator_s.parameters(), 
                                     self.DADA_config.clipping_threshold_d)
            nn.utils.clip_grad_norm_(self.discriminator_t.parameters(), 
                                     self.DADA_config.clipping_threshold_d)
        self.d_optimizer.step()
        # print(f'* D has been trained. loss:{d_loss.item()}')

        #============ train G ============#
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

        if self.DADA_config.lambda_idt_loss > 0 and self.DADA_config.l2_loss_weight > 0:
            # G_S2T(T) should be identity if T is fed: ||G_S2T(T) - T||, whole_traj_s + obs_traj_t
            idt_G_S2T = self.generator_s2t(obs_traj_t, obs_traj_rel_t,
                                        seq_start_end_t)  # [8, nums, 2]
            g_s2t_idt_loss = self.DADA_config.l2_loss_weight * l2_loss(idt_G_S2T, obs_traj_rel_t,
                                                            loss_mask_t[:, :self.previous_length],
                                                            mode='average')
            g_s2t_idt_loss *= self.DADA_config.lambda_idt_loss

            # G_T2S(S) should be identity if S is fed: ||G_T2S(S) - S||, whole_traj_t + obs_traj_s
            idt_G_T2S = self.generator_t2s(obs_traj_s, obs_traj_rel_s,
                                        seq_start_end_s)  # [8, nums, 2]
            g_t2s_idt_loss = self.DADA_config.l2_loss_weight * l2_loss(idt_G_T2S, obs_traj_rel_s,
                                                            loss_mask_s[:, :self.previous_length],
                                                            mode='average')
            g_t2s_idt_loss *= self.DADA_config.lambda_idt_loss
        else:
            g_s2t_idt_loss = 0
            g_t2s_idt_loss = 0

        # Train S-T-S cycle, whole_traj_s + obs_traj_t
        # ||G_S2T(S) - T||
        traj_rel_fake = self.generator_s2t(whole_traj_s, whole_traj_rel_s,
                                        seq_start_end_s)  # G_S2T(S)=T', [20, nums, 2]
        traj_fake = relative_to_abs(traj_rel_fake,
                                    whole_traj_s[0])  # T'_abs, [20, nums, 2]
        scores_fake = self.discriminator_t(traj_fake, traj_rel_fake,
                                        seq_start_end_s)  # D_T(T'), [nums, 20]
        g_1_loss = self.self.g_loss_fn(scores_fake,
                                mode=self.DADA_config.gan_loss_mode)
        
        # ||G_T2S(T') - S||
        reconst_traj_rel = self.generator_t2s(traj_fake, traj_rel_fake,
                                            seq_start_end_s)  # G_T2S(T')=S', [20, nums, 2]
        if self.DADA_config.l2_loss_weight > 0:
            g_rec1_loss = self.DADA_config.l2_loss_weight * l2_loss(reconst_traj_rel, whole_traj_rel_s,
                                                        loss_mask_s[:, :],
                                                        mode='average')
        
        MDE_s, FDE_s = self.get_MDE(reconst_traj_rel, whole_traj_s,
                                    loss_mask_s)
        # print(f'* G_1 process has done. loss:{g_rec1_loss.item()}')

        # Train T-S-T cycle, whole_traj_t + obs_traj_s
        # ||G_T2S(T) - S||
        traj_rel_fake = self.generator_t2s(whole_traj_t, whole_traj_rel_t,
                                        seq_start_end_t)  # G_T2S(T)=S', [20, nums, 2]
        traj_fake = relative_to_abs(traj_rel_fake,
                                    whole_traj_t[0])  # S'_abs, [20, nums, 2]
        scores_fake = self.discriminator_s(traj_fake, traj_rel_fake,
                                        seq_start_end_t)  # D_S(S'), [nums, 20]
        g_2_loss = self.self.g_loss_fn(scores_fake,
                                mode=self.DADA_config.gan_loss_mode)
        # ||G_S2T(S') - T||
        reconst_traj_rel = self.generator_s2t(traj_fake, traj_rel_fake,
                                            seq_start_end_t)  # G_S2T(S')=T', [20, nums, 2]
        if self.DADA_config.l2_loss_weight > 0:
            g_rec2_loss = self.DADA_config.l2_loss_weight * l2_loss(reconst_traj_rel, whole_traj_rel_t,
                                                        loss_mask_t[:, :],
                                                        mode='average')
        # print(f'* G_2 process has done. loss:{g_rec1_loss.item()}')

        g_gen_loss = g_1_loss + g_2_loss
        g_rec_loss = g_rec1_loss + g_rec2_loss
        g_idt_loss = g_s2t_idt_loss + g_t2s_idt_loss
        g_loss = g_gen_loss + g_rec_loss + g_idt_loss
        g_loss.backward()
        if self.DADA_config.clipping_threshold_g > 0:
            nn.utils.clip_grad_norm_(self.generator_s2t.parameters(), \
                                     self.DADA_config.clipping_threshold_g)
            nn.utils.clip_grad_norm_(self.generator_t2s.parameters(), \
                                     self.DADA_config.clipping_threshold_g)
        self.g_optimizer.step()
        # print('* G has been trained.')
        MDE_t, FDE_t = self.get_MDE(reconst_traj_rel, whole_traj_t,
                                    loss_mask_t)
        
        return g_loss+d_t_loss+d_s_loss, MDE_s, FDE_s, MDE_t, FDE_t

    def get_MDE(self, reconst_traj_rel, whole_traj,
                loss_mask):
        # MDE, FDE
        last_obs = whole_traj[[self.previous_length-1]]
        predicted_traj = (reconst_traj_rel[self.previous_length:].cumsum(0) + \
                            last_obs).transpose(0,1) # [agent_num, future_T, 2]
        gt = whole_traj[self.previous_length:].transpose(0,1)  # [agent_num, future_T, 2]
        loss_mask_nan = loss_mask.clone()[None,:,self.previous_length:].transpose(1,0)
        
        diag_mask = torch.tril(torch.ones(self.future_length, self.future_length), 0)[None,...,None].to(
            self.args.device)  # [1,future_len,future_len,1]
        MDE_mask = loss_mask_nan[:,...,None] * diag_mask
        MDE_mask[MDE_mask==0] = torch.nan
        MDE_batch_loss = torch.norm((gt - predicted_traj)[:,None]*MDE_mask,dim=-1).nanmean(0).nanmean(1)

        eye_mask = torch.eye(self.future_length)[None,...,None].to(self.args.device)
        FDE_mask = loss_mask_nan[:,...,None] * eye_mask
        FDE_mask[FDE_mask==0] = torch.nan

        FDE_batch_loss = torch.norm((gt - predicted_traj)[:,None]*FDE_mask,dim=-1).nanmean(0).nanmean(1)
        
        return MDE_batch_loss, FDE_batch_loss

    def forward(self):
        pass
