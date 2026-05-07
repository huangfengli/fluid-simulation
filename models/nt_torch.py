import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d.ml.torch as ml3d

from models.ASCC import ContinuousConv as ASCC


class BoundaryAFF(nn.Module):
    def __init__(self,
                 channels=32,
                 inter_channels=None,
                 conv_type='cconv',
                 stats_channels=6,
                 kernel_size=[4, 4, 4],
                 radius_scale=1.5,
                 particle_radius=0.025,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True):
        super().__init__()
        inter_channels = inter_channels or channels
        self.stats_channels = stats_channels
        self.filter_extent = torch.tensor(
            np.float32(radius_scale * 6 * particle_radius))

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        conv_fn = ml3d.layers.ContinuousConv if conv_type == 'cconv' else ASCC
        window_fn = window_poly6 if use_window else None

        self.cconv1 = conv_fn(kernel_size=kernel_size,
                              in_channels=channels * 2 + stats_channels,
                              filters=inter_channels,
                              activation=None,
                              align_corners=True,
                              interpolation=interpolation,
                              coordinate_mapping=coordinate_mapping,
                              normalize=False,
                              window_function=window_fn,
                              radius_search_ignore_query_points=True)
        self.batchNorm1 = nn.BatchNorm1d(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.cconv2 = conv_fn(kernel_size=kernel_size,
                              in_channels=inter_channels,
                              filters=channels,
                              activation=None,
                              align_corners=True,
                              interpolation=interpolation,
                              coordinate_mapping=coordinate_mapping,
                              normalize=False,
                              window_function=window_fn,
                              radius_search_ignore_query_points=True)
        self.batchNorm2 = nn.BatchNorm1d(channels)
        self.sigmoid = nn.Sigmoid()

    def _stats_or_zeros(self, x, stats):
        if self.stats_channels <= 0:
            return x.new_zeros((x.shape[0], 0))
        if stats is None:
            return x.new_zeros((x.shape[0], self.stats_channels))
        return stats.to(dtype=x.dtype, device=x.device)

    def forward(self, x, y, pos, stats=None):
        stats = self._stats_or_zeros(x, stats)
        xa = torch.cat((x, y, stats), -1)
        xl = self.cconv1(xa, pos, pos, self.filter_extent)
        xl = self.batchNorm1(xl)
        xl = self.relu1(xl)
        xl = self.cconv2(xl, pos, pos, self.filter_extent)
        xl = self.batchNorm2(xl)
        wei = self.sigmoid(xl)
        return 2 * x * wei + 2 * y * (1 - wei)


class BoundaryIAFF(nn.Module):
    def __init__(self,
                 channels=32,
                 inter_channels=64,
                 conv_type='cconv',
                 stats_channels=6,
                 kernel_size=[4, 4, 4],
                 radius_scale=1.5,
                 particle_radius=0.025,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True):
        super().__init__()
        self.stats_channels = stats_channels
        self.filter_extent = torch.tensor(
            np.float32(radius_scale * 6 * particle_radius))

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        conv_fn = ml3d.layers.ContinuousConv if conv_type == 'cconv' else ASCC
        window_fn = window_poly6 if use_window else None

        def Conv(in_channels, filters):
            return conv_fn(kernel_size=kernel_size,
                           in_channels=in_channels,
                           filters=filters,
                           activation=None,
                           align_corners=True,
                           interpolation=interpolation,
                           coordinate_mapping=coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True)

        self.cconv1 = Conv(channels * 2 + stats_channels, inter_channels)
        self.batchNorm1 = nn.BatchNorm1d(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.cconv2 = Conv(inter_channels, channels)
        self.batchNorm2 = nn.BatchNorm1d(channels)

        self.cconv3 = Conv(channels + stats_channels, inter_channels)
        self.batchNorm3 = nn.BatchNorm1d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.cconv4 = Conv(inter_channels, channels)
        self.batchNorm4 = nn.BatchNorm1d(channels)
        self.sigmoid = nn.Sigmoid()

    def _stats_or_zeros(self, x, stats):
        if self.stats_channels <= 0:
            return x.new_zeros((x.shape[0], 0))
        if stats is None:
            return x.new_zeros((x.shape[0], self.stats_channels))
        return stats.to(dtype=x.dtype, device=x.device)

    def forward(self, x, y, pos, stats=None):
        stats = self._stats_or_zeros(x, stats)

        xa = torch.cat((x, y, stats), -1)
        xl = self.cconv1(xa, pos, pos, self.filter_extent)
        xl = self.batchNorm1(xl)
        xl = self.relu1(xl)
        xl = self.cconv2(xl, pos, pos, self.filter_extent)
        xl = self.batchNorm2(xl)
        wei1 = self.sigmoid(xl)
        xo = 2 * x * wei1 + 2 * y * (1 - wei1)

        xo_feat = torch.cat((xo, stats), -1)
        xl = self.cconv3(xo_feat, pos, pos, self.filter_extent)
        xl = self.batchNorm3(xl)
        xl = self.relu2(xl)
        xl = self.cconv4(xl, pos, pos, self.filter_extent)
        xl = self.batchNorm4(xl)
        wei2 = self.sigmoid(xl)
        return 2 * x * wei2 + 2 * y * (1 - wei2)


class NormalTangentialBoundaryProjector(nn.Module):
    def __init__(self, feature_channels=64, stats_channels=6, scale=0.5):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(feature_channels + stats_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, delta, features, stats):
        wall_score = stats[:, 0:1]
        normal = F.normalize(stats[:, 3:6], dim=-1, eps=1e-6)
        params = self.net(torch.cat([features, stats], dim=-1))
        normal_scale = 1.0 + self.scale * wall_score * torch.tanh(
            params[:, 0:1])
        tangent_scale = 1.0 + self.scale * wall_score * torch.tanh(
            params[:, 1:2])
        normal_bias = self.scale * wall_score * torch.tanh(params[:, 2:3])

        normal_delta = (delta * normal).sum(dim=-1, keepdim=True) * normal
        tangent_delta = delta - normal_delta
        return tangent_scale * tangent_delta + normal_scale * normal_delta + normal_bias * normal


class MyParticleNetwork(torch.nn.Module):
    def __init__(
            self,
            kernel_size=[4, 4, 4],
            radius_scale=1.5,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            timestep=1 / 50,
            gravity=(0, -9.81, 0),
            other_feats_channels=0,
            boundary_stats_channels=6,
            boundary_projector=True,
            boundary_projector_scale=0.5,
            **unused_kwargs,
    ):
        super().__init__()
        self.layer_channels = [32, 64, 128, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self.timestep = timestep
        self.boundary_stats_channels = boundary_stats_channels
        self.boundary_projector_enabled = boundary_projector
        self.num_scale_classes = 0
        self.last_scale_logits = []
        self.last_scale_weights = []
        self.last_gate_weights = []
        self.last_state_aux = {}

        gravity = torch.FloatTensor(gravity)
        self.register_buffer('gravity', gravity)

        self._all_convs_cconv = []
        self._all_convs_ascc = []

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, conv_type='cconv', **kwargs):
            conv_fn = ml3d.layers.ContinuousConv if conv_type == 'cconv' else ASCC
            window_fn = window_poly6 if self.use_window else None
            conv = conv_fn(kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)
            if conv_type == 'cconv':
                self._all_convs_cconv.append((name, conv))
            else:
                self._all_convs_ascc.append((name, conv))
            return conv

        aff_kwargs = dict(stats_channels=boundary_stats_channels,
                          kernel_size=kernel_size,
                          radius_scale=radius_scale,
                          particle_radius=particle_radius,
                          coordinate_mapping=coordinate_mapping,
                          interpolation=interpolation,
                          use_window=use_window)

        in_channels = 4 + other_feats_channels

        self.conv0_fluid_cconv = Conv(name="cconv0_fluid",
                                      in_channels=in_channels,
                                      filters=self.layer_channels[0],
                                      activation=None,
                                      conv_type='cconv')
        self.conv0_obstacle_cconv = Conv(name="cconv0_obstacle",
                                         in_channels=3,
                                         filters=self.layer_channels[0],
                                         activation=None,
                                         conv_type='cconv')
        self.dense0_fluid_cconv = nn.Linear(in_channels,
                                            self.layer_channels[0])
        nn.init.xavier_uniform_(self.dense0_fluid_cconv.weight)
        nn.init.zeros_(self.dense0_fluid_cconv.bias)

        self.convs_cconv = []
        self.denses_cconv = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch = 64
            out_ch = self.layer_channels[i]
            dense = nn.Linear(in_ch, out_ch)
            nn.init.xavier_uniform_(dense.weight)
            nn.init.zeros_(dense.bias)
            conv = Conv(name='cconv{}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None,
                        conv_type='cconv')
            setattr(self, 'dense_cconv{}'.format(i), dense)
            setattr(self, 'cconv{}'.format(i), conv)
            self.denses_cconv.append(dense)
            self.convs_cconv.append(conv)

        self.conv0_fluid_ascc = Conv(name="ascc0_fluid",
                                     in_channels=in_channels,
                                     filters=self.layer_channels[0],
                                     activation=None,
                                     conv_type='ascc')
        self.conv0_obstacle_ascc = Conv(name="ascc0_obstacle",
                                        in_channels=3,
                                        filters=self.layer_channels[0],
                                        activation=None,
                                        conv_type='ascc')
        self.dense0_fluid_ascc = nn.Linear(in_channels,
                                           self.layer_channels[0])
        nn.init.xavier_uniform_(self.dense0_fluid_ascc.weight)
        nn.init.zeros_(self.dense0_fluid_ascc.bias)

        self.convs_ascc = []
        self.denses_ascc = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch = 64
            out_ch = self.layer_channels[i]
            dense = nn.Linear(in_ch, out_ch)
            nn.init.xavier_uniform_(dense.weight)
            nn.init.zeros_(dense.bias)
            conv = Conv(name='ascc{}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None,
                        conv_type='ascc')
            setattr(self, 'dense_ascc{}'.format(i), dense)
            setattr(self, 'ascc{}'.format(i), conv)
            self.denses_ascc.append(dense)
            self.convs_ascc.append(conv)

        self.aff_cconv = BoundaryIAFF(channels=32,
                                      inter_channels=64,
                                      conv_type='cconv',
                                      **aff_kwargs)
        self.aff_ascc = BoundaryIAFF(channels=32,
                                     inter_channels=64,
                                     conv_type='ascc',
                                     **aff_kwargs)
        self.aff0 = BoundaryAFF(channels=self.layer_channels[0] * 2,
                                inter_channels=self.layer_channels[0] * 2,
                                conv_type='cconv',
                                **aff_kwargs)
        self.affs = []
        for i in range(1, len(self.layer_channels)):
            ch = self.layer_channels[i]
            aff = BoundaryAFF(channels=ch,
                              inter_channels=ch,
                              conv_type='cconv',
                              **aff_kwargs)
            setattr(self, 'aff{}'.format(i), aff)
            self.affs.append(aff)
        self.resAff = BoundaryAFF(channels=64,
                                  inter_channels=64,
                                  conv_type='cconv',
                                  **aff_kwargs)

        self.boundary_projector = NormalTangentialBoundaryProjector(
            feature_channels=64,
            stats_channels=boundary_stats_channels,
            scale=boundary_projector_scale,
        )

    def integrate_pos_vel(self, pos1, vel1):
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        dt = self.timestep
        pos = pos2 + pos_correction
        vel = (pos - pos1) / dt
        return pos, vel

    def _row_ids_from_row_splits(self, row_splits):
        counts = (row_splits[1:] - row_splits[:-1]).to(torch.long)
        if counts.numel() == 0 or int(counts.sum().item()) == 0:
            return row_splits.new_empty((0,), dtype=torch.long)
        return torch.repeat_interleave(
            torch.arange(counts.shape[0], device=row_splits.device), counts)

    def _aggregate_min(self, values, row_ids, num_queries, fill_value):
        out = values.new_full((num_queries, values.shape[-1]), fill_value)
        if values.numel() == 0:
            return out
        if hasattr(out, 'scatter_reduce_'):
            index = row_ids.unsqueeze(-1).expand_as(values)
            out.scatter_reduce_(0,
                                index,
                                values,
                                reduce='amin',
                                include_self=True)
            return out
        for query_idx in range(num_queries):
            mask = row_ids == query_idx
            if mask.any():
                out[query_idx] = values[mask].min(dim=0).values
        return out

    def _compute_boundary_stats(self, pos, vel, box, box_feats):
        fluid_nns = self.conv0_fluid_cconv.nns
        fluid_row_splits = fluid_nns.neighbors_row_splits.long()
        fluid_counts = (fluid_row_splits[1:] - fluid_row_splits[:-1]).to(
            pos.dtype)
        self.num_fluid_neighbors = fluid_counts

        density_proxy = fluid_counts.unsqueeze(-1) / fluid_counts.mean(
        ).detach().clamp(min=1.0)
        surface_score = torch.exp(-density_proxy)

        wall_dist = pos.new_full((pos.shape[0], 1), 1.0)
        wall_normal = pos.new_zeros((pos.shape[0], 3))
        obstacle_nns = self.conv0_obstacle_cconv.nns
        obstacle_idx = obstacle_nns.neighbors_index.long()
        obstacle_row_splits = obstacle_nns.neighbors_row_splits.long()
        obstacle_row_ids = self._row_ids_from_row_splits(obstacle_row_splits)
        if obstacle_idx.numel() > 0:
            if hasattr(obstacle_nns, 'neighbors_distance'):
                obstacle_dist = torch.sqrt(
                    torch.clamp(obstacle_nns.neighbors_distance,
                                min=1e-12)).unsqueeze(-1)
            else:
                obstacle_dist = torch.norm(box[obstacle_idx] -
                                           pos[obstacle_row_ids],
                                           dim=-1,
                                           keepdim=True)
            wall_dist = self._aggregate_min(obstacle_dist,
                                            obstacle_row_ids,
                                            pos.shape[0],
                                            float(self.filter_extent))
            wall_dist = wall_dist / max(float(self.filter_extent), 1e-6)

            normal_sum = pos.new_zeros((pos.shape[0], 3))
            normal_sum.index_add_(0, obstacle_row_ids, box_feats[obstacle_idx])
            normal_count = torch.bincount(
                obstacle_row_ids,
                minlength=pos.shape[0]).to(pos.dtype).unsqueeze(-1)
            wall_normal = normal_sum / normal_count.clamp(min=1.0)
            wall_normal = F.normalize(wall_normal, dim=-1, eps=1e-6)

        speed = torch.norm(vel, dim=-1, keepdim=True)
        inward_vn = F.relu(-(vel * wall_normal).sum(dim=-1, keepdim=True))
        inward_ratio = inward_vn / speed.clamp(min=1e-6)
        wall_score = torch.exp(-4.0 * wall_dist)
        stats = torch.cat(
            [wall_score, surface_score, inward_ratio, wall_normal], dim=-1)

        self.last_state_aux = {
            'wall_dist': wall_dist,
            'density_proxy': density_proxy,
            'surface_score': surface_score,
            'vorticity_mag': pos.new_zeros((pos.shape[0], 1)),
            'inward_vn': inward_vn,
            'avg_wall_normal': wall_normal,
        }
        self.last_gate_weights = [wall_score]
        return stats

    def compute_correction(self,
                           pos,
                           vel,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None,
                           scale_labels=None,
                           teacher_forcing_alpha=0.0):
        filter_extent = torch.tensor(self.filter_extent,
                                     device=pos.device,
                                     dtype=pos.dtype)
        fluid_feats = [torch.ones_like(pos[:, 0:1]), vel]
        if other_feats is not None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, axis=-1)

        ans_conv0_fluid_cconv = self.conv0_fluid_cconv(
            fluid_feats, pos, pos, filter_extent)
        ans_dense0_fluid_cconv = self.dense0_fluid_cconv(fluid_feats)
        ans_conv0_obstacle_cconv = self.conv0_obstacle_cconv(
            box_feats, box, pos, filter_extent)
        boundary_stats = self._compute_boundary_stats(
            pos, vel, box, box_feats)

        hybrid_aff_cconv = self.aff_cconv(ans_conv0_fluid_cconv,
                                          ans_conv0_obstacle_cconv, pos,
                                          boundary_stats)
        feats_cconv = torch.cat([hybrid_aff_cconv, ans_dense0_fluid_cconv],
                                axis=-1)

        ans_conv0_fluid_ascc = self.conv0_fluid_ascc(
            fluid_feats, pos, pos, filter_extent)
        ans_dense0_fluid_ascc = self.dense0_fluid_ascc(fluid_feats)
        ans_conv0_obstacle_ascc = self.conv0_obstacle_ascc(
            box_feats, box, pos, filter_extent)
        hybrid_aff_ascc = self.aff_ascc(ans_conv0_fluid_ascc,
                                        ans_conv0_obstacle_ascc, pos,
                                        boundary_stats)
        feats_ascc = torch.cat([hybrid_aff_ascc, ans_dense0_fluid_ascc],
                               axis=-1)

        feats_select = self.aff0(feats_cconv, feats_ascc, pos,
                                 boundary_stats)
        self.ans_convs = [feats_select]
        projector_feats = None

        for conv_cconv, dense_cconv, conv_ascc, dense_ascc, aff in zip(
                self.convs_cconv, self.denses_cconv, self.convs_ascc,
                self.denses_ascc, self.affs):
            inp_feats = F.relu(self.ans_convs[-1])

            ans_conv_cconv = conv_cconv(inp_feats, pos, pos, filter_extent)
            ans_dense_cconv = dense_cconv(inp_feats)
            ans_cconv = ans_conv_cconv + ans_dense_cconv

            ans_conv_ascc = conv_ascc(inp_feats, pos, pos, filter_extent)
            ans_dense_ascc = dense_ascc(inp_feats)
            ans_ascc = ans_conv_ascc + ans_dense_ascc

            ans_select = aff(ans_cconv, ans_ascc, pos, boundary_stats)
            if (len(self.ans_convs) == 3
                    and ans_dense_cconv.shape[-1] == self.ans_convs[-2].
                    shape[-1]):
                ans_select = self.resAff(ans_select, self.ans_convs[-2], pos,
                                         boundary_stats)

            if ans_select.shape[-1] == 3 and inp_feats.shape[-1] == 64:
                projector_feats = inp_feats

            self.ans_convs.append(ans_select)

        output_delta = self.ans_convs[-1]
        if self.boundary_projector_enabled and projector_feats is not None:
            output_delta = self.boundary_projector(output_delta,
                                                   projector_feats,
                                                   boundary_stats)

        self.pos_correction = (1.0 / 128) * output_delta
        self.last_scale_logits = []
        self.last_scale_weights = []
        return self.pos_correction

    def forward(self,
                inputs,
                fixed_radius_search_hash_table=None,
                scale_labels=None,
                teacher_forcing_alpha=0.0):
        pos, vel, feats, box, box_feats = inputs
        pos2, vel2 = self.integrate_pos_vel(pos, vel)
        pos_correction = self.compute_correction(
            pos2,
            vel2,
            feats,
            box,
            box_feats,
            fixed_radius_search_hash_table,
            scale_labels=scale_labels,
            teacher_forcing_alpha=teacher_forcing_alpha)
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)
        return pos2_corrected, vel2_corrected