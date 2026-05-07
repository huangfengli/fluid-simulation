#!/usr/bin/env python3
import csv
import importlib
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import sys
import argparse
import yaml
from datetime import datetime
from torch.backends import cudnn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from datasets.dataset_reader_physics_random_gravity import read_data_train, read_data_val
from datasets.dataset_reader_physics import read_data_train, read_data_val
from collections import namedtuple
import glob
import time
import torch
import torch.nn.functional as F
from utils.deeplearningutilities.torch import Trainer, MyCheckpointManager
from evaluate_network import evaluate_torch, evaluate_whole_sequence_torch

_k = 1000
device_ids = [0, 1]
TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(60 * _k, 0.001, 2)
min_err = float('inf')


def create_model(**kwargs):
    from models.bgsp_torch import MyParticleNetwork
    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork(**kwargs)
    return model


def main():
    global min_err
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(  "cfg",
                        type=str,
                        help="The path to the yaml config file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    training_cfg = cfg.get('training', {})
    max_iter = int(training_cfg.get('max_iter', train_params.max_iter))
    base_lr = float(training_cfg.get('base_lr', train_params.base_lr))
    batch_size = int(training_cfg.get('batch_size', train_params.batch_size))
    eval_interval = int(training_cfg.get('eval_interval', _k))
    resume_training = bool(training_cfg.get('resume', True))
    scale_loss_weight = float(training_cfg.get('scale_loss_weight', 0.05))
    teacher_full_steps = int(training_cfg.get('teacher_full_steps', 3000))
    teacher_decay_steps = int(training_cfg.get('teacher_decay_steps', 5000))
    rollout_loss_steps = int(training_cfg.get('rollout_loss_steps', 2))
    max_preprocess_steps = int(training_cfg.get('max_preprocess_steps', 4))
    preprocess_progress_steps = int(
        training_cfg.get('preprocess_progress_steps', 20000))
    preprocess_difficulty_scale = float(
        training_cfg.get('preprocess_difficulty_scale', 1.0))
    input_pos_noise_std = float(training_cfg.get('input_pos_noise_std', 0.0))
    input_vel_noise_std = float(training_cfg.get('input_vel_noise_std', 0.0))
    use_region_loss = bool(training_cfg.get('use_region_loss', False))
    train_window = max_preprocess_steps + rollout_loss_steps + 1

    # the train dir stores all checkpoints and summaries. The dir name is the name of this file combined with the name of the config file
    train_dir = os.path.splitext(
        os.path.basename(__file__))[0] + '_' + os.path.splitext(
            os.path.basename(args.cfg))[0]
    os.makedirs(train_dir, exist_ok=True)
    eval_csv_path = os.path.join(train_dir, 'eval_metrics.csv')

    val_files = sorted(glob.glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst'))) #(20个场景序列，每个序列15帧，300帧)
    train_files = sorted(
        glob.glob(os.path.join(cfg['dataset_dir'], 'train', '*.zst'))) #(200个场景序列，每个序列15帧，共3000帧)

    device = torch.device("cuda")

    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    dataset = read_data_train(files=train_files,
                              batch_size=batch_size,
                              window=train_window,
                              num_workers=2,
                              **cfg.get('train_data', {}))

    data_iter = iter(dataset) # 用iter迭代器来遍历dataset，iter通过next来遍历每一个batch

    trainer = Trainer(train_dir)

    model = create_model(**cfg.get('model', {}))
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model, device_ids=[0,1])

    model.cuda()

    boundaries = [
        15 * _k,
        25 * _k,
        35 * _k,
        45 * _k,
        50 * _k,
        55 * _k
    ]
    # boundaries = [
    #     50 * _k,
    #     60 * _k,
    #     70 * _k,
    #     80 * _k,
    #     100 * _k
    # ]
    lr_values = [
        1.5,
        1,
        0.5,
        0.25,
        0.125,
        0.125 * 0.5
    ]

    def lrfactor_fn(x):
        factor = lr_values[0]
        for b, v in zip(boundaries, lr_values[1:]):
            if x > b:
                factor = v
            else:
                break
        return factor

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=base_lr,
                                 weight_decay=0.001,
                                 eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrfactor_fn)

    step = torch.tensor(0)
    checkpoint_fn = lambda: {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    manager = MyCheckpointManager(checkpoint_fn,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, max_iter + 1,
                                            1 * _k)),
                                  save_interval_minutes=None)

    def append_eval_metrics(step_value, metrics):
        write_header = (not os.path.isfile(eval_csv_path) or
                        os.path.getsize(eval_csv_path) == 0)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(eval_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['timestamp', 'step', 'metric', 'value'])
            for metric_name, metric_value in metrics.items():
                writer.writerow([timestamp, int(step_value), metric_name,
                                 metric_value])

    def euclidean_distance(a, b, epsilon=1e-9):
        return torch.sqrt(torch.sum((a - b)**2, dim=-1) + epsilon)

    def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
        return torch.mean(importance *
                          euclidean_distance(pr_pos, gt_pos)**gamma)

    def region_aware_loss(pr_pos, gt_pos, num_fluid_neighbors, state_aux):
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
        if state_aux is None:
            region_weight = 1.0
        else:
            wall_score = torch.exp(-state_aux['wall_dist']).detach()
            surface_score = state_aux['surface_score'].detach()
            vorticity_score = torch.clamp(state_aux['vorticity_mag'].detach(),
                                          min=0.0,
                                          max=3.0)
            region_weight = 1.0 + 0.30 * wall_score + 0.30 * surface_score + 0.15 * vorticity_score
        return torch.mean(region_weight * importance *
                          euclidean_distance(pr_pos, gt_pos)**gamma)

    def teacher_forcing_alpha(step_value):
        if step_value < teacher_full_steps:
            return 1.0
        decay_end = teacher_full_steps + teacher_decay_steps
        if step_value >= decay_end:
            return 0.0
        progress = (step_value - teacher_full_steps) / max(
            float(teacher_decay_steps), 1.0)
        return 1.0 - progress

    def sample_difficulty(scale_sequence, num_classes):
        valid = [x for x in scale_sequence if x is not None]
        if not valid:
            return 0.0
        denom = max(num_classes - 1, 1)
        per_step = [
            (scale.float().mean() / denom) for scale in valid
        ]
        return torch.stack(per_step).mean().item()

    def preprocess_steps_for_example(step_value, scale_sequence, num_classes):
        progress = min(
            float(step_value) / max(float(preprocess_progress_steps), 1.0),
            1.0)
        difficulty = sample_difficulty(scale_sequence, num_classes)
        target = progress * max_preprocess_steps * (
            1.0 + preprocess_difficulty_scale * difficulty)
        steps = int(round(target))
        max_allowed = max(0, len(scale_sequence) - rollout_loss_steps)
        return max(0, min(max_preprocess_steps, max_allowed, steps)), difficulty

    def scale_loss_fn(scale_logits_list, labels, num_classes=4):
        if (labels is None or not scale_logits_list or
                num_classes is None or num_classes <= 0):
            zero = torch.tensor(0.0, device=next(model.parameters()).device)
            hist = torch.zeros((1,), device=zero.device)
            return zero, zero, hist, hist
        labels = labels.long()
        losses = [
            F.cross_entropy(scale_logits, labels)
            for scale_logits in scale_logits_list
        ]
        scale_loss = sum(losses) / len(losses)
        with torch.no_grad():
            pred = torch.argmax(scale_logits_list[-1], dim=-1)
            acc = (pred == labels).to(torch.float32).mean()
            label_hist = torch.bincount(labels,
                                        minlength=num_classes).to(
                                            torch.float32)
            pred_hist = torch.bincount(pred,
                                       minlength=num_classes).to(
                                           torch.float32)
            label_hist = label_hist / label_hist.sum().clamp(min=1.0)
            pred_hist = pred_hist / pred_hist.sum().clamp(min=1.0)
        return scale_loss, acc, label_hist, pred_hist

    def contrast_gain_value(model):
        gains = [
            torch.tanh(param.detach()).abs().mean()
            for name, param in model.named_parameters()
            if 'contrast_gain' in name
        ]
        if not gains:
            return 0.0
        return float(torch.stack(gains).mean())

    def train(model, batch):
        optimizer.zero_grad()
        pos_losses = []
        scale_losses = []
        scale_accs = []
        label_hists = []
        pred_hists = []
        preprocess_steps_used = []
        sample_difficulties = []
        alpha = teacher_forcing_alpha(trainer.current_step)

        for batch_i in range(batch_size):
            box = batch['box'][batch_i]
            box_normals = batch['box_normals'][batch_i]
            scale_sequence = []
            step_i = 0
            while 'scale{}'.format(step_i) in batch:
                scale_sequence.append(batch['scale{}'.format(step_i)][batch_i])
                step_i += 1

            preprocess_steps, difficulty = preprocess_steps_for_example(
                trainer.current_step,
                scale_sequence,
                model.num_scale_classes)
            preprocess_steps_used.append(
                torch.tensor(float(preprocess_steps), device=device))
            sample_difficulties.append(
                torch.tensor(float(difficulty), device=device))

            pos = batch['pos0'][batch_i]
            vel = batch['vel0'][batch_i]
            if input_pos_noise_std > 0.0:
                pos = pos + input_pos_noise_std * torch.randn_like(pos)
            if input_vel_noise_std > 0.0:
                vel = vel + input_vel_noise_std * torch.randn_like(vel)

            # Warmup rollout W steps without retaining activations.
            for rollout_i in range(preprocess_steps):
                with torch.no_grad():
                    scale_i = None
                    if rollout_i < len(scale_sequence):
                        scale_i = scale_sequence[rollout_i]
                    pos, vel = model((pos, vel, None, box, box_normals),
                                     scale_labels=scale_i,
                                     teacher_forcing_alpha=alpha)

            step_pos_losses = []
            step_scale_losses = []
            step_scale_accs = []
            step_label_hists = []
            step_pred_hists = []
            start_i = preprocess_steps
            end_i = preprocess_steps + rollout_loss_steps
            for rollout_i in range(start_i, end_i):
                scale_i = None
                if rollout_i < len(scale_sequence):
                    scale_i = scale_sequence[rollout_i]
                pos, vel = model((pos, vel, None, box, box_normals),
                                 scale_labels=scale_i,
                                 teacher_forcing_alpha=alpha)
                state_aux = {k: v for k, v in model.last_state_aux.items()}
                gt_pos = batch['pos{}'.format(rollout_i + 1)][batch_i]
                if use_region_loss:
                    pos_l = region_aware_loss(pos, gt_pos,
                                              model.num_fluid_neighbors,
                                              state_aux)
                else:
                    pos_l = loss_fn(pos, gt_pos, model.num_fluid_neighbors)
                step_pos_losses.append(pos_l)
                scale_l, scale_acc, label_hist, pred_hist = scale_loss_fn(
                    model.last_scale_logits,
                    scale_i,
                    num_classes=model.num_scale_classes)
                step_scale_losses.append(scale_l)
                step_scale_accs.append(scale_acc)
                step_label_hists.append(label_hist)
                step_pred_hists.append(pred_hist)

            pos_losses.append(sum(step_pos_losses) / len(step_pos_losses))
            scale_losses.append(sum(step_scale_losses) / len(step_scale_losses))
            scale_accs.append(sum(step_scale_accs) / len(step_scale_accs))
            label_hists.append(torch.stack(step_label_hists).mean(dim=0))
            pred_hists.append(torch.stack(step_pred_hists).mean(dim=0))

        pos_loss = 128 * sum(pos_losses) / batch_size
        scale_loss = sum(scale_losses) / batch_size
        total_loss = pos_loss + scale_loss_weight * scale_loss
        total_loss.backward()
        optimizer.step() #一个batch更新一次

        return (total_loss, pos_loss.detach(), scale_loss.detach(),
                torch.stack(scale_accs).mean().detach(),
                torch.stack(label_hists).mean(dim=0).detach(),
                torch.stack(pred_hists).mean(dim=0).detach(),
                torch.tensor(alpha, device=pos_loss.device),
                torch.stack(preprocess_steps_used).mean().detach(),
                torch.stack(sample_difficulties).mean().detach())

    if resume_training and manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        latest_checkpoint = torch.load(manager.latest_checkpoint)
        step = latest_checkpoint['step']
        incompatible = model.load_state_dict(latest_checkpoint['model'],
                                             strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print('checkpoint mismatch:',
                  'missing=', incompatible.missing_keys,
                  'unexpected=', incompatible.unexpected_keys)
        try:
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scheduler.load_state_dict(latest_checkpoint['scheduler'])
        except ValueError as exc:
            print('skip optimizer/scheduler restore:', exc)


    display_str_list = []
    while trainer.keep_training(step,
                                max_iter, #最大迭代次数50000
                                checkpoint_manager=manager,
                                display_str_list=display_str_list):
        # 每次循环一个batch
        # 所以总共迭代帧应该是max_iter * batch_size
        # 通过不断迭代数据集直到停止，没有分多少个epoch，而是通过iter一直循环下去。 这里的iter()数据遍历完会继续重复
        data_fetch_start = time.time()
        batch = next(data_iter) # iter通过next来遍历每一个batch，其中一个batch有16(batch_size)组场景数据，每组数据包含20个属性，如pos1，vel1(第一帧gt)，pos2，vel2(第二帧gt)等 【每组数据只有两帧】

        batch_torch = {}
        for k, v in batch.items():
            if k == 'box' or k == 'box_normals' or k.startswith('pos') or k.startswith('vel'):
                batch_torch[k] = [torch.from_numpy(x).to(device) for x in v]
            elif k.startswith('scale'):
                batch_torch[k] = [
                    torch.from_numpy(x).long().to(device) for x in v
                ]
        data_fetch_latency = time.time() - data_fetch_start #数据延迟：存储或检索数据包所需的时间
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        (current_loss, current_pos_loss, current_scale_loss, scale_acc,
         label_hist, pred_hist, current_alpha, current_preprocess_steps,
         current_difficulty) = train(model, batch_torch)
        #batch_torch(n, bs, p,3) n是n种属性（作键），bs是batch_size，p是该组场景数据中的点数（各场景点数不相同），3是特征维度为三维（xyz）
        scheduler.step()
        current_contrast_gain = contrast_gain_value(model)
        display_str_list = [
            'loss', float(current_loss),
            'pos_loss', float(current_pos_loss),
            'contrast_gain', current_contrast_gain
        ] #记录每一次迭代的loss

        if trainer.current_step % 10 == 0:
            trainer.summary_writer.add_scalar('TotalLoss', current_loss,
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('PositionLoss',
                                              current_pos_loss,
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('ScaleLoss',
                                              current_scale_loss,
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('scale/accuracy',
                                              scale_acc,
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('scale/teacher_alpha',
                                              current_alpha,
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('rollout/preprocess_steps',
                                              current_preprocess_steps,
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('rollout/difficulty',
                                              current_difficulty,
                                              trainer.current_step)
            for class_i in range(label_hist.shape[0]):
                trainer.summary_writer.add_scalar(
                    'scale/label_class_{}'.format(class_i),
                    label_hist[class_i],
                    trainer.current_step)
                trainer.summary_writer.add_scalar(
                    'scale/pred_class_{}'.format(class_i),
                    pred_hist[class_i],
                    trainer.current_step)
            trainer.summary_writer.add_scalar('LearningRate',
                                              scheduler.get_last_lr()[0],
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('fusion/contrast_gain',
                                              current_contrast_gain,
                                              trainer.current_step)
            if model.last_gate_weights:
                gate_mean = torch.stack([
                    gate.mean() for gate in model.last_gate_weights
                ]).mean()
                trainer.summary_writer.add_scalar('gate/regional_mean',
                                                  gate_mean.item(),
                                                  trainer.current_step)
        # 每10个iteration打印一次loss

        if eval_interval > 0 and (trainer.current_step) % eval_interval == 0:
            eval_metrics = evaluate_torch(model,
                                          val_dataset,
                                          frame_skip=20,
                                          device=device,
                                          **cfg.get('evaluation', {}))
            append_eval_metrics(trainer.current_step, eval_metrics)
            for k, v in eval_metrics.items():
                trainer.summary_writer.add_scalar('eval/' + k, v,
                                                  trainer.current_step)
                if(k == "err_n1" and v < min_err):
                    min_err = v
                    best_model_path = os.path.join(
                        train_dir,
                        str(step.item()) + '_model_weights_best.pt')
                    torch.save({'model': model.state_dict()},
                               best_model_path)
                    print("=================update best model: err_n1=" + str(v) + "===============")

    # 第1000次迭代eval一次

    torch.save({'model': model.state_dict()},
               os.path.join(train_dir, 'model_weights.pt')) #所有batch遍历一遍记录下当前model
    if trainer.current_step == max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.exit(main())
