#!/usr/bin/env python3
import argparse
import glob
import os

import msgpack
import msgpack_numpy
import numpy as np
import zstandard as zstd

msgpack_numpy.patch()


def read_record(path):
    decompressor = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        return msgpack.unpackb(decompressor.decompress(f.read()), raw=False)


def cell_stats(pos, disp, depth, eps=1e-12):
    num_bins = 2 ** int(depth)
    pos_min = pos.min(axis=0)
    pos_max = pos.max(axis=0)
    span = np.maximum(pos_max - pos_min, 1e-6)
    cell = ((pos - pos_min) / span * num_bins).astype(np.int64)
    cell = np.clip(cell, 0, num_bins - 1)
    cell_ids = (cell[:, 0] * num_bins * num_bins +
                cell[:, 1] * num_bins + cell[:, 2])

    _, inverse = np.unique(cell_ids, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)

    sum_disp = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(sum_disp, inverse, disp.astype(np.float64))
    mean_disp = sum_disp / np.maximum(counts[:, None], 1.0)

    residual = disp.astype(np.float64) - mean_disp[inverse]
    residual_sqr = np.sum(residual * residual, axis=1)
    disp_sqr = np.sum(disp.astype(np.float64) * disp.astype(np.float64),
                      axis=1)

    sum_residual_sqr = np.bincount(inverse,
                                   weights=residual_sqr,
                                   minlength=counts.shape[0])
    sum_disp_sqr = np.bincount(inverse,
                               weights=disp_sqr,
                               minlength=counts.shape[0])
    var = sum_residual_sqr / np.maximum(counts, 1.0)
    disp_energy = sum_disp_sqr / np.maximum(counts, 1.0)
    norm_var = var / np.maximum(disp_energy, eps)

    return norm_var[inverse].astype(np.float32), counts[inverse].astype(
        np.float32)


def compute_labels(pos,
                   next_pos,
                   epsilon,
                   min_particles=2,
                   max_particles=128):
    disp = next_pos - pos

    norm1, count1 = cell_stats(pos, disp, depth=1)
    norm2, count2 = cell_stats(pos, disp, depth=2)
    norm3, count3 = cell_stats(pos, disp, depth=3)
    norm4, count4 = cell_stats(pos, disp, depth=4)

    # 0=depth1, 1=depth2, 2=depth3, 3=depth4.
    # Start from a shallow octree level. If motion entropy is still high,
    # split to the next deeper level until the entropy falls below epsilon
    # or the maximum depth is reached.
    labels = np.zeros((pos.shape[0],), dtype=np.uint8)
    labels[norm1 > epsilon] = 1
    labels[(labels == 1) & (norm2 > epsilon)] = 2
    labels[(labels == 2) & (norm3 > epsilon)] = 3

    stats = {
        'norm1': norm1,
        'norm2': norm2,
        'norm3': norm3,
        'norm4': norm4,
        'count1': count1,
        'count2': count2,
        'count3': count3,
        'count4': count4,
    }
    return labels, stats


def label_file(path,
               epsilon,
               min_particles=2,
               max_particles=128):
    data = read_record(path)
    labels = []
    norm_samples = []
    for frame_i in range(len(data) - 1):
        pos = data[frame_i]['pos'].astype(np.float32)
        next_pos = data[frame_i + 1]['pos'].astype(np.float32)
        frame_labels, stats = compute_labels(pos,
                                             next_pos,
                                             epsilon=epsilon,
                                             min_particles=min_particles,
                                             max_particles=max_particles)
        labels.append(frame_labels)
        norm_samples.append(
            np.stack([
                stats['norm1'], stats['norm2'], stats['norm3'],
                stats['norm4']
            ],
                     axis=1))
    return np.stack(labels, axis=0), np.concatenate(norm_samples, axis=0)


def collect_norm_vars(files, max_frames=None):
    samples = []
    frame_count = 0
    for path in files:
        data = read_record(path)
        for frame_i in range(len(data) - 1):
            pos = data[frame_i]['pos'].astype(np.float32)
            next_pos = data[frame_i + 1]['pos'].astype(np.float32)
            disp = next_pos - pos
            norm1, _ = cell_stats(pos, disp, depth=1)
            norm2, _ = cell_stats(pos, disp, depth=2)
            norm3, _ = cell_stats(pos, disp, depth=3)
            norm4, _ = cell_stats(pos, disp, depth=4)
            samples.append(np.stack([norm1, norm2, norm3, norm4], axis=1))
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                return np.concatenate(samples, axis=0)
    return np.concatenate(samples, axis=0)


def print_norm_summary(norm_vars):
    names = ['depth1', 'depth2', 'depth3', 'depth4']
    quantiles = [0.1, 0.25, 0.5, 0.65, 0.75, 0.9, 0.95]
    for i, name in enumerate(names):
        vals = norm_vars[:, i]
        qs = np.quantile(vals, quantiles)
        q_str = ' '.join(
            ['q{:.0f}={:.6f}'.format(q * 100, v)
             for q, v in zip(quantiles, qs)])
        print('{} {}'.format(name, q_str))


def print_label_summary(all_counts):
    total = all_counts.sum()
    if total == 0:
        print('no labels')
        return
    print('label counts:', all_counts.tolist())
    print('label ratio:',
          ['{:.4f}'.format(x) for x in (all_counts / total).tolist()])
    print('class mapping: 0=depth1, 1=depth2, 2=depth3, 3=depth4')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',
                        default='datasets/ours_default_data')
    parser.add_argument('--split', default='train', choices=['train', 'valid'])
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--epsilon', default='auto')
    parser.add_argument('--epsilon-quantile', type=float, default=0.65)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--max-epsilon-frames', type=int, default=20)
    parser.add_argument('--min-particles', type=int, default=2)
    parser.add_argument('--max-particles', type=int, default=512)
    parser.add_argument('--stats-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.dataset_dir, args.split, '*.zst')))
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        raise RuntimeError('no input files found')

    norm_vars = collect_norm_vars(files, max_frames=args.max_epsilon_frames)
    print_norm_summary(norm_vars)

    if args.epsilon == 'auto':
        epsilon = float(np.quantile(norm_vars[:, 0], args.epsilon_quantile))
    else:
        epsilon = float(args.epsilon)
    print('using epsilon {:.8f}'.format(epsilon))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.dataset_dir, 'scale_gt', args.split)
    if not args.stats_only:
        os.makedirs(output_dir, exist_ok=True)

    all_counts = np.zeros((4,), dtype=np.int64)
    for file_i, path in enumerate(files):
        labels, _ = label_file(path,
                               epsilon=epsilon,
                               min_particles=args.min_particles,
                               max_particles=args.max_particles)
        counts = np.bincount(labels.reshape(-1), minlength=4)
        all_counts += counts
        print('{}/{} {} counts={}'.format(file_i + 1, len(files),
                                          os.path.basename(path),
                                          counts.tolist()))
        if args.stats_only:
            continue
        out_path = os.path.join(
            output_dir,
            os.path.basename(path).replace('.msgpack.zst', '.npz'))
        if os.path.exists(out_path) and not args.overwrite:
            continue
        np.savez_compressed(out_path,
                            scale_labels=labels,
                            epsilon=np.float32(epsilon),
                            min_particles=np.int32(args.min_particles),
                            max_particles=np.int32(args.max_particles),
                            class_mapping=np.array(
                                ['depth1', 'depth2', 'depth3', 'depth4']))

    print_label_summary(all_counts)


if __name__ == '__main__':
    main()
