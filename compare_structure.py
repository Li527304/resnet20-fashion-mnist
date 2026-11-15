"""Verify experiment compatibility and compare five trained variants."""
import json
import time

import numpy as np
import torch
from torchvision.datasets import FashionMNIST

from analyze import finish, plt
from infer import CLASSES, load, preprocess
from train import DEFAULT_RESULTS, ROOT, write_json

NAMES = {
    'base': 'ResNet20',
    'se': 'SE',
    'ds': '深度可分离',
    'gelu': 'GELU',
    'gn': 'GroupNorm',
}
COLORS = ['#4a6675', '#16765b', '#bb693a', '#4f73a5', '#805f91']
WARMUP = 50
REPEATS = 100


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


@torch.inference_mode()
def benchmark(runs):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the same-device GPU latency comparison.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = FashionMNIST(ROOT / 'data', train=False)
    image = preprocess(dataset[0][0]).to('cuda')
    timings = {}
    for seed, group in runs.items():
        models = {
            variant: load(variant, seed, 'cuda', directory)
            for variant, directory in group
        }
        samples = {variant: [] for variant in models}
        for model in models.values():
            for _ in range(WARMUP):
                model(image)
        torch.cuda.synchronize()
        # Rotate measurement order so one variant does not always go first.
        order = list(models)
        for round_id in range(5):
            for variant in order[round_id:] + order[:round_id]:
                model = models[variant]
                for _ in range(REPEATS // 5):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    model(image)
                    torch.cuda.synchronize()
                    samples[variant].append((time.perf_counter() - start) * 1000)
        for variant, values in samples.items():
            timings[variant, seed] = {
                'median_ms': float(np.median(values)),
                'p95_ms': float(np.percentile(values, 95)),
            }
    return timings


def plot(summary):
    labels = [NAMES[row['variant']] for row in summary]
    colors = COLORS
    y = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.0), layout='constrained')
    specs = [
        ('accuracy_mean', '测试准确率（%）', 100, '.2f'),
        ('params', '参数量（千）', .001, '.1f'),
        ('macs', '计算量（百万MACs）', .000001, '.2f'),
        ('latency_ms_mean', 'GPU单张推理耗时（ms）', 1, '.3f'),
    ]
    for ax, (key, title, scale, precision) in zip(axes.flat, specs):
        vals = np.array([row[key] * scale for row in summary])
        ax.set_title(title, loc='left', weight='bold', fontsize=13, pad=8)
        ax.set_yticks(y, labels)
        ax.set_ylim(len(labels) - 0.45, -0.55)
        ax.grid(axis='x', alpha=.8)
        ax.set_axisbelow(True)
        if key == 'accuracy_mean':
            std = [row['accuracy_std'] * scale for row in summary]
            ax.errorbar(vals, y, xerr=std, fmt='none', ecolor='#45525c', capsize=3, zorder=2)
            ax.scatter(vals, y, c=colors, s=42, zorder=3)
            left = float(min(vals - np.array(std))) - .16
            right = float(max(vals + np.array(std))) + .35
            padding = np.array(std) + .055
        else:
            ax.barh(y, vals, color=colors, height=.51)
            left, right = 0, float(max(vals)) * 1.26
            padding = np.full(len(vals), right * .025)
        ax.set_xlim(left, right)
        for value, offset, i in zip(vals, padding, y):
            ax.text(value + offset, i, format(value, precision),
                    fontsize=9, va='center', weight='bold', color=colors[i])
    finish(fig, 'structure_comparison.png')


def main():
    base = DEFAULT_RESULTS
    extra = base / 'structure'
    original = read_json(base / 'config.json')
    added = read_json(extra / 'config.json')
    assert {k: v for k, v in original.items() if k != 'variants'} == {
        k: v for k, v in added.items() if k != 'variants'
    }, 'Training settings differ'
    assert read_json(base / 'split.json') == read_json(extra / 'split.json'), 'Data split differs'
    directories = {
        **dict.fromkeys(original['variants'], base),
        **dict.fromkeys(added['variants'], extra),
    }
    assert set(directories) == set(NAMES)
    seeds = original['seeds']
    timings = benchmark({seed: list(directories.items()) for seed in seeds})
    summary = []
    baseline = None
    for variant, directory in directories.items():
        runs, recalls = [], []
        for seed in seeds:
            run = directory / f'{variant}_{seed}'
            metrics = read_json(run / 'metrics.json')
            assert metrics['seed'] == seed and metrics['variant'] == variant
            runs.append(metrics | timings[variant, seed])
            with np.load(run / 'predictions.npz') as stored:
                matrix = stored['confusion']
                assert matrix.trace() == metrics['test_correct']
                recalls.append(matrix.diagonal() / matrix.sum(axis=1))
        accuracy = np.array([r['test_correct'] / r['test_n'] for r in runs])
        if baseline is None:
            assert variant == 'base'
            baseline = accuracy
        assert len({(r['params'], r['macs']) for r in runs}) == 1
        summary.append({
            'variant': variant,
            'name': NAMES[variant],
            'accuracy_mean': float(accuracy.mean()),
            'accuracy_std': float(accuracy.std(ddof=1)),
            'paired_delta_pp': ((accuracy - baseline) * 100).tolist(),
            'macro_f1_mean': float(np.mean([r['macro_f1'] for r in runs])),
            'params': runs[0]['params'],
            'macs': runs[0]['macs'],
            'latency_ms_mean': float(np.mean([r['median_ms'] for r in runs])),
            'mean_recall': dict(zip(CLASSES, np.mean(recalls, axis=0).tolist())),
            'runs': runs,
        })
    result = {
        'gpu': torch.cuda.get_device_name(0),
        'test_image_index': 0,
        'batch_size': 1,
        'warmup': WARMUP,
        'repeats': REPEATS,
        'seeds': seeds,
        'groups': summary,
    }
    write_json(extra / 'summary.json', result)
    plot(summary)
    for row in summary:
        print(f"{row['variant']:5} {row['accuracy_mean']:.4%} "
              f"{row['params']:>7,d} {row['macs']:>10,d} "
              f"{row['latency_ms_mean']:.3f} ms", flush=True)


if __name__ == '__main__':
    main()

