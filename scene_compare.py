"""Record raw failures, diagnose the input, then test fixed background inversion."""
import csv
import hashlib
import json

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.datasets import FashionMNIST

from analyze import finish, plt
from infer import CLASSES, ROOT, load, predict

OUT = ROOT / 'results' / 'scene'
VARIANTS = ('base', 'se', 'ds', 'gelu', 'gn')
NAMES = ('ResNet20', 'SE', '深度可分离', 'GELU', 'GroupNorm')
SEEDS = (17, 29, 43)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def border_mean(image):
    pixels = np.array(image.convert('L').resize((28, 28), Image.Resampling.BILINEAR))
    mask = np.ones((28, 28), dtype=bool)
    mask[2:-2, 2:-2] = False
    return float(pixels[mask].mean())


def evaluate(samples, invert):
    records = []
    for variant in VARIANTS:
        for seed in SEEDS:
            model = load(variant, seed)
            for sample in samples:
                with Image.open(ROOT / sample['path']) as original:
                    image = ImageOps.invert(original.convert('RGB')) if invert else original
                    probabilities = predict(model, image)
                records.append({
                    'variant': variant, 'seed': seed, 'id': sample['id'],
                    'label_id': sample['label_id'], 'label': sample['label'],
                    'predictions': probabilities,
                    'correct': probabilities[0]['class_id'] == sample['label_id'],
                })
    return records


def main():
    torch.set_num_threads(4)
    samples = json.loads((OUT / 'samples.json').read_text(encoding='utf-8'))['samples']
    assert len(samples) == 8
    for sample in samples:
        assert hashlib.sha256((ROOT / sample['path']).read_bytes()).hexdigest() == sample['sha256']
    write('adaptation_protocol.json', {
        'reason': 'New photos have bright backgrounds and often contain people.',
        'method': 'Invert all RGB pixels once, then use the existing infer.preprocess.',
        'selection': 'Use all eight fixed samples, all five architectures, seeds 17,29,43.',
        'weights': 'Existing checkpoints, no fine tuning.',
        'scope': 'Small scene demonstration; no claim of independent generalization performance.',
    })
    raw = evaluate(samples, False)
    original = json.loads((OUT / 'predictions_all_raw_seed17.json').read_text(encoding='utf-8'))
    for record in raw:
        if record['seed'] == 17:
            saved = next(row for row in original['records'] if row['id'] == record['id'])
            assert record['predictions'] == saved['predictions'][record['variant']]
    write('raw_all_seeds.json', raw)
    dataset = FashionMNIST(ROOT / 'data', train=True)
    original_border = [border_mean(Image.open(ROOT / sample['path'])) for sample in samples]
    # Fixed indices from the original training partition, independent of scene outcomes.
    split = json.loads((ROOT / 'results/split.json').read_text(encoding='utf-8'))
    training_border = [border_mean(dataset[index][0]) for index in split['train_ids'][:100]]
    diagnosis = {
        'photo_border_mean_0_255': float(np.mean(original_border)),
        'training_border_mean_0_255': float(np.mean(training_border)),
        'training_reference_count': len(training_border),
        'original_photo_count': len(original_border),
        'limits': ['Inversion does not remove the model or separate multiple garments.',
                   'Only eight fixed photos are used; no real-world accuracy claim.'],
    }
    write('input_diagnosis.json', diagnosis)
    adapted = evaluate(samples, True)
    write('inverted_all_seeds.json', adapted)
    summary = {'n': len(samples), 'seeds': SEEDS, 'diagnosis': diagnosis, 'groups': []}
    for variant, name in zip(VARIANTS, NAMES):
        group = {'variant': variant, 'name': name}
        for stage, rows in [('raw', raw), ('inverted', adapted)]:
            correct = [sum(r['correct'] for r in rows if r['variant'] == variant and r['seed'] == seed)
                       for seed in SEEDS]
            group[stage + '_correct_by_seed'] = correct
            group[stage + '_mean_correct'] = float(np.mean(correct))
        summary['groups'].append(group)
    write('summary.json', summary)
    with (OUT / 'all_predictions.csv').open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['处理方式', '模型', '种子', '商品ID', '实际类别', '预测类别', '预测概率', '正确'])
        for stage, rows in [('原流程', raw), ('背景反色', adapted)]:
            for row in rows:
                prediction = row['predictions'][0]
                writer.writerow([stage, row['variant'], row['seed'], row['id'], row['label'],
                                 prediction['label'], prediction['probability'], int(row['correct'])])
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.1), layout='constrained')
    for ax, rows, title in zip(axes, [raw, adapted],
                              ['原流程的预测', '背景反色后的预测']):
        selected = {(r['variant'], r['id']): r for r in rows if r['seed'] == 17}
        correct = np.array([[selected[v, s['id']]['correct'] for s in samples] for v in VARIANTS])
        ax.imshow(correct, cmap=plt.matplotlib.colors.ListedColormap(['#f4dfd8', '#d6ece1']), vmin=0, vmax=1)
        for i, variant in enumerate(VARIANTS):
            for j, sample in enumerate(samples):
                row = selected[variant, sample['id']]
                ax.text(j, i, row['predictions'][0]['label'], ha='center', va='center', fontsize=10)
        ax.set_xticks(range(len(samples)), [f"{s['id']}\n{s['label']}" for s in samples], fontsize=9)
        ax.set_yticks(range(len(VARIANTS)), NAMES, fontsize=10)
        ax.set_title(title + '（种子17，绿色为正确，浅红为错误）', loc='left', fontsize=13, pad=9)
    finish(fig, 'scene_before_after.png')
    fig, axes = plt.subplots(2, 8, figsize=(12.8, 4.8), layout='constrained')
    for column, sample in enumerate(samples):
        with Image.open(ROOT / sample['path']) as image:
            for row, picture in enumerate([image.convert('RGB'), ImageOps.invert(image.convert('RGB'))]):
                axes[row, column].imshow(picture)
                axes[row, column].axis('off')
        axes[0, column].set_title(f"{sample['id']}\n{sample['label']}", fontsize=10)
    axes[0, 0].set_ylabel('原图')
    axes[1, 0].set_ylabel('反色')
    finish(fig, 'scene_inputs.png')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

