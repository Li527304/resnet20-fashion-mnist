"""Manual clothing boxes, aspect-preserving input, and fixed GrabCut comparison."""
import json
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from analyze import finish, plt
from infer import ROOT, load, predict

OUT = ROOT / 'results' / 'scene'
BOXES = {
    '53759': [9, 21, 50, 68],
    '1855': [11, 23, 49, 69],
    '39386': [16, 9, 45, 68],
    '26994': [14, 10, 42, 65],
    '39716': [15, 18, 42, 74],
    '4988': [15, 21, 44, 62],
    '15970': [10, 21, 50, 71],
    '30805': [8, 20, 52, 74],
}
VARIANTS = ('base', 'se', 'ds', 'gelu', 'gn')
NAMES = ('ResNet20', 'SE', '深度可分离', 'GELU', 'GroupNorm')
SEEDS = (17, 29, 43)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def center(image):
    resized = ImageOps.contain(image, (24, 24), Image.Resampling.BILINEAR)
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    return canvas


def prepare(image, box, segment):
    left, top, right, bottom = box
    assert 0 <= left < right <= image.width and 0 <= top < bottom <= image.height
    gray = np.array(ImageOps.invert(image.convert('L')))
    mask = np.ones(gray.shape, dtype=np.uint8)
    if segment:
        rgb = np.array(image.convert('RGB'))
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.setRNGSeed(2026)
        cv2.grabCut(rgb, mask, (left, top, right-left, bottom-top),
                    np.zeros((1,65), np.float64), np.zeros((1,65), np.float64),
                    5, cv2.GC_INIT_WITH_RECT)
        mask = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        assert mask.any(), 'GrabCut returned an empty foreground'
    crop = Image.fromarray(gray * mask).crop(box)
    return center(crop), mask


def main():
    torch.set_num_threads(4)
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))['samples']
    assert set(BOXES) == {str(s['id']) for s in samples}
    protocol = {
        'manual_boxes_xyxy': BOXES,
        'box_selection': 'Visual garment extent recorded before classifier inference, right and bottom exclusive.',
        'fit': 'Keep aspect ratio, maximum 24x24, center in a black 28x28 image.',
        'gray': 'Invert grayscale before fitting, matching the previous inversion experiment.',
        'segmentation': 'GrabCut rectangle initialization, 5 iterations, RNG seed 2026.',
        'manual_assistance': 'A person identifies the target clothing box; automatic detection is not used.',
        'cv2_version': cv2.__version__, 'seeds': SEEDS,
        'scope': 'Exploration on the same eight previously inspected images; independent testing remains future work.',
    }
    path = OUT/'crop_protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == json.loads(json.dumps(protocol))
    save('crop_protocol.json', protocol)
    prepared = {'crop_fit': {}, 'crop_grabcut': {}}
    display = []
    image_dir = OUT/'crop_inputs'
    image_dir.mkdir(exist_ok=True)
    image_hashes = {}
    for sample in samples:
        source = ROOT/sample['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == sample['sha256']
        with Image.open(source) as original:
            image = original.convert('RGB')
        annotated = image.copy()
        ImageDraw.Draw(annotated).rectangle(BOXES[str(sample['id'])], outline='#e74b32', width=1)
        pictures = [annotated]
        for stage, segment in [('crop_fit', False), ('crop_grabcut', True)]:
            result, mask = prepare(image, BOXES[str(sample['id'])], segment)
            destination = image_dir/f"{sample['id']}_{stage}.png"
            result.save(destination)
            image_hashes[destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
            prepared[stage][sample['id']] = result
            pictures.append(result)
        display.append(pictures)
    save('crop_input_hashes.json', image_hashes)
    summary = {'n': len(samples), 'seeds': SEEDS, 'groups': []}
    stage_records = {}
    for stage in prepared:
        records = []
        for variant in VARIANTS:
            for seed in SEEDS:
                model = load(variant, seed)
                for sample in samples:
                    probabilities = predict(model, prepared[stage][sample['id']])
                    records.append({
                        'variant': variant, 'seed': seed, 'id': sample['id'],
                        'label': sample['label'], 'label_id': sample['label_id'],
                        'predictions': probabilities,
                        'correct': probabilities[0]['class_id'] == sample['label_id'],
                    })
        save(stage+'_all_seeds.json', records)
        stage_records[stage] = records
    prior = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    for variant, name, old in zip(VARIANTS, NAMES, prior['groups']):
        group = dict(old)
        for stage, rows in stage_records.items():
            counts = [sum(row['correct'] for row in rows
                          if row['variant'] == variant and row['seed'] == seed) for seed in SEEDS]
            group[stage+'_correct_by_seed'] = counts
            group[stage+'_mean_correct'] = float(np.mean(counts))
        summary['groups'].append(group)
    save('crop_summary.json', summary)
    fig, axes = plt.subplots(3, 8, figsize=(13.6, 6.4), layout='constrained')
    for col, (sample, pictures) in enumerate(zip(samples, display)):
        for row, picture in enumerate(pictures):
            axes[row,col].imshow(picture, cmap='gray', vmin=0, vmax=255)
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
        axes[0,col].set_title(f"{sample['id']}\n{sample['label']}", fontsize=10)
    for ax, text in zip(axes[:,0], ['人工框选', '裁剪居中', 'GrabCut后']):
        ax.set_ylabel(text, fontsize=11)
    finish(fig,'scene_crop_inputs.png')
    fig, ax = plt.subplots(figsize=(10.5,4.4),layout='constrained')
    stages = ['raw','inverted','crop_fit','crop_grabcut']
    titles = ['原流程','整图反色','裁剪居中＋反色','裁剪＋GrabCut＋居中']
    colors = ['#b4b9bd','#6b8298','#ce9566','#23866a']
    x = np.arange(len(VARIANTS))
    for i,(stage,title,color) in enumerate(zip(stages,titles,colors)):
        means = [g[stage+'_mean_correct'] for g in summary['groups']]
        bars = ax.bar(x+(i-1.5)*.19,means,width=.18,label=title,color=color)
        for bar,value in zip(bars,means):
            ax.text(bar.get_x()+bar.get_width()/2,value+.06,f'{value:.2f}',ha='center',fontsize=8)
    ax.set_xticks(x,NAMES)
    ax.set_ylim(0,8.6)
    ax.set_ylabel('每8张图的正确张数（三种子的平均）')
    ax.grid(axis='y',alpha=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='upper left',ncol=2,fontsize=9)
    finish(fig,'scene_crop_comparison.png')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

