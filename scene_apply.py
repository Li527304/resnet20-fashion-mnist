"""Apply existing seed17 clothing checkpoints to a fixed set of new product photos."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
import time

from PIL import Image
import torch

from analyze import finish, plt
from infer import CLASSES, ROOT, load, predict

OUT = ROOT / 'results' / 'scene'
IMAGES = ROOT / 'scene_images'
DATASET = 'ashraq/fashion-product-images-small'
PAGE = 'https://huggingface.co/datasets/' + DATASET
MAPPING = {'Tshirts': 0, 'Jeans': 1, 'Trousers': 1, 'Dresses': 3, 'Shirts': 6}
TARGETS = (0, 1, 3, 6)


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'CNN-coursework-scene/1.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def download():
    counts = dict.fromkeys(TARGETS, 0)
    selected = []
    for offset in range(0, 1000, 100):
        url = ('https://datasets-server.huggingface.co/rows?dataset=' + DATASET
               + f'&config=default&split=train&offset={offset}&length=100')
        print(f'Fetching rows {offset}-{offset+99}', flush=True)
        page = json.loads(get(url))
        for item in page['rows']:
            row = item['row']
            label = MAPPING.get(row['articleType'])
            if label not in counts or counts[label] >= 2:
                continue
            name = str(row['id']) + '.jpg'
            image = get(row['image']['src'])
            path = IMAGES / name
            path.write_bytes(image)
            with Image.open(path) as original:
                original.verify()
            selected.append({
                'id': row['id'], 'row_index': item['row_idx'],
                'article_type': row['articleType'], 'label_id': label,
                'label': CLASSES[label], 'name': row['productDisplayName'],
                'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                'sha256': hashlib.sha256(image).hexdigest(), 'source': PAGE,
            })
            counts[label] += 1
            print(row['id'], CLASSES[label], flush=True)
        if all(n == 2 for n in counts.values()):
            break
    if not all(n == 2 for n in counts.values()):
        raise RuntimeError(f'Insufficient fixed-order samples: {counts}')
    selected.sort(key=lambda row: (row['label_id'], row['row_index']))
    save(OUT / 'samples.json', {
        'dataset': DATASET, 'source': PAGE,
        'selection': 'For each mapped class take the first two rows in dataset order, before inference.',
        'mapping': MAPPING, 'samples': selected,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    if args.download:
        download()
    manifest = json.loads((OUT / 'samples.json').read_text(encoding='utf-8'))
    models = {variant: load(variant, 17) for variant in ('base', 'se', 'ds', 'gelu', 'gn')}
    records = []
    for item in manifest['samples']:
        path = ROOT / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        with Image.open(path) as image:
            predictions = {v: predict(m, image) for v, m in models.items()}
        records.append(item | {'predictions': predictions})
    summary = {
        'n': len(records), 'source': PAGE,
        'preprocess': 'Unchanged infer.preprocess: grayscale, bilinear 28x28, pad2, normalize.',
        'correct': {
            v: sum(r['predictions'][v][0]['class_id'] == r['label_id'] for r in records)
            for v in models
        }, 'records': records,
    }
    save(OUT / 'predictions_all_raw_seed17.json', summary)
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.8), layout='constrained')
    for ax, row in zip(axes.flat, records):
        with Image.open(ROOT / row['path']) as picture:
            ax.imshow(picture.convert('RGB'))
        ax.axis('off')
        parts = [f"#{row['id']}  实际类别：{row['label']}"]
        for v, name in [('base', '基础'), ('se', 'SE'), ('ds', 'DS'), ('gelu', 'GELU'), ('gn', 'GN')]:
            prediction = row['predictions'][v][0]
            parts.append(f"{name}：{prediction['label']} {prediction['probability']*100:.1f}%")
        ax.set_title('\n'.join(parts), fontsize=10, pad=5)
    finish(fig, 'scene_all_raw.png')
    print(json.dumps({'n': summary['n'], 'correct': summary['correct'],
                      'files': [r['path'] for r in records]}, ensure_ascii=False))


if __name__ == '__main__':
    main()

