# ResNet20 服装分类与结构改进

在 Fashion-MNIST 上训练 ResNet20，再分别换成 SE 通道注意力、深度可分离卷积、GELU 激活和 GroupNorm 归一化作对比，每组跑 17、29、43 三个随机种子。

代码仓库：https://github.com/Li527304/resnet20-fashion-mnist

## 环境

Python 3.10，显卡是 RTX 4060 Laptop，PyTorch 2.5.1 + cu121。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==2.2.6 matplotlib==3.10.9 Pillow==12.3.0 opencv-python==4.10.0.84
```

没有显卡就装 CPU 版 PyTorch，代码不用改，只是训练慢一些。

## 训练和验证

```powershell
python -u train.py
python -u train.py --variants ds gelu gn --results-dir results/structure
python verify.py
python verify.py --results-dir results/structure
```

第一次运行会自动下载 Fashion-MNIST，并由 torchvision 校验 MD5。已经有 `metrics.json` 的目录会跳过，想重新训练就换一个 results 目录，不要新旧配置混着用。`verify.py` 只重做推理，不会重新训练。

## 识别和演示

```powershell
python infer.py figures/sample.png --variant se --seed 17
python app.py
```

网页只监听 127.0.0.1:8765，可以切换五种模型、抽测试图、上传图片，也能在图上框选后裁剪或去背景。上传的图不会自动裁剪，最好用黑底、居中的单件衣物。图片统一转灰度、缩到 28×28、四周补 2 像素到 32×32，再按 0.2860 和 0.3530 归一化。

## 结果文件

- `results/{模型}_{种子}/`：`history.csv` 逐轮损失和准确率，`metrics.json` 精度、F1、参数量、MACs、延迟和权重哈希，`best.pt` 是按验证集选出的权重，`predictions.npz` 存测试集的 logits、标签和混淆矩阵。
- `results/summary.json`、`all_metrics.csv`：三个种子的汇总。
- `results/verification.json`：重新推理后的核对结果。
- `figures/`：直接读结果画的学习曲线、开销对比、混淆矩阵和样例。

参数量统计的是全部可训练参数。MACs 只算卷积和线性层，一次乘加记 1 MAC，近似 FLOPs 是它的两倍。延迟在同一会话里用 GPU 批量 1 测，每个权重预热 50 次、测 100 次，取中位数再按组平均，这一步需要 CUDA。

## 场景图片

`scene_images` 里的 8 张商品图来自 fashion-product-images-small，来源、商品 ID 和哈希记在 `results/scene/samples.json`，图片是在预测之前按数据顺序固定选好的。

```powershell
python scene_apply.py
python scene_compare.py
python scene_crop.py
```

`scene_apply.py` 跑原来的处理流程，`scene_compare.py` 加整图反色，`scene_crop.py` 做人工框选、裁剪居中和 GrabCut 去背景。8 张图上的错分还是不少，这些结果只用来看问题，不能当成新数据集上的精度。图片来源 https://huggingface.co/datasets/ashraq/fashion-product-images-small

## 参考

- ResNet：https://arxiv.org/abs/1512.03385
- SE：https://arxiv.org/abs/1709.01507
- Fashion-MNIST：https://github.com/zalandoresearch/fashion-mnist
