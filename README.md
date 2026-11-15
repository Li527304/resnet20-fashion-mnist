# ResNet20服装分类与结构修改实验

以完整ResNet20为基线，在Fashion-MNIST上对照SE通道注意力、深度可分离卷积（DS）、GELU激活与GroupNorm归一化。每组运行三个随机种子。原配置见[实验方案](实验方案.md)，补充组沿用相同的数据划分、训练轮数、批量和优化设置。

## 安装

建议Python3.10。进入项目目录后创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==2.2.6 matplotlib==3.10.9 Pillow==12.3.0 opencv-python==4.10.0.84
```

无CUDA设备时，安装PyTorch官方CPU版本即可运行，训练耗时会增加。报告生成额外需要python-docx；仅训练和演示不需要它。

## 完整运行

```powershell
python -u train.py
python analyze.py
python verify.py
python -u train.py --variants ds gelu gn --epochs 15 --seeds 17 29 43 --batch-size 256 --results-dir results/structure
python verify.py --results-dir results/structure
python compare_structure.py
python app.py
```

首次运行自动从数据集官方GitHub下载四个压缩文件，并由Torchvision检查MD5。网页只监听本机127.0.0.1，浏览器地址为http://127.0.0.1:8765。可切换原始ResNet20和SE-ResNet20、抽取测试图、上传图片并查看实时分类概率与实验图表。新增三组也可通过命令行识别。关闭服务按Ctrl+C。

本次实测使用F盘已有环境，可直接执行：

```powershell
& 'F:\深度学习实验\课内实验报告\经典视觉算法\_公共资源\环境\vision\python.exe' app.py
```

命令行识别：

```powershell
python infer.py figures/sample.png --variant ds --seed 17 --output results/structure/inference_ds.json
```

自选图片统一转灰度、双线性缩放到28×28、四周补2像素，再按固定均值0.2860和标准差0.3530归一化。上传图不会自动裁剪或反色，应准备黑底、居中的单件衣物。演示使用seed17检查点，未按测试精度选择种子。

## 结果文件

- `results/config.json`：完整训练配置；`split.json`：训练与验证索引。
- `results/{模型}_{种子}/history.csv`：逐轮损失和准确率。
- `results/{模型}_{种子}/metrics.json`：测试精度、F1、复杂度、延迟和检查点哈希。
- `results/{模型}_{种子}/best.pt`：仅由验证集选择的检查点。
- `results/{模型}_{种子}/predictions.npz`：每张测试图的logits、标签和混淆矩阵。
- `results/summary.json`、`all_metrics.csv`：三次运行汇总。
- `results/verification.json`：重新加载六个检查点后的独立核验。
- results/structure/：补充实验的配置、九份权重、训练记录、核验结果和五组比较汇总。
- figures/：直接读取结果生成的学习曲线、开销对比、混淆矩阵和预测样例，structure_comparison.png为五组结构对照图。

训练脚本会跳过已有`metrics.json`的完整运行。重新从零训练时，应将当前项目复制到新的目录，并保留源码、移走新副本中的results后运行；不要混用新旧配置。验证脚本只重做推理，不会重新训练。不同硬件和软件版本可能产生数值差异，提交包中的日志对应报告实测结果。

参数量统计全部可训练参数。MACs仅统计卷积和线性层，单次乘加记1MAC，近似FLOPs为2倍MACs。五组延迟由compare_structure.py在同一会话复测，使用官方测试集首图、GPU批量1，每个检查点预热50次、测量100次并逐次同步。取各检查点中位数，再按组取平均。复测依赖CUDA。

## 提交与代码仓库

代码仓库：https://github.com/Li527304/resnet20-fashion-mnist

当前源码、训练记录与报告在本项目目录。数据集和权重由Git忽略（`.gitignore` 已排除 `data/`、`results/`、`figures/`、`scene_images/` 及权重文件）。仓库只包含源码与运行说明；训练数据会在首次运行 `train.py` 时自动下载，权重需运行 `train.py` 生成，或从完整交付包中把各 `best.pt` 复制到对应 `results/{模型}_{种子}/` 目录。

在本目录推送到该仓库：

```powershell
git remote add origin https://github.com/Li527304/resnet20-fashion-mnist.git
git push -u origin main
```

如果远程已有提交，应先检查并合并，勿强制推送。本仓库附`submission.bundle`时，可用`git clone submission.bundle cnn-coursework`恢复版本记录，再将完整包中的`results/*/best.pt`和`data/`复制到同名目录，立即运行演示；也可自行重新训练得到权重。

## 参考资料

- ResNet原论文：https://arxiv.org/abs/1512.03385
- SE原论文：https://arxiv.org/abs/1709.01507
- Fashion-MNIST数据集与公开基线：https://github.com/zalandoresearch/fashion-mnist

本作业的90%为预设验收目标。与公开基线比较时需要考虑训练划分、网络与优化设置差异，不将本次短周期训练表述为原论文复现。

## 新衣物归类场景

scene_images中有8张固定商品图，来自公开的fashion-product-images-small。来源、商品ID、原标签映射与SHA256记录在results/scene/samples.json。按数据顺序选取四类各前两张图片，在预测之前固定选样。

先运行python scene_apply.py，保存五组模型在原处理流程下的预测。再运行python scene_compare.py，先保存全部15份权重的原图结果，记录输入分析，然后测试固定的RGB反色处理。原图、反色结果分别记录在raw_all_seeds.json和inverted_all_seeds.json，all_predictions.csv保存全部逐图结果。

本地界面的“新衣物样例”按钮可以逐张读取这8张图，选择五种模型之一，并切换原处理流程或背景反色。结果中仍有较多错分，实际使用时需要人工确认。反色只是输入适配，不调整模型权重。对这8张图的复测用于观察问题，不作为新测试集上的泛化精度。

图片来源 https://huggingface.co/datasets/ashraq/fashion-product-images-small

## 人工框选与去背景复测

运行python scene_crop.py，重现固定8张图片的人工裁剪与GrabCut去背景实验。人工框的原图像素坐标在results/scene/crop_protocol.json，预测前固定。先比较裁剪、反色、保持宽高比和居中，再比较增加GrabCut后的变化。两组都沿用已有15份模型权重。逐图结果分别在crop_fit_all_seeds.json和crop_grabcut_all_seeds.json，全部阶段汇总于crop_summary.json。

安装要求新增opencv-python==4.10.0.84。本地演示可在图片上拖动选框，也可输入左、上、右、下四个整数。新衣物样例自动填入已经记录的人工坐标。界面提供裁剪居中及GrabCut去背景两种方式。框的位置由使用者给出。8张已用于分析的图片上的结果属于探索性复测，后续应另外准备新图片验证。

GrabCut实现参考OpenCV官方教程 https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html
