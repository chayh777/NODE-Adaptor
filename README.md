# NODE-Adapter: Neural ODE for Vision-Language Reasoning

本项目实现了论文《NODE-Adapter: Neural Ordinary Differential Equations for Better Vision-Language Reasoning》的核心逻辑，应用于少样本图像分类任务。

## 项目结构

```
NODE-Adaptor/
├── configs/
│   └── config.yaml           # 配置文件
├── data/
│   ├── prompts.py            # M=20 提示模板 + CIFAR-100 加载
│   └── cifar-100-python/     # CIFAR-100 数据集 (自动下载)
├── models/
│   ├── prototype_builder.py  # CrossModalPrototype 类
│   ├── ode_function.py       # ODEFunc 类 (Neural ODE 梯度估计器)
│   └── node_adapter.py       # NODE-Adapter 主模型
├── data_loader/
│   └── fewshot_loader.py     # 少样本数据加载器
├── utils/
│   ├── metrics.py            # 评估指标
│   └── visualization.py      # 可视化模块 (实验日志、图表)
├── results/                  # 实验结果 (图表、报告)
├── train.py                  # 训练脚本
├── eval.py                   # 测试脚本
└── requirements.txt          # 依赖列表
```

## 核心算法

### A. 跨模态原型构建

**Textual Prototype:**
$$P_t = [\bar{t}_1, \bar{t}_2, ..., \bar{t}_N] \in \mathbb{R}^{N \times D}$$

**Visual Prototype:**
$$P_v = [\bar{v}_1, \bar{v}_2, ..., \bar{v}_N] \in \mathbb{R}^{N \times D}$$

**自适应融合:**
$$p_j = \lambda_j \cdot \bar{v}_j + (1 - \lambda_j) \cdot \bar{t}_j$$
$$\lambda_j = \frac{1}{1 + \exp(-\bar{v}_j^{\top} u)}$$

### B. Neural ODE 原型优化

**状态演化方程:**
$$\frac{dp(t)}{dt} = f_{\theta}(p(t), t, S)$$

- 梯度估计器 $f_{\theta}$: 双阶段网络 (时间嵌入 + 8头自注意力)
- 数值求解器: 伴随敏感度方法 (O(1) 内存开销)
- 积分步数: T=30

## 环境配置

```bash
pip install -r requirements.txt
```

或手动安装:
```bash
pip install torch torchvision torchdiffeq transformers clip numpy Pillow tqdm pyyaml
```

## 快速开始

### 训练

```bash
python train.py --num_ways 5 --num_shots 5 --epochs 10
```

参数说明:
- `--num_ways`: 每轮任务的类别数 (默认 5)
- `--num_shots`: 每类支持图像数 (默认 5)
- `--epochs`: 训练轮数 (默认 10)
- `--num_tasks`: 每轮任务数 (默认 100)
- `--num_eval_tasks`: 评估任务数 (默认 1000)

### 评估

```bash
python eval.py --num_ways 5 --num_shots 5 --num_eval_tasks 1000
```

## 可视化

运行评估后会自动生成可视化报告：

```bash
python eval.py --num_ways 20 --num_shots 5 --num_eval_tasks 1000
```

生成内容：
- `results/plots/{exp_name}/accuracy_comparison.png` - 不同 shot 次数对比图
- `results/plots/{exp_name}/confidence_distribution.png` - 预测置信度分布
- `results/plots/{exp_name}/experiment_report.md` - 实验报告 (Markdown)

## 参考文献

Zhang, Y., Cheng, C. W., Yu, K., He, Z., Schönlieb, C. B., & Aviles-Rivero, A. I. (2024). NODE-Adapter: Neural Ordinary Differential Equations for Better Vision-Language Reasoning.

## 依赖版本

- torch >= 1.12.0
- torchvision >= 0.13.0
- torchdiffeq >= 0.2.3
- transformers >= 4.25.0
- clip (openai)
- numpy >= 1.21.0
- Pillow >= 9.0.0
- tqdm >= 4.60.0
- pyyaml >= 5.4.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
