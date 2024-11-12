# Dataloader
Tools and frame for processing SIDD and DND datasets

```
project/
├── config/                  # 配置文件目录
│   ├── config.yaml          # 主配置文件
│   └── experiment_config.yaml # 不同实验的配置文件
├── data/                    # 数据相关模块
│   ├── __init__.py          # 数据加载模块的初始化
│   ├── data_loader.py       # 数据加载和预处理
│   └── augmentations.py     # 数据增强
├── models/                  # 模型定义模块
│   ├── __init__.py          # 模型模块的初始化
│   ├── model_architecture.py # 神经网络模型定义
│   └── loss_functions.py    # 损失函数定义
├── trainers/                 # 训练模块
│   ├── __init__.py          # 训练模块的初始化
│   ├── trainer.py           # 训练过程管理，定义训练循环
│   └── evaluator.py         # 模型评估
├── utils/                   # 工具模块
│   ├── __init__.py          # 工具模块的初始化
│   ├── logger.py            # 日志处理，结合TensorBoard
│   ├── config_parser.py     # 配置文件解析
│   └── metrics.py           # 评估指标
├── inference/               # 推理模块
│   ├── __init__.py          # 推理模块的初始化
│   └── inference.py         # 推理和结果保存
├── experiments/             # 实验结果存储目录
│   ├── logs/                # 日志文件夹（存放TensorBoard日志）
│   └── checkpoints/         # 保存模型的checkpoint文件夹
└── README.md                # 项目说明文件
```
