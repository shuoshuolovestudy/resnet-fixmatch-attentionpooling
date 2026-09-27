# 基于半监督学习的植物病害识别系统

## 项目简介

本科毕业设计。在 ResNet-18 上引入 **Attention Pooling** 抑制土壤、光照等背景干扰，
结合 **FixMatch** 半监督框架降低标注依赖，在仅使用 **5% 标注数据** 的条件下，
将测试准确率从监督基线的 81.75% 提升至 88.16%。

> 郑州轻工业大学 | 软件工程（新型平台软件）22-03 | 2026 届

---

## 核心方法

| 模块 | 说明 |
|------|------|
| **FixMatch** | 弱增强生成伪标签，强增强下强制一致性约束 |
| **Attention Pooling** | 替换 ResNet-18 顶层全局平均池化，通过 1×1 卷积学习空间权重，聚焦病斑区域 |
| **损失函数** | 监督损失 + λ · 无监督损失 |

## 实验结果

PlantVillage 数据集上的消融对比（5% / 10% / 20% 三档标注比例）：

| 标注比例 | 监督基线 | 纯半监督 | 注意力半监督 |
|---|---|---|---|
| 5% | 81.75% | 85.53% | **88.16%** |
| 10% | 86.05% | 90.46% | **94.30%** |
| 20% | 90.87% | 93.10% | **95.71%** |
| 80%（性能上限） | 99.91% | - | - |

两点结论：注意力机制在每一档标注比例上都带来稳定提升；标注比例越低，半监督的相对收益越明显。
`results/logs/` 中记录的训练过程最优测试准确率为 90.79%（5% 档，第 30 轮）。

- 汇总对比：`results/comparison/`
- 逐轮日志：`results/logs/`
- 训练曲线：`results/images/`

## 代码结构

| 文件 | 作用 |
|---|---|
| `train_supervised_resnet18.py` | 监督学习基线 |
| `train_pure_semi.py` | 纯 FixMatch 半监督 |
| `train_attention_semi.py` | Attention Pooling + FixMatch |
| `split_dataset.py` / `split_all.py` | 数据集划分与标注比例索引生成 |
| `model_test_all.py` | 批量评估已训练模型 |
| `app.py` | Flask Web 诊断平台 |
| `templates/` | Web 页面模板 |
| `results/` | 对比数据、训练日志、训练曲线 |

## 数据集

PlantVillage（公开数据集，约 5.4 万张，38 类植物病害）

- 来源：Kaggle - PlantVillage
- 论文：Hughes & Salathé, arXiv:1511.08060, 2015

## 说明

模型权重（`results/models/`，约 430 MB）与数据集索引（`data/`）体积较大，未纳入仓库。
