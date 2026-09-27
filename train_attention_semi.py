import os
import torch
import torch.nn as nn
import time
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
import gc

# ---------------------- 1. 基础配置 ----------------------
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'

# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE= torch.device('cpu')
print(f"当前使用设备：{DEVICE}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ---------------------- 2. 核心超参数 ----------------------


suffix = "20%"

os.makedirs('./results/models', exist_ok=True)
os.makedirs('./results/logs', exist_ok=True)

BATCH_SIZE = 8 if torch.cuda.is_available() else 1
EPOCHS = 30  # **大幅减少epochs，因为是微调**
LEARNING_RATE = 3e-5  # **更低的学习率用于微调**
NUM_CLASSES = 38
LAMBDA_U = 2.0
THRESHOLD = 0.90
INPUT_SIZE = 96

LABELED_TXT = f'./data/labeled_list_{suffix}标注.txt'
UNLABELED_TXT = f'./data/unlabeled_list_{suffix}标注.txt'

SAVE_MODEL_PATH = f'./results/models/semi_attention_{suffix}_best.pth'
LOG_PATH = f'./results/logs/semi_attention_{suffix}.txt'
# 数据增强suffix
weak_aug = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

strong_aug = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_aug = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# ---------------------- 3. 数据集定义（保持原样） ----------------------
class LabeledDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.data = []
        self.transform = transform
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"标注文件不存在：{txt_path}")

        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            valid_lines = 0
            invalid_lines = 0
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    invalid_lines += 1
                    continue
                parts = line.rsplit(' ', 1)
                if len(parts) != 2:
                    invalid_lines += 1
                    print(f"跳过格式错误行（{idx + 1}）：{line}")
                    continue
                img_path = parts[0].strip()
                label_str = parts[1].strip()

                try:
                    label = int(label_str)
                except ValueError:
                    invalid_lines += 1
                    print(f"跳过标签非数字行（{idx + 1}）：标签={label_str}")
                    continue

                img_path = os.path.normpath(img_path)
                full_img_path = os.path.abspath(os.path.join('./data', img_path))
                if os.path.exists(full_img_path):
                    img_path = full_img_path
                elif not os.path.exists(img_path):
                    invalid_lines += 1
                    print(f"跳过路径不存在行（{idx + 1}）：{img_path}")
                    continue

                self.data.append((img_path, label))
                valid_lines += 1

        if len(self.data) == 0:
            raise ValueError(f"标注文件 {txt_path} 解析后无有效样本！")
        print(f"标注文件加载完成：有效样本{valid_lines} | 无效样本{invalid_lines}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"图片加载失败，使用空白图：{img_path} | 错误：{e}")
            img = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE))

        if self.transform is not None:
            img = self.transform(img)
        return img, label


class UnlabeledDataset(Dataset):
    def __init__(self, txt_path, weak_transform, strong_transform):
        self.data = []
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"无标注文件不存在：{txt_path}")

        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            print(f"\n=== 无标签文件读取 ===")
            print(f"文件总行数：{len(lines)}")

            valid_lines = 0
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                line = os.path.normpath(line)
                full_line = os.path.abspath(os.path.join('./data', line))
                if os.path.exists(full_line):
                    line = full_line
                elif not os.path.exists(line):
                    print(f"跳过路径不存在行（{idx + 1}）：{line}")
                    continue

                self.data.append(line)
                valid_lines += 1

            print(f"成功加载无标签数据量：{len(self.data)}")

        if len(self.data) == 0:
            raise ValueError(f"无标注文件 {txt_path} 解析后无有效样本！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"图片加载失败，使用空白图：{img_path} | 错误：{e}")
            img = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE))

        img_weak = self.weak_transform(img)
        img_strong = self.strong_transform(img)
        return img_weak, img_strong


# ---------------------- 4. **预训练模型 + Attention微调** ----------------------
class AttentionPoolingHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=38, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels

        # **轻量级注意力机制**
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # 确保注意力权重在[0,1]之间
        )

        # **分类头**
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        """
        x: [B, 512, 7, 7]
        """
        # 计算注意力权重
        attention_weights = self.attention(x)  # [B, 1, 7, 7]

        # 加权特征
        weighted_x = x * attention_weights  # [B, 512, 7, 7]

        # 分类
        output = self.classifier(weighted_x)  # [B, 38]
        return output


class PretrainedResNet18WithAttention(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        # **加载预训练模型**
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # **保留完整的backbone特征提取器**
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # **冻结大部分层，只微调最后几层**
        for param in self.backbone[4].parameters():  # layer1
            param.requires_grad = False
        for param in self.backbone[5].parameters():  # layer2
            param.requires_grad = False
        for param in self.backbone[6].parameters():  # layer3
            param.requires_grad = False
        # **只微调layer4和attention head**

        # **替换原有的分类层为新的attention head**
        self.head = AttentionPoolingHead(
            in_channels=512,
            num_classes=num_classes,
            dropout=dropout_rate
        )

    def forward(self, x):
        x = self.backbone(x)  # [B, 3, 224, 224] -> [B, 512, 7, 7]
        x = self.head(x)  # [B, 512, 7, 7] -> [B, 38]
        return x


def build_pretrained_resnet18_with_attention(num_classes, dropout_rate=0.1):
    model = PretrainedResNet18WithAttention(
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    return model.to(DEVICE)


# ---------------------- 5. 数据加载（保持原样） ----------------------
try:
    full_labeled_dataset = LabeledDataset(LABELED_TXT, transform=None)
    total_labeled = len(full_labeled_dataset)
    print(f"\n=== 数据拆分 ===")
    print(f"总有标签数据量：{total_labeled}")

    # 固定随机种子分割
    np.random.seed(42)
    indices = np.arange(total_labeled)
    np.random.shuffle(indices)

    train_size = int(0.8 * total_labeled)
    test_size = total_labeled - train_size
    print(f"训练集：{train_size} | 测试集：{test_size}")

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_labeled_subset = torch.utils.data.Subset(full_labeled_dataset, train_indices)
    train_labeled_subset.dataset.transform = weak_aug
    test_labeled_subset = torch.utils.data.Subset(full_labeled_dataset, test_indices)
    test_labeled_subset.dataset.transform = test_aug

    unlabeled_dataset = UnlabeledDataset(UNLABELED_TXT, weak_aug, strong_aug)

    train_labeled_loader = DataLoader(
        train_labeled_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_labeled_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )

    print(f"\n=== 数据加载器 ===")
    print(f"有标签训练集batch数：{len(train_labeled_loader)}")
    print(f"测试集batch数：{len(test_loader)}")
    print(f"无标签数据集batch数：{len(unlabeled_loader)}")

except Exception as e:
    print(f"\n 数据加载失败：{e}")
    exit(1)


# ---------------------- 6. 损失计算（保持原样） ----------------------
def calculate_fixmatch_loss(model, imgs_sup, labels, imgs_weak, imgs_strong, lambda_u, threshold):
    imgs_sup = imgs_sup.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)
    imgs_weak = imgs_weak.to(DEVICE, non_blocking=True)
    imgs_strong = imgs_strong.to(DEVICE, non_blocking=True)

    logits_sup = model(imgs_sup)
    sup_loss = F.cross_entropy(logits_sup, labels, reduction='mean')

    model.eval()
    with torch.no_grad():
        logits_weak = model(imgs_weak)
        pseudo_probs = F.softmax(logits_weak / 0.5, dim=1)
        max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
        mask = max_probs.ge(threshold).float()

    model.train()
    logits_strong = model(imgs_strong)
    strong_loss = F.cross_entropy(logits_strong, pseudo_labels, reduction='none')
    cons_loss = (strong_loss * mask).sum() / (mask.sum() + 1e-8)

    if torch.isnan(cons_loss) or torch.isinf(cons_loss):
        cons_loss = torch.tensor(0.0, device=DEVICE)

    total_loss = sup_loss + lambda_u * cons_loss

    return total_loss, sup_loss, cons_loss


def main():
    model = build_pretrained_resnet18_with_attention(NUM_CLASSES, dropout_rate=0.1)

    # 针对微调的优化器设置
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': model.backbone[7].parameters(), 'lr': LEARNING_RATE},
        {'params': model.head.parameters(), 'lr': LEARNING_RATE * 5}
    ], lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 用于存储指标的列表
    train_loss_list = []
    sup_loss_list = []
    cons_loss_list = []
    test_acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    # 记录累计总耗时
    total_time_list = []

    best_test_acc = 0.0

    # 记录训练开始的绝对时间
    training_start_time = time.time()

    # 准备日志文件
    LOG_DIR = './results/logs'
    IMG_DIR = './results/images'
    os.makedirs(IMG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'semi_attention_{suffix}.csv')

    #写入 CSV 表头
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("轮次,累计耗时/秒,总损失,监督损失,一致性损失,测试准确率,精确率,召回率,F1分数\n")

    print(f"\n 开始训练，指标将记录在 {log_path} 和 {IMG_DIR} 中")

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_loss_epoch = 0.0
        sup_loss_epoch = 0.0
        cons_loss_epoch = 0.0
        step_count = 0

        num_steps = len(train_labeled_loader)
        labeled_iter = iter(train_labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        pbar = tqdm(range(num_steps), desc=f'Epoch {epoch + 1}/{EPOCHS}')


        for step_idx in pbar:
            step_count += 1
            try:
                imgs_weak, imgs_strong = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                imgs_weak, imgs_strong = next(unlabeled_iter)
            try:
                imgs_sup, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(train_labeled_loader)
                imgs_sup, labels = next(labeled_iter)
            try:
                loss, sup_loss, cons_loss = calculate_fixmatch_loss(
                    model, imgs_sup, labels, imgs_weak, imgs_strong,
                    lambda_u=LAMBDA_U, threshold=THRESHOLD
                )
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_loss_epoch += loss.item()
                sup_loss_epoch += sup_loss.item()
                cons_loss_epoch += cons_loss.item()
            except Exception as e:
                print(f"\n {step_idx} 错误: {e}")
                continue

        # 计算 Epoch 平均损失
        avg_total_loss = total_loss_epoch / step_count if step_count > 0 else 0
        avg_sup_loss = sup_loss_epoch / step_count if step_count > 0 else 0
        avg_cons_loss = cons_loss_epoch / step_count if step_count > 0 else 0

        # 将当前Epoch的损失加入列表
        train_loss_list.append(avg_total_loss)
        sup_loss_list.append(avg_sup_loss)
        cons_loss_list.append(avg_cons_loss)

        # 测试阶段
        model.eval()
        all_preds = []
        all_labels = []
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = test_correct / test_total if test_total > 0 else 0

        if len(all_labels) > 0:
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        else:
            precision, recall, f1 = 0, 0, 0

        test_acc_list.append(test_acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # 计算总耗时并保存指标到CSV
        current_total_seconds = time.time() - training_start_time
        total_time_list.append(current_total_seconds)

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch + 1},{current_total_seconds:.2f},"
                    f"{avg_total_loss:.6f},{avg_sup_loss:.6f},{avg_cons_loss:.6f},"
                    f"{test_acc:.6f},{precision:.6f},{recall:.6f},{f1:.6f}\n")

        # 模型保存逻辑
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if os.path.exists(SAVE_MODEL_PATH):
                os.remove(SAVE_MODEL_PATH)
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"\n最新最佳模型保存: {best_test_acc:.4f}")

        # 绘图逻辑
        plt.figure(figsize=(12, 8))

        # 子图 1: 损失函数变化
        plt.subplot(2, 2, 1)
        plt.plot(train_loss_list, label='总损失', marker='o')
        plt.plot(sup_loss_list, label='监督损失', marker='s')
        plt.plot(cons_loss_list, label='一致性损失', marker='^')
        plt.title('训练损失曲线')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        # 子图 2: 测试准确率
        plt.subplot(2, 2, 2)
        plt.plot(test_acc_list, label='测试准确率', color='green', marker='o')
        plt.title('测试准确率曲线')
        plt.xlabel('轮次')
        plt.ylabel('精准率')
        plt.grid(True)

        # 子图 3: F1 Score
        plt.subplot(2, 2, 3)
        plt.plot(f1_list, label='F1 分数', color='orange', marker='o')
        plt.title('F1 分数 曲线')
        plt.xlabel('轮次')
        plt.ylabel('F1 分数')
        plt.grid(True)

        # 子图 4: 总耗时曲线
        plt.subplot(2, 2, 4)
        plt.plot(total_time_list, label='累计耗时', color='red', marker='x')
        plt.title('训练累计耗时')
        plt.xlabel('轮次')
        plt.ylabel('秒')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f'semi_attention_{suffix}.png'))
        plt.close()

        #打印当前 Epoch 信息
        #控制台打印中显示总耗时
        mins = int(current_total_seconds) // 60
        secs = int(current_total_seconds) % 60
        print(f"\n轮次 {epoch + 1} | "
              f"总耗时: {mins}m{secs}s | "
              f"损失: {avg_total_loss:.4f} | "
              f"准确率: {test_acc:.4f} | "
              f"F1: {f1:.4f}")

        scheduler.step()

    #训练结束后的最终处理
    print(f"\n 训练完成! 总耗时: {mins}m{secs}s")
    print(f"🏆 最佳准确率: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()