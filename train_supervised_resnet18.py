import os
import torch
import torch.nn as nn
import time
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib as mpl



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE=torch.device('cpu')

print(f"启动全监督训练 | 当前设备: {DEVICE}")

suffix="_20%"

PERCENT = 0.20
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
INPUT_SIZE = 96
LEARNING_RATE = 3e-5
EPOCHS = 30

LABEL_PATH = r"E:\PyCharm\Semi_Supervised_Plant_Classification\data\label_all.txt"
SAVE_DIR = './results'
os.makedirs(f'{SAVE_DIR}/models', exist_ok=True)
os.makedirs(f'{SAVE_DIR}/images', exist_ok=True)
os.makedirs(f'{SAVE_DIR}/logs', exist_ok=True)
LOG_FILE = f"{SAVE_DIR}/logs/supervised_all{suffix}.csv"



# 数据处理逻辑
class PlantDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, label
        except:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), label

def main():

    # 准备数据列表
    all_raw_data = []
    with open(LABEL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().rsplit(' ', 1)
            if len(parts) == 2: all_raw_data.append((parts[0], int(parts[1])))

    random.shuffle(all_raw_data)


    total_count = len(all_raw_data)
    train_count = int(total_count * PERCENT)
    test_count = int(total_count * 0.20)

    train_paths = all_raw_data[:train_count]
    test_paths = all_raw_data[total_count - test_count:]

    print(f"训练集({int(PERCENT * 100)}%): {len(train_paths)} | 测试集: {len(test_paths)}")

    # 数据增强与预处理
    train_trans = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_trans = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(PlantDataset(train_paths, train_trans), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(PlantDataset(test_paths, test_trans), batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 38)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'loss': [], 'acc': [], 'f1': [], 'precision': [], 'recall': [], 'time': []}
    best_acc = 0.0
    start_time = time.time()

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("轮次,累计耗时/秒,监督损失,测试准确率,精确率,召回率,F1分数\n")

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for i, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # 梯度累积
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix(loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}")

        # 验证逻辑
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                out = model(imgs.to(DEVICE))
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        # 计算详细指标
        avg_loss = running_loss / len(train_loader)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        current_time = time.time() - start_time
        mins, secs = divmod(int(current_time), 60)

        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        history['time'].append(current_time)

        # 打印日志
        print(f"轮次 {epoch + 1} | 耗时: {mins}m{secs}s | 损失: {avg_loss:.4f} | 准确率: {acc:.4f} | F1: {f1:.4f}")

        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{epoch + 1},{current_time:.2f},{avg_loss:.6f},{acc:.6f},{precision:.6f},{recall:.6f},{f1:.6f}\n")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/models/supervised_best{suffix}.pth")

        #  绘图
        plt.figure(figsize=(12, 10))

        # 训练损失
        plt.subplot(2, 2, 1)
        plt.plot(history['loss'], marker='o', markersize=4, linestyle='-', color='red', label='Loss')
        plt.title('训练损失 (Loss)')
        plt.xlabel('轮次 (Epochs)')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)

        # 测试准确率
        plt.subplot(2, 2, 2)
        plt.plot(history['acc'], marker='o', markersize=4, linestyle='-', color='blue', label='Accuracy')
        plt.title('测试准确率 (Accuracy)')
        plt.xlabel('轮次 (Epochs)')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6)

        # 综合指标 (F1, Precision, Recall)
        plt.subplot(2, 2, 3)
        plt.plot(history['f1'], marker='o', markersize=4, color='orange', label='F1-Score')
        plt.plot(history['precision'], marker='s', markersize=3, color='green', alpha=0.6, label='Precision')
        plt.plot(history['recall'], marker='^', markersize=3, color='purple', alpha=0.6, label='Recall')
        plt.title('详细评估指标 (Metrics)')
        plt.xlabel('轮次 (Epochs)')
        plt.ylabel('Score')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)

        # 累计耗时
        plt.subplot(2, 2, 4)
        plt.plot(history['time'], marker='x', markersize=5, linestyle='-', color='teal', label='Time')
        plt.title('累计耗时 (Time / s)')
        plt.xlabel('轮次 (Epochs)')
        plt.ylabel('Seconds')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/images/supervised_curves{suffix}.png")
        plt.close()

        scheduler.step()

    print(f"最优准确率: {best_acc:.4f} | 日志已保存至: {LOG_FILE}")


if __name__ == '__main__':
    main()