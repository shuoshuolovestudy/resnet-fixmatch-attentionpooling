import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_ROOT = os.path.join(BASE_DIR, "plantvillage dataset", "color")


suffix = "5%"

# 测试集设置
SAMPLES_PER_CLASS = 30
BATCH_SIZE = 16
NUM_CLASSES = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 权重文件夹路径
MODEL_DIR = os.path.join(BASE_DIR, "results", "models")

# 配置四个模型路径
MODELS_TO_TEST = {
    "Supervised 80% (性能上限)": {
        "path": os.path.join(MODEL_DIR, "supervised_best.pth"),
        "type": "standard"
    },
    f"Supervised {suffix} (对比基准)": {
        "path": os.path.join(MODEL_DIR, f"supervised_best_{suffix}.pth"),
        "type": "standard"
    },
    f"Pure Semi {suffix} (纯半监督)": {
        "path": os.path.join(MODEL_DIR, f"semi_pure_{suffix}_best.pth"),
        "type": "standard"
    },
    f"Attention Semi {suffix} (注意力半监督)": {
        "path": os.path.join(MODEL_DIR, f"semi_attention_{suffix}_best.pth"),
        "type": "attention"
    }
}

# 结果保存路径
RESULTS_CSV = os.path.join(BASE_DIR, "results", "comparison", f"comparison_results_{suffix}.csv")


# 网络结构定义

class AttentionPoolingHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=38, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        weighted_x = x * attention_weights
        output = self.classifier(weighted_x)
        return output


class ExactMatchModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = resnet18(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.head = AttentionPoolingHead(in_channels=512, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def build_model(model_type):
    if model_type == "attention":
        return ExactMatchModel(NUM_CLASSES).to(DEVICE)
    else:
        # 普通模型也建议统一结构，或者确保权重 Key 能对上
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, NUM_CLASSES)
        return model.to(DEVICE)




#数据集定义
class SampledDataset(Dataset):
    def __init__(self, root_dir, samples_per_class=10, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        if not os.path.exists(root_dir): return
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        random.seed(42)
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            all_imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            sampled_imgs = random.sample(all_imgs, min(samples_per_class, len(all_imgs)))
            for img_name in sampled_imgs:
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]


# 执行测试
def main():
    print(f"启动毕设对比实验测试 (后缀: {suffix}) | 设备: {DEVICE}")

    test_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SampledDataset(TEST_DATA_ROOT, samples_per_class=SAMPLES_PER_CLASS, transform=test_transform)
    if len(dataset) == 0:
        print("错误：未找到测试数据，请检查 TEST_DATA_ROOT。")
        return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    metrics_list = []

    for name, cfg in MODELS_TO_TEST.items():
        print(f"\n[测试中] {name}...")
        if not os.path.exists(cfg["path"]):
            print(f"跳过：找不到权重文件 {os.path.basename(cfg['path'])}")
            continue

        model = build_model(cfg["type"])

        # 加载权重
        sd = torch.load(cfg["path"], map_location=DEVICE)

        # 1. 修复 DataParallel 的 module. 前缀
        if list(sd.keys())[0].startswith('module.'):
            sd = {k[7:]: v for k, v in sd.items()}

        # 2. 尝试加载
        try:
            model.load_state_dict(sd)
        except RuntimeError as e:
            print(f"结构自动匹配失败，尝试强制兼容模式...")
            # 如果是标准模型权重加载到了 Wrapped 结构，或者反之，进行 Key 转换
            new_sd = {}
            for k, v in sd.items():
                nk = k.replace("backbone.", "").replace("head.", "fc.")
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in dataloader:
                outputs = model(imgs.to(DEVICE))
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

        # 计算指标
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)


        metrics_list.append({
            "实验组别": name,
            "Accuracy": f"{acc:.4f}",
            "F1-Score": f"{f1:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}"
        })
        print(f"测试完成: Acc={acc:.4f}")

    # 汇总展示
    if not metrics_list:
        print("没有成功测试任何模型，请检查权重文件路径。")
        return

    df = pd.DataFrame(metrics_list)
    print("\n" + "=" * 50)
    print(f"最终对比实验汇总表 (比例: {suffix})")
    print("=" * 50)
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存至: {RESULTS_CSV}")


if __name__ == '__main__':
    main()