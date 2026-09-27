
import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split

# ==================== 1. 配置参数 ====================
LABELED_RATIO =0.05 # 5% 有标签数据
TEST_RATIO = 0.2  # 20% 测试集数据
RANDOM_SEED = 42

# 🔑 路径配置 (请确认路径正确)
RAW_DATA_DIR = r"E:\PyCharm\Semi_Supervised_Plant_Classification\plantvillage dataset\color"
SAVE_DATA_DIR = r"E:\PyCharm\Semi_Supervised_Plant_Classification\data"

os.makedirs(SAVE_DATA_DIR, exist_ok=True)

# ==================== 2. 核心修复：定义标准标签映射表 ====================

CLASS_TO_INDEX = {
    'Apple___Apple_scab': 0,
    'Apple___Black_rot': 1,
    'Apple___Cedar_apple_rust': 2,
    'Apple___healthy': 3,
    'Blueberry___healthy': 4,
    'Cherry_(including_sour)___Powdery_mildew': 5,
    'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
    'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9,
    'Corn_(maize)___healthy': 10,
    'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12,
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
    'Grape___healthy': 14,
    'Orange___Haunglongbing_(Citrus_greening)': 15,
    'Peach___Bacterial_spot': 16,
    'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18,
    'Pepper,_bell___healthy': 19,
    'Potato___Early_blight': 20,
    'Potato___Late_blight': 21,
    'Potato___healthy': 22,
    'Raspberry___healthy': 23,
    'Soybean___healthy': 24,
    'Squash___Powdery_mildew': 25,
    'Strawberry___Leaf_scorch': 26,
    'Strawberry___healthy': 27,
    'Tomato___Bacterial_spot': 28,
    'Tomato___Early_blight': 29,
    'Tomato___Late_blight': 30,
    'Tomato___Leaf_Mold': 31,
    'Tomato___Septoria_leaf_spot': 32,
    'Tomato___Spider_mites Two-spotted_spider_mite': 33,
    'Tomato___Target_Spot': 34,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
    'Tomato___Tomato_mosaic_virus': 36,
    'Tomato___healthy': 37,
}
# ==================== 3. 数据校验与收集 ====================
print("正在扫描原始数据集...")
data_list = []

if not os.path.exists(RAW_DATA_DIR):
    raise FileNotFoundError(f"错误：原始数据集目录 {RAW_DATA_DIR} 不存在！")

# 遍历类别
for class_name, class_idx in CLASS_TO_INDEX.items():
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        print(f"警告：类别文件夹不存在，跳过: {class_dir}")
        continue

    img_files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in img_files:
        img_path = os.path.join(class_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img.convert('RGB')
            # 使用硬编码的索引
            data_list.append((os.path.abspath(img_path), class_idx))
        except Exception as e:
            print(f"无效图片：{img_path} | 错误：{str(e)}")

print(f"共获取有效图片 {len(data_list)} 张")

if len(data_list) == 0:
    raise ValueError("未获取到任何有效图片，请检查图片路径和格式！")

# ==================== 4. 严格三路拆分 ====================
print("\n正在进行数据集拆分...")

# 第一步：切分出测试集 (20%)
temp_list, test_list = train_test_split(
    data_list,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED,
    stratify=[item[1] for item in data_list]  # 按标签分层
)

print(f"测试集 (Test)：{len(test_list)} 张")

# 第二步：拆分有标签和无标签
target_ratio = LABELED_RATIO / (1 - TEST_RATIO)
labeled_list, unlabeled_list = train_test_split(
    temp_list,
    train_size=target_ratio,
    random_state=RANDOM_SEED,
    stratify=[item[1] for item in temp_list]
)

print(f" 有标签训练集 (Labeled)：{len(labeled_list)} 张")
print(f" 无标签数据 (Unlabeled)：{len(unlabeled_list)} 张")


# ==================== 5. 写入文件 ====================
def write_list_to_file(data_list, file_name, is_unlabeled=False):
    base_name = file_name.replace("_labeled", "").replace("_unlabeled", "").replace("_test", "")
    file_path = os.path.join(SAVE_DATA_DIR, f"{base_name}_{int(LABELED_RATIO * 100)}%标注.txt")

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            if is_unlabeled:
                f.write(f"{item[0]}\n")
            else:
                f.write(f"{item[0]} {item[1]}\n")
    print(f" 已保存：{file_path}")


# 写入三个文件
write_list_to_file(labeled_list, "labeled_list", is_unlabeled=False)
write_list_to_file(unlabeled_list, "unlabeled_list", is_unlabeled=True)
write_list_to_file(test_list, "test_list", is_unlabeled=False)

print("\n数据集拆分完成！")
