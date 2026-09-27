import os
import random

# 严格对齐你半监督定义的映射表
CLASS_TO_INDEX = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3,
    'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 'Grape___healthy': 14,
    'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20,
    'Potato___Late_blight': 21, 'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24,
    'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27,
    'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 'Tomato___Late_blight': 30,
    'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32,
    'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___healthy': 37
}

RAW_DATA_DIR = r"E:\PyCharm\Semi_Supervised_Plant_Classification\plantvillage dataset\color"
OUTPUT_FILE = r"E:\PyCharm\Semi_Supervised_Plant_Classification\data\label_all.txt"


def main():
    all_data = []
    print("正在收集绝对路径...")
    for class_name, label_idx in CLASS_TO_INDEX.items():
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.exists(class_path): continue
        for f in os.listdir(class_path):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                abs_path = os.path.join(class_path, f)
                all_data.append(f"{abs_path} {label_idx}\n")

    random.seed(42)
    random.shuffle(all_data)  # 关键：全随机打乱

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(all_data)
    print(f"已生成打乱后的绝对路径标签文件，共 {len(all_data)} 张图片。")


if __name__ == '__main__':
    main()