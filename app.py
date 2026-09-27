import os
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename


import torch
from torchvision import transforms
from PIL import Image

from model_test_all import ExactMatchModel

app = Flask(__name__)
app.secret_key = 'gggfs'  # 用于 session 加密


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

USERS_FILE = os.path.join(STORAGE_DIR, 'users.json')
RECORDS_FILE = os.path.join(STORAGE_DIR, 'records.json')
DISEASES_FILE = os.path.join(STORAGE_DIR, 'diseases.json')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38
MODEL_PATH = os.path.join(BASE_DIR, "results", "models", "semi_attention_5%_best.pth")


CLASSES =[
    "苹果黑星病(Apple___Apple_scab)",
    "苹果黑腐病(Apple___Black_rot)",
    "苹果雪松锈病(Apple___Cedar_apple_rust)",
    "苹果健康(Apple___healthy)",
    "蓝莓健康(Blueberry___healthy)",
    "樱桃白粉病(Cherry_(including_sour)___Powdery_mildew)",
    "樱桃健康(Cherry_(including_sour)___healthy)",
    "玉米灰斑病(Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot)",
    "玉米普通锈病(Corn_(maize)___Common_rust_)",
    "玉米大斑病(Corn_(maize)___Northern_Leaf_Blight)",
    "玉米健康(Corn_(maize)___healthy)",
    "葡萄黑腐病(Grape___Black_rot)",
    "葡萄黑痘病(Grape___Esca_(Black_Measles))",
    "葡萄叶枯病(Grape___Leaf_blight_(Isariopsis_Leaf_Spot))",
    "葡萄健康(Grape___healthy)",
    "柑橘黄龙病(Orange___Haunglongbing_(Citrus_greening))",
    "桃子细菌性斑点病(Peach___Bacterial_spot)",
    "桃子健康(Peach___healthy)",
    "甜椒细菌性斑点病(Pepper,_bell___Bacterial_spot)",
    "甜椒健康(Pepper,_bell___healthy)",
    "马铃薯早疫病(Potato___Early_blight)",
    "马铃薯晚疫病(Potato___Late_blight)",
    "马铃薯健康(Potato___healthy)",
    "覆盆子健康(Raspberry___healthy)",
    "大豆健康(Soybean___healthy)",
    "南瓜白粉病(Squash___Powdery_mildew)",
    "草莓叶枯病(Strawberry___Leaf_scorch)",
    "草莓健康(Strawberry___healthy)",
    "番茄细菌性斑点病(Tomato___Bacterial_spot)",
    "番茄早疫病(Tomato___Early_blight)",
    "番茄晚疫病(Tomato___Late_blight)",
    "番茄叶霉病(Tomato___Leaf_Mold)",
    "番茄斑枯病(Tomato___Septoria_leaf_spot)",
    "番茄红蜘蛛病(Tomato___Spider_mites Two-spotted_spider_mite)",
    "番茄靶斑病(Tomato___Target_Spot)",
    "番茄黄化曲叶病毒病(Tomato___Tomato_Yellow_Leaf_Curl_Virus)",
    "番茄花叶病毒病(Tomato___Tomato_mosaic_virus)",
    "番茄健康(Tomato___healthy)"
]



model = ExactMatchModel(num_classes=NUM_CLASSES)
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()

# 图像预处理
test_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def read_json(filepath):
    if not os.path.exists(filepath):
        return [] if 'users' in filepath or 'records' in filepath else {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



import torch.nn.functional as F


def predict_image(filepath):
    img = Image.open(filepath).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        # 将输出转换为概率
        probabilities = F.softmax(outputs, dim=1)
        # 获取最大概率值（置信度）和索引
        confidence, predicted = torch.max(probabilities, 1)

    class_name = CLASSES[predicted.item()]
    return class_name, confidence.item()




@app.route('/', methods=['GET'])
def index():
    if 'username' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('user_dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = read_json(USERS_FILE)

        for user in users:
            if user['username'] == username and user['password'] == password:
                session['username'] = username
                session['role'] = user.get('role', 'user')
                return redirect(url_for('index'))
        flash('用户名或密码错误！', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = read_json(USERS_FILE)

        if any(u['username'] == username for u in users):
            flash('用户名已存在！', 'warning')
        else:
            # 只能注册 user 角色
            users.append({"username": username, "password": password, "role": "user"})
            write_json(USERS_FILE, users)
            flash('注册成功，请登录！', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))



@app.route('/user', methods=['GET', 'POST'])
def user_dashboard():
    if 'username' not in session or session.get('role') != 'user':
        return redirect(url_for('login'))

    username = session['username']
    records = read_json(RECORDS_FILE)
    diseases = read_json(DISEASES_FILE)

    # 筛选当前用户的记录
    user_records = [r for r in records if r['user'] == username]
    user_records.reverse()  # 最新的排前面

    prediction_result = None
    disease_info = None
    uploaded_image = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('没有选择文件', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('没有选择文件', 'danger')
            return redirect(request.url)

        if file:
            # 安全处理文件名并加上时间戳防重名
            timestamp = str(int(time.time()))
            filename = timestamp + "_" + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result_class, confidence = predict_image(filepath)

            THRESHOLD = 0.60

            if confidence < THRESHOLD:
                # 🌟 置信度不够时的逻辑
                result_class = "未知类别或非植物"
                prediction_result = "此图片不是植物或者未查出此类别"
                disease_info = {
                    "name": "无法识别",
                    "description": f"系统对该图片的识别置信度仅为 {confidence:.2%}, 低于判定阈值。请确保上传了清晰的植物病害叶片照片。"
                }
            else:
                # 🌟 置信度足够，正常显示结果
                prediction_result = result_class
                disease_info = diseases.get(result_class, None)

            uploaded_image = filename

            # 保存到 records.json
            new_record = {
                "user": username,
                "result": result_class,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename
            }
            records.append(new_record)
            write_json(RECORDS_FILE, records)


            user_records.insert(0, new_record)

    return render_template('user_dashboard.html',
                           records=user_records,
                           prediction=prediction_result,
                           disease_info=disease_info,
                           uploaded_image=uploaded_image)


@app.route('/delete_records', methods=['POST'])
def delete_records():
    if 'username' not in session:
        return redirect(url_for('login'))

    selected_filenames = request.form.getlist('record_ids')

    if not selected_filenames:
        flash("请先选择要删除的记录", "warning")
        return redirect(url_for('user_dashboard'))

    records = read_json(RECORDS_FILE)

    new_records = []
    for rec in records:
        if rec.get('filename') in selected_filenames:
            # 物理删除文件逻辑
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], rec['filename'])
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"删除文件失败: {e}")
            continue
        new_records.append(rec)

    write_json(RECORDS_FILE, new_records)

    flash(f"成功删除 {len(selected_filenames)} 条记录及相关图片", "success")
    return redirect(url_for('user_dashboard'))


@app.route('/admin')
def admin_dashboard():
    if 'username' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    records = read_json(RECORDS_FILE)
    records.reverse()
    return render_template('admin_dashboard.html', records=records)


@app.route('/admin/delete_records', methods=['POST'])
def delete_records_admin():
    # 权限校验：必须是管理员且已登录
    if 'username' not in session or session.get('role') != 'admin':
        flash("权限不足", "danger")
        return redirect(url_for('login'))

    selected_filenames = request.form.getlist('record_ids')

    if not selected_filenames:
        flash("请先勾选要删除的系统记录", "warning")
        return redirect(url_for('admin_dashboard'))

    records = read_json(RECORDS_FILE)
    new_records = []

    count = 0
    for rec in records:
        if rec.get('filename') in selected_filenames:

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], rec['filename'])
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"管理员删除文件失败: {e}")
            count += 1
            continue
        new_records.append(rec)

    write_json(RECORDS_FILE, new_records)

    flash(f"管理员操作成功：已从系统中清除 {count} 条记录及相关文件", "success")
    return redirect(url_for('admin_dashboard'))



if __name__ == '__main__':
    app.run(debug=True, port=5000)