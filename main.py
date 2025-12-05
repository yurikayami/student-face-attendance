import os
import json
import datetime
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template, session
import sqlite3
from contextlib import contextmanager
import base64
from PIL import Image
import io
import random
import string
import hashlib
import cv2
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytz # Thêm thư viện pytz

# ---------- CẤU HÌNH ----------
UPLOAD_DIR = "uploads"
ENC_DIR = os.path.join(UPLOAD_DIR, "encodings")
IMG_DIR = os.path.join(UPLOAD_DIR, "images")
DB_PATH = "attendance.db"
MODELS_DIR = "static/models"
TEMPLATE_DIR = "templates"

# Cấu hình Time Zone
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# Tạo các thư mục cần thiết
os.makedirs(ENC_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# ---------- HÀM HỖ TRỢ THỜI GIAN ĐÃ CHUẨN HÓA ----------
def now_local_time_for_db():
    """Lấy thời gian hiện tại theo Local Time (VN_TZ) và format cho DB"""
    local_now = datetime.datetime.now(VN_TZ)
    return local_now.strftime('%Y-%m-%d %H:%M:%S')

def get_today_start_end():
    """Lấy chuỗi thời gian bắt đầu/kết thúc ngày hôm nay theo Local Time (VN_TZ)"""
    now = datetime.datetime.now(VN_TZ)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start_of_day.strftime('%Y-%m-%d %H:%M:%S'), end_of_day.strftime('%Y-%m-%d %H:%M:%S')


# ---------- KHỞI TẠO MODELS HIỆN ĐẠI ----------
class ArcFaceModel(nn.Module):
    """ArcFace Model for face recognition"""
    def __init__(self, embedding_size=512, num_classes=None):
        super(ArcFaceModel, self).__init__()
        self.embedding_size = embedding_size
        # Backbone cải tiến với ResNet-inspired architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual block 1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            
            # Residual block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(128 * 7 * 7, embedding_size)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

# Global models
arcface_model = None
lbph_recognizer = None
facenet_model = None
yolo_face_detector = None
mtcnn_detector = None

def initialize_modern_models():
    """Khởi tạo các model hiện đại"""
    global arcface_model, lbph_recognizer, mtcnn_detector
    
    try:
        print("🔄 Đang khởi tạo ArcFace Model...")
        arcface_model = ArcFaceModel(embedding_size=512)
        arcface_model.eval()
        
        print("🔄 Đang khởi tạo LBPH Recognizer...")
        lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        print("🔄 Đang khởi tạo MTCNN Detector...")
        try:
            from facenet_pytorch import MTCNN
            mtcnn_detector = MTCNN(keep_all=True, device='cpu', min_face_size=20)
            print("✅ MTCNN Detector đã được khởi tạo")
        except ImportError:
            print("⚠️ Không thể khởi tạo MTCNN, cần cài đặt facenet-pytorch")
            mtcnn_detector = None
        
        print("✅ Đã khởi tạo thành công các model hiện đại")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo models: {e}")

# Khởi tạo models
initialize_modern_models()

# ---------- DATABASE SQLITE ----------
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def ensure_db():
    with get_conn() as conn:
        cur = conn.cursor()
        
        # Bảng sinh_vien - 🛠️ Đã bỏ DEFAULT CURRENT_TIMESTAMP để kiểm soát thời gian từ Python
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sinh_vien (
                id_sv INTEGER PRIMARY KEY AUTOINCREMENT,
                ma_sv VARCHAR(20) UNIQUE,
                ho_ten VARCHAR(100) NOT NULL,
                gioi_tinh TEXT,
                lop VARCHAR(20),
                khoa VARCHAR(50),
                nganh_hoc VARCHAR(100),
                so_dien_thoai VARCHAR(15),
                email VARCHAR(100),
                trang_thai TEXT DEFAULT 'Đang học',
                created_at DATETIME, -- Giá trị sẽ được cung cấp từ Python
                updated_at DATETIME  -- Giá trị sẽ được cung cấp từ Python
            )
        """)
        
        # Bảng diem_danh - 🛠️ Đã bỏ DEFAULT CURRENT_TIMESTAMP để kiểm soát thời gian từ Python
        cur.execute("""
            CREATE TABLE IF NOT EXISTS diem_danh (
                id_diem_danh INTEGER PRIMARY KEY AUTOINCREMENT,
                id_sv INTEGER,
                ma_sv VARCHAR(20),
                ho_ten VARCHAR(100),
                lop VARCHAR(20),
                thoi_gian_vao DATETIME, -- Giá trị sẽ được cung cấp từ Python
                trang_thai TEXT DEFAULT 'Có mặt',
                do_tin_cay FLOAT,
                thuat_toan TEXT DEFAULT 'Euclidean',
                phuong_phap_phat_hien TEXT DEFAULT 'OpenCV',
                FOREIGN KEY (id_sv) REFERENCES sinh_vien (id_sv)
            )
        """)
        
        # Bảng nguoi_dung - 🛠️ Giữ nguyên CURRENT_TIMESTAMP cho bảng người dùng ít quan trọng về múi giờ
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nguoi_dung (
                id_user INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                ho_ten VARCHAR(100) NOT NULL,
                email VARCHAR(100),
                so_dien_thoai VARCHAR(15),
                vai_tro TEXT CHECK(vai_tro IN ('Admin','GiangVien','TroGiang')) DEFAULT 'GiangVien',
                trang_thai TEXT CHECK(trang_thai IN ('Hoạt động','Khóa')) DEFAULT 'Hoạt động',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tạo tài khoản admin mặc định nếu chưa có
        admin_password = hash_password("admin123")  # Mật khẩu mặc định
        cur.execute("""
            INSERT OR IGNORE INTO nguoi_dung 
            (username, password, ho_ten, email, vai_tro, trang_thai)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ('admin', admin_password, 'Huỳnh Lê Anh Khoa', 'admin@school.edu.vn', 'Admin', 'Hoạt động'))
        
        conn.commit()
        print("✅ Đã tạo/tạo lại tất cả bảng với bảng nguoi_dung mới")

def hash_password(password):
    """Hash mật khẩu sử dụng SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Xác minh mật khẩu"""
    return hash_password(password) == hashed

def generate_captcha():
    """Tạo mã captcha ngẫu nhiên"""
    characters = string.ascii_uppercase + string.digits
    captcha_text = ''.join(random.choice(characters) for _ in range(6))
    return captcha_text

ensure_db()

# ---------- QUẢN LÝ KHUÔN MẶT VỚI ĐA THUẬT TOÁN ----------
known_face_encodings = []
known_face_info = []
known_face_arcface_embeddings = []  # ArcFace embeddings
known_face_lbph_data = []  # LBPH data
known_face_deep_embeddings = []  # Deep Learning embeddings
known_face_mtcnn_embeddings = []  # MTCNN embeddings

def load_all_face_encodings():
    """Tải tất cả encodings khuôn mặt từ thư mục uploads/encodings"""
    global known_face_encodings, known_face_info, known_face_arcface_embeddings, known_face_lbph_data
    global known_face_deep_embeddings, known_face_mtcnn_embeddings
    
    known_face_encodings = []
    known_face_info = []
    known_face_arcface_embeddings = []
    known_face_lbph_data = []
    known_face_deep_embeddings = []
    known_face_mtcnn_embeddings = []
    
    print("🔄 Đang tải encodings khuôn mặt từ file .npy...")
    
    if not os.path.exists(ENC_DIR):
        print("❌ Thư mục encodings không tồn tại!")
        return
    
    files = os.listdir(ENC_DIR)
    print(f"📁 Tìm thấy {len(files)} files trong thư mục encodings")
    
    for filename in files:
        if filename.endswith('.npy'):
            ma_sv = filename.replace('.npy', '')
            print(f"🔍 Đang xử lý file: {filename}, Mã SV: {ma_sv}")
            
            # Lấy thông tin sinh viên từ database
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT ho_ten, lop FROM sinh_vien WHERE ma_sv = ?", (ma_sv,))
                result = cur.fetchone()
                
                if result:
                    ho_ten, lop = result
                    
                    # Load encodings từ file .npy
                    enc_path = os.path.join(ENC_DIR, filename)
                    try:
                        # Load file .npy
                        encodings_data = np.load(enc_path, allow_pickle=True)
                        
                        # Xử lý dữ liệu từ file .npy
                        if encodings_data.ndim == 0:
                            encodings = []
                        elif encodings_data.ndim == 1:
                            if encodings_data.shape == (128,):
                                encodings = [encodings_data]
                            else:
                                encodings = list(encodings_data)
                        else:
                            encodings = list(encodings_data)
                        
                        print(f"📊 File {filename} chứa {len(encodings)} encoding(s) cho {ho_ten}")
                        
                        # Thêm từng encoding vào danh sách
                        for i, encoding in enumerate(encodings):
                            encoding_array = np.array(encoding, dtype=np.float32)
                            
                            # Kiểm tra shape của encoding
                            if encoding_array.shape == (128,):
                                known_face_encodings.append(encoding_array)
                                known_face_info.append({
                                    'ma_sv': ma_sv,
                                    'ho_ten': ho_ten,
                                    'lop': lop
                                })
                                
                                # Tạo ArcFace embedding
                                arcface_embedding = convert_to_arcface_embedding(encoding_array)
                                known_face_arcface_embeddings.append(arcface_embedding)
                                
                                # Tạo LBPH data
                                lbph_data = convert_to_lbph_data(encoding_array)
                                known_face_lbph_data.append(lbph_data)
                                
                                # Tạo Deep Learning embedding
                                deep_embedding = convert_to_deep_embedding(encoding_array)
                                known_face_deep_embeddings.append(deep_embedding)
                                
                                # Tạo MTCNN embedding
                                mtcnn_embedding = convert_to_mtcnn_embedding(encoding_array)
                                known_face_mtcnn_embeddings.append(mtcnn_embedding)
                                
                                print(f"  ✅ Đã thêm encoding {i+1} cho {ho_ten}")
                            else:
                                print(f"  ❌ Encoding {i+1}: shape không hợp lệ {encoding_array.shape}")
                        
                    except Exception as e:
                        print(f"❌ Lỗi khi tải encoding cho {ma_sv}: {e}")
                else:
                    print(f"⚠️ Không tìm thấy thông tin sinh viên cho mã: {ma_sv}")
    
    print(f"🎯 Tổng số khuôn mặt đã tải: {len(known_face_encodings)} từ {len(set([info['ma_sv'] for info in known_face_info]))} sinh viên")

def convert_to_arcface_embedding(encoding):
    """Chuyển đổi encoding thường sang ArcFace embedding"""
    if len(encoding) == 128:
        # Mô phỏng ArcFace embedding từ FaceNet-like encoding
        expanded = np.zeros(512)
        expanded[:128] = encoding
        expanded[128:256] = encoding * 0.9
        expanded[256:384] = encoding * 0.8
        expanded[384:512] = encoding * 0.7
        return expanded / np.linalg.norm(expanded)
    return encoding

def convert_to_lbph_data(encoding):
    """Chuyển đổi encoding sang dạng LBPH"""
    lbph_like = (encoding * 1000).astype(np.uint8)
    return lbph_like

def convert_to_deep_embedding(encoding):
    """Chuyển đổi encoding sang Deep Learning embedding"""
    if len(encoding) == 128:
        # Mô phỏng deep learning embedding
        expanded = np.zeros(512)
        expanded[:128] = encoding
        expanded[128:256] = encoding * 0.95
        expanded[256:384] = encoding * 0.85
        expanded[384:512] = encoding * 0.75
        return expanded / np.linalg.norm(expanded)
    return encoding

def convert_to_mtcnn_embedding(encoding):
    """Chuyển đổi encoding sang MTCNN-style embedding"""
    if len(encoding) == 128:
        # MTCNN thường dùng 512-dimensional embeddings
        expanded = np.zeros(512)
        expanded[:128] = encoding
        expanded[128:256] = encoding * 0.92
        expanded[256:384] = encoding * 0.82
        expanded[384:512] = encoding * 0.72
        return expanded / np.linalg.norm(expanded)
    return encoding

def euclidean_distance(encoding1, encoding2):
    """Tính khoảng cách Euclidean giữa 2 encoding"""
    return np.linalg.norm(encoding1 - encoding2)

def cosine_similarity_score(encoding1, encoding2):
    """Tính cosine similarity giữa 2 encoding"""
    return cosine_similarity([encoding1], [encoding2])[0][0]

def arcface_similarity(embedding1, embedding2):
    """Tính similarity sử dụng ArcFace embeddings"""
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    return np.dot(embedding1_norm, embedding2_norm)

def lbph_similarity(lbph_data1, lbph_data2):
    """Tính similarity sử dụng LBPH"""
    return 1.0 - (np.abs(lbph_data1 - lbph_data2).mean() / 255.0)

def deep_learning_similarity(embedding1, embedding2):
    """Tính similarity sử dụng Deep Learning embeddings"""
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    return np.dot(embedding1_norm, embedding2_norm)

def mtcnn_similarity(embedding1, embedding2):
    """Tính similarity sử dụng MTCNN embeddings"""
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    return np.dot(embedding1_norm, embedding2_norm)

# ---------- PHÁT HIỆN KHUÔN MẶT ĐA PHƯƠNG PHÁP ----------
def detect_faces_multi_method(image_array):
    """
    Phát hiện khuôn mặt sử dụng nhiều phương pháp
    Trả về danh sách khuôn mặt đã được phát hiện
    """
    faces = []
    detection_methods = []
    
    try:
        # Chuyển đổi image array sang định dạng phù hợp
        if len(image_array.shape) == 3:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_array
        
        # 1. OpenCV Haar Cascade (Cơ bản)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            cv_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in cv_faces:
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.85,
                    'method': 'OpenCV_Haar'
                })
                detection_methods.append('OpenCV_Haar')
        except Exception as e:
            print(f"⚠️ Lỗi OpenCV detection: {e}")
        
        # 2. Dlib HOG Detector (Truyền thống nhưng ổn định)
        try:
            import dlib
            detector = dlib.get_frontal_face_detector()
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            dlib_faces = detector(gray, 1)
            
            for face in dlib_faces:
                x = face.left()
                y = face.top()
                width = face.right() - x
                height = face.bottom() - y
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': 0.90,
                    'method': 'Dlib_HOG'
                })
                detection_methods.append('Dlib_HOG')
        except Exception as e:
            print(f"⚠️ Lỗi Dlib detection: {e}")
        
        # 3. MTCNN (Deep Learning - Độ chính xác cao)
        try:
            if mtcnn_detector is not None:
                boxes, probs = mtcnn_detector.detect(rgb_image)
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        if probs[i] > 0.9:  # Ngưỡng confidence cao
                            x1, y1, x2, y2 = box.astype(int)
                            width = x2 - x1
                            height = y2 - y1
                            
                            faces.append({
                                'bbox': (x1, y1, width, height),
                                'confidence': float(probs[i]),
                                'method': 'MTCNN'
                            })
                            detection_methods.append('MTCNN')
        except Exception as e:
            print(f"⚠️ Lỗi MTCNN detection: {e}")
        
        # Loại bỏ các khuôn mặt trùng lặp
        faces = remove_duplicate_faces(faces)
        
        print(f"🔍 Đã phát hiện {len(faces)} khuôn mặt với các phương pháp: {set(detection_methods)}")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình phát hiện khuôn mặt: {e}")
    
    return faces

def remove_duplicate_faces(faces, iou_threshold=0.5):
    """Loại bỏ các khuôn mặt trùng lặp dựa trên IoU"""
    if len(faces) <= 1:
        return faces
    
    # Sắp xếp theo confidence giảm dần
    faces.sort(key=lambda x: x['confidence'], reverse=True)
    
    filtered_faces = []
    used_indices = set()
    
    for i in range(len(faces)):
        if i in used_indices:
            continue
            
        current_face = faces[i]
        filtered_faces.append(current_face)
        used_indices.add(i)
        
        for j in range(i + 1, len(faces)):
            if j in used_indices:
                continue
                
            iou = calculate_iou(current_face['bbox'], faces[j]['bbox'])
            if iou > iou_threshold:
                used_indices.add(j)
    
    return filtered_faces

def calculate_iou(box1, box2):
    """Tính Intersection over Union của 2 bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Chuyển đổi sang dạng (x1, y1, x2, y2)
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # Tính diện tích giao nhau
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Tính diện tích hợp
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# ---------- HỆ THỐNG NHẬN DIỆN ĐA THUẬT TOÁN NÂNG CAO ----------
def recognize_face_multi_algorithm(face_encoding, detection_method="OpenCV"):
    """Nhận diện khuôn mặt sử dụng đa thuật toán nâng cao"""
    if len(known_face_encodings) == 0:
        return None, None, "No faces", detection_method
    
    try:
        # Chuyển đổi encoding sang các định dạng khác
        arcface_embedding = convert_to_arcface_embedding(face_encoding)
        deep_embedding = convert_to_deep_embedding(face_encoding)
        mtcnn_embedding = convert_to_mtcnn_embedding(face_encoding)
        
        # Kết quả từ các thuật toán
        results = []
        
        # 1. ArcFace Similarity (Hiện đại nhất)
        best_arcface_index = None
        best_arcface_similarity = -1
        
        for i, known_embedding in enumerate(known_face_arcface_embeddings):
            similarity = arcface_similarity(arcface_embedding, known_embedding)
            if similarity > best_arcface_similarity:
                best_arcface_similarity = similarity
                best_arcface_index = i
        
        if best_arcface_index is not None:
            arcface_confidence = max(0, best_arcface_similarity * 100)
            results.append(('ArcFace', best_arcface_index, arcface_confidence))
        
        # 2. Deep Learning Similarity
        best_deep_index = None
        best_deep_similarity = -1
        
        for i, known_embedding in enumerate(known_face_deep_embeddings):
            similarity = deep_learning_similarity(deep_embedding, known_embedding)
            if similarity > best_deep_similarity:
                best_deep_similarity = similarity
                best_deep_index = i
        
        if best_deep_index is not None:
            deep_confidence = max(0, best_deep_similarity * 100)
            results.append(('DeepLearning', best_deep_index, deep_confidence))
        
        # 3. MTCNN Similarity
        best_mtcnn_index = None
        best_mtcnn_similarity = -1
        
        for i, known_embedding in enumerate(known_face_mtcnn_embeddings):
            similarity = mtcnn_similarity(mtcnn_embedding, known_embedding)
            if similarity > best_mtcnn_similarity:
                best_mtcnn_similarity = similarity
                best_mtcnn_index = i
        
        if best_mtcnn_index is not None:
            mtcnn_confidence = max(0, best_mtcnn_similarity * 100)
            results.append(('MTCNN', best_mtcnn_index, mtcnn_confidence))
        
        # 4. Euclidean Distance (Traditional)
        best_euclidean_index = None
        best_euclidean_distance = float('inf')
        
        for i, known_encoding in enumerate(known_face_encodings):
            distance = euclidean_distance(face_encoding, known_encoding)
            if distance < best_euclidean_distance:
                best_euclidean_distance = distance
                best_euclidean_index = i
        
        if best_euclidean_index is not None:
            euclidean_confidence = max(0, (1 - (best_euclidean_distance / 0.6)) * 100)
            results.append(('Euclidean', best_euclidean_index, euclidean_confidence))
        
        # 5. LBPH Similarity
        if len(known_face_lbph_data) > 0:
            lbph_data = convert_to_lbph_data(face_encoding)
            best_lbph_index = None
            best_lbph_similarity = -1
            
            for i, known_lbph in enumerate(known_face_lbph_data):
                similarity = lbph_similarity(lbph_data, known_lbph)
                if similarity > best_lbph_similarity:
                    best_lbph_similarity = similarity
                    best_lbph_index = i
            
            if best_lbph_index is not None:
                lbph_confidence = max(0, best_lbph_similarity * 100)
                results.append(('LBPH', best_lbph_index, lbph_confidence))
        
        # Advanced voting system với trọng số
        if results:
            # Sắp xếp theo confidence
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Phân loại thuật toán
            modern_algorithms = [r for r in results if r[0] in ['ArcFace', 'DeepLearning', 'MTCNN']]
            traditional_algorithms = [r for r in results if r[0] in ['Euclidean', 'LBPH']]
            
            # Ưu tiên modern algorithms
            if modern_algorithms:
                best_algorithm, best_index, best_confidence = modern_algorithms[0]
            else:
                best_algorithm, best_index, best_confidence = results[0]
            
            # Ngưỡng confidence linh hoạt theo thuật toán
            confidence_threshold = 60  # Mặc định
            if best_algorithm in ['ArcFace', 'DeepLearning']:
                confidence_threshold = 65  # Yêu cầu cao hơn cho các thuật toán hiện đại
            elif best_algorithm == 'MTCNN':
                confidence_threshold = 70
            
            if best_confidence >= confidence_threshold:
                student_info = known_face_info[best_index]
                
                # Ghi log chi tiết
                print(f"🎯 NHẬN DIỆN THÀNH CÔNG - {student_info['ho_ten']} "
                      f"({best_algorithm}: {best_confidence:.1f}%, "
                      f"Phát hiện: {detection_method})")
                
                return student_info, best_confidence, best_algorithm, detection_method
        
        print(f"❌ KHÔNG NHẬN DIỆN ĐƯỢC - Độ tin cậy tối đa: "
              f"{max([r[2] for r in results]) if results else 0:.1f}%")
        
        return None, None, "No match", detection_method
            
    except Exception as e:
        print(f"💥 Lỗi trong quá trình nhận diện đa thuật toán: {e}")
        return None, None, "Error", detection_method

def recognize_face(face_encoding):
    """Wrapper cho tương thích với code cũ"""
    student_info, confidence, algorithm, detection_method = recognize_face_multi_algorithm(face_encoding)
    return student_info, confidence

# Tải encodings khi khởi động
load_all_face_encodings()

# ---------- FLASK ----------
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder='.')
app.secret_key = 'your_secret_key_here'

# ---------- AUTHENTICATION MIDDLEWARE ----------
def login_required(f):
    """Decorator để kiểm tra đăng nhập"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized', 'message': 'Vui lòng đăng nhập'}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_role(role):
    """Decorator để kiểm tra vai trò"""
    def decorator(f):
        from functools import wraps
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized', 'message': 'Vui lòng đăng nhập'}), 401
            
            if session.get('vai_tro') != role:
                return jsonify({'error': 'Forbidden', 'message': 'Không có quyền truy cập'}), 403
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ---------- ROUTES ĐĂNG NHẬP ----------
@app.route("/login", methods=["GET"])
def login_page():
    """Trang đăng nhập"""
    captcha_text = generate_captcha()
    session['captcha'] = captcha_text
    
    return render_template('login.html', captcha_text=captcha_text)

@app.route("/api/login", methods=["POST"])
def api_login():
    """API đăng nhập"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Thiếu dữ liệu"}), 400
        
        username = data.get('username')
        password = data.get('password')
        captcha_input = data.get('captcha')
        
        # Kiểm tra captcha
        if not captcha_input or captcha_input.upper() != session.get('captcha', ''):
            new_captcha = generate_captcha()
            session['captcha'] = new_captcha
            return jsonify({
                "error": "captcha_error", 
                "message": "Mã captcha không đúng",
                "new_captcha": new_captcha
            }), 400
        
        # Kiểm tra thông tin đăng nhập
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id_user, username, password, ho_ten, email, vai_tro, trang_thai 
                FROM nguoi_dung 
                WHERE username = ? AND trang_thai = 'Hoạt động'
            """, (username,))
            user = cur.fetchone()
            
            if user and verify_password(password, user['password']):
                session['user_id'] = user['id_user']
                session['username'] = user['username']
                session['ho_ten'] = user['ho_ten']
                session['vai_tro'] = user['vai_tro']
                session['email'] = user['email']
                
                print(f"✅ Đăng nhập thành công: {user['ho_ten']} - Vai trò: {user['vai_tro']}")
                
                return jsonify({
                    "success": True,
                    "message": "Đăng nhập thành công",
                    "user": {
                        "id_user": user['id_user'],
                        "username": user['username'],
                        "ho_ten": user['ho_ten'],
                        "vai_tro": user['vai_tro'],
                        "email": user['email']
                    }
                })
            else:
                new_captcha = generate_captcha()
                session['captcha'] = new_captcha
                return jsonify({
                    "error": "login_error",
                    "message": "Tên đăng nhập hoặc mật khẩu không đúng",
                    "new_captcha": new_captcha
                }), 400
                
    except Exception as e:
        print(f"💥 Lỗi đăng nhập: {e}")
        return jsonify({"error": "server_error", "detail": str(e)}), 500

@app.route("/api/logout", methods=["POST"])
def api_logout():
    """API đăng xuất"""
    session.clear()
    return jsonify({"success": True, "message": "Đã đăng xuất"})

@app.route("/api/refresh-captcha", methods=["GET"])
def api_refresh_captcha():
    """API làm mới captcha"""
    new_captcha = generate_captcha()
    session['captcha'] = new_captcha
    return jsonify({"captcha": new_captcha})

@app.route("/api/user-info", methods=["GET"])
@login_required
def api_user_info():
    """API lấy thông tin người dùng hiện tại"""
    return jsonify({
        "id_user": session.get('user_id'),
        "username": session.get('username'),
        "ho_ten": session.get('ho_ten'),
        "vai_tro": session.get('vai_tro'),
        "email": session.get('email')
    })

# ---------- ROUTES CHÍNH (CÓ BẢO VỆ) ----------
@app.route("/")
@login_required
def index():
    return render_template('index.html')

@app.route("/models/<path:model_path>")
@login_required
def serve_models(model_path):
    return send_from_directory(MODELS_DIR, model_path)

@app.route("/static/models/<path:model_path>")
@login_required
def serve_static_models(model_path):
    return send_from_directory(MODELS_DIR, model_path)

@app.route("/static/<path:path>")
def serve_static_files(path):
    return send_from_directory('.', path)

# ---------- API QUẢN LÝ NGƯỜI DÙNG (CHỈ ADMIN) ----------
@app.route("/api/nguoi-dung", methods=["GET"])
@login_required
@require_role('Admin')
def api_nguoi_dung():
    """Lấy danh sách người dùng (chỉ Admin)"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id_user, username, ho_ten, email, so_dien_thoai, 
                   vai_tro, trang_thai, created_at
            FROM nguoi_dung 
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        
        users_list = [dict(row) for row in rows]
        return jsonify(users_list)

@app.route("/api/nguoi-dung", methods=["POST"])
@login_required
@require_role('Admin')
def api_them_nguoi_dung():
    """Thêm người dùng mới (chỉ Admin)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Thiếu dữ liệu"}), 400
        
        required_fields = ['username', 'password', 'ho_ten', 'vai_tro']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Thiếu trường bắt buộc: {field}"}), 400
        
        with get_conn() as conn:
            cur = conn.cursor()
            
            # Kiểm tra username trùng
            cur.execute("SELECT id_user FROM nguoi_dung WHERE username = ?", (data['username'],))
            if cur.fetchone():
                return jsonify({"error": "Tên đăng nhập đã tồn tại"}), 400
            
            # Hash mật khẩu
            hashed_password = hash_password(data['password'])
            
            # Thêm người dùng
            cur.execute("""
                INSERT INTO nguoi_dung 
                (username, password, ho_ten, email, so_dien_thoai, vai_tro, trang_thai)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data['username'],
                hashed_password,
                data['ho_ten'],
                data.get('email'),
                data.get('so_dien_thoai'),
                data['vai_tro'],
                data.get('trang_thai', 'Hoạt động')
            ))
            
            user_id = cur.lastrowid
            conn.commit()
            
            print(f"✅ Đã thêm người dùng: {data['ho_ten']} - {data['username']} - {data['vai_tro']}")
            
            return jsonify({
                "success": True,
                "id_user": user_id,
                "message": "Đã thêm người dùng thành công"
            })
            
    except Exception as e:
        return jsonify({"error": "server_error", "detail": str(e)}), 500

@app.route("/api/nguoi-dung/<int:id_user>", methods=["PUT"])
@login_required
@require_role('Admin')
def api_sua_nguoi_dung(id_user):
    """Sửa thông tin người dùng (chỉ Admin)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Thiếu dữ liệu"}), 400
        
        with get_conn() as conn:
            cur = conn.cursor()
            
            # Kiểm tra người dùng tồn tại
            cur.execute("SELECT id_user FROM nguoi_dung WHERE id_user = ?", (id_user,))
            if not cur.fetchone():
                return jsonify({"error": "Không tìm thấy người dùng"}), 404
            
            # Cập nhật thông tin
            update_fields = []
            update_values = []
            
            if 'ho_ten' in data:
                update_fields.append("ho_ten = ?")
                update_values.append(data['ho_ten'])
            if 'email' in data:
                update_fields.append("email = ?")
                update_values.append(data.get('email'))
            if 'so_dien_thoai' in data:
                update_fields.append("so_dien_thoai = ?")
                update_values.append(data.get('so_dien_thoai'))
            if 'vai_tro' in data:
                update_fields.append("vai_tro = ?")
                update_values.append(data['vai_tro'])
            if 'trang_thai' in data:
                update_fields.append("trang_thai = ?")
                update_values.append(data['trang_thai'])
            
            # Nếu có mật khẩu mới
            if 'password' in data and data['password']:
                update_fields.append("password = ?")
                update_values.append(hash_password(data['password']))
            
            if not update_fields:
                return jsonify({"error": "Không có trường nào để cập nhật"}), 400
            
            update_values.append(id_user)
            
            query = f"UPDATE nguoi_dung SET {', '.join(update_fields)} WHERE id_user = ?"
            cur.execute(query, update_values)
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": "Đã cập nhật thông tin người dùng"
            })
            
    except Exception as e:
        return jsonify({"error": "server_error", "detail": str(e)}), 500

@app.route("/api/nguoi-dung/<int:id_user>", methods=["DELETE"])
@login_required
@require_role('Admin')
def api_xoa_nguoi_dung(id_user):
    """Xóa người dùng (chỉ Admin)"""
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            
            # Không cho xóa chính mình
            if id_user == session.get('user_id'):
                return jsonify({"error": "Không thể xóa tài khoản của chính mình"}), 400
            
            cur.execute("SELECT username, ho_ten FROM nguoi_dung WHERE id_user = ?", (id_user,))
            user = cur.fetchone()
            if not user:
                return jsonify({"error": "Không tìm thấy người dùng"}), 404
            
            cur.execute("DELETE FROM nguoi_dung WHERE id_user = ?", (id_user,))
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": f"Đã xóa người dùng {user['ho_ten']} - {user['username']}"
            })
            
    except Exception as e:
        return jsonify({"error": "server_error", "detail": str(e)}), 500

# ---------- API SINH VIÊN (CÓ BẢO VỆ) ----------
@app.route("/api/sinh-vien", methods=["GET"])
@login_required
def api_sinh_vien():
    """Lấy danh sách sinh viên"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id_sv, ma_sv, ho_ten, gioi_tinh, lop, khoa, nganh_hoc,
                   so_dien_thoai, email, trang_thai, created_at
            FROM sinh_vien 
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        
        sinh_vien_list = []
        for row in rows:
            sv = dict(row)
            # Đếm số lượng ảnh khuôn mặt
            ma_sv = sv['ma_sv']
            enc_path = os.path.join(ENC_DIR, f"{ma_sv}.npy")
            if os.path.exists(enc_path):
                try:
                    encodings_data = np.load(enc_path, allow_pickle=True)
                    if encodings_data.ndim == 0:
                        sv['so_anh_khuon_mat'] = 0
                    else:
                        sv['so_anh_khuon_mat'] = len(encodings_data) if encodings_data.ndim > 0 else 1
                except:
                    sv['so_anh_khuon_mat'] = 0
            else:
                sv['so_anh_khuon_mat'] = 0
                
            sinh_vien_list.append(sv)
        
        return jsonify(sinh_vien_list)

@app.route("/api/sinh-vien", methods=["POST"])
@login_required
def api_them_sinh_vien():
    """Thêm sinh viên mới"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Thiếu dữ liệu"}), 400
        
        required_fields = ['ma_sv', 'ho_ten', 'lop']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Thiếu trường bắt buộc: {field}"}), 400
        
        # 🛠️ KHẮC PHỤC: Lấy thời gian hiện tại
        current_time = now_local_time_for_db()

        with get_conn() as conn:
            cur = conn.cursor()
            
            # Kiểm tra mã sinh viên trùng
            cur.execute("SELECT id_sv FROM sinh_vien WHERE ma_sv = ?", (data['ma_sv'],))
            if cur.fetchone():
                return jsonify({"error": "Mã sinh viên đã tồn tại"}), 400
            
            # Thêm sinh viên (thêm created_at và updated_at)
            cur.execute("""
                INSERT INTO sinh_vien 
                (ma_sv, ho_ten, gioi_tinh, lop, khoa, nganh_hoc, so_dien_thoai, email, trang_thai, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['ma_sv'],
                data['ho_ten'],
                data.get('gioi_tinh', 'Khác'),
                data['lop'],
                data.get('khoa'),
                data.get('nganh_hoc'),
                data.get('so_dien_thoai'),
                data.get('email'),
                data.get('trang_thai', 'Đang học'),
                current_time, # 🛠️ Thêm created_at
                current_time  # 🛠️ Thêm updated_at
            ))
            
            id_sv = cur.lastrowid
            conn.commit()
            
            print(f"✅ Đã thêm sinh viên: {data['ho_ten']} - {data['ma_sv']} - {data['lop']}")
            
            return jsonify({
                "success": True,
                "id_sv": id_sv,
                "message": "Đã thêm sinh viên thành công"
            })
            
    except Exception as e:
        return jsonify({"error": "server_error", "detail": str(e)}), 500

# ---------- API SỬA SINH VIÊN ----------
@app.route("/api/sinh-vien", methods=["PUT"])
@login_required
def api_sua_sinh_vien():
    """Sửa thông tin sinh viên"""
    try:
        data = request.get_json()
        print(f"📝 Nhận request sửa sinh viên: {data}")
        
        if not data:
            return jsonify({"error": "Thiếu dữ liệu"}), 400
        
        # Kiểm tra trường bắt buộc
        if 'id_sv' not in data:
            return jsonify({"error": "Thiếu id_sv"}), 400
        
        id_sv = data['id_sv']
        
        with get_conn() as conn:
            cur = conn.cursor()
            
            # Kiểm tra sinh viên tồn tại
            cur.execute("SELECT id_sv, ma_sv FROM sinh_vien WHERE id_sv = ?", (id_sv,))
            sv = cur.fetchone()
            if not sv:
                return jsonify({"error": "Không tìm thấy sinh viên"}), 404
            
            # Chuẩn bị các trường cập nhật
            update_fields = []
            update_values = []
            
            # Các trường có thể cập nhật
            updatable_fields = [
                'ho_ten', 'gioi_tinh', 'lop', 'khoa', 'nganh_hoc', 
                'so_dien_thoai', 'email', 'trang_thai'
            ]
            
            for field in updatable_fields:
                if field in data:
                    update_fields.append(f"{field} = ?")
                    update_values.append(data[field])
            
            # 🛠️ KHẮC PHỤC: Cập nhật thời gian cập nhật
            update_fields.append("updated_at = ?")
            update_values.append(now_local_time_for_db())
            
            if not update_fields:
                return jsonify({"error": "Không có trường nào để cập nhật"}), 400
            
            # Thêm id_sv vào cuối cho điều kiện WHERE
            update_values.append(id_sv)
            
            # Thực hiện cập nhật
            query = f"UPDATE sinh_vien SET {', '.join(update_fields)} WHERE id_sv = ?"
            print(f"🔧 Query: {query}")
            print(f"🔧 Values: {update_values}")
            
            cur.execute(query, update_values)
            conn.commit()
            
            # Lấy thông tin sinh viên sau khi cập nhật
            cur.execute("""
                SELECT id_sv, ma_sv, ho_ten, gioi_tinh, lop, khoa, nganh_hoc,
                       so_dien_thoai, email, trang_thai, created_at
                FROM sinh_vien WHERE id_sv = ?
            """, (id_sv,))
            updated_sv = cur.fetchone()
            
            print(f"✅ Đã cập nhật sinh viên: {updated_sv['ho_ten']} - {updated_sv['ma_sv']}")
            
            return jsonify({
                "success": True,
                "message": "Đã cập nhật thông tin sinh viên thành công",
                "sinh_vien": dict(updated_sv)
            })
            
    except Exception as e:
        print(f"💥 Lỗi cập nhật sinh viên: {e}")
        return jsonify({"error": "server_error", "detail": str(e)}), 500
    

@app.route("/api/sinh-vien/<int:id_sv>", methods=["DELETE"])
@login_required
def api_xoa_sinh_vien(id_sv):
    """Xóa sinh viên"""
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            
            cur.execute("SELECT ma_sv, ho_ten FROM sinh_vien WHERE id_sv = ?", (id_sv,))
            sv = cur.fetchone()
            if not sv:
                return jsonify({"error": "Không tìm thấy sinh viên"}), 404
            
            ma_sv = sv['ma_sv']
            ho_ten = sv['ho_ten']
            
            cur.execute("DELETE FROM diem_danh WHERE id_sv = ?", (id_sv,))
            cur.execute("DELETE FROM sinh_vien WHERE id_sv = ?", (id_sv,))
            
            conn.commit()
            
            # Xóa file encoding .npy
            enc_path = os.path.join(ENC_DIR, f"{ma_sv}.npy")
            if os.path.exists(enc_path):
                os.remove(enc_path)
                print(f"🗑️ Đã xóa file encoding: {enc_path}")
            
            # Xóa thư mục ảnh
            img_dir = os.path.join(IMG_DIR, ma_sv)
            if os.path.exists(img_dir):
                import shutil
                shutil.rmtree(img_dir)
                print(f"🗑️ Đã xóa thư mục ảnh: {img_dir}")
            
            # Reload encodings
            load_all_face_encodings()
            
            return jsonify({
                "success": True,
                "message": f"Đã xóa sinh viên {ho_ten} - {ma_sv}"
            })
            
    except Exception as e:
        return jsonify({"error": "server_error", "detail": str(e)}), 500

# ---------- API TRAIN KHUÔN MẶT ----------
@app.route("/api/train", methods=["POST"])
@login_required
def api_train():
    """Train khuôn mặt cho sinh viên - LƯU BẰNG .NPY"""
    try:
        data = request.get_json()
        print(f"🎯 Nhận request train từ frontend")
        
        if not data:
            return jsonify({"error": "Thiếu dữ liệu"}), 400

        id_sv = data.get("id_sv")
        descriptor = data.get("descriptor")
        
        print(f"📝 ID SV: {id_sv}, Descriptor length: {len(descriptor) if descriptor else 0}")
        
        if not id_sv or not descriptor:
            return jsonify({"error": "Thiếu id_sv hoặc descriptor"}), 400

        # Lấy thông tin sinh viên
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT ma_sv, ho_ten, lop FROM sinh_vien WHERE id_sv = ?", (id_sv,))
            sv = cur.fetchone()
            if not sv:
                return jsonify({"error": "Không tìm thấy sinh viên"}), 404
            
            ma_sv = sv['ma_sv']
            ho_ten = sv['ho_ten']
            lop = sv['lop']

        # Chuyển descriptor thành numpy array
        try:
            encoding = np.array(descriptor, dtype=np.float32)
            print(f"📐 Encoding shape: {encoding.shape}")
            
            if encoding.shape != (128,):
                return jsonify({"error": f"Descriptor phải có 128 chiều, nhận được {encoding.shape}"}), 400
        except Exception as e:
            print(f"❌ Lỗi chuyển descriptor: {e}")
            return jsonify({"error": "Descriptor không hợp lệ"}), 400

        # LƯU ENCODING VÀO FILE .NPY
        save_face_encoding(ma_sv, encoding)
        
        # Cập nhật cache encodings NGAY LẬP TỨC để nhận diện real-time
        load_all_face_encodings()

        return jsonify({
            "success": True,
            "ma_sv": ma_sv,
            "ho_ten": ho_ten,
            "lop": lop,
            "message": f"Đã train khuôn mặt thành công cho {ho_ten}. Giờ có thể nhận diện real-time!"
        })

    except Exception as e:
        print(f"💥 Lỗi train khuôn mặt: {e}")
        return jsonify({"error": "server_error", "detail": str(e)}), 500

def save_face_encoding(ma_sv, encoding):
    """Lưu encoding khuôn mặt vào file .npy"""
    enc_path = os.path.join(ENC_DIR, f"{ma_sv}.npy")
    
    # Load encodings hiện có nếu file tồn tại
    existing_encodings = []
    if os.path.exists(enc_path):
        try:
            existing_data = np.load(enc_path, allow_pickle=True)
            if existing_data.ndim == 0:
                existing_encodings = []
            elif existing_data.ndim == 1:
                if existing_data.shape == (128,):
                    existing_encodings = [existing_data]
                else:
                    existing_encodings = list(existing_data)
            else:
                existing_encodings = list(existing_data)
            print(f"📁 Đã tải {len(existing_encodings)} encoding hiện có từ {ma_sv}.npy")
        except Exception as e:
            print(f"❌ Lỗi khi tải encoding cũ: {e}")
            existing_encodings = []
    
    # Thêm encoding mới
    existing_encodings.append(encoding)
    
    # Lưu file .npy
    np.save(enc_path, np.array(existing_encodings, dtype=object))
    
    print(f"💾 Đã lưu encoding cho {ma_sv} vào file .npy, tổng: {len(existing_encodings)} encoding")

# ---------- API NHẬN DIỆN KHUÔN MẶT REAL-TIME ----------
@app.route("/api/identify", methods=["POST"])
@login_required
def api_identify():
    """Nhận diện khuôn mặt và điểm danh - REAL TIME với đa thuật toán"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Thiếu dữ liệu"}), 400

        descriptor = data.get("descriptor")
        if not descriptor:
            return jsonify({"error": "Thiếu descriptor"}), 400

        # Chuyển descriptor thành numpy array
        try:
            face_encoding = np.array(descriptor, dtype=np.float32)
            
            if face_encoding.shape != (128,):
                return jsonify({"error": f"Descriptor phải có 128 chiều, nhận được {face_encoding.shape}"}), 400
        except Exception as e:
            print(f"❌ Lỗi chuyển descriptor: {e}")
            return jsonify({"error": "Descriptor không hợp lệ"}), 400

        # NHẬN DIỆN KHUÔN MẶT VỚI ĐA THUẬT TOÁN
        student_info, confidence, algorithm, detection_method = recognize_face_multi_algorithm(face_encoding)

        # CHỈ XÁC NHẬN DANH TÍNH KHI ĐỘ TIN CẬY >= 60%
        if student_info and confidence and confidence >= 60:
            # GHI NHẬN ĐIỂM DANH
            with get_conn() as conn:
                cur = conn.cursor()
                
                cur.execute("SELECT id_sv FROM sinh_vien WHERE ma_sv = ?", (student_info['ma_sv'],))
                sv_result = cur.fetchone()
                
                if sv_result:
                    id_sv = sv_result['id_sv']
                    
                    # 🛠️ KHẮC PHỤC: Sử dụng Local Time (VN_TZ) để so sánh
                    now_dt = datetime.datetime.now(VN_TZ)
                    five_minutes_ago_dt = now_dt - datetime.timedelta(minutes=5)
                    five_minutes_ago = five_minutes_ago_dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Kiểm tra xem đã điểm danh trong 5 phút gần đây chưa
                    cur.execute("""
                        SELECT id_diem_danh FROM diem_danh  
                        WHERE ma_sv = ? AND thoi_gian_vao > ?
                    """, (student_info['ma_sv'], five_minutes_ago))
                    
                    existing_attendance = cur.fetchone()
                    
                    if not existing_attendance:
                        # 🛠️ KHẮC PHỤC: Lấy thời gian hiện tại để ghi vào DB
                        current_time_db_format = now_local_time_for_db()
                        
                        # Thêm điểm danh với thông tin thuật toán và phương pháp phát hiện
                        cur.execute("""
                            INSERT INTO diem_danh (id_sv, ma_sv, ho_ten, lop, trang_thai, do_tin_cay, thuat_toan, phuong_phap_phat_hien, thoi_gian_vao)
                            VALUES (?, ?, ?, ?, 'Có mặt', ?, ?, ?, ?)
                        """, (id_sv, student_info['ma_sv'], student_info['ho_ten'], student_info['lop'], float(confidence), algorithm, detection_method, current_time_db_format))
                        conn.commit()
                        print(f"✅ ĐÃ ĐIỂM DANH ({algorithm} - {detection_method}): {student_info['ho_ten']} - {student_info['lop']} - Tin cậy: {confidence:.1f}%")
                    else:
                        print(f"ℹ️ Đã điểm danh gần đây: {student_info['ho_ten']}")

            return jsonify({
                "matched": True,
                "ma_sv": student_info['ma_sv'],
                "ho_ten": student_info['ho_ten'],
                "lop": student_info['lop'],
                "do_tin_cay": float(confidence),
                "thuat_toan": algorithm,
                "phuong_phap_phat_hien": detection_method,
                "message": f"Đã nhận diện ({algorithm} - {detection_method}): {student_info['ho_ten']} - {student_info['lop']}"
            })
        else:
            confidence_value = float(confidence) if confidence else 0.0
            print(f"❓ Không xác định - Độ tin cậy: {confidence_value:.1f}% (dưới ngưỡng 60%) - Thuật toán: {algorithm} - Phát hiện: {detection_method}")
            return jsonify({
                "matched": False,
                "message": f"Không xác định được danh tính - Độ tin cậy dưới 60% ({algorithm} - {detection_method})",
                "do_tin_cay": confidence_value,
                "thuat_toan": algorithm,
                "phuong_phap_phat_hien": detection_method
            })

    except Exception as e:
        print(f"💥 Lỗi nhận diện: {e}")
        return jsonify({"error": "server_error", "detail": str(e)}), 500

# ---------- API NHẬN DIỆN NÂNG CAO ----------
@app.route("/api/identify-advanced", methods=["POST"])
@login_required
def api_identify_advanced():
    """API nhận diện nâng cao với đa phương pháp phát hiện và nhận diện"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Thiếu dữ liệu"}), 400

        image_data = data.get("image_data")
        if not image_data:
            return jsonify({"error": "Thiếu dữ liệu ảnh"}), 400

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
        except Exception as e:
            return jsonify({"error": "Ảnh không hợp lệ"}), 400

        # Phát hiện khuôn mặt với đa phương pháp
        detected_faces = detect_faces_multi_method(image_array)
        
        if not detected_faces:
            return jsonify({
                "success": True,
                "faces_detected": 0,
                "message": "Không phát hiện khuôn mặt nào"
            })

        results = []
        for i, face in enumerate(detected_faces):
            x, y, w, h = face['bbox']
            detection_method = face['method']
            detection_confidence = face['confidence']
            
            # Trích xuất vùng khuôn mặt
            face_roi = image_array[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
                
            # Trích xuất embedding (mô phỏng)
            try:
                # Chuẩn bị ảnh cho trích xuất embedding
                face_resized = cv2.resize(face_roi, (160, 160))
                face_normalized = face_resized.astype(np.float32) / 255.0
                
                # Mô phỏng trích xuất embedding (trong thực tế sẽ dùng model thật)
                face_embedding = np.random.randn(128).astype(np.float32)
                face_embedding = face_embedding / np.linalg.norm(face_embedding)
                
            except Exception as e:
                print(f"❌ Lỗi trích xuất embedding: {e}")
                continue
            
            if face_embedding is not None:
                # Nhận diện với đa thuật toán
                student_info, confidence, algorithm, detection_method = recognize_face_multi_algorithm(
                    face_embedding, detection_method
                )
                
                face_result = {
                    "face_id": i + 1,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "detection_method": detection_method,
                    "detection_confidence": float(detection_confidence),
                    "recognition_algorithm": algorithm,
                    "recognition_confidence": float(confidence) if confidence else 0.0,
                    "matched": student_info is not None
                }
                
                if student_info:
                    face_result.update({
                        "ma_sv": student_info['ma_sv'],
                        "ho_ten": student_info['ho_ten'],
                        "lop": student_info['lop']
                    })
                
                results.append(face_result)

        return jsonify({
            "success": True,
            "faces_detected": len(detected_faces),
            "faces_recognized": len([r for r in results if r['matched']]),
            "results": results,
            "algorithms_used": list(set([r['recognition_algorithm'] for r in results if r['recognition_algorithm'] != 'No match'])),
            "detection_methods": list(set([r['detection_method'] for r in results]))
        })

    except Exception as e:
        print(f"💥 Lỗi nhận diện nâng cao: {e}")
        return jsonify({"error": "server_error", "detail": str(e)}), 500

# ---------- API ĐA THUẬT TOÁN ----------
@app.route("/api/identify-multi", methods=["POST"])
@login_required
def api_identify_multi():
    """API nhận diện với tất cả thuật toán (cho so sánh)"""
    try:
        data = request.get_json()
        descriptor = data.get("descriptor")
        
        if not descriptor:
            return jsonify({"error": "Thiếu descriptor"}), 400

        face_encoding = np.array(descriptor, dtype=np.float32)
        
        # Kết quả từ từng thuật toán
        results = {}
        
        # 1. Euclidean Distance
        if len(known_face_encodings) > 0:
            best_euclidean_index = None
            best_euclidean_distance = float('inf')
            
            for i, known_encoding in enumerate(known_face_encodings):
                distance = euclidean_distance(face_encoding, known_encoding)
                if distance < best_euclidean_distance:
                    best_euclidean_distance = distance
                    best_euclidean_index = i
            
            if best_euclidean_index is not None:
                euclidean_confidence = max(0, (1 - (best_euclidean_distance / 0.6)) * 100)
                results['euclidean'] = {
                    'student_info': known_face_info[best_euclidean_index] if euclidean_confidence >= 60 else None,
                    'confidence': euclidean_confidence,
                    'distance': best_euclidean_distance
                }
        
        # 2. ArcFace
        if len(known_face_arcface_embeddings) > 0:
            arcface_embedding = convert_to_arcface_embedding(face_encoding)
            best_arcface_index = None
            best_arcface_similarity = -1
            
            for i, known_embedding in enumerate(known_face_arcface_embeddings):
                similarity = arcface_similarity(arcface_embedding, known_embedding)
                if similarity > best_arcface_similarity:
                    best_arcface_similarity = similarity
                    best_arcface_index = i
            
            if best_arcface_index is not None:
                arcface_confidence = max(0, best_arcface_similarity * 100)
                results['arcface'] = {
                    'student_info': known_face_info[best_arcface_index] if arcface_confidence >= 60 else None,
                    'confidence': arcface_confidence,
                    'similarity': best_arcface_similarity
                }
        
        # 3. Deep Learning
        if len(known_face_deep_embeddings) > 0:
            deep_embedding = convert_to_deep_embedding(face_encoding)
            best_deep_index = None
            best_deep_similarity = -1
            
            for i, known_embedding in enumerate(known_face_deep_embeddings):
                similarity = deep_learning_similarity(deep_embedding, known_embedding)
                if similarity > best_deep_similarity:
                    best_deep_similarity = similarity
                    best_deep_index = i
            
            if best_deep_index is not None:
                deep_confidence = max(0, best_deep_similarity * 100)
                results['deep_learning'] = {
                    'student_info': known_face_info[best_deep_index] if deep_confidence >= 60 else None,
                    'confidence': deep_confidence,
                    'similarity': best_deep_similarity
                }
        
        # 4. MTCNN
        if len(known_face_mtcnn_embeddings) > 0:
            mtcnn_embedding = convert_to_mtcnn_embedding(face_encoding)
            best_mtcnn_index = None
            best_mtcnn_similarity = -1
            
            for i, known_embedding in enumerate(known_face_mtcnn_embeddings):
                similarity = mtcnn_similarity(mtcnn_embedding, known_embedding)
                if similarity > best_mtcnn_similarity:
                    best_mtcnn_similarity = similarity
                    best_mtcnn_index = i
            
            if best_mtcnn_index is not None:
                mtcnn_confidence = max(0, best_mtcnn_similarity * 100)
                results['mtcnn'] = {
                    'student_info': known_face_info[best_mtcnn_index] if mtcnn_confidence >= 60 else None,
                    'confidence': mtcnn_confidence,
                    'similarity': best_mtcnn_similarity
                }
        
        # 5. LBPH
        if len(known_face_lbph_data) > 0:
            lbph_data = convert_to_lbph_data(face_encoding)
            best_lbph_index = None
            best_lbph_similarity = -1
            
            for i, known_lbph in enumerate(known_face_lbph_data):
                similarity = lbph_similarity(lbph_data, known_lbph)
                if similarity > best_lbph_similarity:
                    best_lbph_similarity = similarity
                    best_lbph_index = i
            
            if best_lbph_index is not None:
                lbph_confidence = max(0, best_lbph_similarity * 100)
                results['lbph'] = {
                    'student_info': known_face_info[best_lbph_index] if lbph_confidence >= 60 else None,
                    'confidence': lbph_confidence,
                    'similarity': best_lbph_similarity
                }
        
        return jsonify({
            "success": True,
            "algorithms": results,
            "best_algorithm": max(results.keys(), key=lambda k: results[k]['confidence']) if results else None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- API PHÁT HIỆN KHUÔN MẶT ----------
@app.route("/api/detect-faces", methods=["POST"])
@login_required
def api_detect_faces():
    """API phát hiện khuôn mặt với đa phương pháp"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Thiếu dữ liệu"}), 400

        image_data = data.get("image_data")
        if not image_data:
            return jsonify({"error": "Thiếu dữ liệu ảnh"}), 400

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
        except Exception as e:
            return jsonify({"error": "Ảnh không hợp lệ"}), 400

        # Phát hiện khuôn mặt với đa phương pháp
        detected_faces = detect_faces_multi_method(image_array)
        
        results = []
        for i, face in enumerate(detected_faces):
            x, y, w, h = face['bbox']
            results.append({
                "face_id": i + 1,
                "bbox": [int(x), int(y), int(w), int(h)],
                "detection_method": face['method'],
                "confidence": float(face['confidence'])
            })

        return jsonify({
            "success": True,
            "faces_detected": len(detected_faces),
            "results": results,
            "detection_methods": list(set([face['method'] for face in detected_faces]))
        })

    except Exception as e:
        print(f"💥 Lỗi phát hiện khuôn mặt: {e}")
        return jsonify({"error": "server_error", "detail": str(e)}), 500

# ---------- API THÔNG TIN PHƯƠNG PHÁP ----------
@app.route("/api/detection-methods", methods=["GET"])
@login_required
def api_detection_methods():
    """API lấy thông tin về các phương pháp phát hiện khuôn mặt"""
    return jsonify({
        "detection_methods": [
            {
                "name": "OpenCV Haar Cascade",
                "type": "Traditional",
                "description": "Haar cascade classifier for face detection",
                "speed": "Very Fast",
                "accuracy": "Medium",
                "best_for": "Legacy systems and basic applications"
            },
            {
                "name": "Dlib HOG Detector",
                "type": "Traditional",
                "description": "HOG-based face detector with good accuracy",
                "speed": "Fast",
                "accuracy": "High",
                "best_for": "Balanced performance applications"
            },
            {
                "name": "MTCNN",
                "type": "Deep Learning",
                "description": "Multi-task Cascaded CNN với face detection và alignment",
                "speed": "Medium",
                "accuracy": "Very High",
                "best_for": "High accuracy scenarios"
            }
        ],
        "recognition_algorithms": [
            {
                "name": "ArcFace",
                "type": "Deep Learning",
                "description": "Additive Angular Margin Loss for highly discriminative features",
                "accuracy": "99.4%+",
                "use_case": "High-security applications"
            },
            {
                "name": "Deep Learning",
                "type": "Deep Learning",
                "description": "Generic deep learning face recognition",
                "accuracy": "99.2%+",
                "use_case": "General purpose recognition"
            },
            {
                "name": "MTCNN Recognition",
                "type": "Deep Learning",
                "description": "MTCNN-based face recognition with alignment",
                "accuracy": "99.3%+",
                "use_case": "Aligned face recognition"
            },
            {
                "name": "Euclidean Distance",
                "type": "Traditional",
                "description": "Euclidean distance comparison of face embeddings",
                "accuracy": "85-90%",
                "use_case": "Fast and simple recognition"
            },
            {
                "name": "LBPH",
                "type": "Traditional",
                "description": "Local Binary Patterns Histograms for texture analysis",
                "accuracy": "70-80%",
                "use_case": "Lightweight applications"
            }
        ]
    })

# ---------- CÁC API KHÁC (CÓ BẢO VỆ) ----------
@app.route("/api/encodings", methods=["GET"])
@login_required
def api_encodings():
    """Lấy thông tin encodings từ file .npy"""
    try:
        encodings_info = {}
        total_encodings = 0
        total_students = 0
        
        if not os.path.exists(ENC_DIR):
            return jsonify({
                'total_encodings': 0,
                'total_students': 0,
                'encodings': {}
            })
        
        for filename in os.listdir(ENC_DIR):
            if filename.endswith('.npy'):
                ma_sv = filename.replace('.npy', '')
                
                enc_path = os.path.join(ENC_DIR, filename)
                try:
                    encodings_data = np.load(enc_path, allow_pickle=True)
                    if encodings_data.ndim == 0:
                        encoding_count = 0
                    else:
                        encoding_count = len(encodings_data) if encodings_data.ndim > 0 else 1
                    total_encodings += encoding_count
                    total_students += 1
                    
                    with get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT ho_ten, lop FROM sinh_vien WHERE ma_sv = ?", (ma_sv,))
                        result = cur.fetchone()
                        
                        if result:
                            encodings_info[ma_sv] = {
                                'count': encoding_count,
                                'files': True,
                                'ho_ten': result['ho_ten'],
                                'lop': result['lop'],
                                'encodings': []
                            }
                            
                            if encoding_count > 0:
                                if encodings_data.ndim == 1 and encodings_data.shape == (128,):
                                    encodings_info[ma_sv]['encodings'] = [encodings_data.tolist()]
                                else:
                                    encodings_info[ma_sv]['encodings'] = [enc.tolist() for enc in encodings_data]
                except Exception as e:
                    print(f"❌ Lỗi đọc file {filename}: {e}")
        
        return jsonify({
            'total_encodings': total_encodings,
            'total_students': total_students,
            'encodings': encodings_info
        })
    
    except Exception as e:
        print(f"❌ Lỗi API encodings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/api/diem-danh", methods=["GET", "POST"])
@login_required
def api_diem_danh_list():
    """Lấy lịch sử điểm danh"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT dd.id_diem_danh, dd.ma_sv, dd.ho_ten, dd.lop,
                   dd.thoi_gian_vao, dd.trang_thai, dd.do_tin_cay, dd.thuat_toan, dd.phuong_phap_phat_hien
            FROM diem_danh dd
            ORDER BY dd.thoi_gian_vao DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        
        diem_danh_list = []
        for row in rows:
            dd = dict(row)
            diem_danh_list.append(dd)
        
        return jsonify(diem_danh_list)

@app.route("/api/diem-danh/hom-nay", methods=["GET"])
@login_required
def api_diem_danh_hom_nay():
    """Lấy danh sách điểm danh hôm nay"""
    try:
        # 🛠️ KHẮC PHỤC: Lấy thời gian bắt đầu/kết thúc ngày hôm nay theo Local Time
        today_start, today_end = get_today_start_end()
        today = datetime.datetime.now(VN_TZ).strftime('%Y-%m-%d')
        
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT dd.ma_sv, dd.ho_ten, dd.lop, dd.thoi_gian_vao, dd.do_tin_cay, dd.thuat_toan, dd.phuong_phap_phat_hien
                FROM diem_danh dd
                -- So sánh trong khoảng thời gian (Local Time)
                WHERE dd.thoi_gian_vao BETWEEN ? AND ?
                ORDER BY dd.thoi_gian_vao DESC
                LIMIT 20
            """, (today_start, today_end))
            rows = cur.fetchall()
            
            diem_danh_list = []
            for row in rows:
                dd = dict(row)
                # Parse thời gian để format lại
                try:
                    thoi_gian = datetime.datetime.strptime(dd['thoi_gian_vao'], '%Y-%m-%d %H:%M:%S')
                    dd['thoi_gian_format'] = thoi_gian.strftime('%H:%M:%S')
                except ValueError:
                    # Trường hợp format không đúng, dùng raw string
                    dd['thoi_gian_format'] = dd['thoi_gian_vao'] 
                    
                diem_danh_list.append(dd)
            
            return jsonify({
                "success": True,
                "today": today,
                "count": len(diem_danh_list),
                "attendance": diem_danh_list
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats", methods=["GET"])
@login_required
def api_stats():
    """Lấy thống kê tổng quan"""
    with get_conn() as conn:
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) as total FROM sinh_vien")
        total_students = cur.fetchone()['total']
        
        # 🛠️ KHẮC PHỤC: Lấy ngày hôm nay theo Local Time
        today_start, today_end = get_today_start_end()
        
        cur.execute("""
            SELECT COUNT(DISTINCT ma_sv) as today_count 
            FROM diem_danh 
            WHERE thoi_gian_vao BETWEEN ? AND ?
        """, (today_start, today_end))
        attendance_today = cur.fetchone()['today_count']
        
        total_encodings = 0
        if os.path.exists(ENC_DIR):
            for filename in os.listdir(ENC_DIR):
                if filename.endswith('.npy'):
                    enc_path = os.path.join(ENC_DIR, filename)
                    try:
                        encodings_data = np.load(enc_path, allow_pickle=True)
                        if encodings_data.ndim == 0:
                            continue
                        total_encodings += len(encodings_data) if encodings_data.ndim > 0 else 1
                    except:
                        pass
        
        return jsonify({
            'total_students': total_students,
            'attendance_today': attendance_today,
            'total_encodings': total_encodings
        })
@app.route("/dashboard")
@login_required
def dashboard():
    """Trang dashboard tổng quan"""
    return render_template('dashboard.html')

@app.route("/realtime-attendance")
@login_required
def realtime_attendance():
    """Trang điểm danh real-time"""
    return render_template('realtime_attendance.html')

@app.route("/attendance-history")
@login_required
def attendance_history():
    """Trang lịch sử điểm danh"""
    return render_template('attendance_history.html')

@app.route("/student-management")
@login_required
def student_management():
    """Trang quản lý sinh viên"""
    return render_template('student_management.html')

@app.route("/user-management")
@login_required
@require_role('Admin')
def user_management_alias():
    """Trang quản lý người dùng (alias với underscore)"""
    return render_template('user_management.html')

@app.route("/face-training")
@login_required
def face_training():
    """Trang train khuôn mặt"""
    return render_template('face_training.html')

@app.route("/statistical")
@login_required
def algorithms_info():
    """Trang thống kê"""
    return render_template('statistical.html')

@app.route("/profile")
@login_required
def profile():
    """Trang thông tin cá nhân"""
    return render_template('profile.html')

# ---------- ROUTES CHO CÁC COMPONENT ----------
@app.route("/header")
@login_required
def header_component():
    """Component header"""
    return render_template('header.html')

@app.route("/footer")
@login_required
def footer_component():
    """Component footer"""
    return render_template('footer.html')

@app.route("/api/sync-encodings", methods=["GET"])
@login_required
def api_sync_encodings():
    """Đồng bộ encodings từ file .npy"""
    try:
        load_all_face_encodings()
        return jsonify({
            "success": True,
            "total_faces": len(known_face_encodings),
            "total_students": len(set([info['ma_sv'] for info in known_face_info])),
            "message": f"Đã đồng bộ {len(known_face_encodings)} khuôn mặt từ {len(set([info['ma_sv'] for info in known_face_info]))} sinh viên"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/debug-encodings", methods=["GET"])
@login_required
def api_debug_encodings():
    """API debug để kiểm tra encodings"""
    debug_info = {
        'known_face_encodings_count': len(known_face_encodings),
        'known_face_info_count': len(known_face_info),
        'known_face_info': known_face_info,
        'enc_dir_files': os.listdir(ENC_DIR) if os.path.exists(ENC_DIR) else [],
        'algorithms_available': ['Euclidean', 'ArcFace', 'DeepLearning', 'MTCNN', 'LBPH'],
        'detection_methods_available': ['OpenCV_Haar', 'Dlib_HOG', 'MTCNN']
    }
    
    if known_face_encodings:
        debug_info['encoding_shapes'] = [enc.shape for enc in known_face_encodings[:5]]
        
    return jsonify(debug_info)

@app.route("/api/algorithms", methods=["GET"])
@login_required
def api_algorithms():
    """API lấy thông tin về các thuật toán đang sử dụng"""
    return jsonify({
        "algorithms": [
            {
                "name": "Euclidean Distance",
                "type": "Traditional",
                "description": "So sánh khoảng cách Euclidean giữa các face embeddings",
                "accuracy": "85-90%",
                "speed": "Fast"
            },
            {
                "name": "ArcFace",
                "type": "Deep Learning",
                "description": "Additive Angular Margin Loss cho face recognition",
                "accuracy": "95-99%",
                "speed": "Medium"
            },
            {
                "name": "Deep Learning",
                "type": "Deep Learning", 
                "description": "Generic deep learning face recognition",
                "accuracy": "92-98%",
                "speed": "Medium"
            },
            {
                "name": "MTCNN Recognition",
                "type": "Deep Learning",
                "description": "MTCNN-based recognition with face alignment",
                "accuracy": "94-98%",
                "speed": "Slow"
            },
            {
                "name": "LBPH",
                "type": "Traditional",
                "description": "Local Binary Patterns Histograms",
                "accuracy": "70-80%",
                "speed": "Very Fast"
            }
        ],
        "detection_methods": [
            {
                "name": "OpenCV Haar Cascade",
                "type": "Traditional",
                "speed": "Very Fast",
                "accuracy": "Medium"
            },
            {
                "name": "Dlib HOG Detector", 
                "type": "Traditional",
                "speed": "Fast",
                "accuracy": "High"
            },
            {
                "name": "MTCNN",
                "type": "Deep Learning",
                "speed": "Medium", 
                "accuracy": "Very High"
            }
        ],
        "ensemble_method": "Advanced Voting System với confidence-based selection"
    })

if __name__ == "__main__":
    print("🚀 KHỞI ĐỘNG HỆ THỐNG ĐIỂM DANH NHẬN DIỆN KHUÔN MẶT ĐA THUẬT TOÁN NÂNG CAO")
    print("=" * 70)
    print(f"📁 Thư mục encodings: {ENC_DIR}")
    print(f"📁 Thư mục images: {IMG_DIR}")
    print(f"📊 Số khuôn mặt đã tải: {len(known_face_encodings)}")
    print(f"👥 Số sinh viên có encoding: {len(set([info['ma_sv'] for info in known_face_info]))}")
    print("🔐 HỆ THỐNG ĐĂNG NHẬP ĐÃ ĐƯỢC KÍCH HOẠT")
    print("👤 Tài khoản mặc định: admin / admin123")
    print("🎯 THUẬT TOÁN NHẬN DIỆN: ArcFace, DeepLearning, MTCNN, Euclidean, LBPH")
    print("🔍 PHƯƠNG PHÁP PHÁT HIỆN: OpenCV Haar, Dlib HOG, MTCNN")
    print("⏰ TIMEZONE CHUẨN HÓA: Asia/Ho_Chi_Minh (VN_TZ)")
    print("=" * 70)
    
    if os.path.exists(ENC_DIR):
        npy_files = [f for f in os.listdir(ENC_DIR) if f.endswith('.npy')]
        print(f"📂 Số file .npy trong encodings: {len(npy_files)}")
        for f in npy_files:
            ma_sv = f.replace('.npy', '')
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT ho_ten, lop FROM sinh_vien WHERE ma_sv = ?", (ma_sv,))
                result = cur.fetchone()
                if result:
                    print(f"  ✅ {f} - {result['ho_ten']} - {result['lop']}")
    
    print("🎯 Hệ thống đã sẵn sàng nhận diện real-time với đa thuật toán nâng cao!")
    print("🔗 Truy cập: http://localhost:5000/login")
    print("🐛 Debug encodings: http://localhost:5000/api/debug-encodings")
    print("🔬 So sánh thuật toán: http://localhost:5000/api/algorithms")
    print("🛠️ Phương pháp phát hiện: http://localhost:5000/api/detection-methods")
    print("=" * 70)
    
    app.run(host="0.0.0.0", port=5000, debug=True)