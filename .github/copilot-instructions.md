# AI Copilot Instructions for Face Recognition Attendance System

## Project Overview
This is a Vietnamese student attendance system using **real-time face recognition**. It's a Flask backend + HTML/JavaScript frontend app that combines multiple face recognition algorithms for robust student attendance tracking.

**Tech Stack:** Python (Flask), SQLite, PyTorch (ArcFace), OpenCV, face-api.js (browser-based detection)

---

## Architecture & Data Flow

### Backend Structure (`main.py`)
- **Core Layers:**
  1. **Database Layer:** SQLite with three main tables:
     - `sinh_vien` (students) - Ma_SV, Ho_ten, Lop, etc.
     - `diem_danh` (attendance) - Records with timestamp, confidence score, algorithm used
     - `nguoi_dung` (users/teachers) - Vai_tro (Admin/GiangVien/TroGiang)
  
  2. **Face Recognition Layer:** Multi-algorithm approach stored globally:
     ```
     known_face_encodings → Arrays of numpy embeddings
     known_face_arcface_embeddings → PyTorch ArcFace models (512-dim)
     known_face_lbph_data → OpenCV LBPH format
     known_face_mtcnn_embeddings → MTCNN detections
     ```
  
  3. **API Routes:** RESTful endpoints return JSON
     - `/api/identify` - Single face match (POST with base64 image)
     - `/api/train` - Generate face encodings (POST image → save .npy file)
     - `/api/diem-danh` - Record/retrieve attendance
     - `/api/sinh-vien` - CRUD operations for students

### Frontend Architecture
- **Pages in `templates/`:**
  - `header.html` / `footer.html` - Shared navigation & layout
  - `dashboard.html` - Overview stats (requires `/api/stats`)
  - `realtime_attendance.html` - Live camera feed + face detection (uses face-api.js)
  - `face_training.html` - Upload student photos → generate encodings
  - `attendance_history.html` - Search/filter attendance records
  - `student_management.html` - CRUD for `sinh_vien` table
  - `user_management.html` - Admin user management (vai_tro based)
  - `statistical.html` - Analytics dashboard

- **Key JavaScript Integration Points:**
  - `face-api.js` - Real-time browser-based face detection
  - AJAX calls to `/api/*` endpoints with JSON payloads
  - Session management via Flask `session` object

---

## Critical Developer Workflows

### Face Encoding Pipeline (Training)
```
User uploads photo (face_training.html)
  ↓
POST /api/train with base64 image
  ↓
detect_faces_multi_method() - tries Haar Cascade → Dlib → MTCNN
  ↓
Extract face region, normalize to 128x128
  ↓
Generate embeddings via:
  • ArcFace (512-dim) → convert_to_arcface_embedding()
  • LBPH → convert_to_lbph_data()
  • Deep Learning → convert_to_deep_embedding()
  ↓
Save all formats to: uploads/encodings/{ma_sv}.npy (numpy array)
  ↓
Reload all encodings via load_all_face_encodings()
```
**Key Files:** `ENC_DIR = "uploads/encodings/"` | `.npy` files are numpy arrays, not plain text

### Attendance Recognition (Real-time)
```
Live camera frame (face-api.js in browser)
  ↓
User clicks "Nhận diện" button
  ↓
POST /api/identify with face descriptor (128-dim from face-api.js)
  ↓
recognize_face_multi_algorithm() compares with all stored encodings
  ↓
Returns: {ma_sv, ho_ten, do_tin_cay, thuat_toan} (confidence + algorithm)
  ↓
Record to diem_danh table with thoi_gian_vao & status
```
**Matching Thresholds:**
- Euclidean: `distance < 0.6` = match
- Cosine: `similarity > 0.5` = match
- ArcFace: `similarity > 0.7` = match

### Running the Application
```bash
# Start Flask server
python main.py  # Runs on http://localhost:5000

# Database initialization
# Auto-runs on startup via ensure_db()
# Creates tables & inserts default admin user (username: admin, password: admin123)
```

---

## Project-Specific Patterns & Conventions

### Naming Conventions
- **Vietnamese identifiers** used throughout (ma_sv = student ID, ho_ten = full name, etc.)
- **Table/field names:** snake_case in Vietnamese (sinh_vien, diem_danh, thoi_gian_vao)
- **Routes use hyphens:** `/realtime-attendance`, `/student-management`

### Time Handling - **CRITICAL**
- **Always use `now_local_time_for_db()`** for database timestamps (not `datetime.now()`)
- All times are in **Asia/Ho_Chi_Minh timezone** (Vietnam)
- Format: `'%Y-%m-%d %H:%M:%S'`
- Use `get_today_start_end()` for daily attendance queries

### Database Patterns
- **Connection Management:** Use `@contextmanager get_conn()` for all DB access
- **Row Factory:** `sqlite3.Row` for dict-like access (e.g., `result['ma_sv']`)
- **Migration Support:** `add_column_if_not_exists()` for schema updates (don't break existing DBs)

### Authentication & Authorization
- **Session-based:** Uses Flask `session['user_id']` and `session['vai_tro']`
- **Decorators:** `@login_required` & `@require_role('Admin')` protect routes
- **Password:** SHA-256 hash via `hash_password()` function
- **Roles:** 'Admin', 'GiangVien' (lecturer), 'TroGiang' (TA)

### Face Recognition Algorithms (Tier System)
When implementing face matching, use **fallback strategy**:
```python
1. Try ArcFace (modern, best accuracy)
2. Fallback to LBPH (lightweight)
3. Fallback to Euclidean distance (basic, fast)
```
Each attendance record stores `thuat_toan` (algorithm used) & `phuong_phap_phat_hien` (detection method).

### Image & File Handling
- **Image uploads:** `POST /api/train` expects **base64-encoded** image data
- **Encoding storage:** `.npy` files in `uploads/encodings/`
- **Student photos:** `uploads/images/` (path created but not fully used yet)
- **Models:** Pre-trained files in `static/models/` (face-api.js + OpenCV models)

---

## Integration Points & External Dependencies

### Face Detection Methods (Multi-approach)
- **OpenCV Haar Cascades:** Fast, baseline
- **Dlib HOG Detector:** More robust
- **MTCNN:** Most accurate (requires facenet-pytorch)
- **browser-based:** face-api.js runs in browser (not backend)

### Key Python Dependencies & Their Roles
```
Flask              → Web framework, routing
SQLite3            → Database (built-in)
numpy              → Embedding arrays & calculations
PyTorch + torch.nn → ArcFace model architecture
scikit-learn       → Cosine similarity calculations
opencv-python      → Image processing, Haar cascades, LBPH
Pillow (PIL)       → Image format handling
facenet-pytorch    → MTCNN face detector
tensorflow         → (Imported but usage unclear)
pytz               → Timezone handling (Vietnam)
```

### Frontend JavaScript Library
- **face-api.js** - Executes face detection in browser, returns descriptors (128-dim vectors)
  - Models loaded from `static/models/face-api.min.js`
  - Requires model files (weights manifests) in same directory

---

## Common Tasks & Code Patterns

### Adding a New Route
```python
@app.route("/new-feature", methods=["GET", "POST"])
@login_required  # Add auth
@require_role('Admin')  # Optional: role-based
def new_feature():
    if request.method == "POST":
        data = request.get_json()
        with get_conn() as conn:
            cur = conn.cursor()
            # DB operation
        return jsonify({"status": "success"})
    return render_template("new_feature.html")
```

### Querying Students
```python
with get_conn() as conn:
    cur = conn.cursor()
    cur.execute("SELECT * FROM sinh_vien WHERE lop = ?", (class_name,))
    students = cur.fetchall()  # Returns list of sqlite3.Row objects
    for student in students:
        print(student['ho_ten'])  # Dict-like access
```

### Saving Face Encodings
```python
# Always save as .npy file named by ma_sv
encoding_path = os.path.join(ENC_DIR, f"{ma_sv}.npy")
np.save(encoding_path, encoding_array)

# Then reload all encodings to update global lists
load_all_face_encodings()
```

### Recording Attendance
```python
with get_conn() as conn:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO diem_danh 
        (id_sv, ma_sv, ho_ten, lop, thoi_gian_vao, do_tin_cay, thuat_toan)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (id_sv, ma_sv, ho_ten, lop, now_local_time_for_db(), confidence, algorithm_name))
    conn.commit()
```

---

## Code Structure Insights

### Global State Management
- Face encodings loaded once at startup → stored in global lists for fast lookup
- Call `load_all_face_encodings()` after training new faces
- Globals are thread-safe enough for Flask (single-threaded by default)

### Error Handling Patterns
- Use try/except around model initialization → print debug messages (not logging)
- Return `{"status": "error", "message": "..."}` JSON for API errors
- Database errors use context manager for automatic rollback on exception

---

## Common Pitfalls to Avoid

1. **Don't use `datetime.now()` directly** - Always use `now_local_time_for_db()` for DB timestamps
2. **Don't mix encoding formats** - Always save/load as `.npy` (numpy arrays), not pickle
3. **Don't forget to reload encodings** - After training new faces, call `load_all_face_encodings()`
4. **Don't break Jinja2 templates** - Keep `{% %}` blocks intact when refactoring HTML
5. **Don't hardcode paths** - Use `os.path.join()` and pre-defined `*_DIR` constants
6. **Session validation missing** - Check `session.get('user_id')` in protected routes
7. **Image base64 format** - Frontend sends images as base64 strings (data:image/jpeg;base64,...)

---

## Testing & Debugging

**No automated tests currently exist.** Manual testing:
```bash
# Test API endpoint
curl -X POST http://localhost:5000/api/identify \
  -H "Content-Type: application/json" \
  -d '{"encoding": [...]}'

# Test database
sqlite3 attendance.db "SELECT COUNT(*) FROM diem_danh;"

# Check loaded encodings
# Debug route: GET /api/debug-encodings
```

---

## Key Files to Reference
- `main.py` (2113 lines) - All backend logic
- `templates/realtime_attendance.html` - Face detection UX pattern
- `templates/face_training.html` - Encoding pipeline UX
- `static/models/` - Face detection models for browser
- `.github/copilot-instructions.md` - This file (you are here)
