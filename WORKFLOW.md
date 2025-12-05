# WORKFLOW: Nâng cấp Hệ thống Điểm danh Face Recognition

Tài liệu này phác thảo quy trình thực hiện các tính năng mới dựa trên `TODO.md`, chia thành các giai đoạn (Phases) để đảm bảo tính logic và ổn định của hệ thống.

(lưu lý quan trọng mỗi lần xong 1 Phase là phải commit dựa án dó )

## Phase 1: Quản lý Hình ảnh & Giao diện (Avatar & Ảnh điểm danh)
**Mục tiêu:** Hiển thị trực quan người dùng và bằng chứng điểm danh.

- [ ] **1.1. Cập nhật Cơ sở dữ liệu (CSDL):**
    - [ ] Thêm trường `avatar_path` vào bảng Sinh viên (Student).
    - [ ] Thêm trường `capture_image_path` vào bảng Lịch sử điểm danh (Attendance).
- [ ] **1.2. Logic "Thêm sinh viên":**
    - [ ] Khi train/thêm sinh viên mới, tự động lưu hình ảnh khuôn mặt đầu tiên phát hiện được làm Avatar.
    - [ ] Lưu file ảnh vào thư mục `static/uploads/avatars/`.
- [ ] **1.3. Logic "Điểm danh":**
    - [ ] Khi nhận diện thành công, lưu ảnh chụp (snapshot) tại thời điểm đó.
    - [ ] Lưu file ảnh vào thư mục `static/uploads/attendance_captures/`.
- [ ] **1.4. Cập nhật Frontend:**
    - [ ] `student_management.html`: Hiển thị Avatar bên cạnh tên sinh viên.
    - [ ] `attendance_history.html`: Hiển thị ảnh chụp thực tế trong bảng lịch sử.

---

## Phase 2: Quản lý Buổi học (Class Session)
**Mục tiêu:** Tạo dữ liệu nền để so sánh thời gian điểm danh. Hệ thống sẽ hoạt động theo **Lịch cố định hàng tuần** (Weekly Schedule).

- [ ] **2.1. Cập nhật CSDL:**
    - [ ] Tạo bảng mới `ClassSession` để lưu thông tin buổi học.
    - [ ] Các trường cần thiết: 
        - `id`: Primary Key
        - `subject_name`: Tên môn học
        - `class_name`: Tên lớp
        - `day_of_week`: **Bắt buộc** (0=Thứ 2, 1=Thứ 3, ... hoặc String "Monday", "Tuesday")
        - `start_time`: Giờ bắt đầu (Format HH:MM)
        - `end_time`: Giờ kết thúc (Format HH:MM)
- [ ] **2.2. Backend API:**
    - [ ] Tạo API để Thêm/Sửa/Xóa buổi học (`/api/add_session`, ...).
    - [ ] Tạo API để lấy danh sách buổi học hiện có.
- [ ] **2.3. Frontend - Tạo Form:**
    - [ ] Thêm giao diện "Quản lý Buổi học".
    - [ ] Form nhập liệu: Tên môn, Tên lớp, Thứ trong tuần (Dropdown), Giờ bắt đầu, Giờ kết thúc.

---

## Phase 3: Logic Nghiệp vụ Điểm danh (Thời gian & Trạng thái)
**Mục tiêu:** Xử lý logic Đi trễ/Đúng giờ và ngăn chặn điểm danh sai thời điểm.

- [ ] **3.1. Xây dựng Logic "Tìm Lớp Học":**
    - [ ] Lấy thời gian hiện tại (Thứ, Giờ, Phút).
    - [ ] Tìm trong DB xem có lớp nào đang diễn ra hoặc sắp diễn ra không.
    - [ ] **Logic Buffer Time (Thời gian đệm):** Chỉ cho phép điểm danh sớm tối đa **X phút** (ví dụ: 30 phút) trước giờ học.
    - [ ] **Xử lý ngoại lệ:** Nếu không tìm thấy lớp nào phù hợp -> Trả về thông báo: **"Hiện không có lớp học nào diễn ra"** và KHÔNG ghi nhận điểm danh.
- [ ] **3.2. Cập nhật Logic Ghi nhận:**
    - [ ] **Trường hợp 1 (Đúng giờ):** 
        - Thời gian: `(Start Time - 30 phút)` <= `Check-in` <= `Start Time`
        - Trạng thái: **"Đã điểm danh" (Present)**
    - [ ] **Trường hợp 2 (Đi trễ):** 
        - Thời gian: `Start Time` < `Check-in` <= `End Time`
        - Trạng thái: **"Đi trễ" (Late)**
    - [ ] **Trường hợp 3 (Quá sớm hoặc Đã hết giờ):** 
        - Từ chối ghi nhận hoặc chỉ ghi log cảnh báo.
- [ ] **3.3. Hiển thị trạng thái:**
    - [ ] Cập nhật màu sắc/nhãn trạng thái trên giao diện `Real-time` và `Lịch sử`.

---

## Phase 4: Tái cấu trúc & Tối ưu (Refactor)
**Mục tiêu:** Sắp xếp lại code cho gọn gàng theo yêu cầu "Cấu trúc lại mã nguồn".

- [ ] **4.1. Tách file `main.py`:**
    - [ ] Chuyển cấu hình DB và Models sang file riêng (ví dụ: `models.py`).
    - [ ] Tách các route xử lý view (render template) và route xử lý API sang các file riêng (Blueprints) nếu cần thiết.
- [ ] **4.2. Kiểm tra luồng:**
    - [ ] Test toàn bộ luồng từ Thêm sinh viên -> Tạo buổi học -> Điểm danh -> Xem báo cáo.
