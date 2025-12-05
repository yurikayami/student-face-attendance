@templates/header.html @templates/footer.html @templates/dashboard.html @templates/student_management.html @templates/index.html @templates/login.html @templates/realtime_attendance.html @templates/attendance_history.html @templates/profile.html @templates/face_training.html @templates/statistical.html @templates/user_management.html

CONTEXT: 
Dự án này là hệ thống điểm danh bằng khuôn mặt dùng Flask. Hiện tại giao diện đang viết bằng CSS thuần rất lộn xộn. Tôi muốn viết lại toàn bộ UI bằng Tailwind CSS để chuyên nghiệp hơn.

TASK: Hãy Refactor (tái cấu trúc) toàn bộ giao diện Frontend theo các bước sau:

Bước 1: Dọn dẹp & Chuẩn bị (QUAN TRỌNG)
1. XÓA file `templates/ealtime_attendance.html`. (Lý do: Đây là file rác bị đặt tên sai, nội dung trùng lặp với `profile.html`. File `templates/realtime_attendance.html` chuẩn đã tồn tại, tuyệt đối không ghi đè lên nó).
2. XÓA các file rác: `DUTRU.HTML`, `*backup.html`.
3. Trong `templates/header.html`:
   - Xóa hết các thẻ <link rel="stylesheet"> trỏ đến file CSS cũ.
   - Thêm Tailwind CSS qua CDN: <script src="https://cdn.tailwindcss.com"></script>
   - Thêm Font Awesome: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

Bước 2: Viết lại Giao diện (Modern UI)
Hãy sửa code HTML của TẤT CẢ các file trong @templates để sử dụng class của Tailwind CSS.
Yêu cầu phong cách thiết kế (Design System):
- Màu chủ đạo: Indigo (indigo-600) và Slate (slate-100 làm nền).
- Layout: 
  - **Sidebar bên trái (Left Sidebar):** Có khả năng thu gọn (Collapsible), cố định (fixed height full). Khi thu gọn chỉ hiện Icon, khi mở hiện cả Text. Có nút Toggle chuyển đổi trạng thái.
  - **Main Content:** Nằm bên phải, tự động co giãn (`flex-1`, `ml-64` hoặc điều chỉnh theo trạng thái sidebar).
- Components:
  - Form: Input có bo góc (rounded-lg), focus ring (ring-2 ring-indigo-500).
  - Bảng (Table): Header màu xám nhạt (bg-gray-50), dòng chẵn/lẻ (even:bg-gray-50), có hover effect.
  - Button: Gradient (bg-gradient-to-r from-indigo-500 to-purple-600), bo góc, có hiệu ứng hover.
  - Card: Shadow mềm (shadow-lg), bo góc (rounded-xl), nền trắng (bg-white).
  - Badge/Alert: Dùng màu sắc rõ ràng (Green cho thành công, Red cho lỗi).
  - Modal: Overlay mờ (bg-gray-500 bg-opacity-75), nội dung căn giữa.

Bước 3: Bảo toàn Logic Flask (BẮT BUỘC)
- Tuyệt đối GIỮ NGUYÊN các thẻ logic của Jinja2 như `{% block content %}`, `{{ url_for(...) }}`, `{% for item in items %}`.
- Không được làm mất tính năng hiển thị dữ liệu của Flask.
- Giữ nguyên các ID của thẻ HTML (ví dụ: `id="video"`, `id="captureVideo"`) để JavaScript hoạt động đúng.

OUTPUT:
Hãy viết lại code đầy đủ cho các file chính: `header.html`, `footer.html`, `dashboard.html`, `student_management.html`, `realtime_attendance.html`, `attendance_history.html`, `profile.html`, `face_training.html`, `statistical.html` và `user_management.html`.
