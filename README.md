# AURA BIOSTUDIO // HỆ THỐNG THỊ GIÁC MÁY TÍNH & SINH TRẮC HỌC KHUÔN MẶT ĐA TẦNG

> **Dự án nghiên cứu ứng dụng Trí tuệ Nhân tạo trong phân tích hình ảnh và hỗ trợ nhận diện con người thông qua các đặc điểm sinh trắc học thể hiện trên hình ảnh.**
> Thiết kế chuẩn **Light Studio** (Ivory White / Obsidian Black / Cobalt Blue Accent) kết nối trực tiếp với Backend **Python FastAPI + OpenCV Deep Learning (YuNet & SFace ONNX)**.

---

## 🏛️ 1. TRIẾT LÝ THIẾT KẾ LIGHT STUDIO

- **Bảng màu định danh:** Nền trắng ngà (`#F9F9F6`), đen obsidian (`#111827`), xám kim loại và điểm nhấn Cobalt Blue (`#2563EB`) cùng Emerald (`#10B981`).
- **Typography hiện đại:** Phân cấp rõ nét giữa kiểu chữ biên tập đọc báo cáo và phông chữ đơn không gian (`JetBrains Mono`) dành cho các tham số đo lường, mã băm SHA-256, tọa độ pixel và vector nhúng.
- **Hệ thống nhận diện thương hiệu:** Biểu trưng Logo gồm **bốn mảnh ghép khớp nối (Jigsaw Puzzle 2x2)** với hiệu ứng chuyển động xoay, phân tách và neo đậu mượt mà vào thanh điều hướng.
- **Bố cục chuyên biệt:** Mỗi công cụ sở hữu một giao diện đặc thù phù hợp hoàn toàn với nghiệp vụ thực tế, không dùng chung khuôn mẫu đơn điệu.
- **Tuân thủ đạo đức AI:** Không giả lập kết quả AI, không tuyên bố độ chính xác tuyệt đối, hiển thị rõ ràng ngưỡng tham chiếu khoa học và khuyến cáo pháp lý.

---

## 🖥️ 2. HỆ THỐNG CÁC MÀN HÌNH CHUYÊN BIỆT

1. **Màn hình khởi động (Splash Screen):**
   - Logo 4 mảnh ghép hình học rời rạc tự động xoay, bay khớp nối chính xác thành một thể thống nhất và di chuyển lên góc trái thanh Header.
   - Nút "Khởi động" trên thanh điều hướng cho phép xem lại hiệu ứng bất kỳ lúc nào.
2. **Trang giới thiệu dự án (`/`):**
   - Tiêu đề biên tập, mô tả chuyên sâu về thị giác máy tính và sinh trắc học.
   - Trực quan hóa quét laser, lưới điểm mốc và các trụ cột công nghệ.
3. **Trung tâm lựa chọn công cụ (`/hub`):**
   - 4 thẻ làm việc chuyên biệt truy cập nhanh: Phân tích hình ảnh, Phân tích văn bản, Đối soát sinh trắc học 1:1, và Nhật ký lịch sử SQLite.
4. **Trang phân tích hình ảnh (`/vision`):**
   - Khung hình ảnh tương tác: Phóng to/thu nhỏ, Lưới trắc địa quang học, Tâm ngắm vi tính, Bộ lọc kênh màu (RGB, Thang xám, Đảo màu, Tương phản).
   - Đo lường thực tế: Độ phân giải, Tỷ lệ khung hình, Dung lượng, Mã băm SHA-256 toàn vẹn.
   - Phân tích độ sắc nét tiêu cự bằng biến thiên Laplacian (Laplacian Variance).
   - Quét mạng nơ-ron YuNet DNN phát hiện vị trí khuôn mặt, ước tính độ tuổi sọ não và hiển thị bounding box tương tác.
   - Giám định bối cảnh thị giác AI và mô-đun mở rộng kiểm tra tính xác thực (AI-Generated Image Forensics).
5. **Trang phân tích văn bản (`/nlp`):**
   - Soạn thảo văn bản điều tra, lời khai nhân chứng hoặc quy chuẩn sinh trắc học.
   - Đo lường viễn trắc thời gian thực: Đếm ký tự, từ ngữ, số đoạn, ước tính Token AI và thời gian đọc.
   - 4 chế độ xử lý: Trích xuất thực thể sinh trắc học (NER: họ tên, tuổi, chiều cao, sẹo, mắt, mũi), Tâm lý & ngữ thái nhân chứng, Đánh giá tính nhất quán logic, Tóm lược hồ sơ nhân thân.
6. **Trang đối soát sinh trắc học 1:1 (`/biometrics`):**
   - Đặt hai khung ảnh đối xứng: Mẫu A (Reference Anchor) và Mẫu B (Target Probe Query).
   - Hai chế độ quan sát: **Song song (Side-by-Side)** với khung nhận diện SVG động và **Thanh trượt chia đôi (Split Slider)** chồng lớp trực quan.
   - Thuật toán trích xuất đặc trưng sâu **OpenCV SFace DNN (128 chiều)**.
   - Đánh giá chỉ số: Độ tương đồng Cosine (Ngưỡng chuẩn `≥ 0.363`), Khoảng cách Euclid L2 (Ngưỡng chuẩn `≤ 1.128`), Khoảng cách liên đồng tử (IPD), Sai biệt độ tuổi giải phẫu.
7. **Trang nhật ký & lịch sử phân tích (`/history`):**
   - Lưu trữ tự động toàn bộ phiên phân tích vào cơ sở dữ liệu SQLite (`biometric_history.db`).
   - Lọc theo danh mục, tìm kiếm từ khóa, xem chi tiết cấu trúc JSON phản hồi thực tế và thao tác xóa bản ghi.

---

## 🔬 3. CƠ SỞ KHOA HỌC & MÔ HÌNH TÍCH HỢP

| Thành phần | Công nghệ / Mô hình | Đặc tả kỹ thuật & Ngưỡng khoa học |
| :--- | :--- | :--- |
| **Phát hiện khuôn mặt** | **OpenCV YuNet DNN** | Mô hình nơ-ron tích chập nhẹ, trích xuất bounding box và 5 điểm mốc (2 mắt, mũi, 2 khóe miệng). |
| **Trích xuất đặc trưng** | **OpenCV SFace DNN** | Không gian nhúng 128 chiều (128-D L2-normalized embeddings) huấn luyện trên Casia-WebFace, kiểm định LFW (99.60% @ FAR 0.001). |
| **Độ tương đồng Cosine** | `cv2.FaceRecognizerSF_FR_COSINE` | **Ngưỡng kết luận cùng đối tượng:** `Cosine Similarity >= 0.363` |
| **Khoảng cách Euclid L2**| `cv2.FaceRecognizerSF_FR_NORM_L2` | **Ngưỡng kết luận cùng đối tượng:** `L2 Distance <= 1.128` |
| **Độ sắc nét hình ảnh** | **Biến thiên toán tử Laplacian** | `Var(Laplacian) > 250`: Rõ nét; `80 - 250`: Đạt chuẩn; `< 80`: Mờ / Thiếu nét. |
| **Ước tính độ tuổi** | **Craniofacial Geometry** | Phân tích tỷ lệ co giãn giữa chiều rộng gò má, chiều cao khuôn mặt và khoảng cách đồng tử - nhân trung. |
| **Bối cảnh AI & NLP** | **Google Gemini AI (Tùy chọn)** | Mô hình `gemini-1.5-flash` phân tích bối cảnh thị giác sâu và bóc tách thực thể ngôn ngữ tự nhiên. |

---

## 🛠️ 4. KIẾN TRÚC DỰ ÁN

```
biometric-ai-studio/
├── frontend/                     # Giao diện React 19 + Vite + React Router
│   ├── public/
│   │   ├── samples/              # Ảnh chân dung thực nghiệm mẫu chuẩn (portrait_a, b, c)
│   │   └── favicon.svg
│   ├── src/
│   │   ├── components/
│   │   │   ├── JigsawLogo.jsx             # Logo 4 mảnh ghép khớp nối giải đố
│   │   │   ├── SplashIntro.jsx            # Màn hình hoạt ảnh xuất hiện logo
│   │   │   ├── HeaderNav.jsx              # Thanh điều hướng React Router + Badge
│   │   │   ├── StatusBadge.jsx            # Giám sát trạng thái Backend tự động
│   │   │   ├── LandingView.jsx            # Màn hình giới thiệu tổng quan
│   │   │   ├── WorkspaceHub.jsx           # Trung tâm điều phối 4 công cụ
│   │   │   ├── ImageAnalysisStudio.jsx    # Không gian phân tích hình ảnh quang học
│   │   │   ├── TextAnalysisStudio.jsx     # Không gian phân tích văn bản NLP
│   │   │   ├── BiometricAnalysisStudio.jsx# Không gian đối soát sinh trắc học 1:1
│   │   │   ├── AnalysisHistoryView.jsx    # Màn hình lịch sử & nhật ký SQLite
│   │   │   └── Footer.jsx                 # Chân trang Light Studio
│   │   ├── hooks/
│   │   │   └── useBackendHealth.js        # Hook tự động ping kiểm tra sức khỏe API
│   │   ├── services/
│   │   │   └── api.js                     # API Client chuẩn hóa lỗi và kết nối FastAPI
│   │   ├── utils/
│   │   │   └── sampleImages.js            # Tiện ích chuyển đổi ảnh sang File Object
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── vercel.json               # Cấu hình định tuyến SPA khi deploy Vercel
│   ├── .env.example
│   └── package.json
│
├── backend/                      # Máy chủ API Python FastAPI
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes_face.py    # POST /api/face/compare (Đối soát 1:1)
│   │   │   ├── routes_image.py   # POST /api/image/analyze (Phân tích ảnh)
│   │   │   ├── routes_text.py    # POST /api/text/analyze (NLP văn bản)
│   │   │   └── routes_history.py # GET, DELETE /api/history (Kho lịch sử)
│   │   ├── core/
│   │   │   ├── config.py         # Cấu hình biến môi trường, CORS, upload limits
│   │   │   ├── face_detection_yunet.onnx      # Trọng số YuNet Face Detector
│   │   │   └── face_recognition_sface.onnx    # Trọng số SFace Face Recognizer
│   │   ├── models/
│   │   │   └── schemas.py        # Pydantic Schemas đồng nhất dữ liệu
│   │   ├── services/
│   │   │   ├── face_service.py   # Triển khai mô hình YuNet & SFace
│   │   │   ├── vision_service.py # Xử lý quang học, băm SHA-256, Gemini Vision
│   │   │   ├── nlp_service.py    # Thống kê từ vựng, trích xuất NER, Gemini NLP
│   │   │   └── history_service.py# Cơ sở dữ liệu SQLite lưu vết lịch sử
│   │   └── main.py               # Điểm khởi chạy FastAPI, CORS, Lifespan
│   ├── Dockerfile                # Docker container hóa backend triển khai Cloud
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
│
├── .gitignore                    # Chặn rò rỉ khóa bí mật, file venv, database
└── README.md
```

---

## ⚡ 5. HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN

### Bước 1: Khởi động Backend (Python FastAPI)

1. Mở cửa sổ dòng lệnh tại thư mục `backend/`:
   ```bash
   cd backend
   ```
2. Cài đặt các gói phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
3. Tạo file cấu hình môi trường `.env` từ file mẫu:
   ```bash
   cp .env.example .env
   ```
   *(Tùy chọn: Nhập `GEMINI_API_KEY` nếu muốn kích hoạt tính năng Gemini Vision & NLP sâu. Nếu không nhập, toàn bộ tính năng phân tích quang học, trắc địa sọ não, nhận diện khuôn mặt YuNet và đối soát sinh trắc học SFace vẫn hoạt động độc lập 100%).*

4. Khởi chạy máy chủ FastAPI:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   - API Docs tương tác (Swagger UI): `http://127.0.0.1:8000/docs`
   - Health check: `http://127.0.0.1:8000/api/health`

---

### Bước 2: Khởi động Frontend (React + Vite)

1. Mở một cửa sổ dòng lệnh mới tại thư mục `frontend/`:
   ```bash
   cd frontend
   ```
2. Cài đặt các gói phụ thuộc Node:
   ```bash
   npm install
   ```
3. Khởi chạy server phát triển Vite:
   ```bash
   npm run dev
   ```
4. Truy cập giao diện ứng dụng tại:
   ```
   http://localhost:5173/
   ```

---

## 🌐 6. HƯỚNG DẪN TRIỂN KHAI THỰC TẾ (PRODUCTION DEPLOYMENT)

### A. Triển khai Frontend lên Vercel
1. Đưa mã nguồn lên GitHub repository.
2. Đăng nhập [Vercel](https://vercel.com/) và import thư mục `frontend/` (Root Directory: `frontend`).
3. Cấu hình biến môi trường trên Vercel:
   - `VITE_API_URL`: Điền URL Backend đã triển khai (ví dụ: `https://your-api.onrender.com/api`).
4. File `frontend/vercel.json` đã được cấu hình sẵn tính năng `rewrites` cho React Router SPA. Nhấn **Deploy**.

### B. Triển khai Backend lên Render / Railway / Fly.io / VPS
Dự án đã tích hợp sẵn `backend/Dockerfile` chuẩn hóa với môi trường Python 3.11-slim cùng các thư viện hệ thống cần thiết cho OpenCV (`libgl1-mesa-glx`, `libglib2.0-0`).

1. **Trên Render / Railway:**
   - Tạo New Web Service từ GitHub repository.
   - Chọn thư mục gốc là `backend/`.
   - Chọn kiểu triển khai: **Docker** (hoặc Python với Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`).
   - Cấu hình Environment Variables:
     - `PORT`: `8000`
     - `GEMINI_API_KEY`: Khóa API Google AI Studio (nếu có)
     - `GEMINI_MODEL`: `gemini-1.5-flash`
2. Sau khi deploy thành công, cập nhật URL của backend vào biến `VITE_API_URL` của frontend.

---

## ⚖️ 7. TUÂN THỦ BẢO MẬT & ĐẠO ĐỨC AI (ETHICAL AI & PRIVACY)

1. **Bảo vệ khóa bí mật:** Không lưu API Key ở client-side; tất cả tác vụ AI bên ngoài đều được trung chuyển và bảo vệ qua backend.
2. **Quyền riêng tư dữ liệu sinh trắc học:** Hệ thống không lưu trữ hình ảnh khuôn mặt của người dùng lâu dài; ảnh tải lên chỉ được xử lý tạm thời trong bộ nhớ RAM khi thực thi đối soát và trích xuất vector.
3. **Giới hạn diễn giải khoa học:** Kết quả so sánh sinh trắc học dựa trên toán tử không gian nhúng của mô hình SFace và tỷ lệ hình học sọ não. Kết quả mang tính chất tham khảo kỹ thuật trong nghiên cứu, **tuyệt đối không sử dụng làm căn cứ pháp lý duy nhất để kết luận danh tính hoặc quy kết hành vi vi phạm.**
