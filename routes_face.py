import time
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..core.config import settings
from ..models.schemas import FaceCompareResponse
from ..services.face_service import compare_two_faces
from ..services.history_service import save_history

router = APIRouter(prefix="/face", tags=["Biometric Face Comparison"])

@router.post("/compare", response_model=FaceCompareResponse)
async def compare_faces(
    image_a: UploadFile = File(..., description="Ảnh nguồn (Reference)"),
    image_b: UploadFile = File(..., description="Ảnh đối chiếu (Query)")
):
    start_time = time.time()

    # Validate file types
    for f in [image_a, image_b]:
        if f.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Định dạng tệp '{f.filename}' ({f.content_type}) không được hỗ trợ. Hỗ trợ JPG, PNG, WebP, BMP."
            )

    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    for content, f in [(bytes_a, image_a), (bytes_b, image_b)]:
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Tệp '{f.filename}' vượt quá giới hạn ({settings.MAX_UPLOAD_SIZE // (1024*1024)}MB)."
            )

    # Decode with OpenCV
    arr_a = np.frombuffer(bytes_a, np.uint8)
    arr_b = np.frombuffer(bytes_b, np.uint8)
    img_a_bgr = cv2.imdecode(arr_a, cv2.IMREAD_COLOR)
    img_b_bgr = cv2.imdecode(arr_b, cv2.IMREAD_COLOR)

    if img_a_bgr is None or img_b_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Không thể giải mã một trong hai hình ảnh tải lên. Vui lòng kiểm tra lại tính toàn vẹn của tệp."
        )

    try:
        comparison_res = compare_two_faces(
            img_a_bgr,
            img_b_bgr,
            image_a.filename or "sample_a.jpg",
            image_b.filename or "sample_b.jpg"
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        model_info = {
            "engine": "OpenCV YuNet DNN (Face Detection & Landmarks) + SFace DNN (128-D Deep Feature Embedding)",
            "metric": "Cosine Similarity (cv2.FaceRecognizerSF_FR_COSINE) & L2 Distance (cv2.FaceRecognizerSF_FR_NORM_L2)",
            "benchmark_threshold": "SFace Official Match Threshold: Cosine >= 0.363 / L2 Distance <= 1.128",
            "limitations": "Kết quả phân tích sinh trắc học dựa trên không gian nhúng mạng nơ-ron sâu SFace (128 chiều) và mô hình phát hiện khuôn mặt YuNet. Chỉ mang tính chất nghiên cứu khoa học tham khảo, không kết luận danh tính tuyệt đối theo pháp lý."
        }

        response_payload = {
            "success": True,
            "sample_a": comparison_res["sample_a"],
            "sample_b": comparison_res["sample_b"],
            "metrics": comparison_res["metrics"],
            "model_info": model_info,
            "processing_time_ms": elapsed_ms
        }

        # Save to history
        verdict = comparison_res["metrics"].verdict_label_vi
        score = comparison_res["metrics"].composite_match_score
        summary = f"Đối soát 1:1: {verdict} (Điểm tương đồng: {int(score * 100)}%)"
        save_history("face_compare", f"So sánh khuôn mặt: {image_a.filename} vs {image_b.filename}", summary, response_payload)

        return FaceCompareResponse(**response_payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống khi đối soát sinh trắc học: {str(e)}")
