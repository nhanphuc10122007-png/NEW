from fastapi import APIRouter, UploadFile, File, HTTPException
from ..core.config import settings
from ..models.schemas import ImageAnalysisResponse
from ..services.vision_service import process_image_full
from ..services.history_service import save_history

router = APIRouter(prefix="/image", tags=["Image Analysis"])

@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng tệp không được hỗ trợ ({file.content_type}). Hỗ trợ JPG, PNG, WebP, BMP."
        )

    # Read bytes
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Dung lượng tệp vượt quá giới hạn cho phép ({settings.MAX_UPLOAD_SIZE // (1024*1024)}MB)."
        )

    try:
        result = process_image_full(content, file.filename or "uploaded_image.jpg", file.content_type)
        
        # Save to history
        faces_count = result["face_detection"]["faces_count"]
        summary = f"Phát hiện {faces_count} khuôn mặt. Độ sắc nét: {result['metadata'].blur_verdict}."
        save_history("image_analysis", f"Phân tích ảnh: {file.filename}", summary, result)

        return ImageAnalysisResponse(
            success=True,
            metadata=result["metadata"],
            face_detection=result["face_detection"],
            ai_vision=result.get("ai_vision"),
            ai_engine_status=result["ai_engine_status"],
            processing_time_ms=result["processing_time_ms"]
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống khi phân tích ảnh: {str(e)}")
