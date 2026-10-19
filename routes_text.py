from fastapi import APIRouter, HTTPException
from ..models.schemas import TextAnalysisRequest, TextAnalysisResponse
from ..services.nlp_service import analyze_text_full
from ..services.history_service import save_history

router = APIRouter(prefix="/text", tags=["Text NLP Analysis"])

@router.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    if len(request.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Văn bản quá ngắn để thực hiện phân tích (tối thiểu 3 ký tự).")

    try:
        result = analyze_text_full(request.text, request.mode, request.language)
        
        # Save to history
        entities_count = len(result["entities"])
        summary = f"Chế độ {request.mode.upper()}: Trích xuất {entities_count} thực thể nhận diện. Token: ~{result['stats']['estimated_tokens']}."
        save_history("text_nlp", f"Phân tích văn bản ({request.mode})", summary, result)

        return TextAnalysisResponse(
            success=True,
            stats=result["stats"],
            mode=result["mode"],
            language=result["language"],
            entities=result.get("entities", []),
            sentiment=result.get("sentiment"),
            credibility=result.get("credibility"),
            summary=result.get("summary"),
            ai_engine_status=result["ai_engine_status"],
            processing_time_ms=result["processing_time_ms"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý văn bản: {str(e)}")
