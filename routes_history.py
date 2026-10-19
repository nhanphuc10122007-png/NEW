from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from ..services.history_service import get_history, delete_history_item, clear_all_history

router = APIRouter(prefix="/history", tags=["Analysis History"])

@router.get("", response_model=List[Dict[str, Any]])
async def list_history(limit: int = 50):
    return get_history(limit=limit)

@router.delete("/{item_id}")
async def delete_item(item_id: str):
    success = delete_history_item(item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi lịch sử này.")
    return {"success": True, "message": f"Đã xóa bản ghi {item_id}"}

@router.delete("")
async def clear_history():
    deleted_count = clear_all_history()
    return {"success": True, "deleted_count": deleted_count, "message": "Đã xóa toàn bộ lịch sử phân tích"}
