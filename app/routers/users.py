from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.dependencies import get_db
from app.models.db_models import User, UserRole
from app.models.schemes import UserResponse
from app.services.security import get_current_admin

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/", response_model=List[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_current_admin),
):
    users = db.query(User).filter(User.is_active == True).all()
    return users

@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id:int,
    db:Session = Depends(get_db),
    admin_user: User = Depends(get_current_admin)
):
    user = db.query(User).filter(User.user_id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@router.delete("/{user_id}")
def soft_delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_current_admin)
):
    user = db.query(User).filter(User.user_id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.role == UserRole.admin:
        raise HTTPException(status_code=404, detail="Cannot delete admin")
    
    user.is_active = False
    db.commit()

    return {"message": "User deactivated"}