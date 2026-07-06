# -*- coding: utf-8 -*-
"""Conversation API routes - no auth, single user."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.database.conversations import (
    list_conversations, create_conversation, get_conversation,
    append_message, delete_conversation,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

class CreateConvRequest(BaseModel):
    title: str = ""

class MessageRequest(BaseModel):
    role: str
    content: str

@router.get("")
async def list_conv():
    convs = list_conversations()
    return [{k: v for k, v in c.items() if k != "messages"} for c in convs]

@router.post("")
async def create_conv(req: CreateConvRequest):
    return create_conversation(req.title or "New Chat")

@router.get("/{conv_id}")
async def get_conv(conv_id: str):
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@router.post("/{conv_id}/messages")
async def add_message(conv_id: str, req: MessageRequest):
    result = append_message(conv_id, req.role, req.content)
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return result

@router.delete("/{conv_id}")
async def delete_conv(conv_id: str):
    if not delete_conversation(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}
