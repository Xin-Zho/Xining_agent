from pydantic import BaseModel


class AuthRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    conversation_id: int
    message: str


class CreateConversationRequest(BaseModel):
    title: str = "新对话"


class CreateAgentTaskRequest(BaseModel):
    description: str
    conversation_id: int | None = None
    agent_mode: str = "react"  # "react" | "plan_solve" | "reflection"
