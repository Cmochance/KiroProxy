#!/usr/bin/env python3
"""
Kiro API 代理服务器 - 增强版
支持多账号轮询、请求日志、配额监控、429自动切换等功能
"""

import json
import uuid
import httpx
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict
from collections import deque
import time
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Kiro API Proxy", docs_url="/docs", redoc_url=None)

# CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 配置 ====================

KIRO_API_URL = "https://q.us-east-1.amazonaws.com/generateAssistantResponse"
MODELS_URL = "https://q.us-east-1.amazonaws.com/ListAvailableModels"
TOKEN_PATH = Path.home() / ".aws/sso/cache/kiro-auth-token.json"
MACHINE_ID = "fa41d5def91e29225c73f6ea8ee0941a87bd812aae5239e3dde72c3ba7603a26"

# ==================== 数据结构 ====================

@dataclass
class Account:
    """账号信息"""
    id: str
    name: str
    token_path: str
    enabled: bool = True
    rate_limited_until: Optional[float] = None
    request_count: int = 0
    error_count: int = 0
    last_used: Optional[float] = None
    
    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self.rate_limited_until and time.time() < self.rate_limited_until:
            return False
        return True
    
    def get_token(self) -> str:
        try:
            with open(self.token_path) as f:
                return json.load(f).get("accessToken", "")
        except:
            return ""

@dataclass
class RequestLog:
    """请求日志"""
    id: str
    timestamp: float
    method: str
    path: str
    model: str
    account_id: Optional[str]
    status: int
    duration_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None

# ==================== 全局状态 ====================

class ProxyState:
    def __init__(self):
        self.accounts: List[Account] = []
        self.current_account_idx: int = 0
        self.request_logs: deque = deque(maxlen=1000)
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.session_locks: Dict[str, str] = {}  # session_id -> account_id
        self.session_timestamps: Dict[str, float] = {}
        self.start_time: float = time.time()
        
        # 初始化默认账号
        self._init_default_account()
    
    def _init_default_account(self):
        if TOKEN_PATH.exists():
            self.accounts.append(Account(
                id="default",
                name="默认账号",
                token_path=str(TOKEN_PATH)
            ))
    
    def get_available_account(self, session_id: Optional[str] = None) -> Optional[Account]:
        """获取可用账号（支持会话粘性）"""
        # 会话粘性：60秒内同一会话使用同一账号
        if session_id and session_id in self.session_locks:
            account_id = self.session_locks[session_id]
            ts = self.session_timestamps.get(session_id, 0)
            if time.time() - ts < 60:
                for acc in self.accounts:
                    if acc.id == account_id and acc.is_available():
                        self.session_timestamps[session_id] = time.time()
                        return acc
        
        # 轮询获取可用账号
        available = [a for a in self.accounts if a.is_available()]
        if not available:
            return None
        
        # 选择请求数最少的账号（负载均衡）
        account = min(available, key=lambda a: a.request_count)
        
        # 更新会话锁定
        if session_id:
            self.session_locks[session_id] = account.id
            self.session_timestamps[session_id] = time.time()
        
        return account
    
    def mark_rate_limited(self, account_id: str, duration_seconds: int = 60):
        """标记账号限流"""
        for acc in self.accounts:
            if acc.id == account_id:
                acc.rate_limited_until = time.time() + duration_seconds
                acc.error_count += 1
                logger.warning(f"账号 {acc.name} 被限流 {duration_seconds}秒")
                break
    
    def add_log(self, log: RequestLog):
        self.request_logs.append(log)
        self.total_requests += 1
        if log.error:
            self.total_errors += 1
    
    def get_stats(self) -> dict:
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": int(uptime),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": f"{(self.total_errors / max(1, self.total_requests) * 100):.1f}%",
            "accounts_total": len(self.accounts),
            "accounts_available": len([a for a in self.accounts if a.is_available()]),
            "recent_logs": len(self.request_logs)
        }

state = ProxyState()

# ==================== 工具函数 ====================

def get_token() -> str:
    account = state.get_available_account()
    if account:
        return account.get_token()
    raise HTTPException(500, "No available account")

def build_headers(token: str) -> dict:
    return {
        "content-type": "application/json",
        "x-amzn-codewhisperer-optout": "true",
        "x-amzn-kiro-agent-mode": "vibe",
        "x-amz-user-agent": f"aws-sdk-js/1.0.27 KiroIDE-0.8.0-{MACHINE_ID}",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
        "amz-sdk-request": "attempt=1; max=3",
        "Authorization": f"Bearer {token}",
    }

def parse_event_stream(raw: bytes) -> str:
    """解析 AWS event-stream 格式"""
    parts = []
    pos = 0
    while pos < len(raw):
        if pos + 12 > len(raw): break
        total_len = int.from_bytes(raw[pos:pos+4], 'big')
        headers_len = int.from_bytes(raw[pos+4:pos+8], 'big')
        if total_len == 0 or total_len > len(raw) - pos: break
        payload_start = pos + 12 + headers_len
        payload_end = pos + total_len - 4
        if payload_start < payload_end:
            try:
                payload = json.loads(raw[payload_start:payload_end].decode('utf-8'))
                if "assistantResponseEvent" in payload:
                    e = payload["assistantResponseEvent"]
                    if "content" in e: parts.append(e["content"])
                elif "content" in payload:
                    parts.append(payload["content"])
            except: pass
        pos += total_len
    return "".join(parts) or "[No response]"

def generate_session_id(messages: list) -> str:
    """基于消息内容生成会话ID"""
    content = json.dumps(messages[:3], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]

# 模型名称映射：外部名称 -> Kiro 内部名称
MODEL_MAPPING = {
    # Claude 3.5 系列 -> Kiro 的 Claude 4 系列
    "claude-3-5-sonnet-20241022": "claude-sonnet-4",
    "claude-3-5-sonnet-latest": "claude-sonnet-4",
    "claude-3-5-sonnet": "claude-sonnet-4",
    "claude-3-5-haiku-20241022": "claude-haiku-4.5",
    "claude-3-5-haiku-latest": "claude-haiku-4.5",
    "claude-3-5-haiku": "claude-haiku-4.5",
    # Claude 3 系列
    "claude-3-opus-20240229": "claude-opus-4.5",
    "claude-3-opus-latest": "claude-opus-4.5",
    "claude-3-opus": "claude-opus-4.5",
    "claude-3-sonnet-20240229": "claude-sonnet-4",
    "claude-3-sonnet": "claude-sonnet-4",
    "claude-3-haiku-20240307": "claude-haiku-4.5",
    "claude-3-haiku": "claude-haiku-4.5",
    # Claude 4 系列（直接映射）
    "claude-4-sonnet": "claude-sonnet-4",
    "claude-4-opus": "claude-opus-4.5",
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    "claude-sonnet-4.5-20250514": "claude-sonnet-4.5",
    # OpenAI GPT 系列 -> Claude (Codex CLI 用)
    "gpt-4o": "claude-sonnet-4",
    "gpt-4o-mini": "claude-haiku-4.5",
    "gpt-4-turbo": "claude-sonnet-4",
    "gpt-4-turbo-preview": "claude-sonnet-4",
    "gpt-4": "claude-sonnet-4",
    "gpt-3.5-turbo": "claude-haiku-4.5",
    "gpt-3.5-turbo-16k": "claude-haiku-4.5",
    # OpenAI o1 系列 -> Claude Opus (推理模型)
    "o1": "claude-opus-4.5",
    "o1-preview": "claude-opus-4.5",
    "o1-mini": "claude-sonnet-4",
    "o3": "claude-opus-4.5",
    "o3-mini": "claude-sonnet-4",
    # Google Gemini 系列 -> Claude (Gemini CLI 用)
    "gemini-2.0-flash": "claude-sonnet-4",
    "gemini-2.0-flash-exp": "claude-sonnet-4",
    "gemini-2.0-flash-thinking": "claude-opus-4.5",
    "gemini-2.0-flash-thinking-exp": "claude-opus-4.5",
    "gemini-1.5-pro": "claude-sonnet-4.5",
    "gemini-1.5-pro-latest": "claude-sonnet-4.5",
    "gemini-1.5-flash": "claude-sonnet-4",
    "gemini-1.5-flash-latest": "claude-sonnet-4",
    "gemini-pro": "claude-sonnet-4",
    "gemini-pro-vision": "claude-sonnet-4",
    # 通用别名
    "sonnet": "claude-sonnet-4",
    "haiku": "claude-haiku-4.5",
    "opus": "claude-opus-4.5",
}

def map_model_name(model: str) -> str:
    """将外部模型名称映射到 Kiro 支持的名称"""
    if not model:
        return "claude-sonnet-4"
    # 直接匹配
    if model in MODEL_MAPPING:
        return MODEL_MAPPING[model]
    # Kiro 原生模型名称直接返回
    kiro_models = {"auto", "claude-sonnet-4.5", "claude-sonnet-4", "claude-haiku-4.5", "claude-opus-4.5"}
    if model in kiro_models:
        return model
    # 模糊匹配
    model_lower = model.lower()
    if "opus" in model_lower:
        return "claude-opus-4.5"
    if "haiku" in model_lower:
        return "claude-haiku-4.5"
    if "sonnet" in model_lower:
        if "4.5" in model_lower or "4-5" in model_lower:
            return "claude-sonnet-4.5"
        return "claude-sonnet-4"
    # 默认
    return "claude-sonnet-4"

# ==================== API 端点 ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.get("/v1/models")
async def models():
    """获取可用模型列表"""
    try:
        account = state.get_available_account()
        if not account:
            raise Exception("No available account")
        
        token = account.get_token()
        headers = {
            "content-type": "application/json",
            "x-amz-user-agent": f"aws-sdk-js/1.0.27 KiroIDE-0.8.0-{MACHINE_ID}",
            "amz-sdk-invocation-id": str(uuid.uuid4()),
            "Authorization": f"Bearer {token}",
        }
        async with httpx.AsyncClient(verify=False, timeout=30) as client:
            resp = await client.get(MODELS_URL, headers=headers, params={"origin": "AI_EDITOR"})
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "object": "list",
                    "data": [
                        {
                            "id": m["modelId"],
                            "object": "model",
                            "owned_by": "kiro",
                            "name": m["modelName"],
                            "description": m.get("description", ""),
                            "rate": m.get("rateMultiplier", 1),
                            "max_tokens": m.get("tokenLimits", {}).get("maxInputTokens", 0)
                        }
                        for m in data.get("models", [])
                    ]
                }
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
    
    # 降级返回静态列表
    return {"object": "list", "data": [
        {"id": "auto", "object": "model", "owned_by": "kiro", "name": "Auto"},
        {"id": "claude-sonnet-4.5", "object": "model", "owned_by": "kiro", "name": "Claude Sonnet 4.5"},
        {"id": "claude-sonnet-4", "object": "model", "owned_by": "kiro", "name": "Claude Sonnet 4"},
        {"id": "claude-haiku-4.5", "object": "model", "owned_by": "kiro", "name": "Claude Haiku 4.5"},
        {"id": "claude-opus-4.5", "object": "model", "owned_by": "kiro", "name": "Claude Opus 4.5"},
    ]}

@app.post("/v1beta/models/{model_name}:generateContent")
@app.post("/v1/models/{model_name}:generateContent")
async def gemini_generate(model_name: str, request: Request):
    """Gemini 协议兼容端点 - 用于 Gemini CLI"""
    start_time = time.time()
    log_id = uuid.uuid4().hex[:8]
    
    body = await request.json()
    contents = body.get("contents", [])
    
    # 提取模型名（去掉 models/ 前缀）
    model_raw = model_name.replace("models/", "")
    model = map_model_name(model_raw)
    
    # 生成会话ID
    session_id = hashlib.sha256(json.dumps(contents[:3], sort_keys=True).encode()).hexdigest()[:16]
    
    # 获取可用账号
    account = state.get_available_account(session_id)
    if not account:
        raise HTTPException(503, "All accounts are rate limited")
    
    # 提取用户消息（Gemini 格式）
    user_msg = ""
    for content in reversed(contents):
        if content.get("role") == "user":
            parts = content.get("parts", [])
            user_msg = " ".join(p.get("text", "") for p in parts if "text" in p)
            break
    
    # 处理 systemInstruction
    system = body.get("systemInstruction", {})
    if system:
        system_parts = system.get("parts", [])
        system_text = " ".join(p.get("text", "") for p in system_parts if "text" in p)
        if system_text:
            user_msg = f"{system_text}\n\n{user_msg}"
    
    token = account.get_token()
    if not token:
        raise HTTPException(500, f"Failed to get token for account {account.name}")
    
    headers = build_headers(token)
    kiro_body = {
        "conversationState": {
            "conversationId": str(uuid.uuid4()),
            "currentMessage": {
                "userInputMessage": {
                    "content": user_msg,
                    "modelId": model,
                    "origin": "AI_EDITOR",
                    "userInputMessageContext": {}
                }
            },
            "chatTriggerType": "MANUAL"
        }
    }
    
    error_msg = None
    status_code = 200
    content = ""
    
    try:
        async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
            resp = await client.post(KIRO_API_URL, headers=headers, json=kiro_body)
            status_code = resp.status_code
            
            if resp.status_code == 429:
                state.mark_rate_limited(account.id, 60)
                new_account = state.get_available_account()
                if new_account and new_account.id != account.id:
                    token = new_account.get_token()
                    headers = build_headers(token)
                    resp = await client.post(KIRO_API_URL, headers=headers, json=kiro_body)
                    status_code = resp.status_code
                    account = new_account
                else:
                    raise HTTPException(429, "Rate limited")
            
            if resp.status_code != 200:
                error_msg = resp.text
                raise HTTPException(resp.status_code, resp.text)
            
            content = parse_event_stream(resp.content)
            account.request_count += 1
            account.last_used = time.time()
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        status_code = 500
        raise HTTPException(500, str(e))
    finally:
        duration = (time.time() - start_time) * 1000
        state.add_log(RequestLog(
            id=log_id,
            timestamp=time.time(),
            method="POST",
            path=f"/v1/models/{model_name}:generateContent",
            model=model,
            account_id=account.id if account else None,
            status=status_code,
            duration_ms=duration,
            error=error_msg
        ))
    
    # Gemini 响应格式
    return {
        "candidates": [{
            "content": {
                "parts": [{"text": content}],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": len(user_msg) // 4,
            "candidatesTokenCount": len(content) // 4,
            "totalTokenCount": (len(user_msg) + len(content)) // 4
        }
    }


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic 协议兼容端点 - 用于 Claude Code CLI"""
    start_time = time.time()
    log_id = uuid.uuid4().hex[:8]
    
    body = await request.json()
    messages = body.get("messages", [])
    model_raw = body.get("model", "claude-sonnet-4")
    model = map_model_name(model_raw)  # 映射模型名称
    max_tokens = body.get("max_tokens", 4096)
    stream = body.get("stream", False)
    system = body.get("system", "")
    
    if not messages:
        raise HTTPException(400, "messages required")
    
    # 生成会话ID
    session_id = generate_session_id(messages)
    
    # 获取可用账号
    account = state.get_available_account(session_id)
    if not account:
        raise HTTPException(503, "All accounts are rate limited")
    
    # 提取用户消息（Anthropic 格式可能有多个 content block）
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                user_msg = content
            elif isinstance(content, list):
                # 处理 content blocks
                user_msg = " ".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
            break
    
    if system:
        user_msg = f"{system}\n\n{user_msg}"
    
    token = account.get_token()
    if not token:
        raise HTTPException(500, f"Failed to get token for account {account.name}")
    
    headers = build_headers(token)
    kiro_body = {
        "conversationState": {
            "conversationId": str(uuid.uuid4()),
            "currentMessage": {
                "userInputMessage": {
                    "content": user_msg,
                    "modelId": model,
                    "origin": "AI_EDITOR",
                    "userInputMessageContext": {}
                }
            },
            "chatTriggerType": "MANUAL"
        }
    }
    
    error_msg = None
    status_code = 200
    content = ""
    
    try:
        async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
            resp = await client.post(KIRO_API_URL, headers=headers, json=kiro_body)
            status_code = resp.status_code
            
            if resp.status_code == 429:
                state.mark_rate_limited(account.id, 60)
                new_account = state.get_available_account()
                if new_account and new_account.id != account.id:
                    token = new_account.get_token()
                    headers = build_headers(token)
                    resp = await client.post(KIRO_API_URL, headers=headers, json=kiro_body)
                    status_code = resp.status_code
                    account = new_account
                else:
                    raise HTTPException(429, "Rate limited")
            
            if resp.status_code != 200:
                error_msg = resp.text
                raise HTTPException(resp.status_code, resp.text)
            
            content = parse_event_stream(resp.content)
            account.request_count += 1
            account.last_used = time.time()
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        status_code = 500
        raise HTTPException(500, str(e))
    finally:
        duration = (time.time() - start_time) * 1000
        state.add_log(RequestLog(
            id=log_id,
            timestamp=time.time(),
            method="POST",
            path="/v1/messages",
            model=model,
            account_id=account.id if account else None,
            status=status_code,
            duration_ms=duration,
            error=error_msg
        ))
    
    # Anthropic 响应格式
    if stream:
        async def generate_anthropic():
            # message_start
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f'msg_{log_id}', 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
            # content_block_start
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            # content_block_delta
            for i in range(0, len(content), 20):
                chunk = content[i:i+20]
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': chunk}})}\n\n"
                await asyncio.sleep(0.02)
            # content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            # message_delta
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': len(content) // 4}})}\n\n"
            # message_stop
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        return StreamingResponse(generate_anthropic(), media_type="text/event-stream")
    
    return {
        "id": f"msg_{log_id}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": len(user_msg) // 4,
            "output_tokens": len(content) // 4
        }
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    start_time = time.time()
    log_id = uuid.uuid4().hex[:8]
    
    body = await request.json()
    messages = body.get("messages", [])
    model_raw = body.get("model", "claude-sonnet-4")
    model = map_model_name(model_raw)  # 映射模型名称
    stream = body.get("stream", False)
    
    if not messages:
        raise HTTPException(400, "messages required")
    
    # 生成会话ID用于会话粘性
    session_id = generate_session_id(messages)
    
    # 获取可用账号
    account = state.get_available_account(session_id)
    if not account:
        # 所有账号都被限流，返回错误
        raise HTTPException(503, "All accounts are rate limited. Please try again later.")
    
    user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    
    token = account.get_token()
    if not token:
        raise HTTPException(500, f"Failed to get token for account {account.name}")
    
    headers = build_headers(token)
    kiro_body = {
        "conversationState": {
            "conversationId": str(uuid.uuid4()),
            "currentMessage": {
                "userInputMessage": {
                    "content": user_msg,
                    "modelId": model,
                    "origin": "AI_EDITOR",
                    "userInputMessageContext": {}
                }
            },
            "chatTriggerType": "MANUAL"
        }
    }
    
    error_msg = None
    status_code = 200
    content = ""
    
    try:
        async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
            resp = await client.post(KIRO_API_URL, headers=headers, json=kiro_body)
            status_code = resp.status_code
            
            if resp.status_code == 429:
                # 限流处理：标记账号并尝试切换
                state.mark_rate_limited(account.id, 60)
                
                # 尝试获取另一个账号
                new_account = state.get_available_account()
                if new_account and new_account.id != account.id:
                    token = new_account.get_token()
                    headers = build_headers(token)
                    resp = await client.post(KIRO_API_URL, headers=headers, json=kiro_body)
                    status_code = resp.status_code
                    account = new_account
                else:
                    raise HTTPException(429, "Rate limited. Please try again later.")
            
            if resp.status_code != 200:
                error_msg = resp.text
                raise HTTPException(resp.status_code, resp.text)
            
            content = parse_event_stream(resp.content)
            account.request_count += 1
            account.last_used = time.time()
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        status_code = 500
        raise HTTPException(500, str(e))
    finally:
        # 记录日志
        duration = (time.time() - start_time) * 1000
        state.add_log(RequestLog(
            id=log_id,
            timestamp=time.time(),
            method="POST",
            path="/v1/chat/completions",
            model=model,
            account_id=account.id if account else None,
            status=status_code,
            duration_ms=duration,
            error=error_msg
        ))
    
    if stream:
        async def generate():
            for chunk in [content[i:i+20] for i in range(0, len(content), 20)]:
                data = {"choices": [{"delta": {"content": chunk}}]}
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    return {
        "id": f"chatcmpl-{log_id}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

# ==================== 管理 API ====================

@app.get("/api/status")
async def status():
    """服务状态"""
    try:
        with open(TOKEN_PATH) as f:
            data = json.load(f)
        return {
            "ok": True, 
            "expires": data.get("expiresAt"),
            "stats": state.get_stats()
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "stats": state.get_stats()}

@app.get("/api/logs")
async def get_logs(limit: int = Query(100, le=1000)):
    """获取请求日志"""
    logs = list(state.request_logs)[-limit:]
    return {
        "logs": [asdict(log) for log in reversed(logs)],
        "total": len(state.request_logs)
    }

@app.get("/api/accounts")
async def get_accounts():
    """获取账号列表"""
    return {
        "accounts": [
            {
                "id": a.id,
                "name": a.name,
                "enabled": a.enabled,
                "available": a.is_available(),
                "request_count": a.request_count,
                "error_count": a.error_count,
                "rate_limited": a.rate_limited_until > time.time() if a.rate_limited_until else False,
                "rate_limited_until": a.rate_limited_until
            }
            for a in state.accounts
        ]
    }

@app.post("/api/accounts")
async def add_account(request: Request):
    """添加账号"""
    body = await request.json()
    name = body.get("name", f"账号{len(state.accounts)+1}")
    token_path = body.get("token_path")
    
    if not token_path or not Path(token_path).exists():
        raise HTTPException(400, "Invalid token path")
    
    account = Account(
        id=uuid.uuid4().hex[:8],
        name=name,
        token_path=token_path
    )
    state.accounts.append(account)
    return {"ok": True, "account": asdict(account)}

@app.delete("/api/accounts/{account_id}")
async def delete_account(account_id: str):
    """删除账号"""
    state.accounts = [a for a in state.accounts if a.id != account_id]
    return {"ok": True}

@app.post("/api/accounts/{account_id}/toggle")
async def toggle_account(account_id: str):
    """启用/禁用账号"""
    for acc in state.accounts:
        if acc.id == account_id:
            acc.enabled = not acc.enabled
            return {"ok": True, "enabled": acc.enabled}
    raise HTTPException(404, "Account not found")

@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    return state.get_stats()

@app.post("/api/speedtest")
async def speedtest():
    """测试API延迟"""
    account = state.get_available_account()
    if not account:
        return {"ok": False, "error": "No available account"}
    
    start = time.time()
    try:
        token = account.get_token()
        headers = {
            "content-type": "application/json",
            "x-amz-user-agent": f"aws-sdk-js/1.0.27 KiroIDE-0.8.0-{MACHINE_ID}",
            "Authorization": f"Bearer {token}",
        }
        async with httpx.AsyncClient(verify=False, timeout=10) as client:
            resp = await client.get(MODELS_URL, headers=headers, params={"origin": "AI_EDITOR"})
            latency = (time.time() - start) * 1000
            return {
                "ok": resp.status_code == 200,
                "latency_ms": round(latency, 2),
                "status": resp.status_code
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "latency_ms": (time.time() - start) * 1000}

@app.get("/api/config/export")
async def export_config():
    """导出配置"""
    return {
        "accounts": [
            {"name": a.name, "token_path": a.token_path, "enabled": a.enabled}
            for a in state.accounts
        ],
        "exported_at": datetime.now().isoformat()
    }

@app.post("/api/config/import")
async def import_config(request: Request):
    """导入配置"""
    body = await request.json()
    accounts = body.get("accounts", [])
    
    for acc_data in accounts:
        if Path(acc_data.get("token_path", "")).exists():
            account = Account(
                id=uuid.uuid4().hex[:8],
                name=acc_data.get("name", "导入账号"),
                token_path=acc_data["token_path"],
                enabled=acc_data.get("enabled", True)
            )
            # 避免重复
            if not any(a.token_path == account.token_path for a in state.accounts):
                state.accounts.append(account)
    
    return {"ok": True, "imported": len(accounts)}

@app.get("/api/token/scan")
async def scan_tokens():
    """扫描系统中的 Kiro token 文件"""
    found = []
    
    # 扫描 AWS SSO cache 目录
    sso_cache = Path.home() / ".aws/sso/cache"
    if sso_cache.exists():
        for f in sso_cache.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    if "accessToken" in data:
                        found.append({
                            "path": str(f),
                            "name": f.stem,
                            "expires": data.get("expiresAt"),
                            "provider": data.get("provider", "unknown"),
                            "region": data.get("region", "unknown")
                        })
            except:
                pass
    
    return {"tokens": found}

@app.post("/api/token/add-from-scan")
async def add_from_scan(request: Request):
    """从扫描结果添加账号"""
    body = await request.json()
    token_path = body.get("path")
    name = body.get("name", "扫描账号")
    
    if not token_path or not Path(token_path).exists():
        raise HTTPException(400, "Token 文件不存在")
    
    # 检查是否已存在
    if any(a.token_path == token_path for a in state.accounts):
        raise HTTPException(400, "该账号已添加")
    
    # 验证 token 格式
    try:
        with open(token_path) as f:
            data = json.load(f)
            if "accessToken" not in data:
                raise HTTPException(400, "无效的 token 文件")
    except json.JSONDecodeError:
        raise HTTPException(400, "无效的 JSON 文件")
    
    account = Account(
        id=uuid.uuid4().hex[:8],
        name=name,
        token_path=token_path
    )
    state.accounts.append(account)
    
    return {"ok": True, "account_id": account.id}

@app.get("/api/kiro/login-url")
async def get_kiro_login_url():
    """获取 Kiro 登录说明"""
    return {
        "message": "Kiro 使用 AWS Identity Center 认证，无法直接 OAuth",
        "instructions": [
            "1. 打开 Kiro IDE",
            "2. 点击登录按钮，使用 Google/GitHub 账号登录",
            "3. 登录成功后，token 会自动保存到 ~/.aws/sso/cache/",
            "4. 本代理会自动读取该 token"
        ],
        "token_path": str(TOKEN_PATH),
        "token_exists": TOKEN_PATH.exists()
    }

@app.post("/api/token/refresh-check")
async def refresh_token_check():
    """检查并刷新所有账号的 token 状态"""
    results = []
    for acc in state.accounts:
        try:
            with open(acc.token_path) as f:
                data = json.load(f)
                expires = data.get("expiresAt", "")
                # 解析过期时间
                from datetime import datetime
                if expires:
                    exp_time = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                    now = datetime.now(exp_time.tzinfo)
                    is_valid = exp_time > now
                    remaining = (exp_time - now).total_seconds() if is_valid else 0
                else:
                    is_valid = False
                    remaining = 0
                
                results.append({
                    "id": acc.id,
                    "name": acc.name,
                    "valid": is_valid,
                    "expires": expires,
                    "remaining_seconds": int(remaining)
                })
        except Exception as e:
            results.append({
                "id": acc.id,
                "name": acc.name,
                "valid": False,
                "error": str(e)
            })
    
    return {"accounts": results}


# ==================== HTML 页面 ====================

HTML_PAGE = '''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kiro API</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='24' viewBox='0 0 20 24' fill='none'%3E%3Cpath d='M3.80081 18.5661C1.32306 24.0572 6.59904 25.434 10.4904 22.2205C11.6339 25.8242 15.926 23.1361 17.4652 20.3445C20.8578 14.1915 19.4877 7.91459 19.1361 6.61988C16.7244 -2.20972 4.67055 -2.21852 2.59581 6.6649C2.11136 8.21946 2.10284 9.98752 1.82846 11.8233C1.69011 12.749 1.59258 13.3398 1.23436 14.3135C1.02841 14.8733 0.745043 15.3704 0.299833 16.2082C-0.391594 17.5095 -0.0998802 20.021 3.46397 18.7186V18.7195L3.80081 18.5661Z' fill='%23ef4444'/%3E%3Cpath d='M10.9614 10.4413C9.97202 10.4413 9.82422 9.25893 9.82422 8.55407C9.82422 7.91791 9.93824 7.4124 10.1542 7.09197C10.3441 6.81003 10.6158 6.66699 10.9614 6.66699C11.3071 6.66699 11.6036 6.81228 11.8128 7.09892C12.0511 7.42554 12.177 7.92861 12.177 8.55407C12.177 9.73591 11.7226 10.4413 10.9616 10.4413H10.9614Z' fill='black'/%3E%3Cpath d='M15.0318 10.4413C14.0423 10.4413 13.8945 9.25893 13.8945 8.55407C13.8945 7.91791 14.0086 7.4124 14.2245 7.09197C14.4144 6.81003 14.6861 6.66699 15.0318 6.66699C15.3774 6.66699 15.6739 6.81228 15.8831 7.09892C16.1214 7.42554 16.2474 7.92861 16.2474 8.55407C16.2474 9.73591 15.793 10.4413 15.0319 10.4413H15.0318Z' fill='black'/%3E%3C/svg%3E">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root { --bg: #fafafa; --card: #fff; --border: #e5e5e5; --text: #1a1a1a; --muted: #666; --accent: #000; --success: #22c55e; --error: #ef4444; --warn: #f59e0b; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0a0a0a; --card: #141414; --border: #262626; --text: #fafafa; --muted: #a3a3a3; --accent: #fff; }
}
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
.container { max-width: 1100px; margin: 0 auto; padding: 2rem 1rem; }
header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }
h1 { font-size: 1.5rem; font-weight: 600; }
.status { font-size: 0.875rem; color: var(--muted); display: flex; align-items: center; gap: 1rem; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.ok { background: var(--success); }
.status-dot.err { background: var(--error); }
.tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.tab { padding: 0.5rem 1rem; border: 1px solid var(--border); background: var(--card); cursor: pointer; font-size: 0.875rem; transition: all 0.2s; border-radius: 6px; }
.tab.active { background: var(--accent); color: var(--bg); border-color: var(--accent); }
.panel { display: none; }
.panel.active { display: block; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
.card h3 { font-size: 1rem; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
.stat-item { text-align: center; padding: 1rem; background: var(--bg); border-radius: 6px; }
.stat-value { font-size: 1.5rem; font-weight: 600; }
.stat-label { font-size: 0.75rem; color: var(--muted); }
.chat-box { height: 350px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; padding: 1rem; margin-bottom: 1rem; background: var(--bg); }
.msg { margin-bottom: 1rem; }
.msg.user { text-align: right; }
.msg span { display: inline-block; max-width: 80%; padding: 0.75rem 1rem; border-radius: 12px; white-space: pre-wrap; word-break: break-word; }
.msg.user span { background: var(--accent); color: var(--bg); }
.msg.ai span { background: var(--card); border: 1px solid var(--border); }
.input-row { display: flex; gap: 0.5rem; }
.input-row input, .input-row select { flex: 1; padding: 0.75rem 1rem; border: 1px solid var(--border); border-radius: 6px; background: var(--card); color: var(--text); font-size: 1rem; }
.input-row input:focus { outline: none; border-color: var(--accent); }
button { padding: 0.75rem 1.5rem; background: var(--accent); color: var(--bg); border: none; border-radius: 6px; cursor: pointer; font-size: 0.875rem; font-weight: 500; transition: opacity 0.2s; }
button:hover { opacity: 0.8; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
button.secondary { background: var(--card); color: var(--text); border: 1px solid var(--border); }
select { padding: 0.5rem; border: 1px solid var(--border); border-radius: 6px; background: var(--card); color: var(--text); }
pre { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; overflow-x: auto; font-size: 0.8rem; }
code { font-family: "SF Mono", Monaco, monospace; }
table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
th { font-weight: 500; color: var(--muted); }
.badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500; }
.badge.success { background: #dcfce7; color: #166534; }
.badge.error { background: #fee2e2; color: #991b1b; }
.badge.warn { background: #fef3c7; color: #92400e; }
@media (prefers-color-scheme: dark) {
  .badge.success { background: #14532d; color: #86efac; }
  .badge.error { background: #7f1d1d; color: #fca5a5; }
  .badge.warn { background: #78350f; color: #fde68a; }
}
.endpoint { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
.method { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.method.get { background: #dcfce7; color: #166534; }
.method.post { background: #fef3c7; color: #92400e; }
@media (prefers-color-scheme: dark) {
  .method.get { background: #14532d; color: #86efac; }
  .method.post { background: #78350f; color: #fde68a; }
}
.copy-btn { padding: 0.25rem 0.5rem; font-size: 0.75rem; background: var(--card); border: 1px solid var(--border); color: var(--text); }
.footer { text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); }
.log-row { font-size: 0.8rem; }
.log-row:hover { background: var(--bg); }
.account-row { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; border: 1px solid var(--border); border-radius: 6px; margin-bottom: 0.5rem; }
.account-info { display: flex; align-items: center; gap: 1rem; }
.account-actions { display: flex; gap: 0.5rem; }
.account-actions button { padding: 0.25rem 0.5rem; font-size: 0.75rem; }
.refresh-btn { padding: 0.25rem 0.5rem; font-size: 0.75rem; cursor: pointer; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1><svg xmlns="http://www.w3.org/2000/svg" width="20" height="24" viewBox="0 0 20 24" fill="none" style="vertical-align:middle;margin-right:8px"><path d="M3.80081 18.5661C1.32306 24.0572 6.59904 25.434 10.4904 22.2205C11.6339 25.8242 15.926 23.1361 17.4652 20.3445C20.8578 14.1915 19.4877 7.91459 19.1361 6.61988C16.7244 -2.20972 4.67055 -2.21852 2.59581 6.6649C2.11136 8.21946 2.10284 9.98752 1.82846 11.8233C1.69011 12.749 1.59258 13.3398 1.23436 14.3135C1.02841 14.8733 0.745043 15.3704 0.299833 16.2082C-0.391594 17.5095 -0.0998802 20.021 3.46397 18.7186V18.7195L3.80081 18.5661Z" fill="#ef4444"/><path d="M10.9614 10.4413C9.97202 10.4413 9.82422 9.25893 9.82422 8.55407C9.82422 7.91791 9.93824 7.4124 10.1542 7.09197C10.3441 6.81003 10.6158 6.66699 10.9614 6.66699C11.3071 6.66699 11.6036 6.81228 11.8128 7.09892C12.0511 7.42554 12.177 7.92861 12.177 8.55407C12.177 9.73591 11.7226 10.4413 10.9616 10.4413H10.9614Z" fill="black"/><path d="M15.0318 10.4413C14.0423 10.4413 13.8945 9.25893 13.8945 8.55407C13.8945 7.91791 14.0086 7.4124 14.2245 7.09197C14.4144 6.81003 14.6861 6.66699 15.0318 6.66699C15.3774 6.66699 15.6739 6.81228 15.8831 7.09892C16.1214 7.42554 16.2474 7.92861 16.2474 8.55407C16.2474 9.73591 15.793 10.4413 15.0319 10.4413H15.0318Z" fill="black"/></svg>Kiro API</h1>
    <div class="status">
      <span class="status-dot" id="statusDot"></span>
      <span id="statusText">检查中...</span>
      <span id="uptime"></span>
    </div>
  </header>
  
  <div class="tabs">
    <div class="tab active" data-tab="chat">对话</div>
    <div class="tab" data-tab="monitor">监控</div>
    <div class="tab" data-tab="accounts">账号</div>
    <div class="tab" data-tab="logs">日志</div>
    <div class="tab" data-tab="api">API</div>
    <div class="tab" data-tab="docs">文档</div>
  </div>
  
  <!-- 对话面板 -->
  <div class="panel active" id="chat">
    <div class="card">
      <div style="display:flex;gap:0.5rem;margin-bottom:1rem">
        <select id="model" style="flex:1"></select>
        <button class="secondary" onclick="clearChat()">清空</button>
      </div>
      <div class="chat-box" id="chatBox"></div>
      <div class="input-row">
        <input type="text" id="input" placeholder="输入消息..." onkeydown="if(event.key==='Enter')send()">
        <button onclick="send()" id="sendBtn">发送</button>
      </div>
    </div>
  </div>
  
  <!-- 监控面板 -->
  <div class="panel" id="monitor">
    <div class="card">
      <h3>服务状态 <button class="refresh-btn secondary" onclick="loadStats()">刷新</button></h3>
      <div class="stats-grid" id="statsGrid"></div>
    </div>
    <div class="card">
      <h3>速度测试</h3>
      <button onclick="runSpeedtest()" id="speedtestBtn">开始测试</button>
      <span id="speedtestResult" style="margin-left:1rem"></span>
    </div>
  </div>
  
  <!-- 账号面板 -->
  <div class="panel" id="accounts">
    <div class="card">
      <h3>账号管理</h3>
      <div style="display:flex;gap:0.5rem;margin-bottom:1rem;flex-wrap:wrap">
        <button class="secondary" onclick="scanTokens()">扫描 Token</button>
        <button class="secondary" onclick="showAddAccount()">手动添加</button>
        <button class="secondary" onclick="checkTokens()">检查有效期</button>
      </div>
      <div id="accountList"></div>
    </div>
    <div class="card" id="scanResults" style="display:none">
      <h3>扫描结果</h3>
      <div id="scanList"></div>
    </div>
    <div class="card">
      <h3>Kiro 登录说明</h3>
      <p style="color:var(--muted);font-size:0.875rem;margin-bottom:0.5rem">
        Kiro 使用 AWS Identity Center 认证，需要通过 Kiro IDE 登录：
      </p>
      <ol style="color:var(--muted);font-size:0.875rem;padding-left:1.5rem;margin-bottom:1rem">
        <li>打开 Kiro IDE</li>
        <li>点击登录，使用 Google/GitHub 账号</li>
        <li>登录成功后 token 自动保存</li>
        <li>点击上方"扫描 Token"添加账号</li>
      </ol>
      <p style="font-size:0.75rem;color:var(--muted)">Token 路径: ~/.aws/sso/cache/kiro-auth-token.json</p>
    </div>
    <div class="card">
      <h3>配置导入/导出</h3>
      <div style="display:flex;gap:0.5rem">
        <button class="secondary" onclick="exportConfig()">导出配置</button>
        <button class="secondary" onclick="document.getElementById('importFile').click()">导入配置</button>
        <input type="file" id="importFile" accept=".json" style="display:none" onchange="importConfig(event)">
      </div>
    </div>
  </div>
  
  <!-- 日志面板 -->
  <div class="panel" id="logs">
    <div class="card">
      <h3>请求日志 <button class="refresh-btn secondary" onclick="loadLogs()">刷新</button></h3>
      <table>
        <thead><tr><th>时间</th><th>路径</th><th>模型</th><th>状态</th><th>耗时</th></tr></thead>
        <tbody id="logTable"></tbody>
      </table>
    </div>
  </div>
  
  <!-- API 面板 -->
  <div class="panel" id="api">
    <div class="card">
      <h3>API 端点</h3>
      <p style="color:var(--muted);font-size:0.875rem;margin-bottom:1rem">支持 OpenAI 和 Anthropic 两种协议</p>
      <h4 style="color:var(--muted);margin-bottom:0.5rem">OpenAI 协议 (Codex CLI)</h4>
      <div class="endpoint"><span class="method post">POST</span><code>/v1/chat/completions</code><button class="copy-btn" onclick="copy('/v1/chat/completions')">复制</button></div>
      <div class="endpoint"><span class="method get">GET</span><code>/v1/models</code><button class="copy-btn" onclick="copy('/v1/models')">复制</button></div>
      <h4 style="color:var(--muted);margin-top:1rem;margin-bottom:0.5rem">Anthropic 协议 (Claude Code CLI)</h4>
      <div class="endpoint"><span class="method post">POST</span><code>/v1/messages</code><button class="copy-btn" onclick="copy('/v1/messages')">复制</button></div>
      <h4 style="margin-top:1rem;color:var(--muted)">Base URL</h4>
      <pre><code id="baseUrl"></code></pre>
      <button class="copy-btn" onclick="copy(location.origin)" style="margin-top:0.5rem">复制</button>
    </div>
    <div class="card">
      <h3>cc-switch 配置</h3>
      <p style="color:var(--muted);font-size:0.875rem;margin-bottom:1rem">在 cc-switch 中添加自定义供应商：</p>
      <h4 style="color:var(--muted);margin-bottom:0.5rem">Claude Code 配置</h4>
      <pre><code>名称: Kiro Proxy
API Key: any-key-works
Base URL: <span class="pyUrl"></span>
模型: claude-sonnet-4</code></pre>
      <h4 style="color:var(--muted);margin-top:1rem;margin-bottom:0.5rem">Codex 配置</h4>
      <pre><code>名称: Kiro Proxy
API Key: any-key-works
Endpoint: <span class="pyUrl"></span>/v1
模型: claude-sonnet-4</code></pre>
    </div>
    <div class="card">
      <h3>cURL 示例</h3>
      <h4 style="color:var(--muted);margin-bottom:0.5rem">OpenAI 格式</h4>
      <pre><code>curl <span id="curlUrl"></span>/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "Hello"}]}'</code></pre>
      <h4 style="color:var(--muted);margin-top:1rem;margin-bottom:0.5rem">Anthropic 格式</h4>
      <pre><code>curl <span class="pyUrl"></span>/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: any-key" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{"model": "claude-sonnet-4", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}'</code></pre>
    </div>
    <div class="card">
      <h3>Python 示例</h3>
      <h4 style="color:var(--muted);margin-bottom:0.5rem">OpenAI SDK</h4>
      <pre><code>from openai import OpenAI

client = OpenAI(base_url="<span class="pyUrl"></span>/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)</code></pre>
      <h4 style="color:var(--muted);margin-top:1rem;margin-bottom:0.5rem">Anthropic SDK</h4>
      <pre><code>from anthropic import Anthropic

client = Anthropic(base_url="<span class="pyUrl"></span>", api_key="not-needed")
response = client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.content[0].text)</code></pre>
    </div>
  </div>
  
  <!-- 文档面板 -->
  <div class="panel" id="docs">
    <div class="card">
      <h3>模型对照表</h3>
      <p style="color:var(--muted);margin-bottom:1rem;font-size:0.875rem">在 cc-switch 中配置时，根据想使用的 Kiro 模型填写对应的模型名称</p>
      <table>
        <thead><tr><th>Kiro 模型</th><th>能力</th><th>Claude Code</th><th>Codex</th><th>Gemini CLI</th></tr></thead>
        <tbody>
          <tr><td><code>claude-sonnet-4</code></td><td>⭐⭐⭐ 推荐</td><td><code>claude-sonnet-4</code></td><td><code>gpt-4o</code></td><td><code>gemini-2.0-flash</code></td></tr>
          <tr><td><code>claude-sonnet-4.5</code></td><td>⭐⭐⭐⭐ 更强</td><td><code>claude-sonnet-4.5</code></td><td><code>gpt-4o</code></td><td><code>gemini-1.5-pro</code></td></tr>
          <tr><td><code>claude-haiku-4.5</code></td><td>⚡ 快速</td><td><code>claude-haiku-4.5</code></td><td><code>gpt-4o-mini</code></td><td><code>gemini-1.5-flash</code></td></tr>
          <tr><td><code>claude-opus-4.5</code></td><td>⭐⭐⭐⭐⭐ 最强</td><td><code>claude-opus-4.5</code></td><td><code>o1</code></td><td><code>gemini-2.0-flash-thinking</code></td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <h3>cc-switch 配置示例</h3>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem">
        <div style="padding:1rem;background:var(--bg);border-radius:6px">
          <h4 style="margin-bottom:0.5rem">Claude Code</h4>
          <pre style="font-size:0.75rem;margin:0"><code>名称: Kiro Proxy
API Key: any
Base URL: <span class="pyUrl"></span>
模型: claude-sonnet-4</code></pre>
        </div>
        <div style="padding:1rem;background:var(--bg);border-radius:6px">
          <h4 style="margin-bottom:0.5rem">Codex</h4>
          <pre style="font-size:0.75rem;margin:0"><code>名称: Kiro Proxy
API Key: any
Endpoint: <span class="pyUrl"></span>/v1
模型: gpt-4o</code></pre>
        </div>
        <div style="padding:1rem;background:var(--bg);border-radius:6px">
          <h4 style="margin-bottom:0.5rem">Gemini CLI</h4>
          <pre style="font-size:0.75rem;margin:0"><code>名称: Kiro Proxy
API Key: any
Base URL: <span class="pyUrl"></span>
模型: gemini-2.0-flash</code></pre>
        </div>
      </div>
    </div>
    <div class="card">
      <h3>自动模型映射</h3>
      <p style="color:var(--muted);margin-bottom:1rem;font-size:0.875rem">CLI 发送的模型名会自动映射到 Kiro 支持的模型</p>
      <table style="font-size:0.8rem">
        <thead><tr><th>CLI 发送</th><th>映射到 Kiro</th></tr></thead>
        <tbody>
          <tr><td><code>claude-3-5-sonnet-*</code></td><td><code>claude-sonnet-4</code></td></tr>
          <tr><td><code>claude-3-opus-*</code></td><td><code>claude-opus-4.5</code></td></tr>
          <tr><td><code>gpt-4o</code> / <code>gpt-4-turbo</code></td><td><code>claude-sonnet-4</code></td></tr>
          <tr><td><code>gpt-4o-mini</code> / <code>gpt-3.5-turbo</code></td><td><code>claude-haiku-4.5</code></td></tr>
          <tr><td><code>o1</code> / <code>o1-preview</code></td><td><code>claude-opus-4.5</code></td></tr>
          <tr><td><code>gemini-2.0-flash</code> / <code>gemini-1.5-flash</code></td><td><code>claude-sonnet-4</code></td></tr>
          <tr><td><code>gemini-1.5-pro</code></td><td><code>claude-sonnet-4.5</code></td></tr>
          <tr><td><code>gemini-2.0-flash-thinking</code></td><td><code>claude-opus-4.5</code></td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <h3>API 端点</h3>
      <p style="color:var(--muted);font-size:0.875rem;margin-bottom:1rem">支持三种协议：OpenAI / Anthropic / Gemini</p>
      <table style="font-size:0.8rem">
        <thead><tr><th>协议</th><th>端点</th><th>用于</th></tr></thead>
        <tbody>
          <tr><td>OpenAI</td><td><code>/v1/chat/completions</code></td><td>Codex CLI</td></tr>
          <tr><td>Anthropic</td><td><code>/v1/messages</code></td><td>Claude Code CLI</td></tr>
          <tr><td>Gemini</td><td><code>/v1/models/{model}:generateContent</code></td><td>Gemini CLI</td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <h3>OpenAI 兼容接口</h3>
      <p style="color:var(--muted);margin-bottom:1rem">本服务提供与 OpenAI API 兼容的接口，可直接用于支持自定义 API 端点的应用。</p>
      <h4 style="margin-top:1rem">请求格式</h4>
      <pre><code>{"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "Hello"}], "stream": false}</code></pre>
      <h4 style="margin-top:1rem">可用模型</h4>
      <ul id="modelList" style="color:var(--muted);padding-left:1.5rem"></ul>
    </div>
  </div>
  
  <div class="footer">Kiro API Proxy - 多账号轮询 | 会话粘性 | 429自动切换</div>
</div>

<script>
const $=s=>document.querySelector(s);
const $$=s=>document.querySelectorAll(s);

// Tabs
$$('.tab').forEach(t=>t.onclick=()=>{
  $$('.tab').forEach(x=>x.classList.remove('active'));
  $$('.panel').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  $('#'+t.dataset.tab).classList.add('active');
  if(t.dataset.tab==='monitor')loadStats();
  if(t.dataset.tab==='logs')loadLogs();
  if(t.dataset.tab==='accounts')loadAccounts();
});

// Status
async function checkStatus(){
  try{
    const r=await fetch('/api/status');
    const d=await r.json();
    $('#statusDot').className='status-dot '+(d.ok?'ok':'err');
    $('#statusText').textContent=d.ok?'已连接':'未连接';
    if(d.stats)$('#uptime').textContent='运行 '+formatUptime(d.stats.uptime_seconds);
  }catch(e){
    $('#statusDot').className='status-dot err';
    $('#statusText').textContent='连接失败';
  }
}
function formatUptime(s){
  if(s<60)return s+'秒';
  if(s<3600)return Math.floor(s/60)+'分钟';
  return Math.floor(s/3600)+'小时'+Math.floor((s%3600)/60)+'分钟';
}
checkStatus();
setInterval(checkStatus,30000);

// URLs
$('#baseUrl').textContent=location.origin;
$('#curlUrl').textContent=location.origin;
$$('.pyUrl').forEach(e=>e.textContent=location.origin);

// Models
async function loadModels(){
  try{
    const r=await fetch('/v1/models');
    const d=await r.json();
    const select=$('#model');
    const list=$('#modelList');
    select.innerHTML='';
    list.innerHTML='';
    (d.data||[]).forEach(m=>{
      const opt=document.createElement('option');
      opt.value=m.id;
      opt.textContent=m.name||m.id;
      if(m.id==='claude-sonnet-4')opt.selected=true;
      select.appendChild(opt);
      const li=document.createElement('li');
      li.innerHTML='<code>'+m.id+'</code> - '+(m.name||m.id);
      list.appendChild(li);
    });
  }catch(e){console.error('加载模型失败:',e)}
}
loadModels();

// Chat
let messages=[];
function addMsg(role,text){
  const box=$('#chatBox');
  const div=document.createElement('div');
  div.className='msg '+(role==='user'?'user':'ai');
  div.innerHTML='<span>'+text.replace(/\\x3c/g,'&lt;').replace(/\\n/g,'<br>')+'</span>';
  box.appendChild(div);
  box.scrollTop=box.scrollHeight;
}
function clearChat(){messages=[];$('#chatBox').innerHTML='';}
async function send(){
  const input=$('#input');
  const text=input.value.trim();
  if(!text)return;
  input.value='';
  addMsg('user',text);
  messages.push({role:'user',content:text});
  $('#sendBtn').disabled=true;
  $('#sendBtn').textContent='...';
  try{
    const res=await fetch('/v1/chat/completions',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:$('#model').value,messages})
    });
    const data=await res.json();
    if(data.choices&&data.choices[0]){
      const reply=data.choices[0].message.content;
      addMsg('ai',reply);
      messages.push({role:'assistant',content:reply});
    }else if(data.detail){
      addMsg('ai','错误: '+data.detail);
    }
  }catch(e){addMsg('ai','请求失败: '+e.message)}
  $('#sendBtn').disabled=false;
  $('#sendBtn').textContent='发送';
}

// Stats
async function loadStats(){
  try{
    const r=await fetch('/api/stats');
    const d=await r.json();
    $('#statsGrid').innerHTML=`
      <div class="stat-item"><div class="stat-value">${d.total_requests}</div><div class="stat-label">总请求</div></div>
      <div class="stat-item"><div class="stat-value">${d.total_errors}</div><div class="stat-label">错误数</div></div>
      <div class="stat-item"><div class="stat-value">${d.error_rate}</div><div class="stat-label">错误率</div></div>
      <div class="stat-item"><div class="stat-value">${d.accounts_available}/${d.accounts_total}</div><div class="stat-label">可用账号</div></div>
    `;
  }catch(e){console.error(e)}
}

// Speedtest
async function runSpeedtest(){
  $('#speedtestBtn').disabled=true;
  $('#speedtestResult').textContent='测试中...';
  try{
    const r=await fetch('/api/speedtest',{method:'POST'});
    const d=await r.json();
    $('#speedtestResult').textContent=d.ok?`延迟: ${d.latency_ms.toFixed(0)}ms`:'测试失败: '+d.error;
  }catch(e){$('#speedtestResult').textContent='测试失败'}
  $('#speedtestBtn').disabled=false;
}

// Logs
async function loadLogs(){
  try{
    const r=await fetch('/api/logs?limit=50');
    const d=await r.json();
    $('#logTable').innerHTML=(d.logs||[]).map(l=>`
      <tr class="log-row">
        <td>${new Date(l.timestamp*1000).toLocaleTimeString()}</td>
        <td>${l.path}</td>
        <td>${l.model||'-'}</td>
        <td><span class="badge ${l.status<400?'success':l.status<500?'warn':'error'}">${l.status}</span></td>
        <td>${l.duration_ms.toFixed(0)}ms</td>
      </tr>
    `).join('');
  }catch(e){console.error(e)}
}

// Accounts
async function loadAccounts(){
  try{
    const r=await fetch('/api/accounts');
    const d=await r.json();
    $('#accountList').innerHTML=(d.accounts||[]).map(a=>`
      <div class="account-row">
        <div class="account-info">
          <span class="badge ${a.available?'success':a.rate_limited?'warn':'error'}">${a.available?'可用':a.rate_limited?'限流':'禁用'}</span>
          <span>${a.name}</span>
          <span style="color:var(--muted);font-size:0.8rem">请求: ${a.request_count}</span>
        </div>
        <div class="account-actions">
          <button class="secondary" onclick="toggleAccount('${a.id}')">${a.enabled?'禁用':'启用'}</button>
          <button class="secondary" onclick="deleteAccount('${a.id}')" style="color:var(--error)">删除</button>
        </div>
      </div>
    `).join('')||'<p style="color:var(--muted)">暂无账号，请点击"扫描 Token"</p>';
  }catch(e){console.error(e)}
}
async function toggleAccount(id){
  await fetch('/api/accounts/'+id+'/toggle',{method:'POST'});
  loadAccounts();
}
async function deleteAccount(id){
  if(confirm('确定删除此账号?')){
    await fetch('/api/accounts/'+id,{method:'DELETE'});
    loadAccounts();
  }
}
function showAddAccount(){
  const path=prompt('输入 Token 文件路径:','/home/'+location.hostname.split('.')[0]+'/.aws/sso/cache/kiro-auth-token.json');
  if(path){
    const name=prompt('账号名称:','账号'+(Date.now()%1000));
    fetch('/api/accounts',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,token_path:path})
    }).then(r=>r.json()).then(d=>{
      if(d.ok)loadAccounts();
      else alert(d.detail||'添加失败');
    });
  }
}

// Token 扫描
async function scanTokens(){
  try{
    const r=await fetch('/api/token/scan');
    const d=await r.json();
    const panel=$('#scanResults');
    const list=$('#scanList');
    if(d.tokens&&d.tokens.length>0){
      panel.style.display='block';
      list.innerHTML=d.tokens.map(t=>`
        <div class="account-row">
          <div class="account-info">
            <span>${t.name}</span>
            <span style="color:var(--muted);font-size:0.75rem">${t.provider} | ${t.region}</span>
            <span style="color:var(--muted);font-size:0.75rem">过期: ${t.expires||'未知'}</span>
          </div>
          <button class="secondary" onclick="addFromScan('${t.path}','${t.name}')">添加</button>
        </div>
      `).join('');
    }else{
      alert('未找到 Kiro token 文件。请先在 Kiro IDE 中登录。');
    }
  }catch(e){alert('扫描失败: '+e.message)}
}

async function addFromScan(path,name){
  try{
    const r=await fetch('/api/token/add-from-scan',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path,name})
    });
    const d=await r.json();
    if(d.ok){
      loadAccounts();
      $('#scanResults').style.display='none';
    }else{
      alert(d.detail||'添加失败');
    }
  }catch(e){alert('添加失败: '+e.message)}
}

async function checkTokens(){
  try{
    const r=await fetch('/api/token/refresh-check',{method:'POST'});
    const d=await r.json();
    let msg='Token 状态:\\n';
    (d.accounts||[]).forEach(a=>{
      const status=a.valid?'✅ 有效':'❌ 已过期';
      const time=a.remaining_seconds>0?` (剩余 ${Math.floor(a.remaining_seconds/60)} 分钟)`:'';
      msg+=`${a.name}: ${status}${time}\\n`;
    });
    alert(msg);
  }catch(e){alert('检查失败: '+e.message)}
}

// Config
async function exportConfig(){
  const r=await fetch('/api/config/export');
  const d=await r.json();
  const blob=new Blob([JSON.stringify(d,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='kiro-config.json';
  a.click();
}
async function importConfig(e){
  const file=e.target.files[0];
  if(!file)return;
  const text=await file.text();
  const data=JSON.parse(text);
  await fetch('/api/config/import',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(data)
  });
  loadAccounts();
  alert('导入成功');
}

function copy(text){navigator.clipboard.writeText(text)}
</script>
</body>
</html>'''

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════╗
║              Kiro API Proxy Server v2.0                   ║
╠═══════════════════════════════════════════════════════════╣
║   功能: 多账号轮询 | 会话粘性 | 429自动切换 | 请求日志          ║
╠═══════════════════════════════════════════════════════════╣
║  Web UI:  http://0.0.0.0:8000                             ║
║  API:     http://0.0.0.0:8000/v1                          ║
║  Docs:    http://0.0.0.0:8000/docs                        ║
╚═══════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
