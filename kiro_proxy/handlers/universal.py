"""通用聊天处理器 - 支持多种输入格式

支持的输入格式:
1. 简单文本: {"prompt": "你好"} 或 {"text": "你好"} 或 {"content": "你好"}
2. OpenAI 格式: {"messages": [{"role": "user", "content": "你好"}]}
3. 带历史: {"prompt": "你好", "history": [...]}
4. 完整配置: {"prompt": "...", "model": "...", "stream": true, "system": "..."}

输出格式: OpenAI Chat Completion 格式
"""
import json
import uuid
import time
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse

from ..config import KIRO_API_URL, map_model_name
from ..core import state, is_retryable_error, stats_manager, flow_monitor, TokenUsage
from ..core.state import RequestLog
from ..core.history_manager import HistoryManager, get_history_config
from ..core.error_handler import classify_error, ErrorType, format_error_log
from ..core.rate_limiter import get_rate_limiter
from ..kiro_api import build_headers, build_kiro_request, parse_event_stream_full, parse_event_stream, is_quota_exceeded_error
from ..converters import (
    generate_session_id,
    convert_openai_tools_to_kiro,
    fix_history_alternation,
    extract_images_from_content,
    MAX_TOOLS,
    truncate_description
)


def normalize_request(body: dict) -> dict:
    """将各种格式的请求标准化为统一格式
    
    Returns:
        {
            "messages": [...],
            "model": str,
            "stream": bool,
            "tools": [...] or None,
            "tool_choice": ... or None
        }
    """
    # 提取模型
    model = body.get("model", "claude-sonnet-4")
    
    # 提取流式标志
    stream = body.get("stream", False)
    
    # 提取工具
    tools = body.get("tools", None)
    tool_choice = body.get("tool_choice", None)
    
    # 提取 system prompt
    system = body.get("system", body.get("system_prompt", ""))
    
    # 已经是 OpenAI 格式
    if "messages" in body and isinstance(body["messages"], list) and len(body["messages"]) > 0:
        messages = body["messages"]
        # 确保有 system 消息
        if system and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system}] + messages
        return {
            "messages": messages,
            "model": model,
            "stream": stream,
            "tools": tools,
            "tool_choice": tool_choice
        }
    
    # 提取用户输入
    user_input = (
        body.get("prompt") or 
        body.get("text") or 
        body.get("content") or 
        body.get("query") or 
        body.get("input") or
        body.get("message") or
        ""
    )
    
    if not user_input:
        raise HTTPException(400, "No input provided. Use 'prompt', 'text', 'content', 'query', 'input', 'message', or 'messages'")
    
    # 构建 messages
    messages = []
    
    # 添加 system 消息
    if system:
        messages.append({"role": "system", "content": system})
    
    # 处理历史消息
    history = body.get("history", body.get("conversation", []))
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", msg.get("text", ""))
            
            # 处理 tool_calls (assistant)
            if role == "assistant" and "tool_calls" in msg:
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": msg["tool_calls"]
                })
            # 处理 tool 结果
            elif role == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content
                })
            else:
                messages.append({"role": role, "content": content})
    
    # 添加当前用户消息
    messages.append({"role": "user", "content": user_input})
    
    return {
        "messages": messages,
        "model": model,
        "stream": stream,
        "tools": tools,
        "tool_choice": tool_choice
    }


def convert_universal_messages_to_kiro(
    messages: List[dict],
    model: str,
    tools: List[dict] = None,
    tool_choice = None
) -> Tuple[str, List[dict], List[dict], List[dict]]:
    """将标准化的消息转换为 Kiro 格式
    
    Returns:
        (user_content, history, tool_results, kiro_tools)
    """
    from ..converters import convert_openai_messages_to_kiro
    return convert_openai_messages_to_kiro(messages, model, tools, tool_choice)


async def handle_universal_chat(request: Request):
    """处理通用聊天请求 - 支持流式和非流式"""
    start_time = time.time()
    log_id = uuid.uuid4().hex[:8]
    
    # 解析请求体
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON: {e}")
    
    # 标准化请求
    normalized = normalize_request(body)
    
    model = map_model_name(normalized["model"])
    messages = normalized["messages"]
    stream = normalized["stream"]
    tools = normalized["tools"]
    tool_choice = normalized["tool_choice"]
    
    print(f"[Universal] Request: model={normalized['model']} -> {model}, messages={len(messages)}, stream={stream}, tools={len(tools) if tools else 0}")
    
    # 获取可用账号
    session_id = generate_session_id(messages)
    account = state.get_available_account(session_id)
    
    if not account:
        raise HTTPException(503, "All accounts are rate limited or unavailable")
    
    # 创建 Flow 记录
    flow_id = flow_monitor.create_flow(
        protocol="universal",
        method="POST",
        path="/api/chat",
        headers=dict(request.headers),
        body=body,
        account_id=account.id,
        account_name=account.name,
    )
    
    # 检查 token 是否即将过期
    if account.is_token_expiring_soon(5):
        print(f"[Universal] Token 即将过期，尝试刷新: {account.id}")
        success, msg = await account.refresh_token()
        if not success:
            print(f"[Universal] Token 刷新失败: {msg}")
    
    token = account.get_token()
    if not token:
        flow_monitor.fail_flow(flow_id, "authentication_error", f"Failed to get token for account {account.name}")
        raise HTTPException(500, f"Failed to get token for account {account.name}")
    
    # 构建请求头
    creds = account.get_credentials()
    headers = build_headers(
        token,
        machine_id=account.get_machine_id(),
        profile_arn=creds.profile_arn if creds else None,
        client_id=creds.client_id if creds else None
    )
    
    # 限速检查
    rate_limiter = get_rate_limiter()
    can_request, wait_seconds, reason = rate_limiter.can_request(account.id)
    if not can_request:
        print(f"[Universal] 限速: {reason}")
        await asyncio.sleep(wait_seconds)
    
    # 转换消息格式
    user_content, history, tool_results, kiro_tools = convert_universal_messages_to_kiro(
        messages, model, tools, tool_choice
    )
    
    # 历史消息预处理
    history_manager = HistoryManager(get_history_config(), cache_key=session_id)
    
    async def call_summary(prompt: str) -> str:
        req = build_kiro_request(prompt, "claude-haiku-4.5", [])
        try:
            async with httpx.AsyncClient(verify=False, timeout=60) as client:
                resp = await client.post(KIRO_API_URL, json=req, headers=headers)
                if resp.status_code == 200:
                    return parse_event_stream(resp.content)
        except Exception as e:
            print(f"[Universal] Summary API 调用失败: {e}")
        return ""
    
    if history_manager.should_summarize(history) or history_manager.should_pre_summary_for_error_retry(history, user_content):
        history = await history_manager.pre_process_async(history, user_content, call_summary)
    else:
        history = history_manager.pre_process(history, user_content)
    
    history = fix_history_alternation(history)
    
    if history_manager.was_truncated:
        print(f"[Universal] {history_manager.truncate_info}")
    
    # 提取图片
    images = []
    if messages:
        last_msg = messages[-1]
        if last_msg.get("role") == "user":
            _, images = extract_images_from_content(last_msg.get("content", ""))
    
    # 构建 Kiro 请求
    kiro_request = build_kiro_request(
        user_content, model, history,
        images=images if images else None,
        tools=kiro_tools if kiro_tools else None,
        tool_results=tool_results if tool_results else None
    )
    
    if stream:
        return await _handle_stream_response(
            kiro_request, headers, account, model, log_id, start_time,
            session_id, flow_id, history, user_content, kiro_tools, images,
            tool_results, history_manager, call_summary
        )
    else:
        return await _handle_non_stream_response(
            kiro_request, headers, account, model, log_id, start_time,
            session_id, flow_id, history, user_content, kiro_tools, images,
            tool_results, history_manager, call_summary
        )


async def _handle_stream_response(
    kiro_request, headers, account, model, log_id, start_time,
    session_id, flow_id, history, user_content, kiro_tools, images,
    tool_results, history_manager, call_summary
):
    """处理流式响应"""
    
    async def generate():
        nonlocal kiro_request, history
        current_account = account
        retry_count = 0
        max_retries = 2
        full_content = ""
        tool_calls_list = []
        
        while retry_count <= max_retries:
            try:
                async with httpx.AsyncClient(verify=False, timeout=300) as client:
                    async with client.stream("POST", KIRO_API_URL, json=kiro_request, headers=headers) as response:
                        
                        # 处理配额超限
                        if response.status_code == 429 or is_quota_exceeded_error(response.status_code, ""):
                            current_account.mark_quota_exceeded("Rate limited (stream)")
                            next_account = state.get_next_available_account(current_account.id)
                            if next_account and retry_count < max_retries:
                                print(f"[Universal Stream] 配额超限，切换账号: {current_account.id} -> {next_account.id}")
                                current_account = next_account
                                headers["Authorization"] = f"Bearer {current_account.get_token()}"
                                retry_count += 1
                                continue
                            
                            if flow_id:
                                flow_monitor.fail_flow(flow_id, "rate_limit_error", "All accounts rate limited", 429)
                            yield _format_error_chunk("rate_limit_error", "All accounts rate limited")
                            return
                        
                        # 处理可重试的服务端错误
                        if is_retryable_error(response.status_code):
                            if retry_count < max_retries:
                                print(f"[Universal Stream] 服务端错误 {response.status_code}，重试 {retry_count + 1}/{max_retries}")
                                retry_count += 1
                                await asyncio.sleep(0.5 * (2 ** retry_count))
                                continue
                            if flow_id:
                                flow_monitor.fail_flow(flow_id, "api_error", "Server error after retries", response.status_code)
                            yield _format_error_chunk("api_error", "Server error after retries")
                            return
                        
                        if response.status_code != 200:
                            error_text = await response.aread()
                            error_str = error_text.decode()
                            
                            error = classify_error(response.status_code, error_str)
                            print(format_error_log(error, current_account.id))
                            
                            if error.should_disable_account:
                                current_account.enabled = False
                                from ..credential import CredentialStatus
                                current_account.status = CredentialStatus.SUSPENDED
                            
                            if error.should_switch_account:
                                next_account = state.get_next_available_account(current_account.id)
                                if next_account and retry_count < max_retries:
                                    current_account = next_account
                                    headers["Authorization"] = f"Bearer {current_account.get_token()}"
                                    retry_count += 1
                                    continue
                            
                            # 内容长度超限重试
                            if error.type == ErrorType.CONTENT_TOO_LONG:
                                truncated_history, should_retry = await history_manager.handle_length_error_async(
                                    history, retry_count, call_summary
                                )
                                if should_retry:
                                    history = truncated_history
                                    kiro_request = build_kiro_request(
                                        user_content, model, history, kiro_tools, images, tool_results
                                    )
                                    retry_count += 1
                                    continue
                            
                            if flow_id:
                                flow_monitor.fail_flow(flow_id, "api_error", error.user_message, response.status_code)
                            yield _format_error_chunk("api_error", error.user_message)
                            return
                        
                        # 标记开始流式传输
                        if flow_id:
                            flow_monitor.start_streaming(flow_id)
                        
                        # 发送开始事件
                        yield _format_stream_chunk(log_id, model, "", is_first=True)
                        
                        full_response = b""
                        
                        async for chunk in response.aiter_bytes():
                            full_response += chunk
                            
                            try:
                                pos = 0
                                while pos < len(chunk):
                                    if pos + 12 > len(chunk):
                                        break
                                    total_len = int.from_bytes(chunk[pos:pos+4], 'big')
                                    if total_len == 0 or total_len > len(chunk) - pos:
                                        break
                                    headers_len = int.from_bytes(chunk[pos+4:pos+8], 'big')
                                    payload_start = pos + 12 + headers_len
                                    payload_end = pos + total_len - 4
                                    
                                    if payload_start < payload_end:
                                        try:
                                            payload = json.loads(chunk[payload_start:payload_end].decode('utf-8'))
                                            content = None
                                            if 'assistantResponseEvent' in payload:
                                                content = payload['assistantResponseEvent'].get('content')
                                            elif 'content' in payload:
                                                content = payload['content']
                                            if content:
                                                full_content += content
                                                if flow_id:
                                                    flow_monitor.add_chunk(flow_id, content)
                                                yield _format_stream_chunk(log_id, model, content)
                                        except Exception:
                                            pass
                                    pos += total_len
                            except Exception:
                                pass
                        
                        # 解析完整响应获取工具调用
                        result = parse_event_stream_full(full_response)
                        tool_uses = result.get("tool_uses", [])
                        
                        # 发送工具调用
                        if tool_uses:
                            tool_calls_list = _convert_tool_uses_to_openai(tool_uses)
                            yield _format_tool_calls_chunk(log_id, model, tool_calls_list)
                        
                        # 发送结束事件
                        finish_reason = "tool_calls" if tool_uses else "stop"
                        yield _format_stream_chunk(log_id, model, "", finish_reason=finish_reason)
                        yield "data: [DONE]\n\n"
                        
                        # 完成 Flow
                        if flow_id:
                            flow_monitor.complete_flow(
                                flow_id,
                                status_code=200,
                                content=full_content,
                                tool_calls=tool_uses,
                                stop_reason=finish_reason,
                                usage=TokenUsage(
                                    input_tokens=result.get("input_tokens", 0),
                                    output_tokens=result.get("output_tokens", 0),
                                ),
                            )
                        
                        current_account.request_count += 1
                        current_account.last_used = time.time()
                        get_rate_limiter().record_request(current_account.id)
                        return
            
            except httpx.TimeoutException:
                if retry_count < max_retries:
                    print(f"[Universal Stream] 请求超时，重试 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                if flow_id:
                    flow_monitor.fail_flow(flow_id, "timeout_error", "Request timeout", 408)
                yield _format_error_chunk("timeout_error", "Request timeout after retries")
                return
            except httpx.ConnectError:
                if retry_count < max_retries:
                    print(f"[Universal Stream] 连接错误，重试 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                if flow_id:
                    flow_monitor.fail_flow(flow_id, "connection_error", "Connection error", 502)
                yield _format_error_chunk("connection_error", "Connection error after retries")
                return
            except Exception as e:
                if is_retryable_error(None, e) and retry_count < max_retries:
                    retry_count += 1
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                if flow_id:
                    flow_monitor.fail_flow(flow_id, "api_error", str(e), 500)
                yield _format_error_chunk("api_error", str(e))
                return
    
    return StreamingResponse(generate(), media_type="text/event-stream")


async def _handle_non_stream_response(
    kiro_request, headers, account, model, log_id, start_time,
    session_id, flow_id, history, user_content, kiro_tools, images,
    tool_results, history_manager, call_summary
):
    """处理非流式响应"""
    current_account = account
    max_retries = 2
    error_msg = None
    status_code = 200
    
    for retry in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(verify=False, timeout=300) as client:
                response = await client.post(KIRO_API_URL, json=kiro_request, headers=headers)
                status_code = response.status_code
                
                # 处理配额超限
                if response.status_code == 429 or is_quota_exceeded_error(response.status_code, response.text):
                    current_account.mark_quota_exceeded("Rate limited")
                    next_account = state.get_next_available_account(current_account.id)
                    if next_account and retry < max_retries:
                        print(f"[Universal] 配额超限，切换账号: {current_account.id} -> {next_account.id}")
                        current_account = next_account
                        headers["Authorization"] = f"Bearer {current_account.get_token()}"
                        continue
                    
                    if flow_id:
                        flow_monitor.fail_flow(flow_id, "rate_limit_error", "All accounts rate limited", 429)
                    raise HTTPException(429, "All accounts rate limited")
                
                # 处理可重试的服务端错误
                if is_retryable_error(response.status_code):
                    if retry < max_retries:
                        print(f"[Universal] 服务端错误 {response.status_code}，重试 {retry + 1}/{max_retries}")
                        await asyncio.sleep(0.5 * (2 ** retry))
                        continue
                    if flow_id:
                        flow_monitor.fail_flow(flow_id, "api_error", "Server error after retries", response.status_code)
                    raise HTTPException(response.status_code, f"Server error after {max_retries} retries")
                
                if response.status_code != 200:
                    error_msg = response.text
                    error = classify_error(response.status_code, error_msg)
                    print(format_error_log(error, current_account.id))
                    
                    if error.should_disable_account:
                        current_account.enabled = False
                        from ..credential import CredentialStatus
                        current_account.status = CredentialStatus.SUSPENDED
                    
                    if error.should_switch_account:
                        next_account = state.get_next_available_account(current_account.id)
                        if next_account and retry < max_retries:
                            current_account = next_account
                            headers["Authorization"] = f"Bearer {current_account.get_token()}"
                            continue
                    
                    # 内容长度超限重试
                    if error.type == ErrorType.CONTENT_TOO_LONG:
                        truncated_history, should_retry = await history_manager.handle_length_error_async(
                            history, retry, call_summary
                        )
                        if should_retry:
                            history = truncated_history
                            kiro_request = build_kiro_request(
                                user_content, model, history, kiro_tools, images, tool_results
                            )
                            continue
                    
                    if flow_id:
                        flow_monitor.fail_flow(flow_id, "api_error", error.user_message, response.status_code)
                    raise HTTPException(response.status_code, error.user_message)
                
                # 解析响应
                result = parse_event_stream_full(response.content)
                current_account.request_count += 1
                current_account.last_used = time.time()
                get_rate_limiter().record_request(current_account.id)
                
                # 构建 OpenAI 格式响应
                text = "".join(result.get("content", []))
                tool_uses = result.get("tool_uses", [])
                
                message = {
                    "role": "assistant",
                    "content": text if text else None
                }
                
                if tool_uses:
                    message["tool_calls"] = _convert_tool_uses_to_openai(tool_uses)
                
                finish_reason = "tool_calls" if tool_uses else "stop"
                
                # 完成 Flow
                if flow_id:
                    flow_monitor.complete_flow(
                        flow_id,
                        status_code=200,
                        content=text,
                        tool_calls=tool_uses,
                        stop_reason=finish_reason,
                        usage=TokenUsage(
                            input_tokens=result.get("input_tokens", 0),
                            output_tokens=result.get("output_tokens", 0),
                        ),
                    )
                
                # 记录日志和统计
                duration = (time.time() - start_time) * 1000
                state.add_log(RequestLog(
                    id=log_id,
                    timestamp=time.time(),
                    method="POST",
                    path="/api/chat",
                    model=model,
                    account_id=current_account.id,
                    status=200,
                    duration_ms=duration,
                    error=None
                ))
                stats_manager.record_request(
                    account_id=current_account.id,
                    model=model,
                    success=True,
                    latency_ms=duration
                )
                
                return {
                    "id": f"chatcmpl-{log_id}",
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": result.get("input_tokens", 0),
                        "completion_tokens": result.get("output_tokens", 0),
                        "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0)
                    }
                }
        
        except HTTPException:
            raise
        except httpx.TimeoutException:
            error_msg = "Request timeout"
            status_code = 408
            if retry < max_retries:
                print(f"[Universal] 请求超时，重试 {retry + 1}/{max_retries}")
                await asyncio.sleep(0.5 * (2 ** retry))
                continue
            if flow_id:
                flow_monitor.fail_flow(flow_id, "timeout_error", "Request timeout", 408)
            raise HTTPException(408, "Request timeout after retries")
        except httpx.ConnectError:
            error_msg = "Connection error"
            status_code = 502
            if retry < max_retries:
                print(f"[Universal] 连接错误，重试 {retry + 1}/{max_retries}")
                await asyncio.sleep(0.5 * (2 ** retry))
                continue
            if flow_id:
                flow_monitor.fail_flow(flow_id, "connection_error", "Connection error", 502)
            raise HTTPException(502, "Connection error after retries")
        except Exception as e:
            error_msg = str(e)
            status_code = 500
            if is_retryable_error(None, e) and retry < max_retries:
                await asyncio.sleep(0.5 * (2 ** retry))
                continue
            if flow_id:
                flow_monitor.fail_flow(flow_id, "api_error", str(e), 500)
            raise HTTPException(500, str(e))
    
    raise HTTPException(503, "All retries exhausted")


def _convert_tool_uses_to_openai(tool_uses: List[dict]) -> List[dict]:
    """将 Kiro 工具调用转换为 OpenAI 格式"""
    tool_calls = []
    for tool_use in tool_uses:
        if tool_use.get("type") == "tool_use":
            tool_calls.append({
                "id": tool_use.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": tool_use.get("name", ""),
                    "arguments": json.dumps(tool_use.get("input", {}))
                }
            })
    return tool_calls


def _format_stream_chunk(log_id: str, model: str, content: str, is_first: bool = False, finish_reason: str = None) -> str:
    """格式化流式响应块"""
    chunk = {
        "id": f"chatcmpl-{log_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    }
    
    if is_first:
        chunk["choices"][0]["delta"] = {"role": "assistant", "content": ""}
    elif content:
        chunk["choices"][0]["delta"] = {"content": content}
    elif finish_reason:
        chunk["choices"][0]["delta"] = {}
    
    return f"data: {json.dumps(chunk)}\n\n"


def _format_tool_calls_chunk(log_id: str, model: str, tool_calls: List[dict]) -> str:
    """格式化工具调用流式响应块"""
    chunk = {
        "id": f"chatcmpl-{log_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": tool_calls
            },
            "finish_reason": None
        }]
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _format_error_chunk(error_type: str, message: str) -> str:
    """格式化错误响应块"""
    error = {
        "error": {
            "type": error_type,
            "message": message
        }
    }
    return f"data: {json.dumps(error)}\n\n"
