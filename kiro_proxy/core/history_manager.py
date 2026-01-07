"""历史消息管理器 - 处理对话长度限制

提供多种策略处理 Kiro API 的输入长度限制：
1. 自动截断 - 保留最近 N 条消息
2. 智能摘要 - 压缩早期消息（需要额外 API 调用）
3. 错误重试 - 捕获错误后截断重试
4. 预估检测 - 发送前预估并截断
"""
import json
import httpx
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class TruncateStrategy(str, Enum):
    """截断策略"""
    NONE = "none"                    # 不截断
    AUTO_TRUNCATE = "auto_truncate"  # 自动截断（保留最近 N 条）
    SMART_SUMMARY = "smart_summary"  # 智能摘要
    ERROR_RETRY = "error_retry"      # 错误时截断重试
    PRE_ESTIMATE = "pre_estimate"    # 预估检测


@dataclass
class HistoryConfig:
    """历史消息配置"""
    # 启用的策略（可多选）
    strategies: List[TruncateStrategy] = field(default_factory=lambda: [TruncateStrategy.ERROR_RETRY])
    
    # 自动截断配置
    max_messages: int = 30           # 最大消息数
    max_chars: int = 150000          # 最大字符数（约 50k tokens）
    
    # 智能摘要配置
    summary_keep_recent: int = 10    # 摘要时保留最近 N 条完整消息
    summary_threshold: int = 100000  # 触发摘要的字符数阈值
    summary_max_length: int = 2000   # 摘要最大长度
    
    # 错误重试配置
    retry_max_messages: int = 15     # 重试时保留的消息数
    max_retries: int = 2             # 最大重试次数
    
    # 预估配置
    estimate_threshold: int = 180000  # 预估阈值（字符数）
    chars_per_token: float = 3.0      # 每 token 约等于多少字符
    
    # 是否添加截断警告
    add_warning_header: bool = True
    
    def to_dict(self) -> dict:
        return {
            "strategies": [s.value for s in self.strategies],
            "max_messages": self.max_messages,
            "max_chars": self.max_chars,
            "summary_keep_recent": self.summary_keep_recent,
            "summary_threshold": self.summary_threshold,
            "summary_max_length": self.summary_max_length,
            "retry_max_messages": self.retry_max_messages,
            "max_retries": self.max_retries,
            "estimate_threshold": self.estimate_threshold,
            "chars_per_token": self.chars_per_token,
            "add_warning_header": self.add_warning_header,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "HistoryConfig":
        strategies = [TruncateStrategy(s) for s in data.get("strategies", ["error_retry"])]
        return cls(
            strategies=strategies,
            max_messages=data.get("max_messages", 30),
            max_chars=data.get("max_chars", 150000),
            summary_keep_recent=data.get("summary_keep_recent", 10),
            summary_threshold=data.get("summary_threshold", 100000),
            summary_max_length=data.get("summary_max_length", 2000),
            retry_max_messages=data.get("retry_max_messages", 15),
            max_retries=data.get("max_retries", 2),
            estimate_threshold=data.get("estimate_threshold", 180000),
            chars_per_token=data.get("chars_per_token", 3.0),
            add_warning_header=data.get("add_warning_header", True),
        )


class HistoryManager:
    """历史消息管理器"""
    
    def __init__(self, config: HistoryConfig = None):
        self.config = config or HistoryConfig()
        self._truncated = False
        self._truncate_info = ""
    
    @property
    def was_truncated(self) -> bool:
        """是否发生了截断"""
        return self._truncated
    
    @property
    def truncate_info(self) -> str:
        """截断信息"""
        return self._truncate_info
    
    def reset(self):
        """重置状态"""
        self._truncated = False
        self._truncate_info = ""
    
    def estimate_tokens(self, text: str) -> int:
        """估算 token 数量"""
        return int(len(text) / self.config.chars_per_token)
    
    def estimate_history_size(self, history: List[dict]) -> Tuple[int, int]:
        """估算历史消息大小
        
        Returns:
            (message_count, char_count)
        """
        char_count = len(json.dumps(history, ensure_ascii=False))
        return len(history), char_count
    
    def truncate_by_count(self, history: List[dict], max_count: int) -> List[dict]:
        """按消息数量截断"""
        if len(history) <= max_count:
            return history
        
        original_count = len(history)
        truncated = history[-max_count:]
        self._truncated = True
        self._truncate_info = f"按数量截断: {original_count} -> {len(truncated)} 条消息"
        return truncated
    
    def truncate_by_chars(self, history: List[dict], max_chars: int) -> List[dict]:
        """按字符数截断"""
        total_chars = len(json.dumps(history, ensure_ascii=False))
        if total_chars <= max_chars:
            return history
        
        original_count = len(history)
        # 从后往前保留
        result = []
        current_chars = 0
        
        for msg in reversed(history):
            msg_chars = len(json.dumps(msg, ensure_ascii=False))
            if current_chars + msg_chars > max_chars and result:
                break
            result.insert(0, msg)
            current_chars += msg_chars
        
        if len(result) < original_count:
            self._truncated = True
            self._truncate_info = f"按字符数截断: {original_count} -> {len(result)} 条消息 ({total_chars} -> {current_chars} 字符)"
        
        return result
    
    def _extract_text(self, content) -> str:
        """从消息内容中提取文本"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts)
        if isinstance(content, dict):
            return content.get("text", "") or content.get("content", "")
        return str(content) if content else ""
    
    def _format_history_for_summary(self, history: List[dict]) -> str:
        """格式化历史消息用于生成摘要"""
        lines = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = self._extract_text(msg.get("content", ""))
            # 截断过长的单条消息
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)
    
    async def generate_summary(self, history: List[dict], api_caller: Callable) -> Optional[str]:
        """生成历史消息摘要
        
        Args:
            history: 需要摘要的历史消息
            api_caller: API 调用函数，签名为 async (prompt: str) -> str
        
        Returns:
            摘要文本，失败返回 None
        """
        if not history:
            return None
        
        formatted = self._format_history_for_summary(history)
        # 限制输入长度
        if len(formatted) > 10000:
            formatted = formatted[:10000] + "\n...(truncated)"
        
        prompt = f"""请简洁地总结以下对话历史的关键信息，包括：
1. 用户的主要目标和需求
2. 已完成的重要操作
3. 当前的工作状态和上下文

对话历史：
{formatted}

请用中文输出摘要，控制在 {self.config.summary_max_length} 字符以内："""
        
        try:
            summary = await api_caller(prompt)
            if summary and len(summary) > self.config.summary_max_length:
                summary = summary[:self.config.summary_max_length] + "..."
            return summary
        except Exception as e:
            print(f"[HistoryManager] 生成摘要失败: {e}")
            return None
    
    async def compress_with_summary(
        self, 
        history: List[dict], 
        api_caller: Callable
    ) -> List[dict]:
        """使用智能摘要压缩历史消息
        
        Args:
            history: 历史消息
            api_caller: API 调用函数
        
        Returns:
            压缩后的历史消息
        """
        total_chars = len(json.dumps(history, ensure_ascii=False))
        if total_chars <= self.config.summary_threshold:
            return history
        
        if len(history) <= self.config.summary_keep_recent:
            return history
        
        # 分离早期消息和最近消息
        keep_recent = self.config.summary_keep_recent
        old_history = history[:-keep_recent]
        recent_history = history[-keep_recent:]
        
        # 生成摘要
        summary = await self.generate_summary(old_history, api_caller)
        
        if not summary:
            # 摘要失败，回退到简单截断
            self._truncated = True
            self._truncate_info = f"摘要生成失败，回退截断: {len(history)} -> {len(recent_history)} 条消息"
            return recent_history
        
        # 构建带摘要的历史
        summary_msg = {
            "role": "user",
            "content": f"[Earlier conversation summary]\n{summary}\n\n[Continuing from recent messages...]"
        }
        
        # 确保消息交替正确
        result = [summary_msg]
        # 添加一个 assistant 响应确保交替
        result.append({
            "role": "assistant",
            "content": "I understand the context. Let's continue."
        })
        result.extend(recent_history)
        
        self._truncated = True
        self._truncate_info = f"智能摘要: {len(history)} -> {len(result)} 条消息 (摘要 {len(summary)} 字符)"
        
        return result
    
    def should_pre_truncate(self, history: List[dict], user_content: str) -> bool:
        """检查是否需要预截断"""
        if TruncateStrategy.PRE_ESTIMATE not in self.config.strategies:
            return False
        
        total_chars = len(json.dumps(history, ensure_ascii=False)) + len(user_content)
        return total_chars > self.config.estimate_threshold
    
    def should_summarize(self, history: List[dict]) -> bool:
        """检查是否需要智能摘要"""
        if TruncateStrategy.SMART_SUMMARY not in self.config.strategies:
            return False
        
        total_chars = len(json.dumps(history, ensure_ascii=False))
        return total_chars > self.config.summary_threshold and len(history) > self.config.summary_keep_recent
    
    def pre_process(self, history: List[dict], user_content: str = "") -> List[dict]:
        """预处理历史消息（发送前，同步版本）
        
        根据配置的策略进行预处理（不包括智能摘要）
        """
        self.reset()
        
        if not history:
            return history
        
        result = history
        
        # 策略 1: 自动截断
        if TruncateStrategy.AUTO_TRUNCATE in self.config.strategies:
            # 先按数量截断
            result = self.truncate_by_count(result, self.config.max_messages)
            # 再按字符数截断
            result = self.truncate_by_chars(result, self.config.max_chars)
        
        # 策略 4: 预估检测
        if TruncateStrategy.PRE_ESTIMATE in self.config.strategies:
            total_chars = len(json.dumps(result, ensure_ascii=False)) + len(user_content)
            if total_chars > self.config.estimate_threshold:
                # 计算需要保留的消息数
                target_chars = int(self.config.estimate_threshold * 0.8)  # 留 20% 余量
                result = self.truncate_by_chars(result, target_chars)
        
        return result
    
    async def pre_process_async(
        self, 
        history: List[dict], 
        user_content: str = "",
        api_caller: Callable = None
    ) -> List[dict]:
        """预处理历史消息（发送前，异步版本，支持智能摘要）
        
        Args:
            history: 历史消息
            user_content: 当前用户消息
            api_caller: API 调用函数，用于生成摘要
        """
        self.reset()
        
        if not history:
            return history
        
        result = history
        
        # 策略 2: 智能摘要（优先级最高）
        if TruncateStrategy.SMART_SUMMARY in self.config.strategies and api_caller:
            if self.should_summarize(result):
                result = await self.compress_with_summary(result, api_caller)
                # 摘要后如果还是太长，继续其他策略
        
        # 策略 1: 自动截断
        if TruncateStrategy.AUTO_TRUNCATE in self.config.strategies:
            result = self.truncate_by_count(result, self.config.max_messages)
            result = self.truncate_by_chars(result, self.config.max_chars)
        
        # 策略 4: 预估检测
        if TruncateStrategy.PRE_ESTIMATE in self.config.strategies:
            total_chars = len(json.dumps(result, ensure_ascii=False)) + len(user_content)
            if total_chars > self.config.estimate_threshold:
                target_chars = int(self.config.estimate_threshold * 0.8)
                result = self.truncate_by_chars(result, target_chars)
        
        return result
    
    def handle_length_error(self, history: List[dict], retry_count: int = 0) -> Tuple[List[dict], bool]:
        """处理长度超限错误
        
        Args:
            history: 历史消息
            retry_count: 当前重试次数
        
        Returns:
            (truncated_history, should_retry)
        """
        # 策略 3: 错误重试
        if TruncateStrategy.ERROR_RETRY not in self.config.strategies:
            return history, False
        
        if retry_count >= self.config.max_retries:
            return history, False
        
        # 根据重试次数逐步减少消息
        factor = 1.0 - (retry_count * 0.3)  # 每次减少 30%
        target_count = max(5, int(self.config.retry_max_messages * factor))
        
        self.reset()
        truncated = self.truncate_by_count(history, target_count)
        
        if len(truncated) < len(history):
            self._truncate_info = f"错误重试截断 (第 {retry_count + 1} 次): {len(history)} -> {len(truncated)} 条消息"
            return truncated, True
        
        return history, False
    
    def get_warning_header(self) -> Optional[str]:
        """获取截断警告头"""
        if not self.config.add_warning_header or not self._truncated:
            return None
        return self._truncate_info


# 全局配置实例
_history_config = HistoryConfig()


def get_history_config() -> HistoryConfig:
    """获取历史消息配置"""
    return _history_config


def set_history_config(config: HistoryConfig):
    """设置历史消息配置"""
    global _history_config
    _history_config = config


def update_history_config(data: dict):
    """更新历史消息配置"""
    global _history_config
    _history_config = HistoryConfig.from_dict(data)


def is_content_length_error(status_code: int, error_text: str) -> bool:
    """检查是否为内容长度超限错误"""
    if "CONTENT_LENGTH_EXCEEDS_THRESHOLD" in error_text:
        return True
    if "Input is too long" in error_text:
        return True
    # 更宽松的匹配
    lowered = error_text.lower()
    if "too long" in lowered and ("input" in lowered or "content" in lowered or "message" in lowered):
        return True
    return False
