# 功能特性

## 多协议支持
- OpenAI: `/v1/chat/completions`
- Anthropic: `/v1/messages`
- Gemini: `/v1/models/{model}:generateContent`

## 多账号管理
- 支持添加多个账号
- 自动轮询负载均衡
- 会话粘性 60 秒

## Token 自动刷新
- 每 5 分钟检查
- 提前 15 分钟刷新

## 配额管理
- 429 自动冷却 5 分钟
- 自动切换账号
- 冷却后自动恢复

## 登录方式
- Google OAuth
- GitHub OAuth
- AWS Builder ID
