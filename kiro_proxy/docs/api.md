# API 端点

## AI 协议
| 协议 | 端点 | 用途 |
|------|------|------|
| OpenAI | POST /v1/chat/completions | Codex CLI |
| OpenAI | GET /v1/models | 模型列表 |
| Anthropic | POST /v1/messages | Claude Code |
| Gemini | POST /v1/models/{model}:generateContent | Gemini CLI |

## 管理 API
- GET /api/accounts - 账号列表
- POST /api/accounts/{id}/refresh - 刷新 Token
- GET /api/accounts/{id}/usage - 查询用量
- GET /api/flows - 流量记录
- POST /api/speedtest - 速度测试

## 模型映射
| 请求 | 实际 |
|------|------|
| gpt-4o | claude-sonnet-4 |
| gpt-4o-mini | claude-haiku-4.5 |
| o1 | claude-opus-4.5 |
