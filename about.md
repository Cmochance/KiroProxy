这份文档总结了我们对 **Kiro IDE** 进行逆向分析和调试的所有关键发现。你可以直接将其喂给你的 AI 辅助工具，以便它为你生成反向代理代码或补丁逻辑。

---

# Kiro IDE 逆向工程与请求分析报告

## 1. 基础环境信息

* **OS**: Arch Linux (x86_64)
* **IDE 名称**: Kiro (基于 VS Code / Code OSS)
* **二进制路径**: `/usr/bin/kiro`
* **源码位置**: `/opt/kiro/resources/app/out/`
* **核心渲染层**: `/opt/kiro/resources/app/out/vs/code/electron-browser/workbench/workbench.html` (及对应的 `workbench.js`)

## 2. 调试入口 (DevTools)

* **常规快捷键**: `Ctrl+Shift+I` 或 `F12` 在某些版本中被禁用。
* **强制开启方式**:
* 启动参数: `--remote-debugging-port=9222`
* 调试地址: `http://127.0.0.1:9222`
* **已知障碍**: 在国内网络环境下，Chrome 尝试从 `chrome-devtools-frontend.appspot.com` 加载调试器前端会导致 404，需使用 `chrome://inspect` 或离线 DevTools 链接。



## 3. 网络与 API 发现

* **已知域名**:
* 更新/元数据: `prod.download.desktop.kiro.dev`
* 潜在 API 域名: `api.kiro.dev`, `auth.kiro.dev`


* **请求特征**:
* 默认使用 HTTPS。
* 存在 DNS 探测行为，若解析失败会触发内部重试逻辑。
* 在启动时会检测代理设置，不兼容的 `--proxy-server` 参数可能导致程序闪退。



## 4. 关键注入与劫持点

### A. 修改 JS 源码 (硬劫持)

通过 `sed` 或编辑器直接修改 `workbench.js` 中的字符串，实现 API 重定向：

```bash
# 替换示例
https://api.kiro.dev -> http://127.0.0.1:8000

```

### B. 运行时劫持 (Console Hook)

在开发者面板中注入以下代码可实时监控并拦截 AI 请求：

```javascript
const _oldFetch = window.fetch;
window.fetch = function(url, options) {
    if (url.includes("kiro.dev")) {
        console.log("拦截到 AI 请求:", url, options);
        // 可在此处修改 url 指向本地服务器
    }
    return _oldFetch(url, options);
};

```

### C. 环境变量保护绕过

Kiro/Electron 校验 SSL 证书，若要通过 Mitmproxy 等工具抓包，必须设置：

```bash
export NODE_TLS_REJECT_UNAUTHORIZED=0

```

## 5. 后续反向代理目标

1. **协议转换**: 将 Kiro 的私有 API 格式转换为 OpenAI 兼容格式。
2. **流量重定向**: 通过修改 `/etc/hosts` 或 JS 源码，将所有 AI 对话请求引导至本地 Python/Node.js 服务器。
3. **鉴权模拟**: 模拟登录成功后的 Token 返回，绕过官方授权检测。

---

**给 AI 的指令建议：**

> “基于以上关于 Kiro IDE 的信息，请帮我写一个 Python (FastAPI/Flask) 后端，用来接收来自该 IDE 的请求，并将其转发给 DeepSeek API，同时请处理好请求头中的鉴权字段转换。”