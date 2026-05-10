# Application Layer - 应用层

> 我们日常用的网络服务（网页、邮件、DNS）底层是什么协议？

## 🎯 应用层的作用

**应用层**是网络协议栈的最顶层，直接为用户和应用程序提供网络服务。

```
常见协议：
- HTTP/HTTPS：网页浏览
- DNS：域名解析
- SMTP/POP3/IMAP：电子邮件
- FTP：文件传输
- SSH：远程登录
- WebSocket：实时通信
```

---

## 🌐 HTTP 协议

### 核心特征

| 特征 | 说明 |
|---|---|
| 无状态 | 每个请求独立，服务器不记忆客户端 |
| 基于 TCP | 可靠传输 |
| 请求-响应模式 | 客户端发起，服务器响应 |
| 文本协议 | 易读易调试 |

### 版本演进

| 版本 | 核心改进 |
|---|---|
| HTTP/0.9 | 只支持 GET |
| HTTP/1.0 | +POST/HEAD, 每个请求新连接 |
| HTTP/1.1 | 持久连接、管道化、Host 头 |
| HTTP/2 | 二进制帧、多路复用、HPACK 头部压缩 |
| HTTP/3 | 基于 QUIC/UDP，0-RTT 连接 |

### 请求/响应格式

```
请求: GET /index.html HTTP/1.1          ← 方法 URI 版本
      Host: www.example.com             ← 请求头 (键值对)
      (空行)
      (可选的请求体)

响应: HTTP/1.1 200 OK                   ← 版本 状态码 原因短语
      Content-Type: text/html           ← 响应头
      Content-Length: 138
      (空行)
      <!DOCTYPE html><html>...          ← 响应体
```

### HTTP 方法

| 方法 | 作用 | 幂等 |
|---|---|---|
| GET | 获取资源 | ✅ |
| POST | 提交数据 | ❌ |
| PUT | 更新资源 | ✅ |
| DELETE | 删除资源 | ✅ |
| HEAD | 获取响应头（无 body） | ✅ |
| PATCH | 部分更新 | ❌ |

### 状态码速览

| 分类 | 含义 | 常见例子 |
|---|---|---|
| 1xx | 信息性 | 101 Switching Protocols |
| 2xx | 成功 | 200 OK, 201 Created |
| 3xx | 重定向 | 301 永久, 302 临时, 304 未修改（缓存） |
| 4xx | 客户端错误 | 400 Bad Request, 401 未认证, 403 禁止, 404 未找到 |
| 5xx | 服务器错误 | 500 内部错误, 502 Bad Gateway, 503 服务不可用 |

---

## 🔐 HTTPS (HTTP + SSL/TLS)

```
┌───────────────────┐
│     HTTP          │
├───────────────────┤  ← 这一层提供加密
│   SSL/TLS 层      │
├───────────────────┤
│     TCP           │
└───────────────────┘
```

- **端口**：HTTPS = 443，HTTP = 80
- **核心功能**：数据加密、身份认证（证书）、数据完整性
- **TLS 握手**：ClientHello → ServerHello + Certificate → 密钥交换 → Finished → 加密通信

---

## 🔍 DNS 协议

### DNS 层次结构

```
根域 (.) → 顶级域 (.com/.org/.cn) → 二级域 (google.com) → 主机名 (www)
```

### DNS 查询流程

客户端 → (递归) 本地DNS → (迭代) 根DNS → .com DNS → google.com DNS → 返回 IP
- **端口**：UDP 53
- **记录类型**：A (IPv4), AAAA (IPv6), CNAME (别名), MX (邮件), NS (域名服务器)

---

## 📧 邮件协议

| 协议 | 端口 | 用途 | 特点 |
|---|---|---|---|
| **SMTP** | 25/587 | 发送 | HELO → MAIL FROM → RCPT TO → DATA |
| **POP3** | 110/995 | 接收 | 下载到本地，服务器删除 |
| **IMAP** | 143/993 | 接收 | 保留服务器，多设备同步 |

---

## 📁 FTP 协议

端口 21 (控制) + 20/随机 (数据)
| 模式 | 数据方向 |
|---|---|
| 主动 (PORT) | 服务器 → 客户端 |
| 被动 (PASV) | 客户端 → 服务器 ✅ 防火墙友好 |

---

## 🔗 概念关系图

```
应用层: HTTP DNS SMTP POP3 IMAP FTP SSH
           │        │
      ┌────┴─┐   ┌─┴────┐
      │HTTPS │   │ TCP  │
      │SSL/  │   └──────┘
      │TLS   │
      └──────┘
           ↓
      传输层 (TCP/UDP)
```

---

> 代码实现（简易 HTTP 服务器、DNS 查询）见 [practices/systems/networks/](../../practices/systems/networks/)
