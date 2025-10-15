# Application Layer - 应用层

> 我们日常使用的网络服务是如何工作的？HTTP、DNS、FTP等协议如何实现？

## 🎯 应用层的作用

**应用层**是网络协议栈的最顶层，直接为用户提供网络服务。

```
功能：
- 为应用程序提供网络服务
- 定义应用间的通信协议
- 数据表示和编码
- 用户接口

常见协议：
- HTTP/HTTPS：网页浏览
- DNS：域名解析
- FTP：文件传输
- SMTP/POP3/IMAP：邮件
- SSH：远程登录
- WebSocket：实时通信

类比：
应用层 = 各种应用软件
浏览器使用HTTP
邮件客户端使用SMTP/POP3
```

---

## 🌐 HTTP协议

### HTTP简介

```
HTTP (HyperText Transfer Protocol) - 超文本传输协议

特点：
- 无状态：每个请求独立
- 基于TCP：可靠传输
- 请求-响应模式：客户端发起
- 文本协议：易读易调试

版本：
- HTTP/0.9：1991年，只支持GET
- HTTP/1.0：1996年，增加POST、HEAD等
- HTTP/1.1：1997年，持久连接、管道化
- HTTP/2：2015年，二进制帧、多路复用
- HTTP/3：2022年，基于QUIC/UDP
```

### HTTP请求方法

```
GET     - 获取资源（幂等）
POST    - 提交数据（非幂等）
PUT     - 更新资源（幂等）
DELETE  - 删除资源（幂等）
HEAD    - 获取响应头（不获取body）
OPTIONS - 查询支持的方法
PATCH   - 部分更新资源
```

### HTTP请求格式

```
请求行：方法 URI HTTP/版本
请求头：键值对
空行
请求体（可选）

例子：
GET /index.html HTTP/1.1
Host: www.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Connection: keep-alive

```

### HTTP响应格式

```
状态行：HTTP/版本 状态码 状态描述
响应头：键值对
空行
响应体

例子：
HTTP/1.1 200 OK
Content-Type: text/html
Content-Length: 138
Connection: keep-alive

<!DOCTYPE html>
<html>
<body>Hello, World!</body>
</html>
```

### HTTP状态码

```python
# 常见HTTP状态码
HTTP_STATUS_CODES = {
    # 1xx：信息性状态码
    100: "Continue",
    101: "Switching Protocols",

    # 2xx：成功
    200: "OK",
    201: "Created",
    204: "No Content",

    # 3xx：重定向
    301: "Moved Permanently",     # 永久重定向
    302: "Found",                 # 临时重定向
    304: "Not Modified",          # 缓存有效

    # 4xx：客户端错误
    400: "Bad Request",
    401: "Unauthorized",          # 未认证
    403: "Forbidden",             # 无权限
    404: "Not Found",
    405: "Method Not Allowed",

    # 5xx：服务器错误
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}

def explain_status(code):
    """解释状态码"""
    if code in HTTP_STATUS_CODES:
        return f"{code} {HTTP_STATUS_CODES[code]}"
    elif 100 <= code < 200:
        return f"{code} 信息性响应"
    elif 200 <= code < 300:
        return f"{code} 成功"
    elif 300 <= code < 400:
        return f"{code} 重定向"
    elif 400 <= code < 500:
        return f"{code} 客户端错误"
    elif 500 <= code < 600:
        return f"{code} 服务器错误"
    else:
        return f"{code} 未知状态码"

# 测试
for code in [200, 301, 404, 500]:
    print(explain_status(code))
```

### 实现简单的HTTP服务器

```python
import socket

class SimpleHTTPServer:
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
        self.routes = {}

    def route(self, path):
        """装饰器：注册路由"""
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator

    def parse_request(self, request):
        """解析HTTP请求"""
        lines = request.split('\r\n')
        request_line = lines[0]
        parts = request_line.split(' ')

        if len(parts) >= 3:
            method = parts[0]
            path = parts[1]
            version = parts[2]

            # 解析请求头
            headers = {}
            for line in lines[1:]:
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    headers[key] = value

            return {
                'method': method,
                'path': path,
                'version': version,
                'headers': headers
            }
        return None

    def build_response(self, status_code, headers, body):
        """构建HTTP响应"""
        status_messages = {
            200: "OK",
            404: "Not Found",
            500: "Internal Server Error"
        }

        status_msg = status_messages.get(status_code, "Unknown")
        response = f"HTTP/1.1 {status_code} {status_msg}\r\n"

        # 添加响应头
        for key, value in headers.items():
            response += f"{key}: {value}\r\n"

        response += "\r\n"  # 空行
        response += body

        return response

    def handle_request(self, request_data):
        """处理请求"""
        request = self.parse_request(request_data)

        if not request:
            return self.build_response(400, {}, "Bad Request")

        path = request['path']

        # 查找路由
        if path in self.routes:
            try:
                body = self.routes[path](request)
                headers = {
                    'Content-Type': 'text/html',
                    'Content-Length': str(len(body)),
                    'Connection': 'close'
                }
                return self.build_response(200, headers, body)
            except Exception as e:
                return self.build_response(500, {}, f"Internal Error: {e}")
        else:
            return self.build_response(404, {}, "404 Not Found")

    def start(self):
        """启动服务器"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(5)

        print(f"HTTP服务器启动在 http://{self.host}:{self.port}")

        try:
            while True:
                client_sock, addr = sock.accept()
                print(f"收到来自 {addr} 的连接")

                # 接收请求
                request_data = client_sock.recv(4096).decode('utf-8')
                print(f"请求:\n{request_data[:200]}...")

                # 处理请求
                response = self.handle_request(request_data)

                # 发送响应
                client_sock.sendall(response.encode('utf-8'))
                client_sock.close()
        except KeyboardInterrupt:
            print("\n服务器关闭")
        finally:
            sock.close()

# 使用
app = SimpleHTTPServer(port=8080)

@app.route('/')
def index(request):
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Simple HTTP Server</title></head>
    <body>
        <h1>Welcome to Simple HTTP Server!</h1>
        <p>This is a basic HTTP server written in Python.</p>
    </body>
    </html>
    """

@app.route('/hello')
def hello(request):
    return "<h1>Hello, World!</h1>"

# 启动服务器（取消注释以运行）
# app.start()
```

### HTTP客户端

```python
import socket

class HTTPClient:
    def __init__(self):
        pass

    def request(self, url, method='GET', headers=None, body=None):
        """发送HTTP请求"""
        # 解析URL
        if url.startswith('http://'):
            url = url[7:]

        if '/' in url:
            host, path = url.split('/', 1)
            path = '/' + path
        else:
            host = url
            path = '/'

        # 默认端口
        if ':' in host:
            host, port = host.split(':')
            port = int(port)
        else:
            port = 80

        # 构建请求
        request = f"{method} {path} HTTP/1.1\r\n"
        request += f"Host: {host}\r\n"

        if headers:
            for key, value in headers.items():
                request += f"{key}: {value}\r\n"

        request += "Connection: close\r\n"
        request += "\r\n"

        if body:
            request += body

        # 发送请求
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.sendall(request.encode('utf-8'))

        # 接收响应
        response = b''
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk

        sock.close()

        # 解析响应
        response_str = response.decode('utf-8', errors='ignore')
        parts = response_str.split('\r\n\r\n', 1)

        headers_str = parts[0]
        body = parts[1] if len(parts) > 1 else ''

        # 解析状态行
        lines = headers_str.split('\r\n')
        status_line = lines[0]
        status_parts = status_line.split(' ', 2)
        status_code = int(status_parts[1])

        return {
            'status_code': status_code,
            'headers': headers_str,
            'body': body
        }

    def get(self, url):
        """GET请求"""
        return self.request(url, 'GET')

# 使用
client = HTTPClient()

# 发送GET请求
response = client.get('http://example.com')
print(f"状态码: {response['status_code']}")
print(f"响应体前100字符:\n{response['body'][:100]}...")
```

---

## 🔐 HTTPS协议

### HTTPS vs HTTP

```
HTTPS = HTTP + SSL/TLS

区别：
┌─────────────────────────────────────┐
│           应用层 (HTTP)              │
├─────────────────────────────────────┤
│  SSL/TLS层 (加密)                    │ ← HTTPS多了这层
├─────────────────────────────────────┤
│           传输层 (TCP)               │
└─────────────────────────────────────┘

特点：
✅ 数据加密：防止窃听
✅ 身份认证：防止冒充
✅ 数据完整性：防止篡改
❌ 性能开销：加密解密
❌ 需要证书：成本

端口：
- HTTP：80
- HTTPS：443
```

### HTTPS握手过程

```
客户端                                 服务器
  │                                      │
  │────── ClientHello ─────────────────→│
  │       (支持的加密算法、随机数)        │
  │                                      │
  │←────── ServerHello ─────────────────│
  │       (选择的加密算法、随机数)        │
  │                                      │
  │←────── Certificate ─────────────────│
  │       (服务器证书)                   │
  │                                      │
  │←────── ServerHelloDone ─────────────│
  │                                      │
  │────── ClientKeyExchange ───────────→│
  │       (加密的预主密钥)               │
  │                                      │
  │────── ChangeCipherSpec ────────────→│
  │       (开始使用加密)                 │
  │                                      │
  │────── Finished ────────────────────→│
  │                                      │
  │←────── ChangeCipherSpec ────────────│
  │                                      │
  │←────── Finished ────────────────────│
  │                                      │
  ├──────── 加密通信开始 ────────────────┤
```

---

## 🔍 DNS协议

### DNS简介

```
DNS (Domain Name System) - 域名系统

作用：将域名转换为IP地址

例子：
www.google.com → 142.250.185.46

为什么需要DNS？
- IP地址难记
- IP地址可能变化
- 负载均衡
- 故障转移
```

### DNS层次结构

```
根域名服务器 (.)
    ↓
顶级域名服务器 (.com, .org, .cn)
    ↓
权威域名服务器 (google.com, baidu.com)
    ↓
子域名 (www.google.com, mail.google.com)

例子：www.google.com.
      │   │      │   │
      │   │      │   └─ 根域
      │   │      └───── 顶级域(.com)
      │   └──────────── 二级域(google)
      └──────────────── 主机名(www)
```

### DNS查询过程

```
1. 递归查询（客户端 → 本地DNS）
   客户端: "www.google.com的IP是什么？"
   本地DNS: "我帮你查！"

2. 迭代查询（本地DNS → 各级DNS服务器）
   本地DNS → 根DNS: "www.google.com?"
   根DNS: "去找.com服务器"

   本地DNS → .com服务器: "www.google.com?"
   .com服务器: "去找google.com服务器"

   本地DNS → google.com服务器: "www.google.com?"
   google.com服务器: "142.250.185.46"

3. 返回结果
   本地DNS → 客户端: "142.250.185.46"
```

### DNS报文格式

```
DNS使用UDP，端口53

报文结构：
┌─────────────┐
│   报头      │ 12字节
├─────────────┤
│   问题      │ 域名查询
├─────────────┤
│   回答      │ 查询结果
├─────────────┤
│   权威      │ 权威服务器
├─────────────┤
│   附加      │ 额外信息
└─────────────┘
```

### 实现DNS查询

```python
import socket
import struct

class DNSQuery:
    def __init__(self, domain):
        self.domain = domain
        self.transaction_id = 0x1234

    def build_query(self):
        """构建DNS查询报文"""
        # DNS头部（12字节）
        header = struct.pack(
            '!HHHHHH',
            self.transaction_id,  # 事务ID
            0x0100,  # 标志（标准查询）
            1,       # 问题数
            0,       # 回答数
            0,       # 权威数
            0        # 附加数
        )

        # 问题部分：域名
        question = b''
        for part in self.domain.split('.'):
            question += bytes([len(part)]) + part.encode()
        question += b'\x00'  # 结束符

        # 查询类型和类
        question += struct.pack('!HH',
                               1,    # 类型：A记录（IPv4地址）
                               1)    # 类：IN（Internet）

        return header + question

    def parse_response(self, response):
        """解析DNS响应"""
        # 解析头部
        header = struct.unpack('!HHHHHH', response[:12])
        transaction_id = header[0]
        flags = header[1]
        questions = header[2]
        answers = header[3]

        # 跳过问题部分
        pos = 12
        while response[pos] != 0:
            length = response[pos]
            pos += length + 1
        pos += 5  # 跳过结束符和类型、类

        # 解析回答
        ips = []
        for _ in range(answers):
            # 名称（压缩格式）
            if (response[pos] & 0xC0) == 0xC0:
                pos += 2
            else:
                while response[pos] != 0:
                    pos += response[pos] + 1
                pos += 1

            # 类型、类、TTL、数据长度
            rtype, rclass, ttl, rdlength = struct.unpack(
                '!HHIH', response[pos:pos+10]
            )
            pos += 10

            # 如果是A记录，解析IP
            if rtype == 1 and rdlength == 4:
                ip = '.'.join(str(b) for b in response[pos:pos+4])
                ips.append(ip)

            pos += rdlength

        return ips

    def query(self, dns_server='8.8.8.8', port=53):
        """发送DNS查询"""
        # 构建查询
        query = self.build_query()

        # 发送UDP请求
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)

        try:
            sock.sendto(query, (dns_server, port))
            response, _ = sock.recvfrom(512)

            # 解析响应
            ips = self.parse_response(response)
            return ips
        except socket.timeout:
            return []
        finally:
            sock.close()

# 使用
dns = DNSQuery('www.google.com')
ips = dns.query()
print(f"www.google.com 的IP地址: {ips}")

# 使用系统的DNS解析（简单方法）
import socket
ip = socket.gethostbyname('www.google.com')
print(f"使用系统解析: {ip}")
```

---

## 📧 邮件协议

### SMTP（发送邮件）

```
SMTP (Simple Mail Transfer Protocol)

端口：25（明文）、587（TLS）

流程：
客户端                              服务器
  │                                   │
  │←────── 220 Welcome ───────────────│
  │                                   │
  │────── HELO client.com ───────────→│
  │                                   │
  │←────── 250 OK ─────────────────────│
  │                                   │
  │────── MAIL FROM:<sender@xxx> ────→│
  │                                   │
  │←────── 250 OK ─────────────────────│
  │                                   │
  │────── RCPT TO:<receiver@xxx> ────→│
  │                                   │
  │←────── 250 OK ─────────────────────│
  │                                   │
  │────── DATA ───────────────────────→│
  │                                   │
  │←────── 354 Start mail input ───────│
  │                                   │
  │────── Subject: Hello ─────────────→│
  │────── Body... ────────────────────→│
  │────── . ──────────────────────────→│ (结束标记)
  │                                   │
  │←────── 250 OK ─────────────────────│
  │                                   │
  │────── QUIT ───────────────────────→│
  │                                   │
  │←────── 221 Bye ────────────────────│
```

### POP3（接收邮件）

```
POP3 (Post Office Protocol 3)

端口：110（明文）、995（SSL）

特点：
- 下载邮件到本地
- 服务器删除邮件（可选）
- 离线访问

命令：
USER username
PASS password
STAT           # 统计邮件数
LIST           # 列出邮件
RETR n         # 下载第n封邮件
DELE n         # 删除第n封邮件
QUIT
```

### IMAP（接收邮件）

```
IMAP (Internet Message Access Protocol)

端口：143（明文）、993（SSL）

特点：
- 邮件保留在服务器
- 可以多设备同步
- 在线访问
- 可以只下载邮件头

vs POP3：
IMAP更现代，支持文件夹、搜索等
```

---

## 📁 FTP协议

### FTP简介

```
FTP (File Transfer Protocol) - 文件传输协议

端口：
- 控制连接：21
- 数据连接：20（主动模式）或随机端口（被动模式）

特点：
- 两个连接：控制 + 数据
- 支持断点续传
- 支持目录操作

模式：
1. 主动模式（PORT）
   - 客户端打开随机端口
   - 服务器从20端口主动连接

2. 被动模式（PASV）
   - 服务器打开随机端口
   - 客户端主动连接
   - 更友好于防火墙
```

---

## 🔗 相关概念

- [传输层](transport-layer.md) - TCP/UDP
- [网络基础与分层模型](network-fundamentals.md) - 分层模型

---

**记住**：
1. 应用层直接为用户提供服务
2. HTTP是无状态的请求-响应协议
3. HTTPS = HTTP + SSL/TLS加密
4. DNS将域名转换为IP地址
5. SMTP用于发送邮件，POP3/IMAP用于接收
6. FTP使用两个连接：控制和数据
7. 理解HTTP状态码的含义
8. 应用层协议基于传输层协议（TCP/UDP）
