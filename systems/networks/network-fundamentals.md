# Network Fundamentals - 网络基础与分层模型

> 为什么需要分层？OSI和TCP/IP模型如何工作？

## 🎯 为什么需要分层？

### 问题：网络通信太复杂

```
如果没有分层，网络协议会是一团乱麻：
- 应用程序需要知道物理层细节
- 每个应用都要实现完整的网络栈
- 无法模块化，难以维护
- 技术更新困难
```

### 解决：分层架构

```
分层的好处：
✅ 每层专注自己的职责
✅ 层与层之间接口清晰
✅ 下层为上层提供服务
✅ 上层不关心下层细节
✅ 易于替换和升级
```

---

## 📊 OSI七层模型

### OSI (Open Systems Interconnection)

```
7. 应用层    - 为应用程序提供网络服务
6. 表示层    - 数据格式转换、加密
5. 会话层    - 建立、管理、终止会话
4. 传输层    - 端到端的可靠传输
3. 网络层    - 路由选择、寻址
2. 数据链路层 - 帧传输、错误检测
1. 物理层    - 物理信号传输
```

### 各层详解

#### 7. 应用层 (Application Layer)

```
功能：
- 为应用程序提供网络服务
- 用户接口

协议：
- HTTP/HTTPS：网页浏览
- FTP：文件传输
- SMTP：邮件发送
- DNS：域名解析

例子：
用户在浏览器输入 www.google.com
↓
应用层使用HTTP协议
```

#### 6. 表示层 (Presentation Layer)

```
功能：
- 数据格式转换
- 数据加密/解密
- 数据压缩

例子：
- JPEG、GIF图片格式
- ASCII、UTF-8编码
- SSL/TLS加密
```

#### 5. 会话层 (Session Layer)

```
功能：
- 建立、管理、终止会话
- 同步和对话控制

例子：
- RPC (Remote Procedure Call)
- SQL会话
- 视频会议的会话管理
```

#### 4. 传输层 (Transport Layer)

```
功能：
- 端到端的数据传输
- 可靠性保证
- 流量控制
- 差错控制

协议：
- TCP：可靠的传输
- UDP：快速的传输

数据单位：段（Segment）
```

#### 3. 网络层 (Network Layer)

```
功能：
- 路径选择（路由）
- 逻辑寻址（IP地址）
- 分组转发

协议：
- IP：寻址和路由
- ICMP：错误报告（ping）
- ARP：地址解析

数据单位：包（Packet）
```

#### 2. 数据链路层 (Data Link Layer)

```
功能：
- 物理地址（MAC地址）
- 帧传输
- 错误检测
- 访问控制

协议：
- Ethernet（以太网）
- Wi-Fi（无线局域网）
- PPP（点对点协议）

数据单位：帧（Frame）
```

#### 1. 物理层 (Physical Layer)

```
功能：
- 物理信号传输
- 比特流传输
- 定义电气特性

介质：
- 双绞线（网线）
- 光纤
- 无线电波

数据单位：比特（Bit）
```

---

## 🌐 TCP/IP四层模型

### TCP/IP模型（实际使用的模型）

```
4. 应用层      - HTTP、FTP、DNS等
3. 传输层      - TCP、UDP
2. 网络层      - IP、ICMP
1. 网络接口层  - Ethernet、Wi-Fi
```

### 与OSI模型的对应关系

```
OSI 7层                TCP/IP 4层

应用层    ┐
表示层    ├──────────→ 应用层
会话层    ┘

传输层    ──────────→ 传输层

网络层    ──────────→ 网络层

数据链路层 ┐
物理层     ┘─────────→ 网络接口层
```

---

## 📦 数据封装与解封装

### 数据封装过程（发送端）

```
应用层：
┌─────────────────┐
│   应用数据      │
└─────────────────┘
        ↓ 添加应用层头部

传输层：
┌──────┬──────────┐
│TCP头 │ 应用数据 │ ← TCP段（Segment）
└──────┴──────────┘
        ↓ 添加IP头部

网络层：
┌──────┬──────┬──────────┐
│IP头  │TCP头 │ 应用数据 │ ← IP包（Packet）
└──────┴──────┴──────────┘
        ↓ 添加帧头和帧尾

数据链路层：
┌─────┬──────┬──────┬──────────┬─────┐
│帧头 │IP头  │TCP头 │ 应用数据 │帧尾 │ ← 帧（Frame）
└─────┴──────┴──────┴──────────┴─────┘
        ↓ 转换为比特流

物理层：
010110101010... ← 比特流（Bits）
```

### 数据解封装过程（接收端）

```
物理层：
010110101010... → 接收比特流
        ↓ 组装成帧

数据链路层：
┌─────┬──────┬──────┬──────────┬─────┐
│帧头 │IP头  │TCP头 │ 应用数据 │帧尾 │
└─────┴──────┴──────┴──────────┴─────┘
        ↓ 去掉帧头和帧尾

网络层：
┌──────┬──────┬──────────┐
│IP头  │TCP头 │ 应用数据 │
└──────┴──────┴──────────┘
        ↓ 去掉IP头

传输层：
┌──────┬──────────┐
│TCP头 │ 应用数据 │
└──────┴──────────┘
        ↓ 去掉TCP头

应用层：
┌─────────────────┐
│   应用数据      │
└─────────────────┘
```

### Python模拟封装过程

```python
class NetworkStack:
    """模拟网络协议栈"""

    @staticmethod
    def application_layer(data):
        """应用层：原始数据"""
        print(f"应用层: 数据 = {data}")
        return data

    @staticmethod
    def transport_layer(data, src_port, dst_port):
        """传输层：添加TCP头"""
        tcp_header = f"[TCP: {src_port}→{dst_port}]"
        segment = tcp_header + data
        print(f"传输层: {segment}")
        return segment

    @staticmethod
    def network_layer(data, src_ip, dst_ip):
        """网络层：添加IP头"""
        ip_header = f"[IP: {src_ip}→{dst_ip}]"
        packet = ip_header + data
        print(f"网络层: {packet}")
        return packet

    @staticmethod
    def datalink_layer(data, src_mac, dst_mac):
        """数据链路层：添加帧头"""
        frame_header = f"[Frame: {src_mac}→{dst_mac}]"
        frame_trailer = "[FCS]"
        frame = frame_header + data + frame_trailer
        print(f"链路层: {frame}")
        return frame

    @staticmethod
    def physical_layer(data):
        """物理层：转为比特流"""
        bits = ''.join(format(ord(c), '08b') for c in data)
        print(f"物理层: {bits[:50]}... ({len(bits)} bits)")
        return bits

# 发送端：封装
print("=== 发送端：数据封装 ===")
data = "Hello, Network!"
data = NetworkStack.application_layer(data)
data = NetworkStack.transport_layer(data, 8080, 80)
data = NetworkStack.network_layer(data, "192.168.1.10", "93.184.216.34")
data = NetworkStack.datalink_layer(data, "AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66")
bits = NetworkStack.physical_layer(data)

print("\n=== 传输中... ===\n")

# 接收端：解封装
print("=== 接收端：数据解封装 ===")
# （简化，实际需要解析各层头部）
print("物理层: 接收比特流")
print("链路层: 去掉帧头和帧尾")
print("网络层: 去掉IP头，检查目标IP")
print("传输层: 去掉TCP头，检查目标端口")
print(f"应用层: 得到数据 = Hello, Network!")
```

---

## 🔍 每层的PDU（协议数据单元）

```
层次            PDU名称        添加的内容
────────────────────────────────────────
应用层          Data          应用数据
传输层          Segment       TCP/UDP头
网络层          Packet        IP头
数据链路层      Frame         帧头+帧尾
物理层          Bits          比特流
```

---

## 📝 分层通信示例

### HTTP请求的完整过程

```python
# 模拟HTTP请求的分层处理

class HTTPRequest:
    """应用层"""
    def __init__(self, url):
        self.url = url
        self.method = "GET"

    def build(self):
        return f"{self.method} {self.url} HTTP/1.1\r\nHost: example.com\r\n\r\n"

class TCPSegment:
    """传输层"""
    def __init__(self, data, src_port, dst_port):
        self.src_port = src_port
        self.dst_port = dst_port
        self.data = data
        self.seq_num = 1000
        self.ack_num = 0

    def build(self):
        header = f"TCP[{self.src_port}→{self.dst_port}, Seq={self.seq_num}]"
        return f"{header}|{self.data}"

class IPPacket:
    """网络层"""
    def __init__(self, data, src_ip, dst_ip):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.ttl = 64
        self.data = data

    def build(self):
        header = f"IP[{self.src_ip}→{self.dst_ip}, TTL={self.ttl}]"
        return f"{header}|{self.data}"

class EthernetFrame:
    """数据链路层"""
    def __init__(self, data, src_mac, dst_mac):
        self.src_mac = src_mac
        self.dst_mac = dst_mac
        self.data = data

    def build(self):
        header = f"ETH[{self.src_mac}→{self.dst_mac}]"
        trailer = "[CRC]"
        return f"{header}|{self.data}|{trailer}"

# 构建完整的网络数据包
print("=== 构建HTTP请求的网络数据包 ===\n")

# 应用层
http = HTTPRequest("/index.html")
app_data = http.build()
print(f"应用层:\n{app_data}\n")

# 传输层
tcp = TCPSegment(app_data, src_port=52341, dst_port=80)
transport_data = tcp.build()
print(f"传输层:\n{transport_data}\n")

# 网络层
ip = IPPacket(transport_data, src_ip="192.168.1.100", dst_ip="93.184.216.34")
network_data = ip.build()
print(f"网络层:\n{network_data}\n")

# 数据链路层
eth = EthernetFrame(network_data,
                    src_mac="AA:BB:CC:DD:EE:FF",
                    dst_mac="11:22:33:44:55:66")
frame = eth.build()
print(f"数据链路层:\n{frame}\n")

print("物理层: 转换为电信号并发送到网络...")
```

---

## 🔄 层与层之间的通信

### 同层通信

```
发送方                                接收方
应用层 ←─────── 应用层协议 ─────────→ 应用层
传输层 ←─────── 传输层协议 ─────────→ 传输层
网络层 ←─────── 网络层协议 ─────────→ 网络层
链路层 ←─────── 链路层协议 ─────────→ 链路层
物理层 ←─────── 物理介质   ─────────→ 物理层

特点：
- 同层使用相同的协议
- 逻辑上直接通信
- 实际上通过下层传递
```

### 相邻层通信

```
上层 ←─ 提供服务 ─→ 下层

接口：
- 服务访问点（SAP）
- 原语（Primitive）

例子：
应用层调用传输层的socket API
传输层调用网络层的IP服务
```

---

## 💡 为什么是七层而不是五层或十层？

### 历史原因

```
OSI模型（理论）：
- 1970s-1980s ISO制定
- 设计完美，但过于复杂
- 推广不成功

TCP/IP模型（实践）：
- 1960s-1970s ARPANET发展而来
- 简单实用
- 成为事实标准
```

### 实际使用

```
现实中常用五层模型：

5. 应用层    - HTTP、DNS、FTP
4. 传输层    - TCP、UDP
3. 网络层    - IP
2. 数据链路层 - Ethernet、Wi-Fi
1. 物理层    - 电信号、光信号

原因：
- OSI的会话层和表示层功能不明确
- 很多协议跨越多层
- 简化模型更易理解
```

---

## 🎓 每层的设备

```
层次            设备
─────────────────────────
应用层          网关、代理服务器
传输层          -
网络层          路由器、三层交换机
数据链路层      交换机、网桥
物理层          集线器、中继器、网卡
```

---

## 🔗 相关概念

- [传输层](transport-layer.md) - TCP/UDP详解
- [网络层](network-layer.md) - IP和路由
- [应用层](application-layer.md) - HTTP等协议

---

**记住**：
1. 分层是为了降低复杂度
2. 每层有明确的职责
3. 下层为上层提供服务
4. 同层协议实现逻辑通信
5. OSI七层是理论，TCP/IP四层是实践
6. 数据在各层间封装和解封装
7. 实际工作中常用五层模型
8. 理解分层是理解网络的基础
