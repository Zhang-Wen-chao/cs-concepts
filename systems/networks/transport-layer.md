# Transport Layer - 传输层

> 如何保证数据可靠传输？TCP和UDP有什么区别？

## 🎯 传输层的作用

**传输层**提供端到端的数据传输服务，是应用程序之间通信的桥梁。

```
功能：
1. 端到端通信：进程到进程
2. 可靠传输：确保数据完整无误
3. 流量控制：防止发送方过快
4. 拥塞控制：防止网络过载
5. 复用与分用：多个应用共享网络

数据单位：段（Segment）

类比：
传输层 = 快递服务
端口号 = 门牌号
TCP = 保证送达的快递
UDP = 普通信件
```

---

## 🔌 端口号

### 端口号的作用

```
端口号：16位（0-65535）

作用：区分同一主机上的不同应用

地址 = IP地址 + 端口号
例子：192.168.1.100:8080

IP地址：标识主机
端口号：标识主机上的应用
```

### 端口号分类

```
知名端口（Well-Known Ports）：0-1023
- 21：FTP
- 22：SSH
- 23：Telnet
- 25：SMTP（邮件发送）
- 53：DNS
- 80：HTTP
- 443：HTTPS
- 3306：MySQL
- 6379：Redis

注册端口（Registered Ports）：1024-49151
- 应用程序注册使用

动态端口（Dynamic Ports）：49152-65535
- 客户端临时使用
```

### 查看端口

```bash
# 查看所有端口
netstat -an

# 查看监听端口
netstat -ln

# 查看特定端口
netstat -an | grep 8080

# Linux查看端口占用
lsof -i :8080

# 查看进程的端口
ps aux | grep nginx
```

---

## 📦 UDP协议

### UDP特点

```
UDP (User Datagram Protocol) - 用户数据报协议

特点：
✅ 无连接：不需要建立连接
✅ 快速：没有握手和确认
✅ 简单：头部开销小（8字节）
✅ 支持广播和多播
❌ 不可靠：不保证送达
❌ 无序：可能乱序到达
❌ 无流量控制
❌ 无拥塞控制

适用场景：
- 实时视频/语音（宁可丢包也不要延迟）
- DNS查询（快速响应）
- 网络游戏（实时性重要）
- 物联网（简单设备）
```

### UDP报文格式

```
UDP报头：8字节

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
┌───────────────────────────────┬───────────────────────────────┐
│          源端口               │          目标端口             │
├───────────────────────────────┼───────────────────────────────┤
│          长度                 │          校验和               │
└───────────────────────────────┴───────────────────────────────┘
│                           数据                                │
└───────────────────────────────────────────────────────────────┘
```

### Python实现UDP通信

```python
import socket

# UDP服务器
def udp_server(host='127.0.0.1', port=9999):
    """UDP服务器"""
    # 创建socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定地址
    sock.bind((host, port))
    print(f"UDP服务器启动在 {host}:{port}")

    while True:
        # 接收数据（最大1024字节）
        data, addr = sock.recvfrom(1024)
        print(f"收到来自 {addr} 的数据: {data.decode()}")

        # 发送响应
        response = f"Echo: {data.decode()}"
        sock.sendto(response.encode(), addr)

# UDP客户端
def udp_client(host='127.0.0.1', port=9999):
    """UDP客户端"""
    # 创建socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # 发送数据（不需要connect）
        message = "Hello, UDP!"
        sock.sendto(message.encode(), (host, port))
        print(f"发送: {message}")

        # 接收响应
        data, addr = sock.recvfrom(1024)
        print(f"收到响应: {data.decode()}")
    finally:
        sock.close()

# 使用
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        udp_server()
    else:
        udp_client()
```

---

## 🔒 TCP协议

### TCP特点

```
TCP (Transmission Control Protocol) - 传输控制协议

特点：
✅ 面向连接：三次握手建立连接
✅ 可靠传输：保证数据送达
✅ 有序：按序到达
✅ 流量控制：滑动窗口
✅ 拥塞控制：防止网络拥塞
✅ 全双工：双向通信
❌ 慢：握手、确认、重传
❌ 开销大：头部20字节+

适用场景：
- 网页浏览（HTTP）
- 文件传输（FTP）
- 邮件（SMTP/POP3）
- 远程登录（SSH）
- 几乎所有需要可靠性的场景
```

### TCP报文格式

```
TCP报头：20-60字节

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
┌───────────────────────────────┬───────────────────────────────┐
│          源端口               │          目标端口             │
├───────────────────────────────┴───────────────────────────────┤
│                        序列号（Sequence Number）              │
├───────────────────────────────────────────────────────────────┤
│                      确认号（Acknowledgment Number）          │
├───────┬───┬─┬─┬─┬─┬─┬─┬───────────────────────────────────────┤
│头长度│保留│U│A│P│R│S│F│             窗口大小                  │
├───────┴───┴─┴─┴─┴─┴─┴─┼───────────────────────────────────────┤
│          校验和           │          紧急指针                 │
├───────────────────────────┴───────────────────────────────────┤
│                        选项（可选）                           │
└───────────────────────────────────────────────────────────────┘

关键字段：
- 序列号：数据的字节编号
- 确认号：期望收到的下一个字节编号
- 标志位：
  - SYN：建立连接
  - ACK：确认
  - FIN：结束连接
  - RST：重置连接
  - PSH：推送数据
  - URG：紧急数据
- 窗口大小：流量控制
```

---

## 🤝 三次握手（连接建立）

### 三次握手过程

```
客户端                              服务器
  │                                   │
  │────── SYN (seq=x) ──────────────→│  第一次握手
  │                                   │  服务器收到SYN，知道客户端要建立连接
  │                                   │
  │←─── SYN-ACK (seq=y, ack=x+1) ───│  第二次握手
  │                                   │  客户端收到SYN-ACK，知道服务器同意
  │                                   │
  │────── ACK (ack=y+1) ────────────→│  第三次握手
  │                                   │  服务器收到ACK，连接建立
  │                                   │
  ├─────────── 连接建立 ─────────────┤
  │                                   │
```

### 为什么是三次？

```
两次握手的问题：
- 旧的重复SYN到达
- 服务器建立连接
- 客户端不知道（已超时）
- 浪费资源

三次握手的好处：
- 确认双方的发送和接收能力
- 防止旧连接请求
- 同步序列号
```

### 模拟三次握手

```python
import random

class TCPHandshake:
    def __init__(self):
        self.state = 'CLOSED'
        self.seq = random.randint(1000, 9999)
        self.ack = 0

    def client_connect(self):
        """客户端发起连接"""
        if self.state != 'CLOSED':
            print("错误：连接已存在")
            return None

        # 第一次握手：发送SYN
        print(f"客户端: 发送 SYN (seq={self.seq})")
        self.state = 'SYN_SENT'

        return {'type': 'SYN', 'seq': self.seq}

    def server_receive_syn(self, syn_packet):
        """服务器接收SYN"""
        if self.state != 'CLOSED':
            print("错误：服务器忙")
            return None

        # 第二次握手：发送SYN-ACK
        self.ack = syn_packet['seq'] + 1
        print(f"服务器: 收到 SYN (seq={syn_packet['seq']})")
        print(f"服务器: 发送 SYN-ACK (seq={self.seq}, ack={self.ack})")
        self.state = 'SYN_RCVD'

        return {'type': 'SYN-ACK', 'seq': self.seq, 'ack': self.ack}

    def client_receive_syn_ack(self, syn_ack_packet):
        """客户端接收SYN-ACK"""
        if self.state != 'SYN_SENT':
            print("错误：状态不正确")
            return None

        # 第三次握手：发送ACK
        self.ack = syn_ack_packet['seq'] + 1
        print(f"客户端: 收到 SYN-ACK (seq={syn_ack_packet['seq']}, ack={syn_ack_packet['ack']})")
        print(f"客户端: 发送 ACK (ack={self.ack})")
        self.state = 'ESTABLISHED'

        return {'type': 'ACK', 'ack': self.ack}

    def server_receive_ack(self, ack_packet):
        """服务器接收ACK"""
        if self.state != 'SYN_RCVD':
            print("错误：状态不正确")
            return False

        print(f"服务器: 收到 ACK (ack={ack_packet['ack']})")
        print("服务器: 连接建立！")
        self.state = 'ESTABLISHED'

        return True

# 模拟三次握手
print("=== TCP三次握手 ===\n")

client = TCPHandshake()
server = TCPHandshake()

# 第一次握手
syn = client.client_connect()
print()

# 第二次握手
syn_ack = server.server_receive_syn(syn)
print()

# 第三次握手
ack = client.client_receive_syn_ack(syn_ack)
print()

# 服务器确认
server.server_receive_ack(ack)

print(f"\n客户端状态: {client.state}")
print(f"服务器状态: {server.state}")
```

---

## 👋 四次挥手（连接终止）

### 四次挥手过程

```
客户端                              服务器
  │                                   │
  │────── FIN (seq=u) ──────────────→│  第一次挥手
  │                                   │  客户端：我要关闭了
  │                                   │
  │←───── ACK (ack=u+1) ─────────────│  第二次挥手
  │                                   │  服务器：我知道了，等我发完数据
  │                                   │
  │                                   │  服务器继续发送剩余数据...
  │                                   │
  │←───── FIN (seq=v) ───────────────│  第三次挥手
  │                                   │  服务器：我也要关闭了
  │                                   │
  │────── ACK (ack=v+1) ────────────→│  第四次挥手
  │                                   │  客户端：好的
  │                                   │
  ├────── TIME_WAIT (2MSL) ──────────┤
  │                                   │
  └────── CLOSED ─────────────────────┘
```

### 为什么是四次？

```
为什么不是三次？
- TCP是全双工的
- 每个方向都需要单独关闭
- 服务器收到FIN后：
  1. 先发ACK（我知道你要关闭）
  2. 发送剩余数据
  3. 再发FIN（我也关闭了）

TIME_WAIT为什么是2MSL？
- MSL（Maximum Segment Lifetime）：报文最大生存时间
- 2MSL：确保对方收到最后的ACK
- 防止旧连接的包干扰新连接
```

### 模拟四次挥手

```python
class TCPClose:
    def __init__(self):
        self.state = 'ESTABLISHED'

    def client_close(self):
        """客户端发起关闭"""
        if self.state != 'ESTABLISHED':
            print("错误：连接未建立")
            return None

        # 第一次挥手：发送FIN
        print("客户端: 发送 FIN")
        self.state = 'FIN_WAIT_1'
        return {'type': 'FIN'}

    def server_receive_fin(self):
        """服务器接收FIN"""
        if self.state != 'ESTABLISHED':
            print("错误：状态不正确")
            return None

        # 第二次挥手：发送ACK
        print("服务器: 收到 FIN")
        print("服务器: 发送 ACK")
        self.state = 'CLOSE_WAIT'
        return {'type': 'ACK'}

    def client_receive_ack(self):
        """客户端接收ACK"""
        if self.state != 'FIN_WAIT_1':
            print("错误：状态不正确")
            return False

        print("客户端: 收到 ACK")
        self.state = 'FIN_WAIT_2'
        return True

    def server_close(self):
        """服务器关闭"""
        if self.state != 'CLOSE_WAIT':
            print("错误：状态不正确")
            return None

        # 第三次挥手：发送FIN
        print("服务器: 发送剩余数据...")
        print("服务器: 发送 FIN")
        self.state = 'LAST_ACK'
        return {'type': 'FIN'}

    def client_receive_fin(self):
        """客户端接收FIN"""
        if self.state != 'FIN_WAIT_2':
            print("错误：状态不正确")
            return None

        # 第四次挥手：发送ACK
        print("客户端: 收到 FIN")
        print("客户端: 发送 ACK")
        self.state = 'TIME_WAIT'
        return {'type': 'ACK'}

    def server_receive_ack(self):
        """服务器接收ACK"""
        if self.state != 'LAST_ACK':
            print("错误：状态不正确")
            return False

        print("服务器: 收到 ACK")
        print("服务器: 连接关闭")
        self.state = 'CLOSED'
        return True

    def time_wait_timeout(self):
        """TIME_WAIT超时"""
        if self.state != 'TIME_WAIT':
            print("错误：状态不正确")
            return False

        print("客户端: TIME_WAIT 超时（2MSL）")
        print("客户端: 连接关闭")
        self.state = 'CLOSED'
        return True

# 模拟四次挥手
print("=== TCP四次挥手 ===\n")

client = TCPClose()
server = TCPClose()

# 第一次挥手
fin1 = client.client_close()
print()

# 第二次挥手
ack1 = server.server_receive_fin()
client.client_receive_ack()
print()

# 第三次挥手
fin2 = server.server_close()
print()

# 第四次挥手
ack2 = client.client_receive_fin()
server.server_receive_ack()
print()

# TIME_WAIT
client.time_wait_timeout()

print(f"\n客户端状态: {client.state}")
print(f"服务器状态: {server.state}")
```

---

## 🔄 可靠传输机制

### 1. 序列号和确认号

```
发送方                              接收方
  │                                   │
  │─── Seq=1000, Data="Hello" ──────→│
  │                                   │
  │←──── Ack=1005 ───────────────────│ 确认收到，期望下一个是1005
  │                                   │
  │─── Seq=1005, Data="World" ──────→│
  │                                   │
  │←──── Ack=1010 ───────────────────│
  │                                   │
```

### 2. 超时重传

```python
import time
import random

class ReliableTransfer:
    def __init__(self, timeout=1.0):
        self.timeout = timeout
        self.seq = 0

    def send_with_retransmission(self, data, max_retries=3):
        """带重传的发送"""
        retries = 0

        while retries < max_retries:
            print(f"尝试 {retries + 1}: 发送数据 (seq={self.seq})")

            # 模拟发送
            ack = self.simulate_send(data)

            if ack:
                print(f"收到ACK (ack={self.seq + len(data)})")
                self.seq += len(data)
                return True
            else:
                print(f"超时！等待 {self.timeout} 秒后重传...")
                time.sleep(self.timeout)
                retries += 1

        print("达到最大重试次数，发送失败")
        return False

    def simulate_send(self, data):
        """模拟发送（可能丢包）"""
        # 30%概率丢包
        if random.random() < 0.3:
            print("  [模拟] 数据包丢失")
            return False
        else:
            print("  [模拟] 数据包成功送达")
            return True

# 使用
transfer = ReliableTransfer(timeout=0.5)
transfer.send_with_retransmission("Hello", max_retries=5)
```

### 3. 滑动窗口

```python
class SlidingWindow:
    def __init__(self, window_size=4):
        self.window_size = window_size
        self.base = 0  # 窗口基序号
        self.next_seq = 0  # 下一个要发送的序号
        self.buffer = {}  # {seq: data}

    def can_send(self):
        """是否可以发送"""
        return self.next_seq < self.base + self.window_size

    def send(self, data):
        """发送数据"""
        if not self.can_send():
            print(f"窗口已满，等待确认...")
            return False

        seq = self.next_seq
        self.buffer[seq] = data
        print(f"发送: seq={seq}, data={data}")
        self.next_seq += 1
        return True

    def receive_ack(self, ack):
        """接收确认"""
        print(f"收到ACK: {ack}")

        if ack > self.base:
            # 滑动窗口
            print(f"窗口滑动: {self.base} → {ack}")
            for seq in range(self.base, ack):
                if seq in self.buffer:
                    del self.buffer[seq]
            self.base = ack

    def show_window(self):
        """显示窗口"""
        print(f"\n当前窗口: [{self.base}, {self.base + self.window_size})")
        print(f"已发送未确认: {list(self.buffer.keys())}")
        print(f"下一个发送: {self.next_seq}\n")

# 使用
window = SlidingWindow(window_size=4)

# 发送多个数据包
for i in range(6):
    window.send(f"Data{i}")
    window.show_window()

# 接收确认
window.receive_ack(2)
window.show_window()

# 继续发送
window.send("Data6")
window.send("Data7")
window.show_window()

# 接收更多确认
window.receive_ack(5)
window.show_window()
```

---

## 🔗 相关概念

- [网络层](network-layer.md) - IP协议
- [应用层](application-layer.md) - HTTP等应用协议

---

**记住**：
1. 传输层提供端到端通信
2. 端口号标识应用进程
3. UDP：快速、不可靠、无连接
4. TCP：可靠、有序、面向连接
5. 三次握手建立连接
6. 四次挥手终止连接
7. TCP通过序列号、确认、重传保证可靠性
8. 滑动窗口实现流量控制
