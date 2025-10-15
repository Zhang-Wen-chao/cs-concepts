# Physical and Data Link Layers - 物理层与数据链路层

> 数据如何在物理介质上传输？如何保证传输的可靠性？

## 🎯 物理层 (Physical Layer)

### 物理层的作用

**物理层**负责在物理介质上传输原始的比特流。

```
功能：
- 定义物理特性（电压、频率）
- 定义接口标准（网线接口）
- 传输比特流（0和1）

类比：
物理层 = 公路
数据 = 汽车
```

---

## 🔌 传输介质

### 1. 双绞线 (Twisted Pair)

```
结构：两根铜线绞在一起

分类：
- UTP（非屏蔽双绞线）：常见的网线
- STP（屏蔽双绞线）：抗干扰更强

类别：
- Cat 5：100 Mbps
- Cat 5e：1 Gbps
- Cat 6：10 Gbps
- Cat 7：10 Gbps+

优点：✅ 便宜 ✅ 灵活 ✅ 易安装
缺点：❌ 距离短（100米） ❌ 易受干扰
```

### 2. 同轴电缆 (Coaxial Cable)

```
结构：中心导线 + 绝缘层 + 屏蔽层

特点：
- 抗干扰能力强
- 传输距离较远

用途：
- 有线电视
- 早期以太网（已淘汰）

优点：✅ 抗干扰 ✅ 带宽高
缺点：❌ 硬 ❌ 贵
```

### 3. 光纤 (Optical Fiber)

```
原理：光的全反射

分类：
- 单模光纤：激光，远距离（几十公里）
- 多模光纤：LED，短距离（几公里）

特点：
- 带宽极高（Tbps）
- 传输距离远
- 不受电磁干扰
- 轻便

优点：✅ 速度快 ✅ 距离远 ✅ 不受干扰
缺点：❌ 贵 ❌ 安装复杂
```

### 4. 无线传输

```
分类：
- 无线电波：Wi-Fi、蓝牙
- 微波：卫星通信
- 红外线：遥控器

Wi-Fi标准：
- 802.11b：11 Mbps
- 802.11g：54 Mbps
- 802.11n：600 Mbps
- 802.11ac：1-7 Gbps
- 802.11ax (Wi-Fi 6)：9.6 Gbps

优点：✅ 移动性 ✅ 灵活
缺点：❌ 易受干扰 ❌ 安全性差
```

---

## 📡 编码与调制

### 数字信号编码

```python
def nrz_encoding(bits):
    """非归零编码 (NRZ)
    高电平 = 1
    低电平 = 0
    """
    signal = []
    for bit in bits:
        signal.append(1 if bit == '1' else -1)
    return signal

def manchester_encoding(bits):
    """曼彻斯特编码
    上跳 = 0
    下跳 = 1

    优点：自带时钟信号
    """
    signal = []
    for bit in bits:
        if bit == '0':
            signal.extend([1, -1])  # 上跳
        else:
            signal.extend([-1, 1])  # 下跳
    return signal

# 示例
bits = "1011"
print(f"原始数据: {bits}")
print(f"NRZ编码: {nrz_encoding(bits)}")
print(f"曼彻斯特编码: {manchester_encoding(bits)}")
```

---

## 🔗 数据链路层 (Data Link Layer)

### 数据链路层的作用

```
功能：
1. 成帧（Framing）：将比特流组织成帧
2. 物理寻址：MAC地址
3. 流量控制：防止接收方过载
4. 错误控制：检测和纠正错误
5. 访问控制：多个设备共享信道

数据单位：帧（Frame）
```

---

## 📦 帧结构

### 以太网帧格式

```
┌─────────┬─────────┬──────┬────────┬──────┬─────┐
│前导码   │目标MAC  │源MAC │类型    │数据  │FCS  │
│8字节    │6字节    │6字节 │2字节   │46-   │4字节│
│         │         │      │        │1500  │     │
└─────────┴─────────┴──────┴────────┴──────┴─────┘

- 前导码：同步用
- 目标MAC：接收方地址
- 源MAC：发送方地址
- 类型：上层协议（0x0800=IP）
- 数据：有效载荷
- FCS：帧校验序列（CRC）
```

### Python模拟以太网帧

```python
import struct
import binascii

class EthernetFrame:
    def __init__(self, dst_mac, src_mac, ether_type, data):
        self.dst_mac = dst_mac
        self.src_mac = src_mac
        self.ether_type = ether_type
        self.data = data

    @staticmethod
    def mac_to_bytes(mac_str):
        """MAC地址转字节：'AA:BB:CC:DD:EE:FF' -> bytes"""
        return bytes.fromhex(mac_str.replace(':', ''))

    def calculate_crc(self, data):
        """计算CRC校验"""
        crc = binascii.crc32(data) & 0xFFFFFFFF
        return struct.pack('!I', crc)

    def build(self):
        """构建以太网帧"""
        # 前导码（简化，实际是硬件添加）
        preamble = b'\xAA' * 7 + b'\xAB'

        # 帧头
        dst = self.mac_to_bytes(self.dst_mac)
        src = self.mac_to_bytes(self.src_mac)
        eth_type = struct.pack('!H', self.ether_type)

        # 数据（填充到最小46字节）
        data = self.data.encode() if isinstance(self.data, str) else self.data
        if len(data) < 46:
            data = data + b'\x00' * (46 - len(data))

        # 计算FCS
        frame_without_fcs = dst + src + eth_type + data
        fcs = self.calculate_crc(frame_without_fcs)

        # 完整帧
        frame = preamble + frame_without_fcs + fcs
        return frame

    def parse(frame_bytes):
        """解析以太网帧"""
        # 跳过前导码
        frame = frame_bytes[8:]

        dst_mac = ':'.join(f'{b:02X}' for b in frame[0:6])
        src_mac = ':'.join(f'{b:02X}' for b in frame[6:12])
        ether_type = struct.unpack('!H', frame[12:14])[0]
        data = frame[14:-4]  # 去掉FCS
        fcs = frame[-4:]

        return {
            'dst_mac': dst_mac,
            'src_mac': src_mac,
            'ether_type': hex(ether_type),
            'data': data,
            'fcs': fcs.hex()
        }

# 使用
frame = EthernetFrame(
    dst_mac='11:22:33:44:55:66',
    src_mac='AA:BB:CC:DD:EE:FF',
    ether_type=0x0800,  # IP协议
    data='Hello, Ethernet!'
)

frame_bytes = frame.build()
print(f"帧大小: {len(frame_bytes)} 字节")
print(f"帧内容: {frame_bytes[:50].hex()}...")

# 解析
parsed = EthernetFrame.parse(frame_bytes)
print(f"\n解析结果:")
print(f"目标MAC: {parsed['dst_mac']}")
print(f"源MAC: {parsed['src_mac']}")
print(f"类型: {parsed['ether_type']}")
print(f"数据: {parsed['data']}")
```

---

## 🏷️ MAC地址

### MAC地址结构

```
MAC地址：48位（6字节）

格式：AA:BB:CC:DD:EE:FF

┌─────────────┬─────────────┐
│  OUI（24位）│ NIC（24位）│
│  厂商标识   │ 网卡编号   │
└─────────────┴─────────────┘

例子：
- 00:1A:2B:3C:4D:5E
  └─ 00:1A:2B 是厂商标识（Intel）
     └─ 3C:4D:5E 是网卡编号

特殊地址：
- FF:FF:FF:FF:FF:FF：广播地址
- 01:00:5E:XX:XX:XX：多播地址
```

### 查看MAC地址

```bash
# Linux/Mac
ifconfig
ip link show

# Windows
ipconfig /all

# Python查看MAC地址
import uuid

def get_mac_address():
    mac = uuid.getnode()
    mac_str = ':'.join(('%012X' % mac)[i:i+2] for i in range(0, 12, 2))
    return mac_str

print(f"本机MAC地址: {get_mac_address()}")
```

---

## 🔄 介质访问控制 (MAC)

### 信道分配

#### 1. 静态分配

```
信道划分：
- FDMA（频分多址）：不同频率
- TDMA（时分多址）：不同时间片
- CDMA（码分多址）：不同编码

优点：✅ 简单 ✅ 无冲突
缺点：❌ 效率低 ❌ 不灵活
```

#### 2. 动态分配

```
随机访问：
- ALOHA
- CSMA/CD（以太网）
- CSMA/CA（Wi-Fi）
```

### CSMA/CD（载波侦听多路访问/冲突检测）

```
以太网使用的协议

工作流程：
1. 载波侦听：检查信道是否空闲
2. 发送数据：如果空闲则发送
3. 冲突检测：边发送边监听
4. 冲突处理：检测到冲突立即停止
5. 随机退避：等待随机时间后重试
```

```python
import random
import time

class CSMACD:
    def __init__(self, max_attempts=16):
        self.max_attempts = max_attempts

    def sense_carrier(self):
        """载波侦听"""
        # 模拟检查信道
        return random.random() > 0.3  # 70%概率空闲

    def detect_collision(self):
        """冲突检测"""
        # 模拟检测冲突
        return random.random() < 0.2  # 20%概率冲突

    def send_frame(self, frame_data):
        """发送帧"""
        attempts = 0

        while attempts < self.max_attempts:
            print(f"尝试 {attempts + 1}:")

            # 1. 载波侦听
            if not self.sense_carrier():
                print("  信道忙，等待...")
                time.sleep(0.01)
                continue

            print("  信道空闲，开始发送...")

            # 2. 发送数据
            # 3. 冲突检测
            if self.detect_collision():
                print("  检测到冲突！")
                attempts += 1

                # 4. 随机退避
                k = min(attempts, 10)
                backoff = random.randint(0, 2**k - 1)
                print(f"  退避时间: {backoff} 时间片")
                time.sleep(backoff * 0.01)
            else:
                print("  发送成功！")
                return True

        print("达到最大重试次数，发送失败")
        return False

# 使用
csma = CSMACD()
csma.send_frame("Hello, Ethernet!")
```

### CSMA/CA（碰撞避免）

```
Wi-Fi使用的协议

与CSMA/CD的区别：
- 无线信号无法边发送边检测冲突
- 使用RTS/CTS握手避免冲突

工作流程：
1. 载波侦听
2. 发送RTS（请求发送）
3. 接收CTS（允许发送）
4. 发送数据
5. 接收ACK（确认）
```

---

## ❌ 错误检测

### 1. 奇偶校验 (Parity Check)

```python
def add_parity_bit(data, even=True):
    """添加奇偶校验位"""
    ones = data.count('1')

    if even:
        # 偶校验：使1的个数为偶数
        parity = '0' if ones % 2 == 0 else '1'
    else:
        # 奇校验：使1的个数为奇数
        parity = '1' if ones % 2 == 0 else '0'

    return data + parity

def check_parity(data_with_parity, even=True):
    """检查奇偶校验"""
    ones = data_with_parity.count('1')

    if even:
        return ones % 2 == 0
    else:
        return ones % 2 == 1

# 使用
data = "1011010"
data_with_parity = add_parity_bit(data, even=True)
print(f"原始数据: {data}")
print(f"添加校验位: {data_with_parity}")

# 检查（无错误）
is_valid = check_parity(data_with_parity, even=True)
print(f"校验结果: {'正确' if is_valid else '错误'}")

# 模拟错误
corrupted = data_with_parity[:-1] + ('0' if data_with_parity[-1] == '1' else '1')
is_valid = check_parity(corrupted, even=True)
print(f"损坏数据: {corrupted}")
print(f"校验结果: {'正确' if is_valid else '错误'}")
```

### 2. CRC（循环冗余校验）

```python
def crc_remainder(data, polynomial):
    """计算CRC余数"""
    # 转为整数
    data_int = int(data, 2)
    poly_int = int(polynomial, 2)

    # 多项式的位数
    poly_len = len(polynomial)

    # 添加0
    data_int <<= (poly_len - 1)

    # 除法
    while data_int.bit_length() >= poly_len:
        shift = data_int.bit_length() - poly_len
        data_int ^= (poly_int << shift)

    # 余数
    return format(data_int, f'0{poly_len-1}b')

def crc_encode(data, polynomial):
    """CRC编码"""
    remainder = crc_remainder(data, polynomial)
    return data + remainder

def crc_check(encoded_data, polynomial):
    """CRC检验"""
    remainder = crc_remainder(encoded_data, polynomial)
    return remainder == '0' * (len(polynomial) - 1)

# 使用（CRC-4）
data = "1101"
polynomial = "10011"  # x^4 + x + 1

encoded = crc_encode(data, polynomial)
print(f"原始数据: {data}")
print(f"多项式: {polynomial}")
print(f"编码后: {encoded}")

# 检验
is_valid = crc_check(encoded, polynomial)
print(f"校验结果: {'正确' if is_valid else '错误'}")

# 模拟错误
corrupted = encoded[:3] + ('0' if encoded[3] == '1' else '1') + encoded[4:]
is_valid = crc_check(corrupted, polynomial)
print(f"损坏数据: {corrupted}")
print(f"校验结果: {'正确' if is_valid else '错误'}")
```

---

## 🔀 交换机 (Switch)

### 交换机的作用

```
功能：
- 工作在数据链路层
- 根据MAC地址转发帧
- 学习MAC地址表
- 隔离冲突域

vs 集线器（Hub）：
集线器：物理层设备，广播到所有端口
交换机：链路层设备，只转发到目标端口
```

### MAC地址表学习

```python
class Switch:
    def __init__(self):
        self.mac_table = {}  # {MAC地址: 端口号}
        self.ports = {}      # {端口号: [连接的设备]}

    def learn_mac(self, mac_address, port):
        """学习MAC地址"""
        if mac_address not in self.mac_table:
            print(f"学习: {mac_address} 在端口 {port}")
            self.mac_table[mac_address] = port
        elif self.mac_table[mac_address] != port:
            print(f"更新: {mac_address} 从端口 {self.mac_table[mac_address]} 到 {port}")
            self.mac_table[mac_address] = port

    def forward_frame(self, src_mac, dst_mac, src_port):
        """转发帧"""
        # 学习源MAC地址
        self.learn_mac(src_mac, src_port)

        # 查找目标端口
        if dst_mac == 'FF:FF:FF:FF:FF:FF':
            # 广播
            print(f"广播帧: {src_mac} → {dst_mac}")
            print(f"  从端口 {src_port} 广播到所有其他端口")
            return list(range(1, 5))  # 假设4个端口

        if dst_mac in self.mac_table:
            # 已知目标
            dst_port = self.mac_table[dst_mac]
            print(f"转发帧: {src_mac} → {dst_mac}")
            print(f"  从端口 {src_port} 到端口 {dst_port}")
            return [dst_port]
        else:
            # 未知目标，泛洪
            print(f"未知目标: {dst_mac}，泛洪")
            print(f"  从端口 {src_port} 泛洪到所有其他端口")
            return [p for p in range(1, 5) if p != src_port]

    def show_mac_table(self):
        """显示MAC地址表"""
        print("\nMAC地址表:")
        print("─" * 40)
        for mac, port in self.mac_table.items():
            print(f"{mac:20} → 端口 {port}")
        print("─" * 40)

# 使用
switch = Switch()

# 模拟帧转发
print("=== 帧1 ===")
switch.forward_frame('AA:BB:CC:DD:EE:01', 'AA:BB:CC:DD:EE:02', src_port=1)

print("\n=== 帧2 ===")
switch.forward_frame('AA:BB:CC:DD:EE:02', 'AA:BB:CC:DD:EE:01', src_port=2)

print("\n=== 帧3 ===")
switch.forward_frame('AA:BB:CC:DD:EE:01', 'AA:BB:CC:DD:EE:02', src_port=1)

print("\n=== 帧4（广播）===")
switch.forward_frame('AA:BB:CC:DD:EE:01', 'FF:FF:FF:FF:FF:FF', src_port=1)

switch.show_mac_table()
```

---

## 🔗 相关概念

- [网络基础与分层模型](network-fundamentals.md) - 分层模型
- [网络层](network-layer.md) - IP协议

---

**记住**：
1. 物理层传输比特流
2. 常见传输介质：双绞线、光纤、无线
3. 数据链路层负责成帧、寻址、错误检测
4. MAC地址是物理地址（48位）
5. 以太网使用CSMA/CD避免冲突
6. CRC用于错误检测
7. 交换机工作在链路层，根据MAC地址转发
8. 交换机通过学习建立MAC地址表
