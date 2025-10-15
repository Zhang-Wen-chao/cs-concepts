# Network Layer - 网络层

> 如何让数据在不同网络间传输？路由器如何选择路径？

## 🎯 网络层的作用

**网络层**负责将数据包从源主机传送到目标主机，即使它们不在同一个网络中。

```
功能：
1. 路由选择：找到最佳路径
2. 逻辑寻址：IP地址
3. 分组转发：将数据包转发到下一跳
4. 拥塞控制：防止网络过载

数据单位：包（Packet）

类比：
网络层 = 邮局系统
IP地址 = 邮政编码
路由器 = 邮局中转站
```

---

## 📍 IP地址

### IPv4地址结构

```
IPv4地址：32位（4字节）

格式：点分十进制
例子：192.168.1.100

二进制：11000000.10101000.00000001.01100100

地址 = 网络号 + 主机号
```

### IP地址分类

```
A类：1.0.0.0    - 126.255.255.255
     第1字节是网络号，后3字节是主机号
     网络数：126个
     每个网络主机数：16,777,214个
     适用：大型网络

B类：128.0.0.0  - 191.255.255.255
     前2字节是网络号，后2字节是主机号
     网络数：16,384个
     每个网络主机数：65,534个
     适用：中型网络

C类：192.0.0.0  - 223.255.255.255
     前3字节是网络号，后1字节是主机号
     网络数：2,097,152个
     每个网络主机数：254个
     适用：小型网络

D类：224.0.0.0  - 239.255.255.255
     多播地址

E类：240.0.0.0  - 255.255.255.255
     保留
```

### 特殊IP地址

```python
def analyze_ip(ip):
    """分析IP地址"""
    parts = [int(x) for x in ip.split('.')]
    first = parts[0]

    # 私有地址
    if first == 10:
        return "A类私有地址"
    elif first == 172 and 16 <= parts[1] <= 31:
        return "B类私有地址"
    elif first == 192 and parts[1] == 168:
        return "C类私有地址"

    # 回环地址
    if first == 127:
        return "回环地址（localhost）"

    # 广播地址
    if all(p == 255 for p in parts):
        return "广播地址"

    # 分类
    if 1 <= first <= 126:
        return "A类公网地址"
    elif 128 <= first <= 191:
        return "B类公网地址"
    elif 192 <= first <= 223:
        return "C类公网地址"
    elif 224 <= first <= 239:
        return "D类多播地址"
    else:
        return "E类保留地址"

# 测试
test_ips = [
    "192.168.1.1",
    "10.0.0.1",
    "127.0.0.1",
    "8.8.8.8",
    "172.16.0.1"
]

for ip in test_ips:
    print(f"{ip:15} → {analyze_ip(ip)}")
```

---

## 🎭 子网掩码与子网划分

### 子网掩码

```
作用：区分网络号和主机号

格式：与IP地址相同的32位

例子：
IP地址：  192.168.1.100
子网掩码：255.255.255.0

二进制：
IP：      11000000.10101000.00000001.01100100
掩码：    11111111.11111111.11111111.00000000
                                      ↑
                                  网络号 | 主机号

网络号 = IP & 子网掩码
       = 192.168.1.0
```

### CIDR表示法

```
CIDR (Classless Inter-Domain Routing)

格式：IP地址/前缀长度

例子：
192.168.1.0/24
           └─ 24位是网络号

等价于：
192.168.1.0，子网掩码 255.255.255.0
```

### 子网划分

```python
class SubnetCalculator:
    def __init__(self, ip, prefix_len):
        self.ip = ip
        self.prefix_len = prefix_len

    def ip_to_int(self, ip):
        """IP地址转整数"""
        parts = [int(x) for x in ip.split('.')]
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]

    def int_to_ip(self, num):
        """整数转IP地址"""
        return f"{(num >> 24) & 0xFF}.{(num >> 16) & 0xFF}.{(num >> 8) & 0xFF}.{num & 0xFF}"

    def get_subnet_mask(self):
        """计算子网掩码"""
        mask = (0xFFFFFFFF << (32 - self.prefix_len)) & 0xFFFFFFFF
        return self.int_to_ip(mask)

    def get_network_address(self):
        """计算网络地址"""
        ip_int = self.ip_to_int(self.ip)
        mask_int = self.ip_to_int(self.get_subnet_mask())
        network = ip_int & mask_int
        return self.int_to_ip(network)

    def get_broadcast_address(self):
        """计算广播地址"""
        ip_int = self.ip_to_int(self.ip)
        mask_int = self.ip_to_int(self.get_subnet_mask())
        network = ip_int & mask_int
        host_bits = 32 - self.prefix_len
        broadcast = network | ((1 << host_bits) - 1)
        return self.int_to_ip(broadcast)

    def get_first_host(self):
        """第一个可用主机地址"""
        network = self.ip_to_int(self.get_network_address())
        return self.int_to_ip(network + 1)

    def get_last_host(self):
        """最后一个可用主机地址"""
        broadcast = self.ip_to_int(self.get_broadcast_address())
        return self.int_to_ip(broadcast - 1)

    def get_total_hosts(self):
        """总主机数"""
        host_bits = 32 - self.prefix_len
        return (2 ** host_bits) - 2  # 减去网络地址和广播地址

    def show_info(self):
        """显示子网信息"""
        print(f"IP地址:      {self.ip}/{self.prefix_len}")
        print(f"子网掩码:    {self.get_subnet_mask()}")
        print(f"网络地址:    {self.get_network_address()}")
        print(f"广播地址:    {self.get_broadcast_address()}")
        print(f"第一个主机:  {self.get_first_host()}")
        print(f"最后一个主机: {self.get_last_host()}")
        print(f"可用主机数:  {self.get_total_hosts()}")

# 使用
subnet = SubnetCalculator("192.168.1.100", 24)
subnet.show_info()

print("\n" + "="*50 + "\n")

# 子网划分示例：将192.168.1.0/24划分成4个子网
print("将 192.168.1.0/24 划分成 4 个子网 (需要2位，/26):\n")
for i in range(4):
    network = f"192.168.1.{i * 64}"
    subnet = SubnetCalculator(network, 26)
    print(f"子网 {i+1}:")
    subnet.show_info()
    print()
```

---

## 📦 IP数据包格式

### IPv4报头

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
┌───────┬───────┬───────────────┬───────────────────────────────┐
│版本   │头长度│服务类型       │          总长度                │
├───────┴───────┴───────────────┼───────────────┬───────────────┤
│           标识                │标志│      片偏移               │
├───────────────┬───────────────┼────┴───────────────────────────┤
│   生存时间    │    协议       │          头部校验和            │
├───────────────┴───────────────┴────────────────────────────────┤
│                        源IP地址                                │
├────────────────────────────────────────────────────────────────┤
│                       目标IP地址                               │
└────────────────────────────────────────────────────────────────┘

关键字段：
- 版本：4（IPv4）
- 头长度：通常20字节
- 总长度：整个IP包的长度
- TTL（生存时间）：防止环路，每经过一个路由器减1
- 协议：上层协议（6=TCP, 17=UDP）
- 源IP、目标IP：发送方和接收方地址
```

### Python构造IP包

```python
import struct
import socket

class IPv4Packet:
    def __init__(self, src_ip, dst_ip, protocol, data):
        self.version = 4
        self.ihl = 5  # 头长度（5个32位字）
        self.tos = 0
        self.total_len = 20 + len(data)  # 头部 + 数据
        self.identification = 54321
        self.flags = 0
        self.fragment_offset = 0
        self.ttl = 64
        self.protocol = protocol
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.data = data

    def checksum(self, data):
        """计算校验和"""
        if len(data) % 2 == 1:
            data += b'\x00'

        s = sum(struct.unpack('!%dH' % (len(data) // 2), data))
        s = (s >> 16) + (s & 0xFFFF)
        s += s >> 16
        return ~s & 0xFFFF

    def build(self):
        """构造IP包"""
        # 版本和头长度
        ver_ihl = (self.version << 4) + self.ihl

        # IP头（不含校验和）
        header = struct.pack(
            '!BBHHHBBH',
            ver_ihl,
            self.tos,
            self.total_len,
            self.identification,
            (self.flags << 13) + self.fragment_offset,
            self.ttl,
            self.protocol,
            0  # 校验和占位
        )

        # 添加源IP和目标IP
        src_ip_bytes = socket.inet_aton(self.src_ip)
        dst_ip_bytes = socket.inet_aton(self.dst_ip)
        header += src_ip_bytes + dst_ip_bytes

        # 计算校验和
        checksum = self.checksum(header)

        # 重新构造头部（含校验和）
        header = struct.pack(
            '!BBHHHBBH',
            ver_ihl,
            self.tos,
            self.total_len,
            self.identification,
            (self.flags << 13) + self.fragment_offset,
            self.ttl,
            self.protocol,
            checksum
        ) + src_ip_bytes + dst_ip_bytes

        return header + self.data

    @staticmethod
    def parse(packet):
        """解析IP包"""
        # 解析头部
        ver_ihl, tos, total_len, identification, flags_offset, ttl, protocol, checksum = \
            struct.unpack('!BBHHHBBH', packet[:12])

        version = ver_ihl >> 4
        ihl = ver_ihl & 0x0F

        src_ip = socket.inet_ntoa(packet[12:16])
        dst_ip = socket.inet_ntoa(packet[16:20])

        data = packet[20:]

        return {
            'version': version,
            'header_length': ihl * 4,
            'total_length': total_len,
            'ttl': ttl,
            'protocol': protocol,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'data': data
        }

# 使用
packet = IPv4Packet(
    src_ip='192.168.1.100',
    dst_ip='8.8.8.8',
    protocol=6,  # TCP
    data=b'Hello, IP!'
)

packet_bytes = packet.build()
print(f"IP包大小: {len(packet_bytes)} 字节")
print(f"IP包(hex): {packet_bytes[:20].hex()}")

# 解析
parsed = IPv4Packet.parse(packet_bytes)
print(f"\n解析结果:")
print(f"版本: IPv{parsed['version']}")
print(f"头长度: {parsed['header_length']} 字节")
print(f"总长度: {parsed['total_length']} 字节")
print(f"TTL: {parsed['ttl']}")
print(f"协议: {parsed['protocol']}")
print(f"源IP: {parsed['src_ip']}")
print(f"目标IP: {parsed['dst_ip']}")
print(f"数据: {parsed['data']}")
```

---

## 🗺️ 路由选择

### 路由表

```
路由表：记录如何到达目标网络

字段：
- 目标网络
- 子网掩码
- 下一跳（网关）
- 接口
- 跳数（度量）

例子：
目标网络          子网掩码        下一跳          接口
192.168.1.0      255.255.255.0   直连            eth0
10.0.0.0         255.0.0.0       192.168.1.1     eth0
0.0.0.0          0.0.0.0         192.168.1.1     eth0  (默认路由)
```

### 查看路由表

```bash
# Linux
route -n
ip route show

# Windows
route print

# Mac
netstat -nr
```

### 路由选择过程

```python
class RoutingTable:
    def __init__(self):
        self.routes = []

    def add_route(self, network, netmask, gateway, interface, metric=1):
        """添加路由条目"""
        self.routes.append({
            'network': network,
            'netmask': netmask,
            'gateway': gateway,
            'interface': interface,
            'metric': metric
        })

    def ip_to_int(self, ip):
        """IP转整数"""
        parts = [int(x) for x in ip.split('.')]
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]

    def match_route(self, dst_ip):
        """查找匹配的路由"""
        dst_int = self.ip_to_int(dst_ip)
        matches = []

        for route in self.routes:
            network_int = self.ip_to_int(route['network'])
            netmask_int = self.ip_to_int(route['netmask'])

            # 检查是否匹配
            if (dst_int & netmask_int) == (network_int & netmask_int):
                # 计算前缀长度（越长越具体）
                prefix_len = bin(netmask_int).count('1')
                matches.append((prefix_len, route))

        if not matches:
            return None

        # 选择最长前缀匹配（最具体的路由）
        matches.sort(reverse=True)
        return matches[0][1]

    def show_table(self):
        """显示路由表"""
        print("路由表:")
        print("─" * 80)
        print(f"{'目标网络':<18} {'子网掩码':<18} {'网关':<18} {'接口':<8} {'跳数'}")
        print("─" * 80)
        for route in self.routes:
            print(f"{route['network']:<18} {route['netmask']:<18} "
                  f"{route['gateway']:<18} {route['interface']:<8} {route['metric']}")
        print("─" * 80)

# 使用
rt = RoutingTable()

# 添加路由
rt.add_route('192.168.1.0', '255.255.255.0', '直连', 'eth0', 0)
rt.add_route('10.0.0.0', '255.0.0.0', '192.168.1.1', 'eth0', 1)
rt.add_route('172.16.0.0', '255.255.0.0', '192.168.1.1', 'eth0', 1)
rt.add_route('0.0.0.0', '0.0.0.0', '192.168.1.1', 'eth0', 1)  # 默认路由

rt.show_table()

# 查找路由
test_ips = ['192.168.1.50', '10.1.2.3', '8.8.8.8']
print("\n路由查找:")
for ip in test_ips:
    route = rt.match_route(ip)
    if route:
        print(f"{ip:<15} → 通过 {route['gateway']:<15} ({route['interface']})")
    else:
        print(f"{ip:<15} → 无匹配路由")
```

---

## 🔀 路由算法

### 1. 距离矢量算法（RIP）

```python
class DistanceVectorRouting:
    def __init__(self, router_id):
        self.router_id = router_id
        self.distance_table = {router_id: {router_id: 0}}  # {目标: {下一跳: 距离}}
        self.neighbors = {}  # {邻居: 距离}

    def add_neighbor(self, neighbor_id, cost):
        """添加邻居"""
        self.neighbors[neighbor_id] = cost
        self.distance_table[neighbor_id] = {neighbor_id: cost}

    def update_from_neighbor(self, neighbor_id, neighbor_table):
        """从邻居接收更新"""
        updated = False

        for dest, routes in neighbor_table.items():
            if dest == self.router_id:
                continue

            if dest not in self.distance_table:
                self.distance_table[dest] = {}

            # 计算通过该邻居到目标的距离
            min_cost = float('inf')
            for next_hop, cost in routes.items():
                new_cost = self.neighbors[neighbor_id] + cost
                if new_cost < min_cost:
                    min_cost = new_cost

            # 更新路由表
            if neighbor_id not in self.distance_table[dest] or \
               self.distance_table[dest][neighbor_id] != min_cost:
                self.distance_table[dest][neighbor_id] = min_cost
                updated = True

        return updated

    def get_best_route(self, dest):
        """获取到目标的最佳路由"""
        if dest not in self.distance_table:
            return None, float('inf')

        best_next_hop = None
        best_cost = float('inf')

        for next_hop, cost in self.distance_table[dest].items():
            if cost < best_cost:
                best_cost = cost
                best_next_hop = next_hop

        return best_next_hop, best_cost

    def show_table(self):
        """显示路由表"""
        print(f"\n路由器 {self.router_id} 的路由表:")
        print("─" * 40)
        print(f"{'目标':<10} {'下一跳':<10} {'距离'}")
        print("─" * 40)

        for dest in sorted(self.distance_table.keys()):
            next_hop, cost = self.get_best_route(dest)
            print(f"{dest:<10} {next_hop:<10} {cost}")
        print("─" * 40)

# 使用
# 网络拓扑：
#   A --- 1 --- B
#   |           |
#   2           3
#   |           |
#   C -----4--- D

router_a = DistanceVectorRouting('A')
router_a.add_neighbor('B', 1)
router_a.add_neighbor('C', 2)

router_b = DistanceVectorRouting('B')
router_b.add_neighbor('A', 1)
router_b.add_neighbor('D', 3)

router_c = DistanceVectorRouting('C')
router_c.add_neighbor('A', 2)
router_c.add_neighbor('D', 4)

router_d = DistanceVectorRouting('D')
router_d.add_neighbor('B', 3)
router_d.add_neighbor('C', 4)

# 模拟几轮更新
routers = [router_a, router_b, router_c, router_d]

for round_num in range(3):
    print(f"\n{'='*50}")
    print(f"第 {round_num + 1} 轮更新")
    print('='*50)

    for router in routers:
        for neighbor_id, cost in router.neighbors.items():
            # 找到邻居路由器
            neighbor = next(r for r in routers if r.router_id == neighbor_id)
            router.update_from_neighbor(neighbor_id, neighbor.distance_table)

    for router in routers:
        router.show_table()
```

### 2. 链路状态算法（OSPF）- Dijkstra算法

```python
import heapq

class LinkStateRouting:
    def __init__(self):
        self.graph = {}  # {节点: {邻居: 权重}}

    def add_link(self, node1, node2, cost):
        """添加链路"""
        if node1 not in self.graph:
            self.graph[node1] = {}
        if node2 not in self.graph:
            self.graph[node2] = {}

        self.graph[node1][node2] = cost
        self.graph[node2][node1] = cost

    def dijkstra(self, source):
        """Dijkstra最短路径算法"""
        distances = {node: float('inf') for node in self.graph}
        distances[source] = 0
        previous = {node: None for node in self.graph}
        pq = [(0, source)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            for neighbor, weight in self.graph[current].items():
                distance = current_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))

        return distances, previous

    def get_path(self, source, dest, previous):
        """获取路径"""
        path = []
        current = dest

        while current is not None:
            path.append(current)
            current = previous[current]

        return list(reversed(path))

    def show_routing_table(self, source):
        """显示路由表"""
        distances, previous = self.dijkstra(source)

        print(f"\n从 {source} 的路由表:")
        print("─" * 50)
        print(f"{'目标':<10} {'下一跳':<10} {'距离':<10} {'路径'}")
        print("─" * 50)

        for dest in sorted(self.graph.keys()):
            if dest == source:
                continue

            path = self.get_path(source, dest, previous)
            next_hop = path[1] if len(path) > 1 else '-'
            path_str = ' → '.join(path)

            print(f"{dest:<10} {next_hop:<10} {distances[dest]:<10} {path_str}")
        print("─" * 50)

# 使用
lsr = LinkStateRouting()

# 构建网络拓扑
#   A --- 1 --- B
#   |           |
#   2           3
#   |           |
#   C -----4--- D

lsr.add_link('A', 'B', 1)
lsr.add_link('A', 'C', 2)
lsr.add_link('B', 'D', 3)
lsr.add_link('C', 'D', 4)

# 显示每个节点的路由表
for node in ['A', 'B', 'C', 'D']:
    lsr.show_routing_table(node)
```

---

## 🔗 相关概念

- [物理层与数据链路层](physical-datalink.md) - 下层协议
- [传输层](transport-layer.md) - 上层协议

---

**记住**：
1. 网络层负责路由选择和逻辑寻址
2. IP地址是32位（IPv4）的逻辑地址
3. 子网掩码用于划分网络号和主机号
4. 路由表记录如何到达目标网络
5. 最长前缀匹配原则
6. 距离矢量算法（RIP）：告诉邻居整个路由表
7. 链路状态算法（OSPF）：告诉所有人邻居链路状态
8. TTL防止路由环路
