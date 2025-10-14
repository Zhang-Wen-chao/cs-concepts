# I/O Systems - I/O系统

> 操作系统如何与外部设备通信？如何高效地处理I/O操作？

## 🎯 什么是I/O系统？

**I/O系统**是操作系统中负责管理输入/输出设备的子系统，处理设备与内存之间的数据传输。

### 生活类比

```
I/O系统 = 邮局系统

- I/O设备 = 邮局网点
- 设备驱动 = 邮局工作人员
- I/O请求 = 邮件
- 缓冲区 = 邮箱
- DMA = 自动分拣机
```

---

## 🖥️ I/O设备分类

### 按数据传输方式

#### 1. 块设备 (Block Device)

```
特点：
- 数据以块为单位传输
- 可随机访问
- 有缓存

例子：
- 硬盘 (HDD)
- 固态硬盘 (SSD)
- U盘
- CD-ROM
```

#### 2. 字符设备 (Character Device)

```
特点：
- 数据以字符（字节）为单位传输
- 顺序访问
- 无缓存

例子：
- 键盘
- 鼠标
- 打印机
- 串口
```

### 按数据传输速率

```
低速设备：< 100 KB/s
- 键盘：10-20 bytes/s
- 鼠标：100 bytes/s

中速设备：100 KB/s - 10 MB/s
- 激光打印机：~100 KB/s
- USB 2.0：480 Mb/s

高速设备：> 10 MB/s
- 硬盘：100-200 MB/s
- SSD：500+ MB/s
- 网卡：1 Gb/s - 10 Gb/s
```

### Linux设备文件

```bash
# 查看设备文件
ls -l /dev/

# 块设备（b开头）
brw-rw---- 1 root disk 8, 0  /dev/sda    # 硬盘
brw-rw---- 1 root disk 8, 1  /dev/sda1   # 分区

# 字符设备（c开头）
crw-rw-rw- 1 root tty  5, 0  /dev/tty    # 终端
crw------- 1 root root 1, 3  /dev/null   # 空设备
crw-rw-rw- 1 root root 1, 8  /dev/random # 随机数
```

---

## 🔌 I/O控制方式

### 1. 轮询 (Polling)

```
CPU不断检查设备状态

while (设备未就绪):
    等待

读取数据
```

```python
def polling_read(device):
    """轮询方式读取"""
    while not device.is_ready():
        pass  # 忙等待（浪费CPU）

    return device.read()

# 示例
class Device:
    def __init__(self):
        self.ready = False
        self.data = None

    def is_ready(self):
        return self.ready

    def read(self):
        self.ready = False
        return self.data

# 使用
device = Device()
# data = polling_read(device)  # CPU一直在忙等待
```

**优点**：
- ✅ 简单
- ✅ 适合快速设备

**缺点**：
- ❌ 浪费CPU时间
- ❌ 不适合慢速设备

---

### 2. 中断驱动 (Interrupt-Driven)

```
设备完成后发送中断信号

CPU执行其他任务
    ↓
设备完成操作
    ↓
发送中断
    ↓
CPU暂停当前任务
    ↓
执行中断处理程序
    ↓
读取数据
    ↓
恢复之前的任务
```

```python
class InterruptController:
    def __init__(self):
        self.handlers = {}  # {中断号: 处理函数}

    def register_handler(self, irq, handler):
        """注册中断处理程序"""
        self.handlers[irq] = handler

    def trigger_interrupt(self, irq):
        """触发中断"""
        if irq in self.handlers:
            print(f"Interrupt {irq} triggered")
            self.handlers[irq]()

class InterruptDevice:
    def __init__(self, controller, irq):
        self.controller = controller
        self.irq = irq
        self.data = None

    def start_io(self):
        """启动I/O操作"""
        print("Device: Starting I/O...")
        # 模拟异步I/O
        import threading
        def complete():
            import time
            time.sleep(0.1)
            self.data = "Hello from device!"
            self.controller.trigger_interrupt(self.irq)

        threading.Thread(target=complete).start()

def io_complete_handler():
    """I/O完成中断处理"""
    print("Interrupt Handler: I/O complete!")
    # 读取数据、更新状态等

# 使用
ic = InterruptController()
ic.register_handler(5, io_complete_handler)

device = InterruptDevice(ic, irq=5)
device.start_io()

print("CPU: Doing other work...")
import time
time.sleep(0.2)
```

**优点**：
- ✅ CPU不浪费在等待上
- ✅ 响应及时

**缺点**：
- ❌ 中断开销
- ❌ 大量数据传输时中断频繁

---

### 3. DMA (Direct Memory Access)

```
设备直接访问内存，无需CPU参与数据传输

CPU发起DMA请求
    ↓
DMA控制器接管
    ↓
数据直接在设备和内存间传输
    ↓
传输完成后发送中断
    ↓
CPU处理完成事件
```

```python
class DMAController:
    def __init__(self, memory):
        self.memory = memory
        self.busy = False

    def transfer(self, device, mem_addr, size, direction='read'):
        """DMA传输"""
        if self.busy:
            return False

        self.busy = True
        print(f"DMA: Starting transfer of {size} bytes")

        # 模拟DMA传输（不占用CPU）
        import threading
        def do_transfer():
            import time
            time.sleep(0.05)  # 模拟传输时间

            if direction == 'read':
                # 从设备读取到内存
                data = device.read_data(size)
                self.memory[mem_addr:mem_addr+size] = data
                print(f"DMA: Read {size} bytes from device to memory[{mem_addr}]")
            else:
                # 从内存写入到设备
                data = self.memory[mem_addr:mem_addr+size]
                device.write_data(data)
                print(f"DMA: Write {size} bytes from memory[{mem_addr}] to device")

            self.busy = False
            # 发送中断通知CPU
            print("DMA: Transfer complete, sending interrupt")

        threading.Thread(target=do_transfer).start()
        return True

class DMADevice:
    def __init__(self):
        self.buffer = bytearray(1024)

    def read_data(self, size):
        return self.buffer[:size]

    def write_data(self, data):
        self.buffer[:len(data)] = data

# 使用
memory = bytearray(1024)
dma = DMAController(memory)
device = DMADevice()

# CPU发起DMA传输
dma.transfer(device, mem_addr=100, size=256, direction='read')

print("CPU: Free to do other work during DMA transfer")
import time
time.sleep(0.1)
```

**优点**：
- ✅ CPU完全解放
- ✅ 高速传输
- ✅ 适合大量数据

**缺点**：
- ❌ 需要专门的DMA硬件
- ❌ 可能与CPU竞争内存总线

---

## 🔧 设备驱动程序 (Device Driver)

### 驱动的作用

```
应用程序
    ↓
操作系统
    ↓
设备驱动 ← 屏蔽硬件细节
    ↓
硬件设备
```

### 驱动的结构

```python
class DeviceDriver:
    """设备驱动抽象类"""

    def __init__(self, device_name):
        self.device_name = device_name
        self.opened = False

    def open(self):
        """打开设备"""
        if self.opened:
            return False
        self.opened = True
        print(f"Driver: Opening device {self.device_name}")
        return True

    def close(self):
        """关闭设备"""
        if not self.opened:
            return False
        self.opened = False
        print(f"Driver: Closing device {self.device_name}")
        return True

    def read(self, buffer, size):
        """读取数据"""
        raise NotImplementedError

    def write(self, buffer, size):
        """写入数据"""
        raise NotImplementedError

    def ioctl(self, cmd, arg):
        """设备控制"""
        raise NotImplementedError

class DiskDriver(DeviceDriver):
    """磁盘驱动"""

    def __init__(self):
        super().__init__("disk0")
        self.data = bytearray(1024 * 1024)  # 1MB

    def read(self, buffer, size, offset=0):
        """读取磁盘"""
        if not self.opened:
            return -1

        print(f"DiskDriver: Reading {size} bytes from offset {offset}")
        buffer[:] = self.data[offset:offset+size]
        return size

    def write(self, buffer, size, offset=0):
        """写入磁盘"""
        if not self.opened:
            return -1

        print(f"DiskDriver: Writing {size} bytes to offset {offset}")
        self.data[offset:offset+size] = buffer[:size]
        return size

    def ioctl(self, cmd, arg):
        """设备控制命令"""
        if cmd == "GET_SIZE":
            return len(self.data)
        elif cmd == "FLUSH":
            print("DiskDriver: Flushing cache")
            return 0
        else:
            return -1

# 使用
driver = DiskDriver()
driver.open()

# 写入数据
write_buf = b"Hello, Disk!"
driver.write(write_buf, len(write_buf), offset=0)

# 读取数据
read_buf = bytearray(100)
driver.read(read_buf, 12, offset=0)
print(f"Read: {read_buf[:12]}")

# 控制命令
size = driver.ioctl("GET_SIZE", None)
print(f"Disk size: {size} bytes")

driver.close()
```

---

## 📦 I/O缓冲

### 为什么需要缓冲？

```
问题：
- 设备速度差异大
- 减少系统调用次数
- 提高数据传输效率

解决：缓冲区
```

### 缓冲策略

#### 1. 无缓冲

```
每次读写直接访问设备

应用 → 设备

优点：简单
缺点：效率低
```

#### 2. 单缓冲

```
操作系统提供一个缓冲区

写入：应用 → 缓冲区 → 设备
读取：设备 → 缓冲区 → 应用
```

```python
class SingleBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = bytearray(size)
        self.count = 0

    def write(self, data):
        """写入缓冲区"""
        if self.count + len(data) > self.size:
            self.flush()

        end = self.count + len(data)
        self.buffer[self.count:end] = data
        self.count = end

    def flush(self):
        """刷新缓冲区到设备"""
        if self.count > 0:
            print(f"Flushing {self.count} bytes to device")
            # 实际写入设备
            self.count = 0

    def read(self, size):
        """从缓冲区读取"""
        if self.count == 0:
            # 从设备填充缓冲区
            print("Filling buffer from device")
            self.count = self.size

        result = bytes(self.buffer[:min(size, self.count)])
        self.count -= len(result)
        return result

# 使用
buf = SingleBuffer(1024)
buf.write(b"Hello")
buf.write(b"World")
buf.flush()
```

#### 3. 双缓冲

```
使用两个缓冲区，一个填充，一个使用

缓冲区A：应用写入
缓冲区B：传输到设备

交换角色，提高效率
```

```python
class DoubleBuffer:
    def __init__(self, size):
        self.size = size
        self.buffers = [bytearray(size), bytearray(size)]
        self.active = 0  # 当前活动缓冲区
        self.count = 0

    def write(self, data):
        """写入活动缓冲区"""
        active_buf = self.buffers[self.active]

        if self.count + len(data) > self.size:
            # 切换缓冲区
            self.switch_buffer()

        end = self.count + len(data)
        active_buf[self.count:end] = data
        self.count = end

    def switch_buffer(self):
        """切换缓冲区"""
        if self.count > 0:
            # 在后台传输当前缓冲区
            print(f"Switching buffer, transferring {self.count} bytes")
            import threading
            def transfer():
                # 模拟传输
                import time
                time.sleep(0.01)
                print("Transfer complete")

            threading.Thread(target=transfer).start()

        # 切换到另一个缓冲区
        self.active = 1 - self.active
        self.count = 0

# 使用
dbuf = DoubleBuffer(1024)
for i in range(10):
    dbuf.write(f"Data chunk {i}\n".encode())
```

#### 4. 循环缓冲区

```
环形缓冲区，生产者-消费者模型

    写指针 →
┌─┬─┬─┬─┬─┬─┬─┬─┐
│ │X│X│X│ │ │ │ │
└─┴─┴─┴─┴─┴─┴─┴─┘
      ↑ 读指针
```

```python
class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = bytearray(size)
        self.read_pos = 0
        self.write_pos = 0
        self.count = 0

    def write(self, data):
        """写入数据"""
        for byte in data:
            if self.count >= self.size:
                # 缓冲区满
                return False

            self.buffer[self.write_pos] = byte
            self.write_pos = (self.write_pos + 1) % self.size
            self.count += 1

        return True

    def read(self, size):
        """读取数据"""
        result = bytearray()

        for _ in range(min(size, self.count)):
            result.append(self.buffer[self.read_pos])
            self.read_pos = (self.read_pos + 1) % self.size
            self.count -= 1

        return bytes(result)

    def available(self):
        """可读取的字节数"""
        return self.count

# 使用
cbuf = CircularBuffer(10)
cbuf.write(b"Hello")
print(f"Available: {cbuf.available()}")
data = cbuf.read(3)
print(f"Read: {data}")
print(f"Available: {cbuf.available()}")
```

---

## 💿 磁盘调度算法

### 磁盘结构

```
磁盘 = 多个盘片
盘片 = 多个磁道（同心圆）
磁道 = 多个扇区

访问时间 = 寻道时间 + 旋转延迟 + 传输时间

寻道时间：移动磁头到目标磁道
旋转延迟：等待扇区旋转到磁头下
传输时间：读写数据
```

### 1. FCFS (First-Come-First-Served)

```python
def fcfs_scheduling(requests, head_start):
    """先来先服务"""
    total_movement = 0
    current = head_start

    print(f"Starting at track {current}")

    for track in requests:
        movement = abs(track - current)
        total_movement += movement
        print(f"Move from {current} to {track} (distance: {movement})")
        current = track

    print(f"Total head movement: {total_movement}")
    return total_movement

# 使用
requests = [98, 183, 37, 122, 14, 124, 65, 67]
fcfs_scheduling(requests, head_start=53)
```

### 2. SSTF (Shortest Seek Time First)

```python
def sstf_scheduling(requests, head_start):
    """最短寻道时间优先"""
    total_movement = 0
    current = head_start
    remaining = requests.copy()

    print(f"Starting at track {current}")

    while remaining:
        # 找最近的请求
        closest = min(remaining, key=lambda x: abs(x - current))
        movement = abs(closest - current)
        total_movement += movement

        print(f"Move from {current} to {closest} (distance: {movement})")
        current = closest
        remaining.remove(closest)

    print(f"Total head movement: {total_movement}")
    return total_movement

# 使用
requests = [98, 183, 37, 122, 14, 124, 65, 67]
sstf_scheduling(requests, head_start=53)
```

### 3. SCAN (电梯算法)

```python
def scan_scheduling(requests, head_start, disk_size=200, direction='up'):
    """SCAN - 电梯算法"""
    total_movement = 0
    current = head_start
    remaining = sorted(requests)

    print(f"Starting at track {current}, direction: {direction}")

    if direction == 'up':
        # 先处理高于当前位置的请求
        upper = [r for r in remaining if r >= current]
        lower = [r for r in remaining if r < current]

        # 向上扫描
        for track in upper:
            movement = abs(track - current)
            total_movement += movement
            print(f"Move from {current} to {track} (distance: {movement})")
            current = track

        # 到达最高端
        if upper and current < disk_size - 1:
            movement = disk_size - 1 - current
            total_movement += movement
            print(f"Move to end: {disk_size - 1} (distance: {movement})")
            current = disk_size - 1

        # 向下扫描
        for track in reversed(lower):
            movement = abs(track - current)
            total_movement += movement
            print(f"Move from {current} to {track} (distance: {movement})")
            current = track

    print(f"Total head movement: {total_movement}")
    return total_movement

# 使用
requests = [98, 183, 37, 122, 14, 124, 65, 67]
scan_scheduling(requests, head_start=53, direction='up')
```

### 4. C-SCAN (循环扫描)

```python
def cscan_scheduling(requests, head_start, disk_size=200):
    """C-SCAN - 循环扫描"""
    total_movement = 0
    current = head_start
    remaining = sorted(requests)

    print(f"Starting at track {current}")

    # 向上的请求
    upper = [r for r in remaining if r >= current]
    lower = [r for r in remaining if r < current]

    # 向上扫描
    for track in upper:
        movement = abs(track - current)
        total_movement += movement
        print(f"Move from {current} to {track} (distance: {movement})")
        current = track

    # 到达最高端
    if upper:
        movement = disk_size - 1 - current
        total_movement += movement
        print(f"Move to end: {disk_size - 1} (distance: {movement})")
        current = disk_size - 1

    # 跳回起始端
    movement = current
    total_movement += movement
    print(f"Jump to start: 0 (distance: {movement})")
    current = 0

    # 处理低位请求
    for track in lower:
        movement = abs(track - current)
        total_movement += movement
        print(f"Move from {current} to {track} (distance: {movement})")
        current = track

    print(f"Total head movement: {total_movement}")
    return total_movement

# 使用
requests = [98, 183, 37, 122, 14, 124, 65, 67]
cscan_scheduling(requests, head_start=53)
```

### 调度算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **FCFS** | 公平、简单 | 效率低 | 负载轻 |
| **SSTF** | 吞吐量高 | 可能饥饿 | 负载重 |
| **SCAN** | 无饥饿 | 中间磁道有利 | 通用 |
| **C-SCAN** | 更公平 | 开销略大 | 负载重 |

---

## 🔗 相关概念

- [进程与线程](processes-threads.md) - I/O与进程调度
- [文件系统](file-systems.md) - 文件I/O

---

**记住**：
1. I/O设备分为块设备和字符设备
2. 三种I/O控制方式：轮询、中断、DMA
3. DMA适合大量数据传输，释放CPU
4. 设备驱动屏蔽硬件细节
5. 缓冲提高I/O效率
6. 磁盘调度算法优化寻道时间
7. SCAN和C-SCAN避免饥饿
8. 中断驱动比轮询高效
