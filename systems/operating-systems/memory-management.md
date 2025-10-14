# Memory Management - 内存管理

> 操作系统如何管理有限的内存？如何让每个程序都觉得自己拥有整个内存空间？

## 🎯 什么是内存管理？

**内存管理**是操作系统负责管理计算机内存资源的机制，包括内存的分配、回收、保护和地址转换。

### 为什么需要内存管理？

```
问题1: 内存不够用
- 程序需要10GB内存，但物理内存只有8GB

问题2: 多个程序共享内存
- 如何防止程序A访问程序B的内存？

问题3: 内存碎片
- 内存被分割成许多小块，无法满足大块需求

解决方案：虚拟内存 + 分页/分段
```

---

## 🧩 地址空间

### 物理地址 vs 虚拟地址

```
虚拟地址 (Virtual Address)
- 程序看到的地址
- 每个进程都有独立的虚拟地址空间
- 例子: 0x00000000 - 0xFFFFFFFF (4GB)

        ↓ MMU转换

物理地址 (Physical Address)
- 实际的内存地址
- 所有进程共享
- 例子: 0x12345000
```

### 进程的地址空间布局

```
高地址 0xFFFFFFFF
    ┌─────────────┐
    │   内核空间   │ ← 操作系统使用
    ├─────────────┤ 0xC0000000 (Linux)
    │     栈      │ ← 向下增长
    │      ↓      │
    │             │
    │      ↑      │
    │     堆      │ ← 向上增长
    ├─────────────┤
    │   BSS段     │ ← 未初始化的全局变量
    ├─────────────┤
    │   数据段    │ ← 已初始化的全局变量
    ├─────────────┤
    │   代码段    │ ← 程序代码（只读）
    └─────────────┘
低地址 0x00000000
```

### 查看进程内存布局

```c
#include <stdio.h>
#include <stdlib.h>

int global_init = 42;        // 数据段
int global_uninit;           // BSS段

int main() {
    int stack_var = 10;      // 栈
    int *heap_var = malloc(sizeof(int));  // 堆

    printf("代码段: %p\n", main);
    printf("数据段: %p\n", &global_init);
    printf("BSS段:  %p\n", &global_uninit);
    printf("堆:     %p\n", heap_var);
    printf("栈:     %p\n", &stack_var);

    free(heap_var);
    return 0;
}
```

---

## 🗺️ 内存管理方式

### 1. 连续内存分配

#### 固定分区

```
┌──────────┐
│  OS      │ 100KB
├──────────┤
│  分区1   │ 200KB
├──────────┤
│  分区2   │ 300KB
├──────────┤
│  分区3   │ 400KB
└──────────┘

缺点：
- 内部碎片：分区比程序大，浪费空间
- 不灵活：分区大小固定
```

#### 动态分区

```
初始状态:
┌──────────────────────┐
│    OS (100KB)        │
├──────────────────────┤
│    空闲 (900KB)      │
└──────────────────────┘

加载P1(200KB):
┌──────────────────────┐
│    OS (100KB)        │
├──────────────────────┤
│    P1 (200KB)        │
├──────────────────────┤
│    空闲 (700KB)      │
└──────────────────────┘

加载P2(300KB):
┌──────────────────────┐
│    OS (100KB)        │
├──────────────────────┤
│    P1 (200KB)        │
├──────────────────────┤
│    P2 (300KB)        │
├──────────────────────┤
│    空闲 (400KB)      │
└──────────────────────┘
```

**分配算法**：

```python
class MemoryBlock:
    def __init__(self, start, size, free=True):
        self.start = start
        self.size = size
        self.free = free

class MemoryManager:
    def __init__(self, total_size):
        self.blocks = [MemoryBlock(0, total_size, free=True)]

    def first_fit(self, size):
        """首次适应：找到第一个足够大的空闲块"""
        for block in self.blocks:
            if block.free and block.size >= size:
                return block
        return None

    def best_fit(self, size):
        """最佳适应：找到最小的足够大的空闲块"""
        best = None
        for block in self.blocks:
            if block.free and block.size >= size:
                if best is None or block.size < best.size:
                    best = block
        return best

    def worst_fit(self, size):
        """最坏适应：找到最大的空闲块"""
        worst = None
        for block in self.blocks:
            if block.free and block.size >= size:
                if worst is None or block.size > worst.size:
                    worst = block
        return worst

    def allocate(self, size, algorithm='first_fit'):
        """分配内存"""
        if algorithm == 'first_fit':
            block = self.first_fit(size)
        elif algorithm == 'best_fit':
            block = self.best_fit(size)
        else:
            block = self.worst_fit(size)

        if block is None:
            return None  # 分配失败

        # 分割块
        if block.size > size:
            new_block = MemoryBlock(block.start + size,
                                   block.size - size,
                                   free=True)
            block.size = size
            idx = self.blocks.index(block)
            self.blocks.insert(idx + 1, new_block)

        block.free = False
        return block.start

# 使用
mm = MemoryManager(1000)
addr1 = mm.allocate(200, 'first_fit')
addr2 = mm.allocate(300, 'best_fit')
print(f"P1 allocated at: {addr1}")
print(f"P2 allocated at: {addr2}")
```

#### 外部碎片

```
问题：释放后产生碎片

初始:
┌────────────┐
│ P1 (200KB) │
├────────────┤
│ P2 (100KB) │
├────────────┤
│ P3 (300KB) │
├────────────┤
│ P4 (100KB) │
└────────────┘

释放P2和P4:
┌────────────┐
│ P1 (200KB) │
├────────────┤
│ 空闲(100KB)│ ← 碎片
├────────────┤
│ P3 (300KB) │
├────────────┤
│ 空闲(100KB)│ ← 碎片
└────────────┘

总共200KB空闲，但无法分配200KB的连续空间！

解决：内存紧缩（压缩）
```

---

## 📄 分页 (Paging)

### 核心思想

- 将物理内存分成**固定大小的页框 (Page Frame)**
- 将虚拟内存分成**固定大小的页 (Page)**
- 页可以映射到任意页框，不需要连续

```
虚拟内存:              物理内存:
┌────────┐            ┌────────┐
│ 页0    │───────────→│ 页框2  │
├────────┤            ├────────┤
│ 页1    │─────┐      │ 页框1  │
├────────┤     │      ├────────┤
│ 页2    │─┐   └─────→│ 页框5  │
├────────┤ │          ├────────┤
│ 页3    │ └─────────→│ 页框0  │
└────────┘            ├────────┤
                      │ 页框3  │
                      ├────────┤
                      │ 页框4  │
                      └────────┘

不需要连续！解决了外部碎片问题
```

### 地址转换

```
虚拟地址 = (页号, 页内偏移)
物理地址 = (页框号, 页内偏移)

例子：
- 页大小 = 4KB = 4096 bytes = 2^12
- 虚拟地址 = 0x12345 (十进制: 74565)

计算：
页号 = 74565 / 4096 = 18
页内偏移 = 74565 % 4096 = 549

查页表得到：页18 → 页框5
物理地址 = 5 * 4096 + 549 = 21029
```

### 页表 (Page Table)

```python
class PageTable:
    def __init__(self, num_pages):
        # 页表项：页号 → (页框号, 有效位, 保护位)
        self.entries = [None] * num_pages

    def map_page(self, page_num, frame_num, valid=True, writable=True):
        """映射页到页框"""
        self.entries[page_num] = {
            'frame': frame_num,
            'valid': valid,
            'writable': writable,
            'accessed': False,
            'dirty': False
        }

    def translate(self, virtual_address, page_size=4096):
        """虚拟地址转换为物理地址"""
        page_num = virtual_address // page_size
        offset = virtual_address % page_size

        if page_num >= len(self.entries) or self.entries[page_num] is None:
            raise Exception("Segmentation Fault: Invalid page")

        entry = self.entries[page_num]

        if not entry['valid']:
            raise Exception("Page Fault: Page not in memory")

        frame_num = entry['frame']
        physical_address = frame_num * page_size + offset

        # 更新访问位
        entry['accessed'] = True

        return physical_address

# 使用
pt = PageTable(1024)  # 1024个页
pt.map_page(0, 5)     # 页0 → 页框5
pt.map_page(1, 10)    # 页1 → 页框10

# 访问虚拟地址
virtual_addr = 0x1234  # 虚拟地址
try:
    physical_addr = pt.translate(virtual_addr)
    print(f"Virtual: 0x{virtual_addr:x} → Physical: 0x{physical_addr:x}")
except Exception as e:
    print(f"Error: {e}")
```

### 页表项结构

```
┌──────────┬────┬────┬────┬────┬─────┐
│ 页框号   │有效│保护│访问│脏位│其他 │
└──────────┴────┴────┴────┴────┴─────┘
    20位     1   2    1    1    7

- 页框号：物理页框的编号
- 有效位 (Valid): 该页是否在内存中
- 保护位 (Protection): 读/写/执行权限
- 访问位 (Accessed): 是否被访问过（用于页面替换）
- 脏位 (Dirty): 是否被修改过（用于写回）
```

---

## 🏗️ 多级页表

### 为什么需要多级页表？

```
问题：页表太大

假设：
- 32位地址空间 = 4GB
- 页大小 = 4KB
- 需要的页数 = 4GB / 4KB = 1M 个页
- 每个页表项 = 4B
- 页表大小 = 1M × 4B = 4MB

每个进程都需要4MB的页表！

解决：多级页表
```

### 二级页表

```
虚拟地址:
┌──────────┬──────────┬────────┐
│ 页目录号 │  页表号  │  偏移  │
└──────────┴──────────┴────────┘
   10位       10位      12位

           ┌────────────┐
           │  页目录    │
           ├────────────┤
           │  entry 0   │───┐
           ├────────────┤   │
           │  entry 1   │   │
           └────────────┘   │
                            ↓
                    ┌────────────┐
                    │   页表     │
                    ├────────────┤
                    │  entry 0   │──→ 页框号
                    ├────────────┤
                    │  entry 1   │
                    └────────────┘

优势：
- 只需要为实际使用的页分配页表
- 大部分进程只用很少的内存
```

### 实现二级页表

```python
class TwoLevelPageTable:
    def __init__(self):
        self.page_directory = {}  # 页目录

    def map_page(self, virtual_page, physical_frame):
        """映射页到页框"""
        # 分解虚拟页号
        dir_index = virtual_page >> 10  # 高10位
        table_index = virtual_page & 0x3FF  # 低10位

        # 创建页表（如果不存在）
        if dir_index not in self.page_directory:
            self.page_directory[dir_index] = {}

        # 设置映射
        self.page_directory[dir_index][table_index] = physical_frame

    def translate(self, virtual_address, page_size=4096):
        """地址转换"""
        virtual_page = virtual_address // page_size
        offset = virtual_address % page_size

        dir_index = virtual_page >> 10
        table_index = virtual_page & 0x3FF

        if dir_index not in self.page_directory:
            raise Exception("Page Directory Entry not present")

        page_table = self.page_directory[dir_index]

        if table_index not in page_table:
            raise Exception("Page Table Entry not present")

        physical_frame = page_table[table_index]
        physical_address = physical_frame * page_size + offset

        return physical_address

# 使用
tlpt = TwoLevelPageTable()
tlpt.map_page(0, 5)      # 虚拟页0 → 物理页框5
tlpt.map_page(1024, 10)  # 虚拟页1024 → 物理页框10

addr = tlpt.translate(0x400000)  # 虚拟地址
print(f"Physical address: 0x{addr:x}")
```

---

## 💨 TLB (Translation Lookaside Buffer)

### 加速地址转换

```
问题：每次访问内存都要查页表，太慢！

解决：TLB = 页表的高速缓存

访问流程：
1. 查TLB (几个CPU周期)
   - 命中 → 直接得到物理地址
   - 未命中 → 查页表
2. 查页表 (几十个CPU周期)
3. 更新TLB

TLB命中率 ~98%，大大加速！
```

### TLB结构

```
┌──────────┬──────────┬────┬────┬────┐
│ 虚拟页号 │ 物理页框 │有效│保护│其他│
└──────────┴──────────┴────┴────┴────┘

例子：
VPN   PFN   Valid
 0  →  5     1
 1  →  10    1
 5  →  3     1
...
```

### 模拟TLB

```python
class TLB:
    def __init__(self, size=16):
        self.size = size
        self.cache = {}  # {virtual_page: physical_frame}
        self.hits = 0
        self.misses = 0

    def lookup(self, virtual_page):
        """查找TLB"""
        if virtual_page in self.cache:
            self.hits += 1
            return self.cache[virtual_page]
        else:
            self.misses += 1
            return None

    def update(self, virtual_page, physical_frame):
        """更新TLB"""
        if len(self.cache) >= self.size:
            # TLB满了，移除最旧的（简化版）
            self.cache.pop(next(iter(self.cache)))

        self.cache[virtual_page] = physical_frame

    def hit_rate(self):
        """计算命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

# 使用
tlb = TLB(size=4)
page_table = {0: 5, 1: 10, 2: 3, 3: 7}

def translate_with_tlb(virtual_page):
    # 先查TLB
    frame = tlb.lookup(virtual_page)
    if frame is not None:
        print(f"TLB Hit: Page {virtual_page} → Frame {frame}")
        return frame

    # TLB未命中，查页表
    print(f"TLB Miss: Page {virtual_page}")
    frame = page_table.get(virtual_page)
    if frame is None:
        raise Exception("Page Fault")

    # 更新TLB
    tlb.update(virtual_page, frame)
    return frame

# 访问序列
for page in [0, 1, 0, 2, 0, 1, 3]:
    translate_with_tlb(page)

print(f"TLB Hit Rate: {tlb.hit_rate():.2%}")
```

---

## 💾 虚拟内存 (Virtual Memory)

### 核心思想

```
虚拟内存 > 物理内存

程序可以使用比物理内存更大的地址空间！

实现：
- 不是所有页都在内存中
- 不在内存中的页保存在磁盘上（Swap）
- 需要时从磁盘加载到内存（缺页中断）
```

### 缺页中断 (Page Fault)

```
1. CPU访问虚拟地址
2. 查页表，发现有效位=0（页不在内存）
3. 触发缺页中断（Page Fault）
4. OS从磁盘加载页到内存
5. 更新页表
6. 重新执行访问指令
```

### 模拟缺页处理

```python
class VirtualMemory:
    def __init__(self, physical_frames, disk_pages):
        self.physical_memory = [None] * physical_frames  # 物理内存
        self.disk = list(range(disk_pages))  # 磁盘上的页
        self.page_table = {}  # 页表
        self.page_faults = 0
        self.next_frame = 0

    def access_page(self, page_num):
        """访问页"""
        # 检查页是否在内存
        if page_num in self.page_table:
            entry = self.page_table[page_num]
            if entry['valid']:
                print(f"Page {page_num} in memory (frame {entry['frame']})")
                entry['accessed'] = True
                return entry['frame']

        # 缺页中断
        print(f"Page Fault: Page {page_num} not in memory")
        self.page_faults += 1

        # 分配页框
        if self.next_frame < len(self.physical_memory):
            frame = self.next_frame
            self.next_frame += 1
        else:
            # 内存满了，需要替换页
            frame = self.select_victim()
            self.evict_page(frame)

        # 加载页从磁盘
        self.load_page_from_disk(page_num, frame)

        return frame

    def load_page_from_disk(self, page_num, frame):
        """从磁盘加载页"""
        print(f"Loading page {page_num} from disk to frame {frame}")
        self.physical_memory[frame] = page_num
        self.page_table[page_num] = {
            'frame': frame,
            'valid': True,
            'accessed': False,
            'dirty': False
        }

    def select_victim(self):
        """选择要替换的页（简单的循环策略）"""
        return self.next_frame % len(self.physical_memory)

    def evict_page(self, frame):
        """驱逐页"""
        page = self.physical_memory[frame]
        if page is not None:
            print(f"Evicting page {page} from frame {frame}")
            self.page_table[page]['valid'] = False

# 使用
vm = VirtualMemory(physical_frames=3, disk_pages=10)

# 访问页序列
for page in [0, 1, 2, 0, 3, 0, 4]:
    vm.access_page(page)
    print()

print(f"Total page faults: {vm.page_faults}")
```

---

## 🔄 页面置换算法

### 1. FIFO (First-In-First-Out)

```python
def fifo_replacement(pages, num_frames):
    """先进先出"""
    frames = []
    page_faults = 0

    for page in pages:
        if page not in frames:
            page_faults += 1
            if len(frames) < num_frames:
                frames.append(page)
            else:
                frames.pop(0)  # 移除最早的
                frames.append(page)
            print(f"Page {page}: Fault. Frames = {frames}")
        else:
            print(f"Page {page}: Hit.   Frames = {frames}")

    return page_faults

# 测试
pages = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
faults = fifo_replacement(pages, num_frames=3)
print(f"\nTotal page faults: {faults}")
```

### 2. LRU (Least Recently Used)

```python
def lru_replacement(pages, num_frames):
    """最近最少使用"""
    frames = []
    page_faults = 0
    recent = []  # 记录使用顺序

    for page in pages:
        if page not in frames:
            page_faults += 1
            if len(frames) < num_frames:
                frames.append(page)
            else:
                # 移除最久未使用的
                lru_page = recent[0]
                frames.remove(lru_page)
                frames.append(page)
                recent.remove(lru_page)
            print(f"Page {page}: Fault. Frames = {frames}")
        else:
            print(f"Page {page}: Hit.   Frames = {frames}")

        # 更新使用顺序
        if page in recent:
            recent.remove(page)
        recent.append(page)

    return page_faults

pages = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
faults = lru_replacement(pages, num_frames=3)
print(f"\nTotal page faults: {faults}")
```

### 3. Clock (时钟算法)

```python
class ClockReplacement:
    def __init__(self, num_frames):
        self.frames = [None] * num_frames
        self.use_bits = [0] * num_frames  # 使用位
        self.hand = 0  # 时钟指针
        self.page_faults = 0

    def access(self, page):
        """访问页"""
        # 检查是否在内存
        if page in self.frames:
            idx = self.frames.index(page)
            self.use_bits[idx] = 1  # 设置使用位
            print(f"Page {page}: Hit.   Frames = {self.frames}")
            return

        # 缺页
        self.page_faults += 1

        # 找空闲页框
        if None in self.frames:
            idx = self.frames.index(None)
            self.frames[idx] = page
            self.use_bits[idx] = 1
        else:
            # 使用时钟算法找victim
            while True:
                if self.use_bits[self.hand] == 0:
                    # 找到victim
                    self.frames[self.hand] = page
                    self.use_bits[self.hand] = 1
                    self.hand = (self.hand + 1) % len(self.frames)
                    break
                else:
                    # 给第二次机会
                    self.use_bits[self.hand] = 0
                    self.hand = (self.hand + 1) % len(self.frames)

        print(f"Page {page}: Fault. Frames = {self.frames}")

# 使用
clock = ClockReplacement(num_frames=3)
pages = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
for page in pages:
    clock.access(page)

print(f"\nTotal page faults: {clock.page_faults}")
```

### 算法对比

| 算法 | 优点 | 缺点 | 复杂度 |
|-----|------|------|--------|
| **FIFO** | 简单 | Belady异常 | O(1) |
| **LRU** | 性能好 | 实现开销大 | O(n) |
| **Clock** | 接近LRU | 近似算法 | O(1) |
| **Optimal** | 最优 | 无法实现(需预知未来) | - |

---

## 🔗 相关概念

- [进程与线程](processes-threads.md) - 进程的地址空间
- [进程同步与互斥](synchronization.md) - 共享内存
- [内存管理](../../fundamentals/programming-concepts/memory-management.md) - 编程中的内存管理

---

**记住**：
1. 虚拟内存让程序以为自己拥有整个地址空间
2. 分页解决了外部碎片问题
3. MMU负责地址转换
4. TLB加速地址转换
5. 缺页中断从磁盘加载页
6. LRU是最接近最优的实用算法
7. 多级页表节省页表空间
8. 每次内存访问都需要地址转换
