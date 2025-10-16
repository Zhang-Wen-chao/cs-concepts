# 存储层次结构 (Memory Hierarchy)

> 从寄存器到硬盘的完整存储体系

## 🎯 核心概念

**核心矛盾：我们需要又大又快又便宜的存储，但现实中这三者不可兼得**

### 解决方案：存储层次结构

```
  速度   容量   成本
   ↑      ↓      ↑
   │      │      │
寄存器  <100字节  极高  <1ns
   │      │      │
L1 Cache ~64KB   高    1-2ns
   │      │      │
L2 Cache ~512KB  较高  3-10ns
   │      │      │
L3 Cache ~8MB    中    10-20ns
   │      │      │
主内存   ~16GB   较低  50-100ns
   │      │      │
SSD     ~1TB     低    0.1ms
   │      │      │
HDD     ~4TB     很低  5-10ms
   ↓      ↑      ↓
```

**关键思想：利用程序的局部性原理，用小容量高速存储配合大容量低速存储**

---

## 1️⃣ 局部性原理

### 为什么存储层次有效？

程序访问数据不是随机的，而是有规律的！

### 时间局部性 (Temporal Locality)

**刚访问过的数据，很可能马上再次访问**

```c
// 例子1：循环变量
for (int i = 0; i < 1000; i++) {  // 'i' 被反复访问
    sum += array[i];              // 'sum' 被反复访问
}

// 例子2：热点函数
void frequently_called() {  // 这个函数的指令会一直在 Cache 中
    // ...
}
```

### 空间局部性 (Spatial Locality)

**访问了某个数据，很可能接下来访问它附近的数据**

```c
// 例子1：顺序访问数组
int sum = 0;
for (int i = 0; i < 1000; i++) {
    sum += array[i];  // array[0], array[1], array[2]... 连续访问
}

// 例子2：访问结构体
struct Point {
    int x, y, z;
};
Point p;
p.x = 1;  // 访问了 x
p.y = 2;  // 接着访问相邻的 y
p.z = 3;  // 再访问相邻的 z
```

### 局部性的实际数据

```
典型程序的内存访问：
  • 90% 的时间访问 10% 的数据
  • 循环占程序执行时间的 90%
  • 90% 的函数调用只涉及 10% 的函数

这就是为什么 Cache 有效！
```

---

## 2️⃣ Cache 基础

### 什么是 Cache？

**Cache = CPU 和内存之间的高速缓存**

```
CPU 需要数据时：
┌────────────────────────────────────┐
│ 1. 先查 Cache                      │
│    ├─ Hit (命中)   → 直接返回数据  │  快！
│    └─ Miss (未命中) → 去内存读取   │  慢...
│                                    │
│ 2. Miss 时从内存读取               │
│    • 读取整个 Cache line (64字节)  │
│    • 存入 Cache                    │
│    • 返回给 CPU                    │
└────────────────────────────────────┘
```

### Cache Line (缓存行)

**Cache 的基本存储单位，通常 64 字节**

```
为什么是 64 字节？
  → 利用空间局部性！

例子：
  int array[16];  // 16 个 int，每个 4 字节 = 64 字节

访问 array[0] 时：
  → Cache 加载整行 array[0-15]
  → 后续访问 array[1-15] 都命中！
```

### Cache 的三个核心问题

```
1. 块放置 (Block Placement)
   → 内存块应该放在 Cache 的哪里？

2. 块查找 (Block Identification)
   → 如何判断数据是否在 Cache？

3. 块替换 (Block Replacement)
   → Cache 满了，替换哪一块？
```

---

## 3️⃣ Cache 映射方式

### 内存地址分解

```
32位地址 = Tag + Index + Offset

┌─────────────┬─────────────┬────────────┐
│  Tag        │  Index      │  Offset    │
│  标签位     │  索引位     │  偏移位    │
└─────────────┴─────────────┴────────────┘
  用于匹配      选择Cache组    块内偏移

示例（64字节 Cache line）：
  Offset: 6 位 (2^6 = 64 字节)
  Index: 根据 Cache 大小决定
  Tag: 剩余位
```

### 1. 直接映射 (Direct Mapped)

**每个内存块只能放在 Cache 的唯一位置**

```
映射规则：
  Cache_Index = (Memory_Address / Block_Size) % Num_Cache_Blocks

示例（8个 Cache 块）：
┌──────────┬──────────────────┐
│ Cache 块 │  可以存放的内存块 │
├──────────┼──────────────────┤
│    0     │  0, 8, 16, 24... │
│    1     │  1, 9, 17, 25... │
│    2     │  2, 10, 18, 26...│
│    ...   │  ...             │
│    7     │  7, 15, 23, 31...│
└──────────┴──────────────────┘
```

**Cache 结构：**
```
┌───────┬──────┬──────────────────────────────┐
│ Valid │ Tag  │     Data (Cache Line)        │
├───────┼──────┼──────────────────────────────┤
│   1   │ 0x10 │ [64 bytes of data]           │  ← Index 0
│   1   │ 0x24 │ [64 bytes of data]           │  ← Index 1
│   0   │  -   │ [invalid]                    │  ← Index 2
│  ...  │ ...  │ ...                          │
└───────┴──────┴──────────────────────────────┘

查找步骤：
  1. 用 Index 选择 Cache 行
  2. 比较 Tag
  3. 检查 Valid 位
  4. Tag 匹配 && Valid = 1 → Hit
```

**优点：** 硬件简单，查找快
**缺点：** 冲突多（地址 0 和 8 会互相踢出）

### 2. 全相联 (Fully Associative)

**内存块可以放在 Cache 的任意位置**

```
┌───────┬──────┬────────────────┐
│ Valid │ Tag  │     Data       │
├───────┼──────┼────────────────┤
│   1   │ 0x1000│ [data]        │  ← 任意块
│   1   │ 0x2000│ [data]        │  ← 任意块
│   1   │ 0xABCD│ [data]        │  ← 任意块
│  ...  │ ...  │ ...            │
└───────┴──────┴────────────────┘

查找步骤：
  1. 同时比较所有 Cache 行的 Tag（并行比较）
  2. 任何一行 Tag 匹配 && Valid = 1 → Hit
```

**优点：** 冲突少，灵活
**缺点：** 硬件复杂（需要并行比较器），成本高

### 3. 组相联 (Set-Associative)

**折中方案：直接映射 + 全相联**

```
N-way Set-Associative:
  • Cache 分成多个组 (Set)
  • 每组有 N 个 Cache 行 (Way)
  • 先用 Index 选择组（直接映射）
  • 再在组内任意位置存放（全相联）

示例：2-way Set-Associative（每组2行）
┌─────────────┬──────────────────┐
│  Set 0      │ Way 0  │ Way 1  │
│  (Index=0)  │ [data] │ [data] │
├─────────────┼────────┼────────┤
│  Set 1      │ Way 0  │ Way 1  │
│  (Index=1)  │ [data] │ [data] │
├─────────────┼────────┼────────┤
│  Set 2      │ Way 0  │ Way 1  │
│  (Index=2)  │ [data] │ [data] │
└─────────────┴────────┴────────┘

查找步骤：
  1. 用 Index 选择 Set
  2. 在 Set 内并行比较 Tag
  3. 匹配成功 → Hit
```

**实际应用：**
- L1 Cache: 8-way
- L2 Cache: 8-way 或 16-way
- L3 Cache: 16-way 或 20-way

### 映射方式对比

```
┌──────────┬────────┬──────┬────────┐
│  映射方式│ 硬件   │ 命中率│ 实际应用│
├──────────┼────────┼──────┼────────┤
│ 直接映射 │ 简单   │ 低   │ 小 Cache│
│ 全相联   │ 复杂   │ 高   │ TLB    │
│ 组相联   │ 中等   │ 中高 │ 主流   │
└──────────┴────────┴──────┴────────┘
```

---

## 4️⃣ Cache 替换策略

### 当 Cache 满了，替换哪一块？

### 1. LRU (Least Recently Used)

**替换最久未使用的块**

```
示例（4个 Cache 块）：
┌────┬────┬────┬────┐
│ A  │ B  │ C  │ D  │  初始状态
└────┴────┴────┴────┘

访问 E（Miss）：
  → 需要替换一个块
  → 查看使用时间：A 最久没用
  → 替换 A

┌────┬────┬────┬────┐
│ E  │ B  │ C  │ D  │  A 被替换
└────┴────┴────┴────┘

访问 B（Hit）：
  → B 的使用时间更新为最新

访问 F（Miss）：
  → 现在 C 是最久未使用
  → 替换 C

┌────┬────┬────┬────┐
│ E  │ B  │ F  │ D  │
└────┴────┴────┴────┘
```

**实现：** 维护使用时间戳或访问队列
**优点：** 命中率高，符合局部性原理
**缺点：** 硬件开销大

### 2. FIFO (First In First Out)

**替换最早进入的块**

```
类似队列：
  → 新块从队尾进入
  → 替换时从队头踢出
```

**优点：** 实现简单
**缺点：** 可能踢出常用数据

### 3. Random (随机)

**随机选择一块替换**

**优点：** 硬件极简
**缺点：** 性能不稳定

### 4. LFU (Least Frequently Used)

**替换使用次数最少的块**

**优点：** 保留热点数据
**缺点：** 需要计数器，对访问模式变化适应慢

### 实际应用

```
L1/L2 Cache: 伪LRU (Pseudo-LRU)
  → 近似 LRU，硬件开销小

L3 Cache: 更复杂的策略
  → 考虑多核竞争
```

---

## 5️⃣ 多级 Cache

### 现代 CPU 的 Cache 层次

```
Intel Core i7 示例：
┌─────────────────────────────────────┐
│           CPU Core 0                │
│  ┌──────────┐      ┌──────────┐    │
│  │ L1 I-Cache│      │ L1 D-Cache│   │  32KB each
│  │   32 KB   │      │   32 KB   │   │  8-way
│  └─────┬─────┘      └─────┬─────┘   │  4 cycles
│        └────────┬──────────┘         │
│           ┌─────▼─────┐              │
│           │ L2 Cache  │              │  256 KB
│           │  256 KB   │              │  4-way
│           └─────┬─────┘              │  12 cycles
└─────────────────┼─────────────────────┘
                  │
       ┌──────────▼──────────┐
       │    L3 Cache (LLC)   │         8 MB
       │  Shared, 8-16 MB    │         16-way
       └──────────┬──────────┘         40 cycles
                  │
       ┌──────────▼──────────┐
       │    Main Memory      │         16 GB
       │      16 GB          │         ~200 cycles
       └─────────────────────┘
```

### L1 Cache 分离设计

**为什么 L1 分为指令 Cache 和数据 Cache？**

```
L1 I-Cache (Instruction):
  • 只存指令
  • 访问模式：顺序访问为主
  • 冲突少

L1 D-Cache (Data):
  • 只存数据
  • 访问模式：随机访问多
  • 需要更复杂的策略

优点：
  • 同时取指和访存（并行）
  • 避免指令和数据竞争 Cache
  • 针对性优化
```

### Inclusive vs Exclusive Cache

```
Inclusive (包含式):
  L3 ⊇ L2 ⊇ L1
  → L1 的数据一定在 L2 和 L3 中

  优点：一致性维护简单
  缺点：重复存储，浪费空间

Exclusive (互斥式):
  L1 ∩ L2 ∩ L3 = ∅
  → 各级 Cache 不重复

  优点：有效容量大
  缺点：一致性复杂

实际：
  • Intel: Inclusive（以前），现在混合
  • AMD: 部分 Exclusive
```

---

## 6️⃣ Cache 写策略

### 写操作的挑战

**问题：CPU 修改了 Cache 中的数据，如何同步到内存？**

### 1. 写直达 (Write-Through)

**每次写操作都同时写 Cache 和内存**

```
CPU 写数据：
  1. 写入 Cache ─┐
  2. 同时写内存 ─┘ 并行或串行

┌─────┐    ┌───────┐    ┌────────┐
│ CPU │───→│ Cache │───→│ Memory │
└─────┘    └───────┘    └────────┘
            写Cache      写内存
```

**优点：** 数据一致性好（Cache 和内存总是同步）
**缺点：** 写操作慢（每次都要访问内存）

**优化：写缓冲 (Write Buffer)**
```
CPU → Cache → Write Buffer → Memory
              ↓
           CPU 继续执行（不等内存写完）
```

### 2. 写回 (Write-Back)

**只写 Cache，脏数据在被替换时才写回内存**

```
1. CPU 写数据 → 只写 Cache
2. 标记为 Dirty（脏）
3. 继续使用
4. 被替换时 → 写回内存

┌─────────────────────────────────┐
│ Cache Line                      │
│ ┌───────┬──────┬──────┬───────┐│
│ │ Valid │ Dirty│ Tag  │ Data  ││
│ │   1   │  1   │ 0x10 │ [...]  ││
│ └───────┴──────┴──────┴───────┘│
└─────────────────────────────────┘
          ↑
       脏位标记
```

**优点：** 写操作快（不需要等内存）
**缺点：** 一致性复杂（需要 Dirty 位）

**实际应用：** 现代 CPU 主要使用 Write-Back

### 写缺失 (Write Miss) 处理

#### Write Allocate (写分配)
```
写 Miss 时：
  1. 先从内存加载数据到 Cache
  2. 再执行写操作

适用：Write-Back Cache
```

#### No-Write Allocate (写不分配)
```
写 Miss 时：
  直接写内存，不加载到 Cache

适用：Write-Through Cache
```

---

## 7️⃣ 虚拟内存与 TLB

### 虚拟内存

**每个进程有独立的虚拟地址空间**

```
进程 A:                    物理内存:
┌───────────────┐         ┌───────────────┐
│ 0x00000000    │         │               │
│   ...         │   ───→  │  Page Frame 5 │
│ 0x12340000    │   ───→  │  Page Frame 2 │
│   ...         │         │               │
│ 0xFFFFFFFF    │         │  Page Frame 1 │
└───────────────┘         └───────────────┘

进程 B:
┌───────────────┐
│ 0x00000000    │   ───→  │  Page Frame 7 │
│   ...         │         └───────────────┘
└───────────────┘

优点：
  • 进程隔离（安全）
  • 内存管理灵活
  • 支持比物理内存更大的地址空间
```

### 地址翻译

```
虚拟地址 → 物理地址

┌──────────────┬─────────────┐
│  VPN         │  Offset     │  虚拟地址
│ (虚拟页号)   │ (页内偏移)  │
└──────┬───────┴─────────────┘
       │
       │ 页表查询
       ▼
┌──────────────┬─────────────┐
│  PFN         │  Offset     │  物理地址
│ (物理帧号)   │ (页内偏移)  │
└──────────────┴─────────────┘

页大小通常: 4KB (2^12 = 4096)
  → Offset: 12 位
```

### TLB (Translation Lookaside Buffer)

**TLB = 地址翻译的 Cache**

```
问题：每次内存访问都要查页表 → 太慢！
解决：缓存最近的地址翻译结果

┌──────────────────────────────┐
│           TLB                │
│  ┌──────┬──────┬──────────┐ │
│  │ VPN  │ PFN  │ Flags    │ │
│  ├──────┼──────┼──────────┤ │
│  │ 0x10 │ 0x42 │ Valid,R,W│ │
│  │ 0x11 │ 0x55 │ Valid,R  │ │
│  │ ...  │ ...  │ ...      │ │
│  └──────┴──────┴──────────┘ │
└──────────────────────────────┘

访问流程：
  1. 查 TLB → Hit: 直接得到物理地址  (快！)
  2. TLB Miss → 查页表 → 更新 TLB   (慢...)
```

**TLB 特点：**
- 容量小（64-512 项）
- 全相联（高命中率）
- 极快（1 cycle）

### 地址翻译 + Cache 访问

```
完整流程：

虚拟地址
   │
   ▼
┌───────┐
│  TLB  │ ─ Hit  → 物理地址
└───┬───┘         │
    │ Miss        ▼
    ▼         ┌─────────┐
┌───────┐     │  Cache  │ ─ Hit  → 数据
│ 页表  │     └─────┬───┘
└───────┘           │ Miss
                    ▼
                ┌────────┐
                │ Memory │
                └────────┘

性能：
  TLB Hit + Cache Hit:  ~1-2 cycles    (最快)
  TLB Hit + Cache Miss: ~100 cycles    (还行)
  TLB Miss + Cache Miss: ~200+ cycles  (很慢)
```

---

## 8️⃣ 实际应用：如何写 Cache 友好的代码

### 1. 顺序访问数组

```c
// ❌ 不友好：按列访问（跳跃访问）
for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
        sum += matrix[i][j];  // 跨越多个 Cache line
    }
}

// ✅ 友好：按行访问（连续访问）
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        sum += matrix[i][j];  // 同一个 Cache line
    }
}

性能差距可达 10 倍！
```

### 2. 数据结构布局

```c
// ❌ Array of Structs (AoS) - 访问不连续
struct Particle {
    float x, y, z;     // 位置
    float vx, vy, vz;  // 速度
    float mass;        // 质量
};
Particle particles[1000];

// 只需要位置时，也会加载速度和质量到 Cache
for (int i = 0; i < 1000; i++) {
    process_position(particles[i].x, particles[i].y);
}

// ✅ Struct of Arrays (SoA) - 访问连续
struct Particles {
    float x[1000], y[1000], z[1000];
    float vx[1000], vy[1000], vz[1000];
    float mass[1000];
};

// 只加载需要的数据
for (int i = 0; i < 1000; i++) {
    process_position(particles.x[i], particles.y[i]);
}
```

### 3. 避免伪共享 (False Sharing)

```c
// ❌ 多核竞争同一 Cache line
struct Counter {
    int count_core0;  // 4 bytes
    int count_core1;  // 4 bytes - 在同一个 Cache line！
};

// Core 0 修改 count_core0 → 无效 Core 1 的 Cache
// Core 1 修改 count_core1 → 无效 Core 0 的 Cache
// 来回无效，性能下降！

// ✅ 填充到不同 Cache line
struct Counter {
    int count_core0;
    char padding[60];  // 填充到 64 字节
    int count_core1;
};
```

### 4. 循环分块 (Loop Tiling)

```c
// ❌ 大矩阵乘法 - Cache 不够
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}

// ✅ 分块计算 - 每块都在 Cache 中
int BLOCK = 64;
for (int ii = 0; ii < N; ii += BLOCK) {
    for (int jj = 0; jj < N; jj += BLOCK) {
        for (int kk = 0; kk < N; kk += BLOCK) {
            // 块内计算
            for (int i = ii; i < ii+BLOCK; i++) {
                for (int j = jj; j < jj+BLOCK; j++) {
                    for (int k = kk; k < kk+BLOCK; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
    }
}

性能提升：2-5 倍！
```

### 5. 数据对齐

```c
// ❌ 未对齐
struct Data {
    char a;      // 1 byte
    int b;       // 4 bytes - 可能未对齐
    short c;     // 2 bytes
};  // 可能跨越多个 Cache line

// ✅ 对齐
struct Data {
    int b;       // 4 bytes - 对齐到 4 字节边界
    short c;     // 2 bytes
    char a;      // 1 byte
    char pad;    // 填充
} __attribute__((aligned(64)));  // 对齐到 Cache line
```

---

## 9️⃣ 性能分析工具

### 查看 Cache 性能

```bash
# Linux perf 工具
perf stat -e cache-references,cache-misses ./program

Performance counter stats:
  1,234,567 cache-references
     12,345 cache-misses  # 1.0% of all cache refs

# 查看详细 Cache 事件
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./program
```

### Cache 友好度评估

```
命中率 = (Cache Hits) / (Cache Hits + Cache Misses)

优秀:  > 99%
良好:  95-99%
一般:  90-95%
差:    < 90%

注意：L1/L2/L3 命中率要分别看！
```

---

## 🔗 与其他概念的联系

### 与操作系统
- **虚拟内存** - OS 管理页表，硬件提供 TLB
- **进程切换** - 需要刷新 TLB 和 Cache
- **缺页异常** - TLB Miss → 页表查询 → 可能触发缺页

参考：`systems/operating-systems/memory-management.md`

### 与编译器优化
- **循环展开** - 减少跳转，提高指令 Cache 命中
- **数据预取** - 编译器插入预取指令
- **对齐优化** - 结构体字段重排

### 与并发编程
- **伪共享** - 多核 Cache 一致性问题
- **内存屏障** - 保证 Cache 一致性
- **锁的实现** - 利用 Cache line 独占

参考：`fundamentals/programming-concepts/concurrency-parallelism.md`

---

## 📚 深入学习

### 推荐资源
- *What Every Programmer Should Know About Memory* - Ulrich Drepper
- *Computer Architecture: A Quantitative Approach* - 第2章
- Intel® 64 and IA-32 Architectures Optimization Reference Manual

### 实践项目
- 实现 Cache 模拟器
- 分析程序的 Cache 性能
- 优化实际程序的 Cache 命中率

### 下一步
- [流水线与并行](./pipelining-parallelism.md) - 理解 CPU 并行执行
- [性能评估](./performance-evaluation.md) - 量化 Cache 对性能的影响
- [计算机组成](./computer-organization.md) - 理解硬件基础

---

**掌握存储层次，你就能写出快如闪电的代码！** ⚡
