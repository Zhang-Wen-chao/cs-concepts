# File Systems - 文件系统

> 文件系统：操作系统如何组织和管理磁盘上的数据？

## 🎯 什么是文件系统？

**文件系统 (File System)** 是操作系统用于管理文件和目录的机制，负责文件的存储、检索、命名和保护。

### 生活类比

```
文件系统 = 图书馆管理系统

- 文件 = 书籍
- 目录 = 书架/分类
- 磁盘块 = 书籍的页
- 文件名 = 书名
- 索引节点 = 图书卡片（记录位置、大小等）
```

---

## 📁 文件的概念

### 文件是什么？

**文件**：存储在外部存储设备上的相关信息的集合。

```
文件的属性：
- 名称：用户可读的标识
- 类型：文本、可执行、图片等
- 位置：磁盘上的位置
- 大小：当前大小、最大大小
- 保护：访问权限
- 时间：创建、修改、访问时间
- 所有者：用户ID
```

### 文件操作

```python
# Python中的文件操作
# 1. 创建/打开
f = open('test.txt', 'w')

# 2. 写入
f.write('Hello, World!\n')

# 3. 关闭
f.close()

# 4. 读取
with open('test.txt', 'r') as f:
    content = f.read()
    print(content)

# 5. 追加
with open('test.txt', 'a') as f:
    f.write('New line\n')

# 6. 定位
with open('test.txt', 'r') as f:
    f.seek(5)  # 跳到第5个字节
    data = f.read(5)  # 读取5个字节
```

### 文件类型

```
1. 普通文件
   - 文本文件：ASCII/UTF-8编码
   - 二进制文件：可执行文件、图片、视频

2. 目录文件
   - 包含文件名和指针

3. 特殊文件（Unix/Linux）
   - 字符设备文件：/dev/tty
   - 块设备文件：/dev/sda
   - 管道文件：用于进程间通信
   - 套接字文件：用于网络通信
```

### Linux文件类型查看

```bash
# ls -l 查看文件类型
ls -l /dev/

# 文件类型标识：
# -  普通文件
# d  目录
# l  符号链接
# c  字符设备
# b  块设备
# p  管道
# s  套接字

# 例子：
drwxr-xr-x  # 目录
-rw-r--r--  # 普通文件
lrwxrwxrwx  # 符号链接
```

---

## 🗂️ 目录结构

### 单级目录

```
根目录
├── file1
├── file2
├── file3
└── file4

缺点：
- 所有文件在同一目录
- 命名冲突
- 不利于组织
```

### 二级目录

```
根目录
├── user1/
│   ├── file1
│   └── file2
├── user2/
│   ├── file1
│   └── file3
└── user3/
    └── file1

优点：
- 不同用户有独立目录
- 避免命名冲突
```

### 树形目录（最常用）

```
/
├── home/
│   ├── user1/
│   │   ├── documents/
│   │   │   └── report.txt
│   │   └── pictures/
│   └── user2/
├── usr/
│   ├── bin/
│   └── lib/
└── etc/
    └── config.conf
```

### 路径

```python
# 绝对路径：从根目录开始
absolute_path = "/home/user1/documents/report.txt"

# 相对路径：从当前目录开始
relative_path = "../pictures/photo.jpg"

# 特殊路径
# .   当前目录
# ..  父目录
# ~   用户主目录
# /   根目录
```

---

## 📊 文件系统实现

### 文件的存储

#### 1. 连续分配

```
文件存储在连续的磁盘块中

例子：file.txt 从块3开始，长度5

磁盘块：
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
└───┴───┴───┴───┴───┴───┴───┴───┘
            └─file.txt─┘

优点：
✅ 简单
✅ 读取快（顺序访问）

缺点：
❌ 外部碎片
❌ 文件增长困难
```

```python
class ContiguousAllocation:
    def __init__(self, disk_size):
        self.disk = [None] * disk_size
        self.files = {}  # {filename: (start, length)}

    def allocate(self, filename, length):
        """连续分配"""
        # 找连续的空闲块
        start = None
        count = 0

        for i in range(len(self.disk)):
            if self.disk[i] is None:
                if start is None:
                    start = i
                count += 1
                if count == length:
                    # 找到足够空间
                    for j in range(start, start + length):
                        self.disk[j] = filename
                    self.files[filename] = (start, length)
                    return True
            else:
                start = None
                count = 0

        return False  # 分配失败

    def read(self, filename):
        """读取文件"""
        if filename not in self.files:
            return None
        start, length = self.files[filename]
        return self.disk[start:start + length]

# 使用
ca = ContiguousAllocation(20)
ca.allocate('file1.txt', 5)
ca.allocate('file2.txt', 3)
print(f"file1.txt: {ca.read('file1.txt')}")
```

#### 2. 链接分配

```
每个磁盘块包含指向下一个块的指针

文件 A:
块3 → 块7 → 块2 → 块10 → NULL

磁盘块：
┌────────────┐
│ 块3: 数据  │
│ next: 7    │
├────────────┤
│ 块7: 数据  │
│ next: 2    │
├────────────┤
│ 块2: 数据  │
│ next: 10   │
├────────────┤
│ 块10: 数据 │
│ next: NULL │
└────────────┘

优点：
✅ 无外部碎片
✅ 文件可以动态增长

缺点：
❌ 随机访问慢（需要顺序遍历）
❌ 指针占用空间
❌ 可靠性差（指针损坏）
```

```python
class LinkedAllocation:
    def __init__(self, disk_size, block_size=512):
        self.disk_size = disk_size
        self.block_size = block_size
        self.blocks = {}  # {block_num: {'data': ..., 'next': ...}}
        self.free_blocks = set(range(disk_size))
        self.files = {}  # {filename: first_block}

    def allocate_block(self):
        """分配一个空闲块"""
        if not self.free_blocks:
            return None
        block = self.free_blocks.pop()
        return block

    def write(self, filename, data):
        """写入文件"""
        # 将数据分成块
        chunks = [data[i:i+self.block_size]
                  for i in range(0, len(data), self.block_size)]

        first_block = None
        prev_block = None

        for chunk in chunks:
            block = self.allocate_block()
            if block is None:
                return False

            self.blocks[block] = {
                'data': chunk,
                'next': None
            }

            if first_block is None:
                first_block = block
                self.files[filename] = first_block

            if prev_block is not None:
                self.blocks[prev_block]['next'] = block

            prev_block = block

        return True

    def read(self, filename):
        """读取文件"""
        if filename not in self.files:
            return None

        data = []
        block = self.files[filename]

        while block is not None:
            data.append(self.blocks[block]['data'])
            block = self.blocks[block]['next']

        return ''.join(data)

# 使用
la = LinkedAllocation(100)
la.write('file1.txt', 'Hello, World! ' * 100)
content = la.read('file1.txt')
print(f"Read: {content[:50]}...")
```

#### 3. 索引分配（最常用）

```
用一个索引块存储所有块的指针

文件 A 的索引块：
┌──────────────┐
│ 块0: 指向4   │
│ 块1: 指向7   │
│ 块2: 指向2   │
│ 块3: 指向10  │
│ ...          │
└──────────────┘

优点：
✅ 支持随机访问
✅ 无外部碎片
✅ 动态增长

缺点：
❌ 索引块占用空间
❌ 小文件浪费（也需要索引块）
```

---

## 🔍 索引节点 (inode)

### Unix/Linux的inode结构

```
inode包含文件元数据：
- 文件大小
- 所有者
- 权限
- 时间戳
- 数据块指针

┌─────────────────────┐
│     inode #123      │
├─────────────────────┤
│ 文件大小: 10KB      │
│ 所有者: user1       │
│ 权限: rw-r--r--     │
│ 创建时间: ...       │
├─────────────────────┤
│ 直接指针 (12个)     │
│  → 块4              │
│  → 块7              │
│  → 块2              │
│  ...                │
├─────────────────────┤
│ 一级间接指针        │
│  → 索引块           │
│     → 块10          │
│     → 块15          │
├─────────────────────┤
│ 二级间接指针        │
├─────────────────────┤
│ 三级间接指针        │
└─────────────────────┘
```

### inode实现

```python
class INode:
    DIRECT_POINTERS = 12
    INDIRECT_POINTERS = 1
    POINTERS_PER_BLOCK = 128

    def __init__(self, inode_num):
        self.inode_num = inode_num
        self.size = 0
        self.owner = None
        self.permissions = 0o644
        self.created = None
        self.modified = None

        # 数据块指针
        self.direct = [None] * self.DIRECT_POINTERS
        self.indirect = None
        self.double_indirect = None
        self.triple_indirect = None

    def get_block_num(self, logical_block):
        """获取逻辑块对应的物理块号"""
        if logical_block < self.DIRECT_POINTERS:
            # 直接指针
            return self.direct[logical_block]

        logical_block -= self.DIRECT_POINTERS

        if logical_block < self.POINTERS_PER_BLOCK:
            # 一级间接
            # 实际需要读取间接块
            return None  # 简化

        # 二级、三级间接...
        return None

class FileSystem:
    def __init__(self):
        self.inodes = {}  # {inode_num: INode}
        self.blocks = {}  # {block_num: data}
        self.next_inode = 1
        self.free_blocks = set(range(1000))

    def create_file(self, filename, owner):
        """创建文件"""
        inode_num = self.next_inode
        self.next_inode += 1

        inode = INode(inode_num)
        inode.owner = owner
        self.inodes[inode_num] = inode

        return inode_num

    def write_block(self, inode_num, logical_block, data):
        """写入数据块"""
        if inode_num not in self.inodes:
            return False

        inode = self.inodes[inode_num]

        # 分配物理块
        if not self.free_blocks:
            return False

        physical_block = self.free_blocks.pop()

        # 存储数据
        self.blocks[physical_block] = data

        # 更新inode
        if logical_block < INode.DIRECT_POINTERS:
            inode.direct[logical_block] = physical_block

        return True

# 使用
fs = FileSystem()
inode = fs.create_file('test.txt', 'user1')
fs.write_block(inode, 0, b'Hello, World!')
print(f"Created file with inode {inode}")
```

### 查看inode信息

```bash
# 查看文件的inode号
ls -i file.txt
# 输出: 12345678 file.txt

# 查看inode详细信息
stat file.txt

# 输出：
#   File: file.txt
#   Size: 1234        Blocks: 8          IO Block: 4096
#   Device: 801h/2049d    Inode: 12345678    Links: 1
#   Access: (0644/-rw-r--r--)  Uid: ( 1000/   user)
#   ...
```

---

## 💽 磁盘空间管理

### 1. 位图 (Bitmap)

```
用位图记录空闲块

0 = 空闲
1 = 已使用

例子：
位图: 11001101...
      ││││││││
      ││││││└└─ 块0,1已用
      ││││└──── 块2空闲
      │││└───── 块3空闲
      ││└────── 块4已用
      │└─────── 块5已用
      └──────── 块6空闲

优点：
✅ 简单高效
✅ 易于找到n个连续空闲块
```

```python
class BitmapAllocator:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        # 用整数数组模拟位图
        self.bitmap = [0] * ((num_blocks + 7) // 8)

    def allocate(self):
        """分配一个空闲块"""
        for i in range(self.num_blocks):
            byte_idx = i // 8
            bit_idx = i % 8

            if not (self.bitmap[byte_idx] & (1 << bit_idx)):
                # 找到空闲块
                self.bitmap[byte_idx] |= (1 << bit_idx)
                return i

        return None  # 无空闲块

    def free(self, block_num):
        """释放块"""
        byte_idx = block_num // 8
        bit_idx = block_num % 8
        self.bitmap[byte_idx] &= ~(1 << bit_idx)

    def is_free(self, block_num):
        """检查块是否空闲"""
        byte_idx = block_num // 8
        bit_idx = block_num % 8
        return not (self.bitmap[byte_idx] & (1 << bit_idx))

# 使用
allocator = BitmapAllocator(100)
block1 = allocator.allocate()
block2 = allocator.allocate()
print(f"Allocated blocks: {block1}, {block2}")

allocator.free(block1)
print(f"Block {block1} is free: {allocator.is_free(block1)}")
```

### 2. 空闲链表

```
将空闲块用链表连接

空闲链表头 → 块5 → 块12 → 块7 → NULL

优点：
✅ 不需要额外空间（在空闲块内存储指针）

缺点：
❌ 难以找到连续空闲块
```

---

## 🗄️ 常见文件系统

### 1. FAT (File Allocation Table)

```
FAT表：记录文件的块链

FAT表：
┌────┬────┬────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │
├────┼────┼────┼────┼────┼────┤
│ -1 │ 3  │ -1 │ 4  │ -1 │ 2  │
└────┴────┴────┴────┴────┴────┘

说明：
- 块0: 文件结束
- 块1: 下一块是3
- 块2: 文件结束
- 块3: 下一块是4
- 块4: 文件结束
- 块5: 下一块是2

文件A: 1 → 3 → 4 → 结束

优点：
✅ 简单
✅ 兼容性好

缺点：
❌ FAT表可能很大
❌ 不支持权限
❌ 文件名长度限制
```

### 2. ext4 (Linux默认)

```
特性：
- 使用inode
- 支持日志
- 支持大文件（16TB）
- 支持大分区（1EB）
- 延迟分配
- 多块分配
```

### 3. NTFS (Windows)

```
特性：
- MFT (Master File Table)
- 支持权限（ACL）
- 文件压缩
- 文件加密
- 卷影复制
- 磁盘配额
```

### 4. ZFS

```
特性：
- Copy-on-Write
- 快照
- 数据完整性检查
- 自动修复
- 压缩
- 去重
```

---

## 🔒 文件权限

### Unix权限模型

```
权限位：rwxrwxrwx
        │││││││││
        │││││││└└─ 其他用户: 读/写/执行
        │││││└──── 组: 读/写/执行
        │││└────── 所有者: 读/写/执行

例子：
-rw-r--r--  1 user group 1234 Jan 1 12:00 file.txt
│││││││││
││││││││└─ 其他用户: 只读
│││││└──── 组: 只读
││└────── 所有者: 读写
│└─────── 类型: - (普通文件)
```

```python
# 查看权限
import os
import stat

path = 'test.txt'
st = os.stat(path)

# 获取权限
mode = st.st_mode

# 检查权限
is_readable = bool(mode & stat.S_IRUSR)
is_writable = bool(mode & stat.S_IWUSR)
is_executable = bool(mode & stat.S_IXUSR)

print(f"Readable: {is_readable}")
print(f"Writable: {is_writable}")
print(f"Executable: {is_executable}")

# 修改权限
os.chmod('test.txt', 0o644)  # rw-r--r--
```

### 访问控制列表 (ACL)

```
更细粒度的权限控制

例子：
user:alice:rw-    # alice可以读写
user:bob:r--      # bob只能读
group:dev:rwx     # dev组可以读写执行
mask::rw-         # 最大权限掩码
```

---

## 📝 文件系统实现示例

```python
class SimpleFileSystem:
    def __init__(self, disk_size=1000):
        self.disk_size = disk_size
        self.inodes = {}
        self.blocks = {}
        self.free_blocks = set(range(disk_size))
        self.next_inode = 1
        self.root = self.create_directory('/')

    def create_directory(self, name):
        """创建目录"""
        inode_num = self.next_inode
        self.next_inode += 1

        self.inodes[inode_num] = {
            'type': 'directory',
            'name': name,
            'children': {}
        }
        return inode_num

    def create_file(self, name, parent_inode):
        """创建文件"""
        inode_num = self.next_inode
        self.next_inode += 1

        self.inodes[inode_num] = {
            'type': 'file',
            'name': name,
            'size': 0,
            'blocks': []
        }

        # 添加到父目录
        self.inodes[parent_inode]['children'][name] = inode_num
        return inode_num

    def write_file(self, inode_num, data):
        """写入文件"""
        if inode_num not in self.inodes:
            return False

        inode = self.inodes[inode_num]
        if inode['type'] != 'file':
            return False

        # 分配块
        block = self.free_blocks.pop() if self.free_blocks else None
        if block is None:
            return False

        self.blocks[block] = data
        inode['blocks'].append(block)
        inode['size'] += len(data)
        return True

    def read_file(self, inode_num):
        """读取文件"""
        if inode_num not in self.inodes:
            return None

        inode = self.inodes[inode_num]
        if inode['type'] != 'file':
            return None

        data = []
        for block in inode['blocks']:
            data.append(self.blocks[block])

        return b''.join(data)

    def list_directory(self, inode_num):
        """列出目录内容"""
        if inode_num not in self.inodes:
            return None

        inode = self.inodes[inode_num]
        if inode['type'] != 'directory':
            return None

        return list(inode['children'].keys())

# 使用
fs = SimpleFileSystem()

# 创建文件
file_inode = fs.create_file('test.txt', fs.root)

# 写入数据
fs.write_file(file_inode, b'Hello, World!')

# 读取数据
content = fs.read_file(file_inode)
print(f"File content: {content}")

# 列出根目录
files = fs.list_directory(fs.root)
print(f"Files in root: {files}")
```

---

## 🔗 相关概念

- [进程与线程](processes-threads.md) - 文件描述符
- [内存管理](memory-management.md) - 文件缓存

---

**记住**：
1. 文件系统管理磁盘上的文件和目录
2. inode存储文件元数据
3. 三种分配方式：连续、链接、索引
4. 索引分配最常用（支持随机访问）
5. 位图管理空闲块
6. 目录是特殊的文件
7. Unix权限模型：所有者/组/其他
8. 不同文件系统有不同特性（FAT、ext4、NTFS）
