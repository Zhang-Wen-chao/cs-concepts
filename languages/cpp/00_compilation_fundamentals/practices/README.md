# Stage 0 实战: 编译四步曲

## 目标
亲手跑一遍预处理→编译→汇编→链接，看到每一步的产物。

## 步骤

### 0. 确认编译器装好
```bash
clang++ --version
# Apple clang version 17.0.0 (clang-1700.4.4.1)
```

### 1. 写一个 hello.cpp
```bash
mkdir -p /tmp/cpp_stage0
cd /tmp/cpp_stage0

cat > hello.cpp << 'EOF'
#include <iostream>
#define GREETING "Hello"

int main() {
    std::cout << GREETING << ", C++!" << std::endl;
    return 0;
}
EOF
```

### 2. 跑四步并看产物
```bash
clang++ -E hello.cpp -o hello.i && head -20 hello.i
clang++ -S hello.cpp -o hello.s && cat hello.s
clang++ -c hello.cpp -o hello.o && file hello.o
clang++ hello.o -o hello && ./hello
```

**预期输出**：
- `hello.i` 开头是 iostream 库的展开内容
- `hello.s` 是汇编代码（看到 `main:` 标签、`mov`、`call` 指令）
- `hello.o` 是 Mach-O 64-bit 目标文件
- `./hello` 输出 `Hello, C++!`

### 3. 制造并理解两类错误

**编译错误**（缺分号）：
```bash
echo 'int main() { return 0' > bad1.cpp   # 缺分号
clang++ -c bad1.cpp                       # 编译错误
```

**链接错误**（只声明不定义）：
```cpp
// link_err.cpp
int add(int a, int b);   // 只声明
int main() { return add(1,2); }
```
```bash
clang++ link_err.cpp -o link_err
# linker error: undefined reference to 'add'
```

### 4. 看符号表
```bash
# 看 .o 里有什么符号
nm hello.o
nm -C hello.o      # -C demangle C++ 名字
```

你应该能看到：
```
0000000000000000 T _main
```

`T` = 在 text 段（代码段）定义的全局符号。

### 5. 多文件项目（手写编译）
创建三个文件：

`math.h`:
```cpp
#pragma once
int add(int a, int b);
int sub(int a, int b);
```

`math.cpp`:
```cpp
#include "math.h"
int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
```

`main.cpp`:
```cpp
#include <iostream>
#include "math.h"

int main() {
    std::cout << "3+5=" << add(3,5) << "\n";
    std::cout << "10-4=" << sub(10,4) << "\n";
    return 0;
}
```

手动编译：
```bash
clang++ -c math.cpp -o math.o
clang++ -c main.cpp -o main.o
clang++ math.o main.o -o my_program
./my_program
```

## 完成后告诉我
- 四步产物都看到了吗？
- 链接错误的报错能讲清楚吗？
- 多文件项目跑通了吗？
- `nm` 输出里看到了什么符号？

任何一步卡住都贴报错出来。
