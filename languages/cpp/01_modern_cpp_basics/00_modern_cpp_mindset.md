# 现代 C++ 思维

## 核心思想

**让编译器帮你管理资源，不要手动管理**

```cpp
// ❌ 旧 C++：手动管理，容易出错
int* p = new int[1000];
// ... 100 行代码 ...
if (error) return;  // 内存泄漏
delete[] p;

// ✅ 现代 C++：自动管理
std::vector<int> v(1000);
// ... 100 行代码 ...
if (error) return;  // 自动释放
```

## 五大核心原则

### 1. RAII（资源绑定对象生命周期）

```cpp
std::vector<int> v(1000);      // 构造时分配
// ...
// 离开作用域自动释放，即使有异常
```

### 2. 智能指针（不手动 new/delete）

```cpp
// ❌ 旧
Widget* w = new Widget();
delete w;

// ✅ 新
auto w = std::make_unique<Widget>();
//       类型：std::unique_ptr<Widget>
//       () 调用 Widget 的构造函数

// 带参数的构造
auto w2 = std::make_unique<Widget>(42);        // 调用 Widget(int)
auto w3 = std::make_unique<Widget>(42, "hi");  // 调用 Widget(int, string)
```

### 3. 标准容器（不造轮子）

```cpp
// ❌ 旧
int* arr = new int[100];

// ✅ 新
std::vector<int> arr(100);
```

### 4. 移动语义（不拷贝大对象）

```cpp
// 自动移动，不拷贝
std::vector<int> create() {
    std::vector<int> v(1000000);
    return v;  // 自动移动，O(1)
}
```

### 5. const 正确性（明确意图）

```cpp
void read_only(const std::string& s);   // 不修改
void modify(std::string& s);             // 修改
void take(std::string&& s);              // 移动
```

## 关键对比

| 旧 C++ | 现代 C++ |
|--------|----------|
| new/delete | unique_ptr/vector |
| 裸指针 | 智能指针 |
| 手动管理 | RAII |
| 拷贝 | 移动 |
| char* | std::string |

## 记住

1. **永远不手动 new/delete**
2. **函数参数用 const&**
3. **默认用 vector**
4. **能 const 就 const**
5. **返回值让编译器优化**
