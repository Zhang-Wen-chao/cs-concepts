# 现代 C++ 学习路径

> 面向系统开发的 C++ 学习（C++11/14/17/20）

## 🎯 学习目标

**不是为了：**
- ❌ 算法竞赛
- ❌ 语言律师（过度关注语言细节）
- ❌ 展示语言特性

**而是为了：**
- ✅ 写出高性能、可维护的系统代码
- ✅ 理解工业界的 C++ 代码（如 bRPC）
- ✅ 掌握现代 C++ 的最佳实践

---

## 📖 学习流程

**每个主题的学习步骤**（以阶段 1 为例）：

```
1. 📄 看文档      01_modern_cpp_basics/00_modern_cpp_mindset.md
   理解概念，快速过一遍（每个文档 50-100 行，5-10 分钟）
   ↓
2. 💻 看代码      01_modern_cpp_basics/practices/00_old_vs_new_cpp.cpp
   看代码示例，理解如何实现（每个文件 40-60 行）
   ↓
3. 🚀 运行代码    g++ -std=c++17 00_old_vs_new_cpp.cpp -o demo && ./demo
   动手实践，验证理解
   ↓
4. 📝 记录小抄    cpp_cheatsheet.md
   提炼核心要点（一句话 + 最小代码示例），方便背诵
```

**关键**：
- 文档、代码、小抄**三者一一对应**
- 每个主题 15-20 分钟完成一轮
- 小抄随学习进度**动态更新**

---

## 📚 学习路径

### 阶段 1：现代 C++ 基础（2-3周）⬅️ 当前阶段

**核心思维转变：**
```cpp
// ❌ 旧 C++ (C++98)
int* p = new int(10);
delete p;  // 容易忘记，内存泄漏

// ✅ 现代 C++ (C++11+)
auto p = std::make_unique<int>(10);
// 自动释放，不会泄漏
```

**学习内容：**
- [x] 00_modern_cpp_mindset.md - 现代 C++ 思维（必读）✅ 2025-12-15
- [x] 01_raii.md - RAII 原则（最重要的概念）✅ 2025-12-17
- [x] 02_smart_pointers.md - 智能指针 ✅ 2025-12-17
- [x] 03_containers.md - 标准容器 ✅ 2025-12-17
- [x] 04_move_semantics.md - 移动语义 ✅ 2025-12-17
- [x] 05_lambda.md - Lambda 表达式 ✅ 2025-12-17
- [x] 06_templates_basics.md - 模板基础 ✅ 2025-12-17

**实践项目：**
- [x] 实现一个 RAII 风格的文件管理类（`practices/07_raii_file_manager.cpp`）
- [x] 用智能指针管理资源的小程序（`practices/08_smart_pointer_resource.cpp`）

---

### 阶段 2：并发编程（2周）

**核心：工业界必备的并发技能**

**学习内容：**
- [x] 01_thread_basics.md - 线程基础 ✅ 2025-12-18
- [x] 02_mutex_locks.md - 互斥锁与 RAII 锁管理 ✅ 2025-12-18
- [x] 03_condition_variable.md - 条件变量 ✅ 2025-12-18
- [x] 04_atomic.md - 原子操作 ✅ 2025-12-19
- [x] 05_async_future.md - 异步编程 ✅ 2025-12-19
- [x] 06_thread_pool.md - 线程池（重要！） ✅ 2025-12-19

**实践项目：**
- [x] 实现生产者-消费者模型（`practices/03_condition_variable.cpp`）
- [x] 实现一个线程池（`practices/06_thread_pool.cpp`）

---

### 阶段 3：系统编程（2-3周）

**核心：理解底层系统**

**学习内容：**
- [x] 01_network_io.md - 网络 I/O（socket、epoll）✅ 2025-12-20
- [x] 02_serialization.md - 序列化（Protobuf）✅ 2025-12-20
- [x] 03_memory_management.md - 内存管理优化 ✅ 2025-12-20
- [x] 04_performance.md - 性能优化 ✅ 2025-12-21

**实践项目：**
- [x] 实现一个简单的 HTTP 服务器 ✅ 2025-12-22
- [x] 实现一个对象池 ✅ 2025-12-22

---

### 阶段 4：综合项目（1-2周）

**实战项目：**
- [x] 简化版 RPC 框架 ✅ 2025-12-21
- [x] bRPC Hello World ✅ 2025-12-21
- [x] 推荐服务（把深度学习模型包装成服务） ✅ 2025-12-22

---

## 💡 学习原则

### 1. 永远使用现代 C++ 风格

```cpp
// ❌ 不要这样写
char* str = new char[100];
strcpy(str, "hello");
// ... 容易忘记 delete[]

// ✅ 这样写
std::string str = "hello";
// 自动管理内存
```

### 2. 遵循 RAII 原则

**RAII = Resource Acquisition Is Initialization**
- 资源在构造函数中获取
- 在析构函数中释放
- 永远不要手动 new/delete

### 3. 使用标准库

```cpp
// ❌ 不要手写数据结构
class MyVector { ... };  // 费时费力，还容易出错

// ✅ 使用标准库
std::vector<int> vec;    // 高效、安全、经过充分测试
```

### 4. 代码风格

遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)：
- 用 `snake_case` 命名变量和函数
- 用 `PascalCase` 命名类
- 用 `kConstantName` 命名常量
- 永远使用 `const` 当变量不变时

---

## 📖 推荐资源

### 必读书籍（按顺序）
1. **《C++ Primer》（第5版）** - 基础入门
2. **《Effective Modern C++》（Scott Meyers）** - 最佳实践
3. **《C++ Concurrency in Action》** - 并发编程

### 在线资源
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/) - C++ 之父的建议
- [cppreference.com](https://en.cppreference.com/) - 标准库参考
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) - 代码风格

---

## 🎯 学习进度跟踪

**开始时间：** 2025-11-27

**阶段 1 进度：** 7/7 ✅✅✅✅✅✅✅ 🎉 完成！

**阶段 2 进度：** 6/6 ✅✅✅✅✅✅ 🎉 完成！

**阶段 3 进度：** 4/4 ✅✅✅✅ 🎉 完成！

**阶段 4 进度：** 3/3 ✅✅✅ 🎉 完成！
