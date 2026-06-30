# 1.5 错误处理 — 问题版（自测用）

> 盖住答案，能说上来就算过关。

## Q1

C++ 有哪些错误处理方式？各在什么场景用？

<details>
<summary>答案</summary>

| 方式 | 场景 | 举例 |
|---|---|---|
| 返回值/错误码 | 轻度、可预见的错误 | 文件不存在、网络超时 |
| 异常（try/catch） | 重度、不可忽略的错误 | 内存不足、访问越界 |
| `assert` | 调试期，永远不该发生的事 | 指针不应为 null |
| `static_assert` | 编译期检查 | 模板参数不合法 |
| `std::optional` | 可能有值也可能没值 | 查询结果 |
| `std::variant` | 几种不同类型之一 | 成功返回 T，失败返回 Error |

</details>

## Q2

Python 里的 `try/except/finally` 和 C++ 的 `try/catch` 有什么关键区别？

<details>
<summary>答案</summary>

- Python：非强制。你觉得可能出错就 try，不 try 也不会崩
- C++：**异常必须处理**。不 catch 就 `std::terminate()` 直接崩

除此之外流程一样：`try` → 执行 → `throw` → 栈展开 → 匹配 `catch` → 执行 `catch` 块

</details>

## Q3

"栈展开" 是什么意思？RAII 和栈展开有什么关系？

<details>
<summary>答案</summary>

抛异常后，C++ 从 `throw` 所在的栈帧开始，逐层向上退出函数，每退一层就**自动调用所有栈上对象的析构函数**，直到找到匹配的 `catch`。

这就是 RAII 的牛逼之处：如果每个资源都被 RAII 封装了，异常发生时**自动释放**，不会有资源泄漏。

```cpp
void foo() {
    auto p = std::make_unique<int>(42);  // RAII
    auto f = std::ifstream("file.txt");   // RAII
    throw std::runtime_error("boom");     // 栈展开 → p 和 f 自动析构
}
```

</details>

## Q4

C++ 异常性能好吗？什么时候该用什么时候不该用？

<details>
<summary>答案</summary>

- **正常路径**：零开销（不抛就没事）
- **抛异常的路**：巨慢（栈展开 + 拷贝异常对象）

**推荐的 80/20 规则**：
- 👍 用异常：不可恢复的错误（内存不足、参数错到不能继续）
- 👎 不用异常：高频、可预期的小错误（比如 `std::map` 里找不到 key — 用 `find` 检查而不是 try）

</details>

## Q5

AddressSanitizer（ASan）是做什么的？怎么用？

<details>
<summary>答案</summary>

运行时工具，编译时插桩，检测内存错误：

```bash
g++ -fsanitize=address -g my_program.cpp -o my_program
./my_program
```

能检测：
- use-after-free
- 栈/堆/全局变量越界
- 双重释放

**不额外花钱、不改代码、跑一遍就抓到 bug。** 你写 C++ 练习时必须开这个编译。

</details>

## Q6

为什么 C++ 比 Python 更容易出现"看起来能跑但实际有 bug"的情况？

<details>
<summary>答案</summary>

Python 运行时帮你检查所有 — 内存越界、类型错误、空指针引用，Python 要么帮你挡了要么报得很清楚。

C++ 的**未定义行为**是核心原因：越界、UAF、double delete 等操作**可能正常工作，可能崩，可能悄无声息地把数据搞坏**。看起来能跑，下一次换个系统/编译器/输入就炸了。

所以 C++ 必须靠 **sanitizer + 静态分析 + 严格的代码习惯** 来弥补。

</details>
