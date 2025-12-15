# Lambda 表达式

## 基本语法

```cpp
[捕获](参数) { 函数体 }
```

## 快速示例

```cpp
// 最简单
auto f = []() { return 42; };

// 带参数
auto add = [](int a, int b) { return a + b; };

// 用于算法
std::vector<int> v = {1, 2, 3, 4};
auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 2; });
```

## 捕获列表

```cpp
int x = 10;
int y = 20;

[]          // 不捕获
[x]         // 按值捕获 x
[&x]        // 按引用捕获 x
[=]         // 按值捕获所有
[&]         // 按引用捕获所有
[=, &x]     // 默认按值，x 按引用
[&, x]      // 默认按引用，x 按值

// 示例
auto f1 = [x]() { return x + 1; };     // x 是拷贝
auto f2 = [&x]() { x = 100; };         // x 是引用，可修改
auto f3 = [=]() { return x + y; };     // 捕获所有
```

## 常用场景

```cpp
// 1. 排序
std::vector<int> v = {3, 1, 4, 1, 5};
std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });  // 降序

// 2. 查找
auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 3; });

// 3. 计数
int count = std::count_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });

// 4. 遍历
std::for_each(v.begin(), v.end(), [](int x) { std::cout << x << " "; });
```

## 常见陷阱

```cpp
// ❌ 按值捕获，修改无效
int x = 10;
auto f = [x]() { x = 20; };  // 编译错误：x 是 const

// ✅ 加 mutable
auto f = [x]() mutable { x = 20; };  // 可以修改拷贝

// ❌ 悬空引用
auto make_lambda() {
    int x = 10;
    return [&x]() { return x; };  // 危险：x 已销毁
}

// ✅ 按值捕获
auto make_lambda() {
    int x = 10;
    return [x]() { return x; };  // 安全
}
```

## 要点

1. **默认按值捕获** `[=]`（安全）
2. **需要修改外部变量用 `[&]`**
3. **立即使用的小函数用 lambda**
4. **不要捕获引用后延迟调用**（悬空引用）
