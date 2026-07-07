# 标准容器

## 容器选择（90%情况）

```cpp
vector          // 默认选择
unordered_map   // 键值查找
unordered_set   // 去重
```

## 时间复杂度

| 容器 | 查找 | 插入 | 删除 |
|-----|------|------|------|
| vector | O(n) | O(1)尾部 | O(1)尾部 |
| unordered_map | O(1) | O(1) | O(1) |
| unordered_set | O(1) | O(1) | O(1) |
| map | O(log n) | O(log n) | O(log n) |

## vector（默认选择）

```cpp
std::vector<int> v = {1, 2, 3};
v.push_back(4);          // 末尾添加
v.emplace_back(5);       // 原地构造（更快）
v[0] = 10;               // 随机访问
v.reserve(1000);         // 预留容量
```

## unordered_map（键值查找）

```cpp
std::unordered_map<std::string, int> m;
m["apple"] = 5;          // 插入/修改
int val = m["apple"];    // 访问
m.erase("apple");        // 删除
if (m.count("key")) {}   // 检查存在
```

## unordered_set（去重）

```cpp
std::unordered_set<int> s = {1, 2, 3};
s.insert(4);             // 插入
s.erase(2);              // 删除
if (s.count(3)) {}       // 检查存在
```

## 常用操作

```cpp
// 遍历（适用所有容器）
for (const auto& item : container) {
    // ...
}

// 大小
container.size();
container.empty();

// 清空
container.clear();
```

## 容器选择决策树

```
需要键值对？
  是 → unordered_map（O(1)查找）
  否 → 需要去重？
         是 → unordered_set
         否 → vector（默认）
```
