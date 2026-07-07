# 序列化

> 将数据结构转换为字节流，用于存储或网络传输

## 核心概念

**序列化 = 对象 → 字节流**
**反序列化 = 字节流 → 对象**

```
内存中的对象          字节流           网络/文件
┌─────────┐       ┌──────┐        ┌─────┐
│ User    │       │010101│        │     │
│ name: A │  →    │101010│   →    │     │
│ age: 25 │       │110011│        │     │
└─────────┘       └──────┘        └─────┘
  序列化           传输/存储        反序列化
```

## 为什么需要序列化？

**问题**：C++ 对象只存在内存中，无法直接：
- 网络传输（RPC、HTTP API）
- 保存到文件
- 跨进程通信

**解决**：序列化为通用格式（JSON、Protobuf、二进制）

## 常见序列化格式

**1. JSON（文本格式）**：
- ✅ 人类可读
- ✅ 跨语言
- ❌ 体积大
- ❌ 解析慢

**2. Protobuf（二进制格式）**：
- ✅ 体积小（比 JSON 小 3-10 倍）
- ✅ 解析快（比 JSON 快 20-100 倍）
- ✅ 强类型、向后兼容
- ❌ 不可读

**3. 自定义二进制**：
- ✅ 最快、最小
- ❌ 不兼容、难维护

## 简单 JSON 序列化（手写）

```cpp
#include <iostream>
#include <string>
#include <sstream>

struct User {
    std::string name;
    int age;

    // 序列化为 JSON
    std::string to_json() const {
        std::ostringstream oss;
        oss << "{"
            << "\"name\":\"" << name << "\","
            << "\"age\":" << age
            << "}";
        return oss.str();
    }
};

// 使用
User user{"Alice", 25};
std::string json = user.to_json();
// {"name":"Alice","age":25}
```

## JSON 库（nlohmann/json）

**安装**：
```bash
# 单头文件库，下载即用
wget https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
```

**基本用法**：
```cpp
#include "json.hpp"
using json = nlohmann::json;

struct User {
    std::string name;
    int age;
};

// 序列化
User user{"Alice", 25};
json j;
j["name"] = user.name;
j["age"] = user.age;
std::string s = j.dump();  // {"age":25,"name":"Alice"}

// 反序列化
json j2 = json::parse(s);
User user2;
user2.name = j2["name"];
user2.age = j2["age"];
```

**自动序列化（宏）**：
```cpp
struct User {
    std::string name;
    int age;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(User, name, age)
};

// 使用
User user{"Alice", 25};
json j = user;              // 自动序列化
User user2 = j.get<User>(); // 自动反序列化
```

## Protobuf

**优势**：工业界标准（gRPC、bRPC 都用它）

**定义消息（.proto 文件）**：
```protobuf
syntax = "proto3";

message User {
    string name = 1;
    int32 age = 2;
}
```

**生成 C++ 代码**：
```bash
protoc --cpp_out=. user.proto
# 生成 user.pb.h 和 user.pb.cc
```

**使用**：
```cpp
#include "user.pb.h"

// 序列化
User user;
user.set_name("Alice");
user.set_age(25);

std::string data;
user.SerializeToString(&data);  // 二进制数据

// 反序列化
User user2;
user2.ParseFromString(data);
std::cout << user2.name() << ", " << user2.age();
```

## 嵌套结构

**Protobuf**：
```protobuf
message Address {
    string city = 1;
    string street = 2;
}

message User {
    string name = 1;
    int32 age = 2;
    Address address = 3;  // 嵌套
}
```

**使用**：
```cpp
User user;
user.set_name("Alice");
user.set_age(25);
user.mutable_address()->set_city("Beijing");
user.mutable_address()->set_street("Main St");
```

## 列表/数组

**Protobuf**：
```protobuf
message UserList {
    repeated User users = 1;  // 列表
}
```

**使用**：
```cpp
UserList list;

User* u1 = list.add_users();
u1->set_name("Alice");

User* u2 = list.add_users();
u2->set_name("Bob");

// 遍历
for (const auto& user : list.users()) {
    std::cout << user.name();
}
```

## 网络传输

**问题**：序列化后的数据如何发送？

**解决**：先发送长度，再发送数据

```cpp
// 发送
std::string data;
user.SerializeToString(&data);

uint32_t len = data.size();
send(sock, &len, sizeof(len), 0);    // 1. 发送长度
send(sock, data.data(), len, 0);     // 2. 发送数据

// 接收
uint32_t len;
recv(sock, &len, sizeof(len), 0);    // 1. 接收长度

std::vector<char> buffer(len);
recv(sock, buffer.data(), len, 0);   // 2. 接收数据

User user;
user.ParseFromArray(buffer.data(), len);
```

## 常见陷阱

### 陷阱 1：字节序问题

```cpp
// ❌ 不同机器字节序不同
uint32_t len = data.size();
send(sock, &len, sizeof(len), 0);  // 大端机器收到乱码

// ✅ 统一用网络字节序
uint32_t len = htonl(data.size());  // 主机序 → 网络序
send(sock, &len, sizeof(len), 0);

// 接收端
uint32_t len;
recv(sock, &len, sizeof(len), 0);
len = ntohl(len);  // 网络序 → 主机序
```

### 陷阱 2：JSON 解析失败

```cpp
// ❌ 不检查错误
json j = json::parse(s);  // s 不是合法 JSON 时崩溃

// ✅ 捕获异常
try {
    json j = json::parse(s);
} catch (const json::parse_error& e) {
    std::cerr << "解析错误: " << e.what();
}
```

### 陷阱 3：Protobuf 未初始化

```cpp
// ❌ 忘记设置必填字段
User user;
user.SerializeToString(&data);  // 某些字段为空

// ✅ 检查是否完整
if (!user.IsInitialized()) {
    std::cerr << "字段未完整设置";
}
```

### 陷阱 4：版本不兼容

```cpp
// ❌ 修改 .proto 文件时改变字段编号
message User {
    string name = 1;
    int32 age = 3;  // 原来是 2，改成 3 → 不兼容
}

// ✅ 永远不要改变编号，只能添加新字段
message User {
    string name = 1;
    int32 age = 2;
    string email = 3;  // 新增字段，编号递增
}
```

## 性能对比

**实测数据（10000 次序列化）**：

| 格式 | 体积 | 序列化时间 | 反序列化时间 |
|------|------|-----------|-------------|
| JSON | 65 字节 | 150ms | 180ms |
| Protobuf | 18 字节 | 8ms | 12ms |

**结论**：Protobuf 比 JSON 快 15-20 倍，体积小 3-4 倍

## 核心要点

1. **序列化**：对象 → 字节流
2. **常见格式**：
   - JSON：可读、跨语言、慢
   - Protobuf：快、小、工业标准
3. **Protobuf 流程**：
   - 定义 .proto → protoc 生成代码 → 使用
4. **网络传输**：
   - 先发长度，再发数据
   - 注意字节序（htonl/ntohl）
5. **向后兼容**：
   - 永远不改字段编号
   - 只能添加新字段
6. **实际应用**：
   - RPC 框架（gRPC、bRPC）
   - 数据存储
   - 微服务通信
