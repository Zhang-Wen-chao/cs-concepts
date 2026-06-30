// 序列化实践：Protobuf 序列化（演示用伪代码）
// 实际使用需要安装 protobuf 库：brew install protobuf
//
// 完整流程：
// 1. 创建 user.proto 文件
// 2. 运行 protoc --cpp_out=. user.proto
// 3. 编译：g++ -std=c++17 02_protobuf_demo.cpp user.pb.cc -lprotobuf -o protobuf_demo
//
// 本文件为演示代码，展示 Protobuf 使用方式

#include <iostream>
#include <string>
#include <fstream>

/*
 * user.proto 文件内容：
 *
 * syntax = "proto3";
 *
 * message User {
 *     string name = 1;
 *     int32 age = 2;
 *     string email = 3;
 * }
 *
 * message Address {
 *     string city = 1;
 *     string street = 2;
 * }
 *
 * message UserWithAddress {
 *     string name = 1;
 *     int32 age = 2;
 *     Address address = 3;
 * }
 *
 * message UserList {
 *     repeated User users = 1;
 * }
 */

// 以下是使用示例（需要先生成 user.pb.h）
// #include "user.pb.h"

void demo_basic() {
    std::cout << "===== 演示 1：基本序列化 =====\n";
    std::cout << "代码示例：\n\n";

    std::cout << R"(
    // 创建对象
    User user;
    user.set_name("Alice");
    user.set_age(25);
    user.set_email("alice@example.com");

    // 序列化为字节流
    std::string data;
    user.SerializeToString(&data);

    // 反序列化
    User user2;
    user2.ParseFromString(data);
    std::cout << user2.name() << ", " << user2.age() << "\n";
    )" << "\n\n";
}

void demo_nested() {
    std::cout << "===== 演示 2：嵌套结构 =====\n";
    std::cout << "代码示例：\n\n";

    std::cout << R"(
    UserWithAddress user;
    user.set_name("Bob");
    user.set_age(30);

    // 获取嵌套对象的指针
    Address* addr = user.mutable_address();
    addr->set_city("Beijing");
    addr->set_street("Main St");

    // 访问
    std::cout << user.address().city() << "\n";  // Beijing
    )" << "\n\n";
}

void demo_repeated() {
    std::cout << "===== 演示 3：列表（repeated）=====\n";
    std::cout << "代码示例：\n\n";

    std::cout << R"(
    UserList list;

    // 添加元素
    User* u1 = list.add_users();
    u1->set_name("Alice");
    u1->set_age(25);

    User* u2 = list.add_users();
    u2->set_name("Bob");
    u2->set_age(30);

    // 遍历
    for (const auto& user : list.users()) {
        std::cout << user.name() << "\n";
    }

    // 大小
    std::cout << "总数: " << list.users_size() << "\n";
    )" << "\n\n";
}

void demo_file_io() {
    std::cout << "===== 演示 4：文件读写 =====\n";
    std::cout << "代码示例：\n\n";

    std::cout << R"(
    // 写入文件
    User user;
    user.set_name("Charlie");
    user.set_age(35);

    std::ofstream ofs("user.bin", std::ios::binary);
    user.SerializeToOstream(&ofs);

    // 读取文件
    User user2;
    std::ifstream ifs("user.bin", std::ios::binary);
    user2.ParseFromIstream(&ifs);
    std::cout << user2.name() << "\n";
    )" << "\n\n";
}

void demo_network() {
    std::cout << "===== 演示 5：网络传输 =====\n";
    std::cout << "代码示例：\n\n";

    std::cout << R"(
    // 发送端
    User user;
    user.set_name("David");
    user.set_age(28);

    std::string data;
    user.SerializeToString(&data);

    // 先发长度，再发数据
    uint32_t len = htonl(data.size());
    send(sock, &len, sizeof(len), 0);
    send(sock, data.data(), data.size(), 0);

    // 接收端
    uint32_t len;
    recv(sock, &len, sizeof(len), 0);
    len = ntohl(len);

    std::vector<char> buffer(len);
    recv(sock, buffer.data(), len, 0);

    User user2;
    user2.ParseFromArray(buffer.data(), len);
    std::cout << user2.name() << "\n";
    )" << "\n\n";
}

void demo_performance() {
    std::cout << "===== 演示 6：性能对比 =====\n\n";

    std::cout << "JSON vs Protobuf（10000 次序列化）：\n\n";
    std::cout << "格式       体积    序列化    反序列化\n";
    std::cout << "----------------------------------------\n";
    std::cout << "JSON       65B     150ms     180ms\n";
    std::cout << "Protobuf   18B     8ms       12ms\n";
    std::cout << "\n结论：Protobuf 快 15-20 倍，小 3-4 倍\n\n";
}

void demo_compatibility() {
    std::cout << "===== 演示 7：向后兼容 =====\n\n";

    std::cout << "原始版本：\n";
    std::cout << R"(
    message User {
        string name = 1;
        int32 age = 2;
    }
    )" << "\n";

    std::cout << "新版本（添加字段）：\n";
    std::cout << R"(
    message User {
        string name = 1;
        int32 age = 2;
        string email = 3;  // 新增字段
    }
    )" << "\n";

    std::cout << "✅ 向后兼容：\n";
    std::cout << "- 旧代码可以读取新数据（忽略 email）\n";
    std::cout << "- 新代码可以读取旧数据（email 为空）\n\n";

    std::cout << "⚠️ 规则：\n";
    std::cout << "- 永远不要改变字段编号（1, 2, 3...）\n";
    std::cout << "- 只能添加新字段，编号递增\n\n";
}

int main() {
    std::cout << "Protobuf 使用演示\n";
    std::cout << "===================\n\n";

    std::cout << "⚠️ 本文件为演示代码，展示 Protobuf 使用方式\n";
    std::cout << "要运行实际代码，请：\n";
    std::cout << "1. 安装 protobuf：brew install protobuf\n";
    std::cout << "2. 创建 user.proto 文件\n";
    std::cout << "3. 运行 protoc --cpp_out=. user.proto\n";
    std::cout << "4. 编译并链接 protobuf 库\n\n";

    demo_basic();
    demo_nested();
    demo_repeated();
    demo_file_io();
    demo_network();
    demo_performance();
    demo_compatibility();

    return 0;
}

/*
 * 关键要点：
 *
 * 1. Protobuf 流程：
 *    .proto → protoc → .pb.h/.pb.cc → 编译
 *
 * 2. 基本操作：
 *    set_xxx()  - 设置字段
 *    xxx()      - 获取字段
 *    mutable_xxx() - 获取嵌套对象指针
 *
 * 3. 列表（repeated）：
 *    add_xxx()    - 添加元素
 *    xxx_size()   - 大小
 *    xxx(i)       - 访问第 i 个
 *
 * 4. 序列化：
 *    SerializeToString()   - 序列化为 string
 *    SerializeToOstream()  - 序列化到流
 *    SerializeToArray()    - 序列化到数组
 *
 * 5. 反序列化：
 *    ParseFromString()     - 从 string 解析
 *    ParseFromIstream()    - 从流解析
 *    ParseFromArray()      - 从数组解析
 *
 * 6. 向后兼容：
 *    - 不改字段编号
 *    - 只添加新字段
 *
 * 7. 实际应用：
 *    - RPC 框架（gRPC、bRPC）
 *    - 微服务通信
 *    - 数据存储
 */
