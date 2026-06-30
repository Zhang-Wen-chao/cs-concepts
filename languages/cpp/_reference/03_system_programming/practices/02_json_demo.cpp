// 序列化实践：JSON 序列化（手写 + nlohmann/json）
// 编译：g++ -std=c++17 02_json_demo.cpp -o json_demo
// 运行：./json_demo
//
// 目的：演示 JSON 序列化的两种方式

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

// ===== 演示 1：手写 JSON 序列化 =====

struct User {
    std::string name;
    int age;
    std::string email;

    // 手写序列化
    std::string to_json() const {
        std::ostringstream oss;
        oss << "{"
            << "\"name\":\"" << name << "\","
            << "\"age\":" << age << ","
            << "\"email\":\"" << email << "\""
            << "}";
        return oss.str();
    }

    // 简单的手写反序列化（仅作演示，实际应用用库）
    static User from_json(const std::string& json) {
        User user;
        // 这里应该用正则或解析库，演示用简化版
        size_t name_pos = json.find("\"name\":\"") + 8;
        size_t name_end = json.find("\"", name_pos);
        user.name = json.substr(name_pos, name_end - name_pos);

        size_t age_pos = json.find("\"age\":") + 6;
        size_t age_end = json.find(",", age_pos);
        user.age = std::stoi(json.substr(age_pos, age_end - age_pos));

        size_t email_pos = json.find("\"email\":\"") + 9;
        size_t email_end = json.find("\"", email_pos);
        user.email = json.substr(email_pos, email_end - email_pos);

        return user;
    }
};

void demo_manual_json() {
    std::cout << "===== 演示 1：手写 JSON 序列化 =====\n";

    User user{"Alice", 25, "alice@example.com"};

    // 序列化
    std::string json = user.to_json();
    std::cout << "序列化结果: " << json << "\n";

    // 反序列化
    User user2 = User::from_json(json);
    std::cout << "反序列化: " << user2.name << ", " << user2.age << ", " << user2.email << "\n\n";
}

// ===== 演示 2：嵌套结构 =====

struct Address {
    std::string city;
    std::string street;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{"
            << "\"city\":\"" << city << "\","
            << "\"street\":\"" << street << "\""
            << "}";
        return oss.str();
    }
};

struct UserWithAddress {
    std::string name;
    int age;
    Address address;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{"
            << "\"name\":\"" << name << "\","
            << "\"age\":" << age << ","
            << "\"address\":" << address.to_json()
            << "}";
        return oss.str();
    }
};

void demo_nested() {
    std::cout << "===== 演示 2：嵌套结构 =====\n";

    UserWithAddress user{
        "Bob",
        30,
        {"Beijing", "Main St"}
    };

    std::string json = user.to_json();
    std::cout << "嵌套序列化: " << json << "\n\n";
}

// ===== 演示 3：数组序列化 =====

struct UserList {
    std::vector<User> users;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\"users\":[";
        for (size_t i = 0; i < users.size(); ++i) {
            oss << users[i].to_json();
            if (i < users.size() - 1) {
                oss << ",";
            }
        }
        oss << "]}";
        return oss.str();
    }
};

void demo_array() {
    std::cout << "===== 演示 3：数组序列化 =====\n";

    UserList list;
    list.users.push_back({"Alice", 25, "alice@example.com"});
    list.users.push_back({"Bob", 30, "bob@example.com"});
    list.users.push_back({"Charlie", 35, "charlie@example.com"});

    std::string json = list.to_json();
    std::cout << "数组序列化: " << json << "\n\n";
}

// ===== 演示 4：文件读写 =====

#include <fstream>

void demo_file_io() {
    std::cout << "===== 演示 4：文件读写 =====\n";

    User user{"David", 28, "david@example.com"};

    // 写入文件
    {
        std::ofstream ofs("user.json");
        ofs << user.to_json();
    }
    std::cout << "已写入 user.json\n";

    // 读取文件
    {
        std::ifstream ifs("user.json");
        std::string json((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
        User user2 = User::from_json(json);
        std::cout << "从文件读取: " << user2.name << ", " << user2.age << "\n";
    }

    std::cout << "\n";
}

int main() {
    demo_manual_json();
    demo_nested();
    demo_array();
    demo_file_io();

    std::cout << "提示：实际项目中应使用 nlohmann/json 或其他成熟的 JSON 库\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. 手写 JSON：
 *    - 适用简单场景
 *    - 容易出错（引号、逗号）
 *
 * 2. 嵌套结构：
 *    - 递归调用 to_json()
 *
 * 3. 数组：
 *    - 用 [] 包裹，逗号分隔
 *
 * 4. 实际应用：
 *    - 用成熟库（nlohmann/json）
 *    - 自动处理转义、类型检查
 *
 * 5. JSON 格式：
 *    {"key": "value", "key2": 123}
 *    - 字符串用双引号
 *    - 数字不用引号
 *    - 逗号分隔
 */
