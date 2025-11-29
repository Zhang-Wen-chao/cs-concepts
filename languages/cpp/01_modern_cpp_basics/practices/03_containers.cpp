/**
 * 标准容器实践示例
 * 编译：g++ -std=c++17 03_containers.cpp -o containers
 * 运行：./containers
 */

#include <iostream>
#include <vector>
#include <deque>
#include <list>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <stack>
#include <queue>
#include <algorithm>
#include <numeric>
#include <string>

// ============ 示例 1：vector 基本用法 ============

void test_vector_basics() {
    std::cout << "\n=== 示例 1: vector 基本用法 ===" << std::endl;

    // 创建
    std::vector<int> v1;                    // 空
    std::vector<int> v2(5);                 // 5 个 0
    std::vector<int> v3(5, 42);             // 5 个 42
    std::vector<int> v4 = {1, 2, 3, 4, 5};  // 初始化列表

    std::cout << "v4 内容: ";
    for (int x : v4) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // 添加元素
    v1.push_back(10);
    v1.push_back(20);
    v1.emplace_back(30);  // 推荐：原地构造

    std::cout << "v1 大小: " << v1.size() << std::endl;
    std::cout << "v1 容量: " << v1.capacity() << std::endl;

    // 访问
    std::cout << "v4[0] = " << v4[0] << std::endl;
    std::cout << "v4.front() = " << v4.front() << std::endl;
    std::cout << "v4.back() = " << v4.back() << std::endl;

    // at() 会检查边界
    try {
        int x = v4.at(100);  // 抛异常
    } catch (const std::out_of_range& e) {
        std::cout << "越界异常: " << e.what() << std::endl;
    }
}

// ============ 示例 2：vector 容量管理 ============

void test_vector_capacity() {
    std::cout << "\n=== 示例 2: vector 容量管理 ===" << std::endl;

    std::vector<int> vec;

    std::cout << "初始 size: " << vec.size() << ", capacity: " << vec.capacity() << std::endl;

    for (int i = 0; i < 10; ++i) {
        vec.push_back(i);
        std::cout << "添加 " << i << " 后，size: " << vec.size()
                  << ", capacity: " << vec.capacity() << std::endl;
    }

    // 预留容量（避免重新分配）
    std::cout << "\n使用 reserve() 预留容量：" << std::endl;
    std::vector<int> vec2;
    vec2.reserve(10);  // 预留 10 个元素的空间
    std::cout << "reserve(10) 后，size: " << vec2.size()
              << ", capacity: " << vec2.capacity() << std::endl;

    for (int i = 0; i < 10; ++i) {
        vec2.push_back(i);
    }
    std::cout << "添加 10 个元素后，size: " << vec2.size()
              << ", capacity: " << vec2.capacity() << std::endl;
}

// ============ 示例 3：push_back vs emplace_back ============

class Point {
public:
    int x, y;

    Point(int x, int y) : x(x), y(y) {
        std::cout << "  Point(" << x << ", " << y << ") 构造" << std::endl;
    }

    Point(const Point& other) : x(other.x), y(other.y) {
        std::cout << "  Point 拷贝构造" << std::endl;
    }

    Point(Point&& other) noexcept : x(other.x), y(other.y) {
        std::cout << "  Point 移动构造" << std::endl;
    }
};

void test_emplace_back() {
    std::cout << "\n=== 示例 3: push_back vs emplace_back ===" << std::endl;

    std::vector<Point> points;
    points.reserve(5);  // 避免重新分配影响观察

    std::cout << "\npush_back(Point(1, 2)):" << std::endl;
    points.push_back(Point(1, 2));  // 构造临时对象，再移动

    std::cout << "\nemplace_back(3, 4):" << std::endl;
    points.emplace_back(3, 4);  // 直接在 vector 中构造（更高效）

    std::cout << "\n结论：emplace_back 更高效（少一次移动）" << std::endl;
}

// ============ 示例 4：vector 遍历和算法 ============

void test_vector_algorithms() {
    std::cout << "\n=== 示例 4: vector 遍历和算法 ===" << std::endl;

    std::vector<int> vec = {5, 2, 8, 1, 9, 3, 7};

    // 范围 for（推荐）
    std::cout << "原始数据: ";
    for (const auto& x : vec) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // 排序
    std::sort(vec.begin(), vec.end());
    std::cout << "排序后: ";
    for (int x : vec) std::cout << x << " ";
    std::cout << std::endl;

    // 查找
    auto it = std::find(vec.begin(), vec.end(), 8);
    if (it != vec.end()) {
        std::cout << "找到 8，位置: " << std::distance(vec.begin(), it) << std::endl;
    }

    // 累加
    int sum = std::accumulate(vec.begin(), vec.end(), 0);
    std::cout << "总和: " << sum << std::endl;

    // 统计
    int count = std::count_if(vec.begin(), vec.end(), [](int x) { return x > 5; });
    std::cout << "大于 5 的数: " << count << std::endl;
}

// ============ 示例 5：map 基本用法 ============

void test_map() {
    std::cout << "\n=== 示例 5: map（有序映射）===" << std::endl;

    std::map<std::string, int> age_map;

    // 插入
    age_map["Alice"] = 25;
    age_map["Bob"] = 30;
    age_map["Charlie"] = 35;
    age_map.insert({"David", 40});

    // 查找
    if (age_map.count("Alice")) {
        std::cout << "Alice 存在，年龄: " << age_map["Alice"] << std::endl;
    }

    auto it = age_map.find("Bob");
    if (it != age_map.end()) {
        std::cout << "Bob 的年龄: " << it->second << std::endl;
    }

    // 遍历（按键排序）
    std::cout << "\n所有人的年龄（按名字排序）：" << std::endl;
    for (const auto& [name, age] : age_map) {  // C++17 结构化绑定
        std::cout << "  " << name << ": " << age << std::endl;
    }

    // 注意：[] 会插入元素
    std::cout << "\n访问不存在的键：" << std::endl;
    std::cout << "Eve 的年龄: " << age_map["Eve"] << std::endl;  // 插入 {"Eve", 0}
    std::cout << "map 大小: " << age_map.size() << std::endl;    // 5（增加了一个）
}

// ============ 示例 6：unordered_map 基本用法 ============

void test_unordered_map() {
    std::cout << "\n=== 示例 6: unordered_map（哈希表）===" << std::endl;

    // 统计单词频率
    std::vector<std::string> words = {
        "apple", "banana", "apple", "cherry", "banana", "apple"
    };

    std::unordered_map<std::string, int> word_count;
    for (const auto& word : words) {
        word_count[word]++;  // 自动初始化为 0
    }

    std::cout << "单词频率：" << std::endl;
    for (const auto& [word, count] : word_count) {
        std::cout << "  " << word << ": " << count << std::endl;
    }

    // 性能对比提示
    std::cout << "\n提示：" << std::endl;
    std::cout << "  - unordered_map 查找 O(1)（平均）" << std::endl;
    std::cout << "  - map 查找 O(log n)" << std::endl;
    std::cout << "  - 大部分情况用 unordered_map" << std::endl;
}

// ============ 示例 7：set 和 unordered_set ============

void test_set() {
    std::cout << "\n=== 示例 7: set 和 unordered_set ===" << std::endl;

    // 有序集合（自动去重、排序）
    std::set<int> ordered_set = {5, 2, 8, 2, 1, 8, 3};
    std::cout << "有序集合（set）: ";
    for (int x : ordered_set) {
        std::cout << x << " ";  // 输出：1 2 3 5 8
    }
    std::cout << std::endl;

    // 无序集合（去重，但无序）
    std::unordered_set<int> unordered_set = {5, 2, 8, 2, 1, 8, 3};
    std::cout << "无序集合（unordered_set）: ";
    for (int x : unordered_set) {
        std::cout << x << " ";  // 顺序不确定
    }
    std::cout << std::endl;

    // 查找
    if (ordered_set.count(3)) {
        std::cout << "3 存在于集合中" << std::endl;
    }

    // 应用：去重
    std::vector<int> vec = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    std::unordered_set<int> unique_nums(vec.begin(), vec.end());
    std::cout << "去重后数量: " << unique_nums.size() << std::endl;
}

// ============ 示例 8：deque 双端队列 ============

void test_deque() {
    std::cout << "\n=== 示例 8: deque（双端队列）===" << std::endl;

    std::deque<int> dq = {1, 2, 3, 4, 5};

    // 两端操作
    dq.push_front(0);   // 头部插入
    dq.push_back(6);    // 尾部插入

    std::cout << "deque 内容: ";
    for (int x : dq) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    dq.pop_front();  // 头部删除
    dq.pop_back();   // 尾部删除

    std::cout << "删除两端后: ";
    for (int x : dq) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // 随机访问
    std::cout << "dq[2] = " << dq[2] << std::endl;
}

// ============ 示例 9：list 双向链表 ============

void test_list() {
    std::cout << "\n=== 示例 9: list（双向链表）===" << std::endl;

    std::list<int> lst = {1, 2, 3, 4, 5};

    // 两端操作
    lst.push_front(0);
    lst.push_back(6);

    std::cout << "list 内容: ";
    for (int x : lst) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    // 中间插入
    auto it = lst.begin();
    ++it;  // 指向第二个元素（值为 1）
    lst.insert(it, 99);  // 在 1 前面插入 99

    std::cout << "插入 99 后: ";
    for (int x : lst) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    std::cout << "\n注意：list 很少使用，vector 通常更快！" << std::endl;
}

// ============ 示例 10：stack 栈 ============

void test_stack() {
    std::cout << "\n=== 示例 10: stack（栈，LIFO）===" << std::endl;

    std::stack<int> stk;

    stk.push(1);
    stk.push(2);
    stk.push(3);

    std::cout << "栈顶元素: " << stk.top() << std::endl;  // 3

    std::cout << "弹出顺序: ";
    while (!stk.empty()) {
        std::cout << stk.top() << " ";
        stk.pop();
    }
    std::cout << std::endl;  // 输出：3 2 1
}

// ============ 示例 11：queue 队列 ============

void test_queue() {
    std::cout << "\n=== 示例 11: queue（队列，FIFO）===" << std::endl;

    std::queue<int> q;

    q.push(1);
    q.push(2);
    q.push(3);

    std::cout << "队首: " << q.front() << std::endl;  // 1
    std::cout << "队尾: " << q.back() << std::endl;   // 3

    std::cout << "弹出顺序: ";
    while (!q.empty()) {
        std::cout << q.front() << " ";
        q.pop();
    }
    std::cout << std::endl;  // 输出：1 2 3
}

// ============ 示例 12：priority_queue 优先队列 ============

void test_priority_queue() {
    std::cout << "\n=== 示例 12: priority_queue（优先队列/堆）===" << std::endl;

    // 大顶堆（默认）
    std::priority_queue<int> max_heap;
    max_heap.push(3);
    max_heap.push(1);
    max_heap.push(4);
    max_heap.push(2);

    std::cout << "大顶堆（从大到小）: ";
    while (!max_heap.empty()) {
        std::cout << max_heap.top() << " ";
        max_heap.pop();
    }
    std::cout << std::endl;  // 输出：4 3 2 1

    // 小顶堆
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;
    min_heap.push(3);
    min_heap.push(1);
    min_heap.push(4);
    min_heap.push(2);

    std::cout << "小顶堆（从小到大）: ";
    while (!min_heap.empty()) {
        std::cout << min_heap.top() << " ";
        min_heap.pop();
    }
    std::cout << std::endl;  // 输出：1 2 3 4
}

// ============ 示例 13：常见陷阱 - 迭代器失效 ============

void test_iterator_invalidation() {
    std::cout << "\n=== 示例 13: 常见陷阱 - 迭代器失效 ===" << std::endl;

    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::cout << "原始数据: ";
    for (int x : vec) std::cout << x << " ";
    std::cout << std::endl;

    // ❌ 错误：删除元素后迭代器失效
    std::cout << "\n错误做法（会崩溃）：" << std::endl;
    std::cout << "（已注释，避免崩溃）" << std::endl;
    /*
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        if (*it == 3) {
            vec.erase(it);  // it 失效！
        }
    }
    */

    // ✅ 正确：erase 返回下一个有效迭代器
    std::cout << "\n正确做法：" << std::endl;
    for (auto it = vec.begin(); it != vec.end(); ) {
        if (*it == 3) {
            it = vec.erase(it);  // erase 返回下一个迭代器
        } else {
            ++it;
        }
    }

    std::cout << "删除 3 后: ";
    for (int x : vec) std::cout << x << " ";
    std::cout << std::endl;
}

// ============ 示例 14：常见陷阱 - 不必要的拷贝 ============

void test_unnecessary_copy() {
    std::cout << "\n=== 示例 14: 常见陷阱 - 不必要的拷贝 ===" << std::endl;

    std::vector<std::string> vec = {
        "This is a long string 1",
        "This is a long string 2",
        "This is a long string 3"
    };

    // ❌ 错误：拷贝每个字符串（慢）
    std::cout << "\n错误做法（拷贝）：" << std::endl;
    int copy_count = 0;
    for (auto str : vec) {  // 拷贝！
        copy_count++;
    }
    std::cout << "拷贝了 " << copy_count << " 次" << std::endl;

    // ✅ 正确：用 const 引用（快）
    std::cout << "\n正确做法（引用）：" << std::endl;
    for (const auto& str : vec) {  // 引用，不拷贝
        std::cout << "  " << str << std::endl;
    }

    std::cout << "\n记住：范围 for 永远用 const auto&" << std::endl;
}

// ============ 示例 15：容器选择建议 ============

void print_container_guide() {
    std::cout << "\n=== 示例 15: 容器选择建议 ===" << std::endl;

    std::cout << "\n推荐使用频率：" << std::endl;
    std::cout << "  ⭐⭐⭐⭐⭐ vector - 90% 的情况" << std::endl;
    std::cout << "  ⭐⭐⭐⭐⭐ unordered_map - 需要键值对" << std::endl;
    std::cout << "  ⭐⭐⭐⭐⭐ unordered_set - 需要去重" << std::endl;
    std::cout << "  ⭐⭐⭐ map/set - 需要有序" << std::endl;
    std::cout << "  ⭐⭐⭐ queue/stack - 特定用途" << std::endl;
    std::cout << "  ⭐⭐ deque - 两端操作" << std::endl;
    std::cout << "  ⭐ list - 很少用" << std::endl;

    std::cout << "\n性能对比：" << std::endl;
    std::cout << "  - vector：随机访问 O(1)，末尾插入 O(1)" << std::endl;
    std::cout << "  - unordered_map：查找/插入 O(1)" << std::endl;
    std::cout << "  - map：查找/插入 O(log n)" << std::endl;
    std::cout << "  - list：任意位置插入 O(1)，但随机访问 O(n)" << std::endl;

    std::cout << "\n选择原则：" << std::endl;
    std::cout << "  1. 默认用 vector" << std::endl;
    std::cout << "  2. 需要键值对 → unordered_map" << std::endl;
    std::cout << "  3. 需要去重 → unordered_set" << std::endl;
    std::cout << "  4. 需要有序 → map/set" << std::endl;
    std::cout << "  5. 不要过早优化" << std::endl;
}

// ============ 主函数 ============

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      标准容器实践示例" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_vector_basics();
        test_vector_capacity();
        test_emplace_back();
        test_vector_algorithms();
        test_map();
        test_unordered_map();
        test_set();
        test_deque();
        test_list();
        test_stack();
        test_queue();
        test_priority_queue();
        test_iterator_invalidation();
        test_unnecessary_copy();
        print_container_guide();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  所有示例运行完成！✅" << std::endl;
        std::cout << "========================================" << std::endl;

        std::cout << "\n关键收获：" << std::endl;
        std::cout << "1. 默认用 vector（90% 的情况）" << std::endl;
        std::cout << "2. 需要键值对用 unordered_map（O(1) 查找）" << std::endl;
        std::cout << "3. 需要去重用 unordered_set" << std::endl;
        std::cout << "4. 用 emplace 而不是 push（更高效）" << std::endl;
        std::cout << "5. 范围 for 用 const auto&（避免拷贝）" << std::endl;
        std::cout << "6. 用标准算法（std::sort, std::find 等）" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
