/**
 * 智能指针实践示例
 * 编译：g++ -std=c++17 02_smart_pointers.cpp -o smart_ptr_examples
 * 运行：./smart_ptr_examples
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>

// ============ 示例 1：unique_ptr 基本用法 ============

void test_unique_ptr_basic() {
    std::cout << "\n=== 示例 1: unique_ptr 基本用法 ===" << std::endl;

    // 创建 unique_ptr
    auto p1 = std::make_unique<int>(42);
    std::cout << "p1 值: " << *p1 << std::endl;

    // ❌ 不能拷贝
    // auto p2 = p1;  // 编译错误

    // ✅ 可以移动（转移所有权）
    auto p2 = std::move(p1);
    std::cout << "p2 值: " << *p2 << std::endl;

    // p1 现在是空的
    if (!p1) {
        std::cout << "p1 现在是空的" << std::endl;
    }

    // 重置
    p2.reset();
    std::cout << "p2 已重置" << std::endl;
}

// ============ 示例 2：unique_ptr 数组 ============

void test_unique_ptr_array() {
    std::cout << "\n=== 示例 2: unique_ptr 管理数组 ===" << std::endl;

    // 动态数组
    auto arr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; ++i) {
        arr[i] = i * 10;
    }

    std::cout << "数组内容: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    // ⚠️ 但更推荐用 vector
    std::cout << "\n更推荐用 vector:" << std::endl;
    std::vector<int> vec(5);
    for (int i = 0; i < 5; ++i) {
        vec[i] = i * 10;
    }
}

// ============ 示例 3：unique_ptr 工厂模式 ============

class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
};

class Circle : public Shape {
public:
    void draw() const override {
        std::cout << "绘制圆形" << std::endl;
    }
};

class Rectangle : public Shape {
public:
    void draw() const override {
        std::cout << "绘制矩形" << std::endl;
    }
};

// 工厂函数返回 unique_ptr
std::unique_ptr<Shape> create_shape(const std::string& type) {
    if (type == "circle") {
        return std::make_unique<Circle>();
    } else if (type == "rectangle") {
        return std::make_unique<Rectangle>();
    }
    return nullptr;
}

void test_unique_ptr_factory() {
    std::cout << "\n=== 示例 3: unique_ptr 工厂模式 ===" << std::endl;

    auto shape1 = create_shape("circle");
    if (shape1) {
        shape1->draw();
    }

    auto shape2 = create_shape("rectangle");
    if (shape2) {
        shape2->draw();
    }
}

// ============ 示例 4：unique_ptr 在容器中 ============

void test_unique_ptr_in_container() {
    std::cout << "\n=== 示例 4: unique_ptr 在容器中 ===" << std::endl;

    std::vector<std::unique_ptr<Shape>> shapes;

    // 必须用 move 或直接创建
    shapes.push_back(std::make_unique<Circle>());
    shapes.push_back(std::make_unique<Rectangle>());
    shapes.push_back(create_shape("circle"));

    std::cout << "绘制所有图形:" << std::endl;
    for (const auto& shape : shapes) {
        if (shape) {
            shape->draw();
        }
    }
}

// ============ 示例 5：shared_ptr 基本用法 ============

void test_shared_ptr_basic() {
    std::cout << "\n=== 示例 5: shared_ptr 基本用法 ===" << std::endl;

    // 创建 shared_ptr
    auto p1 = std::make_shared<int>(100);
    std::cout << "p1 值: " << *p1 << ", 引用计数: " << p1.use_count() << std::endl;

    {
        // 拷贝，引用计数 +1
        auto p2 = p1;
        std::cout << "p2 创建后，引用计数: " << p1.use_count() << std::endl;

        auto p3 = p1;
        std::cout << "p3 创建后，引用计数: " << p1.use_count() << std::endl;

    }  // p2、p3 销毁，引用计数 -2

    std::cout << "p2、p3 销毁后，引用计数: " << p1.use_count() << std::endl;
}

// ============ 示例 6：shared_ptr 共享资源 ============

class Resource {
public:
    Resource(const std::string& name) : name_(name) {
        std::cout << "Resource " << name_ << " 创建" << std::endl;
    }

    ~Resource() {
        std::cout << "Resource " << name_ << " 销毁" << std::endl;
    }

    void use() {
        std::cout << "使用 Resource " << name_ << std::endl;
    }

private:
    std::string name_;
};

void test_shared_ptr_resource() {
    std::cout << "\n=== 示例 6: shared_ptr 共享资源 ===" << std::endl;

    auto res = std::make_shared<Resource>("共享数据");

    // 多个对象共享同一资源
    std::vector<std::shared_ptr<Resource>> users;
    users.push_back(res);
    users.push_back(res);
    users.push_back(res);

    std::cout << "引用计数: " << res.use_count() << std::endl;

    std::cout << "所有用户使用资源:" << std::endl;
    for (auto& user : users) {
        user->use();
    }

    std::cout << "清空 users 向量" << std::endl;
    users.clear();

    std::cout << "引用计数: " << res.use_count() << std::endl;

    std::cout << "离开作用域，res 销毁" << std::endl;
}

// ============ 示例 7：循环引用问题 ============

struct NodeBad {
    std::shared_ptr<NodeBad> next;
    std::shared_ptr<NodeBad> prev;  // ❌ 会导致循环引用
    std::string data;

    NodeBad(const std::string& d) : data(d) {
        std::cout << "Node " << data << " 创建" << std::endl;
    }

    ~NodeBad() {
        std::cout << "Node " << data << " 销毁" << std::endl;
    }
};

void test_circular_reference_bad() {
    std::cout << "\n=== 示例 7a: 循环引用（错误示范）===" << std::endl;

    {
        auto n1 = std::make_shared<NodeBad>("A");
        auto n2 = std::make_shared<NodeBad>("B");

        n1->next = n2;  // A → B
        n2->prev = n1;  // B → A（循环引用！）

        std::cout << "n1 引用计数: " << n1.use_count() << std::endl;  // 2
        std::cout << "n2 引用计数: " << n2.use_count() << std::endl;  // 2

        std::cout << "离开作用域..." << std::endl;
    }  // 💥 内存泄漏！n1 和 n2 都不会被销毁

    std::cout << "（注意：上面的节点没有被销毁！）" << std::endl;
}

// ============ 示例 8：weak_ptr 解决循环引用 ============

struct NodeGood {
    std::shared_ptr<NodeGood> next;  // 强引用
    std::weak_ptr<NodeGood> prev;    // 弱引用（打破循环）
    std::string data;

    NodeGood(const std::string& d) : data(d) {
        std::cout << "Node " << data << " 创建" << std::endl;
    }

    ~NodeGood() {
        std::cout << "Node " << data << " 销毁" << std::endl;
    }
};

void test_weak_ptr_solution() {
    std::cout << "\n=== 示例 8: weak_ptr 解决循环引用 ===" << std::endl;

    {
        auto n1 = std::make_shared<NodeGood>("A");
        auto n2 = std::make_shared<NodeGood>("B");

        n1->next = n2;  // A → B（强引用）
        n2->prev = n1;  // B ⇢ A（弱引用，不增加计数）

        std::cout << "n1 引用计数: " << n1.use_count() << std::endl;  // 1
        std::cout << "n2 引用计数: " << n2.use_count() << std::endl;  // 2

        std::cout << "离开作用域..." << std::endl;
    }  // ✅ 正确释放！

    std::cout << "（节点已正确销毁）" << std::endl;
}

// ============ 示例 9：weak_ptr 基本用法 ============

void test_weak_ptr_basic() {
    std::cout << "\n=== 示例 9: weak_ptr 基本用法 ===" << std::endl;

    std::weak_ptr<int> wp;

    {
        auto sp = std::make_shared<int>(42);
        wp = sp;  // 弱引用，不增加引用计数

        std::cout << "shared_ptr 引用计数: " << sp.use_count() << std::endl;

        // 使用 weak_ptr：先转换成 shared_ptr
        if (auto temp_sp = wp.lock()) {
            std::cout << "通过 weak_ptr 访问值: " << *temp_sp << std::endl;
        }

        std::cout << "离开作用域，shared_ptr 销毁..." << std::endl;
    }

    // 检查对象是否还存活
    if (wp.expired()) {
        std::cout << "对象已被释放" << std::endl;
    }

    // 尝试访问
    if (auto temp_sp = wp.lock()) {
        std::cout << "值: " << *temp_sp << std::endl;
    } else {
        std::cout << "无法访问，对象已释放" << std::endl;
    }
}

// ============ 示例 10：自定义删除器 ============

void test_custom_deleter() {
    std::cout << "\n=== 示例 10: 自定义删除器 ===" << std::endl;

    // 管理 FILE*
    auto file_deleter = [](FILE* f) {
        if (f) {
            std::fclose(f);
            std::cout << "文件已关闭" << std::endl;
        }
    };

    // 创建测试文件
    {
        FILE* f = std::fopen("test.txt", "w");
        if (f) {
            std::fprintf(f, "Hello");
            std::fclose(f);
        }
    }

    // 用 unique_ptr 管理 FILE*
    {
        std::unique_ptr<FILE, decltype(file_deleter)> file(
            std::fopen("test.txt", "r"),
            file_deleter
        );

        if (file) {
            char buffer[100];
            if (std::fgets(buffer, sizeof(buffer), file.get())) {
                std::cout << "读取内容: " << buffer << std::endl;
            }
        }

        std::cout << "离开作用域，自动关闭文件" << std::endl;
    }
}

// ============ 示例 11：性能对比 ============

void test_performance() {
    std::cout << "\n=== 示例 11: 大小对比 ===" << std::endl;

    std::cout << "sizeof(int*):              " << sizeof(int*) << " 字节" << std::endl;
    std::cout << "sizeof(unique_ptr<int>):   " << sizeof(std::unique_ptr<int>) << " 字节" << std::endl;
    std::cout << "sizeof(shared_ptr<int>):   " << sizeof(std::shared_ptr<int>) << " 字节" << std::endl;
    std::cout << "sizeof(weak_ptr<int>):     " << sizeof(std::weak_ptr<int>) << " 字节" << std::endl;

    std::cout << "\n结论：" << std::endl;
    std::cout << "- unique_ptr 和裸指针一样大（零开销）" << std::endl;
    std::cout << "- shared_ptr 是裸指针的 2 倍（有控制块开销）" << std::endl;
}

// ============ 主函数 ============

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      智能指针实践示例" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_unique_ptr_basic();
        test_unique_ptr_array();
        test_unique_ptr_factory();
        test_unique_ptr_in_container();
        test_shared_ptr_basic();
        test_shared_ptr_resource();
        test_circular_reference_bad();
        test_weak_ptr_solution();
        test_weak_ptr_basic();
        test_custom_deleter();
        test_performance();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  所有示例运行完成！✅" << std::endl;
        std::cout << "========================================" << std::endl;

        std::cout << "\n关键收获：" << std::endl;
        std::cout << "1. unique_ptr：独占所有权，90% 的情况用它" << std::endl;
        std::cout << "2. shared_ptr：共享所有权，需要共享时使用" << std::endl;
        std::cout << "3. weak_ptr：打破循环引用" << std::endl;
        std::cout << "4. 永远用 make_unique/make_shared，不手动 new" << std::endl;
        std::cout << "5. unique_ptr 零开销，放心使用" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
