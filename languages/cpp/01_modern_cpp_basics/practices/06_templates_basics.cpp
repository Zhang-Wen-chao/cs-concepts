/**
 * 模板基础实践示例
 * 编译：g++ -std=c++17 06_templates_basics.cpp -o templates
 * 运行：./templates
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <type_traits>

// ============ 示例 1：函数模板基础 ============

template<typename T>
T max_value(T a, T b) {
    return (a > b) ? a : b;
}

void test_function_template_basic() {
    std::cout << "\n=== 示例 1: 函数模板基础 ===" << std::endl;

    // 自动推导类型
    std::cout << "max(3, 5) = " << max_value(3, 5) << std::endl;  // T = int
    std::cout << "max(3.14, 2.71) = " << max_value(3.14, 2.71) << std::endl;  // T = double
    std::cout << "max('a', 'z') = " << max_value('a', 'z') << std::endl;  // T = char

    // 显式指定类型
    std::cout << "max<double>(3, 5) = " << max_value<double>(3, 5) << std::endl;

    std::cout << "\n模板让我们用同一个函数处理不同类型" << std::endl;
}

// ============ 示例 2：多个模板参数 ============

template<typename T1, typename T2>
void print_pair(const T1& a, const T2& b) {
    std::cout << "(" << a << ", " << b << ")" << std::endl;
}

template<typename T1, typename T2>
auto add(T1 a, T2 b) {  // C++14 自动推导返回类型
    return a + b;
}

void test_multiple_template_params() {
    std::cout << "\n=== 示例 2: 多个模板参数 ===" << std::endl;

    print_pair(42, "hello");
    print_pair(3.14, std::string("world"));
    print_pair('A', 100);

    std::cout << "\nadd(1, 2.5) = " << add(1, 2.5) << std::endl;  // int + double = double
    std::cout << "add(1.5, 2) = " << add(1.5, 2) << std::endl;    // double + int = double
}

// ============ 示例 3：类模板 - Stack ============

template<typename T>
class Stack {
    std::vector<T> elements_;

public:
    void push(const T& elem) {
        elements_.push_back(elem);
    }

    void pop() {
        if (!elements_.empty()) {
            elements_.pop_back();
        }
    }

    T top() const {
        if (elements_.empty()) {
            throw std::runtime_error("Stack is empty");
        }
        return elements_.back();
    }

    bool empty() const {
        return elements_.empty();
    }

    size_t size() const {
        return elements_.size();
    }
};

void test_class_template() {
    std::cout << "\n=== 示例 3: 类模板 - Stack ===" << std::endl;

    // int 栈
    Stack<int> int_stack;
    int_stack.push(1);
    int_stack.push(2);
    int_stack.push(3);

    std::cout << "int 栈: ";
    while (!int_stack.empty()) {
        std::cout << int_stack.top() << " ";
        int_stack.pop();
    }
    std::cout << std::endl;

    // string 栈
    Stack<std::string> string_stack;
    string_stack.push("hello");
    string_stack.push("world");

    std::cout << "string 栈: ";
    while (!string_stack.empty()) {
        std::cout << string_stack.top() << " ";
        string_stack.pop();
    }
    std::cout << std::endl;

    std::cout << "\n注意：类模板必须显式指定类型 Stack<int>" << std::endl;
}

// ============ 示例 4：非类型模板参数 ============

template<typename T, size_t N>
class Array {
    T data_[N];

public:
    size_t size() const {
        return N;
    }

    T& operator[](size_t index) {
        return data_[index];
    }

    const T& operator[](size_t index) const {
        return data_[index];
    }

    // 填充
    void fill(const T& value) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] = value;
        }
    }
};

void test_non_type_template_param() {
    std::cout << "\n=== 示例 4: 非类型模板参数 ===" << std::endl;

    Array<int, 5> arr1;
    arr1.fill(42);

    std::cout << "arr1 (大小 " << arr1.size() << "): ";
    for (size_t i = 0; i < arr1.size(); ++i) {
        std::cout << arr1[i] << " ";
    }
    std::cout << std::endl;

    Array<double, 3> arr2;
    arr2[0] = 1.1;
    arr2[1] = 2.2;
    arr2[2] = 3.3;

    std::cout << "arr2 (大小 " << arr2.size() << "): ";
    for (size_t i = 0; i < arr2.size(); ++i) {
        std::cout << arr2[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\n大小 N 是编译期常量，不占用运行时内存" << std::endl;
}

// ============ 示例 5：模板特化 ============

// 通用模板
template<typename T>
class Printer {
public:
    static void print(const T& value) {
        std::cout << "通用: " << value << std::endl;
    }
};

// 完全特化：针对 bool
template<>
class Printer<bool> {
public:
    static void print(bool value) {
        std::cout << "bool 特化: " << (value ? "true" : "false") << std::endl;
    }
};

// 完全特化：针对 const char*
template<>
class Printer<const char*> {
public:
    static void print(const char* value) {
        std::cout << "字符串特化: \"" << value << "\"" << std::endl;
    }
};

void test_template_specialization() {
    std::cout << "\n=== 示例 5: 模板特化 ===" << std::endl;

    Printer<int>::print(42);
    Printer<double>::print(3.14);
    Printer<bool>::print(true);
    Printer<const char*>::print("hello");

    std::cout << "\n特化让我们为特定类型提供定制实现" << std::endl;
}

// ============ 示例 6：函数模板特化 ============

// 通用模板
template<typename T>
T max_func(T a, T b) {
    std::cout << "  [通用版本] ";
    return (a > b) ? a : b;
}

// 特化：针对 const char*
template<>
const char* max_func<const char*>(const char* a, const char* b) {
    std::cout << "  [字符串特化] ";
    return (strcmp(a, b) > 0) ? a : b;
}

void test_function_specialization() {
    std::cout << "\n=== 示例 6: 函数模板特化 ===" << std::endl;

    std::cout << "max(3, 5) = " << max_func(3, 5) << std::endl;

    const char* s1 = "abc";
    const char* s2 = "xyz";
    std::cout << "max(\"abc\", \"xyz\") = " << max_func(s1, s2) << std::endl;

    std::cout << "\n特化版本正确比较字符串内容，而不是指针地址" << std::endl;
}

// ============ 示例 7：偏特化 ============

// 通用模板
template<typename T1, typename T2>
class Pair {
public:
    T1 first;
    T2 second;

    void print() const {
        std::cout << "通用 Pair: (" << first << ", " << second << ")" << std::endl;
    }
};

// 偏特化：两个类型相同
template<typename T>
class Pair<T, T> {
public:
    T first;
    T second;

    void print() const {
        std::cout << "相同类型 Pair: (" << first << ", " << second << ")" << std::endl;
    }
};

// 偏特化：指针类型
template<typename T1, typename T2>
class Pair<T1*, T2*> {
public:
    T1* first;
    T2* second;

    void print() const {
        std::cout << "指针 Pair: (" << *first << ", " << *second << ")" << std::endl;
    }
};

void test_partial_specialization() {
    std::cout << "\n=== 示例 7: 偏特化 ===" << std::endl;

    Pair<int, double> p1;
    p1.first = 1;
    p1.second = 2.5;
    p1.print();

    Pair<int, int> p2;
    p2.first = 10;
    p2.second = 20;
    p2.print();

    int x = 100;
    double y = 200.5;
    Pair<int*, double*> p3;
    p3.first = &x;
    p3.second = &y;
    p3.print();
}

// ============ 示例 8：变参模板 - 递归展开 ============

// 递归终止
void print_recursive() {
    std::cout << std::endl;
}

// 递归展开
template<typename T, typename... Args>
void print_recursive(T first, Args... rest) {
    std::cout << first << " ";
    print_recursive(rest...);
}

void test_variadic_template() {
    std::cout << "\n=== 示例 8: 变参模板（递归）===" << std::endl;

    std::cout << "打印多个参数: ";
    print_recursive(1, 2, 3, "hello", 3.14);

    std::cout << "可以接受任意数量、任意类型的参数" << std::endl;
}

// ============ 示例 9：变参模板 - 折叠表达式（C++17）============

// 求和
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);  // 折叠表达式
}

// 打印（用空格分隔）
template<typename... Args>
void print_fold(Args... args) {
    ((std::cout << args << " "), ...) << std::endl;
}

void test_fold_expression() {
    std::cout << "\n=== 示例 9: 折叠表达式（C++17）===" << std::endl;

    std::cout << "sum(1, 2, 3, 4, 5) = " << sum(1, 2, 3, 4, 5) << std::endl;
    std::cout << "sum(1.5, 2.5, 3.5) = " << sum(1.5, 2.5, 3.5) << std::endl;

    std::cout << "打印多个参数: ";
    print_fold(1, 2, 3, "hello", 3.14);

    std::cout << "\n折叠表达式让变参模板更简洁" << std::endl;
}

// ============ 示例 10：实际应用 - 泛型容器 ============

template<typename T>
class SimpleVector {
    T* data_;
    size_t size_;
    size_t capacity_;

public:
    SimpleVector() : data_(nullptr), size_(0), capacity_(0) {}

    ~SimpleVector() {
        delete[] data_;
    }

    // 禁止拷贝
    SimpleVector(const SimpleVector&) = delete;
    SimpleVector& operator=(const SimpleVector&) = delete;

    // 允许移动
    SimpleVector(SimpleVector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        data_[size_++] = value;
    }

    void reserve(size_t new_capacity) {
        if (new_capacity <= capacity_) return;

        T* new_data = new T[new_capacity];
        for (size_t i = 0; i < size_; ++i) {
            new_data[i] = data_[i];
        }
        delete[] data_;
        data_ = new_data;
        capacity_ = new_capacity;
    }

    size_t size() const { return size_; }
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
};

void test_real_world_example() {
    std::cout << "\n=== 示例 10: 实际应用 - 泛型容器 ===" << std::endl;

    SimpleVector<int> int_vec;
    int_vec.push_back(1);
    int_vec.push_back(2);
    int_vec.push_back(3);

    std::cout << "int vector: ";
    for (size_t i = 0; i < int_vec.size(); ++i) {
        std::cout << int_vec[i] << " ";
    }
    std::cout << std::endl;

    SimpleVector<std::string> str_vec;
    str_vec.push_back("hello");
    str_vec.push_back("world");

    std::cout << "string vector: ";
    for (size_t i = 0; i < str_vec.size(); ++i) {
        std::cout << str_vec[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\n一个模板类可以处理所有类型" << std::endl;
}

// ============ 示例 11：SFINAE 示例 ============

// 只对整数类型有效
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
double_value(T x) {
    std::cout << "  [整数版本] ";
    return x * 2;
}

// 只对浮点类型有效
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
double_value(T x) {
    std::cout << "  [浮点版本] ";
    return x * 2.0;
}

void test_sfinae() {
    std::cout << "\n=== 示例 11: SFINAE（类型约束）===" << std::endl;

    std::cout << "double_value(10) = " << double_value(10) << std::endl;
    std::cout << "double_value(3.14) = " << double_value(3.14) << std::endl;

    // double_value("hello");  // ❌ 编译错误：不满足任何条件

    std::cout << "\nSFINAE 让我们根据类型属性选择不同实现" << std::endl;
}

// ============ 示例 12：模板实例化演示 ============

template<typename T>
class Demo {
public:
    static void show_type() {
        std::cout << "Demo 实例化了一个新类型" << std::endl;
    }
};

void test_instantiation() {
    std::cout << "\n=== 示例 12: 模板实例化 ===" << std::endl;

    std::cout << "每种类型都会生成一份代码：" << std::endl;

    Demo<int>::show_type();     // 实例化 Demo<int>
    Demo<double>::show_type();  // 实例化 Demo<double>
    Demo<std::string>::show_type();  // 实例化 Demo<string>

    std::cout << "\n实例化发生在编译期" << std::endl;
    std::cout << "这就是为什么模板定义必须在头文件中" << std::endl;
}

// ============ 示例 13：模板最佳实践 ============

void print_best_practices() {
    std::cout << "\n=== 示例 13: 模板最佳实践 ===" << std::endl;

    std::cout << "\n1. 函数模板可以自动推导" << std::endl;
    std::cout << "   auto x = max(3, 5);  // T = int" << std::endl;

    std::cout << "\n2. 类模板必须显式指定（C++17 前）" << std::endl;
    std::cout << "   Stack<int> s;  // 必须写 <int>" << std::endl;

    std::cout << "\n3. 模板定义放在头文件中" << std::endl;
    std::cout << "   编译器需要看到完整定义才能实例化" << std::endl;

    std::cout << "\n4. 用 typename 修饰依赖类型" << std::endl;
    std::cout << "   typename T::value_type x;" << std::endl;

    std::cout << "\n5. 特化用于特定类型的优化" << std::endl;
    std::cout << "   template<> class Foo<bool> { ... };" << std::endl;

    std::cout << "\n6. C++17 折叠表达式简化变参模板" << std::endl;
    std::cout << "   return (args + ...);" << std::endl;
}

// ============ 主函数 ============

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      模板基础实践示例" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_function_template_basic();
        test_multiple_template_params();
        test_class_template();
        test_non_type_template_param();
        test_template_specialization();
        test_function_specialization();
        test_partial_specialization();
        test_variadic_template();
        test_fold_expression();
        test_real_world_example();
        test_sfinae();
        test_instantiation();
        print_best_practices();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  所有示例运行完成！✅" << std::endl;
        std::cout << "========================================" << std::endl;

        std::cout << "\n关键收获：" << std::endl;
        std::cout << "1. 模板 = 泛型编程 = 一次编写，处处复用" << std::endl;
        std::cout << "2. 函数模板自动推导，类模板需显式指定" << std::endl;
        std::cout << "3. 实例化发生在编译期" << std::endl;
        std::cout << "4. 定义必须在头文件中" << std::endl;
        std::cout << "5. 特化用于特定类型的定制" << std::endl;
        std::cout << "6. 变参模板处理任意数量参数" << std::endl;
        std::cout << "7. SFINAE 提供类型约束" << std::endl;

        std::cout << "\n🎉 恭喜！现代 C++ 基础（阶段 1）全部完成！" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
