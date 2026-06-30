// 性能优化实践：CPU 缓存优化
// 编译：g++ -std=c++17 -O2 04_cache_demo.cpp -o cache_demo
// 运行：./cache_demo
//
// 目的：演示 CPU 缓存对性能的影响

#include <iostream>
#include <vector>
#include <chrono>
#include <random>

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    const char* name_;
public:
    Timer(const char* name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cout << name_ << ": " << duration.count() << " μs\n";
    }
};

// ===== 演示 1：顺序 vs 随机访问 =====

void demo_sequential_vs_random() {
    std::cout << "===== 演示 1：顺序 vs 随机访问 =====\n";
    const int N = 10000000;
    std::vector<int> data(N);
    for (int i = 0; i < N; ++i) {
        data[i] = i;
    }

    // 顺序访问
    {
        Timer t("顺序访问");
        long long sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += data[i];  // 缓存命中率高
        }
    }

    // 随机访问
    std::vector<int> indices(N);
    for (int i = 0; i < N; ++i) {
        indices[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    {
        Timer t("随机访问");
        long long sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += data[indices[i]];  // 缓存未命中率高
        }
    }

    std::cout << "\n";
}

// ===== 演示 2：AoS vs SoA =====

// Array of Structures（结构数组）
struct ParticleAoS {
    float x, y, z;       // 位置
    float vx, vy, vz;    // 速度
    float r, g, b;       // 颜色
};

// Structure of Arrays（数组结构）
struct ParticlesSoA {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> r, g, b;
};

void demo_aos_vs_soa() {
    std::cout << "===== 演示 2：AoS vs SoA =====\n";
    const int N = 1000000;

    // AoS
    std::vector<ParticleAoS> particles_aos(N);
    for (int i = 0; i < N; ++i) {
        particles_aos[i] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f, 1.0f, 0.0f, 0.0f};
    }

    {
        Timer t("AoS - 更新位置");
        for (auto& p : particles_aos) {
            p.x += p.vx;  // 加载整个结构（36 字节），只用 8 字节
            p.y += p.vy;
            p.z += p.vz;
        }
    }

    // SoA
    ParticlesSoA particles_soa;
    particles_soa.x.resize(N, 1.0f);
    particles_soa.y.resize(N, 2.0f);
    particles_soa.z.resize(N, 3.0f);
    particles_soa.vx.resize(N, 0.1f);
    particles_soa.vy.resize(N, 0.2f);
    particles_soa.vz.resize(N, 0.3f);

    {
        Timer t("SoA - 更新位置");
        for (int i = 0; i < N; ++i) {
            particles_soa.x[i] += particles_soa.vx[i];  // 只加载需要的数据
            particles_soa.y[i] += particles_soa.vy[i];
            particles_soa.z[i] += particles_soa.vz[i];
        }
    }

    std::cout << "SoA 更缓存友好，通常快 2-5 倍\n\n";
}

// ===== 演示 3：数据对齐 =====

// ❌ 不对齐
struct __attribute__((packed)) BadLayout {
    char c;
    int i;
    char c2;
    double d;
};

// ✅ 对齐
struct GoodLayout {
    double d;
    int i;
    char c;
    char c2;
};

void demo_alignment() {
    std::cout << "===== 演示 3：数据对齐 =====\n";
    std::cout << "BadLayout 大小: " << sizeof(BadLayout) << " 字节\n";
    std::cout << "GoodLayout 大小: " << sizeof(GoodLayout) << " 字节\n";

    const int N = 1000000;

    std::vector<BadLayout> bad(N);
    {
        Timer t("不对齐访问");
        for (auto& b : bad) {
            b.d += 1.0;
        }
    }

    std::vector<GoodLayout> good(N);
    {
        Timer t("对齐访问");
        for (auto& g : good) {
            g.d += 1.0;
        }
    }

    std::cout << "\n";
}

// ===== 演示 4：跨步访问 =====

void demo_stride() {
    std::cout << "===== 演示 4：跨步访问 =====\n";
    const int SIZE = 10000;
    std::vector<std::vector<int>> matrix(SIZE, std::vector<int>(SIZE, 1));

    // 按行访问（缓存友好）
    {
        Timer t("按行访问");
        long long sum = 0;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                sum += matrix[i][j];  // 连续访问
            }
        }
    }

    // 按列访问（缓存不友好）
    {
        Timer t("按列访问");
        long long sum = 0;
        for (int j = 0; j < SIZE; ++j) {
            for (int i = 0; i < SIZE; ++i) {
                sum += matrix[i][j];  // 跳跃访问
            }
        }
    }

    std::cout << "按行访问通常快 5-10 倍\n\n";
}

// ===== 演示 5：缓存行填充 =====

// ❌ 伪共享（false sharing）
struct Counter {
    std::atomic<int> count{0};
};

// ✅ 避免伪共享（缓存行填充）
struct alignas(64) PaddedCounter {
    std::atomic<int> count{0};
    char padding[60];  // 填充到 64 字节（一个缓存行）
};

void demo_false_sharing() {
    std::cout << "===== 演示 5：避免伪共享 =====\n";
    std::cout << "Counter 大小: " << sizeof(Counter) << " 字节\n";
    std::cout << "PaddedCounter 大小: " << sizeof(PaddedCounter) << " 字节\n";
    std::cout << "缓存行填充避免多核竞争同一缓存行\n\n";
}

int main() {
    std::cout << "CPU 缓存优化演示\n";
    std::cout << "==================\n\n";

    demo_sequential_vs_random();
    demo_aos_vs_soa();
    demo_alignment();
    demo_stride();
    demo_false_sharing();

    std::cout << "总结：\n";
    std::cout << "1. 顺序访问比随机访问快 10-100 倍\n";
    std::cout << "2. SoA 比 AoS 更缓存友好\n";
    std::cout << "3. 数据对齐提升访问速度\n";
    std::cout << "4. 按行访问比按列快（C++ 行优先）\n";
    std::cout << "5. 缓存行填充避免伪共享\n";

    return 0;
}

/*
 * 关键要点：
 *
 * 1. CPU 缓存层次：
 *    L1：~1ns，32KB
 *    L2：~4ns，256KB
 *    L3：~10ns，8MB
 *    RAM：~100ns
 *
 * 2. 缓存行（Cache Line）：
 *    - 通常 64 字节
 *    - 一次加载 64 字节
 *    - 连续数据一起加载
 *
 * 3. 优化策略：
 *    - 顺序访问
 *    - 连续内存（vector）
 *    - 数据局部性
 *    - SoA 代替 AoS
 *
 * 4. 伪共享（False Sharing）：
 *    - 多核修改同一缓存行
 *    - 缓存行填充解决
 *
 * 5. 性能差异：
 *    - 顺序 vs 随机：10-100 倍
 *    - 按行 vs 按列：5-10 倍
 *    - SoA vs AoS：2-5 倍
 */
