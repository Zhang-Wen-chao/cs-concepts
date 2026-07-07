# 跨语言对比实现题

> 同一个题目用 C++、Go、Python 各实现一次，才能真正理解三语言的工程差异。

## 使用方法

每题三个子目录：`cpp/`、`go/`、`python/`。
先自己实现，再对比 languages/ 下的笔记和参考答案。

## 题目列表

### 1. 并发 Web Crawler

实现一个并发的 Web 爬虫，给定 URL 列表，限制并发数为 5，爬取页面内容并统计成功/失败。

| 语言 | 考察点 |
|------|--------|
| C++ | `std::thread` 线程池 / `std::async`、`std::mutex` 保护共享状态 |
| Go | goroutine + channel + `sync.WaitGroup` + semaphore 限流 |
| Python | `asyncio` + `aiohttp` + `asyncio.Semaphore` |

### 2. CLI 日志分析器

实现一个 CLI 工具，读取 access log，输出统计信息（请求数、错误率、P50/P99 延迟）。

| 语言 | 考察点 |
|------|--------|
| C++ | 文件流、字符串解析、`getline`、`std::map` 聚合 |
| Go | `bufio.Scanner`、flag 包、struct + method |
| Python | click/argparse、generator pipeline、dataclass |

### 3. 键值存储服务

实现一个简单的 HTTP 键值存储（GET/PUT/DELETE），数据在内存中。

| 语言 | 考察点 |
|------|--------|
| C++ | beast/asio HTTP 服务器、`std::unordered_map` + mutex |
| Go | `net/http`、`sync.RWMutex`、`encoding/json` |
| Python | FastAPI/Flask、dict、threading.Lock |

### 4. 生产者-消费者队列

实现一个多生产者-多消费者模型，处理一批任务并收集结果。

| 语言 | 考察点 |
|------|--------|
| C++ | `std::condition_variable` + `std::queue` + `std::mutex` |
| Go | channel + select + `sync.WaitGroup` |
| Python | `queue.Queue` + `threading` / `asyncio.Queue` |

### 5. 简易单元测试框架

实现一个极简的单元测试框架，支持注册测试用例、运行、输出结果。

| 语言 | 考察点 |
|------|--------|
| C++ | 宏、函数指针/`std::function`、RAII 计时器 |
| Go | `testing.T` 模拟、反射（可选） |
| Python | 装饰器、`inspect` 模块、`__call__` |
