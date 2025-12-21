# bRPC 安装指南

## 说明

bRPC 是一个完整的工业级框架，需要安装后才能使用。

**本项目包含**：
- ✅ Protobuf 定义（echo.proto）
- ✅ 示例代码（echo_server.cpp、echo_client.cpp）
- ✅ 完整文档（README.md）

**不包含**：
- ❌ 实际可运行的程序（需要先安装 bRPC）

## 安装步骤

### macOS

```bash
# 1. 安装依赖
brew install protobuf leveldb gflags openssl

# 2. 克隆 bRPC
cd ~/Downloads
git clone https://github.com/apache/brpc.git
cd brpc

# 3. 编译（需要 10-20 分钟）
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

# 4. 安装
sudo make install
```

### Linux (Ubuntu/Debian)

```bash
# 1. 安装依赖
sudo apt-get install -y git g++ make libssl-dev libgflags-dev libprotobuf-dev \
    libprotoc-dev protobuf-compiler libleveldb-dev

# 2. 克隆并编译 bRPC
git clone https://github.com/apache/brpc.git
cd brpc
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

## 验证安装

```bash
# 检查头文件
ls /usr/local/include/brpc

# 检查库文件
ls /usr/local/lib/libbrpc.*

# 测试编译
protoc --version  # 应该显示 protobuf 版本
```

## 编译本项目

**安装 bRPC 后**，回到本项目目录：

```bash
cd /path/to/02_brpc_hello

# 1. 生成 Protobuf 代码
protoc --cpp_out=. echo.proto

# 2. 编译服务器和客户端
# （需要创建 Makefile，见下文）
make

# 3. 运行
./echo_server  # 终端1
./echo_client  # 终端2
```

## Makefile 示例

```makefile
CXX = g++
CXXFLAGS = -std=c++11 -Wall -g
PROTOC = protoc
BRPC_PATH = /usr/local
PROTOBUF_PATH = /usr/local

INCLUDES = -I$(BRPC_PATH)/include -I$(PROTOBUF_PATH)/include
LIBS = -L$(BRPC_PATH)/lib -L$(PROTOBUF_PATH)/lib \
       -lbrpc -lprotobuf -lleveldb -lgflags -lssl -lcrypto -lz

all: echo_server echo_client

# 生成 Protobuf 代码
echo.pb.cc echo.pb.h: echo.proto
	$(PROTOC) --cpp_out=. $<

# 编译服务器
echo_server: echo_server.cpp echo.pb.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

# 编译客户端
echo_client: echo_client.cpp echo.pb.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

clean:
	rm -f echo_server echo_client echo.pb.* *.o

.PHONY: all clean
```

## 为什么不包含可运行程序？

1. **体积大**：bRPC 编译后很大（~100MB）
2. **依赖多**：需要 protobuf、gflags、leveldb 等
3. **学习目的**：
   - 简化版 RPC：理解原理（可直接运行）
   - bRPC：了解工业实践（示例代码 + 文档）

## 简化学习路径

**如果不想安装 bRPC**，可以：

1. ✅ **阅读 README.md**：理解 bRPC 的设计和用法
2. ✅ **查看示例代码**：理解如何使用 bRPC
3. ✅ **对比简化版 RPC**：理解工业级框架的优势
4. ❌ 跳过实际运行（可选）

**核心目标**：理解工业级 RPC 框架的设计思想，而非每个细节实现。

## 推荐学习顺序

1. ✅ 完成简化版 RPC 项目（已完成）
2. ✅ 阅读 bRPC 文档和示例代码（当前）
3. ⏭️ 继续下一个项目（推荐服务）

## 总结

**bRPC 项目的价值**：
- 理解工业级 RPC 框架的设计
- 对比学习：简单 vs 工业级
- 为实际工作做准备

**不必强求**：
- 不一定要安装运行
- 理解思想更重要
- 实际工作中会用到时再深入
