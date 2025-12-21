# bRPC 推荐服务 Demo

> 用 bRPC 将一个“伪推荐模型”包装成 RPC 服务，演示模型在线化的关键步骤。

## 目标

- 使用 Protobuf 定义 `Recommend` RPC 接口
- 实现一个可加载简单模型配置的 bRPC 服务器
- 提供 CLI 客户端，用于构造请求并查看返回的 Top-K 结果
- 打通 `make -> 启动服务 -> 客户端调用` 的完整流程

## 目录结构

```
03_recommend_service/
├── README.md
├── Makefile
├── data/
│   └── model_config.csv
├── recommend.proto
├── server.cpp
└── client.cpp
```

## 模型配置

`data/model_config.csv` 采用三列 CSV：`type,key,value`。

- `type=user` 表示用户偏置，例如 `user,alice,0.8`
- `type=item` 表示物品热度，例如 `item,itemA,1.2`

服务端在启动时读取该文件，若文件不存在则退回内置默认权重。

## 构建

```bash
cd languages/cpp/04_projects/03_recommend_service
make            # 生成 recommend.pb.cc/h、server、client
```

## 运行

启动服务器（默认监听 9000 端口，可改 `--port`，模型文件通过 `--model_path` 指定）：

```bash
./server --port=9000 --model_path=data/model_config.csv
```

发送请求：

```bash
./client \
  --server=127.0.0.1:9000 \
  --user_id=alice \
  --items="itemA:0.2,itemB:0.7,itemC:1.5" \
  --top_k=2
```

在浏览器中打开 `http://localhost:9000` 可查看 bRPC 内置监控；`/vars` `/status` 提供更多状态信息。

## 推荐逻辑

Score 公式：

```
score = user_bias(user) + item_bias(item) + candidate.context_weight
```

- `user_bias` / `item_bias` 来自模型配置
- `context_weight` 由客户端传入（可以表示实时特征，如曝光位权重）
- 得分按降序排序，返回 `top_k` 项，同时附带解释

## TODO

- 支持实时特征（例如 request-level JSON）
- 加入 A/B 实验标签、召回路由
- 集成真正的模型推理（例如 ONNXRuntime）
