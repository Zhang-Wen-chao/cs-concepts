# 智能指针自测（问题版）

> **用法**：盖住下面的内容，自己口头或纸面回答。  
> 答不出来 = 这个点没真懂，回去看 `01_static_typing.md` 的 1.7 节 + 重写 `practices/02_smart_pointers.cpp`。  
> 答得清楚 = 在 `01_static_typing.md` 末尾的 checklist 勾掉 1.7。

## 速记口诀（Aaron 自创, 2026-06-25）

- **unique_ptr**: 一人一物, 独占
- **shared_ptr**: 共享本本记着数 (`use_count`)
- **weak_ptr**: 只看不拿, 防循环引用

回答不上来时先念口诀, 再回头看机制。

---

## unique_ptr

1. `unique_ptr` 的所有权模型是什么？一句话讲完。
2. `unique_ptr` 能不能复制？为什么？
3. 想把 `unique_ptr` 的所有权交给别人，**该用什么操作**？之后原指针还能用吗？
4. `unique_ptr` 的开销跟裸指针比怎么样？什么时候**应该**默认用 `unique_ptr` 而不是 `unique_ptr`？
5. `std::make_unique<int>(42)` 和 `std::unique_ptr<int>(new int(42))` 哪个好？为什么？

## shared_ptr

6. `shared_ptr` 的所有权模型是什么？什么计数决定对象什么时候被销毁？
7. 三个 `shared_ptr` 指向同一个对象，强引用计数是几？其中一个死了，剩几？
8. `shared_ptr` 内部那个"小本本"叫什么？里面至少存了哪几样东西？
9. 跟 `unique_ptr` 比，`shared_ptr` 多出来的开销主要是什么？
10. `std::make_shared<T>()` 比 `shared_ptr<T>(new T)` 强在哪？（答出"几次堆分配"和"缓存友好"两点即可）

## weak_ptr

11. `weak_ptr` 增加**哪种**引用计数？增加还是不增加？
12. 用 `weak_ptr` 之前**必须**做哪一步？API 叫什么名字？
13. lock 之后可能拿不到对象，可能有哪两种情况？
14. 说出 `weak_ptr` 存在的**唯一**核心原因。（一句话）
15. 用链表 `Node { shared_ptr<Node> next; shared_ptr<Node> prev; }` 举例，为什么会内存泄漏？把 `prev` 改成 `weak_ptr` 之后为什么就修好了？（画图或写数字说明引用计数变化）

## 综合应用

16. 写一个 `unique_ptr` move 的 3 行代码，move 之后打印原指针，应该**不崩**也**不打**出值。
17. 写两个 `shared_ptr` 指向同一 `int`，在两个地方分别打印 `.use_count()`，应该都是 2。
18. 写一个最小可复现的循环引用例子（不用 Node，用两个 struct 互相 `shared_ptr` 即可），加 `-fsanitize=address` 编译并跑，确认 sanitizer 报泄漏。然后把其中一个改成 `weak_ptr`，确认不报了。
