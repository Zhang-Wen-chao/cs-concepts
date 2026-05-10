"""
生产者消费者：threading + 信号量实现有界缓冲区
"""
import threading
import time
import random


class BoundedBuffer:
    def __init__(self, size=5):
        self.buffer = []
        self.size = size
        self.mutex = threading.Lock()
        self.empty = threading.Semaphore(size)  # 空位
        self.full = threading.Semaphore(0)  # 满的数据

    def put(self, item):
        self.empty.acquire()
        with self.mutex:
            self.buffer.append(item)
            print(f"[P] 生产 {item}, 缓冲区: {self.buffer}")
        self.full.release()

    def get(self):
        self.full.acquire()
        with self.mutex:
            item = self.buffer.pop(0)
            print(f"[C] 消费 {item}, 缓冲区: {self.buffer}")
        self.empty.release()
        return item


def producer(buf, n):
    for i in range(n):
        time.sleep(random.uniform(0.1, 0.5))
        buf.put(i)


def consumer(buf, n):
    for _ in range(n):
        time.sleep(random.uniform(0.2, 0.6))
        buf.get()


if __name__ == "__main__":
    buf = BoundedBuffer(size=3)
    n_items = 6

    print(f"有界缓冲区（大小 {buf.size}）：生产者-消费者模拟\n")
    threads = [
        threading.Thread(target=producer, args=(buf, n_items)),
        threading.Thread(target=consumer, args=(buf, n_items)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("\n✅ 完成！无死锁、无竞态。")
    print("信号量保证：")
    print("  empty.acquire() → 缓冲区满时生产者等待")
    print("  full.acquire()  → 缓冲区空时消费者等待")
    print("  mutex            → 互斥写缓冲区")
