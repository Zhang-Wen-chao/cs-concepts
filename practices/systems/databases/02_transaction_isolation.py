"""
事务隔离级别：脏读、不可重复读、幻读模拟
"""
import time
import threading


class Database:
    def __init__(self):
        self.data = {"x": 100, "y": 200}
        self.lock = threading.Lock()
        self.pending = {}  # 未提交的修改

    def read(self, key, use_pending=True):
        if use_pending and key in self.pending:
            return self.pending[key]
        return self.data[key]

    def write(self, key, value):
        self.pending[key] = value

    def commit(self):
        with self.lock:
            self.data.update(self.pending)
            self.pending = {}

    def rollback(self):
        self.pending = {}


def demo_dirty_read(db):
    """脏读：事务A修改未提交，事务B读到"""
    print("🔴 脏读演示")

    def txn_a():
        db.write("x", 999)
        time.sleep(0.1)
        # 回滚
        db.rollback()
        print("  事务A: 回滚 (x 恢复 100)")

    def txn_b():
        time.sleep(0.05)
        val = db.read("x")  # 读到未提交的 999
        print(f"  事务B: 读到 x = {val} (脏读！实际应该是 100)")

    threads = [threading.Thread(target=t) for t in [txn_a, txn_b]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"  最终 x = {db.data['x']}\n")


def demo_unrepeatable_read(db):
    """不可重复读：同一事务两次读取结果不同"""
    print("🟡 不可重复读演示")

    def txn_a():
        time.sleep(0.1)
        db.write("x", 50)
        db.commit()
        print("  事务A: 修改 x = 50 并提交")

    def txn_b():
        v1 = db.read("x")
        print(f"  事务B: 第一次读 x = {v1}")
        time.sleep(0.2)
        v2 = db.read("x")  # 事务A 已经修改提交
        print(f"  事务B: 第二次读 x = {v2} (不同了！)")
        if v1 != v2:
            print("  → 不可重复读！")

    threads = [threading.Thread(target=t) for t in [txn_a, txn_b]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print()


if __name__ == "__main__":
    print("事务隔离级别演示\n")
    demo_dirty_read(Database())
    demo_unrepeatable_read(Database())
    print("解决方案：")
    print("  READ_UNCOMMITTED → 脏读、不可重复读、幻读都有可能")
    print("  READ_COMMITTED   → 无脏读")
    print("  REPEATABLE_READ  → 无脏读、无不可重复读")
    print("  SERIALIZABLE     → 全都避免了（性能代价最大）")
