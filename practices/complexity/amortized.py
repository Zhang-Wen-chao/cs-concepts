"""
摊销分析 - Amortized Analysis

动态数组的 append：单次最坏 O(n)，平均 O(1)。
"""


class DynamicArray:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.arr = [None] * self.capacity

    def append(self, item):
        if self.size == self.capacity:
            self._resize()
        self.arr[self.size] = item
        self.size += 1

    def _resize(self):
        self.capacity *= 2
        new_arr = [None] * self.capacity
        for i in range(self.size):
            new_arr[i] = self.arr[i]
        self.arr = new_arr

    def __len__(self):
        return self.size


def simulate(n):
    """模拟 n 次 append，统计每次的耗时分布"""
    da = DynamicArray()
    costs = []
    for i in range(n):
        before = da.capacity
        da.append(i)
        if da.arr is not None and da.capacity != before:
            costs.append(("resize", i + 1, da.capacity))
        else:
            costs.append(("append", i + 1, None))

    return costs


if __name__ == "__main__":
    n = 16
    costs = simulate(n)
    print(f"动态数组 {n} 次 append 的摊销分析:")
    print("-" * 40)
    for kind, idx, new_cap in costs:
        if kind == "resize":
            print(f"  [{idx:3d}] ⚡ 扩容至 {new_cap:3d}: 复制 {new_cap // 2} 个元素 O(n)")
        else:
            print(f"  [{idx:3d}]   直接追加 O(1)")

    print()
    print(f"总 append: {n} 次")
    resize_count = sum(1 for k, _, _ in costs if k == "resize")
    print(f"总扩容: {resize_count} 次 (log₂{n} ≈ {resize_count})")
    print(f"每次 append 摊销成本: O(1)")
