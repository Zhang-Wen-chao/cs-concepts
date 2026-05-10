"""
缓存模拟：直接映射 vs 全关联 vs 组关联，对比命中率
"""
import random


class CacheLine:
    def __init__(self):
        self.tag = None
        self.valid = False


def direct_mapped(accesses, num_lines=8):
    """直接映射：每个内存地址映射到固定行"""
    cache = [CacheLine() for _ in range(num_lines)]
    hits = 0
    for addr in accesses:
        idx = addr % num_lines
        tag = addr // num_lines
        if cache[idx].valid and cache[idx].tag == tag:
            hits += 1
        else:
            cache[idx].tag = tag
            cache[idx].valid = True
    return hits / len(accesses)


def fully_associative_lru(accesses, num_lines=8):
    """全关联 LRU：任意地址可放任意行，最近最少使用"""
    cache = []
    hits = 0
    for addr in accesses:
        if addr in cache:
            hits += 1
            cache.remove(addr)
            cache.append(addr)
        else:
            if len(cache) >= num_lines:
                cache.pop(0)
            cache.append(addr)
    return hits / len(accesses)


def set_associative(accesses, num_sets=4, ways=2):
    """组关联：每组 ways 行，组内 LRU"""
    cache = [[] for _ in range(num_sets)]
    hits = 0
    for addr in accesses:
        s = addr % num_sets
        if addr in cache[s]:
            hits += 1
            cache[s].remove(addr)
            cache[s].append(addr)
        else:
            if len(cache[s]) >= ways:
                cache[s].pop(0)
            cache[s].append(addr)
    return hits / len(accesses)


if __name__ == "__main__":
    random.seed(42)
    # 模拟具有局部性的内存访问模式
    accesses = [random.randint(0, 63) for _ in range(200)]

    print("缓存映射策略命中率对比（200 次访问，8 行缓存）")
    print("-" * 45)
    print(f"  直接映射 (8行):   {direct_mapped(accesses):.1%}")
    print(f"  全关联 LRU (8行): {fully_associative_lru(accesses):.1%}")
    print(f"  组关联 (4组×2路): {set_associative(accesses, 4, 2):.1%}")
    print()
    print("全关联命中率最高但硬件成本最大，直接映射最简单但冲突最多。")
    print("组关联是实际 CPU 中最常用的折衷方案。")
