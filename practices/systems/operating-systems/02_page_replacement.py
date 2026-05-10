"""
页面置换算法：FIFO、LRU、OPT 对比缺页率
"""


def fifo(pages, frames):
    """先进先出"""
    cache, faults = [], 0
    for p in pages:
        if p not in cache:
            faults += 1
            if len(cache) >= frames:
                cache.pop(0)
            cache.append(p)
    return faults


def lru(pages, frames):
    """最近最少使用"""
    cache, faults = [], 0
    for p in pages:
        if p in cache:
            cache.remove(p)
            cache.append(p)
        else:
            faults += 1
            if len(cache) >= frames:
                cache.pop(0)
            cache.append(p)
    return faults


def opt(pages, frames):
    """最优置换（未来最远使用的被替换）"""
    cache, faults = [], 0
    for i, p in enumerate(pages):
        if p not in cache:
            faults += 1
            if len(cache) >= frames:
                # 找到未来最远使用的页面
                farthest, idx = -1, -1
                for j, cp in enumerate(cache):
                    try:
                        next_use = pages[i + 1:].index(cp)
                    except ValueError:
                        next_use = float('inf')
                    if next_use > farthest:
                        farthest, idx = next_use, j
                cache.pop(idx)
            cache.append(p)
    return faults


if __name__ == "__main__":
    pages = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]

    print(f"页面序列: {pages}")
    print()

    for frames in [3, 4]:
        print(f"物理帧数 = {frames}")
        f_fifo = fifo(pages, frames)
        f_lru = lru(pages, frames)
        f_opt = opt(pages, frames)
        print(f"  FIFO 缺页数: {f_fifo}  (缺页率 {f_fifo/len(pages):.0%})")
        print(f"  LRU  缺页数: {f_lru}  (缺页率 {f_lru/len(pages):.0%})")
        print(f"  OPT  缺页数: {f_opt}  (缺页率 {f_opt/len(pages):.0%})")

        if frames == 3:
            # Belady 异常：帧越多缺页越多
            f_fifo_4 = fifo(pages, 4)
            if f_fifo_4 > f_fifo:
                print(f"  ⚠ Belady 异常：4 帧 ({f_fifo_4}) 比 3 帧 ({f_fifo}) 缺页还多！")
        print()
