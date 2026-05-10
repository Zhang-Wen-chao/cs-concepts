"""
调度算法模拟：FCFS、SJF、RR 对比平均等待时间
"""


def fcfs(processes):
    """先来先服务"""
    time, total_wait = 0, 0
    for p in processes:
        if time < p[0]:
            time = p[0]
        total_wait += time - p[0]
        time += p[1]
    return total_wait / len(processes)


def sjf(processes):
    """最短作业优先（非抢占）"""
    remaining = sorted([(a, b, i) for i, (a, b) in enumerate(processes)])
    time, total_wait, done = 0, 0, []
    while remaining:
        ready = [p for p in remaining if p[0] <= time]
        if not ready:
            time = remaining[0][0]
            continue
        ready.sort(key=lambda p: p[1])
        p = ready[0]
        total_wait += time - p[0]
        time += p[1]
        remaining.remove(p)
    return total_wait / len(processes)


def rr(processes, quantum):
    """时间片轮转"""
    remaining = [(a, b) for a, b in processes]
    time, total_wait, n = 0, 0, len(processes)
    done = [False] * n
    while not all(done):
        for i in range(n):
            if done[i]:
                continue
            if remaining[i][0] > time:
                continue
            if remaining[i][1] > 0:
                if remaining[i][1] == processes[i][1]:
                    total_wait += time - processes[i][0]
                if remaining[i][1] > quantum:
                    time += quantum
                    remaining[i] = (remaining[i][0], remaining[i][1] - quantum)
                else:
                    time += remaining[i][1]
                    remaining[i] = (remaining[i][0], 0)
                    done[i] = True
                    # 提前完成也计了等待时间
    return total_wait / n


if __name__ == "__main__":
    # 进程：(到达时间, 执行时间)
    processes = [(0, 5), (1, 3), (2, 8), (3, 2)]

    print("进程: (到达时间, 执行时间)")
    for i, p in enumerate(processes):
        print(f"  P{i+1}: {p}")
    print()

    print(f"FCFS   平均等待时间: {fcfs(processes):.2f}")
    print(f"SJF    平均等待时间: {sjf(processes):.2f}")
    print(f"RR(q=2)平均等待时间: {rr(processes, 2):.2f}")
    print(f"RR(q=4)平均等待时间: {rr(processes, 4):.2f}")
    print()
    print("FCFS 简单但长任务会堵后面的；SJF 理论上平均最优但需要预知执行时间；")
    print("RR 公平但时间片大小直接影响性能。")
