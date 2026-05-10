"""
流水线冒险演示：数据冒险和控制冒险
"""
import time


def simulate_no_hazard():
    """无冒险的理想流水线"""
    print("=== 理想流水线（无冒险）===")
    for cycle in range(1, 8):
        stages = []
        for offset in range(5):
            instr = cycle - offset
            stage_names = ["IF", "ID", "EX", "MEM", "WB"]
            if 1 <= instr <= 4:
                stages.append(f"I{instr}({stage_names[offset]})")
        print(f"  Cycle {cycle}:  {'  '.join(stages)}")
    print("  CPI = 1.0（理想）\n")


def simulate_data_hazard():
    """数据冒险（RAW）：ADD R1, R2, R3 后 SUB R4, R1, R5"""
    print("=== 数据冒险（RAW）：ADD -> SUB 读 R1 ===")
    print("  无转发：插入气泡（stall）")
    for cycle in range(1, 10):
        if cycle <= 5:
            stages = []
            for offset in range(5):
                instr = cycle - offset
                if instr == 1:
                    stg = ["IF", "ID", "EX", "MEM", "WB"][offset]
                    stages.append(f"ADD({stg})")
                elif instr == 2:
                    stg = ["IF", "ID", "BUBBLE", "BUBBLE", "EX"][offset] if cycle > 3 else ["IF", "ID", "ID", "BUBBLE", "BUBBLE"][offset]
                    stages.append(f"SUB({stg})")
            if stages:
                print(f"  Cycle {cycle}:  {'  '.join(stages)}")
    print("  代价：额外 2 个气泡周期\n")


def simulate_control_hazard():
    """控制冒险：分支预测失败"""
    print("=== 控制冒险（分支预测失败）===")
    print("  预测不跳转 → 实际跳转 → 冲刷流水线")
    for cycle in range(1, 9):
        if cycle <= 4:
            stages = [f"BEQ(IF)", f"BEQ(ID)", f"BEQ(EX)", f"FLUSH", f"TARG(IF)"][:cycle]
            print(f"  Cycle {cycle}:  {'  '.join(stages)}")
    print("  代价：冲刷 2~3 个阶段\n")


if __name__ == "__main__":
    simulate_no_hazard()
    simulate_data_hazard()
    simulate_control_hazard()
    print("缓解方案：")
    print("  - 数据冒险：转发（Forwarding），直接将 EX 结果送到 ID")
    print("  - 控制冒险：分支预测 + 延迟槽")
