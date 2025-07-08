import sys

def main():
    """主函数，处理所有逻辑"""

    # 1. 提取出真正的参数，忽略掉脚本名本身 (sys.argv[0])
    args = sys.argv[1:]

    # --- 处理特殊模式 ---
    # 帮助模式: 如果用户输入了 --help，就打印帮助信息并退出
    if "--help" in args:
        print("用法: python calculator.py [选项] [数字1] [数字2] ...")
        print("\n这是一个简单的加法计算器，它会将所有提供的数字相加。")
        print("\n选项:")
        print("  --help     显示此帮助信息并退出。")
        print("  --verbose  运行时显示详细的计算步骤。")
        # 使用 sys.exit() 可以提前终止脚本
        sys.exit(0) 
    
    # 详细模式: 检查 --verbose 是否存在，并将其从参数列表中移除
    # 这样它就不会被当作数字来处理了
    verbose_mode = False
    if "--verbose" in args:
        verbose_mode = True
        args.remove("--verbose") # 移除它，剩下的就都是数字了

    # --- 核心计算逻辑 ---
    # 如果没有提供任何数字，提示用户
    if not args:
        print("错误：请输入至少一个数字进行计算。")
        print("使用 --help 查看帮助。")
        sys.exit(1) # 用非0状态码退出，表示出错了

    # 开始计算
    total = 0.0
    try:
        for number_str in args:
            num = float(number_str) # 将字符串转换为浮点数
            total += num
            if verbose_mode:
                print(f"  加上 {num}，当前总和是 {total}")
    except ValueError:
        # 如果用户输入了非数字（比如 "abc"），float()会抛出ValueError
        print(f"错误：'{number_str}' 不是一个有效的数字！")
        sys.exit(1)

    # 打印最终结果
    print("-" * 20)
    print(f"计算结果是: {total}")

# 这是一个好习惯，确保只有在直接运行此脚本时才执行main()
if __name__ == "__main__":
    main()