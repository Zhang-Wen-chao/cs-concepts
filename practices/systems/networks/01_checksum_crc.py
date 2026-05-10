"""
CRC 循环冗余校验模拟
"""
POLY = 0b1011  # CRC-3 多项式 x³ + x + 1


def crc_encode(data):
    """计算 CRC 校验码并附加"""
    data <<= 3  # 左移 3 位
    remainder = data
    for i in range(data.bit_length() - 1, 2, -1):
        if remainder & (1 << i):
            remainder ^= POLY << (i - 3)
    return (data, remainder)  # (编码后的数据, 校验码)


def crc_check(code):
    """检查 CRC 编码是否有效（余数是否为 0）"""
    remainder = code
    for i in range(code.bit_length() - 1, 2, -1):
        if remainder & (1 << i):
            remainder ^= POLY << (i - 3)
    return remainder == 0


def flip_bit(code, bit_pos):
    """翻转指定位（模拟传输错误）"""
    return code ^ (1 << bit_pos)


if __name__ == "__main__":
    data = 0b110101  # 原始数据：53
    print(f"原始数据: {bin(data)} ({data})")
    print(f"多项式:   {bin(POLY)}")

    encoded, crc = crc_encode(data)
    print(f"编码后:   {bin(encoded)}")
    print(f"校验码:   {bin(crc)} ({crc})")
    print(f"CRC 检查: {'通过 ✅' if crc_check(encoded) else '失败 ❌'}")
    print()

    # 模拟错误
    corrupted = flip_bit(encoded, 3)
    print(f"传输中第 3 位翻转: {bin(corrupted)}")
    print(f"CRC 检查: {'通过 ✅' if crc_check(corrupted) else '失败 ❌（检测到错误）'}")
