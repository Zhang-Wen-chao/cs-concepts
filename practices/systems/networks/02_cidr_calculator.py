"""
CIDR 子网计算器：IP/掩码 → 网络地址、广播地址、可用主机数
"""
import ipaddress


def cidr_info(cidr):
    net = ipaddress.IPv4Network(cidr, strict=False)
    print(f"CIDR: {cidr}")
    print(f"  网络地址:     {net.network_address}")
    print(f"  广播地址:     {net.broadcast_address}")
    print(f"  子网掩码:     {net.netmask}")
    print(f"  可用主机数:   {net.num_addresses - 2}")
    print(f"  主机范围:     {net.network_address + 1} ~ {net.broadcast_address - 1}")
    print()


def subnet_split(cidr, prefix):
    """将 CIDR 分成 prefix 长度的子网"""
    net = ipaddress.IPv4Network(cidr, strict=False)
    subnets = list(net.subnets(new_prefix=prefix))
    print(f"{cidr} 分成 /{prefix} 子网：{len(subnets)} 个")
    for i, sub in enumerate(subnets[:4]):  # 只显示前 4 个
        print(f"  {i + 1}. {sub}")
    if len(subnets) > 4:
        print(f"  ... 共 {len(subnets)} 个")
    print()


if __name__ == "__main__":
    cidr_info("192.168.1.0/24")
    cidr_info("10.0.0.0/16")
    subnet_split("192.168.1.0/24", 26)  # /24 → /26，4 个子网
