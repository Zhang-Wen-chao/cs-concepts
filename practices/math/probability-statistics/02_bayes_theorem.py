"""
贝叶斯定理：疾病检测例子
P(病|阳性) = P(阳性|病) * P(病) / P(阳性)
"""


def bayes_disease(disease_rate, sensitivity, false_positive_rate):
    """
    参数:
        disease_rate: 患病率 P(病)
        sensitivity: 检测灵敏度 P(阳性|病) 
        false_positive_rate: 误报率 P(阳性|无病)
    """
    p_disease = disease_rate
    p_positive_given_disease = sensitivity
    p_positive_given_healthy = false_positive_rate
    p_healthy = 1 - p_disease

    # 全概率公式：P(阳性)
    p_positive = p_positive_given_disease * p_disease + p_positive_given_healthy * p_healthy

    # 贝叶斯公式：P(病|阳性)
    p_disease_given_positive = p_positive_given_disease * p_disease / p_positive

    return p_disease_given_positive


if __name__ == "__main__":
    # 常见场景：患病率 1%，检测灵敏度 99%，误报率 5%
    result = bayes_disease(0.01, 0.99, 0.05)
    print(f"患病率 1%，灵敏度 99%，误报率 5%")
    print(f"→ 阳性后真正患病概率: {result:.2%}")
    print()

    # 敏感性分析：不同患病率的影响
    print(f"{'患病率':>10}  {'P(病|阳性)':>15}")
    print("-" * 28)
    for rate in [0.001, 0.01, 0.1, 0.5]:
        prob = bayes_disease(rate, 0.99, 0.05)
        print(f"{rate:>10.1%}  {prob:>15.2%}")
    print()
    print("启示：即使检测很准，低患病率下阳性结果也未必是真的")
