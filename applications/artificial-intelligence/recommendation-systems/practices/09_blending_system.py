"""
混排（Blending）- Python实现
推荐系统的最后一环：广告穿插、运营位、低质过滤

作者: Zhang Wenchao
日期: 2025-11-22

====================================================================
📖 混排在推荐链路中的位置
====================================================================

完整链路：
召回 → 粗排 → 精排 → 重排 → 混排 → 展示

混排是推荐系统的"最后一公里"：
- 重排输出：高质量的推荐列表
- 混排输出：用户最终看到的内容

混排的任务：
1️⃣ 广告穿插：在推荐内容中插入广告
2️⃣ 运营位插入：热门活动、运营内容
3️⃣ 低质过滤：过滤违规、低质、重复内容
4️⃣ 个性化调整：VIP用户减少广告
5️⃣ 频控：限制同一内容/广告的曝光频次

====================================================================
🎯 混排的核心思想
====================================================================

平衡三方利益：
- 用户体验：不要太多广告
- 平台收益：广告要有曝光
- 内容质量：保证推荐内容的质量

混排策略：

1️⃣ 广告穿插策略
   - 固定位置：第3、8、15位插广告
   - 动态位置：根据用户等级调整
   - 广告质量：eCPM（期望收益）排序

2️⃣ 运营位策略
   - 热门活动：置顶或固定位置
   - 新人专区：针对新用户
   - 时间敏感：限时活动优先展示

3️⃣ 低质过滤
   - 标题党、低质内容
   - 重复内容（7天内看过的）
   - 违规内容（敏感词、虚假信息）

4️⃣ 频控策略
   - 用户维度：同一广告 1天最多3次
   - 物品维度：同一商品 1天最多2次
   - 类别维度：同类商品不超过5个

====================================================================
🏗️ 本实现：完整的混排系统
====================================================================

组件：
1. ContentItem：推荐内容
2. AdItem：广告内容
3. OperationItem：运营内容
4. FrequencyController：频控器
5. QualityFilter：质量过滤器
6. BlendingEngine：混排引擎

混排流程：
重排结果 → 频控 → 低质过滤 → 广告穿插 → 运营位插入 → 最终列表

====================================================================
📊 混排评价指标
====================================================================

用户体验：
- 广告比例：< 20%
- 内容质量：平均分数 > 阈值
- 多样性：类别覆盖度

平台收益：
- 广告曝光量
- 预期收益（eCPM × 曝光量）
- 点击率（CTR）

整体指标：
- 用户留存率
- 停留时长
- DAU（日活跃用户）

====================================================================
"""

import numpy as np
from typing import List, Dict, Set
from collections import defaultdict
from datetime import datetime, timedelta

# 设置随机种子
np.random.seed(42)


# ============ 1. 内容类型定义 ============

class ContentItem:
    """推荐内容"""

    def __init__(self, item_id, score, category, title, author, quality_score):
        self.item_id = item_id
        self.score = score  # 精排/重排分数
        self.category = category
        self.title = title
        self.author = author
        self.quality_score = quality_score  # 质量分（0-1）
        self.type = 'content'

    def __repr__(self):
        return f"Content(ID={self.item_id}, score={self.score:.3f}, cat={self.category})"


class AdItem:
    """广告内容"""

    def __init__(self, ad_id, ecpm, category, title, ad_quality):
        self.ad_id = ad_id
        self.ecpm = ecpm  # 期望收益（元）
        self.category = category
        self.title = title
        self.ad_quality = ad_quality  # 广告质量（0-1）
        self.type = 'ad'

    def __repr__(self):
        return f"Ad(ID={self.ad_id}, eCPM={self.ecpm:.2f})"


class OperationItem:
    """运营内容"""

    def __init__(self, op_id, priority, category, title, op_type):
        self.op_id = op_id
        self.priority = priority  # 优先级（越高越重要）
        self.category = category
        self.title = title
        self.op_type = op_type  # 活动类型：'hot'/'new_user'/'limited_time'
        self.type = 'operation'

    def __repr__(self):
        return f"Operation(ID={self.op_id}, priority={self.priority}, type={self.op_type})"


# ============ 2. 频控器 ============

class FrequencyController:
    """频次控制器

    功能：
    - 用户维度：同一内容/广告不重复曝光
    - 时间维度：7天内看过的内容不再推荐
    - 类别维度：同类内容不超过阈值
    """

    def __init__(self):
        # 用户历史：{user_id: {item_id: last_seen_time}}
        self.user_history = defaultdict(dict)

        # 广告曝光：{user_id: {ad_id: count_today}}
        self.ad_exposure = defaultdict(lambda: defaultdict(int))

    def check_and_update(self, user_id, item, current_time=None):
        """
        检查是否通过频控，并更新记录

        返回:
            True: 通过频控
            False: 不通过（需要过滤）
        """
        if current_time is None:
            current_time = datetime.now()

        if item.type == 'content':
            # 内容频控：7天内看过的不推荐
            item_id = item.item_id
            if item_id in self.user_history[user_id]:
                last_seen = self.user_history[user_id][item_id]
                if (current_time - last_seen).days < 7:
                    return False

            # 更新历史
            self.user_history[user_id][item_id] = current_time
            return True

        elif item.type == 'ad':
            # 广告频控：同一广告1天最多3次
            ad_id = item.ad_id
            if self.ad_exposure[user_id][ad_id] >= 3:
                return False

            # 更新曝光
            self.ad_exposure[user_id][ad_id] += 1
            return True

        else:
            # 运营内容：不频控
            return True

    def reset_daily(self):
        """每天重置广告曝光计数"""
        self.ad_exposure.clear()


# ============ 3. 质量过滤器 ============

class QualityFilter:
    """质量过滤器

    功能：
    - 低质内容过滤
    - 标题党检测
    - 违规内容过滤
    """

    def __init__(self, quality_threshold=0.5):
        self.quality_threshold = quality_threshold

        # 违规关键词（示例）
        self.blacklist_keywords = ['标题党', '震惊', '违规', '低俗']

    def filter(self, item):
        """
        检查是否通过质量过滤

        返回:
            True: 通过
            False: 不通过（需要过滤）
        """
        if item.type == 'content':
            # 质量分过滤
            if item.quality_score < self.quality_threshold:
                return False

            # 标题检测
            if any(keyword in item.title for keyword in self.blacklist_keywords):
                return False

        elif item.type == 'ad':
            # 广告质量过滤
            if item.ad_quality < 0.6:
                return False

        return True


# ============ 4. 混排引擎 ============

class BlendingEngine:
    """混排引擎

    功能：
    - 广告穿插
    - 运营位插入
    - 频控和质量过滤
    """

    def __init__(self, user_id, user_level='normal'):
        self.user_id = user_id
        self.user_level = user_level  # 'vip' / 'normal' / 'new'

        self.freq_controller = FrequencyController()
        self.quality_filter = QualityFilter(quality_threshold=0.5)

        # 广告位配置（根据用户等级）
        if user_level == 'vip':
            self.ad_positions = [10, 20]  # VIP用户少广告
        elif user_level == 'normal':
            self.ad_positions = [3, 8, 15, 23]  # 普通用户
        else:  # new
            self.ad_positions = [5, 12, 20]  # 新用户

        # 运营位配置
        self.operation_positions = [0] if user_level == 'new' else [0, 10]

    def blend(self, contents: List[ContentItem], ads: List[AdItem],
              operations: List[OperationItem], target_size=20):
        """
        混排主函数

        参数:
            contents: 重排后的推荐内容
            ads: 候选广告（已按eCPM排序）
            operations: 运营内容
            target_size: 目标列表长度

        返回:
            blended_list: 混排后的列表
        """
        # 1. 质量过滤
        contents = [item for item in contents if self.quality_filter.filter(item)]
        ads = [item for item in ads if self.quality_filter.filter(item)]

        # 2. 频控
        contents = [
            item for item in contents
            if self.freq_controller.check_and_update(self.user_id, item)
        ]
        ads = [
            item for item in ads
            if self.freq_controller.check_and_update(self.user_id, item)
        ]

        # 3. 初始化混排列表
        blended_list = []

        # 4. 运营位插入（优先级最高）
        operation_map = {}
        for pos in self.operation_positions:
            if operations:
                # 选择优先级最高的运营内容
                op = max(operations, key=lambda x: x.priority)
                operation_map[pos] = op
                operations.remove(op)

        # 5. 混排：内容 + 广告
        content_idx = 0
        ad_idx = 0

        for pos in range(target_size):
            # 运营位
            if pos in operation_map:
                blended_list.append(operation_map[pos])
                continue

            # 广告位
            if pos in self.ad_positions and ad_idx < len(ads):
                blended_list.append(ads[ad_idx])
                ad_idx += 1
                continue

            # 内容位
            if content_idx < len(contents):
                blended_list.append(contents[content_idx])
                content_idx += 1
            else:
                # 内容不足，填充广告
                if ad_idx < len(ads):
                    blended_list.append(ads[ad_idx])
                    ad_idx += 1

        return blended_list


# ============ 5. 评估指标 ============

def calculate_ad_ratio(blended_list):
    """计算广告比例"""
    ad_count = sum(1 for item in blended_list if item.type == 'ad')
    return ad_count / len(blended_list) if blended_list else 0


def calculate_expected_revenue(blended_list):
    """计算预期收益"""
    revenue = sum(item.ecpm for item in blended_list if item.type == 'ad')
    return revenue


def calculate_content_quality(blended_list):
    """计算内容质量（平均分数）"""
    content_items = [item for item in blended_list if item.type == 'content']
    if not content_items:
        return 0
    return np.mean([item.score for item in content_items])


# ============ 6. 模拟数据生成 ============

def generate_test_data():
    """生成测试数据"""
    # 生成推荐内容
    contents = []
    for i in range(50):
        item = ContentItem(
            item_id=i,
            score=np.random.uniform(0.5, 0.95),
            category=np.random.randint(0, 10),
            title=f"推荐内容_{i}",
            author=f"作者_{np.random.randint(0, 20)}",
            quality_score=np.random.uniform(0.3, 1.0)
        )
        contents.append(item)
    contents.sort(key=lambda x: x.score, reverse=True)

    # 生成广告
    ads = []
    for i in range(20):
        ad = AdItem(
            ad_id=i,
            ecpm=np.random.uniform(0.5, 5.0),
            category=np.random.randint(0, 5),
            title=f"广告_{i}",
            ad_quality=np.random.uniform(0.4, 1.0)
        )
        ads.append(ad)
    ads.sort(key=lambda x: x.ecpm, reverse=True)

    # 生成运营内容
    operations = [
        OperationItem(0, priority=10, category=0, title="热门活动", op_type='hot'),
        OperationItem(1, priority=8, category=1, title="新人专区", op_type='new_user'),
        OperationItem(2, priority=6, category=2, title="限时优惠", op_type='limited_time')
    ]

    return contents, ads, operations


# ============ 7. 对比实验 ============

def compare_user_levels():
    """对比不同用户等级的混排结果"""
    contents, ads, operations = generate_test_data()

    print("\n" + "=" * 60)
    print("不同用户等级的混排结果对比")
    print("=" * 60)

    user_levels = ['vip', 'normal', 'new']

    for level in user_levels:
        engine = BlendingEngine(user_id=f"user_{level}", user_level=level)
        blended = engine.blend(contents, ads, operations, target_size=20)

        ad_ratio = calculate_ad_ratio(blended)
        revenue = calculate_expected_revenue(blended)
        quality = calculate_content_quality(blended)

        print(f"\n{level.upper()} 用户:")
        print(f"  广告比例: {ad_ratio:.2%}")
        print(f"  预期收益: ¥{revenue:.2f}")
        print(f"  内容质量: {quality:.3f}")

        print(f"  混排结果（前10）:")
        for i, item in enumerate(blended[:10]):
            print(f"    {i+1}. {item}")


# ============ 主函数 ============

def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  混排（Blending）- Python实现")
    print("  推荐系统的最后一环")
    print("=" * 60)

    # 生成测试数据
    print("\n" + "=" * 60)
    print("生成测试数据")
    print("=" * 60)

    contents, ads, operations = generate_test_data()

    print(f"推荐内容: {len(contents)} 个")
    print(f"候选广告: {len(ads)} 个")
    print(f"运营内容: {len(operations)} 个")

    # 对比不同用户等级
    compare_user_levels()

    # 详细展示一个案例
    print("\n" + "=" * 60)
    print("详细案例：普通用户的混排过程")
    print("=" * 60)

    engine = BlendingEngine(user_id="user_normal", user_level='normal')
    blended = engine.blend(contents, ads, operations, target_size=25)

    print(f"\n混排后的列表（共{len(blended)}项）:")
    for i, item in enumerate(blended):
        if item.type == 'content':
            print(f"  {i+1}. [内容] {item.title} (分数={item.score:.3f}, 质量={item.quality_score:.2f})")
        elif item.type == 'ad':
            print(f"  {i+1}. [广告] {item.title} (eCPM=¥{item.ecpm:.2f})")
        else:
            print(f"  {i+1}. [运营] {item.title} (优先级={item.priority})")

    # 统计
    print(f"\n统计信息:")
    print(f"  广告比例: {calculate_ad_ratio(blended):.2%}")
    print(f"  预期收益: ¥{calculate_expected_revenue(blended):.2f}")
    print(f"  内容质量: {calculate_content_quality(blended):.3f}")

    content_count = sum(1 for item in blended if item.type == 'content')
    ad_count = sum(1 for item in blended if item.type == 'ad')
    op_count = sum(1 for item in blended if item.type == 'operation')

    print(f"  内容: {content_count}, 广告: {ad_count}, 运营: {op_count}")

    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. 混排的核心任务
   ✓ 广告穿插：在推荐内容中插入广告
   ✓ 运营位插入：热门活动、运营内容
   ✓ 低质过滤：过滤违规、低质内容
   ✓ 频控：限制同一内容/广告的曝光频次

2. 混排策略
   ✓ 固定位置策略：第3、8、15位插广告
   ✓ 用户等级策略：VIP减少广告
   ✓ 质量优先策略：低质内容过滤

3. 三方平衡
   ✓ 用户体验：广告比例 < 20%
   ✓ 平台收益：广告曝光 × eCPM
   ✓ 内容质量：平均质量分 > 阈值

4. 关键组件
   ✓ 频控器：防止重复曝光
   ✓ 质量过滤器：过滤低质内容
   ✓ 混排引擎：执行混排逻辑

5. 工业实践
   ✓ AB测试：测试不同广告比例
   ✓ 实时调整：根据用户反馈动态调整
   ✓ 个性化：不同用户不同策略

6. 推荐系统完整链路
   ✅ 召回：双塔模型（百万 → 几千）
   ✅ 粗排：轻量级模型（几千 → 几百）
   ✅ 精排：DIN、DeepFM（几百 → 几十）
   ✅ 重排：MMR、打散（优化多样性）
   ✅ 混排：广告穿插、运营位（最终展示）

7. 业务价值
   ✓ 优化用户体验（多样性、质量）
   ✓ 提升平台收益（广告收入）
   ✓ 平衡短期和长期目标
    """)

    print("\n🎉 恭喜！推荐系统完整链路学习完成！")
    print("\n你已经掌握了：")
    print("  ✅ 召回：双塔模型")
    print("  ✅ 粗排：轻量级模型")
    print("  ✅ 精排：Wide & Deep、DeepFM、DIN")
    print("  ✅ 多任务学习：CTR + CVR")
    print("  ✅ 重排：MMR、打散策略")
    print("  ✅ 混排：广告穿插、质量控制")
    print("\n这些是工业界推荐系统的核心技术栈！")


if __name__ == "__main__":
    main()
