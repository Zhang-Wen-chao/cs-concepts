"""
B+树搜索模拟：插入数字，按序遍历
"""


class BPlusNode:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []


class BPlusTree:
    def __init__(self, order=4):
        self.order = order  # 阶数
        self.root = BPlusNode(is_leaf=True)

    def insert(self, key):
        root = self.root
        if len(root.keys) == self.order - 1:
            new_root = BPlusNode()
            new_root.children.append(root)
            self._split(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        if node.is_leaf:
            node.keys.append(key)
            node.keys.sort()
        else:
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            if len(node.children[i].keys) == self.order - 1:
                self._split(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def _split(self, parent, i):
        node = parent.children[i]
        mid = self.order // 2
        new_node = BPlusNode(is_leaf=node.is_leaf)
        new_node.keys = node.keys[mid:]
        parent.keys.insert(i, node.keys[mid - 1])
        node.keys = node.keys[:mid - 1]
        if not node.is_leaf:
            new_node.children = node.children[mid:]
            node.children = node.children[:mid]
        parent.children.insert(i + 1, new_node)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if key in node.keys:
            return True
        if node.is_leaf:
            return False
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        return self._search(node.children[i], key)

    def traverse(self):
        result = []
        self._traverse(self.root, result)
        return result

    def _traverse(self, node, result):
        if node.is_leaf:
            result.extend(node.keys)
        else:
            for i, child in enumerate(node.children):
                self._traverse(child, result)

    def print_tree(self):
        print(f"B+树（阶数 {self.order}）")
        self._print(self.root, 0)
        print()

    def _print(self, node, level):
        print("  " * level + f"[{'L' if node.is_leaf else 'I'}] keys={node.keys}")
        if not node.is_leaf:
            for child in node.children:
                self._print(child, level + 1)


if __name__ == "__main__":
    tree = BPlusTree(order=4)
    for key in [3, 7, 1, 9, 5, 2, 8, 4, 6]:
        tree.insert(key)

    tree.print_tree()
    print(f"按序遍历: {tree.traverse()}")
    print(f"搜索 5: {'找到' if tree.search(5) else '未找到'}")
    print(f"搜索 10: {'找到' if tree.search(10) else '未找到'}")
