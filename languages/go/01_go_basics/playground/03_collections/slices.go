package slices // 演示“切片 + map”最常用的工具函数

/*
Usage:

	# 进入 playground 目录后运行
	go fmt ./...                 # 统一风格
	go test ./...                # 跑完全部表驱动测试
	go test . -run TestChunk     # 只跑 Chunk 相关测试
	go test . -run TestMergeCounters
*/

import "fmt"

// Chunk 演示如何把切片按固定窗口切分。
// 参数说明：
//   - nums: 输入切片
//   - size: 每个块的最大长度（必须 >0）
//
// 返回：
//   - 切片的切片（[][]int），每个子切片都是新复制的，外部修改不影响原数据
//   - 错误信息（size<=0 时触发）
//
// 语法提示（Go 与 C++/Java 写法不同）：
//
//	func Chunk(nums []int, size int) ([][]int, error)
//	├─ 第一对括号 -> 参数列表；nums 类型是 []int（切片），size 是 int。
//	└─ 第二对括号 -> 返回值列表；可以一次返回两个结果（[][]int 与 error）。
func Chunk(nums []int, size int) ([][]int, error) {
	if size <= 0 {
		// fmt.Errorf 结合 %w 可向上传播错误；本例只需简单错误即可。
		return nil, fmt.Errorf("size must be > 0")
	}

	if len(nums) == 0 {
		return nil, nil
	}

	// make([][]int, 0, capacity)：创建一个长度为 0 的切片，但提前申请好容量，避免频繁扩容。
	// (len(nums)+size-1)/size 等价于 math.Ceil(len/size)，推测最多需要多少块。
	chunks := make([][]int, 0, (len(nums)+size-1)/size)
	for i := 0; i < len(nums); i += size {
		end := i + size
		if end > len(nums) {
			end = len(nums)
		}
		// append([]int(nil), ...)：通过把 nil 切片作为起点，强制复制一份底层数组，防止外部修改影响结果。
		block := append([]int(nil), nums[i:end]...)
		chunks = append(chunks, block)
	}

	return chunks, nil
}

// MergeCounters 将 delta 中的计数合并至 base，展示 map 的引用语义（按引用传递，不会复制底层数据）。
// 签名：func MergeCounters(base map[string]int, delta map[string]int) map[string]int
//   - 参数：base/delta 都是 map[string]int；传入 nil 的 base 也会被函数初始化。
//   - 返回：合并后的 map（与 base 共享底层存储），调用方可继续读取或写入。
func MergeCounters(base map[string]int, delta map[string]int) map[string]int {
	if base == nil {
		// nil map 不能写入，因此一旦发现 base 为 nil 就先 make 一个。
		base = make(map[string]int, len(delta))
	}
	for k, v := range delta {
		// map 默认零值是 0，因此可以直接 += 而无需判断 key 是否存在。
		base[k] += v
	}
	return base
}
