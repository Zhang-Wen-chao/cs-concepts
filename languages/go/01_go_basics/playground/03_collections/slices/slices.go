package slices

import "fmt"

// Chunk 将切片按固定窗口切分，并复制每个块避免共享底层数组。
func Chunk(nums []int, size int) ([][]int, error) {
	if size <= 0 {
		return nil, fmt.Errorf("size must be > 0")
	}

	if len(nums) == 0 {
		return nil, nil
	}

	chunks := make([][]int, 0, (len(nums)+size-1)/size)
	for i := 0; i < len(nums); i += size {
		end := i + size
		if end > len(nums) {
			end = len(nums)
		}
		block := append([]int(nil), nums[i:end]...) // 拷贝，避免外部修改影响结果
		chunks = append(chunks, block)
	}

	return chunks, nil
}

// MergeCounters 将 delta 中的计数合并至 base，展示 map 的引用语义。
func MergeCounters(base map[string]int, delta map[string]int) map[string]int {
	if base == nil {
		base = make(map[string]int, len(delta))
	}
	for k, v := range delta {
		base[k] += v
	}
	return base
}
