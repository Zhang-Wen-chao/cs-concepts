package transform // 泛型容器/算法示例：Map/Filter/Keys

/*
Usage:

	cd languages/go/01_go_basics/playground/06_generics
	go fmt ./...
	go test ./...
	go test . -run TestMapSlice
*/

// MapSlice 将切片映射为另一类型。
// 签名：func MapSlice[T any, R any](in []T, fn func(T) R) []R
//   - 类型参数：[T any, R any] 声明输入/输出类型；any 等同于 interface{}。
//   - 参数：in 是输入切片，fn 是对每个元素执行的函数。
//   - 返回：与输入等长的切片，元素类型为 R。
func MapSlice[T any, R any](in []T, fn func(T) R) []R {
	out := make([]R, len(in))
	for i, v := range in {
		out[i] = fn(v) // 调用传入的函数，按顺序写回
	}
	return out
}

// Filter 返回满足 predicate 的元素。
// 签名：func Filter[T any](in []T, keep func(T) bool) []T —— 参数是输入切片 + 判定函数，返回被保留的新切片。
func Filter[T any](in []T, keep func(T) bool) []T {
	out := make([]T, 0, len(in))
	for _, v := range in {
		// keep 是 predicate：返回 true 时表示元素应该被保留。
		if keep(v) {
			out = append(out, v)
		}
	}
	return out
}

// Keys 抽取 map 的所有键，演示 comparable 约束：只有可比较的类型才能当作 map key。
// 签名：func Keys[K comparable, V any](m map[K]V) []K —— 传入 map，取出所有键，返回顺序未定义的切片。
func Keys[K comparable, V any](m map[K]V) []K {
	out := make([]K, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
