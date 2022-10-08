package app

func nextPowOf2(n int) int {
	k := 1
	for k < n {
		k = k << 1
	}
	return k
}
