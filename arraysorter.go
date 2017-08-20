package liblinear

import "math"

func reversedMergeSortDefault(a []float64) {
	reversedMergeSort(a, 0, len(a))
}

func reversedMergeSort(x []float64, off int, len int) {
	// Insertion sort on smallest array
	if len < 7 {
		for i := off; i < len+off; i++ {
			for j := i; j > off && x[j-1] < x[j]; j-- {
				swapFloat64Array(x, j, j-1)
			}
		}
		return
	}

	m := off + (len >> 1)
	if len > 7 {
		l := off
		n := off + len - 1
		if len > 40 { // Big arrays, pseudomedian of 9
			s := len / 8
			l = med3(x, l, l+s, l+2*s)
			m = med3(x, m-s, m, m+s)
			n = med3(x, n-2*s, n-s, n)
		}
		m = med3(x, l, m, n)
	}
	v := x[m]

	//Establish invariant: v* (<v)* (>v)* v*
	a := off
	b := a
	c := off + len - 1
	d := c

	for {
		for b <= c && x[b] >= v {
			if x[b] == v {
				swapFloat64Array(x, a, b)
				a++
			}
			b++
		}
		for c >= b && x[c] <= v {
			if x[c] == v {
				swapFloat64Array(x, c, d)
				d--
			}
			c--
		}
		if b > c {
			break
		}
		swapFloat64Array(x, b, c)
		b++
		c--
	}

	var s, n int = 0, off + len
	s = int(math.Min(float64(a-off), float64(b-a)))
	vecswap(x, off, b-s, s)
	s = int(math.Min(float64(d-c), float64(n-d-1)))
	vecswap(x, b, n-s, s)

	// Recursively sort non-partition elements
	s = b - a
	if s > 1 {
		reversedMergeSort(x, off, s)
	}
	s = d - c
	if s > 1 {
		reversedMergeSort(x, n-s, s)
	}

}

func med3(x []float64, a int, b int, c int) int {

	if lessThan(x[a], x[b]) {
		if lessThan(x[b], x[c]) {
			return b
		}

		if lessThan(x[a], x[c]) {
			return c
		}

		return a
	}

	if greaterThan(x[b], x[c]) {
		return b
	}

	if greaterThan(x[a], x[c]) {
		return c
	}

	return a
}

/**
 * Swaps x[a .. (a+n-1)] with x[b .. (b+n-1)].
 */
func vecswap(x []float64, a int, b int, n int) {
	for i := 0; i < n; i, a, b = i+1, a+1, b+1 {
		swapFloat64Array(x, a, b)
	}
}

func greaterThan(a float64, b float64) bool {
	if a > b {
		return true
	}
	return false
}

func lessThan(a float64, b float64) bool {
	if a < b {
		return true
	}
	return false
}
