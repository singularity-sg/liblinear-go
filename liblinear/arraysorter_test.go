package liblinear

import (
	"math/rand"
	"testing"
)

var r = rand.New(rand.NewSource(0))

func assertDescendingOrder(t *testing.T, array []float64) {
	before := array[0]
	for _, d := range array {
		if d == 0.0 && before == -0.0 {
			continue
		}

		if d > before {
			t.Fail()
		}

		before = d
	}
}

func shuffleArray(array []float64) {
	for i := 0; i < len(array); i++ {
		j := r.Intn(len(array))
		swapFloat64Array(array, i, j)
	}
}

func TestReversedMergesort(t *testing.T) {

	for k := 1; k <= 16*8096; k *= 2 {
		array := make([]float64, k)
		for i := 0; i < len(array); i++ {
			array[i] = r.Float64()
		}

		reversedMergeSortDefault(array)
		assertDescendingOrder(t, array)
	}

}

func TestReversedMergesortWithMeanValues(t *testing.T) {
	array := []float64{1.0, -0.0, -1.1, 2.0, 3.0, 0.0, 4.0, -0.0, 0.0}
	shuffleArray(array)
	reversedMergeSortDefault(array)
	assertDescendingOrder(t, array)
}
