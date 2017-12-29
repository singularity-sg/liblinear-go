package liblinear

import (
	"fmt"
	"testing"

	"github.com/lexandro/go-assert"
)

func TestGetIntArrayPointer(t *testing.T) {
	foo := []int{1, 2, 3, 4, 6}
	pFoo := IntArrayPointer{foo, 2}
	assert.Equals(t, 3, pFoo.get(0))
	assert.Equals(t, 4, pFoo.get(1))
	assert.Equals(t, 6, pFoo.get(2))

	defer func() {
		if r := recover(); r != nil {
			assert.Equals(t, "runtime error: index out of range", fmt.Sprintf("%v", r))
		}
	}()
	pFoo.get(3)
}

func TestSetIntArrayPointer(t *testing.T) {
	foo := []int{1, 2, 3, 4, 6}
	pFoo := IntArrayPointer{foo, 2}
	pFoo.set(2, 5)
	assert.Equals(t, []int{1, 2, 3, 4, 5}, foo)

	defer func() {
		if r := recover(); r != nil {
			assert.Equals(t, "runtime error: index out of range", fmt.Sprintf("%v", r))
		}
	}()
	pFoo.set(3, 0)
}

func TestGetDoubleArrayPointer(t *testing.T) {
	foo := []float64{1, 2, 3, 4, 6}
	pFoo := DoubleArrayPointer{foo, 2}

	assert.Equals(t, float64(3), pFoo.get(0))
	assert.Equals(t, float64(4), pFoo.get(1))
	assert.Equals(t, float64(6), pFoo.get(2))

	defer func() {
		if r := recover(); r != nil {
			assert.Equals(t, "runtime error: index out of range", fmt.Sprintf("%v", r))
		}
	}()
	pFoo.get(3)
}

func TestSetDoubleArrayPointer(t *testing.T) {
	foo := []float64{1, 2, 3, 4, 6}
	pFoo := DoubleArrayPointer{foo, 2}
	pFoo.set(2, 5)
	assert.Equals(t, []float64{1, 2, 3, 4, 5}, foo)

	defer func() {
		if r := recover(); r != nil {
			assert.Equals(t, "runtime error: index out of range", fmt.Sprintf("%v", r))
		}
	}()
	pFoo.set(3, 0)
}

// @Test
// public void testSetDoubleArrayPointer() {
// 	double[] foo = new double[] {1, 2, 3, 4, 6};
// 	DoubleArrayPointer pFoo = new DoubleArrayPointer(foo, 2);
// 	pFoo.set(2, 5);
// 	assertThat(foo).isEqualTo(new double[] {1, 2, 3, 4, 5});
// 	try {
// 		pFoo.set(3, 0);
// 		fail("ArrayIndexOutOfBoundsException expected");
// 	} catch (ArrayIndexOutOfBoundsException e) {}
// }
