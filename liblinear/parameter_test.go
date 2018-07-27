package liblinear

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSetWeights(t *testing.T) {
	param := NewParameter(L2R_L1LOSS_SVC_DUAL, 100, 1e-3, 0.1, 1000)

	var expected []float64
	assert.Equal(t, expected, param.weight)
	assert.Equal(t, 0, param.GetNumWeights())

	if err := param.SetWeight([]float64{0, 1, 2, 3, 4, 5}, []int{1, 1, 1, 1, 2, 3}); err != nil {
		t.Fail()
	}

	assert.Equal(t, 6, param.GetNumWeights())

	if err := param.SetWeight([]float64{0, 1, 2, 3, 4, 5}, []int{1}); err == nil {
		t.Fail()
	}

}

func TestGetWeights(t *testing.T) {
	param := NewParameter(L2R_L1LOSS_SVC_DUAL, 100, 1e-3, 0.1, 1000)
	weights := []float64{0, 1, 2, 3, 4, 5}
	weightLabels := []int{1, 1, 1, 1, 2, 3}

	param.SetWeight(weights, weightLabels)

	assert.Equal(t, param.GetWeights(), weights)
	param.GetWeights()[0]++
	assert.Equal(t, param.GetWeights(), weights)

	assert.Equal(t, param.GetWeightLabels(), weightLabels)
	param.GetWeightLabels()[0]++
	assert.Equal(t, param.GetWeightLabels(), weightLabels)
}

func TestSetC(t *testing.T) {
	param := NewParameter(L2R_L1LOSS_SVC_DUAL, 100, 1e-3, 0.1, 1000)
	if err := param.SetC(0.0001); err != nil {
		t.Fail()
	}
	assert.Equal(t, 0.0001, param.c)

	param.SetC(100)
	assert.Equal(t, 100.0, param.c)

	if err := param.SetC(-1); err == nil {
		t.Fail()
	}

	if err := param.SetC(0); err == nil {
		t.Fail()
	}

}

func TestSetEps(t *testing.T) {
	param := NewParameter(L2R_L1LOSS_SVC_DUAL, 100, 1e-3, 0.1, 1000)
	if err := param.SetC(0.0001); err != nil {
		t.Fail()
	}
	assert.Equal(t, 0.0001, param.c)

	param.SetC(100)
	assert.Equal(t, 100.0, param.c)

	if err := param.SetC(-1); err == nil {
		t.Fail()
	}

	if err := param.SetC(0); err == nil {
		t.Fail()
	}
}

func TestSetSolverType(t *testing.T) {
	param := NewParameter(L2R_L1LOSS_SVC_DUAL, 100, 1e-3, 0.1, 1000)
	for _, solverType := range SolverTypeValues() {
		if err := param.SetSolverType(solverType); err != nil {
			t.Fail()
		}
		assert.Equal(t, param.solverType, solverType)
	}

	var nilSolverType *SolverType
	if err := param.SetSolverType(nilSolverType); err == nil {
		t.Fail()
	}

}
