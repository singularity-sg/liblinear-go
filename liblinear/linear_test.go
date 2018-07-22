package liblinear

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

var tempDir, _ = ioutil.TempDir("", "temp")

func TestTrainPredict(t *testing.T) {

	var x = make([][]Feature, 4)
	x[0] = make([]Feature, 2)
	x[1] = make([]Feature, 1)
	x[2] = make([]Feature, 1)
	x[3] = make([]Feature, 3)

	x[0][0] = NewFeatureNode(1, 1)
	x[0][1] = NewFeatureNode(2, 1)

	x[1][0] = NewFeatureNode(3, 1)
	x[2][0] = NewFeatureNode(3, 1)

	x[3][0] = NewFeatureNode(1, 2)
	x[3][1] = NewFeatureNode(2, 1)
	x[3][2] = NewFeatureNode(4, 1)

	var y = []float64{0, 1, 1, 0}

	var prob = &Problem{
		Bias: -1,
		L:    4,
		N:    4,
		X:    x,
		Y:    y,
	}

	for _, solver := range solverTypeValues {
		for C := 0.1; C <= 100; C *= 1.2 {
			if C < 0.2 {
				if solver == L1R_L2LOSS_SVC {
					continue
				}
			}
			if C < 0.7 {
				if solver == L1R_LR {
					continue
				}
			}

			if solver.IsSupportVectorRegression() {
				continue
			}

			param := NewParameter(solver, C, 0.1, 0.1, 1000)
			model, _ := Train(prob, param)

			featureWeights := model.GetFeatureWeights()
			if solver == MCSVM_CS {
				assert.Equal(t, 8, len(featureWeights))
			} else {
				assert.Equal(t, 4, len(featureWeights))
			}

			var i = 0
			for _, value := range prob.Y {
				prediction := Predict(model, prob.X[i])
				assert.Equal(t, value, prediction, fmt.Sprintf("assertion failed for solverType %v", model.SolverType.Name()))

				if model.IsProbabilityModel() {
					estimates := make([]float64, model.NumClass)
					probabilityPrediction := PredictProbability(model, prob.X[i], estimates)
					assert.Equal(t, prediction, probabilityPrediction)

					if estimates[int(probabilityPrediction)] < (1.0 / float64(model.NumClass)) {
						t.Fail()
					}

					var estimationSum float64
					for _, estimate := range estimates {
						estimationSum += estimate
					}

					if estimationSum <= 0.9 || estimationSum >= 1.1 {
						t.Fail()
					}
				}
				i++
			}
		}
	}
}

func TestCrossValidation(t *testing.T) {
	numClasses := random.Intn(10) + 1

	prob := CreateRandomProblem(numClasses)

	param := NewParameter(L2R_LR, 10, 0.01, 0.1, 1000)
	nrFold := 10
	target := make([]float64, prob.L)
	crossValidation(prob, param, nrFold, target)

	for _, val := range target {
		intVal := int(val)
		assert.True(t, intVal >= 0 && intVal < numClasses)
	}
}

func TestLoadSaveModel(t *testing.T) {
	for _, solverType := range SolverTypeValues() {
		model := CreateRandomModel()
		model.SolverType = solverType
		fName := "modeltest-" + solverType.Name()

		if f, err := os.Create(fName); err == nil {
			defer f.Close()
			SaveModel(f, model)
		}

		var loadedModel *Model
		if of, err := os.Open(fName); err == nil {
			defer of.Close()
			loadedModel = LoadModel(of)
		}

		assert.ObjectsAreEqualValues(model, loadedModel)
	}
}

func TestLoadEmptyModel(t *testing.T) {
	tmpFile, _ := ioutil.TempFile(tempDir, "tempFile")
	writer := bufio.NewWriter(tmpFile)

	lines := []string{"solver_type L2R_LR",
		"nr_class 2",
		"label 1 2",
		"nr_feature 0",
		"bias -1.0",
		"w"}
	writeLines(writer, lines)
	tmpFile.Close()

	tmpFile, _ = os.Open(tmpFile.Name())
	defer tmpFile.Close()

	loadedModel := LoadModel(tmpFile)
	assert.Equal(t, L2R_LR, loadedModel.SolverType)
	assert.Contains(t, loadedModel.Label, 1, 2)
	assert.Equal(t, loadedModel.NumClass, 2)
	assert.Equal(t, loadedModel.NumFeatures, 0)
	assert.Equal(t, loadedModel.GetFeatureWeights(), []float64{})
	assert.Equal(t, loadedModel.Bias, -1.0)
}

func TestLoadSimpleModel(t *testing.T) {
	tmpFile, _ := ioutil.TempFile(tempDir, "tempFile")
	writer := bufio.NewWriter(tmpFile)

	lines := []string{"solver_type L2R_L2LOSS_SVR",
		"nr_class 2",
		"label 1 2",
		"nr_feature 6",
		"bias -1.0",
		"w",
		"0.1 0.2 0.3 ",
		"0.4 0.5 0.6 "}

	writeLines(writer, lines)
	tmpFile.Close()

	tmpFile, _ = os.Open(tmpFile.Name())
	defer tmpFile.Close()

	loadedModel := LoadModel(tmpFile)

	assert.Equal(t, L2R_L2LOSS_SVR, loadedModel.SolverType)
	assert.Equal(t, 2, loadedModel.NumClass)
	assert.Equal(t, 6, loadedModel.NumFeatures)
	assert.Equal(t, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, loadedModel.GetFeatureWeights())
	assert.InDelta(t, -1.0, loadedModel.Bias, 0.001)
}

func TestLoadIllegalModel(t *testing.T) {
	tmpFile, _ := ioutil.TempFile(tempDir, "tempFile")
	writer := bufio.NewWriter(tmpFile)

	lines := []string{"solver_type L2R_L2LOSS_SVR",
		"nr_class 2",
		"label 1 2",
		"nr_feature 10",
		"bias -1.0",
		"w",
		"0.1 0.2 0.3 ",
		"0.4 0.5 " + repeat("0", 1024),
	}

	writeLines(writer, lines)

	tmpFile, _ = os.Open(tmpFile.Name())
	defer tmpFile.Close()

	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Recovered from panic [%v]\n", r)
		}
	}()
	_ = LoadModel(tmpFile)
}

func TestPredictProbabilityWrongSolver(t *testing.T) {
	l := 1
	n := 1
	x := make([][]Feature, l)
	y := make([]float64, l)
	for i := 0; i < l; i++ {
		x[i] = []Feature{}
		y[i] = float64(i)
	}
	prob := NewProblem(l, n, y, x, 0)
	param := NewParameter(L2R_L1LOSS_SVC_DUAL, 10, 0.1, 0.1, 1000)

	model, _ := Train(prob, param)

	defer func() {
		if r := recover(); r != nil {
			assert.Equal(t, r, "probability output is only supported for logistic regression. This is currently only supported by the following solvers: L2R_LR, L1R_LR, L2R_LR_DUAL")
		}
	}()

	PredictProbability(model, prob.X[0], []float64{0})
	t.FailNow()

}

/**
 * compared input/output values with the C version (1.51)
 *
 * <pre>
 * IN:
 * res prob.l = 4
 * res prob.n = 4
 * 0: (2,1) (4,1)
 * 1: (1,1)
 * 2: (3,1)
 * 3: (2,2) (3,1) (4,1)
 *
 * TRANSPOSED:
 *
 * res prob.l = 4
 * res prob.n = 4
 * 0: (2,1)
 * 1: (1,1) (4,2)
 * 2: (3,1) (4,1)
 * 3: (1,1) (4,1)
 * </pre>
 */
func TestTranspose(t *testing.T) {
	l := 4
	n := 4
	bias := float64(-1)
	x := make([][]Feature, 4)
	x[0] = make([]Feature, 2)
	x[1] = make([]Feature, 1)
	x[2] = make([]Feature, 1)
	x[3] = make([]Feature, 3)

	x[0][0] = NewFeatureNode(2, 1)
	x[0][1] = NewFeatureNode(4, 1)

	x[1][0] = NewFeatureNode(1, 1)
	x[2][0] = NewFeatureNode(3, 1)

	x[3][0] = NewFeatureNode(2, 2)
	x[3][1] = NewFeatureNode(3, 1)
	x[3][2] = NewFeatureNode(4, 1)

	y := make([]float64, 4)
	y[0] = 0
	y[1] = 1
	y[2] = 1
	y[3] = 0

	prob := NewProblem(l, n, y, x, bias)

	transposed := transpose(prob)

	assert.Equal(t, len(transposed.X[0]), 1)
	assert.Equal(t, len(transposed.X[1]), 2)
	assert.Equal(t, len(transposed.X[2]), 2)
	assert.Equal(t, len(transposed.X[3]), 2)

	assert.Equal(t, transposed.X[0][0], NewFeatureNode(2, 1))
	assert.Equal(t, transposed.X[1][0], NewFeatureNode(1, 1))
	assert.Equal(t, transposed.X[1][1], NewFeatureNode(4, 2))
	assert.Equal(t, transposed.X[2][0], NewFeatureNode(3, 1))
	assert.Equal(t, transposed.X[2][1], NewFeatureNode(4, 1))
	assert.Equal(t, transposed.X[3][0], NewFeatureNode(1, 1))
	assert.Equal(t, transposed.X[3][1], NewFeatureNode(4, 1))

	assert.Equal(t, transposed.Y, prob.Y)

}

/**
 *
 * compared input/output values with the C version (1.51)
 *
 * <pre>
 * IN:
 * res prob.l = 5
 * res prob.n = 10
 * 0: (1,7) (3,3) (5,2)
 * 1: (2,1) (4,5) (5,3) (7,4) (8,2)
 * 2: (1,9) (3,1) (5,1) (10,7)
 * 3: (1,2) (2,2) (3,9) (4,7) (5,8) (6,1) (7,5) (8,4)
 * 4: (3,1) (10,3)
 *
 * TRANSPOSED:
 *
 * res prob.l = 5
 * res prob.n = 10
 * 0: (1,7) (3,9) (4,2)
 * 1: (2,1) (4,2)
 * 2: (1,3) (3,1) (4,9) (5,1)
 * 3: (2,5) (4,7)
 * 4: (1,2) (2,3) (3,1) (4,8)
 * 5: (4,1)
 * 6: (2,4) (4,5)
 * 7: (2,2) (4,4)
 * 8:
 * 9: (3,7) (5,3)
 * </pre>
 */
func TestTranspose2(t *testing.T) {
	l := 5
	n := 10
	bias := float64(-1)
	x := make([][]Feature, 5)
	x[0] = make([]Feature, 3)
	x[1] = make([]Feature, 5)
	x[2] = make([]Feature, 4)
	x[3] = make([]Feature, 8)
	x[4] = make([]Feature, 2)

	x[0][0] = NewFeatureNode(1, 7)
	x[0][1] = NewFeatureNode(3, 3)
	x[0][2] = NewFeatureNode(5, 2)

	x[1][0] = NewFeatureNode(2, 1)
	x[1][1] = NewFeatureNode(4, 5)
	x[1][2] = NewFeatureNode(5, 3)
	x[1][3] = NewFeatureNode(7, 4)
	x[1][4] = NewFeatureNode(8, 2)

	x[2][0] = NewFeatureNode(1, 9)
	x[2][1] = NewFeatureNode(3, 1)
	x[2][2] = NewFeatureNode(5, 1)
	x[2][3] = NewFeatureNode(10, 7)

	x[3][0] = NewFeatureNode(1, 2)
	x[3][1] = NewFeatureNode(2, 2)
	x[3][2] = NewFeatureNode(3, 9)
	x[3][3] = NewFeatureNode(4, 7)
	x[3][4] = NewFeatureNode(5, 8)
	x[3][5] = NewFeatureNode(6, 1)
	x[3][6] = NewFeatureNode(7, 5)
	x[3][7] = NewFeatureNode(8, 4)

	x[4][0] = NewFeatureNode(3, 1)
	x[4][1] = NewFeatureNode(10, 3)

	y := make([]float64, 5)
	y[0] = 0
	y[1] = 1
	y[2] = 1
	y[3] = 0
	y[4] = 1

	prob := NewProblem(l, n, y, x, bias)

	transposed := transpose(prob)

	assert.Equal(t, len(transposed.X[0]), 3)
	assert.Equal(t, len(transposed.X[1]), 2)
	assert.Equal(t, len(transposed.X[2]), 4)
	assert.Equal(t, len(transposed.X[3]), 2)
	assert.Equal(t, len(transposed.X[4]), 4)
	assert.Equal(t, len(transposed.X[5]), 1)
	assert.Equal(t, len(transposed.X[7]), 2)
	assert.Equal(t, len(transposed.X[7]), 2)
	assert.Equal(t, len(transposed.X[8]), 0)
	assert.Equal(t, len(transposed.X[9]), 2)

	assert.Equal(t, transposed.X[0][0], NewFeatureNode(1, 7))
	assert.Equal(t, transposed.X[0][1], NewFeatureNode(3, 9))
	assert.Equal(t, transposed.X[0][2], NewFeatureNode(4, 2))

	assert.Equal(t, transposed.X[1][0], NewFeatureNode(2, 1))
	assert.Equal(t, transposed.X[1][1], NewFeatureNode(4, 2))

	assert.Equal(t, transposed.X[2][0], NewFeatureNode(1, 3))
	assert.Equal(t, transposed.X[2][1], NewFeatureNode(3, 1))
	assert.Equal(t, transposed.X[2][2], NewFeatureNode(4, 9))
	assert.Equal(t, transposed.X[2][3], NewFeatureNode(5, 1))

	assert.Equal(t, transposed.X[3][0], NewFeatureNode(2, 5))
	assert.Equal(t, transposed.X[3][1], NewFeatureNode(4, 7))

	assert.Equal(t, transposed.X[4][0], NewFeatureNode(1, 2))
	assert.Equal(t, transposed.X[4][1], NewFeatureNode(2, 3))
	assert.Equal(t, transposed.X[4][2], NewFeatureNode(3, 1))
	assert.Equal(t, transposed.X[4][3], NewFeatureNode(4, 8))

	assert.Equal(t, transposed.X[5][0], NewFeatureNode(4, 1))

	assert.Equal(t, transposed.X[6][0], NewFeatureNode(2, 4))
	assert.Equal(t, transposed.X[6][1], NewFeatureNode(4, 5))

	assert.Equal(t, transposed.X[7][0], NewFeatureNode(2, 2))
	assert.Equal(t, transposed.X[7][1], NewFeatureNode(4, 4))

	assert.Equal(t, transposed.X[9][0], NewFeatureNode(3, 7))
	assert.Equal(t, transposed.X[9][1], NewFeatureNode(5, 3))

	assert.Equal(t, transposed.Y, prob.Y)

}

/**
 * compared input/output values with the C version (1.51)
 *
 * IN:
 * res prob.l = 3
 * res prob.n = 4
 * 0: (1,2) (3,1) (4,3)
 * 1: (1,9) (2,7) (3,3) (4,3)
 * 2: (2,1)
 *
 * TRANSPOSED:
 *
 * res prob.l = 3
 *      * res prob.n = 4
 * 0: (1,2) (2,9)
 * 1: (2,7) (3,1)
 * 2: (1,1) (2,3)
 * 3: (1,3) (2,3)
 *
 */
func TestTranspose3(t *testing.T) {
	l := 3
	n := 4
	bias := float64(0)
	x := make([][]Feature, 4)
	x[0] = make([]Feature, 3)
	x[1] = make([]Feature, 4)
	x[2] = make([]Feature, 1)
	x[3] = make([]Feature, 3)

	x[0][0] = NewFeatureNode(1, 2)
	x[0][1] = NewFeatureNode(3, 1)
	x[0][2] = NewFeatureNode(4, 3)

	x[1][0] = NewFeatureNode(1, 9)
	x[1][1] = NewFeatureNode(2, 7)
	x[1][2] = NewFeatureNode(3, 3)
	x[1][3] = NewFeatureNode(4, 3)

	x[2][0] = NewFeatureNode(2, 1)

	x[3][0] = NewFeatureNode(3, 2)

	y := make([]float64, 3)

	prob := NewProblem(l, n, y, x, bias)

	transposed := transpose(prob)

	assert.Equal(t, len(transposed.X), 4)
	assert.Equal(t, len(transposed.X[0]), 2)
	assert.Equal(t, len(transposed.X[1]), 2)
	assert.Equal(t, len(transposed.X[2]), 2)
	assert.Equal(t, len(transposed.X[3]), 2)

	assert.Equal(t, transposed.X[0][0], NewFeatureNode(1, 2))
	assert.Equal(t, transposed.X[0][1], NewFeatureNode(2, 9))

	assert.Equal(t, transposed.X[1][0], NewFeatureNode(2, 7))
	assert.Equal(t, transposed.X[1][1], NewFeatureNode(3, 1))

	assert.Equal(t, transposed.X[2][0], NewFeatureNode(1, 1))
	assert.Equal(t, transposed.X[2][1], NewFeatureNode(2, 3))

	assert.Equal(t, transposed.X[3][0], NewFeatureNode(1, 3))
	assert.Equal(t, transposed.X[3][1], NewFeatureNode(2, 3))

}

func TestTrainUnsortedProblem(t *testing.T) {

	x := make([][]Feature, 4)
	x[0] = make([]Feature, 2)

	x[0][0] = NewFeatureNode(2, 1)
	x[0][1] = NewFeatureNode(1, 1)

	prob := NewProblem(1, 2, []float64{0, 0, 0, 0}, x, -1)
	param := NewParameter(L2R_LR, 10, 0.1, 0.1, 1000)

	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Recovered from panic [%v]\n", r)
		}
	}()

	Train(prob, param)

	t.FailNow()
}

func TestTrainTooLargeProblem(t *testing.T) {

	l := 1000
	n := 20000000
	y := make([]float64, l)
	x := make([][]Feature, l)

	for i := 0; i < l; i++ {
		x[i] = []Feature{}
		y[i] = float64(i)
	}

	prob := NewProblem(l, n, y, x, 0.0)

	for _, solverType := range solverTypeValues {
		if solverType.IsSupportVectorRegression() {
			continue
		}

		param := NewParameter(solverType, 10, 0.1, 0.1, 1000)

		defer func() {
			if r := recover(); r != nil {
				fmt.Printf("Recovered from panic [%v]\n", r)
			}
		}()

		Train(prob, param)
	}

}

func repeat(mystring string, numOfRepetitions int) string {
	var val string
	for i := 0; i < numOfRepetitions; i++ {
		val += mystring
	}
	return val
}

func writeLines(writer *bufio.Writer, lines []string) {

	for _, line := range lines {
		writer.WriteString(line + "\n")
	}
	writer.Flush()
}
