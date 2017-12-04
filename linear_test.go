package liblinear

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sort"
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

				if model.isProbabilityModel() {
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

	prob := createRandomProblem(numClasses)

	param := NewParameter(L2R_LR, 10, 0.01, 0.1, 1000)
	nrFold := 10
	target := make([]float64, prob.L)
	CrossValidation(prob, param, nrFold, target)

	for _, val := range target {
		intVal := int(val)
		assert.True(t, intVal >= 0 && intVal < numClasses)
	}
}

func TestLoadSaveModel(t *testing.T) {
	for _, solverType := range SolverTypeValues() {
		model := createRandomModel()
		model.SolverType = solverType
		fName := "modeltest-" + solverType.Name()

		if f, err := os.Create(fName); err == nil {
			defer f.Close()
			saveModel(f, model)
		}

		var loadedModel *Model
		if of, err := os.Open(fName); err == nil {
			defer of.Close()
			loadedModel = loadModel(of)
		}

		assert.ObjectsAreEqualValues(model, loadedModel)
	}
}

func TestLoadEmptyModel(t *testing.T) {
	tmpFile, _ := ioutil.TempFile(tempDir, "tempFile")
	writer := bufio.NewWriter(tmpFile)

	writer.WriteString("solver_type L2R_LR\n")
	writer.WriteString("nr_class 2\n")
	writer.WriteString("label 1 2\n")
	writer.WriteString("nr_feature 0\n")
	writer.WriteString("bias -1.0\n")
	writer.WriteString("w\n")
	writer.Flush()
	tmpFile.Close()

	tmpFile, _ = os.Open(tmpFile.Name())
	defer tmpFile.Close()

	loadedModel := loadModel(tmpFile)
	assert.Equal(t, L2R_LR, loadedModel.SolverType)
	assert.Contains(t, loadedModel.Label, 1, 2)
	assert.Equal(t, loadedModel.NumClass, 2)
	assert.Equal(t, loadedModel.NumFeatures, 0)
	assert.Equal(t, loadedModel.GetFeatureWeights(), []float64{})
	assert.Equal(t, loadedModel.Bias, -1.0)
}

func createRandomModel() *Model {
	label := []int{1, math.MaxInt32, 2}
	solverType := L2R_LR
	w := make([]float64, len(label)*300)
	for i := 0; i < len(w); i++ {
		w[i] = round(random.Float64()*100000, 0, 0) / 10000
	}
	w[random.Int31n(int32(len(w)))] = 0.0
	w[random.Int31n(int32(len(w)))] = -0.0

	numFeature := len(w)/len(label) - 1
	nrClass := len(label)

	return NewModel(2, label, nrClass, numFeature, solverType, w)
}

func createRandomProblem(numClasses int) *Problem {
	var l = random.Intn(100) + 1
	var n = random.Intn(100) + 1
	prob := NewProblem(l, n, make([]float64, l), make([][]Feature, l), -1.0)

	for i := 0; i < prob.L; i++ {
		prob.Y[i] = float64(random.Intn(numClasses))
		randomNumbers := make(map[int]struct{})
		num := random.Intn(prob.N) + 1
		for j := 0; j < num; j++ {
			randomNumbers[random.Intn(prob.N)+1] = struct{}{}
		}

		var randomIndices []int
		for k := range randomNumbers {
			randomIndices = append(randomIndices, k)
		}

		sort.Ints(randomIndices)

		prob.X[i] = make([]Feature, len(randomIndices))
		for j := 0; j < len(randomIndices); j++ {
			prob.X[i][j] = NewFeatureNode(randomIndices[j], random.Float64())
		}
	}

	return prob
}

func round(val float64, roundOn float64, places int) (newVal float64) {
	var round float64
	pow := math.Pow(10, float64(places))
	digit := pow * val
	_, div := math.Modf(digit)
	if div >= roundOn {
		round = math.Ceil(digit)
	} else {
		round = math.Floor(digit)
	}
	newVal = round / pow
	return
}
