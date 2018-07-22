package liblinear

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"strconv"
	"strings"
	"time"
)

// Training is the struct command to hold
type Training struct {
	Bias            float64
	FindC           bool
	CSpecified      bool
	SolverSpecified bool
	CrossValidation bool
	InputFilename   string
	ModelFilename   string
	NrFold          int
	Param           *Parameter
	Prob            *Problem
}

//NewTraining creates a new training type
func NewTraining(bias float64, findC bool, cSpecified bool, solverSpecified bool, crossValidation bool, inputFile string, outputFile string, nrFold int, param *Parameter, problem *Problem) *Training {
	return &Training{Bias: bias, FindC: findC, CSpecified: cSpecified, SolverSpecified: solverSpecified, CrossValidation: crossValidation, InputFilename: inputFile, ModelFilename: outputFile, NrFold: nrFold, Param: param, Prob: problem}
}

//DoFindParameterC searches for the best parameters for C
func (t *Training) DoFindParameterC() {
	var startC float64
	var maxC float64 = 1024

	if t.CSpecified {
		startC = t.Param.C
	} else {
		startC = -1.0
	}

	logger.Printf("Doing parameter search with %d-fold cross validation.\n", t.NrFold)
	result := findParameterC(t.Prob, t.Param, t.NrFold, startC, maxC)
	logger.Printf("Best C = %g  CV accuracy = %g\n", result.bestC, 100.0*result.bestRate)
}

//DoCrossValidation does just that
func (t *Training) DoCrossValidation() {

	target := make([]float64, t.Prob.L)
	totalError := 0.0
	var sumv, sumy, sumvv, sumyy, sumvy float64

	start := time.Now()
	crossValidation(t.Prob, t.Param, t.NrFold, target)
	log.Printf("time: %d ns\n", time.Now().Sub(start))

	if t.Param.SolverType.IsSupportVectorRegression() {
		for i := 0; i < t.Prob.L; i++ {
			y := t.Prob.Y[i]
			v := target[i]
			totalError += (v - y) * (v - y)
			sumv += v
			sumy += y
			sumvv += v * v
			sumyy += y * y
			sumvy += v * y
		}
		log.Printf("Cross Validation Mean squared error = %g\n", totalError/float64(t.Prob.L))
		log.Printf("Cross Validation Squared correlation coefficient = %g\n",
			(float64(t.Prob.L)*sumvy-sumv*sumy)*(float64(t.Prob.L)*sumvy-sumv*sumy)/(float64(t.Prob.L)*sumvv-sumv*sumv)*(float64(t.Prob.L)*sumyy-sumy*sumy))
	} else {
		totalCorrect := 0
		for i := 0; i < t.Prob.L; i++ {
			if target[i] == t.Prob.Y[i] {
				totalCorrect++
			}
		}
		log.Printf("correct: %d\n", totalCorrect)
		log.Printf("Cross Validation Accuracy = %g%%\n", float64(100.0*totalCorrect/t.Prob.L))
	}
}

//ReadProblem reads the problem based on the inputstream and bias
func ReadProblem(inputStream io.Reader, bias float64) *Problem {

	var err error
	scanner := bufio.NewScanner(inputStream)
	vy := make([]float64, 0)
	vx := make([][]Feature, 0)
	maxIndex := 0
	lineNr := 0

	for scanner.Scan() {
		lineNr++
		line := scanner.Text()

		tokens := strings.Split(strings.TrimSpace(line), " ")
		var v float64
		if v, err = strconv.ParseFloat(tokens[0], 64); err != nil {
			log.Fatalf("Unable to read line: %f, err=%v", v, err)
		}
		vy = append(vy, v)

		m := len(tokens) - 1
		var x []Feature
		if bias >= 0 {
			x = make([]Feature, m+1)
		} else {
			x = make([]Feature, m)
		}

		var prevKey int64
		for i := 1; i < len(tokens); i++ {
			t := tokens[i]

			keyVal := strings.Split(t, ":")
			if len(keyVal) != 2 {
				panic(fmt.Sprintf("Token format is incorrect %v", keyVal))
			}

			var key int64
			if key, err = strconv.ParseInt(keyVal[0], 10, 32); err != nil {
				panic(fmt.Sprintf("Error with the key val %s, err=%v", keyVal[0], err))
			}

			if key <= 0 {
				panic(fmt.Sprintf("Invalid index %d on line %d", key, lineNr))
			}

			if prevKey >= key {
				panic(fmt.Sprintf("The indices must be sorted in ascending order (line %d)", lineNr))
			} else {
				prevKey = key
			}

			var val float64
			if val, err = strconv.ParseFloat(keyVal[1], 64); err != nil {
				panic(fmt.Sprintf("Error with the token val %s, err=%v", keyVal[1], err))
			}

			x[i-1] = NewFeatureNode(int(key), val)

		}
		if m > 0 {
			maxIndex = int(math.Max(float64(maxIndex), float64(x[m-1].GetIndex())))
		}
		vx = append(vx, x)
	}

	return constructProblem(vy, vx, maxIndex, bias)
}

func constructProblem(vy []float64, vx [][]Feature, maxIndex int, bias float64) *Problem {

	l := len(vy)
	x := make([][]Feature, l)
	n := maxIndex

	if bias >= 0 {
		n++
	}

	for i := 0; i < l; i++ {
		x[i] = vx[i]
		if bias >= 0 {
			if x[i][len(x[i])-1] != nil {
				log.Fatalf("error while constructin problem...")
			}
			x[i][len(x[i])-1] = NewFeatureNode(maxIndex+1, bias)
		}
	}

	y := make([]float64, l)
	for i := 0; i < l; i++ {
		y[i] = vy[i]
	}

	return NewProblem(l, n, y, x, bias)
}
