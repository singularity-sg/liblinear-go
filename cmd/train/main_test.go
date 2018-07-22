package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"limhan.info/liblinear-go/liblinear"
	"limhan.info/liblinear-go/test"
)

var tempDir, _ = ioutil.TempDir("", "temp")

func TestDoCrossValidationOnIrisDataSet(t *testing.T) {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	for _, solver := range liblinear.SolverTypeValues() {
		os.Args = []string{"-v=5", fmt.Sprintf("-s=%d", solver.Id()), "-q", "-if=../../testdata/iris.scale"}
		main()
	}
}

func TestFindBestCOnIrisDataSet(t *testing.T) {
	os.Args = []string{"-C", "-if=../../testdata/iris.scale"}
	main()
}

func TestParseCommandLine(t *testing.T) {
	for _, solverType := range liblinear.SolverTypeValues() {
		training := parseTrainingFromArgs([]string{"-B=5.3", fmt.Sprintf("-s=%d", solverType.Id()), "-p=0.01", "-if=../../testdata/iris.scale"})
		if training.FindC == true {
			t.Errorf("FindC should be false")
		}
		if training.NrFold != 0 {
			t.Errorf("NrFold should be 0")
		}

		param := training.Param
		if param.SolverType != solverType {
			t.Errorf("SolverType of param should be compatible with the training")
		}

		switch solverType.Id() {

		case 0:
			fallthrough
		case 2:
			fallthrough
		case 5:
			fallthrough
		case 6:
			if param.Eps != 0.01 {
				t.Errorf("The solvertype %d should have eps 0.01 but was %f", solverType.Id(), param.Eps)
			}
		case 11:
			if param.Eps != 0.001 {
				t.Errorf("The solvertype %d should have eps 0.001 but was %f", solverType.Id(), param.Eps)
			}
		case 7:
			fallthrough
		default:
			if param.Eps != 0.1 {
				t.Errorf("The solvertype %d should have eps 0.1 but was %f", solverType.Id(), param.Eps)
			}

			if training.Bias != 5.3 {
				t.Errorf("The bias should be 5.3 but was %f!", training.Bias)
			}

			if param.P != 0.01 {
				t.Errorf("The P value should be 0.01 but was %f", param.P)
			}
		}
	}
}

func TestFindCNoSolverSpecified(t *testing.T) {

	os.Args = []string{"-C", "-if=../../testdata/iris.scale", "-of=../../testdata/iris.scale.model"}
	train := parseTrainingFromArgs(os.Args)

	if !train.FindC {
		t.Error("FindC should be true")
	}

	if train.NrFold != 5 {
		t.Error("Number of Folds should be 5")
	}

	if train.Param.SolverType != liblinear.L2R_L2LOSS_SVC {
		t.Error("The solvertype should be L2R_L2LOSS_SVC")
	}

	if train.Param.Eps != 0.01 {
		t.Error("The param eps should be 0.01")
	}

	if train.Param.P != 0.1 {
		t.Error("The param P should be 0.1")
	}
}

func TestFindCSolverAndNumFoldsSpecified(t *testing.T) {

	os.Args = []string{"-s=0", "-v=10", "-C", "-if=../../testdata/iris.scale", "-of=../../testdata/iris.scale.model"}
	train := parseTrainingFromArgs(os.Args)

	if !train.FindC {
		t.Error("FindC should be true")
	}

	if train.NrFold != 10 {
		t.Error("Number of Folds should be 10")
	}

	if train.Param.SolverType != liblinear.L2R_LR {
		t.Error("The solvertype should be L2R_LR")
	}

	if train.Param.Eps != 0.01 {
		t.Error("The param eps should be 0.01")
	}

	if train.Param.P != 0.1 {
		t.Error("The param P should be 0.1")
	}
}

func TestParseWeights(t *testing.T) {

	os.Args = []string{"-v=10", "-c=10", "-w1=1.234", "-if=../../testdata/iris.scale", "-of=../../testdata/iris.scale.model"}
	train := parseTrainingFromArgs(os.Args)

	if train.Param.WeightLabel[0] != 1 {
		t.Errorf("Weight labels should be 1 but was %v", train.Param.WeightLabel[0])
	}

	if train.Param.Weight[0] != 1.234 {
		t.Errorf("Weight should be 1.234 but was %v", train.Param.Weight[0])
	}

	os.Args = []string{"-w1=1.234", "-w2=0.12", "-w3=7", "-if=../../testdata/iris.scale", "-of=../../testdata/iris.scale.model"}
	train = parseTrainingFromArgs(os.Args)

	if !reflect.DeepEqual(train.Param.WeightLabel, []int{1, 2, 3}) {
		t.Error("The weights labels should be [1,2,3]")
	}

	if !reflect.DeepEqual(train.Param.Weight, []float64{1.234, 0.12, 7}) {
		t.Errorf("The weights should be [1.234, 0.12, 7] but was %v", train.Param.Weight)
	}
}

func TestReadProblem(t *testing.T) {

	var f *os.File
	var err error
	if f, err = ioutil.TempFile("", "train_tmp.txt"); err != nil {
		t.Errorf("Unable to create temp file due to error %+v", err)
	}

	lines := []string{
		"1 1:1 3:1 4:1 6:1",
		"2 2:1 3:1 5:1 7:1",
		"1 3:1 5:1",
		"1 1:1 4:1 7:1",
		"2 4:1 5:1 7:1",
	}

	test.WriteToFile(f, lines)

	if f, err = os.Open(f.Name()); err != nil {
		t.Errorf("Unable to read file %+v", err)
	}
	problem := liblinear.ReadProblem(f, 1.0)

	assert.Equal(t, 1.0, problem.Bias)
	assert.Equal(t, len(lines), len(problem.Y))
	assert.Equal(t, problem.Y, []float64{1.0, 2.0, 1.0, 1.0, 2.0})
	assert.Equal(t, 8, problem.N)
	assert.Equal(t, problem.L, len(problem.Y))
	assert.Equal(t, len(problem.X), len(problem.Y))
}

func TestReadProblemWithEmptyLine(t *testing.T) {

	var f *os.File
	var err error
	if f, err = ioutil.TempFile("", "train_tmp.txt"); err != nil {
		t.Errorf("Unable to create temp file due to error %+v", err)
	}

	lines := []string{
		"1 1:1 3:1 4:1 6:1",
		"2",
	}

	test.WriteToFile(f, lines)

	if f, err = os.Open(f.Name()); err != nil {
		t.Errorf("Unable to read file %+v", err)
	}
	problem := liblinear.ReadProblem(f, -1.0)

	assert.Equal(t, -1.0, problem.Bias)
	assert.Equal(t, len(lines), len(problem.Y))
	assert.Equal(t, problem.Y, []float64{1.0, 2.0})
	assert.Equal(t, 6, problem.N)
	assert.Equal(t, problem.L, len(problem.Y))
	assert.Equal(t, len(problem.X), len(problem.Y))

	assert.Equal(t, 4, len(problem.X[0]))
	assert.Equal(t, 0, len(problem.X[1]))
}
func TestReadProblemUnsorted(t *testing.T) {
	defer func() {
		if msg := recover(); msg != nil {
			assert.Equal(t, "The indices must be sorted in ascending order (line 3)", msg)
		}
	}()

	var f *os.File
	var err error
	if f, err = ioutil.TempFile("", "train_tmp.txt"); err != nil {
		t.Errorf("Unable to create temp file due to error %+v", err)
	}

	lines := []string{
		"1 1:1 3:1 4:1 6:1",
		"2 2:1 3:1 5:1 7:1",
		"1 3:1 5:1 4:1",
	}

	test.WriteToFile(f, lines)

	if f, err = os.Open(f.Name()); err != nil {
		t.Errorf("Unable to read file %+v", err)
	}

	_ = liblinear.ReadProblem(f, -1.0)

}

func TestReadProblemWithInvalidIndex(t *testing.T) {
	defer func() {
		if msg := recover(); msg != nil {
			assert.Equal(t, "Invalid index -4 on line 2", msg)
		}
	}()

	var f *os.File
	var err error
	if f, err = ioutil.TempFile("", "train_tmp.txt"); err != nil {
		t.Errorf("Unable to create temp file due to error %+v", err)
	}

	lines := []string{
		"1 1:1 3:1 4:1 6:1",
		"2 2:1 3:1 5:1 -4:1",
	}

	test.WriteToFile(f, lines)

	if f, err = os.Open(f.Name()); err != nil {
		t.Errorf("Unable to read file %+v", err)
	}

	_ = liblinear.ReadProblem(f, -1.0)
}
