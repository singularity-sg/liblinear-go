package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"limhan.info/liblinear-go/liblinear"
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
		training := parseTrainingFromArgs([]string{"-B=5.3", fmt.Sprintf("-s=%d", solverType.Id()), "-p=0.01", "-if=model-filename"})
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
