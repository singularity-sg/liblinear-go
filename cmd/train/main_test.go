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
		os.Args = []string{"-v=5", fmt.Sprintf("-s=%d", solver.Id()), "-q", "-v=3", "-if=../../testdata/iris.scale"}
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
	}
}

func TestExitWithHelp(t *testing.T) {
	exitWithHelp()
}
