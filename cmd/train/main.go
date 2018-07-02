package main

import (
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"limhan.info/liblinear-go/liblinear"
)

var err error

func parseTrainingFromArgs(args []string) *liblinear.Training {
	var s int
	var c int
	var p float64
	var e int
	var bias float64
	var v int
	var findC bool
	var quiet bool

	var cSpecified bool
	var solverSpecified bool
	var crossValidation bool

	var inputFilename string
	var modelFilename string
	var nrFold int
	var param *liblinear.Parameter
	var prob *liblinear.Problem

	param = liblinear.NewParameter(liblinear.L2R_L2LOSS_SVC_DUAL, 1.0, math.Inf(1), 0.1, 1000)
	bias = -1.0

	for i := 0; i < len(args); i++ {

		if i != len(args)-1 && args[i][0:1] != "-" {
			exitWithHelp()
		}

		tokens := strings.Split(args[i], "=")
		flag := tokens[0]
		var val string
		if len(tokens) > 1 {
			val = tokens[1]
		}

		switch flag {
		case "-s":
			id, _ := strconv.Atoi(val)
			param.SolverType = liblinear.GetById(id)
			solverSpecified = true
		case "-c":
			c, _ := strconv.Atoi(val)
			param.C = float64(c)
			cSpecified = true
		case "-e":
			eps, _ := strconv.Atoi(val)
			param.Eps = float64(eps)
		case "-B":
			var b int
			if b, err = strconv.Atoi(val); err != nil {
				log.Fatalf("bias value is not valid %f", bias)
			}
			bias = float64(b)
		case "-v":
			crossValidation = true
			nrFold, _ = strconv.Atoi(val)
			if nrFold < 2 {
				log.Fatal("n-fold cross validation: n must be >= 2")
				exitWithHelp()
			}
		case "-q":
			//Disable debug
		case "-if":
			inputFilename = val
		case "-of":
			modelFilename = val
		case "-C":
			findC = true
		default:
			if flag[0:2] == "-w" {
				weightLabel, _ := strconv.Atoi(flag[2:])
				weight, _ := strconv.Atoi(val)
				param.WeightLabel = append(param.WeightLabel, weightLabel)
				param.Weight = append(param.Weight, float64(weight))
				break
			}
			log.Fatalf("Unknown option : %c", args[i][1])
			exitWithHelp()
		}

		log.Println(args[i])
	}

	if findC {
		if !crossValidation {
			nrFold = 5
		}
		if !solverSpecified {
			log.Println("Solver not specified. Using -s 2")
			param.SolverType = liblinear.L2R_L2LOSS_SVC
		} else if param.SolverType != liblinear.L2R_LR && param.SolverType != liblinear.L2R_L2LOSS_SVC {
			log.Fatal("Warm-start parameter search only available for -s 0 and -s 2")
			// exitWithHelp()
		}
	}

	if param.Eps == math.Inf(-1) {
		switch param.SolverType {
		case liblinear.L2R_LR:
			fallthrough
		case liblinear.L2R_L2LOSS_SVC:
			param.Eps = 0.01
		case liblinear.L2R_L2LOSS_SVR:
			param.Eps = 0.001
		case liblinear.L2R_L2LOSS_SVC_DUAL:
			fallthrough
		case liblinear.L2R_L1LOSS_SVC_DUAL:
			fallthrough
		case liblinear.MCSVM_CS:
			fallthrough
		case liblinear.L2R_LR_DUAL:
			param.Eps = 0.1
		case liblinear.L1R_L2LOSS_SVC:
			fallthrough
		case liblinear.L1R_LR:
			param.Eps = 0.01
		case liblinear.L2R_L1LOSS_SVR_DUAL:
			fallthrough
		case liblinear.L2R_L2LOSS_SVR_DUAL:
			param.Eps = 0.1
		default:
			log.Fatalf("unknown solver type %v", param.SolverType)
		}
	}

	return liblinear.NewTraining(bias, findC, cSpecified, solverSpecified, crossValidation, inputFilename, modelFilename, nrFold, param, prob)
}

func exitWithHelp() {
	log.Fatalf("Usage: train [options] -if training_set_file [-of model_file]\n" +
		"options:\n" +
		"-s type : set type of solver (default 1)\n" +
		"  for multi-class classification\n" +
		"    0 -- L2-regularized logistic regression (primal)\n" +
		"    1 -- L2-regularized L2-loss support vector classification (dual)\n" +
		"    2 -- L2-regularized L2-loss support vector classification (primal)\n" +
		"    3 -- L2-regularized L1-loss support vector classification (dual)\n" +
		"    4 -- support vector classification by Crammer and Singer\n" +
		"    5 -- L1-regularized L2-loss support vector classification\n" +
		"    6 -- L1-regularized logistic regression\n" +
		"    7 -- L2-regularized logistic regression (dual)\n" +
		"  for regression\n" +
		"   11 -- L2-regularized L2-loss support vector regression (primal)\n" +
		"   12 -- L2-regularized L2-loss support vector regression (dual)\n" +
		"   13 -- L2-regularized L1-loss support vector regression (dual)\n" +
		"-c cost : set the parameter C (default 1)\n" +
		"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n" +
		"-e epsilon : set tolerance of termination criterion\n" +
		"   -s 0 and 2\\n" + "       |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" +
		"       where f is the primal function and pos/neg are # of\n" +
		"       positive/negative data (default 0.01)\\n" + "   -s 11\n" +
		"       |f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n" +
		"   -s 1, 3, 4 and 7\\n" + "       Dual maximal violation <= eps; similar to libsvm (default 0.1)\n" +
		"   -s 5 and 6\n" +
		"       |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n" +
		"       where f is the primal function (default 0.01)\n" +
		"   -s 12 and 13\n" +
		"       |f'(alpha)|_1 <= eps |f'(alpha0)|,\n" +
		"       where f is the dual function (default 0.1)\n" +
		"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n" +
		"-wi weight: weights adjust the parameter C of different classes (see README for details)\n" +
		"-v n: n-fold cross validation mode\n" +
		"-C : find parameter C (only for -s 0 and 2)\n" +
		"-of : Model filename\n" +
		"-if : Input filename\n" +
		"-q : quiet mode (no outputs)\n")
}

func main() {

	training := parseTrainingFromArgs(os.Args)
	training.ReadProblem()

	if training.FindC {
		training.DoFindParameterC()
	} else if training.CrossValidation {
		training.DoCrossValidation()
	} else {
		model, err := liblinear.Train(training.Prob, training.Param)
		var f *os.File
		if f, err = os.Open(training.ModelFilename); err != nil {
			log.Fatalf("Unable to open file %v", err)
		}
		liblinear.SaveModel(f, model)
		defer f.Close()
	}

}
