package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"

	"limhan.info/liblinear-go/liblinear"
)

var flagPredictProbability bool

//DoPredict reads from a reader stream and writes the results to the outputstream
func DoPredict(reader io.Reader, writer io.Writer, model *liblinear.Model) {
	var correct, total int
	var errorCnt float64
	var sump, sumt, sumpp, sumtt, sumpt float64

	nrClass := model.NumClass

	var probEstimates []float64

	var n int
	var nrFeature = model.NumFeatures

	if model.Bias >= 0 {
		n = nrFeature + 1
	} else {
		n = nrFeature
	}

	if flagPredictProbability && !model.IsProbabilityModel() {
		panic("probability output is only supported for logistic regression")
	}

	if flagPredictProbability {
		labels := model.Label
		probEstimates = make([]float64, nrClass)

		io.WriteString(writer, "labels")
		for j := 0; j < nrClass; j++ {
			io.WriteString(writer, fmt.Sprintf(" %d", labels[j]))
		}
		io.WriteString(writer, "\n")
	}

	var err error
	var line string

	scanner := bufio.NewScanner(reader)

	for scanner.Scan() {
		var x []liblinear.Feature
		var targetLabel float64
		line = scanner.Text()

		tokens := strings.Split(line, " ")

		label := tokens[0]
		if targetLabel, err = strconv.ParseFloat(label, 64); err != nil {
			panic(fmt.Sprintf("Wrong input format at line %d : %v", total+1, err))
		}

		for i := 1; i < len(tokens); i++ {
			split := strings.SplitN(tokens[i], ":", 2)
			if len(split) < 2 {
				panic(fmt.Sprintf("Wrong input format at line %d", total+1))
			}

			var tmp int64
			var idx int
			var val float64
			if tmp, err = strconv.ParseInt(split[0], 10, 32); err != nil {
				panic(fmt.Sprintf("The index %s cannot be parsed", split[0]))
			}
			idx = int(tmp)

			if val, err = strconv.ParseFloat(split[1], 64); err != nil {
				panic(fmt.Sprintf("The val %s cannot be parsed", split[1]))
			}

			if idx <= nrFeature {
				node := liblinear.NewFeatureNode(idx, val)
				x = append(x, node)
			}
		}

		if model.Bias >= 0 {
			node := liblinear.NewFeatureNode(n, model.Bias)
			x = append(x, node)
		}

		nodes := make([]liblinear.Feature, len(x))
		for i := 0; i < len(nodes); i++ {
			nodes[i] = x[i]
		}

		var predictLabel float64

		if flagPredictProbability {
			if probEstimates == nil {
				panic("Problem estimates cannot be nil")
			}
			predictLabel = liblinear.PredictProbability(model, nodes, probEstimates)
			io.WriteString(writer, fmt.Sprintf("%g", predictLabel))
			for j := 0; j < model.NumClass; j++ {
				io.WriteString(writer, fmt.Sprintf(" %g", probEstimates[j]))
			}

			io.WriteString(writer, "\n")
		} else {
			predictLabel = liblinear.Predict(model, nodes)
			io.WriteString(writer, fmt.Sprintf(" %g\n", predictLabel))
		}

		if predictLabel == targetLabel {
			correct++
		}

		errorCnt += (predictLabel - targetLabel) * (predictLabel - targetLabel)
		sump += predictLabel
		sumt += targetLabel
		sumpp += predictLabel * predictLabel
		sumtt += targetLabel * targetLabel
		sumpt += predictLabel * targetLabel
		total++
	}

	if model.SolverType.IsSupportVectorRegression() {
		log.Printf("Mean squared error = %g (regression)\n", errorCnt/float64(total))
		val := ((float64(total)*sumpt - sump*sumt) * (float64(total)*sumpt - sump*sumt)) / ((float64(total)*sumpp - sump*sump) * (float64(total)*sumtt - sumt*sumt))
		log.Printf("Squared correlation coefficient = %g (regression)\n", val)
	} else {
		log.Printf("Accuracy = %g%% (%d/%d)\n", float64(correct)/float64(total)*100, correct, total)
	}
}

func exitWithHelp() {
	log.Fatalf("Usage: predict [options] -if=test_file -mf=model_file -of=output_file\n" +
		"options:\n" +
		"-if test_file\n" +
		"-of output_file\n" +
		"-mf model_file\n" +
		"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only\n" +
		"-q quiet mode (no outputs)\n")
}

func main() {

	var err error
	var inputFile *os.File
	var outputFile *os.File
	var modelFile *os.File

	for _, arg := range os.Args[1:] {

		flagVal := strings.SplitN(arg, "=", 2)
		switch flagVal[0] {

		case "-b":
			if b, err := strconv.ParseInt(flagVal[1], 10, 32); err == nil {
				flagPredictProbability = b != 0
			} else {
				exitWithHelp()
			}
		case "-mf":
			if modelFile, err = os.Open(flagVal[1]); err != nil {
				log.Fatalf("Unable to open modelfile %s", flagVal[1])
			}
		case "-if":
			if inputFile, err = os.Open(flagVal[1]); err != nil {
				log.Fatalf("Unable to open inputfile %s", flagVal[1])
			}
		case "-of":
			if outputFile, err = os.Open(flagVal[1]); err != nil {
				log.Fatalf("Unable to open outputfile %s", flagVal[1])
			}
		case "-h", "-help":
			fallthrough
		default:
			exitWithHelp()

		}

	}

	if len(os.Args) < 2 {
		exitWithHelp()
	}

	model := liblinear.LoadModel(modelFile)
	DoPredict(inputFile, outputFile, model)
}
