package liblinear

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strconv"
)

const (
	version int = 211
)

var logger = log.New(os.Stdout, "[liblinear] ", log.LstdFlags)

var random = rand.New(rand.NewSource(0))

func crossValidation(prob *Problem, param *Parameter, nrFold int, target []float64) {
	var i int
	var l = prob.L
	var perm = make([]int, l)

	if nrFold > l {
		nrFold = l
		logger.Println("WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)")
	}
	var foldStart = make([]int, nrFold+1)

	for i = 0; i < l; i++ {
		perm[i] = i
	}

	for i = 0; i < l; i++ {
		j := i + random.Intn(l-i)
		swapIntArray(perm, i, j)
	}

	for i = 0; i <= nrFold; i++ {
		foldStart[i] = i * l / nrFold
	}

	for i = 0; i < nrFold; i++ {
		begin := foldStart[i]
		end := foldStart[i+1]
		var j, k int
		tempL := l - (end - begin)
		subProb := NewProblem(tempL, prob.N, make([]float64, tempL), make([][]Feature, tempL), prob.Bias)

		k = 0
		for j = 0; j < begin; j++ {
			subProb.X[k] = prob.X[perm[j]]
			subProb.Y[k] = prob.Y[perm[j]]
			k++
		}

		for j = end; j < l; j++ {
			subProb.X[k] = prob.X[perm[j]]
			subProb.Y[k] = prob.Y[perm[j]]
			k++
		}

		if subModel, err := Train(subProb, param); err == nil {
			for j = begin; j < end; j++ {
				target[perm[j]] = Predict(subModel, prob.X[perm[j]])
			}
		}
	}
}

func findParameterC(prob *Problem, param *Parameter, nrFold int, startC float64, maxC float64) *ParameterSearchResult {
	// variables for CV
	l := prob.L
	perm := make([]int, l)
	target := make([]float64, l)
	subProb := make([]*Problem, nrFold)

	// variables for warm start
	var ratio float64 = 2
	prevW := make([][]float64, nrFold)
	numUnchangedW := 0
	param1 := param

	if nrFold > l {
		nrFold = l
		log.Println("WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)")
	}

	foldStart := make([]int, nrFold+1)

	for i := 0; i < l; i++ {
		perm[i] = i
	}

	for i := 0; i < l; i++ {
		j := i + int(rand.Int63n(int64(l-i)))
		swapIntArray(perm, i, j)
	}

	for i := 0; i <= nrFold; i++ {
		foldStart[i] = i * l / nrFold
	}

	for i := 0; i < nrFold; i++ {
		begin := foldStart[i]
		end := foldStart[i+1]
		subProbL := l - (end - begin)

		subProb[i] = NewProblem(subProbL, prob.N, make([]float64, subProbL), make([][]Feature, subProbL), prob.Bias)

		k := 0
		for j := 0; j < begin; j++ {
			subProb[i].X[k] = prob.X[perm[j]]
			subProb[i].Y[k] = prob.Y[perm[j]]
			k++
		}

		for j := end; j < l; j++ {
			subProb[i].X[k] = prob.X[perm[j]]
			subProb[i].Y[k] = prob.Y[perm[j]]
			k++
		}

	}

	bestC := math.NaN()
	bestRate := 0.0

	if startC <= 0 {
		startC = calcStartC(prob, param)
	}
	param1.c = startC

	for param1.c <= maxC {
		for i := 0; i < nrFold; i++ {
			begin := foldStart[i]
			end := foldStart[i+1]

			param1.InitSol = prevW[i]

			var subModel *Model
			var err error
			if subModel, err = Train(subProb[i], param1); err != nil {
				log.Fatalf("Unable to train subproblem... %v", subProb[i])
			}

			var totalWSize int
			if subModel.NumClass == 2 {
				totalWSize = subProb[i].N
			} else {
				totalWSize = subProb[i].N * subModel.NumClass
			}

			if prevW[i] == nil {
				prevW[i] = make([]float64, totalWSize)
				for j := 0; j < totalWSize; j++ {
					prevW[i][j] = subModel.W[j]
				}
			} else if numUnchangedW >= 0 {
				normWDiff := 0.0
				for j := 0; j < totalWSize; j++ {
					normWDiff += (subModel.W[j] - prevW[i][j]) * (subModel.W[j] - prevW[i][j])
					prevW[i][j] = subModel.W[j]
				}
				normWDiff = math.Sqrt(normWDiff)

				if normWDiff > 1e-15 {
					numUnchangedW = -1
				}
			} else {
				for j := 0; j < totalWSize; j++ {
					prevW[i][j] = subModel.W[j]
				}
			}

			for j := begin; j < end; j++ {
				target[perm[j]] = Predict(subModel, prob.X[perm[j]])
			}
		}

		totalCorrect := 0
		for i := 0; i < prob.L; i++ {
			if target[i] == prob.Y[i] {
				totalCorrect++
			}
		}

		currentRate := float64(totalCorrect) / float64(prob.L)
		if currentRate > bestRate {
			bestC = param1.c
			bestRate = currentRate
		}

		log.Printf("log2c=%7.2f\trate=%g\n", math.Log(param1.c)/math.Log(2.0), 100.0*currentRate)
		numUnchangedW++

		if numUnchangedW == 3 {
			break
		}

		param1.c = param1.c * ratio

	}

	if param1.c > maxC && maxC > startC {
		log.Printf("warning: maximum C reached.\n")
	}

	return &ParameterSearchResult{bestC: bestC, bestRate: bestRate}
}

// Predict uses the model to predict the result based on the input features x
func Predict(model *Model, x []Feature) float64 {
	decValues := make([]float64, model.NumClass)
	return predictValues(model, x, decValues)
}

func predictValues(model *Model, x []Feature, decValues []float64) float64 {
	var n int
	if model.Bias >= 0 {
		n = model.NumFeatures + 1
	} else {
		n = model.NumFeatures
	}

	w := model.W

	var nrW int
	if model.NumClass == 2 && model.SolverType != MCSVM_CS {
		nrW = 1
	} else {
		nrW = model.NumClass
	}

	for i := 0; i < nrW; i++ {
		decValues[i] = 0
	}

	for _, lx := range x {
		idx := lx.GetIndex()
		// the dimension of testing data may exceed that of training
		if idx <= n {
			for i := 0; i < nrW; i++ {
				decValues[i] += w[(idx-1)*nrW+i] * lx.GetValue()
			}
		}
	}

	if model.NumClass == 2 {
		if model.SolverType.IsSupportVectorRegression() {
			return decValues[0]
		}
		if decValues[0] > 0 {
			return float64(model.Label[0])
		}

		return float64(model.Label[1])
	}

	var decMaxIdx int
	for i := 1; i < model.NumClass; i++ {
		if decValues[i] > decValues[decMaxIdx] {
			decMaxIdx = i
		}
	}

	return float64(model.Label[decMaxIdx])
}

//PredictProbability gives the probability estimates of each class given the features
func PredictProbability(model *Model, x []Feature, probEstimates []float64) float64 {
	if !model.IsProbabilityModel() {
		sb := "probability output is only supported for logistic regression"
		sb = sb + ". This is currently only supported by the following solvers: "
		var i int
		for _, solverType := range SolverTypeValues() {
			if solverType.IsLogisticRegressionSolver() {
				if i > 0 {
					sb = sb + ", "
				}
				i++
				sb = sb + solverType.Name()
			}
		}
		panic(sb)
	}
	nrClass := model.NumClass
	var nrW int
	if nrClass == 2 {
		nrW = 1
	} else {
		nrW = nrClass
	}

	label := predictValues(model, x, probEstimates)
	for i := 0; i < nrW; i++ {
		probEstimates[i] = 1 / (1 + math.Exp(-probEstimates[i]))
	}

	if nrClass == 2 { // for binary classification
		probEstimates[1] = 1. - probEstimates[0]
	} else {
		var sum float64
		for i := 0; i < nrClass; i++ {
			sum += probEstimates[i]
		}

		for i := 0; i < nrClass; i++ {
			probEstimates[i] = probEstimates[i] / sum
		}
	}

	return label
}

// Train uses the Problem and Parameters to create a training model
func Train(prob *Problem, param *Parameter) (*Model, error) {

	for _, nodes := range prob.X {
		indexBefore := 0
		for _, n := range nodes {
			if n.GetIndex() <= indexBefore {
				panic("feature nodes must be sorted by index in ascending order")
			}
			indexBefore = n.GetIndex()
		}
	}

	if param.InitSol != nil && param.solverType.Name() != L2R_LR.Name() && param.solverType.Name() != L2R_L2LOSS_SVC.Name() {
		panic("Initial-solution specification supported only for solver L2R_LR and L2R_L2LOSS_SVC")
	}

	l := prob.L
	n := prob.N
	wSize := prob.N

	model := &Model{}

	if prob.Bias >= 0 {
		model.NumFeatures = n - 1
	} else {
		model.NumFeatures = n
	}

	model.SolverType = param.solverType
	model.Bias = prob.Bias

	if param.solverType.IsSupportVectorRegression() {
		model.W = make([]float64, wSize)
		model.NumClass = 2
		model.Label = nil

		checkProblemSize(n, model.NumClass)

		trainOne(prob, param, model.W, 0, 0)
	} else {
		perm := make([]int, l)

		// group training data of the same class
		var rv = groupClasses(prob, perm)
		nrClass := rv.nrClass
		label := rv.label
		start := rv.start
		count := rv.count

		checkProblemSize(n, nrClass)

		model.NumClass = nrClass
		model.Label = make([]int, nrClass)
		for i := 0; i < nrClass; i++ {
			model.Label[i] = label[i]
		}

		// calculate weighted C
		weightedC := make([]float64, nrClass)
		for i := 0; i < nrClass; i++ {
			weightedC[i] = param.c
		}
		for i := 0; i < param.GetNumWeights(); i++ {
			var j int
			for j = 0; j < nrClass; j++ {
				if param.weightLabel[i] == label[j] {
					break
				}
			}
			if j == nrClass {
				panic(fmt.Sprintf("class label %d specified in weight is not found", param.weightLabel[i]))
			}
			weightedC[j] *= param.weight[i]
		}

		// constructing the subproblem
		x := make([][]Feature, l)
		for i := 0; i < l; i++ {
			x[i] = prob.X[perm[i]]
		}

		subProb := NewProblem(l, n, make([]float64, l), make([][]Feature, l), 0)

		for k := 0; k < subProb.L; k++ {
			subProb.X[k] = x[k]
		}

		// multi-class svm by Crammer and Singer
		if param.solverType == MCSVM_CS {
			model.W = make([]float64, n*nrClass)
			for i := 0; i < nrClass; i++ {
				for j := start[i]; j < start[i]+count[i]; j++ {
					subProb.Y[j] = float64(i)
				}
			}

			solver := NewSolverMCSVMCS(subProb, nrClass, weightedC, param.eps, 100000)
			solver.solve(model.W)
		} else {
			if nrClass == 2 {
				model.W = make([]float64, wSize)

				e0 := start[0] + count[0]
				k := 0
				for ; k < e0; k++ {
					subProb.Y[k] = 1
				}
				for ; k < subProb.L; k++ {
					subProb.Y[k] = -1
				}

				if param.InitSol != nil {
					for i := 0; i < wSize; i++ {
						model.W[i] = param.InitSol[i]
					}
				} else {
					for i := 0; i < wSize; i++ {
						model.W[i] = 0
					}
				}

				trainOne(subProb, param, model.W, weightedC[0], weightedC[1])
			} else {
				model.W = make([]float64, wSize*nrClass)
				w := make([]float64, wSize)
				for i := 0; i < nrClass; i++ {
					si := start[i]
					ei := si + count[i]

					k := 0
					for ; k < si; k++ {
						subProb.Y[k] = -1
					}
					for ; k < ei; k++ {
						subProb.Y[k] = 1
					}
					for ; k < subProb.L; k++ {
						subProb.Y[k] = -1
					}

					if param.InitSol != nil {
						for j := 0; j < wSize; j++ {
							w[j] = param.InitSol[j*nrClass+i]
						}
					} else {
						for j := 0; j < wSize; j++ {
							w[j] = 0
						}
					}

					trainOne(subProb, param, w, weightedC[i], param.c)

					for j := 0; j < n; j++ {
						model.W[j*nrClass+i] = w[j]
					}
				}
			}
		}

	}
	return model, nil
}

// Calculate the initial C for parameter selection
func calcStartC(prob *Problem, param *Parameter) float64 {
	xTx := 0.0
	maxXTx := 0.0
	for i := 0; i < prob.L; i++ {
		xTx = 0
		for _, xi := range prob.X[i] {
			val := xi.GetValue()
			xTx += val * val
		}
		if xTx > maxXTx {
			maxXTx = xTx
		}
	}

	minC := 1.0
	if param.solverType == L2R_LR {
		minC = 1.0 / (float64(prob.L) * maxXTx)
	} else if param.solverType == L2R_L2LOSS_SVC {
		minC = 1.0 / (2 * float64(prob.L) * maxXTx)
	}

	return math.Pow(2, math.Floor(math.Log(minC)/math.Log(2.0)))
}

func trainOne(prob *Problem, param *Parameter, w []float64, cp float64, cn float64) {
	eps := param.eps
	epsCg := 0.1

	if param.InitSol != nil {
		epsCg = 0.5
	}

	pos := 0
	for i := 0; i < prob.L; i++ {
		if prob.Y[i] > 0 {
			pos++
		}
	}

	neg := prob.L - pos
	primalSolverTol := eps * math.Max(math.Min(float64(pos), float64(neg)), 1) / float64(prob.L)

	switch param.solverType {

	case L2R_LR:
		c := make([]float64, prob.L, prob.L)
		for i := 0; i < prob.L; i++ {
			if prob.Y[i] > 0 {
				c[i] = cp
			} else {
				c[i] = cn
			}
		}
		funObj := NewL2RLRFunc(prob, c)
		tronObj := NewTron(funObj, primalSolverTol, param.MaxIters, epsCg)
		tronObj.tron(w)

	case L2R_L2LOSS_SVC:
		c := make([]float64, prob.L, prob.L)
		for i := 0; i < prob.L; i++ {
			if prob.Y[i] > 0 {
				c[i] = cp
			} else {
				c[i] = cn
			}
		}
		funObj := NewL2RL2SvcFunc(prob, c)
		tronObj := NewTron(funObj, primalSolverTol, param.MaxIters, epsCg)
		tronObj.tron(w)

	case L2R_L2LOSS_SVC_DUAL:
		solveL2RL1L2Svc(prob, w, eps, cp, cn, L2R_L2LOSS_SVC_DUAL, param.MaxIters)

	case L2R_L1LOSS_SVC_DUAL:
		solveL2RL1L2Svc(prob, w, eps, cp, cn, L2R_L1LOSS_SVC_DUAL, param.MaxIters)

	case L1R_L2LOSS_SVC:
		probCol := transpose(prob)
		solveL1RL2Svc(probCol, w, primalSolverTol, cp, cn, param.MaxIters)
	case L1R_LR:
		probCol := transpose(prob)
		solveL1RLR(probCol, w, primalSolverTol, cp, cn, param.MaxIters)
	case L2R_LR_DUAL:
		solveL2RLRDual(prob, w, eps, cp, cn, param.MaxIters)
	case L2R_L2LOSS_SVR:
		c := make([]float64, prob.L)
		for i := 0; i < prob.L; i++ {
			c[i] = param.c
		}

		funObj := NewL2RL2SvrFunc(prob, c, param.P)
		tronObj := NewTron(funObj, param.eps, param.MaxIters, epsCg)
		tronObj.tron(w)
	case L2R_L1LOSS_SVR_DUAL:
		fallthrough
	case L2R_L2LOSS_SVR_DUAL:
		solveL2RL1L2Svr(prob, w, param)

	default:
		panic(fmt.Sprintf("unknown solver type : %+v", param.solverType))
	}
}

func solveL2RL1L2Svr(prob *Problem, w []float64, param *Parameter) {
	l := prob.L
	C := param.c
	p := param.P
	wSize := prob.N
	eps := param.eps
	var i, s, iter int
	maxIter := param.MaxIters
	activeSize := l
	index := make([]int, l)

	var d, G, H float64
	GMaxOld := math.Inf(1)
	var GMaxNew, GNorm1New float64
	GNorm1Init := -1.0 // Gnorm1Init is initialized at the first iteration
	var beta = make([]float64, l)
	QD := make([]float64, l)
	y := prob.Y

	// L2R_L2LOSS_SVR_DUAL
	lambda := []float64{0.5 / C}
	upperBound := []float64{math.Inf(1)}

	if param.solverType == L2R_L1LOSS_SVR_DUAL {
		lambda[0] = 0
		upperBound[0] = C
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for i = 0; i < l; i++ {
		beta[i] = 0
	}

	for i = 0; i < wSize; i++ {
		w[i] = 0
	}

	for i = 0; i < l; i++ {
		xi := prob.X[i]
		QD[i] = SparseOperatorNrm2Sq(xi)
		SparseOperatorAxpy(beta[i], xi, w)

		index[i] = i
	}

	for iter < maxIter {
		GMaxNew = 0
		GNorm1New = 0

		for i = 0; i < activeSize; i++ {
			j := i + random.Intn(activeSize-i)
			swapIntArray(index, i, j)
		}

		for s = 0; s < activeSize; s++ {
			i = index[s]
			G = -y[i] + lambda[GETI_SVR(i)]*beta[i]
			H = QD[i] + lambda[GETI_SVR(i)]

			xi := prob.X[i]
			G += SparseOperatorDot(w, xi)

			Gp := G + p
			Gn := G - p
			var violation float64
			if beta[i] == 0 {
				if Gp < 0 {
					violation = -Gp
				} else if Gn > 0 {
					violation = Gn
				} else if Gp > GMaxOld && Gn < -GMaxOld {
					activeSize--
					swapIntArray(index, s, activeSize)
					s--
					continue
				}
			} else if beta[i] >= upperBound[GETI_SVR(i)] {
				if Gp > 0 {
					violation = Gp
				} else if Gp < -GMaxOld {
					activeSize--
					swapIntArray(index, s, activeSize)
					s--
					continue
				}
			} else if beta[i] <= -upperBound[GETI_SVR(i)] {
				if Gn < 0 {
					violation = -Gn
				} else if Gn > GMaxOld {
					activeSize--
					swapIntArray(index, s, activeSize)
					s--
					continue
				}
			} else if beta[i] > 0 {
				violation = math.Abs(Gp)
			} else {
				violation = math.Abs(Gn)
			}

			GMaxNew = math.Max(GMaxNew, violation)
			GNorm1New += violation

			// obtain Newton direction d
			if Gp < H*beta[i] {
				d = -Gp / H
			} else if Gn > H*beta[i] {
				d = -Gn / H
			} else {
				d = -beta[i]
			}

			if math.Abs(d) < 1.0e-12 {
				continue
			}

			betaOld := beta[i]
			beta[i] = math.Min(math.Max(beta[i]+d, -upperBound[GETI_SVR(i)]), upperBound[GETI_SVR(i)])
			d = beta[i] - betaOld

			if d != 0 {
				SparseOperatorAxpy(d, xi, w)
			}
		}

		if iter == 0 {
			GNorm1Init = GNorm1New
		}
		iter++
		if iter%10 == 0 {
			fmt.Print(".")
		}

		if GNorm1New <= eps*GNorm1Init {
			if activeSize == l {
				break
			} else {
				activeSize = l
				fmt.Print("*")
				GMaxOld = math.Inf(1)
				continue
			}
		}

		GMaxOld = GMaxNew
	}

	logger.Printf("\n[solveL2RL1L2Svr] optimization finished, #iter = %d\n", iter)
	if iter >= maxIter {
		logger.Printf("\n[solveL2RL1L2Svr]WARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n")
	}

	// calculate objective value
	var v float64
	var nSV int
	for i = 0; i < wSize; i++ {
		v += w[i] * w[i]
	}
	v = 0.5 * v
	for i = 0; i < l; i++ {
		v += p*math.Abs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI_SVR(i)]*beta[i]*beta[i]
		if beta[i] != 0 {
			nSV++
		}
	}

	logger.Printf("[solveL2RL1L2Svr] Objective value = %g\n", v)
	logger.Printf("[solveL2RL1L2Svr] nSV = %d\n", nSV)
}

/**
 * A coordinate descent algorithm for
 * the dual of L2-regularized logistic regression problems
 *<pre>
 *  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i) ,
 *     s.t.      0 <= \alpha_i <= upper_bound_i,
 *
 *  where Qij = yi yj xi^T xj and
 *  upper_bound_i = Cp if y_i = 1
 *  upper_bound_i = Cn if y_i = -1
 *
 * Given:
 * x, y, Cp, Cn
 * eps is the stopping tolerance
 *
 * solution will be put in w
 *
 * See Algorithm 5 of Yu et al., MLJ 2010
 *</pre>
 *
 * @since 1.7
 */
func solveL2RLRDual(prob *Problem, w []float64, eps float64, cp float64, cn float64, maxIters int) {
	l := prob.L
	wSize := prob.N
	var i, s, iter int
	xTx := make([]float64, l)
	index := make([]int, l)

	alpha := make([]float64, 2*l)
	y := make([]int8, l)
	maxInnerIter := 100
	innerEps := 1e-2
	innerEpsMin := math.Min(1e-8, eps)
	upperBound := []float64{cn, 0, cp}

	for i = 0; i < l; i++ {
		if prob.Y[i] > 0 {
			y[i] = 1
		} else {
			y[i] = -1
		}
	}

	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for i = 0; i < l; i++ {
		alpha[2*i] = math.Min(0.001*upperBound[GETI(y, i)], 1e-8)
		alpha[2*i+1] = upperBound[GETI(y, i)] - alpha[2*i]
	}

	for i = 0; i < wSize; i++ {
		w[i] = 0
	}

	for i = 0; i < l; i++ {
		xi := prob.X[i]
		xTx[i] = SparseOperatorNrm2Sq(xi)
		SparseOperatorAxpy(float64(y[i])*alpha[2*i], xi, w)
		index[i] = i
	}

	for iter < maxIters {
		for i = 0; i < l; i++ {
			j := i + random.Intn(l-i)
			swapIntArray(index, i, j)
		}
		newtonIter := 0
		GMax := 0.0
		for s = 0; s < l; s++ {
			i = index[s]
			yi := y[i]
			C := upperBound[GETI(y, i)]
			var yWTx, xisq float64 = 0, xTx[i]
			xi := prob.X[i]
			yWTx = float64(yi) * SparseOperatorDot(w, xi)
			a, b := xisq, yWTx

			// Decide to minimize g_1(z) or g_2(z)
			var ind1, ind2, sign int = 2 * i, 2*i + 1, 1
			if 0.5*a*(alpha[ind2]-alpha[ind1])+b < 0 {
				ind1 = 2*i + 1
				ind2 = 2 * i
				sign = -1
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			alphaOld := alpha[ind1]
			z := alphaOld

			if C-z < 0.5*C {
				z = 0.1 * z
			}

			gp := a*(z-alphaOld) + float64(sign)*b + math.Log(z/(C-z))
			GMax = math.Max(GMax, math.Abs(gp))

			// Newton method on the sub-problem
			eta := 0.1
			innerIter := 0
			for innerIter <= maxInnerIter {
				if math.Abs(gp) < innerEps {
					break
				}
				gpp := a + C/(C-z)/z
				tmpz := z - gp/gpp
				if tmpz <= 0 {
					z *= eta
				} else {
					// tmpz in (0,C)
					z = tmpz
				}
				gp = a*(z-alphaOld) + float64(sign)*b + math.Log(z/(C-z))
				newtonIter++
				innerIter++
			}

			if innerIter > 0 { //update w
				alpha[ind1] = z
				alpha[ind2] = C - z
				SparseOperatorAxpy(float64(sign)*(z-alphaOld)*float64(yi), xi, w)
			}
		}

		iter++
		if iter%10 == 0 {
			fmt.Print(".")
		}

		if GMax < eps {
			break
		}

		if newtonIter <= l/10 {
			innerEps = math.Max(innerEpsMin, 0.1*innerEps)
		}
	}

	logger.Printf("\n[solveL2RLRDual] optimization finished, #iter = %d\n", iter)
	if iter >= maxIters {
		logger.Printf("\n[solveL2RLRDual] WARNING: reaching max nunber of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n")
	}

	//calculate objective value

	var v float64
	for i = 0; i < wSize; i++ {
		v += w[i] * w[i]
	}
	v *= 0.5
	for i = 0; i < l; i++ {
		v += alpha[2*i]*math.Log(alpha[2*i]) + alpha[2*i+1]*math.Log(alpha[2*i+1]) - upperBound[GETI(y, i)]*math.Log(upperBound[GETI(y, i)])
	}
	logger.Printf("[solveL2RLRDual] Objective value = %g\n", v)
}

func solveL1RLR(probCol *Problem, w []float64, eps float64, cp float64, cn float64, maxIters int) {
	l := probCol.L
	wSize := probCol.N
	var j, s, newtonIter, iter int
	var maxNewtonIter = 100
	var maxNumLineSearch = 20

	var activeSize int
	var QPActiveSize int

	var nu = 1e-12
	var innerEps float64 = 1
	var sigma = 0.01
	var wNorm, wNormNew float64
	var z, G, H float64
	var GNorm1Init = -1.0
	var GMaxOld = math.Inf(1)
	var GMaxNew, GNorm1New float64
	var QPGMaxOld = math.Inf(1)
	var QPGMaxNew, QPGNorm1New float64
	var delta, negsumXtd, cond float64

	index := make([]int, wSize)
	y := make([]int8, l)
	HDiag := make([]float64, wSize)
	Grad := make([]float64, wSize)
	wpd := make([]float64, wSize)
	xjnegSum := make([]float64, wSize)
	xTd := make([]float64, l)
	expWtx := make([]float64, l)
	expWtxNew := make([]float64, l)
	tau := make([]float64, l)
	D := make([]float64, l)
	C := []float64{cn, 0, cp}

	// Initial w can be set here
	for j = 0; j < wSize; j++ {
		w[j] = 0
	}

	for j = 0; j < l; j++ {
		if probCol.Y[j] > 0 {
			y[j] = 1
		} else {
			y[j] = -1
		}

		expWtx[j] = 0
	}

	wNorm = 0
	for j = 0; j < wSize; j++ {
		wNorm += math.Abs(w[j])
		wpd[j] = w[j]
		index[j] = j
		xjnegSum[j] = 0
		for _, x := range probCol.X[j] {
			ind := x.GetIndex() - 1
			val := x.GetValue()
			expWtx[ind] += w[j] * val
			if y[ind] == -1 {
				xjnegSum[j] += C[GETI(y, ind)] * val
			}
		}
	}

	for j = 0; j < l; j++ {
		expWtx[j] = math.Exp(expWtx[j])
		tauTmp := 1 / (1 + expWtx[j])
		tau[j] = C[GETI(y, j)] * tauTmp
		D[j] = C[GETI(y, j)] * expWtx[j] * tauTmp * tauTmp
	}

	for newtonIter < maxNewtonIter {
		GMaxNew = 0
		GNorm1New = 0
		activeSize = wSize

		for s = 0; s < activeSize; s++ {
			j = index[s]
			HDiag[j] = nu
			Grad[j] = 0

			tmp := 0.0
			for _, x := range probCol.X[j] {
				ind := x.GetIndex() - 1
				HDiag[j] += x.GetValue() * x.GetValue() * D[ind]
				tmp += x.GetValue() * tau[ind]
			}
			Grad[j] = -tmp + xjnegSum[j]

			Gp := Grad[j] + 1
			Gn := Grad[j] - 1

			var violation float64
			if w[j] == 0 {
				if Gp < 0 {
					violation = -Gp
				} else if Gn > 0 {
					violation = Gn
					// outer-level shrinking
				} else if Gp > GMaxOld/float64(l) && Gn < -GMaxOld/float64(l) {
					activeSize--
					swapIntArray(index, s, activeSize)
					s--
					continue
				}
			} else if w[j] > 0 {
				violation = math.Abs(Gp)
			} else {
				violation = math.Abs(Gn)
			}

			GMaxNew = math.Max(GMaxNew, violation)
			GNorm1New += violation
		}

		if newtonIter == 0 {
			GNorm1Init = GNorm1New
		}

		if GNorm1New <= eps*GNorm1Init {
			break
		}

		iter = 0
		QPGMaxOld = math.Inf(1)
		QPActiveSize = activeSize

		for i := 0; i < l; i++ {
			xTd[i] = 0
		}

		// optimize QP over wpd
		for iter < maxIters {
			QPGMaxNew = 0
			QPGNorm1New = 0

			for j = 0; j < QPActiveSize; j++ {
				i := random.Intn(QPActiveSize - j)
				swapIntArray(index, i, j)
			}

			for s = 0; s < QPActiveSize; s++ {
				j = index[s]
				H = HDiag[j]

				G = Grad[j] + (wpd[j]-w[j])*nu
				for _, x := range probCol.X[j] {
					ind := x.GetIndex() - 1
					G += x.GetValue() * D[ind] * xTd[ind]
				}

				Gp := G + 1
				Gn := G - 1
				var violation float64

				if wpd[j] == 0 {
					if Gp < 0 {
						violation = -Gp
					} else if Gn > 0 {
						violation = Gn
						// inner-level shrinking
					} else if Gp > QPGMaxOld/float64(l) && Gn < -QPGMaxOld/float64(l) {
						QPActiveSize--
						swapIntArray(index, s, QPActiveSize)
						s--
						continue
					}
				} else if wpd[j] > 0 {
					violation = math.Abs(Gp)
				} else {
					violation = math.Abs(Gn)
				}

				QPGMaxNew = math.Max(QPGMaxNew, violation)
				QPGNorm1New += violation

				// obtain solution of 1 variable problem
				if Gp < H*wpd[j] {
					z = -Gp / H
				} else if Gn > H*wpd[j] {
					z = -Gn / H
				} else {
					z = -wpd[j]
				}

				if math.Abs(z) < 1.0e-12 {
					continue
				}

				z = math.Min(math.Max(z, -10.0), 10.0)

				wpd[j] += z

				x := probCol.X[j]

				SparseOperatorAxpy(z, x, xTd)
			}

			iter++

			if QPGNorm1New <= innerEps*GNorm1Init {
				//inner stopping
				if QPActiveSize == activeSize {
					break
				} else {
					QPActiveSize = activeSize
					QPGMaxOld = math.Inf(1)
					continue
				}
			}

			QPGMaxOld = QPGMaxNew
		}

		if iter >= maxIters {
			logger.Print("[solveL1RLR] WARNING: reaching max number of inner iterations\n")
		}

		delta = 0
		wNormNew = 0
		for j := 0; j < wSize; j++ {
			delta += Grad[j] * (wpd[j] - w[j])
			if wpd[j] != 0 {
				wNormNew += math.Abs(wpd[j])
			}
		}
		delta += wNormNew - wNorm

		negsumXtd = 0
		for i := 0; i < l; i++ {
			if y[i] == -1 {
				negsumXtd += C[GETI(y, i)] * xTd[i]
			}
		}

		numLineSearch := 0

		for numLineSearch = 0; numLineSearch < maxNumLineSearch; numLineSearch++ {
			cond = wNormNew - wNorm + negsumXtd - sigma*delta

			for i := 0; i < l; i++ {
				expXtd := math.Exp(xTd[i])
				expWtxNew[i] = expWtx[i] * expXtd
				cond += C[GETI(y, i)] * math.Log((1+expWtxNew[i])/(expXtd+expWtxNew[i]))
			}
			if cond <= 0 {
				wNorm = wNormNew
				for j = 0; j < wSize; j++ {
					w[j] = wpd[j]
				}

				for i := 0; i < l; i++ {
					expWtx[i] = expWtxNew[i]
					tauTmp := 1 / (1 + expWtx[i])
					tau[i] = C[GETI(y, i)] * tauTmp
					D[i] = C[GETI(y, i)] * expWtx[i] * tauTmp * tauTmp
				}
				break
			} else {
				wNormNew = 0
				for j = 0; j < wSize; j++ {
					wpd[j] = (w[j] + wpd[j]) * 0.5
					if wpd[j] != 0 {
						wNormNew += math.Abs(wpd[j])
					}
				}
				delta *= 0.5
				negsumXtd *= 0.5
				for i := 0; i < l; i++ {
					xTd[i] *= 0.5
				}
			}
		}

		// Recompute some info due to too many line search steps
		if numLineSearch >= maxNumLineSearch {
			for i := 0; i < l; i++ {
				expWtx[i] = 0
			}

			for i := 0; i < wSize; i++ {
				if w[i] == 0 {
					continue
				}
				x := probCol.X[i]
				SparseOperatorAxpy(w[i], x, expWtx)
			}

			for i := 0; i < l; i++ {
				expWtx[i] = math.Exp(expWtx[i])
			}
		}

		if iter == 1 {
			innerEps *= 0.25
		}

		newtonIter++
		GMaxOld = GMaxNew

		logger.Printf("[solveL1RLR] iter %3d #CD cycles %d\n", newtonIter, iter)
	}

	logger.Println("===============================")
	logger.Printf("[solveL1RLR] optimization finished, #iter = %d\n", newtonIter)
	if newtonIter >= maxNewtonIter {
		logger.Println("WARNING: reaching max number of iterations")
	}

	// calculate objective value
	var v float64
	var nnz int

	for j = 0; j < wSize; j++ {
		if w[j] != 0 {
			v += math.Abs(w[j])
			nnz++
		}
	}

	for j = 0; j < l; j++ {
		if y[j] == 1 {
			v += C[GETI(y, j)] * math.Log(1+1/expWtx[j])
		} else {
			v += C[GETI(y, j)] * math.Log(1+expWtx[j])
		}
	}

	logger.Printf("[solveL1RLR] Objective value = %g\n", v)
	logger.Printf("[solveL1RLR] #nonzeros/#features = %d/%d\n", nnz, wSize)

}

func transpose(prob *Problem) *Problem {
	l := prob.L
	n := prob.N

	colPtr := make([]int, n+1)
	probCol := NewProblem(l, n, make([]float64, l), make([][]Feature, n), prob.Bias)

	for i := 0; i < l; i++ {
		probCol.Y[i] = prob.Y[i]
	}

	for i := 0; i < l; i++ {
		for _, x := range prob.X[i] {
			colPtr[x.GetIndex()]++
		}
	}

	for i := 0; i < n; i++ {
		probCol.X[i] = make([]Feature, colPtr[i+1])
		colPtr[i] = 0
	}

	for i := 0; i < l; i++ {
		for _, x := range prob.X[i] {
			index := x.GetIndex() - 1
			probCol.X[index][colPtr[index]] = NewFeatureNode(i+1, x.GetValue())
			colPtr[index]++
		}
	}

	return probCol
}

func solveL1RL2Svc(probCol *Problem, w []float64, eps float64, cp float64, cn float64, maxIter int) {
	l := probCol.L
	wSize := probCol.N
	var j, s, iter int
	activeSize := wSize
	maxNumLineSearch := 20

	sigma := 0.01
	var d, GLoss, G, H float64
	GMaxOld := math.Inf(1)
	var GMaxNew, GNorm1New float64
	GNorm1Init := -1.0
	var dOld, dDiff float64
	var lossOld, lossNew float64
	var appxcond, cond float64

	index := make([]int, wSize)
	y := make([]int8, l)
	b := make([]float64, l)
	xjSq := make([]float64, wSize)

	C := []float64{cn, 0, cp}

	for j = 0; j < wSize; j++ {
		w[j] = 0
	}

	for j = 0; j < l; j++ {
		b[j] = 1
		if probCol.Y[j] > 0 {
			y[j] = 1
		} else {
			y[j] = -1
		}
	}

	for j = 0; j < wSize; j++ {
		index[j] = j
		xjSq[j] = 0
		for _, xi := range probCol.X[j] {
			ind := xi.GetIndex() - 1
			xi.SetValue(xi.GetValue() * float64(y[ind]))
			val := xi.GetValue()
			b[ind] -= w[j] * val

			xjSq[j] += C[GETI(y, ind)] * val * val
		}
	}

	for iter < maxIter {
		GMaxNew = 0
		GNorm1New = 0

		for j = 0; j < activeSize; j++ {
			i := j + random.Intn(activeSize-j)
			swapIntArray(index, i, j)
		}

		for s = 0; s < activeSize; s++ {
			j = index[s]
			GLoss = 0
			H = 0

			for _, xi := range probCol.X[j] {
				ind := xi.GetIndex() - 1
				if b[ind] > 0 {
					val := xi.GetValue()
					tmp := C[GETI(y, ind)] * val
					GLoss -= tmp * b[ind]
					H += tmp * val
				}
			}

			GLoss *= 2
			G = GLoss
			H *= 2
			H = math.Max(H, 1e-12)

			var Gp = G + 1
			var Gn = G - 1
			var violation = 0.0

			if w[j] == 0 {
				if Gp < 0 {
					violation = -Gp
				} else if Gn > 0 {
					violation = Gn
				} else if Gp > GMaxOld/float64(l) && Gn < -GMaxOld/float64(l) {
					activeSize--
					swapIntArray(index, s, activeSize)
					s--
					continue
				}
			} else if w[j] > 0 {
				violation = math.Abs(Gp)
			} else {
				violation = math.Abs(Gn)
			}

			GMaxNew = math.Max(GMaxNew, violation)
			GNorm1New += violation

			// obtain Newton direction d
			if Gp < H*w[j] {
				d = -Gp / H
			} else if Gn > H*w[j] {
				d = -Gn / H
			} else {
				d = -w[j]
			}

			if math.Abs(d) < 1.0e-12 {
				continue
			}

			delta := math.Abs(w[j]+d) - math.Abs(w[j]) + G*d
			dOld = 0

			var numLineSearch int

			for numLineSearch = 0; numLineSearch < maxNumLineSearch; numLineSearch++ {
				dDiff = dOld - d
				cond = math.Abs(w[j]+d) - math.Abs(w[j]) - sigma*delta

				appxcond = xjSq[j]*d*d + GLoss*d + cond
				if appxcond <= 0 {
					x := probCol.X[j]
					SparseOperatorAxpy(dDiff, x, b)
					break
				}

				if numLineSearch == 0 {
					lossOld = 0
					lossNew = 0
					for _, x := range probCol.X[j] {
						ind := x.GetIndex() - 1
						if b[ind] > 0 {
							lossOld += C[GETI(y, ind)] * b[ind] * b[ind]
						}
						bNew := b[ind] + dDiff*x.GetValue()
						b[ind] = bNew
						if bNew > 0 {
							lossNew += C[GETI(y, ind)] * bNew * bNew
						}
					}
				} else {
					lossNew = 0
					for _, x := range probCol.X[j] {
						ind := x.GetIndex() - 1
						bNew := b[ind] + dDiff*x.GetValue()
						b[ind] = bNew
						if bNew > 0 {
							lossNew += C[GETI(y, ind)] * bNew * bNew
						}
					}
				}

				cond = cond + lossNew - lossOld
				if cond <= 0 {
					break
				} else {
					dOld = d
					d *= 0.5
					delta *= 0.5
				}
			}

			w[j] += d

			//recompute b[] if line search takes too many steps
			if numLineSearch >= maxNumLineSearch {
				fmt.Print("#")
				for i := 0; i < l; i++ {
					b[i] = 1
				}
				for i := 0; i < wSize; i++ {
					if w[i] == 0 {
						continue
					}
					x := probCol.X[i]
					SparseOperatorAxpy(-w[i], x, b)
				}
			}
		}

		if iter == 0 {
			GNorm1Init = GNorm1New
		}

		iter++
		if iter%10 == 0 {
			fmt.Print(".")
		}

		if GNorm1New <= eps*GNorm1Init {
			if activeSize == wSize {
				break
			} else {
				activeSize = wSize
				fmt.Print("*")
				GMaxOld = math.Inf(1)
				continue
			}
		}

		GMaxOld = GMaxNew
	}

	logger.Printf("\n[solveL1RL2Svcoptimization finished, #iter = %d\n", iter)
	if iter >= maxIter {
		logger.Printf("\n[solveL1RL2Svc] WARNING: reaching max number of iterations\n")
	}

	//calculate objective value
	v := 0.0
	nnz := 0
	for j := 0; j < wSize; j++ {
		for _, x := range probCol.X[j] {
			x.SetValue(x.GetValue() * probCol.Y[x.GetIndex()-1])
		}
		if w[j] != 0 {
			v += math.Abs(w[j])
			nnz++
		}
	}

	for j := 0; j < l; j++ {
		if b[j] > 0 {
			v += C[GETI(y, j)] * b[j] * b[j]
		}
	}

	logger.Printf("[solveL1RL2Svc] Objective value = %g\n", v)
	logger.Printf("[solveL1RL2Svc] #nonzeros/#features = %d/%d\n", nnz, wSize)

}

func solveL2RL1L2Svc(prob *Problem, w []float64, eps float64, cp float64, cn float64, solverType *SolverType, maxIter int) {
	l := prob.L
	wSize := prob.N
	var i, s, iter int
	var C, d, G float64
	QD := make([]float64, l)
	index := make([]int, l)
	alpha := make([]float64, l)
	y := make([]int8, l)
	activeSize := l

	// PG: projected gradient, for shrinking and stopping
	var PG float64
	var PGMaxOld = math.Inf(1)
	var PGMinOld = math.Inf(-1)
	var PGMaxNew, PGMinNew float64

	// default solverType: L2R_L2LOSS_SVC_DUAL
	diag := []float64{0.5 / cn, 0, 0.5 / cp}
	upperBound := []float64{math.Inf(1), 0, math.Inf(1)}
	if solverType == L2R_L1LOSS_SVC_DUAL {
		diag[0] = 0
		diag[2] = 0
		upperBound[0] = cn
		upperBound[2] = cp
	}

	for i = 0; i < l; i++ {
		if prob.Y[i] > 0 {
			y[i] = 1
		} else {
			y[i] = -1
		}
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for i = 0; i < l; i++ {
		alpha[i] = 0
	}

	for i = 0; i < wSize; i++ {
		w[i] = 0
	}

	for i = 0; i < l; i++ {
		QD[i] = diag[GETI(y, i)]
		xi := prob.X[i]
		QD[i] += SparseOperatorNrm2Sq(xi)
		SparseOperatorAxpy(float64(y[i])*alpha[i], xi, w)
		index[i] = i
	}

	for iter < maxIter {
		PGMaxNew = math.Inf(-1)
		PGMinNew = math.Inf(1)

		for i = 0; i < activeSize; i++ {
			j := i + random.Intn(activeSize-i)
			swapIntArray(index, i, j)
		}

		for s = 0; s < activeSize; s++ {
			i = index[s]
			yi := y[i]
			xi := prob.X[i]

			G = float64(yi)*SparseOperatorDot(w, xi) - 1

			C = upperBound[GETI(y, i)]
			G += alpha[i] * diag[GETI(y, i)]

			PG = 0.0
			if alpha[i] == 0 {
				if G > PGMaxOld {
					activeSize--
					swapIntArray(index, s, activeSize)
					s--
					continue
				} else if G < 0 {
					PG = G
				}
			} else if alpha[i] == C {
				if G < PGMinOld {
					activeSize--
					swapIntArray(index, s, activeSize)
					s--
					continue
				} else if G > 0 {
					PG = G
				}
			} else {
				PG = G
			}

			PGMaxNew = math.Max(PGMaxNew, PG)
			PGMinNew = math.Min(PGMinNew, PG)

			if math.Abs(PG) > 1.0e-12 {
				alphaOld := alpha[i]
				alpha[i] = math.Min(math.Max(alpha[i]-G/QD[i], 0.0), C)
				d = (alpha[i] - alphaOld) * float64(yi)
				SparseOperatorAxpy(d, xi, w)
			}
		}

		iter++
		if iter%10 == 0 {
			fmt.Print(".")
		}

		if PGMaxNew-PGMinNew <= eps {
			if activeSize == l {
				break
			} else {
				activeSize = l
				fmt.Print("*")
				PGMaxOld = math.Inf(1)
				PGMaxOld = math.Inf(-1)
				continue
			}
		}
		PGMaxOld = PGMaxNew
		PGMinOld = PGMinNew

		if PGMaxOld <= 0 {
			PGMaxOld = math.Inf(1)
		}
		if PGMinOld >= 0 {
			PGMinOld = math.Inf(-1)
		}

	}

	logger.Printf("\n[solveL1RL2Svc] optimization finished, #iter = %d\n", iter)

	if iter >= maxIter {
		logger.Printf("\n[solveL1RL2Svc] WARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n")
	}

	// calculate objective value

	var v float64
	nSV := 0
	for i = 0; i < wSize; i++ {
		v += w[i] * w[i]
	}
	for i = 0; i < l; i++ {
		v += alpha[i] * (alpha[i]*diag[GETI(y, i)] - 2)
		if alpha[i] > 0 {
			nSV++
		}
	}
	logger.Printf("[solveL1RL2Svc] Objective value = %g\n", v/2)
	logger.Printf("[solveL1RL2Svc] nSV = %d\n", nSV)
}

func swapIntArray(array []int, idxA int, idxB int) {
	temp := array[idxA]
	array[idxA] = array[idxB]
	array[idxB] = temp
}

func swapFloat64Array(array []float64, idxA int, idxB int) {
	temp := array[idxA]
	array[idxA] = array[idxB]
	array[idxB] = temp
}

func swapIntArrayPointer(array *IntArrayPointer, idxA int, idxB int) {
	temp := array.get(idxA)
	array.set(idxA, array.get(idxB))
	array.set(idxB, temp)
}

// GETI method corresponds to the following define in the C version:
func GETI(y []int8, i int) int {
	return int(y[i] + 1)
}

// GETI_SVR :To support weights for instances, use GETI(i) (i)
func GETI_SVR(i int) int {
	return 0
}

// SaveModel saves the model file
func SaveModel(modelFile *os.File, model *Model) {

	w := bufio.NewWriter(modelFile)

	nrFeature := model.NumFeatures
	wSize := nrFeature
	if model.Bias >= 0 {
		wSize++
	}

	nrW := model.NumClass
	if model.NumClass == 2 && model.SolverType != MCSVM_CS {
		nrW = 1
	}

	w.WriteString(fmt.Sprintf("solver_type %s\n", model.SolverType.Name()))
	w.WriteString(fmt.Sprintf("nr_class %d\n", model.NumClass))

	if model.Label != nil {
		w.WriteString("label")
		for i := 0; i < model.NumClass; i++ {
			w.WriteString(fmt.Sprintf(" %d", model.Label[i]))
		}
		w.WriteString("\n")
	}

	w.WriteString(fmt.Sprintf("nr_feature %d\n", nrFeature))
	w.WriteString(fmt.Sprintf("bias %.16g\n", model.Bias))

	w.WriteString("w\n")
	for i := 0; i < wSize; i++ {
		for j := 0; j < nrW; j++ {
			value := model.W[i*nrW+j]
			if value == 0.0 {
				w.WriteString(fmt.Sprintf("%d ", 0))
			} else {
				w.WriteString(fmt.Sprintf("%.16g ", value))
			}
		}
		w.WriteString("\n")
	}

	w.Flush()
}

func LoadModel(f *os.File) *Model {

	var r = regexp.MustCompile("\\s+")
	var bias float64
	var solverType *SolverType
	var nrClasses, nrFeatures int64
	var label []int

	reader := bufio.NewReader(f)

header:
	for {
		line, err := reader.ReadString('\n')

		if err == io.EOF {
			break
		}

		if err != nil {
			panic("unable to read file :" + f.Name())
		}

		split := r.Split(line, -1)

		switch split[0] {

		case "solver_type":
			solver := getSolverType(split[1])
			if solver == nil {
				panic("unknown solver type")
			}
			solverType = solver

		case "nr_class":
			nrClasses, _ = strconv.ParseInt(split[1], 10, 32)

		case "nr_feature":
			nrFeatures, _ = strconv.ParseInt(split[1], 10, 32)

		case "bias":
			bias, _ = strconv.ParseFloat(split[1], 32)

		case "w":
			break header

		case "label":
			label = make([]int, nrClasses)
			for i := 0; i < int(nrClasses); i++ {
				val, _ := strconv.ParseInt(split[i+1], 10, 32)
				label[i] = int(val)
			}

		default:
			panic("unknown text in model file: [" + line + "]")
		}
	}
	wSize := nrFeatures
	if bias >= 0 {
		wSize++
	}

	nrW := nrClasses
	if nrClasses == 2 && solverType != MCSVM_CS {
		nrW = 1
	}

	w := make([]float64, wSize*nrW)
	buffer := make([]rune, 128)

	for i := int64(0); i < wSize; i++ {
		for j := int64(0); j < nrW; j++ {
			b := 0

		WeightLoop:
			for {
				ch, s, _ := reader.ReadRune()

				if s > 1 || ch == 0xfffd {
					panic("unexpected EOF")
				}

				switch ch {
				case ' ':
					w[i*nrW+j], _ = strconv.ParseFloat(string(buffer[:b]), 64)
					break WeightLoop
				case '\n':
					continue
				default:
					if b >= len(buffer) {
						panic(fmt.Sprintf("illegal weight in model file at index %d with string content %s, is not terminated with a whitespace character, or is longer than expected (%d characters max).", i*nrW+j, string(buffer[:b]), len(buffer)))
					}
					buffer[b] = ch
					b++
				}
			}
		}
	}

	model := NewModel(float64(bias), label, int(nrClasses), int(nrFeatures), solverType, w)

	return model
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

func CreateRandomModel() *Model {
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

func CreateRandomProblem(numClasses int) *Problem {
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

func checkProblemSize(n int, numClass int) {
	if n >= math.MaxInt32/numClass || n*numClass < 0 {
		panic(fmt.Sprintf("'number of classes' * 'number of instances' is too large: %d * %d", numClass, n))
	}
}
