package liblinear

// Parameter contains the weights for solving
type Parameter struct {
	C           float64
	Eps         float64 // Stopping criteria
	MaxIters    int
	SolverType  *SolverType
	Weight      []float64
	WeightLabel []int
	P           float64
	InitSol     []float64
}

func newParameter(solverType *SolverType, c float64, eps float64, maxIters int, p float64) *Parameter {
	parameter := &Parameter{
		SolverType: solverType,
		C:          c,
		Eps:        eps,
		MaxIters:   maxIters,
		P:          p,
	}

	return parameter
}


