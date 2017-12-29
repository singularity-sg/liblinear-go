package liblinear

// Parameter contains the weights for solving
type Parameter struct {
	c           float64
	eps         float64 // Stopping criteria
	maxIters    int
	solverType  *SolverType
	weight      []float64
	weightLabel []int
	p           float64
	initSol     []float64
}

// NewParameter constructs a Parameter
func NewParameter(solverType *SolverType, c float64, eps float64, p float64, maxIters int) *Parameter {
	parameter := &Parameter{
		solverType: solverType,
		c:          c,
		eps:        eps,
		maxIters:   maxIters,
		p:          p,
	}

	return parameter
}

// GetNumWeights gets the weights
func (p *Parameter) GetNumWeights() int {
	if p.weight == nil {
		return 0
	}
	return len(p.weight)
}
