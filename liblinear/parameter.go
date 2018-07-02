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

// NewParameter constructs a Parameter
func NewParameter(SolverType *SolverType, c float64, eps float64, p float64, maxIters int) *Parameter {
	parameter := &Parameter{
		SolverType: SolverType,
		C:          c,
		Eps:        eps,
		MaxIters:   maxIters,
		P:          p,
	}

	return parameter
}

// GetNumWeights gets the weights
func (p *Parameter) GetNumWeights() int {
	if p.Weight == nil {
		return 0
	}
	return len(p.Weight)
}
