package liblinear

import (
	"fmt"
)

// Parameter contains the weights for solving
type Parameter struct {
	c           float64
	eps         float64 // Stopping criteria
	MaxIters    int
	solverType  *SolverType
	weight      []float64
	weightLabel []int
	P           float64
	InitSol     []float64
}

// NewParameter constructs a Parameter
func NewParameter(solverType *SolverType, c float64, eps float64, p float64, maxIters int) *Parameter {
	parameter := &Parameter{
		solverType: solverType,
		c:          c,
		eps:        eps,
		MaxIters:   maxIters,
		P:          p,
	}

	return parameter
}

// SetWeight sets the weight labels and weights.
// It returns an error if the weight and weightLabel length is different
func (p *Parameter) SetWeight(weight []float64, weightLabel []int) error {
	if len(weight) != len(weightLabel) {
		return fmt.Errorf("The weight has a different length compared to weightLabel. %d vs %d", len(weight), len(weightLabel))
	}

	p.weight = weight
	p.weightLabel = weightLabel

	return nil
}

// GetNumWeights gets the weights
func (p *Parameter) GetNumWeights() int {
	if p.weight == nil {
		return 0
	}
	return len(p.weight)
}

// AddWeight adds a new weight to the parameter
func (p *Parameter) AddWeight(weight float64, weightLabel int) {
	p.weight = append(p.weight, weight)
	p.weightLabel = append(p.weightLabel, weightLabel)
}

// GetWeight returns the weight at the specified index
func (p *Parameter) GetWeight(idx int) float64 {
	return p.weight[idx]
}

// GetWeights returns a copy of the weights in the parameter
func (p *Parameter) GetWeights() []float64 {
	wCopy := make([]float64, len(p.weight))

	for idx, x := range p.weight {
		wCopy[idx] = x
	}

	return wCopy
}

// GetWeightLabels return a copy of the weightLabels in the parameter
func (p *Parameter) GetWeightLabels() []int {
	wlCopy := make([]int, len(p.weightLabel))

	for idx, x := range p.weightLabel {
		wlCopy[idx] = x
	}

	return wlCopy
}

// GetWeightLabel returns the label at the specified index
func (p *Parameter) GetWeightLabel(idx int) int {
	return p.weightLabel[idx]
}

// SetC does what it says
func (p *Parameter) SetC(c float64) error {
	if c <= 0 {
		return fmt.Errorf("c should not be <= 0")
	}

	p.c = c
	return nil
}

// SetEps does what it says
func (p *Parameter) SetEps(eps float64) error {
	if eps <= 0 {
		return fmt.Errorf("eps should not be <= 0")
	}

	p.eps = eps
	return nil
}

// GetEps returns the eps
func (p *Parameter) GetEps() float64 {
	return p.eps
}

// SetSolverType does just that
func (p *Parameter) SetSolverType(solverType *SolverType) error {
	if solverType == nil {
		return fmt.Errorf("The solvertype cannot be nil")
	}
	p.solverType = solverType
	return nil
}

// GetSolverType returns the solverType for this param
func (p *Parameter) GetSolverType() *SolverType {
	return p.solverType
}
