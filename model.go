package liblinear

// Model is a struct containing the data of a trained model
type Model struct {
	Bias        float64
	Label       []int
	NumClass    int
	NumFeatures int
	SolverType  *SolverType
	W           []float64
}

func NewModel(bias float64, label []int, numClass int, numFeatures int, solverType *SolverType, w []float64) *Model {
	return &Model{
		Bias:        bias,
		Label:       label,
		NumClass:    numClass,
		NumFeatures: numFeatures,
		SolverType:  solverType,
		W:           w,
	}
}
