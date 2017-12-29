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

// NewModel creates a new instance of struct
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

/*
 * The array w gives feature weights; its size is
 * nr_feature*nr_class but is nr_feature if nr_class = 2. We use one
 * against the rest for multi-class classification, so each feature
 * index corresponds to nr_class weight values. Weights are
 * organized in the following way
 *
 * <pre>
 * +------------------+------------------+------------+
 * | nr_class weights | nr_class weights |  ...
 * | for 1st feature  | for 2nd feature  |
 * +------------------+------------------+------------+
 * </pre>
 *
 * If bias &gt;= 0, x becomes [x; bias]. The number of features is
 * increased by one, so w is a (nr_feature+1)*nr_class array. The
 * value of bias is stored in the variable bias.
 * @see #getBias()
 * @return a <b>copy of</b> the feature weight array as described
 */
// GetFeatureWeights returns the weights*/
func (model *Model) GetFeatureWeights() []float64 {
	var weights = make([]float64, len(model.W))
	copy(weights, model.W)

	return weights
}

/**
 * @return true for logistic regression solvers
 */
func (model *Model) isProbabilityModel() bool {
	return model.SolverType.IsLogisticRegressionSolver()
}
