package liblinear

// L2R_LR : L2-regularized logistic regression (primal)
var L2R_LR = NewSolverType(0, "L2R_LR", true, false)

// L2R_L2_LOSS_SVC_DUAL : L2-regularized L2-loss support vector classification (dual)
var L2R_L2LOSS_SVC_DUAL = NewSolverType(1, "L2R_L2LOSS_SVC_DUAL", false, false)

// L2R_L2_LOSS_SVC : L2-regularized L2-loss support vector classification (primal)
var L2R_L2LOSS_SVC = NewSolverType(2, "L2R_L2LOSS_SVC", false, false)

// L2R_L1_LOSS_SVC_DUAL : L2-regularized L1-loss support vector classification (dual)
var L2R_L1LOSS_SVC_DUAL = NewSolverType(3, "L2R_L1LOSS_SVC_DUAL", false, false)

// MCSVM_CS : multi-class support vector classification by Crammer and Singer
var MCSVM_CS = NewSolverType(4, "MCSVM_CS", false, false)

// L1R_L2LOSS_SVC : L1-regularized L2-loss support vector classification
var L1R_L2LOSS_SVC = NewSolverType(5, "L1R_L2LOSS_SVC", false, false)

// L1R_LR : L1-regularized logistic regression
var L1R_LR = NewSolverType(6, "L1R_LR", true, false)

// L2R_LR_DUAL : L2-regularized logistic regression (dual)
var L2R_LR_DUAL = NewSolverType(7, "L2R_LR_DUAL", true, false)

// L2R_L2LOSS_SVR : L2-regularized L2-loss support vector regression (dual)
var L2R_L2LOSS_SVR = NewSolverType(11, "L2R_L2LOSS_SVR", false, true)

// L2R_L2LOSS_SVR_DUAL : L2-regularized L1-loss support vector regression (dual)
var L2R_L2LOSS_SVR_DUAL = NewSolverType(12, "L2R_L2LOSS_SVR_DUAL", false, true)

// L2R_L1LOSS_SVR_DUAL : L1-regularized L2-loss support vector regression (primal)
var L2R_L1LOSS_SVR_DUAL = NewSolverType(13, "L2R_L1LOSS_SVR_DUAL", false, true)

var solverTypeValues = []*SolverType{
	L2R_LR,
	L2R_L2LOSS_SVC_DUAL,
	L2R_L2LOSS_SVC,
	L2R_L1LOSS_SVC_DUAL,
	MCSVM_CS,
	L1R_L2LOSS_SVC,
	L1R_LR,
	L2R_LR_DUAL,
	L2R_L2LOSS_SVR,
	L2R_L2LOSS_SVR_DUAL,
	L2R_L1LOSS_SVR_DUAL,
}

// SolverType describes the properties of the solver
type SolverType struct {
	name                     string
	logisticRegressionSolver bool
	supportVectorRegression  bool
	id                       int
}

// NewSolverType returns a new SolverType based on input fields
func NewSolverType(id int, name string, logisticRegressionSolver bool, supportVectorRegression bool) *SolverType {
	return &SolverType{
		id:   id,
		name: name,
		logisticRegressionSolver: logisticRegressionSolver,
		supportVectorRegression:  supportVectorRegression,
	}
}

// SolverTypeValues gives a list of SolverTypes
func SolverTypeValues() []*SolverType {
	return solverTypeValues
}

// Name is nameless
func (solverType *SolverType) Name() string {
	return solverType.name
}

// IsSupportVectorRegression returns if this solver type supports vector regression
func (solverType *SolverType) IsSupportVectorRegression() bool {
	return solverType.supportVectorRegression
}

// IsLogisticRegressionSolver returns if this solver type is a logistic regression solver
func (solverType *SolverType) IsLogisticRegressionSolver() bool {
	return solverType.logisticRegressionSolver
}
