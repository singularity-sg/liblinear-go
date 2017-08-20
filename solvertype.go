package liblinear

// L2R_LR : L2-regularized logistic regression (primal)
var L2R_LR = NewSolverType(0, true, false)

// L2R_L2_LOSS_SVC_DUAL : L2-regularized L2-loss support vector classification (dual)
var L2R_L2LOSS_SVC_DUAL = NewSolverType(1, false, false)

// L2R_L2_LOSS_SVC : L2-regularized L2-loss support vector classification (primal)
var L2R_L2LOSS_SVC = NewSolverType(2, false, false)

// L2R_L1_LOSS_SVC_DUAL : L2-regularized L1-loss support vector classification (dual)
var L2R_L1LOSS_SVC_DUAL = NewSolverType(3, false, false)

// MCSVM_CS : multi-class support vector classification by Crammer and Singer
var MCSVM_CS = NewSolverType(4, false, false)

// L1R_L2LOSS_SVC : L1-regularized L2-loss support vector classification
var L1R_L2LOSS_SVC = NewSolverType(5, false, false)

// L1R_LR : L1-regularized logistic regression
var L1R_LR = NewSolverType(6, true, false)

// L2R_LR_DUAL : L2-regularized logistic regression (dual)
var L2R_LR_DUAL = NewSolverType(7, true, false)

// L2R_L2LOSS_SVR : L2-regularized L2-loss support vector regression (dual)
var L2R_L2LOSS_SVR = NewSolverType(11, false, true)

// L2R_L2LOSS_SVR_DUAL : L2-regularized L1-loss support vector regression (dual)
var L2R_L2LOSS_SVR_DUAL = NewSolverType(12, false, true)

// L2R_L1LOSS_SVR_DUAL : L1-regularized L2-loss support vector regression (primal)
var L2R_L1LOSS_SVR_DUAL = NewSolverType(13, false, true)

// SolverType describes the properties of the solver
type SolverType struct {
	logisticRegressionSolver bool
	supportVectorRegression  bool
	id                       int
}

// NewSolverType returns a new SolverType based on input fields
func NewSolverType(id int, logisticRegressionSolver bool, supportVectorRegression bool) *SolverType {
	return &SolverType{
		id: id,
		logisticRegressionSolver: logisticRegressionSolver,
		supportVectorRegression:  supportVectorRegression,
	}
}

// IsSupportVectorRegression returns if this solver type supports vector regression
func (solverType *SolverType) IsSupportVectorRegression() bool {
	return solverType.supportVectorRegression
}

// IsLogisticRegressionSolver returns if this solver type is a logistic regression solver
func (solverType *SolverType) IsLogisticRegressionSolver() bool {
	return solverType.logisticRegressionSolver
}
