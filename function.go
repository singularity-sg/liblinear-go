package liblinear

// Function provides the interface for different implementations of logistic regression functions
type Function interface {
	fun(w []float64) float64

	grad(w []float64, g []float64)

	hv(s []float64, hs []float64)

	getNrVariable() int
}
