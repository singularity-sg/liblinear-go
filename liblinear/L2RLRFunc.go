package liblinear

import "math"

// L2RLRFunc provides the structure for this function
type L2RLRFunc struct {
	c    []float64
	z    []float64
	d    []float64
	prob *Problem
}

// NewL2RLRFunc a new instance of this struct
func NewL2RLRFunc(prob *Problem, c []float64) *L2RLRFunc {
	return &L2RLRFunc{
		prob: prob,
		c:    c,
		z:    make([]float64, prob.L),
		d:    make([]float64, prob.L),
	}
}

func (fn *L2RLRFunc) fun(w []float64) float64 {
	var f float64

	y := fn.prob.Y
	l := fn.prob.L
	wSize := fn.getNrVariable()

	fn.xv(w, fn.z)

	for i := 0; i < wSize; i++ {
		f += w[i] * w[i]
	}
	f /= 2.0

	for i := 0; i < l; i++ {
		yz := y[i] * fn.z[i]
		if yz >= 0 {
			f += fn.c[i] * math.Log(1+math.Exp(-yz))
		} else {
			f += fn.c[i] * (-yz + math.Log(1+math.Exp(yz)))
		}
	}

	return f
}

func (fn *L2RLRFunc) grad(w []float64, g []float64) {
	y := fn.prob.Y
	l := fn.prob.L
	wSize := fn.getNrVariable()

	for i := 0; i < l; i++ {
		fn.z[i] = 1 / (1 + math.Exp(-y[i]*fn.z[i]))
		fn.d[i] = fn.z[i] * (1 - fn.z[i])
		fn.z[i] = fn.c[i] * (fn.z[i] - 1) * y[i]
	}

	fn.xTv(fn.z, g)

	for i := 0; i < wSize; i++ {
		g[i] = w[i] + g[i]
	}
}

func (fn *L2RLRFunc) xv(v []float64, xv []float64) {
	l := fn.prob.L
	x := fn.prob.X

	for i := 0; i < l; i++ {
		xv[i] = SparseOperatorDot(v, x[i])
	}
}

func (fn *L2RLRFunc) xTv(v []float64, xTv []float64) {
	l := fn.prob.L
	wSize := fn.getNrVariable()
	x := fn.prob.X

	for i := 0; i < wSize; i++ {
		xTv[i] = 0
	}

	for i := 0; i < l; i++ {
		SparseOperatorAxpy(v[i], x[i], xTv)
	}
}

func (fn *L2RLRFunc) hv(s []float64, hs []float64) {
	var i int
	l := fn.prob.L
	wSize := fn.getNrVariable()
	x := fn.prob.X

	for i = 0; i < wSize; i++ {
		hs[i] = 0
	}

	for i = 0; i < l; i++ {
		xi := x[i]
		xTs := SparseOperatorDot(s, xi)

		xTs = fn.c[i] * fn.d[i] * xTs

		SparseOperatorAxpy(xTs, xi, hs)
	}

	for i = 0; i < wSize; i++ {
		hs[i] = s[i] + hs[i]
	}
}

func (fn *L2RLRFunc) getNrVariable() int {
	return fn.prob.N
}
