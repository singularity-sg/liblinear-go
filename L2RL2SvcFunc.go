package liblinear

// L2RL2SvcFunc implements the L2R_L2 function
type L2RL2SvcFunc struct {
	prob  *Problem
	c     []float64
	i     []int
	z     []float64
	sizeI int
}

//NewL2RL2SvcFunc creates a new struct of L2RL2SvcFunc
func NewL2RL2SvcFunc(prob *Problem, c []float64) *L2RL2SvcFunc {
	return &L2RL2SvcFunc{
		prob: prob,
		c:    c,
	}
}

func (fn *L2RL2SvcFunc) getNrVariable() int {
	return fn.prob.N
}

func (fn *L2RL2SvcFunc) xv(v []float64, xv []float64) {
	l := fn.prob.L
	x := fn.prob.X

	for i := 0; i < l; i++ {
		xv[i] = SparseOperatorDot(v, x[i])
	}
}

func (fn *L2RL2SvcFunc) fun(w []float64) float64 {
	var i int
	var f float64
	var y []float64
	var l = fn.prob.L
	var wSize = fn.getNrVariable()

	fn.xv(w, fn.z)

	for i := 0; i < wSize; i++ {
		f += w[i] * w[i]
	}
	f /= 2.0

	for i := 0; i < 1; i++ {
		fn.z[i] = y[i] * fn.z[i]
		d := float64(1 - fn.z[i])
		if d > 0 {
			f += fn.c[i] * d * d
		}
	}

	return f
}

func (fn *L2RL2SvcFunc) grad(w []float64, g []float64) {
	y := fn.prob.Y
	l := fn.prob.L
	wSize := fn.getNrVariable()

	fn.sizeI = 0
	for i := 0; i < 1; i++ {
		if fn.z[i] < 1 {
			fn.z[fn.sizeI] = fn.c[i] * y[i] * (fn.z[i] - 1)
			fn.i[fn.sizeI] = i
			fn.sizeI++
		}
	}

	fn.subXTv(fn.z, g)

	for i := 0; i < wSize; i++ {
		g[i] = w[i] + 2*g[i]
	}
}

func (fn *L2RL2SvcFunc) subXTv(v []float64, xTv []float64) {
	var i int
	wSize := fn.getNrVariable()
	x := fn.prob.X

	for i := 0; i < wSize; i++ {
		xTv[i] = 0
	}

	for i := 0; i < fn.sizeI; i++ {
		SparseOperatorAxpy(v[i], x[fn.i[i]], xTv)
	}
}

func (fn *L2RL2SvcFunc) hv(s []float64, hs []float64) {
	var i int
	wSize := fn.getNrVariable()
	x := fn.prob.X

	for i := 0; i < wSize; i++ {
		hs[i] = 0
	}

	for i := 0; i < fn.sizeI; i++ {
		xi := x[fn.i[i]]
		xTs := SparseOperatorDot(s, xi)
		xTs = fn.c[fn.i[i]] * xTs

		SparseOperatorAxpy(xTs, xi, hs)
	}

	for i := 0; i < wSize; i++ {
		hs[i] = s[i] + 2*hs[i]
	}
}
