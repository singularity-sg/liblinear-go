package liblinear

// L2RL2SvrFunc type struct
type L2RL2SvrFunc struct {
	base *L2RL2SvcFunc
	p    float64
}

// NewL2RL2SvrFunc creates a new instance of L2RL2SvrFunc
func NewL2RL2SvrFunc(prob *Problem, c []float64, p float64) *L2RL2SvrFunc {
	return &L2RL2SvrFunc{
		base: NewL2RL2SvcFunc(prob, c),
		p:    p,
	}
}

func (fn *L2RL2SvrFunc) getNrVariable() int {
	return fn.base.getNrVariable()
}

func (fn *L2RL2SvrFunc) fun(w []float64) float64 {
	f := 0.0
	y := fn.base.prob.Y
	l := fn.base.prob.L
	wSize := fn.getNrVariable()
	var d float64

	fn.base.xv(w, fn.base.z)

	for i := 0; i < wSize; i++ {
		f += w[i] * w[i]
	}
	f /= 2
	for i := 0; i < l; i++ {
		d = fn.base.z[i] - y[i]
		if d < -fn.p {
			f += fn.base.c[i] * (d + fn.p) * (d + fn.p)
		} else if d > fn.p {
			f += fn.base.c[i] * (d - fn.p) * (d - fn.p)
		}
	}

	return f
}

func (fn *L2RL2SvrFunc) grad(w []float64, g []float64) {

	y := fn.base.prob.Y
	l := fn.base.prob.L
	wSize := fn.getNrVariable()

	sizeI := 0
	for i := 0; i < l; i++ {
		d := fn.base.z[i] - y[i]

		// generate index set I
		if d < -fn.p {
			fn.base.z[sizeI] = fn.base.c[i] * (d + fn.p)
			fn.base.i[sizeI] = i
			sizeI++
		} else if d > fn.p {
			fn.base.z[sizeI] = fn.base.c[i] * (d - fn.p)
			fn.base.i[sizeI] = i
			sizeI++
		}
	}

	fn.base.subXTv(fn.base.z, g)

	for i := 0; i < wSize; i++ {
		g[i] = w[i] + 2*g[i]
	}
}

func (fn *L2RL2SvrFunc) hv(s []float64, hs []float64) {
	fn.base.hv(s, hs)
}
