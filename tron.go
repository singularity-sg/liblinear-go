package liblinear

import (
	"math"

	"github.com/tevino/abool"
)

// Tron is Trust Region Newton Method optimization
type Tron struct {
	funObj  Function
	eps     float64
	maxIter int
	epsCg   float64
}

//NewTron returns a Tron struct
func NewTron(funObj Function, eps float64, maxIter int, epsCg float64) *Tron {
	return &Tron{
		funObj:  funObj,
		eps:     eps,
		maxIter: maxIter,
		epsCg:   epsCg,
	}
}

func (tr *Tron) tron(w []float64) {
	// Parameters for updating the iterates.
	eta0 := 1e-4
	eta1 := 0.25
	eta2 := 0.75

	// Parameters for updating the trust region size delta.
	sigma1 := 0.25
	sigma2 := 0.5
	sigma3 := 4.0

	var n = tr.funObj.getNrVariable()
	var i, cgIter int
	var delta, snorm, one float64 = 0, 0, 1.0
	var alpha, f, fnew, prered, actred, gs float64
	var search, iter int = 1, 1

	s, r, g := make([]float64, n), make([]float64, n), make([]float64, n)

	w0 := make([]float64, n)
	for i = 0; i < n; i++ {
		w0[i] = 0
	}
	tr.funObj.fun(w0)
	tr.funObj.grad(w0, g)
	gnorm0 := euclideanNorm(g)

	f = tr.funObj.fun(w)
	tr.funObj.grad(w, g)
	delta = euclideanNorm(g)
	gnorm := delta

	if gnorm <= tr.eps*gnorm0 {
		search = 0
	}

	iter = 1

	wNew := make([]float64, n)
	var reachBoundary = abool.New()

	for iter <= tr.maxIter && search != 0 {
		cgIter = tr.trcg(delta, g, s, r, reachBoundary)

		copy(wNew, w)
		daxpy(one, s, wNew)

		gs = dot(g, s)
		prered = -0.5 * (gs - dot(s, r))
		fnew = tr.funObj.fun(wNew)

		// Compute the actual reduction.
		actred = f - fnew

		// On the first iteration, adjust the initial step bound.
		snorm = euclideanNorm(s)

		if iter == 1 {
			delta = math.Min(delta, snorm)
		}

		// Compute prediction alpha*snorm of the step.
		if fnew-f-gs <= 0 {
			alpha = sigma3
		} else {
			alpha = math.Max(sigma1, -0.5*(gs/(fnew-f-gs)))
		}

		// Update the trust region bound according to the ratio of actual to
		// predicted reduction.
		if actred < eta0*prered {
			delta = math.Min(math.Max(alpha, sigma1)*snorm, sigma2*delta)
		} else if actred < eta1*prered {
			delta = math.Max(sigma1*delta, math.Min(alpha*snorm, sigma2*delta))
		} else if actred < eta2*prered {
			delta = math.Max(sigma1*delta, math.Min(alpha*snorm, sigma3*delta))
		} else {
			if reachBoundary.IsSet() {
				delta = sigma3 * delta
			} else {
				delta = math.Max(delta, math.Min(alpha*snorm, sigma3*delta))
			}
		}

		logger.Printf("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cgIter)

		if actred > eta0*prered {
			iter++
			copy(w, wNew)
			f = fnew
			tr.funObj.grad(w, g)
			gnorm = euclideanNorm(g)
			if gnorm <= tr.eps*gnorm0 {
				break
			}
		}

		if f < -1.0e+32 {
			logger.Println("WARNING: f < -1.0e+32")
			break
		}

		if prered <= 0 {
			logger.Println("WARNING: prered <= 0")
			break
		}

		if math.Abs(actred) <= 1.0e-12*math.Abs(f) && math.Abs(prered) <= 1.0e-12*math.Abs(f) {
			logger.Println("WARNING: actred and prered too small")
			break
		}
	}

}

func (tr *Tron) trcg(delta float64, g []float64, s []float64, r []float64, reachBoundary *abool.AtomicBool) int {

	n := tr.funObj.getNrVariable()
	var one float64 = 1

	var d = make([]float64, n)
	var hd = make([]float64, n)
	var rTr, rNewTrNew, cgTol float64

	for i := 0; i < n; i++ {
		s[i] = 0
		r[i] = -g[i]
		d[i] = r[i]
	}

	cgTol = tr.epsCg * euclideanNorm(g)

	cgIter := 0
	rTr = dot(r, r)

	for {
		if euclideanNorm(r) <= cgTol {
			break
		}

		cgIter++

		tr.funObj.hv(d, hd)

		alpha := rTr / dot(d, hd)
		daxpy(alpha, d, s)

		if euclideanNorm(s) > delta {
			logger.Println("cg reaches trust region boundary")
			reachBoundary.Set()
			alpha = -alpha
			daxpy(alpha, d, s)

			std := dot(s, d)
			sts := dot(s, s)
			dtd := dot(d, d)
			dsq := delta * delta
			rad := math.Sqrt(std*std + dtd*(dsq-sts))

			if std >= 0 {
				alpha = (dsq - sts) / (std + rad)
			} else {
				alpha = (rad - std) / dtd
			}

			daxpy(alpha, d, s)
			alpha = -alpha
			daxpy(alpha, hd, r)
			break
		}

		alpha = -alpha
		daxpy(alpha, hd, r)
		rNewTrNew = dot(r, r)
		beta := rNewTrNew / rTr
		scale(beta, d)
		daxpy(one, r, d)
		rTr = rNewTrNew
	}

	return cgIter
}

// constant times a vector plus a vector
func daxpy(constant float64, vector1 []float64, vector2 []float64) {
	if constant == 0 {
		return
	}

	if len(vector1) != len(vector2) {
		panic("The length of the vectors are different!")
	}

	for i := 0; i < len(vector1); i++ {
		vector2[i] += constant * vector1[i]
	}
}

// returns the dot product of two vectors
func dot(vector1 []float64, vector2 []float64) float64 {

	var product float64
	if len(vector1) != len(vector2) {
		panic("Cannot calculate as the vectors are of different lengths!")
	}
	for i := 0; i < len(vector1); i++ {
		product += vector1[i] * vector2[i]
	}
	return product

}

// returns the euclidean norm of a vector
func euclideanNorm(vector []float64) float64 {

	n := len(vector)

	if n < 1 {
		return 0
	}

	if n == 1 {
		return math.Abs(vector[0])
	}

	// this algorithm is (often) more accurate than just summing up the squares and taking the square-root afterwards
	var scale float64 // scaling factor that is factored out
	var sum = 1.0     // basic sum of squares from which scale has been factored out
	for i := 0; i < n; i++ {
		if vector[i] != 0 {
			abs := math.Abs(vector[i])
			// try to get the best scaling factor
			if scale < abs {
				t := scale / abs
				sum = 1 + sum*(t*t)
				scale = abs
			} else {
				t := abs / scale
				sum += t * t
			}
		}
	}

	return scale * math.Sqrt(sum)
}

// scales a vector by a constant
func scale(constant float64, vector []float64) {
	if constant == 1.0 {
		return
	}

	for i := 0; i < len(vector); i++ {
		vector[i] *= constant
	}
}
