package liblinear

import (
	"fmt"
	"math"
)

/**
 * A coordinate descent algorithm for
 * multi-class support vector machines by Crammer and Singer
 *
 * <pre>
 * min_{\alpha} 0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
 * s.t. \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
 *
 * where e^m_i = 0 if y_i = m,
 * e^m_i = 1 if y_i != m,
 * C^m_i = C if m = y_i,
 * C^m_i = 0 if m != y_i,
 * and w_m(\alpha) = \sum_i \alpha^m_i x_i
 *
 * Given:
 * x, y, C
 * eps is the stopping tolerance
 *
 * solution will be put in w
 *
 * See Appendix of LIBLINEAR paper, Fan et al. (2008)
 * </pre>
 */

// SolverMCSVMCS provides the services for solving
type SolverMCSVMCS struct {
	B       []float64
	C       []float64
	eps     float64
	G       []float64
	maxIter int
	wSize   int
	l       int
	nrClass int
	prob    *Problem
}

// NewSolverMCSVMCS is the cOnstructor for struct SolverMCSVMCS
func NewSolverMCSVMCS(prob *Problem, nrClass int, weightedC []float64, eps float64, maxIter int) *SolverMCSVMCS {
	return &SolverMCSVMCS{
		wSize:   prob.N,
		l:       prob.L,
		nrClass: nrClass,
		eps:     eps,
		maxIter: maxIter,
		prob:    prob,
		C:       weightedC,
		B:       make([]float64, nrClass),
		G:       make([]float64, nrClass),
	}
}

// GETI is an internal function
func (solver *SolverMCSVMCS) GETI(i int) int {
	return int(solver.prob.Y[i])
}

func (solver *SolverMCSVMCS) beShrunk(i int, m int, yi int, alphaI float64, minG float64) bool {
	var bound float64

	if m == yi {
		bound = solver.C[solver.GETI(i)]
	}

	if alphaI == bound && solver.G[m] < minG {
		return true
	}

	return false
}

func (solver *SolverMCSVMCS) solveSubProblem(Ai float64, yi int, Cyi float64, activeI int, alphaNew []float64) {
	var r int

	if activeI <= len(solver.B) {
		panic("unable to solver subproblem")
	}

	D := make([]float64, activeI)
	copy(D, solver.B)

	if yi < activeI {
		D[yi] += Ai * Cyi
	}

	reversedMergeSortDefault(D)

	beta := D[0] - Ai*Cyi
	for r = 1; r < activeI && beta < float64(r)*D[r]; r++ {
		beta += D[r]
	}
	beta /= float64(r)

	for r = 0; r < activeI; r++ {
		if r == yi {
			alphaNew[r] = math.Min(Cyi, (beta-solver.B[r])/Ai)
		} else {
			alphaNew[r] = math.Min(0.0, (beta-solver.B[r])/Ai)
		}
	}

}

func (solver *SolverMCSVMCS) solve(w []float64) {
	var i, m, s int
	var iter int
	var maxIter = solver.maxIter
	var l, nrClass = solver.l, solver.nrClass
	var eps, wSize = solver.eps, solver.wSize
	alpha := make([]float64, l*nrClass)
	alphaNew := make([]float64, nrClass)
	index := make([]int, l)
	QD := make([]float64, l)
	dInd := make([]int, l)
	dVal := make([]float64, nrClass)
	alphaIndex := make([]int, nrClass*l)
	yIndex := make([]int, l)
	activeSize := l
	activeSizeI := make([]int, l)

	epsShrink := math.Max(10.0*eps, 1.0)
	startFromAll := true

	// Initial alpha can be set here. Note that
	// sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
	// alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
	// alpha[i*nr_class+m] <= 0 if prob->y[i] != m
	// If initial alpha isn't zero, uncomment the for loop below to initialize w

	for i = 0; i < l*nrClass; i++ {
		alpha[i] = 0
	}

	for i = 0; i < wSize*nrClass; i++ {
		w[i] = 0
	}

	for i = 0; i < l; i++ {
		for m = 0; m < nrClass; m++ {
			alphaIndex[i*nrClass+m] = m
		}
		QD[i] = 0
		for _, xi := range solver.prob.X[i] {
			val := xi.GetValue()
			QD[i] += val * val

			// Uncomment the for loop if initial alpha isn't zero
			// for m = 0; m<nrClass; m++ {
			//  w[(xi->index-1)*nrClass+m] += alpha[i*nrClass+m]*val
			//}
		}
		activeSizeI[i] = nrClass
		yIndex[i] = int(solver.prob.Y[i])
		index[i] = i
	}

	alphaI := NewDoubleArrayPointer(alpha, 0)
	alphaIndexI := NewIntArrayPointer(alphaIndex, 0)

	for iter < maxIter {
		stopping := math.Inf(-1)
		for i = 0; i < activeSize; i++ {
			j := i + random.Intn(activeSize-i)
			swapIntArray(index, i, j)
		}

		for s = 0; s < activeSize; s++ {
			i = index[s]
			aI := QD[i]
			alphaI.setOffset(i * nrClass)
			alphaIndexI.setOffset(i * nrClass)

			if aI > 0 {
				for m = 0; m < activeSize; m++ {
					solver.G[m] = 1
				}
				if yIndex[i] < activeSizeI[i] {
					solver.G[yIndex[i]] = 0
				}

				for _, xi := range solver.prob.X[i] {
					wOffset := (xi.GetIndex() - 1) * nrClass
					for m = 0; m < activeSizeI[i]; m++ {
						solver.G[m] += w[wOffset+alphaIndexI.get(m)] * xi.GetValue()
					}
				}

				minG := math.Inf(1)
				maxG := math.Inf(-1)

				for m = 0; m < activeSizeI[i]; m++ {
					if alphaI.get(alphaIndexI.get(m)) < 0 && solver.G[m] < minG {
						minG = solver.G[m]
					}
					if solver.G[m] > maxG {
						maxG = solver.G[m]
					}
				}

				if yIndex[i] < activeSizeI[i] {
					if alphaI.get(int(solver.prob.Y[i])) < solver.C[solver.GETI(i)] && solver.G[yIndex[i]] < minG {
						minG = solver.G[yIndex[i]]
					}
				}

				for m = 0; m < activeSizeI[i]; m++ {
					if solver.beShrunk(i, m, yIndex[i], alphaI.get(alphaIndexI.get(m)), minG) {
						activeSizeI[i]--
						for activeSizeI[i] > m {
							if !solver.beShrunk(i, activeSizeI[i], yIndex[i], alphaI.get(alphaIndexI.get(activeSizeI[i])), minG) {
								swapIntArrayPointer(alphaIndexI, m, activeSizeI[i])
								swapFloat64Array(solver.G, m, activeSizeI[i])
								if yIndex[i] == activeSizeI[i] {
									yIndex[i] = m
								} else if yIndex[i] == m {
									yIndex[i] = activeSizeI[i]
								}
								break
							}
							activeSizeI[i]--
						}
					}
				}

				if activeSizeI[i] <= 1 {
					activeSize--
					swapIntArray(index, s, activeSize)
					s--
					continue
				}

				if maxG-minG <= 1e-12 {
					continue
				} else {
					stopping = math.Max(maxG-minG, stopping)
				}

				for m = 0; m < activeSizeI[i]; m++ {
					solver.B[m] = solver.G[m] - aI*alphaI.get(alphaIndexI.get(m))
				}

				solver.solveSubProblem(aI, yIndex[i], solver.C[solver.GETI(i)], activeSizeI[i], alphaNew)

				nzD := 0
				for m = 0; m < activeSizeI[i]; m++ {
					d := alphaNew[m] - alphaI.get(alphaIndexI.get(m))
					alphaI.set(alphaIndexI.get(m), alphaNew[m])
					if math.Abs(d) >= 1e-12 {
						dInd[nzD] = alphaIndexI.get(m)
						dVal[nzD] = d
						nzD++
					}
				}

				for _, xi := range solver.prob.X[i] {
					// wI *float64 = &w[(xi.GetIndex()-1)*nrClass];
					wOffset := (xi.GetIndex() - 1) * nrClass
					for m = 0; m < nzD; m++ {
						w[wOffset+dInd[m]] += dVal[m] * xi.GetValue()
					}
				}
			}
		}

		iter++

		if iter%10 == 0 {
			fmt.Print(".")
		}

		if stopping < epsShrink {
			if stopping < eps && startFromAll {
				break
			} else {
				activeSize = l
				for i = 0; i < l; i++ {
					activeSizeI[i] = nrClass
				}
				fmt.Print("*")
				epsShrink = math.Max(epsShrink/2, eps)
				startFromAll = true
			}
		} else {
			startFromAll = false
		}
	}

	fmt.Printf("\noptimization finished, #iter = %d\n", iter)
	if iter >= maxIter {
		fmt.Printf("\nWARNING: reaching max number of iterations\n")
	}

	// calculate objective value

	var v float64
	nSV := 0
	for i = 0; i < wSize*nrClass; i++ {
		v += w[i] * w[i]
	}
	v = 0.5 * v
	for i = 0; i < l*nrClass; i++ {
		v += alpha[i]
		if math.Abs(alpha[i]) > 0 {
			nSV++
		}
	}
	for i = 0; i < l; i++ {
		v -= alpha[i*nrClass+int(solver.prob.Y[i])]
	}
	fmt.Printf("Objective value = %f\n", v)
	fmt.Printf("nSV = %d\n", nSV)
}
