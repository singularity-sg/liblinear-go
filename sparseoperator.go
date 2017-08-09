package liblinear

// SparseOperatorNrm2Sq is the equivalent of nrm2sq
func SparseOperatorNrm2Sq(x []Feature) float64 {
	var ret float64

	for _, feature := range x {
		ret += feature.GetValue() * feature.GetValue()
	}

	return ret
}

// SparseOperatorDot is the equivalent of dot
func SparseOperatorDot(s []float64, x []Feature) float64 {
	var ret float64

	for _, feature := range x {
		ret += s[feature.GetIndex()-1] * feature.GetValue()
	}

	return ret
}

// SparseOperatorAxpy is the equivalent of axpy
func SparseOperatorAxpy(a float64, x []Feature, y []float64) {
	for _, feature := range x {
		y[feature.GetIndex()-1] += a * feature.GetValue()
	}
}
