package liblinear

// ParameterSearchResult stores the result of the parameter search
type ParameterSearchResult struct {
	bestC    float64
	bestRate float64
}

// NewParameterSearchResult returns a new instance of this struct
func NewParameterSearchResult(bestC float64, bestRate float64) *ParameterSearchResult {
	return &ParameterSearchResult{
		bestC:    bestC,
		bestRate: bestRate,
	}
}
