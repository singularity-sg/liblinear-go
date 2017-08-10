package liblinear

// DoubleArrayPointer is a data structure to represent an array with offset
type DoubleArrayPointer struct {
	array  []float64
	offset int
}

func (dap *DoubleArrayPointer) setOffset(offset int) {
	if dap.offset < 0 || dap.offset >= len(dap.array) {
		panic("offset must be between 0 and the length of the array")
	}
	dap.offset = offset
}

func (dap *DoubleArrayPointer) get(index int) float64 {
	return dap.array[dap.offset+index]
}

func (dap *DoubleArrayPointer) set(index int, value float64) {
	dap.array[dap.offset+index] = value
}

// NewDoubleArrayPointer is a constructor for the DoubleArrayPointer
func NewDoubleArrayPointer(array []float64, offset int) *DoubleArrayPointer {
	return &DoubleArrayPointer{
		array:  array,
		offset: offset,
	}
}
