package liblinear

// IntArrayPointer is a data structure to represent an array with offset
type IntArrayPointer struct {
	array  []int
	offset int
}

func (dap *IntArrayPointer) setOffset(offset int) {
	if dap.offset < 0 || dap.offset >= len(dap.array) {
		panic("offset must be between 0 and the length of the array")
	}
	dap.offset = offset
}

func (dap *IntArrayPointer) get(index int) int {
	return dap.array[dap.offset+index]
}

func (dap *IntArrayPointer) set(index int, value int) {
	dap.array[dap.offset+index] = value
}

// NewIntArrayPointer is a constructor for the IntArrayPointer
func NewIntArrayPointer(array []int, offset int) *IntArrayPointer {
	return &IntArrayPointer{
		array:  array,
		offset: offset,
	}
}
