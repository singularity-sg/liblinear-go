package liblinear

// Feature contains an index and value
type Feature interface {
	GetIndex() int
	GetValue() float64
	SetValue(val float64)
}
