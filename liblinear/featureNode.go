package liblinear

// FeatureNode implements a Feature
type FeatureNode struct {
	index int
	value float64
}

// NewFeatureNode returns a new FeatureNode
func NewFeatureNode(index int, value float64) *FeatureNode {
	return &FeatureNode{
		index: index,
		value: value,
	}
}

// GetIndex does just that
func (f *FeatureNode) GetIndex() int {
	return f.index
}

// GetValue does just that
func (f *FeatureNode) GetValue() float64 {
	return f.value
}

// SetValue does just that
func (f *FeatureNode) SetValue(val float64) {
	f.value = val
}
