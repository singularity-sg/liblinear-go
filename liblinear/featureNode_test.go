package liblinear

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConstructorIndexZero(t *testing.T) {
	assert.NotNil(t, NewFeatureNode(0, 0))
}

func TestConstructorIndexNegative(t *testing.T) {
	assert.NotNil(t, NewFeatureNode(-1, 0))
}

func TestConstructorHappy(t *testing.T) {
	fn := NewFeatureNode(25, 27.39)
	assert.Equal(t, 25, fn.GetIndex())
	assert.Equal(t, 27.39, fn.GetValue())

	fn = NewFeatureNode(1, -0.22222)
	assert.Equal(t, 1, fn.GetIndex())
	assert.Equal(t, -0.22222, fn.GetValue())
}
