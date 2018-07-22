package main

import (
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"limhan.info/liblinear-go/liblinear"
)

func TestDoPredictCorruptLine(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			assert.Equal(t, "Wrong input format at line 1", r)
		}
	}()

	model := liblinear.CreateRandomModel()
	assert.True(t, model.NumClass >= 2)
	assert.True(t, model.NumFeatures >= 10)

	testWithLines(fmt.Sprintf("%d abc\n", model.Label[0]), os.Stdout, model)
}

func TestDoPredictCorruptLine2(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			assert.Equal(t, "The index  cannot be parsed", r)
		}
	}()
	model := liblinear.CreateRandomModel()
	assert.True(t, model.NumClass >= 2)
	assert.True(t, model.NumFeatures >= 10)

	testWithLines(fmt.Sprintf("%d :1\n", model.Label[0]), os.Stdout, model)
}

func TestDoPredict(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			assert.FailNow(t, "There should not be any panicking but we saw this error", r)
		}
	}()
	model := liblinear.CreateRandomModel()
	assert.True(t, model.NumClass >= 2)
	assert.True(t, model.NumFeatures >= 10)

	testWithLines(fmt.Sprintf("%d 1:0.32393\n%d 2:-71.555 9:88223\n", model.Label[0], model.Label[1]), os.Stdout, model)
}

func testWithLines(str string, writer io.Writer, model *liblinear.Model) {
	DoPredict(strings.NewReader(str), writer, model)
}
