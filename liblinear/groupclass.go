package liblinear

// GroupClassesReturn result of a group of classes
type GroupClassesReturn struct {
	count   []int
	label   []int
	nrClass int
	start   []int
}

// NewGroupClassesReturn is a constructor for GroupClassesReturn
func newGroupClassesReturn(nrClass int, label []int, start []int, count []int) *GroupClassesReturn {
	return &GroupClassesReturn{
		nrClass: nrClass,
		label:   label,
		start:   start,
		count:   count,
	}
}

func groupClasses(prob *Problem, perm []int) *GroupClassesReturn {
	l := prob.L
	maxNrClass := 16
	nrClass := 0

	label := make([]int, maxNrClass)
	count := make([]int, maxNrClass)
	dataLabel := make([]int, l)

	var i int

	for i = 0; i < l; i++ {
		var thisLabel = int(prob.Y[i])
		var j int
		for j = 0; j < nrClass; j++ {
			if thisLabel == label[j] {
				count[j]++
				break
			}
		}
		dataLabel[i] = j
		if j == nrClass {
			if nrClass == maxNrClass {
				maxNrClass *= 2
				dstLabel := make([]int, maxNrClass)
				copy(dstLabel, label)
				label = dstLabel
				dstCount := make([]int, maxNrClass)
				copy(dstCount, count)
				count = dstCount
			}
			label[nrClass] = thisLabel
			count[nrClass] = 1
			nrClass++
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if nrClass == 2 && label[0] == -1 && label[1] == 1 {
		swapIntArray(label, 0, 1)
		swapIntArray(count, 0, 1)
		for i = 0; i < l; i++ {
			if dataLabel[i] == 0 {
				dataLabel[i] = 1
			} else {
				dataLabel[i] = 0
			}
		}
	}

	start := make([]int, nrClass)
	start[0] = 0
	for i = 1; i < nrClass; i++ {
		start[i] = start[i-1] + count[i-1]
	}
	for i = 0; i < l; i++ {
		perm[start[dataLabel[i]]] = i
		start[dataLabel[i]]++
	}

	start[0] = 0
	for i = 1; i < nrClass; i++ {
		start[i] = start[i-1] + count[i-1]
	}

	return newGroupClassesReturn(nrClass, label, start, count)
}
