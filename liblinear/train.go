package liblinear

// Train is the struct command to hold
type Train struct {
	bias            float64
	findC           bool
	CSpecified      bool
	solverSpecified bool
	crossValidation bool
	inputFilename   string
	modelFilename   string
	nrFold          int
	param           Parameter
	prob            Problem
}

func (t *Train) doFindParameterC() {
	var startC float64
	var maxC int = 1024

	if t.CSpecified {
		startC = t.param.c
	} else {
		startC = -1.0
	}

	logger.Println("Doing parameter search with %d-fold cross validation.\n", t.nrFold)
	var result ParameterSearchResult = 

}
