package liblinear

/**
 * <p>Describes the problem</p>
 *
 * For example, if we have the following training data:
 * <pre>
 *  LABEL       ATTR1   ATTR2   ATTR3   ATTR4   ATTR5
 *  -----       -----   -----   -----   -----   -----
 *  1           0       0.1     0.2     0       0
 *  2           0       0.1     0.3    -1.2     0
 *  1           0.4     0       0       0       0
 *  2           0       0.1     0       1.4     0.5
 *  3          -0.1    -0.2     0.1     1.1     0.1
 *
 *  and bias = 1, then the components of problem are:
 *
 *  l = 5
 *  n = 6
 *
 *  y -&gt; 1 2 1 2 3
 *
 *  x -&gt; [ ] -&gt; (2,0.1) (3,0.2) (6,1) (-1,?)
 *       [ ] -&gt; (2,0.1) (3,0.3) (4,-1.2) (6,1) (-1,?)
 *       [ ] -&gt; (1,0.4) (6,1) (-1,?)
 *       [ ] -&gt; (2,0.1) (4,1.4) (5,0.5) (6,1) (-1,?)
 *       [ ] -&gt; (1,-0.1) (2,-0.2) (3,0.1) (4,1.1) (5,0.1) (6,1) (-1,?)
 * </pre>
 */
type Problem struct {
	L    int         //the number of training data
	N    int         //the number of features (including the bias feature if bias >= 0)
	Y    []float64   //an array containing the target values
	X    [][]Feature //array of sparse features
	Bias float64     //If bias &gt;= 0, we assume that one additional feature is added to the end of each data instance
}

// NewProblem gives you a Problem exactly
func NewProblem(L int, N int, Y []float64, X [][]Feature, Bias float64) *Problem {
	return &Problem{
		L:    L,
		N:    N,
		Y:    Y,
		X:    X,
		Bias: Bias,
	}
}
