package liblinear

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTrainPredict(t *testing.T) {

	var x = make([][]Feature, 4)
	x[0] = make([]Feature, 2)
	x[1] = make([]Feature, 1)
	x[2] = make([]Feature, 1)
	x[3] = make([]Feature, 3)

	x[0][0] = NewFeatureNode(1, 1)
	x[0][1] = NewFeatureNode(2, 1)

	x[1][0] = NewFeatureNode(3, 1)
	x[2][0] = NewFeatureNode(3, 1)

	x[3][0] = NewFeatureNode(1, 2)
	x[3][1] = NewFeatureNode(2, 1)
	x[3][2] = NewFeatureNode(4, 1)

	var y = []float64{0, 1, 1, 0}

	var prob = &Problem{
		Bias: -1,
		L:    4,
		N:    4,
		X:    x,
		Y:    y,
	}

	for _, solver := range solverTypeValues {
		for C := 0.1; C <= 100; C *= 1.2 {
			if C < 0.2 {
				if solver == L1R_L2LOSS_SVC {
					continue
				}
			}
			if C < 0.7 {
				if solver == L1R_LR {
					continue
				}
			}

			if solver.IsSupportVectorRegression() {
				continue
			}

			param := NewParameter(solver, C, 0.1, 1000, 0.1)
			model, _ := Train(prob, param)

			featureWeights := model.GetFeatureWeights()
			if solver == MCSVM_CS {
				assert.Equal(t, 8, len(featureWeights))
			} else {
				assert.Equal(t, 4, len(featureWeights))
			}

			var i = 0
			for _, value := range prob.Y {
				prediction := Predict(model, prob.X[i])
				assert.Equal(t, value, prediction)

				if model.isProbabilityModel() {
					estimates := make([]float64, model.NumClass)
					probabilityPrediction := PredictProbability(model, prob.X[i], estimates)
					assert.Equal(t, prediction, probabilityPrediction)

					if estimates[int(probabilityPrediction)] < (1.0 / float64(model.NumClass)) {
						t.Fail()
					}

					var estimationSum float64
					for _, estimate := range estimates {
						estimationSum += estimate
					}

					if estimationSum <= 0.9 || estimationSum >= 1.1 {
						t.Fail()
					}
				}
				i++
			}
		}
	}
}

// 			 int i = 0;
// 			 for (double value : prob.y) {
// 				 double prediction = Linear.predict(model, prob.x[i]);
// 				 assertThat(prediction).as("prediction with solver " + solver).isEqualTo(value);
// 				 if (model.isProbabilityModel()) {
// 					 double[] estimates = new double[model.getNrClass()];
// 					 double probabilityPrediction = Linear.predictProbability(model, prob.x[i], estimates);
// 					 assertThat(probabilityPrediction).isEqualTo(prediction);
// 					 assertThat(estimates[(int)probabilityPrediction]).isGreaterThanOrEqualTo(1.0 / model.getNrClass());
// 					 double estimationSum = 0;
// 					 for (double estimate : estimates) {
// 						 estimationSum += estimate;
// 					 }
// 					 assertThat(estimationSum).isEqualTo(1.0, offset(0.001));
// 				 }
// 				 i++;
// 			 }
// 		 }
// 	 }
//  }
