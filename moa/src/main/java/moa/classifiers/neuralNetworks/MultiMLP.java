package moa.classifiers.neuralNetworks;

import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;



public class MultiMLP extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {

        private static final long serialVersionUID = 1L;
        private static final int NUMBER_OF_MLPS = 10;


        protected MLP[] nn = null;

        @Override
        public void resetLearningImpl() {

        }

        @Override
        public void trainOnInstanceImpl(Instance instance) {
            if(this.nn == null)
                initNNs(instance);

            for (int i = 0 ; i < this.nn.length ; i++) {
                this.nn[i].trainOnInstance(instance);
            }
        }

        @Override
        public double[] getVotesForInstance(Instance instance) {
            int min_index = 0;
            if(this.nn == null) {
                initNNs(instance);
            }else {
                double min_estimation = Double.MIN_VALUE;
                for (int i = 0 ; i < this.nn.length ; i++) {
                    if (this.nn[i].estimator.getEstimation() < min_estimation){
                        min_index = i;
                    }
                }
            }
            return this.nn[min_index].getVotesForInstance(instance);
        }

        @Override
        public boolean isRandomizable() {
            return false;
        }

        @Override
        public void getModelDescription(StringBuilder arg0, int arg1) {
        }

        @Override
        protected Measurement[] getModelMeasurementsImpl() {
            return null;
        }

        protected void initNNs(Instance instance) {
            //	net_config = [
//	{'optimizer_type': OP_TYPE_SGD_NC, 'l_rate': 0.03},
//	{'optimizer_type': OP_TYPE_SGD_NC, 'l_rate': 0.05},
//	{'optimizer_type': OP_TYPE_SGD_NC, 'l_rate': 0.07},
//	{'optimizer_type': OP_TYPE_RMSPROP_NC, 'l_rate': 0.01},
//	{'optimizer_type': OP_TYPE_ADAGRAD, 'l_rate': 0.03},
//	{'optimizer_type': OP_TYPE_ADAGRAD_NC, 'l_rate': 0.03},
//	{'optimizer_type': OP_TYPE_ADAGRAD, 'l_rate': 0.07},
//	{'optimizer_type': OP_TYPE_ADAGRAD, 'l_rate': 0.09},
//	{'optimizer_type': OP_TYPE_ADAGRAD_NC, 'l_rate': 0.09},
//	{'optimizer_type': OP_TYPE_ADAM, 'l_rate': 0.01},
//	{'optimizer_type': OP_TYPE_ADAM_NC, 'l_rate': 0.01},
//			]
            this.nn = new MLP[4];
//            this.nn[0] = new MLP(MLP.OPTIMIZER_SGD,0.03f);
            this.nn[0] = new MLP();
            this.nn[0].optimizerTypeOption.setChosenIndex(MLP.OPTIMIZER_SGD);
            this.nn[0].learningRateOption.setValue(0.03f);
//            this.nn[1] = new MLP(MLP.OPTIMIZER_SGD,0.05f);
            this.nn[1] = new MLP();
            this.nn[1].optimizerTypeOption.setChosenIndex(MLP.OPTIMIZER_SGD);
            this.nn[1].learningRateOption.setValue(0.05f);
//            this.nn[2] = new MLP(MLP.OPTIMIZER_SGD,0.07f);
            this.nn[2] = new MLP();
            this.nn[2].optimizerTypeOption.setChosenIndex(MLP.OPTIMIZER_SGD);
            this.nn[2].learningRateOption.setValue(0.07f);
////            this.nn[3] = new MLP(MLP.OPTIMIZER_RMSPROP,0.01f);
//            this.nn[3] = new MLP();
//            this.nn[3].optimizerTypeOption.setChosenIndex(MLP.OPTIMIZER_RMSPROP);
//            this.nn[3].learningRateOption.setValue(0.01f);
////            this.nn[4] = new MLP(MLP.OPTIMIZER_ADAGRAD,0.03f);
//            this.nn[4] = new MLP();
//            this.nn[4].optimizerTypeOption.setChosenIndex(MLP.OPTIMIZER_ADAGRAD);
//            this.nn[4].learningRateOption.setValue(0.03f);
////            this.nn[5] = new MLP(MLP.OPTIMIZER_ADAGRAD,0.07f);
//            this.nn[5] = new MLP();
//            this.nn[5].optimizerTypeOption.setChosenIndex(MLP.OPTIMIZER_ADAGRAD);
//            this.nn[5].learningRateOption.setValue(0.07f);
////            this.nn[6] = new MLP(MLP.OPTIMIZER_ADAGRAD,0.09f);
//            this.nn[6] = new MLP();
//            this.nn[6].optimizerTypeOption.setChosenIndex(MLP.OPTIMIZER_ADAGRAD);
//            this.nn[6].learningRateOption.setValue(0.09f);
////            this.nn[7] = new MLP(MLP.OPTIMIZER_ADAM,0.01f);
            this.nn[3] = new MLP();
            this.nn[3].optimizerTypeOption.setChosenIndex(MLP.OPTIMIZER_ADAM);
            this.nn[3].learningRateOption.setValue(0.01f);
        }

        @Override
        public ImmutableCapabilities defineImmutableCapabilities() {
//            if (this.getClass() == moa.classifiers.meta.StreamingRandomPatches.class)
                return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
//            else
//                return new ImmutableCapabilities(Capability.VIEW_STANDARD);
        }
}
