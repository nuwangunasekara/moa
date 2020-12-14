package moa.classifiers.neuralNetworks;

import com.github.javacliparser.FlagOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;



public class MultiMLP extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;
    protected MLP[] nn = null;
    protected int featureValuesArraySize = 0;

    public FlagOption useOneHotEncode = new FlagOption("useOneHotEncode", 'h',
            "use one hot encoding");

    @Override
    public void resetLearningImpl() {

    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        if(this.nn == null){
            initNNs(instance);
        }

        float[] featureValues = new float [featureValuesArraySize];
        MLP.setFeatureValuesArray(instance, featureValues, useOneHotEncode.isSet());
        double class_value[] = {instance.classValue()};

        for (int i = 0 ; i < this.nn.length ; i++) {
            this.nn[i].initializeNetwork(instance);
            this.nn[i].trainOnInstance(featureValues, class_value);
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
        class MLPConfigs{
            int optimizerType;
            float learningRate = 0.0f;
            MLPConfigs(int optimizerType, float learningRate){
                this.optimizerType = optimizerType;
                this.learningRate = learningRate;
            }
        }
        MLPConfigs nnConfigs[] = new MLPConfigs[7];
        nnConfigs[0] = new MLPConfigs(MLP.OPTIMIZER_SGD, 0.03f);
        nnConfigs[1] = new MLPConfigs(MLP.OPTIMIZER_SGD, 0.05f);
        nnConfigs[2] = new MLPConfigs(MLP.OPTIMIZER_SGD, 0.07f);
        nnConfigs[3] = new MLPConfigs(MLP.OPTIMIZER_ADAM, 0.01f);
        nnConfigs[4] = new MLPConfigs(MLP.OPTIMIZER_ADAM, 0.03f);
        nnConfigs[5] = new MLPConfigs(MLP.OPTIMIZER_ADAM, 0.07f);
        nnConfigs[6] = new MLPConfigs(MLP.OPTIMIZER_ADAM, 0.09f);
//            nnConfigs[0] = new MLPConfigs(MLP.OPTIMIZER_RMSPROP, 0.01f);
//            nnConfigs[0] = new MLPConfigs(MLP.OPTIMIZER_ADAGRAD, 0.03f);
//            nnConfigs[0] = new MLPConfigs(MLP.OPTIMIZER_ADAGRAD, 0.07f);
//            nnConfigs[0] = new MLPConfigs(MLP.OPTIMIZER_ADAGRAD, 0.09f);

        this.nn = new MLP[nnConfigs.length];
        for(int i=0; i < nnConfigs.length; i++){
            this.nn[i] = new MLP();
            this.nn[i].optimizerTypeOption.setChosenIndex(nnConfigs[i].optimizerType);
            this.nn[i].learningRateOption.setValue(nnConfigs[i].learningRate);
            this.nn[i].useOneHotEncode.setValue(useOneHotEncode.isSet());
        }

        featureValuesArraySize = MLP.getFeatureValuesArraySize(instance, useOneHotEncode.isSet());
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
//            if (this.getClass() == moa.classifiers.meta.StreamingRandomPatches.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
//            else
//                return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
