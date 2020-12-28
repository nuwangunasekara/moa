package moa.classifiers.neuralNetworks;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;

import java.util.concurrent.*;


public class MultiMLP extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {

    private static final long serialVersionUID = 1L;
    protected MLP[] nn = null;
    protected int featureValuesArraySize = 0;
    protected long samplesSeen = 0;
    protected MLP.NormalizeInfo[] normalizeInfo = null;
    private float[] featureValues = null;
    private double [] class_value = null;
    private ExecutorService exService = null;

    public FlagOption useOneHotEncode = new FlagOption("useOneHotEncode", 'h',
            "use one hot encoding");

    public FlagOption useNormalization = new FlagOption("useNormalization", 'n',
            "Normalize data");

    public MultiChoiceOption trainingMethodOption = new MultiChoiceOption("trainingMethod", 't',
            "The training method to use: Sequential, Use threads",
            new String[]{"Sequential", "UseThreads"},
            new String[]{"Sequential", "UseThreads"}, 1);

    public static final int TRAIN_SEQUENTIAL = 0;
    public static final int TRAIN_USE_THREADS = 1;

    @Override
    public void resetLearningImpl() {
        if (nn != null) {
            exService.shutdownNow();
            exService = null;
            for (int i = 0; i < this.nn.length; i++) {
                nn[i] = null;
            }
            nn = null;
            featureValuesArraySize = 0;
            samplesSeen = 0;
            normalizeInfo = null;
            featureValues = null;
            class_value = null;
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        if(this.nn == null){
            initNNs(instance);
        }

        MLP.setFeatureValuesArray(instance, featureValues, useOneHotEncode.isSet(),false, normalizeInfo, samplesSeen);
        class_value[0] = instance.classValue();

        switch (this.trainingMethodOption.getChosenIndex()) {
            case TRAIN_SEQUENTIAL:
                for (MLP mlp : this.nn) {
                    mlp.initializeNetwork(instance);
                    mlp.trainOnFeatureValues(featureValues, class_value);
                }
                break;
            case TRAIN_USE_THREADS:
                class TrainThread implements Callable<Boolean> {
                    private final MLP nn;
                    private final float[] featureValues;
                    private final double [] class_value;

                    public TrainThread(MLP nn, float[] featureValues, double [] class_value){
                        this.nn = nn;
                        this.featureValues = featureValues;
                        this.class_value = class_value;
                    }

                    @Override
                    public Boolean call() {
                        try {
                            this.nn.trainOnFeatureValues(this.featureValues, this.class_value);
//                        System.out.println(Thread.currentThread().getName());
                        } catch (NullPointerException e){
//                            System.err.println("Error: Model failed during training.");
                            e.printStackTrace();
                            System.exit(1);
                        }
                        return Boolean.TRUE;
                    }
                }
//                TrainThread[] trainThreadArray = new TrainThread[this.nn.length];
                final Future<Boolean> [] runFuture = new Future[this.nn.length];
                for (int i =0; i < this.nn.length; i++) {
//                    trainThreadArray[i] = new TrainThread(this.nn[i], this.featureValues, this.class_value);
                    runFuture[i] = exService.submit(new TrainThread(this.nn[i], this.featureValues, this.class_value));
                }
                int runningCount = this.nn.length;
                while (runningCount != 0){
                    runningCount = 0;
                    for (int i =0; i < this.nn.length; i++) {
                        try {
                            final Boolean returnedValue = runFuture[i].get();
                            if (!returnedValue.equals(Boolean.TRUE)){
                                runningCount++;
                            }
                        } catch (InterruptedException | ExecutionException e) {
                            e.printStackTrace();
                        }
                    }
                }
//                for (int i =0; i < trainThreadArray.length; i++) {
//                    trainThreadArray[i].start();
//                }

//                for (int i =0; i < trainThreadArray.length; i++) {
//                    try {
//                        trainThreadArray[i].join();
////                        trainThreadArray[i].
//                    }catch(Exception e) {
//                        System.err.println("Error: Model "+ i +" failed during training.");
//                        e.printStackTrace();
//                    }
//                }
                break;
        }
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        int min_index = 0;
        samplesSeen ++;
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

        MLP.setFeatureValuesArray(instance, featureValues, useOneHotEncode.isSet(), true, normalizeInfo, samplesSeen);
        return this.nn[min_index].getVotesForFeatureValues(instance, featureValues);
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
            private final int optimizerType;
            private final float learningRate;

            MLPConfigs(int optimizerType, float learningRate){
                this.optimizerType = optimizerType;
                this.learningRate = learningRate;
            }
        }
        MLPConfigs [] nnConfigs = {
                new MLPConfigs(MLP.OPTIMIZER_SGD, 0.03f),
//                new MLPConfigs(MLP.OPTIMIZER_SGD, 0.05f),
                new MLPConfigs(MLP.OPTIMIZER_SGD, 0.07f),
                new MLPConfigs(MLP.OPTIMIZER_RMSPROP, 0.01f),
                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD, 0.03f),
                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD_RESET, 0.03f),
                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD, 0.07f),
                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD, 0.09f),
                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD_RESET, 0.09f),
                new MLPConfigs(MLP.OPTIMIZER_ADAM_RESET, 0.01f),
                new MLPConfigs(MLP.OPTIMIZER_ADAM, 0.01f)};

        this.nn = new MLP[nnConfigs.length];
        for(int i=0; i < nnConfigs.length; i++){
            this.nn[i] = new MLP();
            this.nn[i].optimizerTypeOption.setChosenIndex(nnConfigs[i].optimizerType);
            this.nn[i].learningRateOption.setValue(nnConfigs[i].learningRate);
            this.nn[i].useOneHotEncode.setValue(useOneHotEncode.isSet());

            this.nn[i].initializeNetwork(instance);
        }

        exService = Executors.newFixedThreadPool(nnConfigs.length);

        class_value = new double[1];
        featureValuesArraySize = MLP.getFeatureValuesArraySize(instance, useOneHotEncode.isSet());
        featureValues = new float [featureValuesArraySize];
        if (useNormalization.isSet()) {
            normalizeInfo = new MLP.NormalizeInfo[featureValuesArraySize];
            for(int i=0; i < normalizeInfo.length; i++){
                normalizeInfo[i] = new MLP.NormalizeInfo();
            }
        }
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
//            if (this.getClass() == moa.classifiers.meta.StreamingRandomPatches.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
//            else
//                return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
