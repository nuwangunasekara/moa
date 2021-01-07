package moa.classifiers.neuralNetworks;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
//import moa.evaluation.BasicClassificationPerformanceEvaluator;

import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
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
    private FileWriter statsDumpFile;
//    private BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();
    private static DecimalFormat decimalFormat = new DecimalFormat("0.00000");

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
                    private final MLP mlp;
                    private final float[] featureValues;
                    private final double [] class_value;

                    public TrainThread(MLP mlp, float[] featureValues, double [] class_value){
                        this.mlp = mlp;
                        this.featureValues = featureValues;
                        this.class_value = class_value;
                    }

                    @Override
                    public Boolean call() {
                        try {
                            this.mlp.trainOnFeatureValues(this.featureValues, this.class_value);
//                        System.out.println(Thread.currentThread().getName());
                        } catch (NullPointerException e){
//                            System.err.println("Error: Model failed during training.");
                            e.printStackTrace();
                            System.exit(1);
                        }
                        return Boolean.TRUE;
                    }
                }

//                start threads
                int numberOfMLPsToTrain = 3; // min value should be 2
                int indexOfTheLastNetworkToTrain = (int)(samplesSeen % this.nn.length);
                if (samplesSeen < 500){
                    numberOfMLPsToTrain = this.nn.length;
                    indexOfTheLastNetworkToTrain = this.nn.length - 1;
                }else{
                    if (indexOfTheLastNetworkToTrain < numberOfMLPsToTrain){
//                        We train NN s up till numberOfMLPsToTrain - 2 any how.
                        indexOfTheLastNetworkToTrain = numberOfMLPsToTrain + indexOfTheLastNetworkToTrain; // need to make sure that the array doesn't overflow
                    }

                    Arrays.sort(this.nn, new Comparator<MLP>() {
                        @Override
                        public int compare(MLP o1, MLP o2) {
                            return Double.compare(o1.estimator.getEstimation(), o2.estimator.getEstimation());
                        }
                    });
                }

                final Future<Boolean> [] runFuture = new Future[numberOfMLPsToTrain];
                for (int i =0; i < numberOfMLPsToTrain; i++) {
                    if (i <= numberOfMLPsToTrain - 2){
                        runFuture[i] = exService.submit(new TrainThread(this.nn[i], this.featureValues, this.class_value));
                    }else {
                        // i == numberOfMLPsToTrain - 1
                        runFuture[i] = exService.submit(new TrainThread(this.nn[indexOfTheLastNetworkToTrain], this.featureValues, this.class_value));
                    }
                }

//                wait for threads to complete
                int runningCount = numberOfMLPsToTrain;
                while (runningCount != 0){
                    runningCount = 0;
                    for (int i =0; i < numberOfMLPsToTrain; i++) {
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

                break;
        }
    }

    private void printStats(){
        for (int i = 0 ; i < this.nn.length ; i++) {
            try {
                statsDumpFile.write(this.nn[i].samplesSeen + "," + this.nn[i].numberOfNeuronsIn2Power.getValue() + this.nn[i].optimizerTypeOption.getChosenLabel() + decimalFormat.format(this.nn[i].learningRateOption.getValue()) + "," + this.nn[i].estimator.getEstimation() + "," + this.nn[i].chosenCount + "\n");
                statsDumpFile.flush();
            } catch (IOException e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        int min_index = 0;
        samplesSeen ++;
        if(this.nn == null) {
            initNNs(instance);
        }else {
            double min_estimation = Double.MAX_VALUE;
            for (int i = 0 ; i < this.nn.length ; i++) {
                if (this.nn[i].estimator.getEstimation() < min_estimation){
                    min_estimation = this.nn[i].estimator.getEstimation();
                    min_index = i;
                }
            }
        }
        MLP.setFeatureValuesArray(instance, featureValues, useOneHotEncode.isSet(), true, normalizeInfo, samplesSeen);
        this.nn[min_index].chosenCount++;
//        this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());
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
        printStats();
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
            private final int numberOfNeuronsIn2Power;
            private final int optimizerType;
            private final float learningRate;

            MLPConfigs(int numberOfNeuronsIn2Power, int optimizerType, float learningRate){
                this.numberOfNeuronsIn2Power = numberOfNeuronsIn2Power;
                this.optimizerType = optimizerType;
                this.learningRate = learningRate;
            }
        }

        MLPConfigs [] nnConfigs = {
//                new MLPConfigs(10, MLP.OPTIMIZER_RMSPROP, 0.01f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAGRAD, 0.03f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAGRAD, 0.07f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAGRAD,0.09f),

//                new MLPConfigs(10, MLP.OPTIMIZER_SGD, 0.00075f),
//                new MLPConfigs(10, MLP.OPTIMIZER_SGD, 0.0005f),
//                new MLPConfigs(10, MLP.OPTIMIZER_SGD, 0.0075f),
//                new MLPConfigs(10, MLP.OPTIMIZER_SGD, 0.005f),

//                new MLPConfigs(10, MLP.OPTIMIZER_SGD, 0.000025f),
//                new MLPConfigs(10, MLP.OPTIMIZER_SGD, 0.00025f),
//                new MLPConfigs(10, MLP.OPTIMIZER_SGD, 0.0025f),
//                new MLPConfigs(10, MLP.OPTIMIZER_SGD, 0.025f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.000075f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.000025f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.00075f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.00025f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.0075f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.005f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.0025f),
//                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.025f),
        };

        List<MLPConfigs> nnConfigsArrayList = new ArrayList<MLPConfigs>(Arrays.asList(nnConfigs));
        float [] denominator = {100.0f, 1000.f, 10000.0f};
        float [] numerator = {1.0f, 2.5f, 5.0f, 7.5f};
        for (int n=0; n < numerator.length; n++){
            for (int d=0; d < denominator.length; d++){
                float lr = numerator[n]/denominator[d];
                nnConfigsArrayList.add(new MLPConfigs(8, MLP.OPTIMIZER_SGD, lr));
                nnConfigsArrayList.add(new MLPConfigs(8, MLP.OPTIMIZER_ADAM, lr));
            }
        }
        nnConfigs = nnConfigsArrayList.toArray(nnConfigs);

        this.nn = new MLP[nnConfigs.length];
        for(int i=0; i < nnConfigs.length; i++){
            this.nn[i] = new MLP();
            this.nn[i].optimizerTypeOption.setChosenIndex(nnConfigs[i].optimizerType);
            this.nn[i].learningRateOption.setValue(nnConfigs[i].learningRate);
            this.nn[i].useOneHotEncode.setValue(useOneHotEncode.isSet());
            this.nn[i].numberOfNeuronsIn2Power.setValue(nnConfigs[i].numberOfNeuronsIn2Power);

            this.nn[i].initializeNetwork(instance);
        }

        try {
            statsDumpFile = new FileWriter("NN_loss.csv");
            statsDumpFile.write("samplesSeenAtTrain,optimizer_type_learning_rate,estimated_loss,chosen_counts\n");
            statsDumpFile.flush();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        exService = Executors.newFixedThreadPool(nnConfigs.length);

        class_value = new double[1];
        featureValuesArraySize = MLP.getFeatureValuesArraySize(instance, useOneHotEncode.isSet());
        System.out.println("Number of features before one-hot encode: " + instance.numInputAttributes() + " : Number of features after one-hot encode: " + featureValuesArraySize);
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
