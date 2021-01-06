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
                int numberOfMLPsToTrain = 4;
                if ((samplesSeen < 500) || (samplesSeen % 10 == 0)){
                    numberOfMLPsToTrain = this.nn.length;
                }else{
                    Arrays.sort(this.nn, new Comparator<MLP>() {
                        @Override
                        public int compare(MLP o1, MLP o2) {
                            return Double.compare(o1.estimator.getEstimation(), o2.estimator.getEstimation());
                        }
                    });
                }

                final Future<Boolean> [] runFuture = new Future[numberOfMLPsToTrain];
                for (int i =0; i < numberOfMLPsToTrain; i++) {
                    runFuture[i] = exService.submit(new TrainThread(this.nn[i], this.featureValues, this.class_value));
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
                statsDumpFile.write(samplesSeen + "," + this.nn[i].numberOfNeuronsIn2Power.getValue() + this.nn[i].optimizerTypeOption.getChosenLabel() + decimalFormat.format(this.nn[i].learningRateOption.getValue()) + "," + this.nn[i].estimator.getEstimation() + "," + this.nn[i].chosenCount + "\n");
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
                new MLPConfigs(8, MLP.OPTIMIZER_SGD, 0.03f),
//                new MLPConfigs(MLP.OPTIMIZER_SGD, 0.05f),
                new MLPConfigs(8, MLP.OPTIMIZER_SGD, 0.07f),
//                new MLPConfigs(MLP.OPTIMIZER_RMSPROP, 0.01f),
//                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD, 0.03f),
//                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD_RESET, 0.03f),
//                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD, 0.07f),
//                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD, 0.09f),
//                new MLPConfigs(MLP.OPTIMIZER_ADAGRAD_RESET, 0.09f),
                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.0005f),
                new MLPConfigs(10, MLP.OPTIMIZER_ADAM_RESET, 0.0005f),
                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.001f),
                new MLPConfigs(10, MLP.OPTIMIZER_ADAM_RESET, 0.001f),
                new MLPConfigs(10, MLP.OPTIMIZER_ADAM, 0.005f),
                new MLPConfigs(10, MLP.OPTIMIZER_ADAM_RESET, 0.005f),
                new MLPConfigs(8, MLP.OPTIMIZER_ADAM, 0.03f),
                new MLPConfigs(8, MLP.OPTIMIZER_ADAM_RESET, 0.03f),

        };

//        if (numberOfNeuronsIn2Power.getValue() <= 8) {
            List<MLPConfigs> nnConfigsArrayList = new ArrayList<MLPConfigs>(Arrays.asList(nnConfigs));
            nnConfigsArrayList.add(new MLPConfigs(8, MLP.OPTIMIZER_SGD, 0.0005f));
            nnConfigsArrayList.add(new MLPConfigs(8, MLP.OPTIMIZER_SGD, 0.001f));
            nnConfigsArrayList.add(new MLPConfigs(8, MLP.OPTIMIZER_ADAM, 0.07f));
            nnConfigsArrayList.add(new MLPConfigs(8, MLP.OPTIMIZER_ADAM_RESET, 0.07f));
            nnConfigs = nnConfigsArrayList.toArray(nnConfigs);
//        }


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
            statsDumpFile.write("id,optimizer_type_learning_rate,estimated_loss,chosen_counts\n");
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
