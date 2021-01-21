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
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.evaluation.BasicClassificationPerformanceEvaluator;

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
    private BasicClassificationPerformanceEvaluator performanceEvaluator = new BasicClassificationPerformanceEvaluator();
    private int instancesToProcessSinceLastDrift = 0;
    private long driftsDetectedPer1000Samples = 0;
    private long lastDriftDetectedAt = 0;
    private long learnedOnDriftsFor1000Samples = 0;
    private long totalDriftsDetected = 0;
    private long avgMLPsFor1000Samples = 0;

    private int numberOfMLPsToTrainAtStart;
    private int numberOfTopMLPsToTrainAtStart;
    private int maxInstancesToTrainAtStart;

    private int numberOfMLPsToTrainOffDrift;
    private int numberOfTopMLPsToTrainOffDrift;

    private int numberOfMLPsToTrainOnDrift;
    private int numberOfTopMLPsToTrainOrOnDrift;
    private int instancesToProcessAfterADrift;




    private ADWIN driftDetector = new ADWIN(1.0E-3);

    private static DecimalFormat decimalFormat = new DecimalFormat("0.00000");

    public FlagOption useOneHotEncode = new FlagOption("useOneHotEncode", 'h',
            "use one hot encoding");

    public FlagOption useNormalization = new FlagOption("useNormalization", 'n',
            "Normalize data");

    public static final int TRAIN_SEQUENTIAL = 0;
    public static final int TRAIN_USE_THREADS = 1;

    public MultiChoiceOption trainingMethodOption = new MultiChoiceOption("trainingMethod", 't',
            "The training method to use: Sequential, Use threads",
            new String[]{"Sequential", "UseThreads"},
            new String[]{"Sequential", "UseThreads"}, TRAIN_USE_THREADS);

    public static final int SELECT_USING_LOSS = 0;
    public static final int SELECT_USING_ACCURACY = 1;

    public MultiChoiceOption modelSelectionCriteria = new MultiChoiceOption("modelSelectionCriteria", 'm',
            "The model selection criteria",
            new String[]{"loss", "accuracy"},
            new String[]{"loss", "accuracy"}, SELECT_USING_LOSS);

    public MultiChoiceOption deviceTypeOption = new MultiChoiceOption("deviceType", 'd',
            "Choose device to run the model(use CPU if GPUs are not available)",
            new String[]{"GPU","CPU"},
            new String[]{"GPU (use CPU if not available)", "CPU"},
            MLP.deviceTypeOptionCPU);

    public IntOption numberOfInstancesToTrainAtStartOption = new IntOption(
            "numberOfInstancesToTrainAtStart",
            's',
            "Number of instances to train at start",
            100, 0, Integer.MAX_VALUE);

    public IntOption numberOfMLPsToTrainOffDriftOption = new IntOption(
            "numberOfMLPsToTrainOffDrift",
            'o',
            "Number of MLPs to train off drift",
            2, 2, Integer.MAX_VALUE);

    public IntOption numberOfMLPsToTrainOnDriftOption = new IntOption(
            "numberOfMLPsToTrainOnDrift",
            'O',
            "number of MLPs to train on drift",
            10, 2, Integer.MAX_VALUE);

    public IntOption numberOfNeuronsIn2Power = new IntOption(
            "numberOfNeuronsIn2Power",
            'N',
            "Number of neurons in the 1st layer in 2's power",
            8, 2, 20);


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
//        InstanceExample instanceExample = new InstanceExample(instance);

        switch (this.trainingMethodOption.getChosenIndex()) {
            case TRAIN_SEQUENTIAL:
                for (MLP mlp : this.nn) {
                    mlp.initializeNetwork(instance);
                    mlp.trainOnFeatureValues(featureValues, class_value/*, instanceExample*/);
                }
                break;
            case TRAIN_USE_THREADS:
                class TrainThread implements Callable<Boolean> {
                    private final MLP mlp;
                    private final float[] featureValues;
                    private final double [] class_value;
//                    private final InstanceExample instanceExample;

                    public TrainThread(MLP mlp, float[] featureValues, double [] class_value/*, InstanceExample instanceExample*/){
                        this.mlp = mlp;
                        this.featureValues = featureValues;
                        this.class_value = class_value;
//                        this.instanceExample = instanceExample;
                    }

                    @Override
                    public Boolean call() {
                        try {
                            this.mlp.trainOnFeatureValues(this.featureValues, this.class_value/*, instanceExample*/);
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
                int numberOfMLPsToTrain = numberOfMLPsToTrainOffDrift;
                int numberOfTopMLPsToTrain = numberOfTopMLPsToTrainOffDrift;
                if (samplesSeen < maxInstancesToTrainAtStart){
                    numberOfMLPsToTrain = numberOfMLPsToTrainAtStart;
                    numberOfTopMLPsToTrain = numberOfTopMLPsToTrainAtStart;
                }else if (instancesToProcessSinceLastDrift > 0){
                    numberOfMLPsToTrain = numberOfMLPsToTrainOnDrift;
                    numberOfTopMLPsToTrain = numberOfTopMLPsToTrainOrOnDrift;
                    learnedOnDriftsFor1000Samples++;
//                    System.out.println("@ "+samplesSeen
//                            +" driftsDetectedPer1000Samples : "+ driftsDetectedPer1000Samples
//                            +" learnedOnDriftsFor1000Samples : " + learnedOnDriftsFor1000Samples
//                            +" numberOfMLPsToTrain : " + numberOfMLPsToTrain);
                    instancesToProcessSinceLastDrift --;
                }
                avgMLPsFor1000Samples += numberOfMLPsToTrain;

                if (modelSelectionCriteria.getChosenIndex() == SELECT_USING_LOSS){
                    Arrays.sort(this.nn, new Comparator<MLP>() {
                        @Override
                        public int compare(MLP o1, MLP o2) {
                            return Double.compare(o1.getLossEstimation(), o2.getLossEstimation());
                        }
                    });
                }else{
                    Arrays.sort(this.nn, new Comparator<MLP>() {
                        @Override
                        public int compare(MLP o1, MLP o2) {
                            return Double.compare(o1.getAccuracy(), o2.getAccuracy());
                        }
                    });
                }

                final Future<Boolean> [] runFuture = new Future[numberOfMLPsToTrain];
                for (int i =0; i < numberOfMLPsToTrain; i++) {
                    int nnIndex;
                    if (i < numberOfTopMLPsToTrain){
                        // top most train
                        if (modelSelectionCriteria.getChosenIndex() == SELECT_USING_LOSS) {
                            nnIndex = i;
                        }else {
                            nnIndex = this.nn.length - 1 - i;
                        }
                    }else {
                        // Random train
                        int offSet = (int) ((samplesSeen + i) % (this.nn.length  - numberOfTopMLPsToTrain));
                        if (modelSelectionCriteria.getChosenIndex() == SELECT_USING_LOSS) {
                            nnIndex = numberOfTopMLPsToTrain + offSet;
                        }else{
                            nnIndex = this.nn.length - 1 - numberOfTopMLPsToTrain - offSet;
                        }
                    }
                    runFuture[i] = exService.submit(new TrainThread(this.nn[nnIndex], this.featureValues, this.class_value/*, instanceExample*/));
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
                statsDumpFile.write(samplesSeen + ","
                        + this.nn[i].samplesSeen + ","
                        + this.nn[i].numberOfNeuronsIn2Power.getValue() +"_" + this.nn[i].optimizerTypeOption.getChosenLabel() +"_" + decimalFormat.format(this.nn[i].learningRateOption.getValue()) + "_" + this.nn[i].deltaForADWIN + ","
                        + performanceEvaluator.getPerformanceMeasurements()[1].getValue() + ","
                        + this.nn[i].getAccuracy() + ","
                        + this.nn[i].accumulatedLoss/this.nn[i].samplesSeen + ","
                        + this.nn[i].lossEstimator.getEstimation() + ","
                        + this.nn[i].chosenCount + ","
                        + totalDriftsDetected + ","
                        + driftsDetectedPer1000Samples + ","
                        + learnedOnDriftsFor1000Samples + ","
                        + avgMLPsFor1000Samples/1000 + "\n");
                statsDumpFile.flush();
            } catch (IOException e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        int chosenIndex = 0;
        double [] votes;
        samplesSeen ++;
        if( (samplesSeen % 1000) == 1){
            driftsDetectedPer1000Samples = 0;
            learnedOnDriftsFor1000Samples = 0;
            avgMLPsFor1000Samples = 0;
        }
        if(this.nn == null) {
            initNNs(instance);
        }else {
            if (modelSelectionCriteria.getChosenIndex() == SELECT_USING_LOSS) {
                double minEstimation = Double.MAX_VALUE;
                for (int i = 0 ; i < this.nn.length ; i++) {
                    if (this.nn[i].getLossEstimation() < minEstimation){
                        minEstimation = this.nn[i].getLossEstimation();
                        chosenIndex = i;
                    }
                }
            }else{
                double maxAcc = Double.MIN_VALUE;
                for (int i = 0 ; i < this.nn.length ; i++) {
                    if (this.nn[i].getAccuracy() > maxAcc){
                        maxAcc = this.nn[i].getAccuracy();
                        chosenIndex = i;
                    }
                }
            }
        }
        MLP.setFeatureValuesArray(instance, featureValues, useOneHotEncode.isSet(), true, normalizeInfo, samplesSeen);
        this.nn[chosenIndex].chosenCount++;
        votes = this.nn[chosenIndex].getVotesForFeatureValues(instance, featureValues);
        performanceEvaluator.addResult(new InstanceExample(instance), votes);
        double lastAcc = driftDetector.getEstimation();
        driftDetector.setInput(performanceEvaluator.getPerformanceMeasurements()[1].getValue());
        if (driftDetector.getChange() && (driftDetector.getEstimation() < lastAcc)){
            if ((samplesSeen - lastDriftDetectedAt) > instancesToProcessAfterADrift){
                instancesToProcessSinceLastDrift = instancesToProcessAfterADrift;
                lastDriftDetectedAt = samplesSeen;
            }
            totalDriftsDetected++;
            driftsDetectedPer1000Samples++;
        }
        return votes;
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
//        return new Measurement[]{
//                new Measurement("drifts detected", driftsDetected)};
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
            private final double deltaForADWIN;

            MLPConfigs(int numberOfNeuronsIn2Power, int optimizerType, float learningRate, double deltaForADWIN){
                this.numberOfNeuronsIn2Power = numberOfNeuronsIn2Power;
                this.optimizerType = optimizerType;
                this.learningRate = learningRate;
                this.deltaForADWIN = deltaForADWIN;
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
        float [] denominator = {10.0f, 100.0f, 1000.0f, 10000.0f, 100000.0f};
        float [] numerator = {1.0f, 1.25f, 2.5f, 3.75f, 5.0f, 6.25f, 7.5f};
        for (int n=0; n < numerator.length; n++){
            for (int d=0; d < denominator.length; d++){
                float lr = numerator[n]/denominator[d];
                nnConfigsArrayList.add(new MLPConfigs(numberOfNeuronsIn2Power.getValue(), MLP.OPTIMIZER_ADAM, lr, 1.0E-3));
            }
        }
        nnConfigs = nnConfigsArrayList.toArray(nnConfigs);

        this.nn = new MLP[nnConfigs.length];
        for(int i=0; i < nnConfigs.length; i++){
            this.nn[i] = new MLP();
            this.nn[i].optimizerTypeOption.setChosenIndex(nnConfigs[i].optimizerType);
            this.nn[i].learningRateOption.setValue(nnConfigs[i].learningRate);
            this.nn[i].useOneHotEncode.setValue(useOneHotEncode.isSet());
            this.nn[i].deviceTypeOption.setChosenIndex(deviceTypeOption.getChosenIndex());
            this.nn[i].numberOfNeuronsIn2Power.setValue(nnConfigs[i].numberOfNeuronsIn2Power);
            this.nn[i].deltaForADWIN = nnConfigs[i].deltaForADWIN;

            this.nn[i].initializeNetwork(instance);
        }

        try {
            statsDumpFile = new FileWriter("NN_loss.csv");
            statsDumpFile.write("id," +
                    "samplesSeenAtTrain," +
                    "optimizer_type_learning_rate_delta," +
                    "acc,model_acc," +
                    "avg_loss," +
                    "estimated_loss," +
                    "chosen_counts," +
                    "totalDriftsDetected," +
                    "driftsDetectedPer1000Samples," +
                    "learnedOnDriftsFor1000Samples," +
                    "avgMLPsFor1000Samples" +
                    "\n");
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

        numberOfMLPsToTrainAtStart = nn.length;
        numberOfTopMLPsToTrainAtStart = numberOfMLPsToTrainAtStart;
        maxInstancesToTrainAtStart = numberOfInstancesToTrainAtStartOption.getValue();

        numberOfMLPsToTrainOffDrift = numberOfMLPsToTrainOffDriftOption.getValue();
        numberOfTopMLPsToTrainOffDrift = numberOfMLPsToTrainOffDrift /2;

        numberOfMLPsToTrainOnDrift = numberOfMLPsToTrainOnDriftOption.getValue();
        numberOfTopMLPsToTrainOrOnDrift = numberOfMLPsToTrainOnDrift /2;
        instancesToProcessAfterADrift = 3 * (nn.length);
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
//            if (this.getClass() == moa.classifiers.meta.StreamingRandomPatches.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
//            else
//                return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
