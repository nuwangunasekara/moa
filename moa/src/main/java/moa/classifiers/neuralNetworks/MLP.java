/*
 *    kNN.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.neuralNetworks;

import ai.djl.engine.Engine;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ADWIN;
//import moa.core.InstanceExample;
import moa.core.Measurement;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.MultiChoiceOption;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.optimizer.Optimizer;
//import moa.evaluation.BasicClassificationPerformanceEvaluator;

import java.text.DecimalFormat;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class MLP extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

	public static class NormalizeInfo{
		double sumOfValues = 0.0f;
		double sumOfSquares = 0.0f;
	}

	public static final int OPTIMIZER_SGD = 0;
	public static final int OPTIMIZER_RMSPROP = 1;
	public static final int OPTIMIZER_RMSPROP_RESET = 2;
	public static final int OPTIMIZER_ADAGRAD = 3;
	public static final int OPTIMIZER_ADAGRAD_RESET = 4;
	public static final int OPTIMIZER_ADAM = 5;
	public static final int OPTIMIZER_ADAM_RESET = 6;

	protected long samplesSeen = 0;
	protected long trainedCount = 0;
	protected NormalizeInfo[] normalizeInfo = null;

	private float[] pFeatureValues = null;
	private double [] pClassValue = null;

	public FloatOption learningRateOption = new FloatOption(
			"learningRate",
			'r',
			"Learning Rate",
			0.03, 0.0000001, 1.0);

	public FloatOption backPropLossThreshold = new FloatOption(
			"backPropLossThreshold",
			'b',
			"Back propagation loss threshold",
			0.5, 0.0, Math.pow(10,10));

	public MultiChoiceOption optimizerTypeOption = new MultiChoiceOption("optimizer", 'o',
			"Choose optimizer",
			new String[]{"SGD", "RMSPROP", "RMSPROP_RESET", "ADAGRAD", "ADAGRAD_RESET", "ADAM", "ADAM_RESET"},
			new String[]{"oSGD", "oRMSPROP", "oRMSPROP_RESET", "oADAGRAD", "oADAGRAD_RESET", "oADAM", "oADAM_RESET"},
			0);

	public FlagOption useOneHotEncode = new FlagOption("useOneHotEncode", 'h',
			"use one hot encoding");

	public FlagOption useNormalization = new FlagOption("useNormalization", 'n',
			"Normalize data");

	public IntOption numberOfNeuronsInL1InLog2 = new IntOption(
			"numberOfNeuronsInL1InLog2",
			'N',
			"Number of neurons in the 1st layer in log2",
			10, 2, 20);

	public IntOption numberOfLayers = new IntOption(
			"numberOfLayers",
			'L',
			"Number of layers",
			1, 1, 4);

	public static final int deviceTypeOptionGPU = 0;
	public static final int deviceTypeOptionCPU = 1;
	public MultiChoiceOption deviceTypeOption = new MultiChoiceOption("deviceType", 'd',
			"Choose device to run the model(use CPU if GPUs are not available)",
			new String[]{"GPU","CPU"},
			new String[]{"GPU (use CPU if not available)", "CPU"},
			deviceTypeOptionGPU);

	public FlagOption resetModel = new FlagOption("resetModel", 'R',
			"Reset the model (with weights) along with the optimizer if applicable (only for optimizers which support reset) on concept drafts");

//	public FloatOption deltaForADWINOption = new FloatOption(
//			"deltaForADWINOption",
//			'd',
//			"Delta for ADWI",
//			1.0E-10, 1.0E-20, 1.0);
	public double deltaForADWIN = 1.0E-5;

	public long optimizerResetCount = 0;

	@Override
    public String getPurposeString() {
        return "NN: special.";
    }

	public ADWIN lossEstimator;
	public float accumulatedLoss = 0;
	public int chosenCount = 0;
	public long modelResetCount = 0;
	public String modelName;


	protected Model nnmodel = null;
	protected Trainer trainer = null;
	protected int featureValuesArraySize = 0;
	private transient NDManager trainingNDManager;
	private transient NDManager testingNDManager;
	private int numberOfClasses;
	private double [] votes;
	private boolean resetOptimiser = false;
//	private BasicClassificationPerformanceEvaluator performanceEvaluator = new BasicClassificationPerformanceEvaluator();
//	private ADWIN accEstimator;
	private Device[] devices;
	private int gpuCount;
	private static DecimalFormat decimalFormat = new DecimalFormat("0.00000");


	@Override
	public void setModelContext(InstancesHeader context) {
		try {
//			this.window = new Instances(context,0); //new StringReader(context.toString())
//			this.window.setClassIndex(context.classIndex());
		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

    @Override
    public void resetLearningImpl() {
    }

    public double getAccuracy(){
//		return performanceEvaluator.getPerformanceMeasurements()[1].getValue();
//		return accEstimator.getEstimation();
		return 0.0f;
	}

	public void trainOnFeatureValues(float[] featureValues, double [] classValue /*, InstanceExample instanceExample*/) {
		samplesSeen ++;
		try{
			NDManager childNDManager = trainingNDManager.newSubManager();
			NDList d = new NDList(childNDManager.create(featureValues));
			NDList l = new NDList(childNDManager.create(classValue));

			GradientCollector collector = trainer.newGradientCollector();
			NDList preds = trainer.forward(d, l);
//			for (int i = 0; i < votes.length; i++) {
//				votes[i] = (double) preds.get(0).toFloatArray()[i];
//			}
//			performanceEvaluator.addResult(instanceExample, votes);
//			accEstimator.setInput(performanceEvaluator.getPerformanceMeasurements()[1].getValue());
			NDArray lossValue = trainer.getLoss().evaluate(l, preds);
			accumulatedLoss += lossValue.getFloat();

			double previousLossEstimation = lossEstimator.getEstimation();
			if (lossValue.getFloat() == 0.0f){
//				System.out.println("Zero loss");
				this.lossEstimator.setInput(0.0);
			}else{
				if (lossValue.getFloat() > backPropLossThreshold.getValue()){
					trainedCount++;
					try {
						collector.backward(lossValue);
						trainer.step(); // enforce the calculated weights
					}catch (IllegalStateException e)
					{
//					trainer.step() throws above exception if all gradients are zero.
					}
				}
				this.lossEstimator.setInput(lossValue.getFloat());
			}
			//			print weights
//			System.out.println(nnmodel.getBlock().getChildren().get("02Linear").getParameters().get("weight").getArray());

			if (resetOptimiser && lossEstimator.getChange() && (previousLossEstimation < lossEstimator.getEstimation()) ){
//				System.out.println("Resetting optimizer:" + optimizerTypeOption.getChosenLabel() + " learning rate: " + decimalFormat.format(learningRateOption.getValue()));
				optimizerResetCount++;
				if (resetModel.isSet()) {
					setModel();
					modelResetCount ++;
				}
				setTrainer();
			}

			collector.close();
			preds.close();
			lossValue.close();
			d.close();
			l.close();
			childNDManager.close();
		}catch (Exception e) {
			System.err.println(e);
			e.printStackTrace();
			System.exit(1);
		}
	}

    @Override
    public void trainOnInstanceImpl(Instance inst) {
		initializeNetwork(inst);

//		setFeatureValuesArray(inst, pFeatureValues, useOneHotEncode.isSet(), false, normalizeInfo, samplesSeen);
		pClassValue[0] = inst.classValue();

		trainOnFeatureValues(pFeatureValues, pClassValue/*, new InstanceExample(inst)*/);
    }

    public double[] getVotesForFeatureValues(Instance inst, float[] featureValues) {
		initializeNetwork(inst);

		try {
			NDManager childNDManager = testingNDManager.newSubManager();
			NDList d = new NDList(childNDManager.create(featureValues));
			NDList preds = trainer.evaluate(d);

			for (int i = 0; i < inst.numClasses(); i++) {
				votes[i] = (double) preds.get(0).toFloatArray()[i];
			}
			preds.close();
			d.close();
			childNDManager.close();
		}catch (Exception e) {
			System.err.println(e);
			e.printStackTrace();
			System.exit(1);
		}

		return votes;
    }

	@Override
	public double[] getVotesForInstance(Instance inst) {

		initializeNetwork(inst);

		setFeatureValuesArray(inst, pFeatureValues, useOneHotEncode.isSet(), true, normalizeInfo, samplesSeen);
		return getVotesForFeatureValues(inst, pFeatureValues);
	}


	@Override
	public ImmutableCapabilities defineImmutableCapabilities() {
//		if (this.getClass() == StreamingRandomPatches.class)
			return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
//		else
//			return new ImmutableCapabilities(Capability.VIEW_STANDARD);
	}

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return false;
    }

    public static int getFeatureValuesArraySize(Instance inst, boolean useOneHotEncoding){
		int totalOneHotEncodedSize = 0;
		int totalOneHotEncodedInstances = 0;
		for(int i=0; i < inst.numInputAttributes(); i++){
			if (useOneHotEncoding && inst.attribute(i).isNominal() && (inst.attribute(i).numValues() > 2) ){
				totalOneHotEncodedSize += inst.attribute(i).numValues();
				totalOneHotEncodedInstances ++;
			}
		}
		return inst.numInputAttributes() + totalOneHotEncodedSize - totalOneHotEncodedInstances;
	}

	public static double getNormalizedValue(double value, double sumOfValues, double sumOfSquares, long samplesSeen){
		// Normalize data
		double variance = 0.0f;
		double sd;
		double mean = 0.0f;
		if (samplesSeen > 1){
			mean = sumOfValues / samplesSeen;
			variance = (sumOfSquares - ((sumOfValues * sumOfValues) / samplesSeen)) / samplesSeen;
		}
		sd = Math.sqrt(variance);
		if (sd > 0.0f){
			return (value - mean) / (3 * sd);
		} else{
			return 0.0f;
		}
	}

    public static void setFeatureValuesArray(Instance inst, float[] featureValuesArrayToSet, boolean useOneHotEncoding, boolean testing, NormalizeInfo[] normalizeInfo, long samplesSeen){
		int totalOneHotEncodedSize = 0;
		int totalOneHotEncodedInstances = 0;
		for(int i=0; i < inst.numInputAttributes(); i++){
			int index = i + totalOneHotEncodedSize - totalOneHotEncodedInstances;
			if (useOneHotEncoding && inst.attribute(i).isNominal() && (inst.attribute(i).numValues() > 2) ){
				// Do one hot-encoding
				featureValuesArrayToSet[index + (int)inst.value(i)] = 1.0f;
				totalOneHotEncodedSize += inst.attribute(i).numValues();
				totalOneHotEncodedInstances ++;
			}else
			{
				if( inst.attribute(i).isNumeric() && (normalizeInfo != null) && (normalizeInfo[index] != null) ){
					// Normalize data
					if (testing) {
						normalizeInfo[index].sumOfSquares += inst.value(i) * inst.value(i);
						normalizeInfo[index].sumOfValues += inst.value(i);
					}
					featureValuesArrayToSet[index] = (float) getNormalizedValue(inst.value(i), normalizeInfo[index].sumOfValues, normalizeInfo[index].sumOfSquares, samplesSeen);
				}else{
					featureValuesArrayToSet[index] = (float) inst.value(i);
				}
			}
		}

//		if (testing && samplesSeen < 2) {
//			for(int i=0; i < featureValuesArrayToSet.length; i++) {
//				System.out.print(featureValuesArrayToSet[i]);
//				System.out.print(",");
//			}
//		}
//		if (testing && samplesSeen < 2){
//			System.out.print("\n");
//		}
	}

	public void initializeNetwork(Instance inst) {
		if (nnmodel != null){
			return;
		}

		votes = new double [inst.numClasses()];

		pClassValue =  new double[1];
		featureValuesArraySize = getFeatureValuesArraySize(inst, useOneHotEncode.isSet());
		pFeatureValues = new float [featureValuesArraySize];
		if (useNormalization.isSet()) {
			normalizeInfo = new NormalizeInfo[featureValuesArraySize];
			for(int i=0; i < normalizeInfo.length; i++){
				normalizeInfo[i] = new NormalizeInfo();
			}
		}

		try {
			gpuCount = Device.getGpuCount();
			devices = Device.getDevices(gpuCount);

			numberOfClasses = inst.numClasses();
			setModel();

			lossEstimator = new ADWIN(deltaForADWIN);
//			accEstimator = new ADWIN(deltaForADWIN);

			switch (this.optimizerTypeOption.getChosenIndex()){
				case MLP.OPTIMIZER_RMSPROP_RESET:
				case MLP.OPTIMIZER_ADAGRAD_RESET:
				case MLP.OPTIMIZER_ADAM_RESET:
					resetOptimiser = true;
					break;
				default:
					resetOptimiser = false;
					break;
			}
			setTrainer();
		}catch (Exception e) {
			System.err.println(e);
			e.printStackTrace();
		}
		modelName = "L" + numberOfLayers.getValue() + "_L1n" + numberOfNeuronsInL1InLog2.getValue() +"_" + optimizerTypeOption.getChosenLabel() +"_" + decimalFormat.format(learningRateOption.getValue()) + "_" + deltaForADWIN;
	}

	public double getLossEstimation(){
		if (samplesSeen == 0)
			return 0.0f;
//		return accumulatedLoss/samplesSeen;
		return lossEstimator.getEstimation();
	}

	protected void setModel(){
		try{
			nnmodel = Model.newInstance("mlp", ((deviceTypeOption.getChosenIndex() == deviceTypeOptionGPU) && (gpuCount > 0)) ? Device.gpu() : Device.cpu());
			// Construct neural network and set it in the block
			Integer [] neuronsInEachHiddenLayer = {};
			List<Integer> neuronsInEachHiddenLayerArrayList = new ArrayList<Integer>(Arrays.asList(neuronsInEachHiddenLayer));
			for(int i=0; i < numberOfLayers.getValue(); i++){
				neuronsInEachHiddenLayerArrayList.add((int) Math.pow(2, numberOfNeuronsInL1InLog2.getValue() - i));
			}
			neuronsInEachHiddenLayer = neuronsInEachHiddenLayerArrayList.toArray(neuronsInEachHiddenLayer);
			Block block = new Mlp(featureValuesArraySize, numberOfClasses, Arrays.stream(neuronsInEachHiddenLayer).mapToInt(Integer::intValue).toArray());
//		Block block = new Mlp(featureValuesArraySize, inst.numClasses(), new int[] {2});
			nnmodel.setBlock(block);
			trainingNDManager = Engine.getInstance().newBaseManager(nnmodel.getNDManager().getDevice());
			testingNDManager = Engine.getInstance().newBaseManager(nnmodel.getNDManager().getDevice());
			System.out.println("Model using Device: " + nnmodel.getNDManager().getDevice() + "Trainer using Device: " + trainingNDManager.getDevice() + "Tester using Device: " + testingNDManager.getDevice());

		}catch (Exception e) {
			System.err.println(e);
			e.printStackTrace();
		}
	}

	protected void setTrainer(){
		if (trainer != null){
			trainer.close();
			trainer = null;
		}
		try {
		Tracker learningRateTracker = Tracker.fixed((float) this.learningRateOption.getValue());
		Optimizer optimizer;

		switch(this.optimizerTypeOption.getChosenIndex()) {
			case MLP.OPTIMIZER_RMSPROP_RESET:
			case MLP.OPTIMIZER_RMSPROP:
				optimizer = Optimizer.rmsprop().optLearningRateTracker(learningRateTracker).build();
				break;
			case MLP.OPTIMIZER_ADAGRAD_RESET:
			case MLP.OPTIMIZER_ADAGRAD:
				optimizer = Optimizer.adagrad().optLearningRateTracker(learningRateTracker).build();
				break;
			case MLP.OPTIMIZER_ADAM_RESET:
			case MLP.OPTIMIZER_ADAM:
				optimizer = Optimizer.adam().optLearningRateTracker(learningRateTracker).build();
				break;
			case MLP.OPTIMIZER_SGD:
			default:
				optimizer = Optimizer.sgd().setLearningRateTracker(learningRateTracker).build();
				break;
		}
		//softmaxCrossEntropyLoss is a standard loss for classification problems
		Loss loss = Loss.softmaxCrossEntropyLoss();
		DefaultTrainingConfig config = new DefaultTrainingConfig(loss);
//				.optOptimizer(optimizer);
//				.addEvaluator(new Accuracy()) // Use accuracy so we humans can understand how accurate the model is
//				.addTrainingListeners(TrainingListener.Defaults.logging());
		config.optOptimizer(optimizer);
		trainer = nnmodel.newTrainer(config);
		trainer.initialize(new Shape(1, featureValuesArraySize));
		}catch (Exception e) {
			System.err.println(e);
			e.printStackTrace();
		}
	}
}