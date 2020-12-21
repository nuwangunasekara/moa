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

import com.github.javacliparser.FlagOption;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.Measurement;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.MultiChoiceOption;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.optimizer.Optimizer;


public class MLP extends AbstractClassifier implements MultiClassClassifier, Regressor {

    private static final long serialVersionUID = 1L;

	public static class NormalizeInfo{
		double sumOfValues = 0.0f;
		double sumOfSquares = 0.0f;
	}

	public static final int OPTIMIZER_SGD = 0;
	public static final int OPTIMIZER_RMSPROP = 1;
	public static final int OPTIMIZER_ADAGRAD = 2;
	public static final int OPTIMIZER_ADAM = 3;

	protected long samplesSeen = 0;
	protected NormalizeInfo[] normalizeInfo = null;

	public FloatOption learningRateOption = new FloatOption(
			"learningRate",
			'r',
			"Learning Rate",
			0.03, 0.001, 1.0);

	public MultiChoiceOption optimizerTypeOption = new MultiChoiceOption("optimizer", 'o',
			"Choose optimizer",
			new String[]{"SGD", "RMSPROP", "ADAGRAD", "ADAM"},
			new String[]{"oSGD", "oRMSPROP", "oADAGRAD", "oADAM"},
			0);

	public FlagOption useOneHotEncode = new FlagOption("useOneHotEncode", 'h',
			"use one hot encoding");

	public FlagOption useNormalization = new FlagOption("useNormalization", 'n',
			"Normalize data");

	@Override
    public String getPurposeString() {
        return "NN: special.";
    }

    public ADWIN estimator = new ADWIN();


	protected Model nnmodel = null;
	protected Trainer trainer = null;
	protected int featureValuesArraySize = 0;

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

	public void trainOnFeatureValues(float[] featureValues, double class_value[]) {
		NDList d = new NDList(trainer.getManager().create(featureValues));
		NDList l = new NDList(trainer.getManager().create(class_value));
//		GradientCollector collector = trainer.newGradientCollector();
		boolean calculateGradients = true;
		try (GradientCollector collector = trainer.newGradientCollector()) {
			NDList preds = trainer.forward(d, l);
			NDArray lossValue = trainer.getLoss().evaluate(l, preds);
			if (lossValue.getFloat() == 0.0f){
//				System.out.println("Zero loss");
				calculateGradients = false;
			}
			//			print weights
//			System.out.println(nnmodel.getBlock().getChildren().get("02Linear").getParameters().get("weight").getArray());
			if (calculateGradients) {
				collector.backward(lossValue);
				this.estimator.setInput(lossValue.getFloat());
			}else {
				this.estimator.setInput(0.0);
			}
			lossValue.close();
		}catch (Exception e) {
			System.err.println(e);
			e.printStackTrace();
		}
		if (calculateGradients) {
			trainer.step(); // enforce the calculated weights
		}
		d.close();
		l.close();
	}

    @Override
    public void trainOnInstanceImpl(Instance inst) {
		initializeNetwork(inst);

		float[] featureValues = new float [featureValuesArraySize];
		setFeatureValuesArray(inst, featureValues, useOneHotEncode.isSet(), false, normalizeInfo, samplesSeen);
		double class_value[] = {inst.classValue()};

		trainOnFeatureValues(featureValues, class_value);
    }

    public double[] getVotesForFeatureValues(Instance inst, float[] featureValues) {
		initializeNetwork(inst);
		double v [] = new double [inst.numClasses()];

		NDList d = new NDList(trainer.getManager().create(featureValues));
		NDList preds = trainer.evaluate(d);

		for(int i=0; i<inst.numClasses(); i++){
			v[i] = (double)preds.get(0).toFloatArray()[i];
		}

		preds.close();
		d.close();

		return v;
    }

	@Override
	public double[] getVotesForInstance(Instance inst) {
		samplesSeen ++;
		initializeNetwork(inst);
		float[] featureValues = new float [featureValuesArraySize];
		setFeatureValuesArray(inst, featureValues, useOneHotEncode.isSet(), true, normalizeInfo, samplesSeen);
		return getVotesForFeatureValues(inst, featureValues);
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
		double sd = 0.0f;
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

		featureValuesArraySize = getFeatureValuesArraySize(inst, useOneHotEncode.isSet());
		if (useNormalization.isSet()) {
			normalizeInfo = new NormalizeInfo[featureValuesArraySize];
			for(int i=0; i < normalizeInfo.length; i++){
				normalizeInfo[i] = new NormalizeInfo();
			}
		}

//  ====Different learning rate trackers ===========================================
		//set the learning rate
		//the amount the weights are adjusted based on loss (ie errors)
		//dictates how much to change the model in response to errors
		//sometimes called the step size
//		MultiFactorTracker learningRateTracker = new MultiFactorTracker.Builder()
//				.setBaseValue(0.03f)
//				.setSteps(new int [] {1})
//				.optFactor(1.0f)
//				.build();

//		LinearTracker learningRateTracker = new LinearTracker.Builder()
//				.setBaseValue(0.03f)
//				.optMaxUpdates(1)
////				.optMinValue(0.02f)
////				.optMaxValue(0.03f)
//				.optSlope(0.01f)
//				.build();

//		FactorTracker learningRateTracker = new FactorTracker.Builder()
//				.setBaseValue(0.03f)
//				.optMaxUpdates(1)
////				.optMinValue(0.029f)
//				.setFactor(1.0f)
//				.build();

		//set optimization technique, Stochastic Gradient Descent (SGD)
		//makes small adjustments to the network configuration to decrease errors
		//minimizes loss (i.e. errors) to produce better and faster results
//		Optimizer optimizer =
//				Optimizer.sgd()
////						.setRescaleGrad(1.0f / BATCH_SIZE)
//						.setLearningRateTracker(learningRateTracker)
////						.optMomentum(0.9f)
////						.optWeightDecays(0.001f)
//						.optClipGrad(1f)
//						.build();
//  ====================================================================

//		Tracker learningRateTracker = Tracker.fixed(0.03f);

		try {
			nnmodel = Model.newInstance("mlp", Device.cpu());
			// Construct neural network and set it in the block
			Block block = new Mlp(featureValuesArraySize, inst.numClasses(), new int[] {1024});
//		Block block = new Mlp(featureValuesArraySize, inst.numClasses(), new int[] {2});
			nnmodel.setBlock(block);

			Tracker learningRateTracker = Tracker.fixed((float) this.learningRateOption.getValue());
			Optimizer optimizer;

			switch(this.optimizerTypeOption.getChosenIndex()) {
				case MLP.OPTIMIZER_SGD:
					optimizer = Optimizer.sgd().setLearningRateTracker(learningRateTracker).build();
					break;
				case MLP.OPTIMIZER_RMSPROP:
					optimizer = Optimizer.rmsprop().optLearningRateTracker(learningRateTracker).build();
					break;
				case MLP.OPTIMIZER_ADAGRAD:
					optimizer = Optimizer.adagrad().optLearningRateTracker(learningRateTracker).build();
					break;
				case MLP.OPTIMIZER_ADAM:
					optimizer = Optimizer.adam().optLearningRateTracker(learningRateTracker).build();
					break;
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