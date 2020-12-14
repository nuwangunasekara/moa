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

import java.lang.reflect.Array;
import java.util.Arrays;


public class MLP extends AbstractClassifier implements MultiClassClassifier, Regressor {

    private static final long serialVersionUID = 1L;

	public static final int OPTIMIZER_SGD = 0;
	public static final int OPTIMIZER_RMSPROP = 1;
	public static final int OPTIMIZER_ADAGRAD = 2;
	public static final int OPTIMIZER_ADAM = 3;

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


    @Override
    public String getPurposeString() {
        return "NN: special.";
    }

    public ADWIN estimator = new ADWIN();


	protected Model nnmodel = null;
	protected Trainer trainer = null;
	protected double classVotesInit[];
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

    @Override
    public void trainOnInstanceImpl(Instance inst) {
		if (nnmodel == null){
			initializeNetwork(inst);
		}
		float[] featureValues = new float [featureValuesArraySize];
		int totalOneHotEncodedSize = 0;
		int totalOneHotEncodedInstances = 0;
		for(int i=0; i < inst.numInputAttributes(); i++){
			if (useOneHotEncode.isSet() && inst.attribute(i).isNominal()){
				featureValues[ i + totalOneHotEncodedSize - totalOneHotEncodedInstances + (int)inst.value(i)] = 1.0f;
				totalOneHotEncodedSize += inst.attribute(i).numValues();
				totalOneHotEncodedInstances ++;
			}else
			{
				featureValues[ i + totalOneHotEncodedSize - totalOneHotEncodedInstances] = (float) inst.value(i);
			}
		}
		double class_value[] = {inst.classValue()};
		NDList d = new NDList(trainer.getManager().create(featureValues));
		NDList l = new NDList(trainer.getManager().create(class_value));
//		GradientCollector collector = trainer.newGradientCollector();
		try (GradientCollector collector = trainer.newGradientCollector()) {
			NDList preds = trainer.forward(d, l);
			NDArray lossValue = trainer.getLoss().evaluate(l, preds);
			if (lossValue.getFloat() == 0.0f){
				System.out.println("Zero loss");
//				lossValue.add(0.0001f);
			}
			//			print weights
			System.out.println(nnmodel.getBlock().getChildren().get("02Linear").getParameters().get("weight").getArray());
			collector.backward(lossValue);
//			System.out.println(nnmodel.getBlock().getChildren().get("02Linear").getParameters().get("weight").getArray());
			this.estimator.setInput(lossValue.getFloat());
			lossValue.close();
		}catch (Exception e) {
			System.err.println(e);
			e.printStackTrace();
		}
		trainer.step(); // enforce the calculated weights
		d.close();
		l.close();
    }

	@Override
    public double[] getVotesForInstance(Instance inst) {
		if (nnmodel == null){
			initializeNetwork(inst);
		}
		double v [] = classVotesInit.clone();
		float[] featureValues = new float [featureValuesArraySize];
		int totalOneHotEncodedSize = 0;
		int totalOneHotEncodedInstances = 0;
		for(int i=0; i < inst.numInputAttributes(); i++){
			if (useOneHotEncode.isSet() && inst.attribute(i).isNominal()){
				featureValues[ i + totalOneHotEncodedSize - totalOneHotEncodedInstances + (int)inst.value(i)] = 1.0f;
				totalOneHotEncodedSize += inst.attribute(i).numValues();
				totalOneHotEncodedInstances ++;
			}else
			{
				featureValues[ i + totalOneHotEncodedSize - totalOneHotEncodedInstances] = (float) inst.value(i);
			}
		}

		NDList d = new NDList(trainer.getManager().create(featureValues));
		NDList preds = trainer.evaluate(d);

		for(int i=0;i<inst.numClasses();i++){
			v[i] = (double)preds.get(0).toFloatArray()[i];
		}

		preds.close();
		d.close();

		return v;
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

    private void setFeatureValuesArray(Instance inst, float[] featureValues){
		int totalOneHotEncodedSize = 0;
		int totalOneHotEncodedInstances = 0;
		for(int i=0; i < inst.numInputAttributes(); i++){
			if (inst.attribute(i).isNominal()){
				featureValues[ i + totalOneHotEncodedSize - totalOneHotEncodedInstances + (int)inst.value(i)] = 1.0f;
				totalOneHotEncodedSize += inst.attribute(i).numValues();
				totalOneHotEncodedInstances ++;
			}else
			{
				featureValues[ i + totalOneHotEncodedSize - totalOneHotEncodedInstances] = (float) inst.value(i);
			}
		}
	}

	private void initializeNetwork(Instance inst) {
		classVotesInit = new double [inst.numClasses()];
		for(int i=0;i<inst.numClasses();i++){
			classVotesInit[i]=0.0f;
		}

		int totalOneHotEncodedSize = 0;
		int totalOneHotEncodedInstances = 0;
		for(int i=0; i < inst.numInputAttributes(); i++){
			if (inst.attribute(i).isNominal()){
				totalOneHotEncodedSize += inst.attribute(i).numValues();
				totalOneHotEncodedInstances ++;
			}
		}
		featureValuesArraySize = inst.numInputAttributes() + totalOneHotEncodedSize - totalOneHotEncodedInstances;

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