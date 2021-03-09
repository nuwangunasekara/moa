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


import com.github.javacliparser.FileOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.evaluation.BasicClassificationPerformanceEvaluator;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.stream.Stream;


public class DatasetStats extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    class DatasetClass{
    	String name = null;
		int classIndex = -1;
		long count = 0;
		double percentage = 0.0;

		public DatasetClass(String name, int classIndex, long count) {
			this.name = name;
			this.classIndex = classIndex;
			this.count = count;
		}

		@Override
		public String toString() {
			return "DatasetClass{" +
					"name='" + name + '\'' +
					", classIndex=" + classIndex +
					", count=" + count +
					", percentage=" + percentage +
					'}';
		}
	}

	private HashMap<Double, DatasetClass> datasetClasses = null;
    private long instanceID = 0;
    private static double [] dummyVotes;

    private void initClasses(Instance inst){
    	dummyVotes = new double[inst.numClasses()];
		datasetClasses = new HashMap<Double, DatasetClass>(inst.numClasses());
	}



	@Override
	public double[] getVotesForInstance(Instance inst) {

		if ( datasetClasses == null ){
			initClasses(inst);
		}
		instanceID++;
		if (datasetClasses.containsKey(inst.classValue())) {
			datasetClasses.get(inst.classValue()).count++;
		}else {
			datasetClasses.put(inst.classValue(), new DatasetClass(String.valueOf(inst.classValue()), inst.classIndex(), 0));
		}
		return dummyVotes;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
    	long totalInstances = 0;
		ArrayList<DatasetClass> list = new ArrayList<>();
		for (Map.Entry<Double, DatasetClass> e : datasetClasses.entrySet()){
			System.out.println("Key: " + e.getKey()
					+ " Value: " + e.getValue());
			totalInstances += e.getValue().count;
			list.add(e.getValue());
		}
		Collections.sort(list, new Comparator<DatasetClass>() {
			public int compare(DatasetClass c1, DatasetClass c2) {
				return Long.compare(c1.count, c2.count);
			}
		});
		for (DatasetClass datasetClass : list) {
			datasetClass.percentage = datasetClass.count/(double) totalInstances *100;
			System.out.println(datasetClass);
		}
		return null;
	}

	@Override
	public String getPurposeString() {
		return "Print dataset info";
	}

	@Override
	public void setModelContext(InstancesHeader context) { }

	@Override
	public void trainOnInstanceImpl(Instance inst) { }

    @Override
    public void resetLearningImpl() { }

	@Override
	public ImmutableCapabilities defineImmutableCapabilities() {
		return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
	}

    @Override
    public void getModelDescription(StringBuilder out, int indent) { }

    public boolean isRandomizable() {
        return false;
    }
}