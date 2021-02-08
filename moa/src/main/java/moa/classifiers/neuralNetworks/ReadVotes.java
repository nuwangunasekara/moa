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


public class ReadVotes extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

	class Ensemble {
		String name;
		private BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();
	}

	private HashMap<String, Ensemble> ensemble = null;

	private long instanceID = 0;

	private static final int tokenID_id = 0;
	private static final int tokenID_modelName = 1;
	private static final int tokenID_estimateLoss = 2;
	private static final int tokenID_classValue = 3;
	private static final int tokenID_classIndex = 4;
	private static final int tokenID_votesStart = 5;

	private BufferedReader inputReader = null;
	private BasicClassificationPerformanceEvaluator performanceByMinLoss = new BasicClassificationPerformanceEvaluator();
	private BasicClassificationPerformanceEvaluator performanceByMajorityVote = new BasicClassificationPerformanceEvaluator();
	private double [] accumulatedVotesForMajorityVote = {};


	public FileOption votesFileOption = new FileOption("votesFile", 'f',
			"Votes File.", null, "csv", true);

	protected static final int USE_MIN_LOSS = 0;
	protected static final int USE_MAJORITY_VOTE = 1;
	public MultiChoiceOption votesSelectionCriteria = new MultiChoiceOption("votesSelectionCriteria", 'c',
			"The votes selection criteria",
			new String[]{"min_loss", "majority_vote"},
			new String[]{"min_loss", "majority_vote"}, USE_MIN_LOSS);

	public IntOption numberOfLearners = new IntOption(
			"numberOfLearners",
			'e',
			"Number of learners",
			8, 1, Integer.MAX_VALUE);



	static class InstanceVotes {
		long id = 0;
		String name = null;
		double loss = 0.0 ;
		double classValue = 0.0;
		int classIndex = -1;
		double [] votes;

		public InstanceVotes() {
		}

		public InstanceVotes(InstanceVotes from) {
			this.id = from.id;
			this.name = from.name;
			this.loss = from.loss;
			this.classValue = from.classValue;
			this.classIndex = from.classIndex;
			this.votes = from.votes;
		}

		@Override
		public String toString() {
			return "InstanceVotes{" +
					"id=" + id +
					", name='" + name + '\'' +
					", loss=" + loss +
					", classValue=" + classValue +
					", classIndex=" + classIndex +
					", votes=" + votes +
					'}';
		}
	}

	private InstanceVotes readVotesFromLine(String inputFileLine){
		InstanceVotes tmpInstanceVotes = new InstanceVotes();
		StringTokenizer inputFileTokenizer = new StringTokenizer(inputFileLine, ",[]");
		int tokenId = 0;
		ArrayList<Double> tmpVotesArrayList = new ArrayList<Double>();
		Double [] tmpVotesD = {};
		while (inputFileTokenizer.hasMoreTokens())
		{
			String tokenStr = inputFileTokenizer.nextToken().trim();
			switch (tokenId){
				case tokenID_id:
					tmpInstanceVotes.id = Long.parseLong(tokenStr);
					break;
				case tokenID_modelName:
					tmpInstanceVotes.name = tokenStr;
					break;
				case tokenID_estimateLoss:
					tmpInstanceVotes.loss = Double.parseDouble(tokenStr);
					break;
				case tokenID_classValue:
					tmpInstanceVotes.classValue = Double.parseDouble(tokenStr);
					break;
				case tokenID_classIndex:
					tmpInstanceVotes.classIndex = Integer.parseInt(tokenStr);
					break;
				case tokenID_votesStart:
				default:
					tmpVotesArrayList.add(Double.parseDouble(tokenStr));
					break;
			}
			tokenId++;
		}
		tmpVotesD = tmpVotesArrayList.toArray(tmpVotesD);
		tmpInstanceVotes.votes = Stream.of(tmpVotesD).mapToDouble(Double::doubleValue).toArray();
		return tmpInstanceVotes;
	}

	public ArrayList<InstanceVotes> readVotesForInstanceID(Instance inst){
		long lineCount = 0;
		ArrayList<InstanceVotes> votesForInstanceID = new ArrayList<InstanceVotes>();

		try {
			String inputFileLine = inputReader.readLine();
			while (inputFileLine != null) {
//                System.out.println(inputFileLine);
				InstanceVotes tmpInstVotesForSingleLearner = readVotesFromLine(inputFileLine);
				votesForInstanceID.add(new InstanceVotes(tmpInstVotesForSingleLearner));
				lineCount++;
				if (instanceID == 1){ // init each ensemble
					ensemble.put(tmpInstVotesForSingleLearner.name, new Ensemble());
				}
				ensemble.get(tmpInstVotesForSingleLearner.name).evaluator.addResult(new InstanceExample(inst),tmpInstVotesForSingleLearner.votes);
				if (lineCount == 1){
					accumulatedVotesForMajorityVote = new double [tmpInstVotesForSingleLearner.votes.length];
				}
				for (int i = 0; i < accumulatedVotesForMajorityVote.length; i++){
					double acc = ensemble.get(tmpInstVotesForSingleLearner.name).evaluator.getPerformanceMeasurements()[1].getValue();
					accumulatedVotesForMajorityVote[i] += tmpInstVotesForSingleLearner.votes[i] * acc;
				}
				if ( lineCount == numberOfLearners.getValue() ) {
					break;
				}
				inputFileLine = inputReader.readLine();
			}

			if (inputFileLine == null){
				inputReader.close();
			}
		} catch (Exception e) {
			System.out.println(e.getMessage());
			System.exit(1);
		}
		return votesForInstanceID;
	}

	private void initReader(){
		try {
			inputReader = new BufferedReader(new FileReader(votesFileOption.getFile()));
			//read out the header line
			String inputFileLine = inputReader.readLine();
		} catch (Exception e) {
			System.out.println(e.getMessage());
			System.exit(1);
		}
		ensemble = new HashMap<String, Ensemble>();
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		double [] votesByMinLoss;

		int chosenIndex = 0;

		if ( inputReader == null ){
			initReader();
		}

		instanceID++;

		ArrayList<InstanceVotes> votesForID = readVotesForInstanceID(inst);
		if ((instanceID != votesForID.get(0).id) || (votesForID.get(0).id != votesForID.get(votesForID.size()-1).id)){
			System.out.println("Something wrong: " + instanceID + votesForID);
		}

		double minEstimation = Double.MAX_VALUE;
		for (int i = 0 ; i < votesForID.size() ; i++) {
			if (votesForID.get(i).loss < minEstimation){
				minEstimation = votesForID.get(i).loss;
				chosenIndex = i;
			}
		}
		votesByMinLoss = votesForID.get(chosenIndex).votes;
		performanceByMinLoss.addResult(new InstanceExample(inst), votesByMinLoss);
		performanceByMajorityVote.addResult(new InstanceExample(inst), accumulatedVotesForMajorityVote);

		if (votesSelectionCriteria.getChosenIndex() == USE_MIN_LOSS){
			return votesByMinLoss;
		}else{
			return accumulatedVotesForMajorityVote;
		}
	}

	@Override
	public String getPurposeString() {
		return "Read votes from a file";
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
    protected Measurement[] getModelMeasurementsImpl() {
//		System.out.println(performanceByMinLoss.getPerformanceMeasurements()[1].getValue());
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) { }

    public boolean isRandomizable() {
        return false;
    }
}