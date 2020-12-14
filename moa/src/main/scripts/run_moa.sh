#java -cp classes/ -javaagent:classes/sizeofag-1.0.4.jar -Xmx50g -Xms50m -Xss1g moa.DoTask "EvaluateInterleavedTestThenTrain -l (lazy.kNN) -s  generators.WaveformGenerator -i 100000000 -f 1000000"#moa_path='/Scratch/ng98/JavaSetup1/moa'
if [ $# -lt 4 ]; then
  echo "Usage: $0 <moa_path_classes_path> <maven_repo> <dataset_dir> <out_csv_dir>"
  echo "e.g:   $0 ~/Desktop/moa-fork/moa/target/classes/ ~/.m2/repository/ ~/Desktop/datasets/NEW/unzipped/ ~/Desktop/results"
  exit 1
fi
moa_path_classes_path=$1
maven_repo=$2
dataset_dir=$3
out_csv_dir=$4

jar_paths="$(for j in $(find $maven_repo -name '*.jar');do printf '%s:' $j; done)"
class_paths="$jar_pathsS$moa_path_classes_path/"
java_agent_path="$(find $maven_repo -name 'sizeofag-1.0.4.jar')"

java_cmd="java -cp $class_paths"
javaagent_cmd="-javaagent:$java_agent_path"


for dataset in elecNormNew;
#for dataset in internet_ads_Normalized_Randomized WISDM_ar_v1.1_transformed nomao_Normalized elecNormNew airlines_Normalized covtypeNorm kdd99 AGR_a_Normalized AGR_g_Normalized RBF_f RBF_m LED_a LED_g spam_corpus;
do
  in_file="${dataset_dir}/${dataset}.arff"
  out_file="${out_csv_dir}/${dataset}.csv"

  if [ -f $out_file ]; then
    echo "$out_file already available"
  fi



#  learner='(meta.StreamingRandomPatches -s 10)'
  learner='neuralNetworks.MLP'
#  learner='lazy.kNN'

  echo "$java_cmd $javaagent_cmd -Xmx50g -Xms50m -Xss1g moa.DoTask \"EvaluateInterleavedTestThenTrain -l ($learner) -s (ArffFileStream -f $in_file) -i 10000000 -f 10000000 -q 10000000 -d $out_file\""
  $java_cmd $javaagent_cmd -Xmx50g -Xms50m -Xss1g moa.DoTask "EvaluateInterleavedTestThenTrain -l ($learner) -s (ArffFileStream -f $in_file) -i 10000000 -f 10000000 -q 10000000 -d $out_file"
done