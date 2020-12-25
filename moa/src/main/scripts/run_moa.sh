#trap "kill 0" EXIT

if [ $# -lt 2 ]; then
  echo "Usage: $0 <dataset_dir> <out_csv_dir>"
  echo "e.g:   $0 ~/Desktop/datasets/NEW/unzipped/ ~/Desktop/results"
  exit 1
fi

dataset_dir=$1
out_csv_dir=$2

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
REPO=$BASEDIR/../../target/classes
MAVEN_REPO="$(realpath ~)/.m2/repository"
JAR_PATHS="$(for j in $(find $MAVEN_REPO -name '*.jar');do printf '%s:' $j; done)"
CLASSPATH="$JAR_PATHS$REPO/"
JAVA_AGENT_PATH="$(find $MAVEN_REPO -name 'sizeofag-1.0.4.jar')"


JCMD=java
case $(uname)  in

  Darwin)
    if [ -f "$(/usr/libexec/java_home -v 1.8.0_271)/bin/java" ]
    then
      JCMD="$(/usr/libexec/java_home -v 1.8.0_271)/bin/java"
    fi
    echo "MacOS"
    ;;

  Linux)
    if [ -f "$JAVA_HOME/bin/java" ]
    then
      JCMD="$JAVA_HOME/bin/java"
    fi
    echo "Linux"
    ;;

  *)
    JCMD=java
    ;;
esac

# learner='meta.StreamingRandomPatches -s 10'
learner='neuralNetworks.MultiMLP -h -n -t UseThreads'
# learner='lazy.kNN'

log_file="${out_csv_dir}/full.log"

echo "Full results log file = $log_file"


rm -f $log_file

#dataset=(WISDM_ar_v1.1_transformed elecNormNew covtypeNorm kdd99 RBF_f RBF_m spam_corpus LED_g LED_a nomao airlines AGR_a AGR_g)
dataset=(WISDM_ar_v1.1_transformed elecNormNew)
re_run_count=0
task_failed=0

for (( i=0; i<${#dataset[@]}; i++ ))
do
  task_failed=0
  echo "Dataset = ${dataset[$i]}"
  in_file="${dataset_dir}/${dataset[$i]}.arff"
  out_file="${out_csv_dir}/${dataset[$i]}.csv"
  tmp_log_file="${out_csv_dir}/${dataset[$i]}.log"

  if [ -f $out_file ]; then
    echo "$out_file already available"
  fi

  rm -f tmp_log_file

  exp_cmd="moa.DoTask \"EvaluateInterleavedTestThenTrain -l ($learner) -s (ArffFileStream -f $in_file) -i 10000000 -f 10000000 -q 10000000 -d $out_file\" &>$tmp_log_file &"
  echo "\n$exp_cmd\n"
  echo "\n$exp_cmd\n" > $tmp_log_file
"$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx8g -Xms50m -Xss1g \
  -javaagent:"$JAVA_AGENT_PATH" \
  moa.DoTask "EvaluateInterleavedTestThenTrain -l ($learner) -s (ArffFileStream -f $in_file) -i 10000000 -f 10000000 -q 10000000 -d $out_file" &>$tmp_log_file &

  if [ -z $! ]; then
    task_failed=1
  else
    PID=$!
    echo "PID=$PID : $exp_cmd"
    sleep 5

    while [ $(grep -m 1 -c 'Task completed' $tmp_log_file ) -lt 1 ];
    do
      sleep 60
      if ! ps -p $PID &>/dev/null;
      then
        task_failed=1
        break
      esle
        echo -ne "Waiting for exp with $PID to finish\r"
      fi
    done

    cat $tmp_log_file >> $log_file
    kill $PID
  fi


  if [ $task_failed -eq 0 ]; then
    re_run_count=0
  else
    echo "Task=$i dataset=${dataset[$i]} PID=$PID ) failed."
    if [ $re_run_count -lt 2 ]; then
      re_run_count=$((re_run_count+1))
      echo "Re-running it for the $re_run_count time."
      i=$((i-1))
    else
      echo "Not Re-running it for the $((re_run_count+1)) time."
      re_run_count=0
    fi
  fi

done