#trap "kill 0" EXIT
print_usage()
{
  echo "Usage: $0 <dataset_dir> <out_csv_dir> <djl_cache_dir> <local_maven_repo>"
  echo "e.g:   $0 ~/Desktop/datasets/NEW/unzipped/ ~/Desktop/results ~/Desktop/djl.ai/ ~/Desktop/m2_cache/ "
  echo "e.g:   $0 /Scratch/ng98/datasets/NEW/unzipped/ /Scratch/ng98/JavaSetup1/resultsNN/Exp17_test/ /Scratch/ng98/JavaSetup1/djl.ai/ /Scratch/ng98/JavaSetup1/local_m2/"
}

if [ $# -lt 2 ]; then
  print_usage
  exit 1
fi

dataset_dir=$1
out_csv_dir=$2

if [ $# -gt 2 ]; then
  if [ -d "$3" ]; then
    export DJL_CACHE_DIR=$3
  else
    echo "DJL_CACHE_DIR can not be set. Directory $3 is not available."
    print_usage
    exit 1
  fi
fi

MAVEN_REPO="$(realpath ~)/.m2/repository"
if [ $# -gt 3 ]; then
  if [ -d "$4" ]; then
    MAVEN_REPO="$4"
    export MAVEN_OPTS="-Dmaven.repo.local=$4"
  else
    echo "MAVEN_OPTS=-Dmaven.repo.local can not be set. Directory $4 is not available."
    print_usage
    exit 1
  fi
fi

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
REPO=$BASEDIR/../../target/classes
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

#learner='meta.StreamingRandomPatches -s 10'
learner='neuralNetworks.MultiMLP -h -n -t UseThreads'
# learner='lazy.kNN'

log_file="${out_csv_dir}/full.log"

echo "Full results log file = $log_file"


rm -f $log_file

dataset=(spam_corpus WISDM_ar_v1.1_transformed elecNormNew nomao covtypeNorm kdd99 airlines RBF_f RBF_m LED_g LED_a AGR_a AGR_g)
#dataset=(elecNormNew)
max_repeat=4
datasets_to_repeat=(WISDM_ar_v1.1_transformed elecNormNew nomao)
declare -a repeat_exp_count
for i in "${datasets_to_repeat[@]}"
do
    repeat_exp_count+=(${max_repeat})
done

re_run_count=0
task_failed=0

for (( i=0; i<${#dataset[@]}; i++ ))
do
  sleep 60
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
  echo -e "\n$exp_cmd\n"
  echo -e "\n$exp_cmd\n" > $tmp_log_file
time "$JCMD" \
  -classpath "$CLASSPATH" \
  -Xmx16g -Xms50m -Xss1g \
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
    echo "Child processors============================="
    #Store the current Process ID, we don't want to kill the current executing process id
    CURPID=$$

    # This is process id, parameter passed by user
    ppid=$PID

    if [ -z $ppid ] ; then
       echo No PID given.
    fi

    arraycounter=1
    while true
    do
            FORLOOP=FALSE
            # Get all the child process id
            for i in `ps -ef| awk '$3 == '$ppid' { print $2 }'`
            do
                    if [ $i -ne $CURPID ] ; then
                            procid[$arraycounter]=$i
                            arraycounter=`expr $arraycounter + 1`
                            ppid=$i
                            FORLOOP=TRUE
                    fi
            done
            if [ "$FORLOOP" = "FALSE" ] ; then
               arraycounter=`expr $arraycounter - 1`
               ## We want to kill child process id first and then parent id's
               while [ $arraycounter -ne 0 ]
               do
                 echo "killing ${procid[$arraycounter]}"
                 kill -9 "${procid[$arraycounter]}" >/dev/null
                 arraycounter=`expr $arraycounter - 1`
               done
             break
            fi
    done
    echo "Child processors============================="
    kill $PID
  fi

  if [ -f NN_loss.csv ]; then
    mv NN_loss.csv ${dataset[$i]}_NN_loss.csv
  fi

  cat $tmp_log_file >> $log_file

  if [ $task_failed -eq 0 ]; then
    re_run_count=0
    echo "Task=$i dataset=${dataset[$i]} PID=$PID ) was successful."
    for (( j=0; j<${#datasets_to_repeat[@]}; j++ ))
    do
      if [ "${dataset[$i]}" == "${datasets_to_repeat[$j]}" ]; then
        if [ $((${repeat_exp_count[$j]})) -gt 0 ]; then
          repeat_exp_count[$j]=$((${repeat_exp_count[$j]} - 1))
          echo "Repeat ${dataset[$i]} for the $((max_repeat - ${repeat_exp_count[$j]})) time"
          i=$((i-1))
          break
        fi
      fi
    done
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
