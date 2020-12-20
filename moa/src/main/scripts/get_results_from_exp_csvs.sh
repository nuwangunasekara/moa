RESULTS_DIR='/Scratch/ng98/JavaSetup1/resultsNN/Exp2/Results/'

find  $RESULTS_DIR -iname '*.txt' |sed 's/ /\\ /g' |xargs awk '{print $0}'|grep  'Learner' -v |sed "s/ArffFileStream -f ${RESULTS_DIR//\//\\\/}" |sed 's/.arff//g' | awk -F ',' '{printf "%30s, %30s, %40s, %40s, %40s\n", $1,$2,$5,$6,$8}'
