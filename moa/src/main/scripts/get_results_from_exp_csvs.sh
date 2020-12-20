RESULTS_DIR='/Scratch/ng98/JavaSetup1/resultsNN/Exp2/Results/'
DATASET_DIR='/Scratch/ng98/datasets/NEW/unzipped/'

find  $RESULTS_DIR -iname '*.txt' |sed 's/ /\\ /g' | xargs awk '{print $0}' |grep  'Learner' -v |sed "s#ArffFileStream -f ${DATASET_DIR//\//\\\/}##" |sed 's/.arff//g' | awk -F ',' '{printf "%30s, %40s, %40s, %40s, %40s\n", $1,$2,$5,$6,$8}'
