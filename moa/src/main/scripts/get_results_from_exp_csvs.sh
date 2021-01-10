if [ $# -lt 2 ]; then
  echo "Usage: $0 <dataset_dir> <results_csv_dir> <file_search_pattern> <exclude_file_search_pattern>"
  echo "e.g:   $0 /Scratch/ng98/datasets/NEW/unzipped/ /Scratch/ng98/JavaSetup1/resultsNN/Exp8/ '*.txt' 'NN_loss'"
  exit 1
fi

DATASET_DIR=$1"/"
RESULTS_DIR=$2"/"
if [ $# -gt 2 ]; then
  s_pattern=$3
else
  s_pattern='*.csv'
fi

if [ $# -gt 3 ]; then
  exclude_s_pattern=$4
else
  exclude_s_pattern=' '
fi

if [ $( find  $RESULTS_DIR -iname "$s_pattern" | grep -v "$exclude_s_pattern" | wc -l ) -eq 0 ]; then
  echo "No results files to check."
  exit 0
fi
if [ $( find  $RESULTS_DIR -iname "$s_pattern" | grep -v "$exclude_s_pattern" | sed 's/ /\\ /g'  | xargs ls -t | tail -n 1 |xargs grep -c 'Wall Time (Actual Time)' ) -eq 1 ]; then
  p_pattern='{printf "%30s, %40s, %40s, %40s, %40s\n", $1,$2,$6,$7,$9}'
else
  p_pattern='{printf "%30s, %40s, %40s, %40s, %40s\n", $1,$2,$5,$6,$8}'
fi

find  $RESULTS_DIR -iname "$s_pattern" | grep -v "$exclude_s_pattern" | sed 's/ /\\ /g'  | xargs ls -t | tail -n 1 |xargs head -n 1 |awk -F ',' "$p_pattern"
find  $RESULTS_DIR -iname "$s_pattern" | grep -v "$exclude_s_pattern" | sed 's/ /\\ /g'  | xargs awk '{print $0}' |grep  'Learner' -v |sed "s#ArffFileStream -f ##g" |sed "s#${DATASET_DIR}##g" |sed "s#/\/#\\\#g" |sed 's/.arff//g' | awk -F ',' "$p_pattern"
