results_dir=$1
headers=$(grep --exclude full.log -m 1 -r "MinEstimatedLoss" $results_dir)
echo "stream,"
echo "$headers"
grep --exclude full.log -A 1 -r "MinEstimatedLoss" $results_dir |grep -v "MinEstimatedLoss" |grep -v '\-\-' |sed "s#$results_dir##g"|sed 's#/neuralNetworks.ReadVotes_##g' | sed 's#.log-#,#g'