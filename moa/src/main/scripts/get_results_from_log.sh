if [ $# -lt 1 ]; then
  echo "Usage: $0 <log_file>"
  exit 1
fi

log_file=$1
print_next=0
while IFS= read -r line
do
  if [ $(echo $line | grep -c -E '[0-9A-Za-z_\.]+arff') -eq 1 ]; then
    echo -e "dataset                           : $(echo $line | grep -o -E '[0-9A-Za-z_\.]+arff' | sed 's/.arff//g')"
  fi

  if [ $print_next -eq 1 ]; then
    echo -e "classifications correct (percent) : $(echo $line | awk -F ',' '{print $5}')"
    echo -e "evaluation time (cpu seconds)     : $(echo $line | awk -F ',' '{print $2}')"
    echo -e "model cost (RAM-Hours)            : $(echo $line | awk -F ',' '{print $3}')"
    echo -e "classified instances              : $(echo $line | awk -F ',' '{print $4}')"
    echo -e "Kappa Statistic (percent)         : $(echo $line | awk -F ',' '{print $6}')"
    echo -e "Kappa Temporal Statistic (percent): $(echo $line | awk -F ',' '{print $7}')"
    echo -e "Kappa M Statistic (percent)       : $(echo $line | awk -F ',' '{print $8}')"
#    echo -e "learning evaluation instances     : $(echo $line | awk -F ',' '{print $1}')"
    echo "========================================================================="
  fi
  if [ $(echo "$line" | grep -c 'learning evaluation') -eq 1 ]; then
    print_next=1
  else
    print_next=0
  fi
done < $log_file