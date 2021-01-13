#!/bin/bash

command='echo -e "HOST=$(hostname).$(dnsdomainname) \t CPU_COUNT=$(( $(lscpu -e=CPU |tail -n1) + 1 )) \t TOP=$(ps h -Ao user,comm,pcpu --sort=-pcpu |head -n 1) \t NO_OF_USERS=$(who |wc -l) \t MEM=$(free -h | grep -i 'Mem:')"'

gpu_query_gpu_command='nvidia-smi --query-gpu=gpu_name,pci.bus_id,utilization.gpu,utilization.memory,memory.total,memory.free --format=csv'
gpu_query_compute_apps_command='nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_gpu_memory,process_name --format=csv'
gpu_query_accounted_apps_command='nvidia-smi --query-accounted-apps=gpu_bus_id,pid,gpu_utilization,mem_utilization,max_memory_usage  --format=csv'

if [ $# -gt 0 ]; then
  if [ "$1" == "local" ]; then
    print_server_info "$1" show_gpu
    exit 0
  fi
fi

domain_name='cms.waikato.ac.nz'

print_server_info()
{
  server_name="$1"
  
  if [ "$1" == "local" ]; then
    ssh_command=""
  else
    ssh_command="ssh -l ng98 -t ${server_name}"
  fi

  echo "========================================================================================================================"
  ${ssh_command}  "$command" | grep "HOST="
  if [ $# -gt 1 ]; then
    if [ "$2" == "show_gpu" ]; then
      echo "------------------------------------------------------------------------------------------------------------------------"
      ${ssh_command}  "$gpu_query_gpu_command" | awk -F ',' '{printf "%20s %20s %20s %20s %20s %20s\n", $1,$2,$3,$4,$5,$6}'
      echo "........................................................................................................................"
      ${ssh_command}  "$gpu_query_compute_apps_command" | awk -F ',' '{printf "%20s %20s %20s %20s %40s\n", " ",$1,$2,$3,$4}'
      echo "........................................................................................................................"
      ${ssh_command}  "$gpu_query_accounted_apps_command" | awk -F ',' '{printf "%20s %20s %20s %20s %20s %20s\n", " ",$1,$2,$3,$4,$5}'
      echo "........................................................................................................................"
    fi
  fi
}

echo -e "\nServers with up to one GPU, all with 64 CPU RAM:\n"
for i in duet quartet quatern;
do
  print_server_info "${i}.${domain_name}" show_gpu
done


echo -e "\nServers with up to one GPU, all with 64 CPU RAM:\n"
for i in $(seq 11 25);
do
  print_server_info "ml-${i}.${domain_name}" show_gpu
done

echo -e "\nOld ML servers, somewhat slower, no GPUs, 16GB of RAM:\n"
for i in 21 31 33 35;
do
  print_server_info "ml64-${i}.${domain_name}"
done
