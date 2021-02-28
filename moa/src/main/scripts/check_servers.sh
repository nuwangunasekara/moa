#!/bin/bash

command='echo -e "HOST=$(hostname).$(dnsdomainname) \t CPU_COUNT=$(( $(lscpu -e=CPU |tail -n1) + 1 )) \t TOP=$(ps h -Ao user,comm,pcpu --sort=-pcpu |head -n 1) \t NO_OF_USERS=$(who |wc -l) \t MEM=$(free -h | grep -i 'Mem:')"'

gpu_query_gpu_command='nvidia-smi --query-gpu=gpu_name,pci.bus_id,utilization.gpu,utilization.memory,memory.total,memory.free --format=csv'
gpu_get_driver='nvidia-smi'
gpu_get_cuda_version='nvcc --version'
gpu_query_compute_apps_command='nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_gpu_memory,process_name --format=csv'
gpu_query_accounted_apps_command='nvidia-smi --query-accounted-apps=gpu_bus_id,pid,gpu_utilization,mem_utilization,max_memory_usage  --format=csv'
ssd_usage_command='if [ -d /Scratch/ng98/ ]; then echo "SSD usage: $(du -h -d 0 /Scratch/ng98/)"; else  echo "SSD usage: N/A"; fi'

domain_name='cms.waikato.ac.nz'
show_gpu="do_not_show_gpu"
if [ $# -gt 1 ]; then
  if [ "$2" == "show_gpu" ]; then
    show_gpu="show_gpu"
  fi
fi

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
      ${ssh_command}  "$gpu_get_driver"
      echo "........................................................................................................................"
      ${ssh_command}  "$gpu_get_cuda_version"
      echo "........................................................................................................................"
      ${ssh_command}  "$gpu_query_gpu_command" | awk -F ',' '{printf "%20s %20s %20s %20s %20s %20s\n", $1,$2,$3,$4,$5,$6}'
      echo "........................................................................................................................"
      ${ssh_command}  "$gpu_query_compute_apps_command" | awk -F ',' '{printf "%20s %20s %20s %20s %40s\n", " ",$1,$2,$3,$4}'
      echo "........................................................................................................................"
      ${ssh_command}  "$gpu_query_accounted_apps_command" | awk -F ',' '{printf "%20s %20s %20s %20s %20s %20s\n", " ",$1,$2,$3,$4,$5}'
      echo "........................................................................................................................"
      ${ssh_command}  "$ssd_usage_command"
      echo "........................................................................................................................"
    fi
  fi
}

if [ $# -gt 0 ]; then
  if [ "$1" == "local" ]; then
    print_server_info "$1" $show_gpu
    exit 0
  else
    print_server_info "$1" $show_gpu
    exit 0
  fi
fi

echo -e "\nServers with more than 1 GPU with NVLink, all with 64BB CPU RAM:\n"
for i in duet quartet quatern;
do
  print_server_info "${i}.${domain_name}" $show_gpu
done

echo -e "\nServers with NO GPU, all with 64GB CPU RAM:\n"
for i in $(seq 11 12);
do
  print_server_info "ml-${i}.${domain_name}" $show_gpu
done

echo -e "\nServers with up to 1 GPU, all with 64 CPU RAM:\n"
for i in $(seq 13 25);
do
  print_server_info "ml-${i}.${domain_name}" $show_gpu
done

echo -e "\nOld ML servers, somewhat slower, NO GPUs, 16GB of RAM:\n"
for i in 20 21 24 25 27 28 29 31 32 33 34 35;
do
  print_server_info "ml64-${i}.${domain_name}"
done
