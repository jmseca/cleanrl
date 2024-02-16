#!/bin/bash

log_file="logautofile.log"
names_file="model_names.txt"

truncate -s 0 $log_file

for ((i=0; i<=10; i++)); do
    file="toRun/dqn_atari0${i}.py"

    read model_name

    echo $model_name > "model_names.txt"

    echo "About to Run $file" >> "$log_file"
    python3 "$file" < "model_names.txt"
    echo "FINISHED $file" >> "$log_file"

    read model_descr
    #if cp -r "../runs/${model_name}/" toSend; then
    echo "$model_descr" >> "runs/${model_name}/specs.txt"
    echo "SPECS written ${model_name}" >> "$log_file"

    echo "" >> "$log_file"
    echo "" >> "$log_file"
    #else
    #    echo "Error: Failed to copy ${model_name}" >> "$log_file"
    #fi

done
