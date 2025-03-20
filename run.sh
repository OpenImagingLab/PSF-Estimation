#!/bin/bash

start_time=$(date +%s)
python main.py
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$((runtime % 60))

echo "Program runtime: ${hours} hours ${minutes} minutes ${seconds} seconds"

