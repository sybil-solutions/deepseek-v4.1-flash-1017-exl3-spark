. ~/venv/bin/activate; export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH DSV41_CUDA_MEM_FRACTION=0.95
M=/data/DeepSeek-V4.1-Flash-1017; C=/data/cal_1017_v2; cd /data/recipe/scripts/quantize
date; time python p4_pass0.py $M $C/cal.safetensors /data/stats_v2 --index $C/cal_index.json || exit 1
date; time python p4_sensitivity.py $M $C/cal.safetensors /data/sens_v2.json || exit 1
date; echo P0S_DONE
