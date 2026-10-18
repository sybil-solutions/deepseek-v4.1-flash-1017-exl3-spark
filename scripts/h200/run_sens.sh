. ~/venv/bin/activate; export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH DSV41_CUDA_MEM_FRACTION=0.95
cd /data/recipe/scripts/quantize
date; time python p4_sensitivity.py /data/DeepSeek-V4.1-Flash-1017 /data/cal_1017_v2/cal.safetensors /data/sens_v2.json || exit 1
echo SENS_DONE
