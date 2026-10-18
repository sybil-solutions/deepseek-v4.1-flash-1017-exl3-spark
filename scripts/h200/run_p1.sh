. ~/venv/bin/activate; export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH DSV41_CUDA_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
cd /data/recipe/scripts/quantize; mkdir -p /data/p1_T
date; time python -u p4_pass1.py /data/DeepSeek-V4.1-Flash-1017 /data/cal_1017_v2/cal.safetensors /data/plan_T_270.json /data/p1_T --layers 0-39
echo P1_EXIT=$?
