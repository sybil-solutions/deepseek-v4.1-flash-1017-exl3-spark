#!/bin/bash
# Arm T finish on one H200: setup -> dense K4 (wo_a K5) + head K4 -> panel KLD ref + quant -> HF upload.
# Writes /data/STAGE (current step) and /data/FINISH_OK or /data/FINISH_FAIL; the Mac watcher terminates the pod on either.
set -o pipefail
S() { echo "$(date -u +%FT%TZ) $*" | tee -a /data/STAGE.log; echo "$*" > /data/STAGE; }
fail() { S "FAIL $*"; touch /data/FINISH_FAIL; up_logs; exit 1; }
R=0xSero/DeepSeek-V4.1-Flash-1017-EXL3-wip; M=/data/DeepSeek-V4.1-Flash-1017; Q=/data/recipe/scripts/quantize
export PATH=$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 DSV41_CUDA_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1
up_logs() { ~/venv/bin/hf upload $R /data/logs finish/logs --commit-message "finish logs" >/dev/null 2>&1; }
mkdir -p /data/logs
S setup-venv
. ~/venv/bin/activate || fail venv
S download
( hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir $M > /data/logs/dl_model.log 2>&1 ) &
P1=$!
( for pat in "p1_T/*" "inputs/cal_1017_v2/*" "inputs/plan_T_270.json"; do hf download $R --include "$pat" --local-dir /data/wip || exit 1; done > /data/logs/dl_wip.log 2>&1 ) &
P2=$!
S setup-cuda
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update >/dev/null 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y cuda-compat-12-8 cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8 cuda-cccl-12-8 libcublas-dev-12-8 cuda-profiler-api-12-8 > /data/logs/apt.log 2>&1 || fail apt
uv pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 > /data/logs/pip.log 2>&1 || fail torch
uv pip install -q "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.0/exllamav3-1.6.0+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl" tilelang==0.1.12 transformers tokenizers "safetensors>=0.7.0" numpy sympy Pillow tqdm >> /data/logs/pip.log 2>&1 || fail pip
python -c "import torch,exllamav3,tilelang;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))" >> /data/logs/pip.log 2>&1 || fail cuda-check
[ -d /data/exllamav3 ] || { git clone -q https://github.com/turboderp-org/exllamav3.git /data/exllamav3 && git -C /data/exllamav3 checkout -q 151539c7; } || fail exl3-clone
wait $P1 || fail dl-model; wait $P2 || fail dl-wip
S verify
python -I /data/verify.py > /data/logs/verify.log 2>&1; grep -q "bad \[\]" /data/logs/verify.log || fail verify
n=$(ls /data/wip/p1_T/layer_*.part0.safetensors | wc -l); [ "$n" = 40 ] || fail "p1 layers $n"
S panel
( cd $Q/caldata && python build.py $M /data/recipe/data/caldata/conversations.jsonl /data/exllamav3/exllamav3/conversion/standard_cal_data /data/panel_build > /data/logs/panel.log 2>&1 ) || fail panel
sha256sum /data/panel_build/panel.safetensors >> /data/logs/panel.log
grep -q 39e0db2f0b58aa432a512aed7b425e3feb1b424c9e02cc315be39bc9b7b71cf2 /data/logs/panel.log || fail "panel hash mismatch"
cd $Q
S dense
python -u p4_dense.py $M /data/wip/inputs/cal_1017_v2/cal.safetensors /data/wip/p1_T /data/dense_T --k 4 --k-map attn.wo_a=5 --head 4 > /data/logs/dense.log 2>&1 || fail dense
python p4_dense.py --index /data/dense_T >> /data/logs/dense.log 2>&1 || fail dense-index
ls /data/dense_T/dense_head.safetensors >/dev/null || fail "no head"
hf upload $R /data/dense_T dense_T --commit-message "dense K4 wo_a K5 head K4" > /dev/null 2>&1 || fail upload-dense
S kld-ref
python -u p4_kld.py $M /data/panel_build/panel.safetensors /data/panel_build/panel_index.json /data/kld_ref --mode ref > /data/logs/kld_ref.log 2>&1 || fail kld-ref
S kld-quant
python -u p4_kld.py $M /data/panel_build/panel.safetensors /data/panel_build/panel_index.json /data/kld_ref --mode quant --tensors /data/wip/p1_T --dense /data/dense_T > /data/logs/kld_quant.log 2>&1 || fail kld-quant
S upload
hf upload $R /data/kld_ref kld_T --exclude "*.pt" --commit-message "panel KLD arm T" > /dev/null 2>&1 || fail upload-kld
up_logs
S done; touch /data/FINISH_OK
