# DeepSeek-V4.1-Flash-1017 EXL3 2.7 bpw for 2x DGX Spark

The full recipe behind [`0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw`](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw):
an EXL3 trellis quantization of `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, made with
[sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) @ `cc59c360` (Apache-2.0) and
served on two NVIDIA DGX Sparks with that project's vLLM stack.

The method, the allocator, the EXL3 encoder harness, the serving patches and the plugin kernels are sudoingX's. This
repo adds what changed for the 1017 checkpoint: a calibration set of native 1017 replies, a fix that makes DeepSeek's
reference implementation run on Hopper (H200), the bit allocation for 1017, and the measured result.

## Result

Panel: sudoingX's published KLD panel, unchanged (199 conversations, 316,944 tokens, 254,846 scored assistant
positions, sha256 `39e0db2f…`). Reference: DeepSeek's own `inference/model.py` on 1017, streamed layer by layer on one
H200. Lower KLD is closer to the native model.

| | this quant (1017) |
|---|---|
| routed-expert bits, memory / per decoded token | 2.70 / 2.90 |
| KLD vs native, mean over positions (nats) | **0.0304** (95 % CI 0.0268-0.0343, bootstrap over the 199 conversations) |
| KLD, the 8 domains weighted equally | 0.0512 |
| KLD p99 | 0.435 |
| top-1 agreement | 94.95 % |
| panel perplexity, native -> quant | 1.6010 -> 1.6231 |

| domain | share of positions | KLD | top-1 |
|---|---|---|---|
| code | 65 % | 0.0182 | 95.83 % |
| math | 4 % | 0.0230 | 94.72 % |
| agentic | 7 % | 0.0277 | 94.96 % |
| long | 6 % | 0.0379 | 95.20 % |
| knowledge | 6 % | 0.0568 | 92.05 % |
| writing | 2 % | 0.0795 | 91.56 % |
| multilingual | 3 % | 0.0827 | 93.35 % |
| chat | 7 % | 0.0835 | 90.68 % |

For context: sudoingX's `DeepSeek-V4.1-Flash-EXL3-2.7bpw` measures 0.0243 / 95.57 % on the same panel at the same
budget, but on the previous checkpoint (`2cba9e42`). That is a different base model, so the two numbers are not a
like-for-like comparison. The panel's replies were written by the previous checkpoint, which is why 1017's own
perplexity on it is higher (1.601 vs 1.498). Two diagnostics recorded before the result point to 1017 being harder to
compress at this budget: its measured EXL3 error energy per K is 1.75x / 2.4x / 3.7x sudoingX's table at K2 / K3 / K4
(layers 0-18), and its sensitivity alphas are 1.6-2x theirs on every layer. The controlled comparison (the same
pipeline on 1017 with sudoingX's `cal_v3` calibration, arm B of `PROTOCOL.md`) has not been run, so whether our
calibration helps or hurts on 1017 is open.

Raw file: `evidence/kld_panel.json` (`per_seq` holds each conversation's KL sum). `python3 scripts/bootstrap_ci.py
evidence/kld_panel.json` reproduces the interval.

## What is in the checkpoint

| part | encoding |
|---|---|
| routed experts, layers 0-39 (15,360) | EXL3 K2 / K3 / K4 / K5 per expert: 6,278 / 7,447 / 1,600 / 35 (`data/plan_T_270.json`), 2.70 bpw, 85.75 GiB a rank at TP2 |
| dense linears (wq_a + wkv, wq_b, wo_b, shared expert) | EXL3 K4 |
| `wo_a` | EXL3 K5 |
| LM head | EXL3 K4 |
| DSpark drafter (`mtp.0-2`) | official precision (FP8 dense, MXFP4 experts) in this release |
| Engram tables | not shipped: read from DeepSeek's own two shards at serve time |
| everything else | the official 1017 tensors |

The drafter is not quantized yet: sudoingX's drafter encode (`p4_drafter.py`, `p4_drafter_experts.py`) needs input
Hessians captured from a live serve on the Sparks. It changes speed and memory, never the output (the target model
verifies every drafted token).

## Reproduce

Everything ran on one rented H200 (141 GB) with 1.5 TB of disk, plus a 4x RTX PRO 6000 box serving 1017 for the
calibration replies. Wall times are the H200's.

1. **Code.** `git clone https://github.com/sudoingX/dgx-spark-deepseek-v41 && git -C dgx-spark-deepseek-v41 checkout
   cc59c3609e8e671418f39eb92aa16b0e4f770b77 && git -C dgx-spark-deepseek-v41 apply
   ../patches/0001-dsv41-1017-over-sudoingx-cc59c36.patch`. The patch:
   - `dsv41q/refmodel.py`: on compute capability < 10, the reference runs its FP8 / FP4 GEMMs with float32
     power-of-two activation scales. TileLang's float -> e8m0 conversion in `act_quant` is broken on sm_90: every
     ue8m0 scale came out as 2^-127 and every GEMM returned about 0. Relative GEMM error with the fix: FP8 0.0266, FP4
     0.0262. Blackwell keeps the original path (`DSV41_REF_F32_SCALES=auto|1|0`). Also adds a host-RAM fallback for
     the Engram tables when the GPU cannot hold them.
   - `caldata/generate.py`: `GEN_URL` / `GEN_MODEL` point it at a local OpenAI-compatible server on the native
     weights. No output cap, so every reply runs to its own end.
   - `caldata/generate_muse.py`, `caldata/build_1017.py` (new): native 1017 replies for the owner calibration
     prompts, and the token set built to the family shares below.
   - `p4_sensitivity.py`: the sequence filter also accepts the new source labels.
   - `p4_kld.py`: writes each conversation's KL sum and position count (`per_seq`) for the bootstrap.
   - `p4_assemble.py`: `--source` records the checkpoint revision in the index metadata.
2. **Weights.** `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170`
   (476 GB). `scripts/h200/verify.py` checks every file against its LFS sha256.
3. **Calibration replies** (on any server running native 1017, no `max_tokens`):
   `GEN_URL=http://<server>/v1/chat/completions GEN_MODEL=<name> python3 caldata/generate.py --set base ...`
   for sudoingX's 299 base calibration prompts, and `python3 caldata/generate_muse.py <prompts.jsonl> muse_1017.jsonl`
   for the owner prompt set (605 items).
4. **Calibration set.** `python3 caldata/build_1017.py <model> muse_1017.jsonl base_1017.jsonl
   <exllamav3>/exllamav3/conversion/standard_cal_data cal_1017_v2`: 927 sequences, 1,216,384 tokens in 2,048-token
   windows, all native 1017 text. Token shares: agentic 34.5 %, coding (TS, Rust, Go, C, C++, HTML, math) 24.7 %,
   religion + philosophy 19.7 %, multilingual 9.9 %, structured 9.9 %, random tokens 1.4 %. sha256 `aafa6309…`
   (`data/cal_1017_v2_report.json`). For comparison, sudoingX's `cal_v3` is 766 sequences / 853,106 tokens.
5. **Pass 0 + sensitivity** (`scripts/h200/run_p0s.sh`, 46 min + 31 min): `p4_pass0.py <model> cal.safetensors
   stats_v2 --index cal_index.json`, then `p4_sensitivity.py <model> cal.safetensors sens_v2.json`
   (`data/sens_v2.json`).
6. **Allocation** (seconds): `p4_allocate.py stats_v2 --bpw 2.70 --decode-bpw 2.90 --alpha sens_v2.json --eps
   2:0.026,3:0.0042,4:0.00058,5:0.000096 --importance tshare --protect-share 0.2 --floor-layers 37,38,39:3 -o
   plan_T_270.json`. These are sudoingX's plan v3 settings unchanged: super experts protected at K5, the DSpark target
   layers 37-39 at K3 or more.
7. **Pass 1** (`scripts/h200/run_p1.sh`, 858 min on one H200): `p4_pass1.py <model> cal.safetensors plan_T_270.json
   p1_T --layers 0-39`. This is the sequential encode, each layer calibrated on the stream through the already
   quantized layers.
8. **Dense + head** (56 min): `p4_dense.py <model> cal.safetensors p1_T dense_T --k 4 --k-map attn.wo_a=5 --head 4`,
   then `p4_dense.py --index dense_T`.
9. **KLD** (26 + 25 min): `caldata/build.py` builds the panel from sudoingX's `data/caldata/conversations.jsonl`
   (hash `39e0db2f…` required), then `p4_kld.py <model> panel.safetensors panel_index.json kld --mode ref`, then
   `--mode quant --tensors p1_T --dense dense_T`. Steps 8-9 are `scripts/h200/finish.sh`.
10. **Assemble** (CPU, no GPU): `p4_assemble.py <model> p1_T plan_T_270.json <out> --dense dense_T --source
    deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`.

Total GPU time: about 17 h on one H200.

Logs for every step: `evidence/logs/`. The evaluation protocol, written before any quantized result existed:
`PROTOCOL.md`.

The owner prompt set behind `muse_1017.jsonl` includes private agent traces, so it is not published. Steps 3-4
publish only the generator, the builder, the shares and the hash. With sudoingX's public base prompts alone, step 4
builds a smaller set that is fully reproducible.

## Serve on 2x DGX Spark

Image: `ghcr.io/sybil-solutions/deepseek-v4.1-flash-exl3-spark` (sybil-solutions/local-ai-images, directory
`deepseek-v4.1-flash-exl3-spark`). It is sudoingX's `dsv41-serve:v4` stack (vLLM `6e517b15` for sm_121, patches
0001-0008 and 0010-0021, exllamav3 1.6.0, the `vllm_exl3_moe` plugin), built by GitHub Actions from pinned archives
with signed provenance. Registry recipe: `sybil-solutions/local-ai-registry`,
`vllm-deepseek-v4.1-flash-1017-exl3-2.7bpw-spark-256k-2x`.

```bash
# on both Sparks, same absolute path (about 400 GB: 199 GB of this repo + DeepSeek's 203 GB Engram shards)
hf download 0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw --local-dir ~/models/dsv41-1017-exl3
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir ~/models/dsv41-1017-exl3
(cd ~/models/dsv41-1017-exl3 && sha256sum -c MANIFEST.sha256)

# from a control machine with passwordless ssh to both (sudoingX's launcher, at the pinned commit)
git clone https://github.com/sudoingX/dgx-spark-deepseek-v41 ~/dsv41_recipe && git -C ~/dsv41_recipe checkout cc59c360
IMAGE=ghcr.io/sybil-solutions/deepseek-v4.1-flash-exl3-spark@sha256:<digest> \
HEAD=<rank-0 host> WORKER=<rank-1 host> MODEL_DIR=$HOME/models/dsv41-1017-exl3 \
~/dsv41_recipe/scripts/launch/dsv41_tp2.sh up
```

The launcher's default `RECIPE_ENV` carries a DSpark round-cost table (`VLLM_DSPARK_ROUND_MS`) measured on sudoingX's
weights, whose drafter experts are EXL3 K3. With this release's native-precision drafter, that table has to be
re-measured (`scripts/bench/depth_sweep.sh`). See the registry recipe for the exact environment.

## Credits and licences

- Method, scripts, serving patches and plugin: sudoingX, [dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41)
  (Apache-2.0), including TensorFold's RDMA ring, L2 prefetch and prompt kernels (bertholomus, Apache-2.0).
- EXL3: turboderp, [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT).
- Model: DeepSeek, `deepseek-ai/DeepSeek-V4.1-Flash-1017` (MIT).
- This repo: Apache-2.0 (the patch modifies Apache-2.0 files and keeps their headers).
