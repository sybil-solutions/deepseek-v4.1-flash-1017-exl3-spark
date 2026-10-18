# Evaluation protocol (pre-registered 2026-10-17, before any quantized result existed)
Written before pass1; any change after a panel number is seen must be recorded here with its reason, and a change that
was informed by panel results can only be judged on a NEW held-out panel.

Question: does native-1017 calibration at the owner composition give a lower KLD than sudoingx's method on the same
checkpoint, at the same memory and decode budget?

Arms (same checkpoint 1017 @ 1bae0e10, same code, same allocator settings = plan v3: 2.70 bpw, decode cap 2.90,
eps 2:0.026/3:0.0042/4:0.00058/5:0.000096, importance tshare, protect-share 0.2, floor 37-39:K3, dense K4 / wo_a K5 /
head K4, drafter as v3; the only difference is the calibration set):
- B (baseline): sudoingx's published cal_v3 (853,106 tok, sha256 as in their RECIPE.md) -> pass0 -> sens -> plan -> pass1.
- T (treatment): cal_1017_v2 (1,216,384 tok, sha256 aafa6309...) -> same pipeline.
- R (record, reference only): sudoingx's published quant on its own checkpoint, 0.0243 / 95.57 % - a different base model,
  so it is context, not the comparison. Optional harness check: re-score R on our harness against the old revision and
  require agreement with 0.0243 within its CI.

Primary metric: mean per-position KL(ref || quant) over the 254,846 scored assistant positions of sudoingx's panel
(199 conversations, sha256 39e0db2f...), teacher-forced, reference = DeepSeek inference/model.py on 1017 (H200, fp32
power-of-two scales, validated below). Secondary: top-1 agreement, domain-equal mean KLD, per-domain KLD, p99 KLD.
Statistics: paired by conversation; 95 % bootstrap CI (10,000 resamples of the 199 conversations) for mean KLD of B, T
and the paired difference T - B. Decision: T beats B iff the CI of T - B lies below 0. "Beats the record" is claimed
only as: T's mean KLD and CI vs the 0.0243 figure, stated as a cross-checkpoint comparison.

Validity checks (all before scoring B / T):
1. Reference validity: H200 reference top-k logprobs vs pop-os vLLM serving 1017 on the same token sequences (different
   kernels, so small nonzero KL is expected; report the value).
2. Leakage: no panel prompt shares an 8-gram with any calibration conversation (T and B); report counts.
3. Determinism: re-score 10 panel conversations twice; identical results required.
4. Every artifact hashed (cal, plan, layer files, panel) and recorded in ledger.jsonl.
No tuning on the panel. One pre-specified secondary held-out panel (1017-native replies to the panel prompts) may be
reported alongside, never used to choose settings.

## Results
- 2026-10-18 arm T (finish pod d1e08407, terminated 15:32 UTC): panel KLD mean 0.03039 (95 % CI 0.0268-0.0343, bootstrap
  over 199 conversations), domain-equal 0.0512, top-1 94.95 %, p99 0.435, ppl 1.601 -> 1.623. Per domain KLD: code 0.0182,
  math 0.0230, agentic 0.0277, long 0.0379, knowledge 0.0568, writing 0.0795, multilingual 0.0827, chat 0.0835.
  Versus record R 0.0243 / 95.57 % (different checkpoint): not beaten; the whole CI is above 0.0243. Arm B not yet run,
  so the T-vs-B question is open. Evidence: HF 0xSero/DeepSeek-V4.1-Flash-1017-EXL3-wip kld_T/kld.json.
