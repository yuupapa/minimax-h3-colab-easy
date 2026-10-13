# Instructions for a MiniMax H3 setup agent

Default: https://colab.research.google.com/github/yuupapa/minimax-h3-colab-easy/blob/main/MiniMax_H3_T2V_I2V_720p_DEV_Colab.ipynb . Read README and docs/LICENSING.md. This is a self-contained cloud Colab kit, not a Mac ML installer.

## Scope and authorization

- Use Colab cloud GPU only. On Mac, only read/rebuild the kit and run stdlib/mock checks. Never install ML dependencies, download model weights or generate locally.
- Resolve the user's purpose, finite count and whole-session CU ceiling before allocating GPU. A typed input change is a new queue request; the explicit count is the number to run. Never infer a 40-video request from a future plan. No extra purchase/subscription/auto-billing/tunnel/public API/Drive mount.
- Preserve current human approvals for unchanged source/license/use/image scope. Do not repeatedly ask permission already supplied. New images require applicable rights/cloud-upload and content/orientation review. Do not invent license agreement or broader access.
- Never read/extract credentials, cookies, auth files or tokens. Let the user handle login/challenges. No implicit HF token use.
- Publishing this repository requires human authorization for this destination. Never publish private prompts/images/videos/account/runtime state. Generation and publication budgets are separate.

## Continuous flow (supersedes former one-trial-per-mode rules)

1. Prepare notebook input/source before GPU: selected mode, VIDEO_COUNT1..100, PROMPT or PROMPTS_JSON with one/count-matching strings, seed, source/license/image fields and session ceiling. Legacy TRIAL_APPROVED field names now record permission to use that mode, not a single-use output limit.
2. Connect A10080GB/highRAM and observe actual CU/rate/elapsed time/resources. Driver580+, RAM about167GiB/free140, cache180GiB + environment25GiB (shared205GiB) still required. Recheck actual balance; a plan label does not prove entitlement/hardware. Do not open Files while disconnected when GPU start is forbidden; it may auto-connect.
3. Run input -> setup -> generate -> save. Setup/download is reused within that VM. Each validated clip calls the unchanged native one-clip model process sequentially; no resident-model or multi-GPU claim.
4. Different prompt/image/seed/count/mode: run input to select a new queue, then generate. Existing queues and output directories are retained. Mode changes invalidate old I2V image selection and require current mode/image review, without resetting CU deadlines.
5. Same current input: generation reexecution returns existing results. NEW_RUN is for intentionally making another queue of the same input. Consume NEW_RUN/RESUME_FAILED_QUEUE during input selection; generation re-click does not rearm retries.
6. A stop or failure terminates remaining work and preserves completed per-clip ZIPs. The save cell can package current completed clips even if the queue failed. Review/resolve the failure, set RESUME_FAILED_QUEUE for the same input and run input again; completed jobs are skipped and unfinished jobs use fresh output directories. Never automatically retry or downgrade resolution/LoRA.
7. Stop with Runtime -> Interrupt execution; the foreground process cleanup remains in finally. A graceful stop marker is also supported by serial_queue.request_stop in an authorized cloud process. Never rely on another queued notebook cell to interrupt a busy kernel.
8. Keep a global consumption/deadline ledger through all queues and reconnects. Do not increase same-session budget, extend deadlines by input changes, or hide failed attempts. Budget can exceed the former5CU trial cap on the720 routes only when explicitly set within actual balance. The legacy Easy544 route remains1..2/max5CU.
9. Validate counts/native vs delivered size/AVdecode and actual quality. Preview shows the last completed clip and completed/requested counts. Save the exact current queue/run/ZIP; stale saved flags cannot confirm a new queue/archive. Earlier outputs remain available for saving. Images are not included in delivery ZIPs.
10. Before work ends or CU expires, save completed outputs and use Runtime -> Disconnect and delete runtime. Verify actual rate0 and active0. Model-process exit, preview, browser close or success prints do not prove a stopped allocation. Auto-expiry unassignment is GPU-unverified and unsaved VM files may disappear.

## Routes and evidence

- FL720: T2V text / I2V one first keyframe, transformer, FL Turbo4step v1.0 768p BF16, shift6/3, alpha8/scale1. native1280x736 cropped8px top/bottom to1280x720, no upscale, fixed5seconds/124frames24fps/container5.175s. Each mode had one separate-cleanVM core GPU trial; standard browser upload UI, new queue control and warm switching are not newly GPU tested. Current changes have offline mock regression evidence only.
- R2V: MiniMax_H3_R2V_720p_Colab.ipynb, separate VM/venv, transformer_ref and semantic references/Picture1, never I2V image conditioning. Finite queues1..100 now supported, one reference image per queue, change it for another queue. near720_crop is measured in one example;544_upscale remains GPU-unverified. Preserve model/vendor/lock/native bytes.
- Easy544: MiniMax_H3_Easy_Colab.ipynb retains the old SGLang234dependency environment and1..2 request behavior. Never install it into FL720/R2V's67dependency VM.
- Old per-mode one-output/consumed-permission restrictions were prototype policy and are removed on720 routes. Do not claim all updated paths GPU-verified or guaranteed speed/quality. No API-only full2K/native1080p. Audio decode is not subjective listening.

## Offline validation

python3 scripts/build_fl720_notebook.py
python3 scripts/build_r2v_notebook.py
python3 scripts/validate_fl720.py
python3 -m unittest discover -s tests -q

Rebuild all changed embeddings. Preserve vendor revisions/licenses/immutable native generation settings, consent/cloud/count/budget/process-cleanup/stop protections. Keep output cells and personal artifacts out of the release. Read server main before publishing; update with expected-head lease, verify every remote blob and CI state afterward. Do not start GPU tests for orchestration changes unless indispensable and explicitly budgeted.
