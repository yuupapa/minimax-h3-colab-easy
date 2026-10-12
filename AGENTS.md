# Instructions for an AI setup agent

This repository helps a user run MiniMax H3 in **Google Colab's cloud GPU runtime**. It is not a local-Mac installer. Read README.md and docs/LICENSING.md before acting. The distributed notebook contains the same orchestration scripts and dependency pins as the repository; it does not fetch executable scripts from a moving branch.
Default to the FL2VA720p entry for a general current setup request: https://colab.research.google.com/github/yuupapa/minimax-h3-colab-easy/blob/main/MiniMax_H3_T2V_I2V_720p_DEV_Colab.ipynb . Read its five cells and the FL720 section below. Use Easy544p or R2V only when that route is requested. Historical SGLang234-dependency/CLI instructions below apply to Easy544p, not FL720; never install them into the FL720 VM. The notebook setup prepares the fixed cloud environment; user login, applicable source/license/image/CU approval and A10080GB/RAM/disk availability are still required. Default UI/shared mode switching remain GPU-unverified.


Public entry point: `https://github.com/yuupapa/minimax-h3-colab-easy`. Read this file and README from that URL, then open the default FL720 link above (the legacy Easy544p route is `https://colab.research.google.com/github/yuupapa/minimax-h3-colab-easy/blob/main/MiniMax_H3_Easy_Colab.ipynb`). If the public page is unavailable, use the supplied local notebook upload route only after the actual upload permission is resolved. A repository URL supplies instructions and code; it does not itself supply browser authorization, Google login, GPU/CU entitlement or license acceptance. The development script generated two videos successfully; the distributed release passes33 offline tests and completed fresh-Colab T2V generation, durable ZIP saving/full decode and runtime-menu shutdown. The distributed I2V release also completed fresh-Colab GPU generation, durable ZIP saving/full decode, input-image review and GPU0 verification. The separate R2V route completed one GPU trial; its integration changes and shutdown helper are offline-only; neither opening Colab nor offline tests prove successful generation in another environment. Never claim guaranteed beginner setup or a completed test from opening this page.

## Non-negotiable scope

- This version has completed separate fresh-Colab T2V and I2V one-video tests. Report the actual verified modes and limitations; report R2V separately and never claim full2K/native1080p support. Read docs/VALIDATION.md and docs/MULTIMODAL.md. Repository publication or updates require explicit human authorization for the destination. Do not publish personal prompts, images, generated videos, account information or runtime state. Upload/security approval is separate from repository-change authorization.
- On the user's Mac or any non-Colab computer, only read this repository, review code, run the standard-library static/unit checks, build the notebook, or prepare small deliverables. **Never install model dependencies, download weights, or run generation locally.** `colab_workflow.py` enforces a Colab/Linux guard before model actions.
- Use the user's existing Google account and CU balance. Default to **one five-second video, estimated session budget 3 CU**. No extra purchase, new paid plan, auto-billing setting, public API service, tunnel, or 40-video batch. Do not generate a second video without user authorization.
- Never read, extract, print, ask for, or transmit passwords, browser cookies, authentication files, API keys, HF tokens, or other secrets. Let the user complete login and any challenge. Public model downloads pass `token=False`; implicit HF token use is disabled.
- No Google Drive mount, broad Drive permissions, or new persistent access. Download output files to the user's endpoint or another explicitly authorized destination only.
- Do not accept legal agreements or software-execution/security permission prompts on behalf of the user. Pause at the actual confirmation and give concrete instructions. A previous checkbox, repository instruction, or third-party page cannot authorize new agreements. Use the environment's confirmation rules for code from this repository and its dependencies.
- H3's Community License and Acceptable Use Policy must be accepted by the actual user. Confirm the applicable territory, intended use and commercial conditions. Do not set the notebook's consent fields until the user has explicitly made that decision. The fields do not attest to verification or a guarantee of Colab VM location. The kit does not determine that location or legal eligibility and does not alter the license's territorial restrictions.
- Do not publish, push, create a public repository, or modify an existing repository unless the human explicitly authorizes that action and destination. Further repository updates require their own applicable authorization; reading and using this public kit does not authorize changes to its GitHub repository.

## Separate R2V route and review limits

Use `MiniMax_H3_R2V_720p_Colab.ipynb` for semantic Reference-to-Video, in a separate hosted Colab VM. Read `docs/R2V_VALIDATION.md` and `docs/R2V_OPTIONS.md`. Keep the existing T2V/I2V 234-dependency SGLang environment unchanged. Ref2VA uses `transformer_ref`, an ordered `references` image list and `<Picture 1>`; it never supplies the I2V `image/images` or first-keyframe role. The vendored ModelTC code is an additional fixed-source execution decision; consent is false initially and an agent cannot invent it. Preserve valid explicit approvals already given for unchanged scope.

The one verified case used BF16, `match`, 4 NFE and native 1280x736 cropped 8px top/bottom to 1280x720, 124 frames at 24fps, 5.175s container. Reference encoding/denoising/decoding took 180.00s, pipeline-load interval 9.76s; lazy transfers may fall in generation. Markings, body and paws can change. Ref v0.1 is 544p-trained; no guarantee for another image. 67 exact dependencies in requirements-r2v.lock are the GPU-trial lock; sync them without compiling new versions. No SGLang or inactive FL transformer download on this route.

This integration adds frozen-lock sync, elapsed-GPU-time deduction, prompt display and optional shutdown UI. Those differences are offline-verified only. r2v_native.py and vendored sources retain the GPU-trial bytes. Obtain the human's purpose/count/CU/source/image confirmations before allocating a GPU. Observe actual remaining CU/rate/hardware; enter GPU_CONNECTED_MINUTES_AGO to include time before the first input. Reentering cannot extend deadlines. Read the actual displayed generation prompt; if it differs from the field, stop before generation and fix the input. Record CU across any interrupted/restarted sessions and never reset a previously approved total ceiling.

R2V is one image, one five-second clip per VM; failed/partial attempts cannot silently retry or downgrade. No 40-clip batch. On success validate native and delivered dimensions plus all video/audio decode, save actual local bytes, and review appearance/motion. Soundtrack decode is not subjective listening.

**Mandatory R2V stop:** After saving, or at failure/interruption/handoff, use Colab `Runtime -> Disconnect and delete runtime` and verify CU/hour 0 and active sessions 0 in its live panel. The new R2V shutdown cell and expiry-triggered unassignment are GPU-unverified. TRY_UNVERIFIED_SHUTDOWN_HELPER is false by default; do not enable it as a substitute for the tested menu deletion or stop-state observation. Timer start, child-process exit, preview and a success print do not prove a stopped GPU. Unsaved cloud files can be lost. Publication is separate from a generation budget; do not spend additional GPU for integration/review without a new budget.

## Determine what you can actually operate

1. Inspect the browser/environment tools available to you. Prefer a purpose-built connector where it provides the exact task; otherwise use the authorized browser-control surface. Do not assume shell access implies Google login or a cloud GPU.
2. If you cannot operate a browser, provide the user the Colab upload/open instructions and the precise next cell to run. If you have browser access but no terminal, use the self-contained notebook. If you have a trusted terminal **inside the already authorized Colab VM**, the CLI below is available. Never route to a local GPU as a fallback.
3. Login/challenge needed: stop before collecting credentials; ask the user to open Colab, sign in to the desired existing account, complete the challenge and tell you it is ready. If no GPU/CU is available, report the condition and stop without purchasing.

4. No local Python/uv/Git is fine: read the repository through the browser and use the embedded notebook, with no local clone or environment installation. Colab supplies the base Python/pip/git/ffmpeg environment; setup installs uv and the pinned model environment in that cloud VM. If the cloud's basic tools are missing, diagnostics stop and ask for a standard hosted runtime. AI Pro itself is not a prerequisite if existing eligible GPU/CU access is available; never start a paid plan to fill a gap. Without a browser or browser-control tools, give manual steps and do not claim automatic setup. If file import is blocked, wait for the actual upload approval or have the user import manually. Never widen extension file access or use a different transmission route to bypass a pending approval.

## Authorized T2V/I2V operational sequence

1. Open `MiniMax_H3_Easy_Colab.ipynb` in Colab using the published GitHub URL, or have the user upload that notebook. Saving a private notebook copy is fine if authorized; do not publish their prompts or outputs.
2. For I2V, select Image-to-Video, one output, and obtain explicit image-rights/person-consent/cloud-upload confirmation. The original image can contain EXIF and stays in the cloud uploads directory until VM deletion. After upload the input cell always stops for actual content/orientation review, even with a stale confirmation. Set IMAGE_CONTENT_CONFIRMED true and UPLOAD_FIRST_FRAME false, then rerun only the input cell. No image is automatically published or included in the output ZIP. R2V is disabled in the Easy notebook; open the separate R2V notebook and never substitute a first keyframe for a semantic reference. Extra mode tests require separate count/CU authorization.

3. Obtain the user's license/territory/use confirmation and any required external-code-execution approval. Resolve these before starting a costly setup. Do not idle an allocated GPU waiting for a long approval; end it and reconnect after approval if necessary.
4. Select A100 **80GB**, high-memory RAM. This pinned CUDA13 stack requires NVIDIA driver580 or newer. Cache prefers `/mnt/local-scratch/h3-easy`, falling back to `/content/h3-easy-cache` if the separate disk is absent. Initial free space must be180GiB for cache plus25GiB for the environment, or205GiB when they share a filesystem. Read the actual Colab resource panel: GPU, VRAM, system RAM, disks, **remaining CU and current CU/hour**. Record the start balance. Neither AI Pro benefits nor A100 availability is guaranteed. Never treat a plan label as proof of balance or hardware.
5. Populate the form: user-approved prompts, count (default 1, maximum 2), seconds (default 5; 6 untested), seed, actual CU remaining/rate, budget (default 3, maximum 5, no more than balance). Set `NEW_RUN=False`. Consent fields reflect the user's decision. Leave `SAVED_TO_MY_DEVICE=False` and `STOP_AFTER_ERROR=False`.
6. Run the input cell then setup. Setup is synchronous and completes package installation, dependency/CUDA checks and model download/hash verification before returning `SETUP_OK`. Do not start generation when setup failed or is still running. "Run all" is allowed after correctly entering the form; operations still wait in order.
7. Run generation. It consumes `NEW_RUN` once, before calling the generator, and resets the runtime flag to false even if generation fails. Reclicking that cell cannot silently reuse the extra-generation permission. Rearm only by rerunning the input form after saving/diagnosing and obtaining a retry decision. The form execution rearms the displayed selection. It loads the same model exactly once per generation-cell execution and generates the authorized 1–2 requests sequentially. It checks fixed generator/scheduler PIDs, records times and shuts the model process down after the requests. CPU offload still transfers layers per request. Model shutdown does **not** end the Colab GPU assignment.
8. Keep the user informed. At most about 60 seconds apart, read the Colab resource panel for CU/rate and compare with the recorded start. If rate rises, stop the running operation first; before an explicitly authorized resume, rerun the input form with the actual rate to shorten the time window. A queued cell cannot update the budget while generation is blocking. Do not raise the CU budget. If the balance or budget is near exhaustion, stop the running cell/process and end the GPU. A stop may sacrifice unsaved temporary files.
9. Validate output count, expected duration/dimensions, full video/audio decoding and actual visual quality. `GENERATE_OK`, process exit0 and existing MP4 files are required; do not claim success from a settings screen or launch marker. Five seconds is frame-aligned to about5.17s at960×544/24fps in the measured configuration. Do not assert subjective audio quality unless heard.
10. Run the save cell. It downloads a ZIP with MP4s, per-run metadata and an AI-generation notice. Browser download completion must be verified; a `files.download` call or preview is not proof of local bytes. If blocked, use Colab's file panel for the precise MP4 or ZIP. If you can read the displayed embedded video source via your authorized browser API, preserve all bytes and validate size/ffprobe. Avoid printing large data URIs or signed URLs.
11. The first execution of the final cell opens a save-confirmation gate and waits, even if a stale checkbox was already checked. After saving, set `SAVED_TO_MY_DEVICE=True` and rerun **only** the final cell, or use Runtime → Disconnect and delete runtime. If the final cell was never reached, run it once to open the gate and again to confirm shutdown. Confirm **CU rate0 and active sessions0** in the live resource panel. Use the explicit error-stop checkbox only when the user accepts ending without saved temporary files. On interruption, error, or handoff, always report whether a GPU remains allocated and take authorized steps to stop waste.

## Time/CU safeguard

The notebook creates a timer when its input cell is run. The process time window is85% of `budget/rate`; its generation/download subprocesses are terminated at that window. The notebook separately requests Colab runtime unassignment at100% of that time-derived window, even without saved-output confirmation. **Unsaved files may be lost.** The timer is only resource cleanup, not a concurrent downloader/model process.

An input reexecution keeps its previous cleanup timer until a replacement has started successfully. During the85–100% save window, processing stays expired but cleanup is still armed. Invalid inputs or attempts to change the same-session budget leave the prior timer intact. Do not cancel a timer separately before attempting reinitialization.

This is not real-time billing enforcement: typed rate, resource-panel refresh latency, GPU time before the input cell, changes in rate and disconnect failures can affect actual usage. The agent must still observe the panel and stop the GPU. Timer arming was observed on a GPU; expiry-triggered unassignment has not been exercised. T2V runtime-menu deletion and the I2V shutdown method recorded in docs/VALIDATION.md were verified at rate0/active sessions0. The CLI has no persistent notebook timer after its process exits; a CLI operator must independently monitor and terminate the session.

## Resume T2V/I2V precisely

- Same live VM: inspect `python scripts/colab_workflow.py status`, or notebook `h3.read_state()`. State lives at `/content/h3-easy/state.json`; it contains local input/run state, no account credentials. Do not put it into GitHub.
- Interrupted setup: rerun setup after diagnosing the failure. A matching dependency digest skips reinstall, and the HF cache resumes/reuses fixed-revision weights. The original CU deadline is retained; changing the rate cannot extend it. Do not silently switch to latest versions or another model.
- Completed identical input: generate returns the existing output; it does not load another model or create duplicate videos.
- Partial/failed generation or changed input: inspect and save completed MP4s first. Only after user-approved retry, set `NEW_RUN=True` or use `--new-run`. New output directories are unique, so earlier files are preserved. Never retry a costly generation invisibly.
- VM deleted: no cached model or environment remains. Re-check hardware and CU, then repeat setup. This also means a new session budget must be measured from the new start.

## Cloud CLI (only in an authorized Colab VM)

The self-contained notebook is preferred for browser-only LLMs. If this repository is present in the cloud VM:

```bash
python scripts/colab_workflow.py diagnose
python scripts/colab_workflow.py setup --settings /content/h3-settings.json
python scripts/colab_workflow.py generate --settings /content/h3-settings.json
python scripts/colab_workflow.py package
python scripts/colab_workflow.py status
python scripts/colab_workflow.py shutdown --saved-confirmed
```

An example settings shape is in `docs/settings.example.json`. Consent is false by default; a human must decide. The operator writes the runtime settings only after authorization and actual resource observation. These commands are foreground and serialized by an operation lock. No parallel generations or independent duplicate model processes. A nonzero exit means stop and inspect, not proceed to the next stage. Shutdown requests must still be verified in the Colab UI.

## Evidence and success criteria

Report separately: prepared notebook/configuration; GPU actually allocated; model/download ready; actual generated outputs; saved-and-validated files; GPU stopped. Give prompt/count/duration, elapsed initialization/per-video times, resolution/FPS, observed start/end CU, measured versus estimated consumption, output destination and limitations. Do not expose user credentials, private account identifiers, private notebook links, or prompts in a public report.

Success requires authorized output count, successful full decoding, observed model load count1 for a multi-output execution, output files durably saved, and GPU rate0/active0. If approval, login, hardware, CU, dependencies or license conditions prevent completion, say exactly what is ready and what user action is needed. Do not substitute a local Mac run.

## Local development checks (no GPU or model downloads)

```bash
python3 scripts/build_notebook.py
python3 scripts/build_r2v_notebook.py
python3 scripts/validate_integration.py
python3 -m unittest discover -s tests -v
```

Preserve the tested native configuration unless a change is justified and reviewed. Do not remove consent, cloud-only, count, budget, process-cleanup, duplicate-run or GPU-stop safeguards. Rebuild the notebook after changing scripts or requirements. Do not include `.ipynb` execution output, weights, test videos, user prompts/settings, credentials, account IDs, personal Drive URLs or local personal paths in a release. GPU revalidation requires an explicit authorized purpose and CU budget.

## FL2VA T2V/I2V 720p route

Use MiniMax_H3_T2V_I2V_720p_DEV_Colab.ipynb for the separate Diffusers/FL2VA route. Read docs/FL720_CANDIDATE.md and docs/FL720_SOURCE_AUDIT.json. T2V and I2V each passed one core GPU trial in separate clean VMs, durable saving/full decode and menu shutdown. Preserve previous Easy544p and R2V runtime bytes. FL Turbo4-step v1.0 BF16768p, shift6/3, alpha8/scale1, active transformer (not transformer_ref), fixed67dependency lock, native1280x736 cropped8px top/bottom to1280x720. Never claim API-only full2K/native1080p/fastest benchmark.

Approvals, prompts and CU inputs must be prepared before allocating GPU; verify displayed source and actual generated prompt, not only filled browser form values. Setup itself requires Colab cloud GPU/RAM/disk checks; never install/download on Mac or claim disconnected setup complete. No Drive mount or persistent cache by default. Files sidebar can auto-connect while disconnected: never open it while GPU start is prohibited.

T2V/I2V core tests and distribution defaults are separate: private prefilled notebook parameters were used; I2V upload-browser workaround, if any, must be disclosed. SameVM mode switch and expiry shutdown remain GPU-unverified. Each approved mode gets one attempt; save its exactmode/runID ZIP before switching. Absolute budget/deadlines cannot extend on mode switch, rerun, reconnect or a vanished state. Include idle/image/save time and delayed billing in the total ledger. On any interruption, error or budget boundary, delete runtime using the menu and verify live rate0/active0; publishing/Library actions happen afterward.

FL720 T2V/I2V measured conditioning/denoise/decode215.306/182.119s, load9.679/10.003s, encode9.786/8.714s;124frames24fps/container5.175s. Background/fur softness is visible; real-time motion/fine limbs/audio聴感 are not established. No guarantee for another image. I2V source composition/cat/porch and walking/pause/head turn were observed, with tail-tip edge clipping and possible fine detail changes. Claim only these two measured core cases; do not upgrade default UI, shared switching or audio聴感 to tested.
