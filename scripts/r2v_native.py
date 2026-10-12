"""Cloud venv entrypoint. ML/provider code is imported only after the guard."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

import r2v_workflow as flow


def preflight():
    # Host-approved isolated Colab child only; checked before any ML import.
    flow.guard.require_cloud(verified_child=True)
    flow.verify_vendor()
    import torch
    assert sys.version_info[:2]==(3,12)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,0)
    assert metadata.version('torch')=='2.14.1'
    direct=json.loads(metadata.distribution('diffusers').read_text('direct_url.json'))
    assert direct['vcs_info']['commit_id']==flow.DIFFUSERS_REV
    sys.path.insert(0,str(flow.KIT/'vendor/modeltc'))
    import inference_minimax_h3 as upstream
    from diffusers import ComponentsManager
    from diffusers import MiniMaxH3Scheduler
    from minimax_h3_ref2va_pipeline import resolve_reference_image_size
    assert callable(ComponentsManager.enable_auto_cpu_offload)
    assert upstream.resolve_output_size(0.9,'16:9')==(1280,736)
    assert upstream.resolve_output_size(0.5,'16:9')==(960,544)
    sched=MiniMaxH3Scheduler()
    sched.set_timesteps(5)
    assert len(sched.timesteps)==4, 'Need exactly four transformer evaluations'
    assert max(resolve_reference_image_size(4096,4096,target_width=1280,target_height=736,mode='match'))<2048
    versions={x:metadata.version(x) for x in ('torch','torchvision','diffusers','transformers','peft','accelerate','huggingface-hub','av')}
    (flow.guard.ROOT/'api_preflight.json').write_text(json.dumps({'import_ok':True,'versions':versions,'diffusers_commit':flow.DIFFUSERS_REV},indent=2))
    print('R2V_API_PREFLIGHT_OK',flush=True)
    return torch,upstream


def generate():
    flow.guard.require_cloud(verified_child=True)
    state=flow.guard.read_state()
    if state.get('native_parent_pid')!=os.getppid() or state.get('cloud_boot_id')!=flow.guard.boot_id():
        raise flow.WorkflowError('Guarded Colab parent is required.')
    flow.guard.deadline_remaining()
    run=json.loads((flow.guard.ROOT/'run.json').read_text())
    settings=flow.validate(run['settings'])
    torch,upstream=preflight()
    out=Path(run['output_dir']);job_path=out/'private_job.json'
    job_path.write_text(json.dumps(run['jobs']))
    jobs=upstream.build_jobs(job_path)
    assert len(jobs)==1 and jobs[0].task=='ref2va' and not jobs[0].image_paths
    assert len(jobs[0].reference_specs)==1 and jobs[0].reference_specs[0].kind=='image'
    assert upstream.resolve_jobs_workflow(jobs)=='ref2va'
    args=argparse.Namespace(model_id=run['download']['model_path'],device='cuda',fsdp2=False,no_cpu_offload=False,
        memory_reserve_margin='12GB',video_shift=12.,audio_shift=3., lora_path=Path(run['download']['lora_path']),
        lora_alpha=8,lora_scale=1.,fuse_lora=False,attention_backend=None)
    # No FlashAttention3/Hopper kernels; leave the standard attention backend.
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    pipe=upstream.load_pipeline(args,upstream.DistributedContext(),workflow='ref2va')
    torch.cuda.synchronize();loaded=time.monotonic()
    name,transformer=upstream.get_active_transformer(pipe,'ref2va');assert name=='transformer_ref'
    calls=[]
    def before(_module,_inputs):
        torch.cuda.synchronize();calls.append({'start':time.monotonic()})
    def after(_module,_inputs,_output):
        torch.cuda.synchronize();calls[-1]['seconds']=time.monotonic()-calls[-1].pop('start')
    hooks=[transformer.register_forward_pre_hook(before),transformer.register_forward_hook(after)]
    try:
        ref_start=time.monotonic();references=upstream._load_references(jobs[0].reference_specs)
        reference_file_seconds=time.monotonic()-ref_start
        width,height=run['profile']['native']
        torch.cuda.synchronize();gen_start=time.monotonic()
        with torch.inference_mode():
            result=pipe(prompt=jobs[0].prompt,height=height,width=width,num_frames=124,num_inference_steps=5,
                        generator=torch.Generator().manual_seed(settings['seed']),output_type='np',
                        output=['videos','audio','sampling_rate'],references=references,reference_resize_mode='match')
        torch.cuda.synchronize();gen_end=time.monotonic()
        if len(calls)!=4:raise RuntimeError(f'Expected 4 transformer evaluations, got {len(calls)}')
        encode_start=time.monotonic();upstream.save_result_video(result,out/'native.mp4',24)
        metrics={'mode':'ref2va','reference_role':'semantic reference, not a first frame',
                 'reference_sha256':settings['reference']['sha256'],'reference_resize_mode':'match',
                 'native_size':[width,height],'requested_seconds':5,'frame_aligned_seconds':124/24,
                 'load_seconds':loaded-start,'reference_file_read_seconds':reference_file_seconds,
                 'reference_encode_denoise_decode_seconds':gen_end-gen_start,
                 'native_encode_seconds':time.monotonic()-encode_start,'transformer_evaluations':len(calls),
                 'transformer_forward_seconds':[x['seconds'] for x in calls],
                 'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                 'cuda_peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                 'process_peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                 'api_preflight':json.loads((flow.guard.ROOT/'api_preflight.json').read_text())}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    finally:
        for hook in hooks:hook.remove()
        del pipe


if __name__=='__main__':
    if sys.argv[1:] == ['preflight']:preflight()
    elif sys.argv[1:] == ['generate']:generate()
    else:raise SystemExit('Expected preflight or generate')
