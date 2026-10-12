"""Cloud venv entrypoint. ML/provider code is imported only after the guard."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

import fl720_workflow as flow


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
    from diffusers import ModularPipeline
    assert callable(ComponentsManager.enable_auto_cpu_offload)
    assert upstream.resolve_output_size(0.9,'16:9')==(1280,736)
    assert upstream.resolve_output_size(0.5,'16:9')==(960,544)
    sched=MiniMaxH3Scheduler()
    sched.set_timesteps(5)
    assert len(sched.timesteps)==4, 'Need exactly four transformer evaluations'
    assert upstream.resolve_output_size(0.98,'16:9')==(1344,768)
    assert callable(ModularPipeline.from_pretrained) and callable(upstream._load_rgb_image)
    assert upstream.TASK_BY_IMAGE_COUNT=={0:'t2va',1:'i2va',2:'fl2va'}
    versions={x:metadata.version(x) for x in ('torch','torchvision','diffusers','transformers','peft','accelerate','huggingface-hub','av')}
    (flow.guard.ROOT/'api_preflight.json').write_text(json.dumps({'import_ok':True,'versions':versions,'diffusers_commit':flow.DIFFUSERS_REV},indent=2))
    print('FL720_API_PREFLIGHT_OK',flush=True)
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
    expected_task='t2va' if settings['mode']=='t2v' else 'i2va'
    expected_images=0 if settings['mode']=='t2v' else 1
    assert len(jobs)==1 and jobs[0].task==expected_task
    assert len(jobs[0].image_paths)==expected_images and not jobs[0].reference_specs
    assert upstream.resolve_jobs_workflow(jobs)=='fl2va'
    args=argparse.Namespace(model_id=run['download']['model_path'],device='cuda',fsdp2=False,no_cpu_offload=False,
        memory_reserve_margin='12GB',video_shift=6.,audio_shift=3., lora_path=Path(run['download']['lora_path']),
        lora_alpha=8,lora_scale=1.,fuse_lora=False,attention_backend=None)
    # No FlashAttention3/Hopper kernels; leave the standard attention backend.
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    pipe=upstream.load_pipeline(args,upstream.DistributedContext(),workflow='fl2va')
    torch.cuda.synchronize();loaded=time.monotonic()
    name,transformer=upstream.get_active_transformer(pipe,'fl2va');assert name=='transformer'
    calls=[]
    def before(_module,_inputs):
        torch.cuda.synchronize();calls.append({'start':time.monotonic()})
    def after(_module,_inputs,_output):
        torch.cuda.synchronize();calls[-1]['seconds']=time.monotonic()-calls[-1].pop('start')
    hooks=[transformer.register_forward_pre_hook(before),transformer.register_forward_hook(after)]
    try:
        input_start=time.monotonic()
        conditioning={}
        if settings['mode']=='i2v':
            conditioning['image']=upstream._load_rgb_image(jobs[0].image_paths[0])
        image_file_seconds=time.monotonic()-input_start
        width,height=run['profile']['native']
        torch.cuda.synchronize();gen_start=time.monotonic()
        with torch.inference_mode():
            result=pipe(prompt=jobs[0].prompt,height=height,width=width,num_frames=124,num_inference_steps=5,
                        generator=torch.Generator().manual_seed(settings['seed']),output_type='np',
                        output=['videos','audio','sampling_rate'],**conditioning)
        torch.cuda.synchronize();gen_end=time.monotonic()
        if len(calls)!=4:raise RuntimeError(f'Expected 4 transformer evaluations, got {len(calls)}')
        encode_start=time.monotonic();upstream.save_result_video(result,out/'native.mp4',24)
        metrics={'mode':settings['mode'],'model_workflow':'fl2va',
                 'image_role':'first keyframe' if settings['mode']=='i2v' else 'none',
                 'first_frame_sha256':settings['first_frame']['sha256'] if settings['mode']=='i2v' else None,
                 'lora_training_size':[1344,768],'lora_file':flow.LORA_FILE,
                 'video_shift':6.,'audio_shift':3.,'lora_alpha':8,'lora_scale':1.,
                 'native_size':[width,height],'requested_seconds':5,'frame_aligned_seconds':124/24,
                 'load_seconds':loaded-start,'image_file_read_seconds':image_file_seconds,
                 'conditioning_denoise_decode_seconds':gen_end-gen_start,
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
