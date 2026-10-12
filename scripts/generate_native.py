# Original orchestration code, MIT. Model licenses apply separately.
import json,time,pathlib,os,traceback

def main():
    from colab_workflow import require_cloud,read_state
    require_cloud(verified_child=True)
    state=read_state()
    assert state.get('setup_complete') and state.get('license_accepted') and state.get('eligible_use_confirmed'), 'Setup and user license confirmation required.'
    from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
    out=pathlib.Path(os.environ['H3_OUTDIR'])
    report=json.loads(pathlib.Path(os.environ['H3_DOWNLOAD_REPORT']).read_text())
    inputs=json.loads((out/'inputs.json').read_text())
    assert 1 <= len(inputs['prompts']) <= 2, 'This test permits at most two outputs.'
    mode=inputs.get('mode','t2v')
    assert mode in ('t2v','i2v'), 'Ref2VA is not enabled in this candidate.'
    first_frame=None
    if mode=='i2v':
        from colab_workflow import verify_first_frame
        assert len(inputs['prompts'])==1, 'I2V candidate permits one output.'
        first_frame=verify_first_frame(inputs['first_frame'])
    config={'model_path':'MiniMaxAI/MiniMax-H3','revision':'42ed227ee7df40d41602854ae760620d6eb651fe','model_variant':'fl2va','backend':'sglang','trust_remote_code':False,'num_gpus':1,'tp_size':1,'ulysses_degree':1,'performance_mode':'memory','layerwise_offload_components':['dit','text_encoder','vae'],'dit_layerwise_resident_layers':20,'layerwise_resident_layers':'video_vae=36','attention_backend':'torch_sdpa','enable_torch_compile':False,'warmup_mode':'off','lora_path':report['lora_path'],'lora_scale':1.0,'lora_alpha':8,'lora_merge_mode':'auto'}
    summary={'status':'initializing','parent_pid':os.getpid(),'model_load_count':0,'requests':[],'config':config,'inputs':inputs}
    def save():
        (out/'sequence_report.json').write_text(json.dumps(summary,indent=2))
    save()
    generator=None
    try:
        load_start=time.monotonic()
        generator=DiffGenerator.from_pretrained(**config)
        summary.update(status='generating',model_load_count=1,model_load_seconds=round(time.monotonic()-load_start,3),generator_identity=id(generator),scheduler_pids=[p.pid for p in generator.local_scheduler_process or []])
        save()
        print('MODEL_LOAD_ONCE',json.dumps({k:summary[k] for k in ['model_load_count','model_load_seconds','parent_pid','generator_identity','scheduler_pids']}),flush=True)
        for i,prompt in enumerate(inputs['prompts'],1):
            name=f'h3_sequence_{i:02d}.mp4'
            sampling={'prompt':prompt,'task':'t2va','conditions':[],'target':{'short_edge':544,'aspect_ratio':'16:9','duration_seconds':inputs['duration_seconds']},'flow_shift':12.0,'audio_flow_shift':3.0,'num_inference_steps':4,'seed':inputs['seeds'][i-1],'num_outputs_per_prompt':1,'quality':'lossless','save_output':True,'output_path':str(out),'output_file_name':name,'perf_dump_path':str(out/f'performance_{i:02d}.json')}
            if mode=='i2v':
                sampling.update(task='fl2va',conditions=[{'type':'image','uri':first_frame.as_uri(),'role':'keyframe','frame_index':0}])
            params=MiniMaxH3SamplingParams(**sampling)
            current_pids=[p.pid for p in generator.local_scheduler_process or []]
            assert current_pids==summary['scheduler_pids'], 'Scheduler process changed.'
            event={'index':i,'status':'running','generator_identity':id(generator),'scheduler_pids':current_pids,'parent_pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sampling':sampling}
            summary['requests'].append(event); save()
            print('REQUEST_BEGIN',json.dumps({'index':i,'model_load_count':summary['model_load_count'],'generator_identity':id(generator),'scheduler_pids':current_pids}),flush=True)
            print('VALIDATED_NATIVE_REQUEST',params.build_request_extra(),flush=True)
            started=time.monotonic()
            result=generator.generate(sampling_params_kwargs=sampling)
            elapsed=time.monotonic()-started
            video=out/name
            assert result is not None and video.exists() and video.stat().st_size>0, f'Request {i} failed.'
            event.update(status='completed',generation_seconds=round(elapsed,3),ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),output=name,bytes=video.stat().st_size)
            save()
            print('REQUEST_COMPLETE',json.dumps({k:event[k] for k in ['index','status','generation_seconds','output','bytes','generator_identity','scheduler_pids']}),flush=True)
        summary['status']='completed'; save()
        print('SEQUENCE_REPORT',json.dumps(summary),flush=True)
    except Exception as exc:
        summary.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        save(); traceback.print_exc(); raise
    finally:
        if generator is not None:
            generator.shutdown()
            summary['generator_shutdown']=True
            save()
            print('GENERATOR_SHUTDOWN',flush=True)

if __name__=='__main__':
    main()
