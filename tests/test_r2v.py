"""Meaningful offline guard/data/representation checks; no provider execution."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
import r2v_workflow as r
import build_r2v_notebook as builder


class Ref2VAChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.patches=[patch.object(r.guard,'ROOT',self.root),patch.object(r.guard,'STATE',self.root/'state.json')]
        for p in self.patches:p.start()
        # Synthetic header fixture for the stdlib managed-image validator only.
        # No provider/ML/Pillow/ffmpeg execution or local video generation.
        data=b'\x89PNG\r\n\x1a\n'+b'\x00\x00\x00\rIHDR'+struct.pack('>II',1024,1024)+b'\x08\x02\x00\x00\x00'+b'\0'*8
        sha=hashlib.sha256(data).hexdigest();p=self.root/'inputs'/(sha+'.png');p.parent.mkdir();p.write_bytes(data)
        self.settings={'gpu_trial_approved':True,'source_execution_accepted':True,'license_accepted':True,
             'eligible_use_confirmed':True,'image_use_confirmed':True,'image_content_confirmed':True,
             'higher_resolution_risk_accepted':True,'profile':'near720_crop','prompts':['A toy in <Picture 1> on a beach.'],
             'count':1,'seconds':5,'seed':42,'cu_remaining':100.,'cu_rate':6.77,'cu_budget':5.,
             'reference':{'path':str(p),'sha256':sha,'width':1024,'height':1024}}

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def test_semantic_reference_job_never_first_frame(self):
        s=r.validate(self.settings);job=r.jobs_document(s)['examples'][0]
        self.assertEqual(job['task'],'ref2va');self.assertEqual(job['references'],[{'type':'image','path':s['reference']['path']}])
        self.assertNotIn('image',job);self.assertNotIn('images',job)

    def test_all_consents_and_new_gpu_approval_required(self):
        for key in ('gpu_trial_approved','source_execution_accepted','license_accepted','eligible_use_confirmed',
                    'image_use_confirmed','image_content_confirmed','higher_resolution_risk_accepted'):
            with self.subTest(key=key):
                s=copy.deepcopy(self.settings);s[key]=False
                with self.assertRaises(r.WorkflowError):r.validate(s)

    def test_one_five_second_request_only(self):
        for count,seconds in [(0,5),(101,5),(True,5),(1,6)]:
            s=dict(self.settings,count=count,seconds=seconds)
            with self.assertRaises(r.WorkflowError):r.validate(s)

    def test_budget_finite_and_no_purchase_or_extension(self):
        for key,value in [('cu_budget',0),('cu_budget',101),('cu_rate',float('nan')),('cu_remaining',0)]:
            with self.subTest(key=key,value=value):
                s=dict(self.settings);s[key]=value
                with self.assertRaises(r.WorkflowError):r.validate(s)

    def test_only_one_matching_picture_reference(self):
        for prompt in ('a toy','<Picture 2> walks','<Picture 1> and <Picture 2>'):
            with self.assertRaises(r.WorkflowError):r.validate(dict(self.settings,prompts=[prompt]))

    def test_reference_hash_path_and_aspect(self):
        for update in ({'path':'https://example.org/ref.png'},{'sha256':'0'*64},{'width':5000,'height':64}):
            s=copy.deepcopy(self.settings);s['reference'].update(update)
            with self.assertRaises(r.WorkflowError):r.validate(s)

    def test_profiles_distinguish_native_and_delivery(self):
        p=r.PROFILES['near720_crop'];self.assertEqual(p['native'],[1280,736]);self.assertEqual(p['delivery'],[1280,720])
        self.assertEqual(p['filter'],'crop=1280:720:0:8,setsar=1')
        self.assertNotIn('scale=',p['filter'])
        p=r.PROFILES['544_upscale'];self.assertEqual(p['native'],[960,544])
        self.assertIn('crop=960:540:0:2,scale=1280:720',p['filter'])
        self.assertEqual(720%32,16)  # do not send invalid native height720 to this pipeline

    def test_provider_resolution_ladder_static_only(self):
        tree=ast.parse((ROOT/'vendor/modeltc/resolution_util.py').read_text())
        ladder=next(n for n in tree.body if isinstance(n,ast.AnnAssign) and getattr(n.target,'id',None)=='PIXEL_LADDER_16_9')
        values=ast.literal_eval(ladder.value)
        self.assertEqual(values[.9],(1280,736));self.assertEqual(values[.5],(960,544))

    def test_vendor_revision_and_integrity(self):
        r.verify_vendor()
        for e in json.loads((ROOT/'vendor/modeltc/manifest.json').read_text()):
            if e['path'].endswith('.py'):ast.parse((ROOT/'vendor/modeltc'/e['path']).read_text())

    def test_only_ref_diffusers_components_downloaded(self):
        self.assertIn('transformer_ref/*',r.PATTERNS)
        for item in ('transformer/*','Ref2VA/*','FL2VA/*','*'):self.assertNotIn(item,r.PATTERNS)
        self.assertIn('ref2v_turbo_4step',r.LORA_FILE)

    def test_source_match_policy_and_transformer_ref_static(self):
        source=(ROOT/'scripts/r2v_native.py').read_text();tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        pipeline_calls=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='pipe']
        self.assertEqual(len(pipeline_calls),1)
        kw={x.arg:x.value for x in pipeline_calls[0].keywords}
        self.assertEqual(ast.literal_eval(kw['num_inference_steps']),5)
        self.assertEqual(ast.literal_eval(kw['num_frames']),124)
        self.assertEqual(ast.literal_eval(kw['reference_resize_mode']),'match')
        self.assertIn('references',kw);self.assertNotIn('image',kw)
        self.assertIn("workflow='ref2va'",source);self.assertIn('len(calls)!=4',source)
        self.assertIn('attention_backend=None',source)

    def test_cloud_guard_before_setup_and_generation_side_effects(self):
        for fn in (r.setup,r.generate,r.upload_reference,r.arm_budget):
            with patch.object(r.guard,'require_cloud',side_effect=r.WorkflowError('Mac blocked')) as guard, \
                 patch.object(r.guard,'run_waiting') as execute:
                with self.assertRaisesRegex(r.WorkflowError,'Mac blocked'):fn(self.settings)
                guard.assert_called_once();execute.assert_not_called()
        self.assertFalse((self.root/'venv').exists())

    def test_child_preflight_guard_before_ml_import(self):
        spec=importlib.util.spec_from_file_location('r2v_native_test',ROOT/'scripts/r2v_native.py')
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        with patch.object(r.guard,'require_cloud',side_effect=r.WorkflowError('Mac blocked')):
            with self.assertRaisesRegex(r.WorkflowError,'Mac blocked'):native.preflight()

    def test_isolated_guard_does_not_modify_published_module(self):
        import colab_workflow
        self.assertEqual(colab_workflow.ROOT,Path('/content/h3-easy'))
        self.assertNotEqual(r.guard.STATE,colab_workflow.STATE)
        self.assertNotEqual(r.guard.VENV,colab_workflow.VENV)

    def test_failed_attempt_never_regenerates(self):
        with patch.object(r.guard,'require_cloud'),patch.object(r.guard,'deadline_remaining',return_value=1000.):
            r.select_request(self.settings)
        state=r.guard.read_state();state['generation_queues'][state['active_queue_id']]['status']='failed';r.guard.write_state(state)
        with patch.object(r.guard,'require_cloud'),patch.object(r,'initialize_budget'),patch.object(r.guard,'deadline_remaining',return_value=1000.),patch.object(r,'_queued_one') as execute:
            with self.assertRaisesRegex(r.WorkflowError,'再開'):r.select_request(self.settings)
            execute.assert_not_called()

    def test_completed_duplicate_returns_existing_archive(self):
        out=self.root/'completed';out.mkdir();(out/'delivery_720p.mp4').write_bytes(b'fixture')
        archive=self.root/'already.zip';archive.write_bytes(b'fixture')
        r.guard.write_state({'generation_attempted':True,'generation_complete':True,'fingerprint':r.fingerprint(self.settings),
                            'output_dir':str(out),'archive_path':str(archive)})
        with patch.object(r.guard,'require_cloud'),patch.object(r,'initialize_budget'),patch.object(r.guard,'deadline_remaining',return_value=1000.),patch.object(r.guard,'run_waiting') as execute:
            self.assertEqual(r.generate(self.settings),str(archive));execute.assert_not_called()
            changed=r.select_request(dict(self.settings,seed=43))
            self.assertEqual(changed['seed'],43)

    def test_probe_rejects_704_as_720_and_wrong_frames(self):
        info={'streams':[{'codec_type':'video','width':1280,'height':720,'nb_frames':'124','avg_frame_rate':'24/1'},
                         {'codec_type':'audio'}],'format':{'duration':'5.175'}}
        self.assertEqual(r.check_probe(info,1280,720)['height'],720)
        for key,value in [('height',704),('nb_frames','123'),('avg_frame_rate','30/1')]:
            bad=copy.deepcopy(info);bad['streams'][0][key]=value
            with self.assertRaises(r.WorkflowError):r.check_probe(bad,1280,720)

    def test_save_and_discard_shutdown_gate(self):
        with patch.object(r.guard,'require_cloud'),patch.object(r.guard,'shutdown') as stop:
            with self.assertRaises(r.WorkflowError):r.shutdown()
            stop.assert_not_called();r.shutdown(saved=True);stop.assert_called_once_with(saved_confirmed=True)
            stop.reset_mock();r.shutdown(stop_after_error=True);stop.assert_called_once()

    def test_inference_env_offline_and_no_implicit_token(self):
        with patch.dict('os.environ',{'HF_TOKEN':'test-value','HUGGING_FACE_HUB_TOKEN':'test-value'}):
            env=r.environment(offline=True)
        self.assertEqual(env['HF_HUB_OFFLINE'],'1');self.assertEqual(env['HF_HUB_DISABLE_IMPLICIT_TOKEN'],'1')
        self.assertNotIn('HF_TOKEN',env);self.assertNotIn('HUGGING_FACE_HUB_TOKEN',env)

    def test_notebook_selfcontained_matches_sources_with_consents_off(self):
        nb=json.loads(builder.build().read_text());code=[c for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),5)
        for c in code:ast.parse(c['source']);self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])
        tree=ast.parse(code[0]['source'])
        embedded=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='EMBEDDED_KIT'))
        for name in builder.FILES:self.assertEqual(embedded[name],(ROOT/name).read_text())
        for name in ('GPU_TRIAL_APPROVED','SOURCE_EXECUTION_ACCEPTED','LICENSE_ACCEPTED','ELIGIBLE_USE_CONFIRMED',
                     'IMAGE_USE_CONFIRMED','IMAGE_CONTENT_CONFIRMED','HIGHER_RESOLUTION_RISK_ACCEPTED','UPLOAD_REFERENCE'):
            v=next(n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
            self.assertIs(ast.literal_eval(v),False)
        for marker in ('SAVED_TO_MY_DEVICE = False','STOP_AFTER_ERROR = False'):self.assertIn(marker,code[-1]['source'])

    def test_stale_saved_and_discard_flags_cannot_stop_first_successful_pass(self):
        nb=json.loads(builder.build().read_text());source=nb['cells'][-1]['source']
        # Simulate a persisted prior form selection; no Colab/native calls.
        source=source.replace('SAVED_TO_MY_DEVICE = False','SAVED_TO_MY_DEVICE = True',1)
        source=source.replace('STOP_AFTER_ERROR = False','STOP_AFTER_ERROR = True',1)
        source=source.replace('TRY_UNVERIFIED_SHUTDOWN_HELPER = False','TRY_UNVERIFIED_SHUTDOWN_HELPER = True',1)
        r.guard.write_state({'generation_complete':True,'archive_path':'run_new.zip'})
        ns={'r2v':r}
        with patch.object(r,'shutdown') as stop:
            exec(compile(source,'<stop-cell>','exec'),ns)
            stop.assert_not_called()
            self.assertFalse(ns['SAVED_TO_MY_DEVICE']);self.assertFalse(ns['STOP_AFTER_ERROR'])
            exec(compile(source,'<stop-cell>','exec'),ns)
            stop.assert_called_once_with(saved=True,stop_after_error=True)

    def test_preflight_failure_blocks_model_download_and_generation(self):
        def fail_preflight(command,label,**kwargs):
            if label=='r2v_api_preflight':raise r.WorkflowError('API mismatch')
        with patch.object(r.guard,'require_cloud'),patch.object(r,'initialize_budget'),patch.object(r.guard,'deadline_remaining',return_value=1000.), \
             patch.object(r.guard,'inspect_resources',return_value={}),patch.object(r.guard,'boot_id',return_value='mock-boot'), \
             patch.object(r.guard,'SCRATCH',self.root/'cache'),patch.object(r.guard,'run_waiting',side_effect=fail_preflight) as execute:
            with self.assertRaisesRegex(r.WorkflowError,'API mismatch'):r.setup(self.settings)
            self.assertTrue(r.guard.read_state()['setup_failed']);self.assertFalse(r.guard.read_state().get('setup_complete'))
            self.assertNotIn('download_ref_weights',[call.args[1] for call in execute.call_args_list])
            execute.reset_mock()
            with self.assertRaisesRegex(r.WorkflowError,'SETUP_OK'):r.generate(self.settings)
            execute.assert_not_called()

    def test_changed_environment_lock_blocks_a_new_gpu_attempt(self):
        (self.root/'requirements-r2v.lock').write_text('modified')
        r.guard.write_state({'setup_complete':True,'kit_digest':r.kit_digest(),'requirements_sha256':'0'*64})
        with patch.object(r.guard,'require_cloud'),patch.object(r,'initialize_budget'),patch.object(r.guard,'deadline_remaining',return_value=1000.),patch.object(r.guard,'run_waiting') as execute:
            with self.assertRaisesRegex(r.WorkflowError,'変更'):r.generate(self.settings)
            execute.assert_not_called();self.assertFalse(r.guard.read_state().get('generation_attempted'))

    def test_decode_failure_stops_before_720_export_or_packaging(self):
        lock=self.root/'requirements-r2v.lock';lock.write_text('mock-lock')
        (self.root/'download_report.json').write_text('{}')
        r.guard.write_state({'setup_complete':True,'kit_digest':r.kit_digest(),
                            'requirements_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),
                            'cu_remaining_at_start':100.})
        def fail_decode(command,label,**kwargs):
            if label=='decode_native':raise r.WorkflowError('Corrupt frame')
        info={'streams':[{'codec_type':'video','width':1280,'height':736,'nb_frames':'124','avg_frame_rate':'24/1'},
                         {'codec_type':'audio'}],'format':{'duration':'5.175'}}
        with patch.object(r.guard,'require_cloud'),patch.object(r,'initialize_budget'),patch.object(r.guard,'deadline_remaining',return_value=1000.), \
             patch.object(r.guard,'inspect_resources'),patch.object(r.guard,'boot_id',return_value='mock-boot'), \
             patch.object(r,'probe',return_value=info),patch.object(r.guard,'run_waiting',side_effect=fail_decode) as execute:
            with self.assertRaisesRegex(r.WorkflowError,'Corrupt frame'):r.generate(self.settings)
            self.assertTrue(r.guard.read_state()['generation_failed']);self.assertFalse(r.guard.read_state()['generation_complete'])
            self.assertNotIn('export_exact720',[c.args[1] for c in execute.call_args_list])
            self.assertFalse(list(self.root.glob('*.zip')))
            execute.reset_mock()
            with self.assertRaisesRegex(r.WorkflowError,'再開'):r.generate(self.settings)
            execute.assert_not_called()


if __name__=='__main__':unittest.main()
