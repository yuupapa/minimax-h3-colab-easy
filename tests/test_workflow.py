import importlib.util
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import struct
import zlib
import unittest
from unittest.mock import patch,Mock
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
import colab_workflow as h3
from validate_distribution import validate


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name,value in [('ROOT',self.root),('STATE',self.root/'state.json'),('SCRATCH',self.root/'scratch'),('VENV',self.root/'venv')]:
            p=patch.object(h3,name,value);p.start();self.addCleanup(p.stop)
        self.s={'license_accepted':True,'eligible_use_confirmed':True,'prompts':['A cat walks across a porch.'],'count':1,'seconds':5,'seed':42,'cu_remaining':10,'cu_rate':6.77,'cu_budget':3}

    def test_distribution_privacy_and_embedded_code(self):
        self.assertEqual(validate()['pinned_dependencies'],234)

    def test_mac_cloud_guard_precedes_any_subprocess(self):
        with patch.object(h3.platform,'system',return_value='Darwin'),patch.object(h3.subprocess,'Popen') as start:
            with self.assertRaises(h3.WorkflowError):h3.setup(self.s)
            with self.assertRaises(h3.WorkflowError):h3.generate(self.s)
            start.assert_not_called()
        self.assertFalse(h3.STATE.exists())

    def test_human_consent_required(self):
        for key in ['license_accepted','eligible_use_confirmed']:
            with self.subTest(key=key):
                with self.assertRaises(h3.WorkflowError):h3.validate_inputs(dict(self.s,**{key:False}))

    def test_inputs_and_budget_reject_unapproved_or_invalid_work(self):
        cases=[{'count':40},{'count':2.0},{'seconds':10},{'prompts':['']},{'seed':-1},{'cu_budget':11},{'cu_remaining':0},{'cu_rate':float('nan')},{'cu_rate':True},{'cu_budget':6},{'count':2}]
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(h3.WorkflowError):h3.validate_inputs(dict(self.s,**changes))

    def test_two_six_second_requests_are_allowed_but_not_executed(self):
        value=dict(self.s,count=2,seconds=6,prompts=['Cat','Boat'])
        self.assertEqual(h3.validate_inputs(value)['count'],2)

    def frame_fixture(self, data=b'normalized image bytes'):
        def chunk(kind,value):
            return struct.pack('>I',len(value))+kind+value+struct.pack('>I',zlib.crc32(kind+value)&0xffffffff)
        color=hashlib.sha256(data).digest()[:3]
        data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,64,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\x00'+color*64)*64))+chunk(b'IEND',b'')
        digest=hashlib.sha256(data).hexdigest()
        p=self.root/'inputs'/(digest+'.png');p.parent.mkdir(exist_ok=True);p.write_bytes(data)
        return {'path':str(p),'sha256':digest}

    def test_t2v_fingerprint_and_native_sampling_remain_unchanged(self):
        value={k:self.s[k] for k in ('prompts','count','seconds','seed')}
        original=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        self.assertEqual(h3.fingerprint(self.s),original)
        self.assertEqual(h3.fingerprint(dict(self.s,mode='t2v')),original)

    def test_i2v_requires_both_human_confirmations_and_one_output(self):
        s=dict(self.s,mode='i2v',first_frame=self.frame_fixture(),image_use_confirmed=True,image_content_confirmed=True)
        self.assertEqual(h3.validate_inputs(s)['mode'],'i2v')
        for changes in [{'image_use_confirmed':False},{'image_content_confirmed':False},{'count':2,'prompts':['Cat','Boat']}]:
            with self.assertRaises(h3.WorkflowError):h3.validate_inputs(dict(s,**changes))

    def test_i2v_changed_image_changes_fingerprint_and_tampering_is_rejected(self):
        s=dict(self.s,mode='i2v',first_frame=self.frame_fixture(),image_use_confirmed=True,image_content_confirmed=True)
        changed=dict(s,first_frame=self.frame_fixture(b'new image'))
        self.assertNotEqual(h3.fingerprint(s),h3.fingerprint(changed))
        Path(s['first_frame']['path']).write_bytes(b'tampered')
        with self.assertRaises(h3.WorkflowError):h3.validate_inputs(s)
        with self.assertRaises(h3.WorkflowError):h3.fingerprint(s)

    def test_image_urls_arbitrary_files_symlinks_and_r2v_fail_before_subprocess(self):
        frame=self.frame_fixture()
        outside=self.root/'private.png';outside.write_bytes(b'secret')
        for path in ['https://example.com/image.png',str(outside),'file:///tmp/image.png']:
            with self.assertRaises(h3.WorkflowError):h3.verify_first_frame(dict(frame,path=path))
        p=Path(frame['path']);p.unlink();p.symlink_to(outside)
        with self.assertRaises(h3.WorkflowError):h3.verify_first_frame(frame)
        with patch.object(h3,'require_cloud'),patch.object(h3.subprocess,'Popen') as run:
            for action in (h3.setup,h3.generate):
                with self.assertRaises(h3.WorkflowError):action(dict(self.s,mode='r2v'))
            run.assert_not_called()

    def test_image_cloud_and_consent_checks_precede_upload_or_decoder(self):
        with patch.object(h3.platform,'system',return_value='Darwin'):
            with self.assertRaises(h3.WorkflowError):h3.upload_first_frame(image_use_confirmed=True)
            with self.assertRaises(h3.WorkflowError):h3.prepare_first_frame('/tmp/image.png',image_use_confirmed=True)
        with patch.object(h3,'require_cloud'):
            with self.assertRaises(h3.WorkflowError):h3.upload_first_frame(image_use_confirmed=False)
            with self.assertRaises(h3.WorkflowError):h3.prepare_first_frame('/tmp/image.png',image_use_confirmed=True)
            p=self.root/'uploads'/'test';p.parent.mkdir();p.write_bytes(b'')
            with self.assertRaises(h3.WorkflowError):h3.prepare_first_frame(p,image_use_confirmed=True)

    def test_native_i2v_routes_first_keyframe_without_changing_t2v_parameters(self):
        tree=ast.parse((ROOT/'scripts/generate_native.py').read_text())
        sampling=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='sampling' for x in n.targets))
        route=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.value.id=='sampling' for c in ast.walk(n)))
        code=compile(ast.fix_missing_locations(ast.Module(body=[sampling,route],type_ignores=[])),'native-request','exec')
        base={'prompt':'Move slowly','inputs':{'duration_seconds':5,'seeds':[42]},'i':1,'name':'clip.mp4','out':self.root,'first_frame':Path(self.frame_fixture()['path'])}
        a=dict(base,mode='t2v');exec(code,a)
        b=dict(base,mode='i2v');exec(code,b)
        self.assertEqual(a['sampling']['task'],'t2va');self.assertEqual(a['sampling']['conditions'],[])
        self.assertEqual(b['sampling']['task'],'fl2va')
        self.assertEqual(b['sampling']['conditions'],[{'type':'image','uri':base['first_frame'].as_uri(),'role':'keyframe','frame_index':0}])
        for key in set(a['sampling'])-{'task','conditions'}:self.assertEqual(a['sampling'][key],b['sampling'][key])

    def test_notebook_new_image_always_stops_even_with_stale_confirmation(self):
        nb=json.loads((ROOT/'MiniMax_H3_Easy_Colab.ipynb').read_text())
        first=''.join(nb['cells'][1]['source'])
        code='settings=None\n'+first[first.index('_base_settings='):]
        frame=self.frame_fixture()
        fake=SimpleNamespace(validate_inputs=h3.validate_inputs,schedule_budget_shutdown=lambda *a,**k:None,deadline_remaining=lambda:100,upload_first_frame=lambda **k:frame,ROOT=self.root,read_state=lambda:{'last_input_image':frame})
        env={'Path':Path,'h3':fake,'settings':self.s,'LICENSE_ACCEPTED':True,'ELIGIBLE_USE_CONFIRMED':True,'PROMPT':self.s['prompts'][0],'SECOND_PROMPT':'','VIDEO_COUNT':1,'DURATION_SECONDS':5,'SEED':42,'CU_REMAINING':10,'CU_PER_HOUR':6.77,'CU_BUDGET':3,'GENERATION_MODE':'Image-to-Video','IMAGE_USE_CONFIRMED':True,'IMAGE_CONTENT_CONFIRMED':True,'UPLOAD_FIRST_FRAME':True,'FIRST_FRAME_PATH':''}
        display_module=SimpleNamespace(Image=lambda **k:None,display=lambda *a:None)
        with patch.dict(sys.modules,{'IPython':SimpleNamespace(),'IPython.display':display_module}):
            with self.assertRaisesRegex(RuntimeError,'始点画像を確認'):exec(code,env)
            self.assertIsNone(env['settings'])
            self.assertFalse(env['UPLOAD_FIRST_FRAME']);self.assertFalse(env['IMAGE_CONTENT_CONFIRMED'])
            env['IMAGE_CONTENT_CONFIRMED']=True
            exec(code,env)
            self.assertEqual(env['settings']['mode'],'i2v')
            # A failed replacement upload must also consume old image consent.
            env.update(UPLOAD_FIRST_FRAME=True,IMAGE_CONTENT_CONFIRMED=True)
            fake.upload_first_frame=Mock(side_effect=RuntimeError('upload failed'))
            with self.assertRaisesRegex(RuntimeError,'upload failed'):exec(code,env)
            self.assertIsNone(env['settings'])
            self.assertFalse(env['IMAGE_CONTENT_CONFIRMED']);self.assertEqual(env['FIRST_FRAME_PATH'],'')

    def test_budget_reentry_does_not_reset_or_extend(self):
        with patch.object(h3,'require_cloud'),patch.object(h3.time,'time',return_value=100):
            first=h3.initialize_budget(self.s)
        with patch.object(h3,'require_cloud'),patch.object(h3.time,'time',return_value=200):
            second=h3.initialize_budget(self.s)
            self.assertEqual(first['deadline_epoch'],second['deadline_epoch'])
            lower=h3.initialize_budget(dict(self.s,cu_rate=5))
            self.assertEqual(first['deadline_epoch'],lower['deadline_epoch'])
            higher=h3.initialize_budget(dict(self.s,cu_rate=10))
            self.assertLess(higher['deadline_epoch'],first['deadline_epoch'])
            with self.assertRaises(h3.WorkflowError):h3.initialize_budget(dict(self.s,cu_budget=4))

    def test_expired_budget_refuses_restart(self):
        h3.write_state({'deadline_epoch':10})
        with patch.object(h3.time,'time',return_value=20):
            with self.assertRaises(h3.BudgetExceeded):h3.deadline_remaining()

    def test_budget_timer_is_cleanup_only_and_cancelable(self):
        with patch.object(h3,'require_cloud'),patch.object(h3.threading,'Timer') as timer:
            h3.schedule_budget_shutdown(self.s)
            timer.return_value.start.assert_called_once()
            self.assertTrue(timer.return_value.daemon)
            self.assertEqual(len(timer.call_args.args),2)
            timer.return_value.cancel()

    def test_expired_processing_window_still_arms_cleanup_before_replacing_old_timer(self):
        with patch.object(h3,'require_cloud'),patch.object(h3.time,'time',return_value=100):
            first=h3.initialize_budget(self.s)
        previous=Mock();order=Mock()
        with patch.object(h3,'require_cloud'),patch.object(h3.time,'time',return_value=first['deadline_epoch']+1),patch.object(h3.threading,'Timer') as timer:
            order.attach_mock(timer.return_value,'replacement');order.attach_mock(previous,'previous')
            h3.schedule_budget_shutdown(self.s,previous_timer=previous)
            self.assertGreater(timer.call_args.args[0],0)
            self.assertLess(timer.call_args.args[0],first['shutdown_deadline_epoch']-first['deadline_epoch'])
            self.assertEqual([x[0] for x in order.mock_calls],['replacement.start','previous.cancel'])
            with self.assertRaises(h3.BudgetExceeded):h3.deadline_remaining()

    def test_budget_change_preserves_previous_cleanup_timer(self):
        with patch.object(h3,'require_cloud'),patch.object(h3.time,'time',return_value=100):h3.initialize_budget(self.s)
        previous=Mock()
        with patch.object(h3,'require_cloud'),patch.object(h3.time,'time',return_value=101),patch.object(h3.threading,'Timer') as timer:
            with self.assertRaises(h3.WorkflowError):h3.schedule_budget_shutdown(dict(self.s,cu_budget=4),previous_timer=previous)
            previous.cancel.assert_not_called();timer.assert_not_called()

    def test_timer_start_failure_preserves_previous_cleanup_timer(self):
        previous=Mock()
        with patch.object(h3,'require_cloud'),patch.object(h3.threading,'Timer') as timer:
            timer.return_value.start.side_effect=RuntimeError('thread failure')
            with self.assertRaises(RuntimeError):h3.schedule_budget_shutdown(self.s,previous_timer=previous)
            previous.cancel.assert_not_called()

    def test_past_cleanup_deadline_arms_immediate_shutdown(self):
        with patch.object(h3,'require_cloud'),patch.object(h3.time,'time',return_value=100):first=h3.initialize_budget(self.s)
        with patch.object(h3,'require_cloud'),patch.object(h3.time,'time',return_value=first['shutdown_deadline_epoch']+1),patch.object(h3.threading,'Timer') as timer:
            h3.schedule_budget_shutdown(self.s)
            self.assertEqual(timer.call_args.args[0],0)
            timer.return_value.start.assert_called_once()

    def test_budget_expiry_terminates_foreground_process_group(self):
        h3.write_state({'deadline_epoch':10});process=Mock(pid=123,poll=Mock(return_value=None))
        with patch.object(h3.subprocess,'Popen',return_value=process),patch.object(h3.time,'time',return_value=20),patch.object(h3.os,'killpg') as kill:
            with self.assertRaises(h3.BudgetExceeded):h3.run_waiting(['unused'],'test: stage')
            kill.assert_called_once_with(123,h3.signal.SIGTERM)
            process.wait.assert_called_once_with(timeout=10)

    def test_keyboard_interruption_also_reaps_subprocess(self):
        h3.write_state({'deadline_epoch':999999999999});process=Mock(pid=123,poll=Mock(return_value=None))
        with patch.object(h3.subprocess,'Popen',return_value=process),patch.object(h3.time,'sleep',side_effect=KeyboardInterrupt),patch.object(h3.os,'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):h3.run_waiting(['unused'],'test: stage')
            kill.assert_called_once()

    def test_setup_waits_all_phases_then_sets_ready(self):
        with patch.object(h3,'require_cloud'),patch.object(h3,'inspect_resources',return_value={'gpu':'test'}),patch.object(h3,'run_waiting') as wait:
            state=h3.setup(self.s)
            commands=[call.args[0] for call in wait.call_args_list]
            self.assertEqual(len(commands),5)
            self.assertIn('uv==0.13.0',commands[0])
            self.assertIn('--build-constraint',commands[2])
            self.assertEqual(commands[-1][-1],str(h3.ROOT/'download.py'))
            self.assertTrue(state['setup_complete'])
            self.assertIn('token=False',(h3.ROOT/'download.py').read_text())
            compile((h3.ROOT/'download.py').read_text(),'download','exec')

    def test_failed_setup_never_marks_ready(self):
        with patch.object(h3,'require_cloud'),patch.object(h3,'inspect_resources',return_value={}),patch.object(h3,'run_waiting',side_effect=h3.WorkflowError('install failed')):
            with self.assertRaises(h3.WorkflowError):h3.setup(self.s)
            self.assertFalse(h3.read_state().get('setup_complete',False))

    def test_previous_ready_is_invalidated_on_resource_or_install_failure(self):
        for stage in ['resources','install']:
            h3.write_state({'setup_complete':True})
            with self.subTest(stage=stage),patch.object(h3,'require_cloud'),patch.object(h3,'inspect_resources',side_effect=h3.WorkflowError('resource failure') if stage=='resources' else None,return_value={}),patch.object(h3,'run_waiting',side_effect=h3.WorkflowError('install failure')):
                with self.assertRaises(h3.WorkflowError):h3.setup(self.s)
                self.assertFalse(h3.read_state()['setup_complete'])

    def test_changed_dependency_digest_blocks_new_generation(self):
        h3.write_state({'setup_complete':True,'requirements_digest':'old'})
        with patch.object(h3,'require_cloud'),patch.object(h3,'run_waiting') as wait:
            with self.assertRaisesRegex(h3.WorkflowError,'固定依存'):h3.generate(self.s)
            wait.assert_not_called()

    def test_stale_save_checkbox_cannot_stop_first_run_all(self):
        nb=json.loads((ROOT/'MiniMax_H3_Easy_Colab.ipynb').read_text())
        stop=''.join(nb['cells'][5]['source']).replace('SAVED_TO_MY_DEVICE = False','SAVED_TO_MY_DEVICE = True')
        fake=Mock();scope={'_h3_stop_gate_open':False}
        with patch.dict(sys.modules,{'google':Mock(),'google.colab':fake}):
            exec(stop,scope)
            fake.runtime.unassign.assert_not_called()
            exec(stop,scope)
            fake.runtime.unassign.assert_called_once()

    def test_generation_reclick_does_not_reuse_new_run_permission_even_after_failure(self):
        nb=json.loads((ROOT/'MiniMax_H3_Easy_Colab.ipynb').read_text());tree=ast.parse(''.join(nb['cells'][3]['source']))
        prefix=[]
        for node in tree.body:
            if isinstance(node,ast.Import):break
            prefix.append(node)
        code=compile(ast.Module(body=prefix,type_ignores=[]),'generation-cell','exec')
        for failed in [False,True]:
            fake=Mock();fake.generate.return_value=self.root
            if failed:fake.generate.side_effect=h3.WorkflowError('failed generation')
            scope={'NEW_RUN':True,'settings':self.s,'h3':fake}
            for unused in range(2):
                if failed:
                    with self.assertRaises(h3.WorkflowError):exec(code,scope)
                else:exec(code,scope)
            self.assertEqual([c.kwargs['new_run'] for c in fake.generate.call_args_list],[True,False])
            self.assertFalse(scope['NEW_RUN'])

    def test_resource_guard_accounts_for_shared_disk_capacity_and_cuda_driver(self):
        stat=h3.Path.stat;read=h3.Path.read_text
        device=h3.SCRATCH.parent.stat().st_dev
        mem='MemTotal: 175164620 kB\nMemAvailable: 167772160 kB\n'
        cases=[(True,210,210,580,None),(True,195,195,580,'205GiB'),(False,181,26,580,None),(False,179,26,580,'180GiB'),(False,190,24,580,'25GiB'),(False,190,26,579,'ドライバ')]
        for shared,cache_free,content_free,driver,error in cases:
            with self.subTest(shared=shared,cache=cache_free,content=content_free,driver=driver),patch.object(h3,'require_cloud'),patch.object(h3,'read_state',return_value={}),patch.object(h3.platform,'machine',return_value='x86_64'),patch.object(h3.shutil,'which',return_value='/usr/bin/tool'),patch.object(h3.subprocess,'run',return_value=SimpleNamespace(stdout=f'A100-SXM4-80GB, 81920, 81920, {driver}.0\n')),patch.object(h3.Path,'read_text',lambda p,*a,**k:mem if str(p)=='/proc/meminfo' else read(p,*a,**k)),patch.object(h3.Path,'stat',lambda p,*a,**k:SimpleNamespace(st_dev=device if shared else device+1) if str(p)=='/content' else stat(p,*a,**k)),patch.object(h3.shutil,'disk_usage',side_effect=lambda p:SimpleNamespace(free=(content_free if str(p)=='/content' else cache_free)*1024**3)),patch.object(h3,'run_waiting') as wait:
                if error:
                    with self.assertRaisesRegex(h3.WorkflowError,error):h3.inspect_resources()
                else:
                    self.assertEqual(h3.inspect_resources()['shared_disk'],shared)
                wait.assert_not_called()

    def test_native_cli_refuses_mac_before_sglang_import(self):
        if sys.platform=='darwin':
            result=subprocess.run([sys.executable,str(ROOT/'scripts/generate_native.py')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('クラウドGPU専用',result.stderr)
            self.assertNotIn('No module named',result.stderr)

    def test_isolated_venv_child_requires_same_guarded_parent_and_vm(self):
        h3.write_state({'native_parent_pid':100,'cloud_boot_id':'same-live-vm'})
        with patch.object(h3.importlib.util,'find_spec',return_value=None),patch.object(h3.platform,'system',return_value='Linux'),patch.object(h3.Path,'is_dir',return_value=True),patch.object(h3,'boot_id',return_value='same-live-vm'),patch.object(h3.os,'getppid',return_value=100):
            h3.require_cloud(verified_child=True)
            with self.assertRaises(h3.WorkflowError):h3.require_cloud()
            with patch.object(h3.os,'getppid',return_value=101):
                with self.assertRaises(h3.WorkflowError):h3.require_cloud(verified_child=True)
            with patch.object(h3,'boot_id',return_value='other-vm'):
                with self.assertRaises(h3.WorkflowError):h3.require_cloud(verified_child=True)
            with patch.object(h3.platform,'system',return_value='Darwin'):
                with self.assertRaises(h3.WorkflowError):h3.require_cloud(verified_child=True)

    def test_repeat_completed_input_reuses_files_without_loading(self):
        out=self.root/'done';out.mkdir();(out/'video.mp4').write_bytes(b'test')
        (out/'sequence_report.json').write_text(json.dumps({'requests':[{'output':'video.mp4'}]}))
        h3.write_state({'setup_complete':True,'last_run':{'fingerprint':h3.fingerprint(self.s),'status':'completed','outdir':str(out)}})
        with patch.object(h3,'require_cloud'),patch.object(h3,'run_waiting') as wait:
            self.assertEqual(h3.generate(self.s),out);wait.assert_not_called()

    def test_partial_or_changed_input_requires_explicit_new_run(self):
        h3.write_state({'setup_complete':True,'last_run':{'fingerprint':h3.fingerprint(self.s),'status':'failed_or_interrupted','outdir':str(self.root)}})
        with patch.object(h3,'require_cloud'),patch.object(h3,'run_waiting') as wait:
            with self.assertRaises(h3.WorkflowError):h3.generate(self.s)
            with self.assertRaises(h3.WorkflowError):h3.generate(dict(self.s,prompts=['New boat']))
            wait.assert_not_called()

    def test_shutdown_does_not_delete_unconfirmed_outputs(self):
        with patch.object(h3,'require_cloud'):
            with self.assertRaises(h3.WorkflowError):h3.shutdown(False)


if __name__=='__main__':unittest.main()
