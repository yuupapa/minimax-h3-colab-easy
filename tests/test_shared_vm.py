"""Same-VM setup reuse, mode/image/output binding and one total budget."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
import fl720_workflow as f
import build_fl720_notebook as builder
import test_fl720 as fixtures


class SharedVMChecks(unittest.TestCase):
    def setUp(self):
        fixtures.FL720Checks.setUp(self)
        self.extra=[patch.object(f.guard,'require_cloud'),patch.object(f.time,'time',return_value=10000.),
                    patch.object(f.guard.threading,'Timer')]
        for p in self.extra:p.start()
        f.guard.write_state({})
        f.initialize_budget(self.settings)
        f.activate_mode(self.settings)
        f.select_request(self.settings)

    def tearDown(self):
        for p in reversed(self.extra):p.stop()
        fixtures.FL720Checks.tearDown(self)

    i2v=fixtures.FL720Checks.i2v

    def complete_first(self, saved=False):
        out=self.root/'outputs'/'run_t2v_fixture';out.mkdir(parents=True)
        (out/'delivery_720p.mp4').write_bytes(b'control-plane fixture; no video generation')
        archive=self.root/'run_t2v_fixture.zip';archive.write_bytes(b'control-plane fixture')
        state=f.guard.read_state()
        state['runs']={'t2v':{'attempted':True,'complete':True,'failed':False,'saved_confirmed':saved,
                             'run_id':'t2v_fixture','fingerprint':f.fingerprint(self.settings),
                             'output_dir':str(out),'archive_path':str(archive)}}
        queue=state['generation_queues'][state['active_queue_id']]
        queue['jobs'][0].update(status='complete',result=state['runs']['t2v']);queue['status']='complete'
        f.guard.write_state(state)
        return state['runs']['t2v']

    def switch_and_select_i2v(self):
        s=self.i2v()
        f.initialize_budget(s)
        self.assertTrue(f.activate_mode(s))
        f.confirm_mode(s)
        state=f.guard.read_state();state['last_input_image']=self.frame;f.guard.write_state(state)
        f.select_request(s)
        return s

    def test_one_session_has_one_budget_and_two_mode_slots(self):
        state=f.guard.read_state()
        self.assertEqual(state['cu_budget'],5.)
        self.assertEqual(state['approved_modes'],['i2v','t2v'])
        self.assertEqual(state['active_mode'],'t2v')
        self.assertNotIn('runs',state)

    def test_selected_modes_can_change_without_extending_budget(self):
        before=f.guard.read_state()
        f.initialize_budget(dict(self.settings,approved_modes=['t2v']))
        after=f.guard.read_state()
        self.assertEqual(before['deadline_epoch'],after['deadline_epoch'])
        self.assertEqual(after['approved_modes'],['t2v'])

    def test_unauthorized_second_mode_cannot_activate(self):
        s=dict(self.i2v(),approved_modes=['t2v'])
        before=f.guard.read_state()
        with self.assertRaisesRegex(f.WorkflowError,'未承認'):f.activate_mode(s)
        self.assertEqual(before,f.guard.read_state())

    def test_changing_to_a_new_per_mode_budget_is_rejected(self):
        old=Mock();before=f.guard.read_state()
        with self.assertRaisesRegex(f.WorkflowError,'CU上限'):
            f.arm_budget(dict(self.i2v(),cu_budget=4.),previous_timer=old)
        old.cancel.assert_not_called()
        self.assertEqual(before['deadline_epoch'],f.guard.read_state()['deadline_epoch'])

    def test_mode_switch_and_lower_rate_do_not_extend_either_deadline(self):
        self.complete_first(saved=True)
        before=f.guard.read_state()
        with patch.object(f.time,'time',return_value=10500.):
            f.initialize_budget(dict(self.i2v(),cu_rate=3.,cu_remaining=97.))
            f.activate_mode(self.i2v())
        after=f.guard.read_state()
        for key in ('session_started_epoch','deadline_epoch','shutdown_deadline_epoch','cu_budget'):
            self.assertEqual(before[key],after[key])

    def test_switch_preserves_unsaved_completed_output(self):
        first=self.complete_first(saved=False)
        self.assertTrue(f.activate_mode(self.i2v()))
        self.assertEqual(f.guard.read_state()['runs']['t2v'],first)
        self.assertTrue(Path(first['archive_path']).is_file())

    def test_wrong_save_id_or_archive_cannot_open_mode_switch(self):
        first=self.complete_first()
        for rid,path in [('i2v_old',first['archive_path']),('t2v_fixture','wrong.zip')]:
            with self.subTest(rid=rid,path=path),self.assertRaisesRegex(f.WorkflowError,'保存確認の対象'):
                f.confirm_saved(self.settings,run_id=rid,archive_path=path)
        self.assertFalse(f.guard.read_state()['generation_queues'][f.guard.read_state()['active_queue_id']]['saved_confirmed'])

    def test_switch_erases_old_image_and_request_but_preserves_saved_output(self):
        first=self.complete_first(saved=True)
        state=f.guard.read_state();state['last_input_image']=self.frame;f.guard.write_state(state)
        self.assertTrue(f.activate_mode(self.i2v()))
        state=f.guard.read_state()
        self.assertEqual(state['active_mode'],'i2v')
        self.assertIsNone(state['active_fingerprint'])
        self.assertFalse(state['mode_reviewed'])
        self.assertNotIn('last_input_image',state)
        self.assertEqual(state['runs']['t2v'],first)

    def test_stale_previous_settings_cannot_preview_save_or_generate_new_mode(self):
        self.complete_first(saved=True);f.activate_mode(self.i2v())
        with patch.object(f.guard,'run_waiting') as run:
            for fn in (f.current_output,f.generate):
                with self.subTest(fn=fn.__name__),self.assertRaisesRegex(f.WorkflowError,'現在の入力'):
                    fn(self.settings)
        run.assert_not_called()

    def test_i2v_requires_mode_review_and_a_new_current_image(self):
        self.complete_first(saved=True);f.activate_mode(self.i2v())
        with patch.object(f.guard,'upload_first_frame') as upload:
            with self.assertRaisesRegex(f.WorkflowError,'切替'):f.upload_first_frame(self.i2v())
        upload.assert_not_called()
        with self.assertRaisesRegex(f.WorkflowError,'現在のモード'):f.select_request(self.i2v())
        f.confirm_mode(self.i2v())
        with self.assertRaisesRegex(f.WorkflowError,'新しい始点画像'):f.select_request(self.i2v())

    def test_mode_pause_ignores_stale_checkbox_and_invalidates_notebook_pointers(self):
        self.complete_first(saved=True)
        state=f.guard.read_state();state['last_input_image']=self.frame;f.guard.write_state(state)
        ns=self.input_namespace()
        ns.update(settings=self.settings,archive_path='previous.zip',_fl720_save_gate_target=('t2v','old','old.zip'))
        tail='settings=None\narchive_path=None\ncandidate='+builder.BOOT.split('candidate=',1)[1]
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'モード切替'):exec(compile(tail,'<switch>','exec'),ns)
        self.assertIsNone(ns['settings']);self.assertIsNone(ns['archive_path'])
        self.assertFalse(ns['MODE_SWITCH_CONFIRMED']);self.assertFalse(ns['IMAGE_CONTENT_CONFIRMED'])
        self.assertFalse(ns['UPLOAD_FIRST_FRAME']);self.assertIsNone(ns['_fl720_save_gate_target'])
        self.assertNotIn('last_input_image',f.guard.read_state())

    def input_namespace(self):
        return {'fl720':f,'Path':Path,'settings':None,'GPU_TRIAL_APPROVED':True,
                'T2V_TRIAL_APPROVED':True,'I2V_TRIAL_APPROVED':True,'MODE_SWITCH_CONFIRMED':True,
                'SOURCE_EXECUTION_ACCEPTED':True,'LICENSE_ACCEPTED':True,'ELIGIBLE_USE_CONFIRMED':True,
                'HIGHER_RESOLUTION_RISK_ACCEPTED':True,'GENERATION_MODE':'Image-to-Video',
                'IMAGE_USE_CONFIRMED':True,'IMAGE_CONTENT_CONFIRMED':True,'UPLOAD_FIRST_FRAME':True,
                'json':json,'VIDEO_COUNT':1,'PROMPTS_JSON':'','NEW_RUN':False,'RESUME_FAILED_QUEUE':False,'PROMPT':'A boat slowly moves.','SEED':42,'CU_REMAINING':97.,'CU_PER_HOUR':6.77,
                'CU_BUDGET':5.,'GPU_CONNECTED_MINUTES_AGO':0.}

    def test_switch_review_does_not_silently_reuse_old_png(self):
        self.complete_first(saved=True);f.activate_mode(self.i2v());f.confirm_mode(self.i2v())
        ns=self.input_namespace();ns['UPLOAD_FIRST_FRAME']=False
        tail='candidate='+builder.BOOT.split('candidate=',1)[1]
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'UPLOAD_FIRST_FRAME'):exec(compile(tail,'<no-image>','exec'),ns)
        self.assertIsNone(ns['settings'])
        self.assertIsNone(f.guard.read_state()['active_fingerprint'])

    def test_save_cell_resets_stale_ack_then_marks_exact_mode_and_run(self):
        first=self.complete_first()
        download=Mock();files=types.ModuleType('google.colab.files');files.download=download
        colab=types.ModuleType('google.colab');colab.files=files
        ns={'fl720':f,'settings':self.settings}
        source=builder.SAVE.replace('SAVED_CURRENT_RUN = False','SAVED_CURRENT_RUN = True',1)
        with patch.dict(sys.modules,{'google.colab':colab,'google.colab.files':files}),contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source,'<save>','exec'),ns)
            self.assertFalse(ns['SAVED_CURRENT_RUN'])
            self.assertFalse(f.guard.read_state()['generation_queues'][f.guard.read_state()['active_queue_id']]['saved_confirmed'])
            exec(compile(source,'<save>','exec'),ns)
        download.assert_called_once_with(first['archive_path'])
        self.assertTrue(f.guard.read_state()['generation_queues'][f.guard.read_state()['active_queue_id']]['saved_confirmed'])

    def test_save_cell_cannot_download_previous_mode_after_switch(self):
        self.complete_first(saved=True);f.activate_mode(self.i2v())
        download=Mock();colab=types.ModuleType('google.colab');colab.files=types.SimpleNamespace(download=download)
        with patch.dict(sys.modules,{'google.colab':colab}),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(f.WorkflowError):
                exec(compile(builder.SAVE,'<save-stale>','exec'),{'fl720':f,'settings':self.settings})
        download.assert_not_called()

    def test_processing_expiry_blocks_new_mode_but_allows_bound_save_ack(self):
        first=self.complete_first()
        deadline=f.guard.read_state()['deadline_epoch']
        with patch.object(f.time,'time',return_value=deadline+1):
            f.confirm_saved(self.settings,run_id=first['run_id'],archive_path=first['archive_path'])
            with self.assertRaises(f.guard.BudgetExceeded):f.activate_mode(self.i2v())
        self.assertEqual(f.guard.read_state()['active_mode'],'t2v')

    def test_failure_aborts_other_mode_without_destroying_completed_run(self):
        first=self.complete_first(saved=True)
        state=f.guard.read_state();state['session_failed']=True;f.guard.write_state(state)
        self.assertTrue(f.activate_mode(self.i2v()))
        self.assertEqual(f.guard.read_state()['runs']['t2v'],first)

    def test_one_shared_setup_and_download_for_two_serial_mode_generations(self):
        labels=[]
        def phase(command,label,**kwargs):
            labels.append(label)
            if label=='download_fl720_weights':
                (self.root/'download_report.json').write_text(json.dumps({'model_path':'fixture','lora_path':'fixture'}))
            elif label=='fl720_generation':
                state=f.guard.read_state();entry=state['runs'][state['active_mode']];out=Path(entry['output_dir'])
                (out/'native.mp4').write_bytes(b'orchestration fixture; no video generation')
                (out/'metrics.json').write_text('{}')
            elif label=='export_exact720':
                Path(command[-1]).write_bytes(b'export orchestration fixture; no real ffmpeg')
        def probe(path):
            return {'streams':[{'codec_type':'video','width':1280,'height':736 if path.name=='native.mp4' else 720,
                                'nb_frames':'124','avg_frame_rate':'24/1'},{'codec_type':'audio'}],
                    'format':{'duration':'5.175'}}
        with patch.object(f,'inspect_resources',return_value={}),patch.object(f.guard,'boot_id',return_value='offline'), \
             patch.object(f.guard,'run_waiting',side_effect=phase),patch.object(f,'probe',side_effect=probe),contextlib.redirect_stdout(io.StringIO()):
            f.setup(self.settings);first_archive=f.generate(self.settings)
            first=f.current_output(self.settings)
            f.confirm_saved(self.settings,run_id=first['run_id'],archive_path=first['archive_path'])
            before=f.guard.read_state()['deadline_epoch']
            second=self.switch_and_select_i2v()
            f.setup(second);second_archive=f.generate(second)
            end=f.current_output(second)
        for label in ('install_uv','create_fl720_venv','sync_fl720_environment','fl720_api_preflight','download_fl720_weights'):
            self.assertEqual(labels.count(label),1,label)
        self.assertEqual(labels.count('fl720_generation'),2)
        self.assertNotEqual(first_archive,second_archive)
        self.assertEqual(before,f.guard.read_state()['deadline_epoch'])
        self.assertEqual(set(f.guard.read_state()['runs']),{'t2v','i2v'})
        self.assertFalse(end['saved_confirmed'])
        for archive,mode in ((first_archive,'t2v'),(second_archive,'i2v')):
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(json.loads(z.read('metrics.json'))['mode'],mode)
                self.assertFalse(any(n.endswith('.png') for n in z.namelist()))

    def test_failed_or_changed_request_in_a_consumed_mode_cannot_retry(self):
        self.complete_first(saved=True)
        for updated in (dict(self.settings,prompts=['A different shot']),dict(self.settings,seed=43)):
            before=f.guard.read_state()['active_queue_id']
            f.select_request(updated)
            self.assertNotEqual(before,f.guard.read_state()['active_queue_id'])

    def test_original_one_attempt_candidate_state_is_not_migrated_into_new_slots(self):
        f.guard.write_state({'generation_attempted':True,'generation_complete':True})
        with self.assertRaisesRegex(f.WorkflowError,'旧候補'):f.initialize_budget(self.settings)

    def test_invalid_modes_fail_cleanly_before_scope_or_timer_mutation(self):
        for approved in ([],[['t2v']],['t2v','t2v'],['r2v'],None):
            before=f.guard.read_state();old=Mock()
            with self.subTest(approved=approved),self.assertRaises(f.WorkflowError):
                f.arm_budget(dict(self.settings,approved_modes=approved),previous_timer=old)
            old.cancel.assert_not_called();self.assertEqual(before,f.guard.read_state())


if __name__=='__main__':unittest.main()
