"""Exercise real controllers/journals, mocking only GPU/process operations."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'scripts'))
import fl720_workflow as fl
import r2v_workflow as ref
import serial_queue
import test_fl720 as fixtures


class ContinuousChecks(unittest.TestCase):
    def setUp(self):
        fixtures.FL720Checks.setUp(self)
        self.patches += [patch.object(fl.guard,'require_cloud'), patch.object(fl.time,'time',return_value=1791800000.),
                         patch.object(fl,'inspect_resources',return_value={}), patch.object(fl.guard,'boot_id',return_value='fixture')]
        for p in self.patches[4:]:p.start()
        fl.guard.write_state({})
        self.calls=[];self.fail_at=None;self.stop_after=None
        self.phase=patch.object(fl.guard,'run_waiting',side_effect=self.fake_phase);self.phase.start()
        self.probe=patch.object(fl,'probe',side_effect=self.fake_probe);self.probe.start()
        self.settings=self.choose(self.settings)
        fl.setup(self.settings)

    def tearDown(self):
        self.phase.stop();self.probe.stop()
        fixtures.FL720Checks.tearDown(self)

    def choose(self, settings):
        fl.initialize_budget(settings)
        if fl.activate_mode(settings):fl.confirm_mode(settings)
        if settings['mode']=='i2v':
            state=fl.guard.read_state();state['last_input_image']=settings['first_frame'];fl.guard.write_state(state)
        return fl.select_request(settings)

    def fake_phase(self, command, label, **kwargs):
        if label=='download_fl720_weights':
            (self.root/'download_report.json').write_text('{}')
        elif label=='fl720_generation':
            job=json.loads((self.root/'run.json').read_text());self.calls.append(job['settings'])
            if self.fail_at==len(self.calls):raise fl.WorkflowError('fixture failure')
            out=Path(job['output_dir']);(out/'native.mp4').write_bytes(b'no GPU/video generation')
            (out/'metrics.json').write_text('{}')
            if self.stop_after==len(self.calls):serial_queue.request_stop(fl.guard)
        elif label=='export_exact720':Path(command[-1]).write_bytes(b'no ffmpeg execution')

    def fake_probe(self, path):
        return {'streams':[{'codec_type':'video','width':1280,'height':736 if path.name=='native.mp4' else 720,
                            'nb_frames':'124','avg_frame_rate':'24/1'},{'codec_type':'audio'}],
                'format':{'duration':'5.175'}}

    def test_changed_prompt_seed_and_image_generate_again_preserving_outputs(self):
        first=fl.generate(self.settings)
        changed=self.choose(dict(self.settings,prompts=['Another shot'],seed=51))
        second=fl.generate(changed)
        image=self.choose(dict(changed,mode='i2v',first_frame=self.frame,image_use_confirmed=True,image_content_confirmed=True))
        third=fl.generate(image)
        data=Path(self.frame['path']).read_bytes()+b'new image fixture';digest=hashlib.sha256(data).hexdigest()
        path=self.root/'inputs'/(digest+'.png');path.write_bytes(data)
        image=self.choose(dict(image,first_frame=dict(self.frame,path=str(path),sha256=digest)))
        fourth=fl.generate(image)
        text=self.choose(dict(changed,mode='t2v'));fifth=fl.generate(text)
        self.assertEqual(len(self.calls),5)
        self.assertEqual(len({first,second,third,fourth,fifth}),5)
        self.assertTrue(all(Path(p).is_file() for p in (first,second,third,fourth,fifth)))

    def test_finite_queue_uses_exact_prompts_and_seeds_no_extra_call(self):
        s=self.choose(dict(self.settings,count=3,prompts=['One','Two','Three']))
        archive=fl.generate(s)
        self.assertEqual([x['prompts'][0] for x in self.calls],['One','Two','Three'])
        self.assertEqual([x['seed'] for x in self.calls],[42,43,44])
        self.assertTrue(all(x['count']==1 for x in self.calls))
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip());self.assertEqual(json.loads(z.read('queue_manifest.json'))['completed_count'],3)
            self.assertEqual(sum(n.endswith('delivery_720p.mp4') for n in z.namelist()),3)
            self.assertFalse(any(n.endswith('.png') for n in z.namelist()))
        self.assertEqual(fl.generate(s),archive);self.assertEqual(len(self.calls),3)

    def test_same_input_new_run_is_one_explicit_new_queue(self):
        first=fl.generate(self.settings)
        s=self.choose(dict(self.settings,new_run=True));self.assertFalse(s['new_run'])
        second=fl.generate(s)
        self.assertNotEqual(first,second);self.assertEqual(fl.generate(s),second)
        self.assertEqual(len(self.calls),2)

    def test_failure_saves_completed_and_resume_skips_them(self):
        s=self.choose(dict(self.settings,count=3));self.fail_at=2
        with self.assertRaisesRegex(fl.WorkflowError,'fixture failure'):fl.generate(s)
        partial=fl.current_output(s);self.assertEqual(partial['completed_count'],1)
        old=partial['archive_path'];self.assertTrue(Path(old).is_file())
        with self.assertRaises(fl.WorkflowError):fl.generate(s)
        with self.assertRaises(fl.WorkflowError):self.choose(s)
        self.fail_at=None;s=self.choose(dict(s,resume_failed_queue=True))
        final=fl.generate(s)
        self.assertEqual([x['seed'] for x in self.calls],[42,43,43,44])
        self.assertEqual(fl.current_output(s)['completed_count'],3)
        self.assertNotEqual(old,final);self.assertTrue(Path(old).is_file())

    def test_stop_request_prevents_next_clip_and_keeps_completed(self):
        s=self.choose(dict(self.settings,count=3));self.stop_after=1
        with self.assertRaises(KeyboardInterrupt):fl.generate(s)
        self.assertEqual(len(self.calls),1);self.assertEqual(fl.current_output(s)['completed_count'],1)
        self.stop_after=None;s=self.choose(dict(s,resume_failed_queue=True));fl.generate(s)
        self.assertEqual(len(self.calls),3)

    def test_queue_changes_do_not_extend_budget_or_allow_unbounded_counts(self):
        before=fl.guard.read_state()
        for count in (0,101,True,float('inf')):
            with self.assertRaises(fl.WorkflowError):self.choose(dict(self.settings,count=count))
        s=self.choose(dict(self.settings,count=40))
        self.assertEqual(fl.guard.read_state()['deadline_epoch'],before['deadline_epoch'])
        self.assertEqual(s['count'],40)
        with self.assertRaises(fl.WorkflowError):self.choose(dict(s,cu_budget=6.))

    def test_route_budget_above_trial_cap_and_legacy_route_unchanged(self):
        self.assertEqual(fl.validate(dict(self.settings,cu_budget=20.))['cu_budget'],20.)
        import colab_workflow
        with self.assertRaises(colab_workflow.WorkflowError):colab_workflow.validate_inputs(fl.budget_settings(dict(self.settings,cu_budget=20.)))

    def test_journal_cannot_expand_the_explicit_requested_count(self):
        state=fl.guard.read_state();queue=state['generation_queues'][state['active_queue_id']]
        queue['jobs'].append({'index':1,'status':'pending','attempts':[]});fl.guard.write_state(state)
        with self.assertRaisesRegex(fl.WorkflowError,'本数'):fl.generate(self.settings)
        self.assertEqual(self.calls,[])

    def test_r2v_controller_uses_same_finite_queue_and_preserves_reference_role(self):
        rr=self.root/'ref';rr.mkdir();inp=rr/'inputs';inp.mkdir()
        src=Path(self.frame['path']);dest=inp/src.name;dest.write_bytes(src.read_bytes())
        s=dict(self.settings,profile='near720_crop',count=2,prompts=['A <Picture 1> scene','Another <Picture 1> scene'],
               image_use_confirmed=True,image_content_confirmed=True,reference=dict(self.frame,path=str(dest)))
        calls=[]
        with patch.object(ref.guard,'ROOT',rr),patch.object(ref.guard,'STATE',rr/'state.json'),patch.object(ref.guard,'require_cloud'), \
             patch.object(ref.guard,'inspect_resources',return_value={}),patch.object(ref.guard,'boot_id',return_value='fixture'):
            ref.initialize_budget(s);s=ref.select_request(s)
            lock=rr/'requirements-r2v.lock';lock.write_bytes((fixtures.ROOT/'requirements-r2v.lock').read_bytes())
            (rr/'download_report.json').write_text('{}');state=ref.guard.read_state()
            state.update(setup_complete=True,kit_digest=ref.kit_digest(),requirements_sha256=hashlib.sha256(lock.read_bytes()).hexdigest());ref.guard.write_state(state)
            def phase(command,label,**kwargs):
                if label=='r2v_generation':
                    job=json.loads((rr/'run.json').read_text());calls.append(job)
                    out=Path(job['output_dir']);(out/'native.mp4').write_bytes(b'fixture');(out/'metrics.json').write_text('{}')
                elif label=='export_exact720':Path(command[-1]).write_bytes(b'fixture')
            with patch.object(ref.guard,'run_waiting',side_effect=phase),patch.object(ref,'probe',side_effect=self.fake_probe):
                archive=ref.generate(s);self.assertEqual(ref.generate(s),archive)
                s=ref.select_request(dict(s,count=1,seed=56,prompts=['New <Picture 1> scene']));ref.generate(s)
            self.assertEqual(len(calls),3)
            for job in calls:
                example=job['jobs']['examples'][0]
                self.assertIn('references',example);self.assertNotIn('image',example)
            self.assertTrue(Path(archive).is_file())


if __name__=='__main__':unittest.main()
