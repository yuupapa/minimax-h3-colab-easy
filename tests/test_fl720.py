"""Offline FL720 guards, upstream job parser and export integration checks."""
import ast
import contextlib
import copy
import dataclasses
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from typing import Mapping
from unittest.mock import Mock, patch
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import fl720_workflow as f
import build_fl720_notebook as builder
import colab_workflow as easy
import r2v_workflow as ref


def stdlib_upstream():
    """Execute only job/parser definitions, with no vendor imports or ML code."""
    spec = importlib.util.spec_from_file_location('fl720_resolution', ROOT / 'vendor/modeltc/resolution_util.py')
    res = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(res)
    tree = ast.parse((ROOT / 'vendor/modeltc/inference_minimax_h3.py').read_text())
    names = {'ReferenceSpec', 'GenerationJob', '_read_text', '_resolve_media_path', '_parse_images',
             '_parse_generation_settings', '_build_prompt', 'build_jobs', 'resolve_jobs_workflow'}
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    fixture = types.ModuleType('fl720_pure_upstream')
    fixture.__dict__.update(Path=Path, Mapping=Mapping, json=json, math=math, dataclass=dataclasses.dataclass,
                            validate_megapixels=res.validate_megapixels, validate_aspect_ratio=res.validate_aspect_ratio,
                            resolve_output_size=res.resolve_output_size)
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in (
                'FPS', 'DEFAULT_DURATION', 'DEFAULT_MEGAPIXELS', 'DEFAULT_ASPECT_RATIO', 'TASK_ALIASES', 'TASK_BY_IMAGE_COUNT'):
            fixture.__dict__[n.targets[0].id] = ast.literal_eval(n.value)
    ast_body = ast.parse('from __future__ import annotations').body + body
    with patch.dict(sys.modules, {fixture.__name__: fixture}):
        exec(compile(ast.Module(body=ast_body, type_ignores=[]), '<upstream-stdlib-only>', 'exec'), fixture.__dict__)
    return fixture


class FL720Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch.object(f.guard, 'ROOT', self.root), patch.object(f.guard, 'STATE', self.root / 'state.json'),
                        patch.object(f.guard, 'SCRATCH', self.root / 'cache'), patch.object(f.guard, 'VENV', self.root / 'venv')]
        for p in self.patches: p.start()
        data = b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR' + struct.pack('>II', 960, 544) + b'\x08\x02\x00\x00\x00' + b'\0' * 8
        digest = hashlib.sha256(data).hexdigest()
        image = self.root / 'inputs' / (digest + '.png')
        image.parent.mkdir()
        image.write_bytes(data)
        self.frame = {'path': str(image), 'sha256': digest, 'width': 960, 'height': 544}
        self.settings = {'gpu_trial_approved': True, 'source_execution_accepted': True, 'license_accepted': True,
                         'eligible_use_confirmed': True, 'higher_resolution_risk_accepted': True,
                         'mode': 't2v', 'profile': 'native736_crop720', 'prompts': ['A red sailboat slowly moves.'],
                         'count': 1, 'seconds': 5, 'seed': 42, 'cu_remaining': 100., 'cu_rate': 6.77,
                         'cu_budget': 5., 'gpu_minutes_before_input': 0., 'approved_modes':['t2v','i2v']}
        f.guard.write_state({'session_schema':2,'approved_modes':['i2v','t2v'],'active_mode':'t2v',
                             'mode_reviewed':True,'active_fingerprint':f.fingerprint(self.settings)})

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def i2v(self):
        return dict(self.settings, mode='i2v', first_frame=copy.deepcopy(self.frame),
                    image_use_confirmed=True, image_content_confirmed=True)

    def ready_generation(self, settings=None):
        settings=settings or self.settings
        lock = self.root / 'requirements-r2v.lock'
        lock.write_bytes((ROOT / 'requirements-r2v.lock').read_bytes())
        (self.root / 'download_report.json').write_text('{}')
        state=f.guard.read_state()
        state.update({'setup_complete': True, 'kit_digest': f.kit_digest(),
                             'requirements_sha256': hashlib.sha256(lock.read_bytes()).hexdigest(),
                             'cu_remaining_at_start': 100.,'active_mode':settings['mode'],
                             'mode_reviewed':True,'active_fingerprint':f.fingerprint(settings),
                             'last_input_image':settings.get('first_frame')})
        f.guard.write_state(state)

    def test_upstream_t2v_parser_gets_zero_images_and_fl_transformer(self):
        upstream = stdlib_upstream()
        path = self.root / 'job.json'
        path.write_text(json.dumps(f.jobs_document(f.validate(self.settings))))
        jobs = upstream.build_jobs(path)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].task, 't2va')
        self.assertEqual(jobs[0].image_paths, ())
        self.assertEqual(jobs[0].reference_specs, ())
        self.assertEqual(jobs[0].num_frames, 124)
        self.assertEqual(upstream.resolve_jobs_workflow(jobs), 'fl2va')

    def test_upstream_i2v_parser_gets_one_first_keyframe_and_no_semantic_reference(self):
        upstream = stdlib_upstream()
        doc = f.jobs_document(f.validate(self.i2v()))
        self.assertEqual(doc['examples'][0]['image'], self.frame['path'])
        self.assertNotIn('references', doc['examples'][0])
        path = self.root / 'job.json'
        path.write_text(json.dumps(doc))
        jobs = upstream.build_jobs(path)
        self.assertEqual(jobs[0].task, 'i2va')
        self.assertEqual(jobs[0].image_paths, (Path(self.frame['path']).resolve(),))
        self.assertEqual(upstream.resolve_jobs_workflow(jobs), 'fl2va')

    def test_semantic_reference_and_stale_image_in_t2v_are_rejected(self):
        for change in ({'mode': 'ref2va'}, {'reference': self.frame}, {'references': [self.frame]}, {'first_frame': self.frame}):
            with self.subTest(change=next(iter(change))), self.assertRaises(f.WorkflowError):
                f.validate(dict(self.settings, **change))

    def test_one_five_second_trial_and_fixed_native_profile_only(self):
        for change in ({'count': 2}, {'count': 40}, {'count': True}, {'seconds': 6}, {'profile': '544_upscale'}):
            with self.subTest(change=change), self.assertRaises(f.WorkflowError):
                f.validate(dict(self.settings, **change))

    def test_trial_source_license_and_resolution_consents_required(self):
        for key in ('gpu_trial_approved', 'source_execution_accepted', 'license_accepted',
                    'eligible_use_confirmed', 'higher_resolution_risk_accepted'):
            with self.subTest(key=key), self.assertRaises(f.WorkflowError):
                f.validate(dict(self.settings, **{key: False}))

    def test_i2v_rights_content_and_actual_image_dimensions_required(self):
        for key in ('image_use_confirmed', 'image_content_confirmed'):
            with self.subTest(key=key), self.assertRaises(f.WorkflowError):
                f.validate(dict(self.i2v(), **{key: False}))
        s = self.i2v()
        s['first_frame']['width'] = 1280
        with self.assertRaisesRegex(f.WorkflowError, '寸法記録'): f.validate(s)

    def test_cloud_guard_blocks_mac_before_install_upload_generation_or_timer(self):
        for fn in (f.setup, f.generate, f.upload_first_frame, f.arm_budget):
            with patch.object(f.guard, 'require_cloud', side_effect=f.WorkflowError('Mac blocked')), \
                 patch.object(f.guard, 'run_waiting') as execute:
                with self.assertRaisesRegex(f.WorkflowError, 'Mac blocked'): fn(self.settings)
                execute.assert_not_called()
        self.assertFalse((self.root / 'venv').exists())

    def test_native_preflight_guards_before_any_model_import(self):
        spec = importlib.util.spec_from_file_location('fl720_native_fixture', ROOT / 'scripts/fl720_native.py')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        with patch.object(f.guard, 'require_cloud', side_effect=f.WorkflowError('Mac blocked')):
            with self.assertRaisesRegex(f.WorkflowError, 'Mac blocked'): native.preflight()

    def test_t2v_cannot_call_the_upload_helper(self):
        with patch.object(f.guard, 'require_cloud'), patch.object(f.guard, 'upload_first_frame') as upload:
            with self.assertRaisesRegex(f.WorkflowError, 'T2V'): f.upload_first_frame(self.settings)
            upload.assert_not_called()

    def test_fl768_checkpoint_and_inactive_ref_exclusion(self):
        self.assertEqual(f.LORA_FILE, 'minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors')
        self.assertEqual(f.LORA_SHA, '1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c')
        self.assertIn('transformer/*', f.PATTERNS)
        for pattern in ('transformer_ref/*', 'FL2VA/*', 'Ref2VA/*', '*'):
            self.assertNotIn(pattern, f.PATTERNS)

    def test_native_recipe_uses_shift6_3_four_evaluations_and_fl_only(self):
        src = (ROOT / 'scripts/fl720_native.py').read_text()
        tree = ast.parse(src)
        args = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'argparse.Namespace')
        values = {k.arg: k.value for k in args.keywords}
        for key, value in (('video_shift', 6.), ('audio_shift', 3.), ('lora_alpha', 8),
                           ('lora_scale', 1.), ('fuse_lora', False), ('attention_backend', None)):
            self.assertEqual(ast.literal_eval(values[key]), value)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'pipe']
        self.assertEqual(len(calls), 1)
        kw = {k.arg: k.value for k in calls[0].keywords}
        self.assertEqual(ast.literal_eval(kw['num_inference_steps']), 5)
        self.assertEqual(ast.literal_eval(kw['num_frames']), 124)
        self.assertNotIn('references', kw)
        self.assertIn("conditioning['image']=upstream._load_rgb_image", src)
        self.assertIn("workflow='fl2va'", src)
        self.assertIn("assert name=='transformer'", src)
        self.assertIn('len(calls)!=4', src)

    def test_native_resolution_and_crop_are_not_544_upscale(self):
        upstream = stdlib_upstream()
        self.assertEqual(upstream.resolve_output_size(f.PROFILE['megapixels'],'16:9'),tuple(f.PROFILE['native']))
        self.assertNotEqual(upstream.resolve_output_size(.5,'16:9'),tuple(f.PROFILE['native']))
        self.assertEqual(f.PROFILE['native'], [1280, 736])
        self.assertEqual(f.PROFILE['delivery'], [1280, 720])
        self.assertEqual(f.PROFILE['filter'], 'crop=1280:720:0:8,setsar=1')
        self.assertNotIn('scale=', f.PROFILE['filter'])

    def test_fl_state_budget_and_environment_are_isolated_from_existing_routes(self):
        self.assertNotEqual(f.guard.VENV, easy.VENV)
        self.assertNotEqual(f.guard.VENV, ref.guard.VENV)
        before = (easy.ROOT, easy.STATE, ref.guard.ROOT, ref.guard.STATE)
        with patch.object(f.guard, 'require_cloud'), patch.object(f.time, 'time', return_value=10000.):
            first = f.initialize_budget(dict(self.settings, gpu_minutes_before_input=5.))
            later = f.initialize_budget(dict(self.settings, cu_rate=3.))
        self.assertEqual(first['session_started_epoch'], 9700.)
        self.assertEqual(first['deadline_epoch'], later['deadline_epoch'])
        self.assertEqual((easy.ROOT, easy.STATE, ref.guard.ROOT, ref.guard.STATE), before)

    def test_new_trial_requires_an_almost_idle_card_before_expensive_setup(self):
        with patch.object(f.guard,'inspect_resources',return_value={'free_vram_mib':80000}):
            with self.assertRaisesRegex(f.WorkflowError,'80,500'): f.inspect_resources()
        with patch.object(f.guard,'inspect_resources',return_value={'free_vram_mib':81000}):
            self.assertEqual(f.inspect_resources()['free_vram_mib'],81000)

    def test_invalid_input_keeps_existing_cleanup_timer(self):
        old = Mock()
        with patch.object(f.guard, 'require_cloud'), patch.object(f.guard.threading, 'Timer') as timer:
            with self.assertRaises(f.WorkflowError):
                f.arm_budget(dict(self.settings, gpu_minutes_before_input=float('nan')), previous_timer=old)
        old.cancel.assert_not_called()
        timer.assert_not_called()

    def test_preflight_failure_stops_before_large_model_download(self):
        def phase(command, label, **kwargs):
            if label == 'sync_fl720_environment':
                self.assertEqual((self.root / 'requirements-r2v.lock').read_bytes(), (ROOT / 'requirements-r2v.lock').read_bytes())
            if label == 'fl720_api_preflight': raise f.WorkflowError('API incompatible')
        with patch.object(f.guard, 'require_cloud'), patch.object(f, 'initialize_budget'), \
             patch.object(f, 'inspect_resources', return_value={}), patch.object(f.guard, 'boot_id', return_value='offline'), \
             patch.object(f.guard, 'run_waiting', side_effect=phase) as run:
            with self.assertRaisesRegex(f.WorkflowError, 'API incompatible'): f.setup(self.settings)
        self.assertEqual([c.args[1] for c in run.call_args_list],
                         ['install_uv', 'create_fl720_venv', 'sync_fl720_environment', 'fl720_api_preflight'])
        self.assertTrue(f.guard.read_state()['setup_failed'])

    def test_setup_does_not_automatically_retry_after_failure(self):
        state=f.guard.read_state();state.update(setup_attempted=True,setup_failed=True);f.guard.write_state(state)
        with patch.object(f.guard, 'require_cloud'), patch.object(f, 'initialize_budget'), \
             patch.object(f, 'inspect_resources', return_value={}), patch.object(f.guard, 'run_waiting') as run:
            with self.assertRaisesRegex(f.WorkflowError, '自動再試行'): f.setup(self.settings)
        run.assert_not_called()

    def test_changed_lock_blocks_generation_before_consuming_attempt(self):
        self.ready_generation()
        (self.root / 'requirements-r2v.lock').write_text('changed')
        with patch.object(f.guard, 'require_cloud'), patch.object(f, 'initialize_budget'), patch.object(f.guard, 'run_waiting') as run:
            with self.assertRaisesRegex(f.WorkflowError, '変更'): f.generate(self.settings)
        run.assert_not_called()
        self.assertFalse(f.guard.read_state().get('generation_attempted'))

    def test_failed_attempt_cannot_silently_regenerate_or_switch_mode(self):
        state=f.guard.read_state();state.update(session_failed=True);f.guard.write_state(state)
        with patch.object(f.guard, 'require_cloud'), patch.object(f, 'initialize_budget'), patch.object(f.guard, 'run_waiting') as run:
            for s in (self.settings, self.i2v()):
                with self.assertRaisesRegex(f.WorkflowError, '消費済み'): f.generate(s)
        run.assert_not_called()

    def test_completed_identical_input_returns_existing_file(self):
        output=self.root/'completed';output.mkdir();(output/'delivery_720p.mp4').write_bytes(b'fixture')
        archive=self.root/'existing.zip';archive.write_bytes(b'fixture')
        state=f.guard.read_state()
        state['runs']={'t2v':{'attempted':True,'complete':True,'fingerprint':f.fingerprint(self.settings),
                              'run_id':'t2v_fixture','archive_path':str(archive),'output_dir':str(output)}}
        f.guard.write_state(state)
        with patch.object(f.guard, 'require_cloud'), patch.object(f, 'initialize_budget'), patch.object(f.guard, 'run_waiting') as run:
            self.assertEqual(f.generate(self.settings), str(archive))
            with self.assertRaisesRegex(f.WorkflowError, '現在の入力'): f.generate(self.i2v())
        run.assert_not_called()

    def test_mode_and_image_hash_are_part_of_duplicate_fingerprint(self):
        self.assertNotEqual(f.fingerprint(self.settings), f.fingerprint(self.i2v()))
        s = self.i2v()
        s['first_frame']['sha256'] = '0' * 64
        self.assertNotEqual(f.fingerprint(self.i2v()), f.fingerprint(s))

    def test_export_and_archive_follow_full_decode_and_never_include_input_image(self):
        self.ready_generation(self.i2v())
        def phase(command, label, **kwargs):
            state=f.guard.read_state();out=Path(state['runs'][state['active_mode']]['output_dir'])
            if label == 'fl720_generation':
                (out / 'native.mp4').write_bytes(b'offline orchestration fixture, not a video')
                (out / 'metrics.json').write_text('{}')
            elif label == 'export_exact720':
                self.assertIn('crop=1280:720:0:8,setsar=1', command)
                (out / 'delivery_720p.mp4').write_bytes(b'offline export fixture, not a video')
        def info(width, height):
            return {'streams': [{'codec_type': 'video', 'width': width, 'height': height,
                                 'nb_frames': '124', 'avg_frame_rate': '24/1'}, {'codec_type': 'audio'}],
                    'format': {'duration': '5.175'}}
        with patch.object(f.guard, 'require_cloud'), patch.object(f, 'initialize_budget'), patch.object(f, 'inspect_resources'), \
             patch.object(f.guard, 'boot_id', return_value='offline'), patch.object(f, 'probe', side_effect=[info(1280,736), info(1280,720)]), \
             patch.object(f.guard, 'run_waiting', side_effect=phase) as run:
            archive = f.generate(self.i2v())
        self.assertEqual([c.args[1] for c in run.call_args_list], ['fl720_generation', 'decode_native', 'export_exact720', 'decode_delivery'])
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(set(z.namelist()), {'native.mp4', 'delivery_720p.mp4', 'metrics.json', 'AI_GENERATED.txt', 'requirements-r2v.lock'})
            self.assertIsNone(z.testzip())
        self.assertTrue(f.guard.read_state()['runs']['i2v']['complete'])

    def test_decode_failure_blocks_export_archive_and_retry(self):
        self.ready_generation()
        def phase(command, label, **kwargs):
            if label == 'decode_native': raise f.WorkflowError('bad frame')
        probe = {'streams': [{'codec_type':'video','width':1280,'height':736,'nb_frames':'124','avg_frame_rate':'24/1'},
                             {'codec_type':'audio'}], 'format': {'duration':'5.175'}}
        with patch.object(f.guard, 'require_cloud'), patch.object(f, 'initialize_budget'), patch.object(f, 'inspect_resources'), \
             patch.object(f.guard, 'boot_id', return_value='offline'), patch.object(f, 'probe', return_value=probe), \
             patch.object(f.guard, 'run_waiting', side_effect=phase) as run:
            with self.assertRaisesRegex(f.WorkflowError, 'bad frame'): f.generate(self.settings)
            self.assertEqual([c.args[1] for c in run.call_args_list], ['fl720_generation', 'decode_native'])
            run.reset_mock()
            with self.assertRaisesRegex(f.WorkflowError, '消費済み'): f.generate(self.settings)
            run.assert_not_called()
        self.assertFalse(list(self.root.glob('*.zip')))
        self.assertTrue(f.guard.read_state()['runs']['t2v']['failed'])

    def test_all_notebook_consents_off_exact_embeds_and_no_outputs(self):
        nb = json.loads(builder.build().read_text())
        code = [c for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code), 5)
        tree = ast.parse(code[0]['source'])
        assignments = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                       if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                       and (isinstance(n.value,ast.Constant) or n.targets[0].id=='EMBEDDED_KIT')}
        for name in ('GPU_TRIAL_APPROVED','SOURCE_EXECUTION_ACCEPTED','LICENSE_ACCEPTED','ELIGIBLE_USE_CONFIRMED',
                     'HIGHER_RESOLUTION_RISK_ACCEPTED','IMAGE_USE_CONFIRMED','IMAGE_CONTENT_CONFIRMED','UPLOAD_FIRST_FRAME'):
            self.assertIs(assignments[name], False)
        self.assertEqual(assignments['CU_REMAINING'], 0)
        self.assertEqual(assignments['CU_PER_HOUR'], 0)
        for name, text in assignments['EMBEDDED_KIT'].items(): self.assertEqual(text, (ROOT/name).read_text())
        for cell in code:
            ast.parse(cell['source'])
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])

    def test_three_route_candidate_preserves_published_files_and_passes_privacy_audit(self):
        import validate_fl720
        report=validate_fl720.validate()
        self.assertEqual(report['published_files_preserved_byte_for_byte'],35)
        self.assertEqual(report['updated_entry_docs'],2)
        self.assertEqual(report['notebooks'],3)
        self.assertEqual(report['shared_frozen_diffusers_dependencies'],67)

    def test_candidate_notebook_rebuilds_identically(self):
        p=ROOT/'MiniMax_H3_T2V_I2V_720p_DEV_Colab.ipynb'
        before=p.read_bytes()
        builder.build()
        self.assertEqual(before,p.read_bytes())

    def test_readme_three_colab_links_resolve_to_unexecuted_consent_off_notebooks(self):
        prefix = 'https://colab.research.google.com/github/yuupapa/minimax-h3-colab-easy/blob/main/'
        readme = (ROOT/'README.md').read_text()
        names = ('MiniMax_H3_Easy_Colab.ipynb', 'MiniMax_H3_R2V_720p_Colab.ipynb',
                 'MiniMax_H3_T2V_I2V_720p_DEV_Colab.ipynb')
        for name in names:
            self.assertIn(prefix + name, readme)
            nb = json.loads((ROOT/name).read_text())
            code = [c for c in nb['cells'] if c['cell_type'] == 'code']
            tree = ast.parse(''.join(code[0]['source']))
            flags = {target.id: ast.literal_eval(node.value)
                     for node in tree.body if isinstance(node, ast.Assign)
                     for target in node.targets if isinstance(target, ast.Name)
                     and target.id in ('LICENSE_ACCEPTED', 'ELIGIBLE_USE_CONFIRMED')}
            self.assertEqual(flags, {'LICENSE_ACCEPTED': False, 'ELIGIBLE_USE_CONFIRMED': False})
            for cell in code:
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_manual_stop_cell_has_no_runtime_unassignment_or_model_call(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            exec(compile(builder.STOP, '<manual-stop>', 'exec'), {})
        self.assertIn('接続解除して削除', out.getvalue())
        self.assertIn('CU/h 0・稼働0', out.getvalue())
        self.assertNotIn('runtime.unassign', builder.STOP)
        self.assertNotIn('shutdown(', builder.STOP)

    def test_new_image_upload_always_stops_even_with_stale_content_checkbox(self):
        state=f.guard.read_state();state.update(active_mode='i2v',mode_reviewed=True);f.guard.write_state(state)
        ns = {'fl720': f, 'Path': Path, 'settings': None,
              'T2V_TRIAL_APPROVED':True,'I2V_TRIAL_APPROVED':True,'MODE_SWITCH_CONFIRMED':True,
              'GPU_TRIAL_APPROVED':True,'SOURCE_EXECUTION_ACCEPTED':True,'LICENSE_ACCEPTED':True,
              'ELIGIBLE_USE_CONFIRMED':True,'HIGHER_RESOLUTION_RISK_ACCEPTED':True,
              'GENERATION_MODE':'Image-to-Video','IMAGE_USE_CONFIRMED':True,'IMAGE_CONTENT_CONFIRMED':True,
              'UPLOAD_FIRST_FRAME':True,'PROMPT':'A red boat slowly moves.','SEED':42,
              'CU_REMAINING':100.,'CU_PER_HOUR':6.77,'CU_BUDGET':5.,'GPU_CONNECTED_MINUTES_AGO':0.}
        tail = 'candidate=' + builder.BOOT.split('candidate=',1)[1]
        display = types.ModuleType('IPython.display')
        display.Image=lambda **kwargs:None
        display.display=lambda *args:None
        with patch.object(f.guard,'require_cloud'), patch.object(f.guard,'deadline_remaining',return_value=600.), patch.object(f,'arm_budget'), patch.object(f,'upload_first_frame',return_value=self.frame), \
             patch.dict(sys.modules,{'IPython.display':display}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'画像を確認'): exec(compile(tail,'<upload>','exec'),ns)
        self.assertIsNone(ns['settings'])
        self.assertFalse(ns['IMAGE_CONTENT_CONFIRMED'])
        self.assertFalse(ns['UPLOAD_FIRST_FRAME'])


if __name__=='__main__': unittest.main()
