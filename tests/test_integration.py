"""Cross-route release and elapsed-budget regression checks; stdlib only."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import build_notebook
import build_r2v_notebook
import colab_workflow as easy
import r2v_workflow as r
import validate_integration as release
import test_r2v as ref_tests


class ReleaseChecks(unittest.TestCase):
    def test_integrated_release_preserves_both_measured_cores_and_dependencies(self):
        report = release.validate()
        self.assertEqual(report['t2v_i2v_pins'], 234)
        self.assertEqual(report['r2v_pins'], 67)
        self.assertEqual(report['outputs_in_distribution'], 0)

    def test_two_notebooks_rebuild_deterministically(self):
        for filename, builder in [('MiniMax_H3_Easy_Colab.ipynb', build_notebook),
                                  ('MiniMax_H3_R2V_720p_Colab.ipynb', build_r2v_notebook)]:
            before = (ROOT / filename).read_bytes()
            builder.build()
            self.assertEqual((ROOT / filename).read_bytes(), before)

    def test_easy_rejects_reference_mode_before_setup_or_image_upload(self):
        nb = json.loads((ROOT / 'MiniMax_H3_Easy_Colab.ipynb').read_text())
        tree = ast.parse(release.source(nb['cells'][1]))
        gate = next(n for n in tree.body if isinstance(n, ast.If)
                    and 'GENERATION_MODE not in' in ast.unparse(n.test))
        with self.assertRaisesRegex(RuntimeError, 'MiniMax_H3_R2V_720p'):
            exec(compile(ast.Module(body=[gate], type_ignores=[]), '<route-gate>', 'exec'),
                 {'GENERATION_MODE': 'Reference-to-Video (別ノートブック)'})

    def test_default_r2v_stop_cell_only_instructs_manual_menu_deletion(self):
        nb = json.loads((ROOT / 'MiniMax_H3_R2V_720p_Colab.ipynb').read_text())
        stop = Mock()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            exec(compile(release.source(nb['cells'][-1]), '<stop>', 'exec'), {'r2v': types.SimpleNamespace(shutdown=stop)})
        stop.assert_not_called()
        self.assertIn('終了補助は実行しません', out.getvalue())
        self.assertIn('使用率0・稼働0', out.getvalue())

    def test_privacy_audit_rejects_private_identifiers_secrets_and_runtime_state(self):
        poisons = [('extra.md', 'lib' + 'file_' + 'a' * 32),
                   ('extra.md', 'https://colab.research.google.com/' + 'drive/' + 'a' * 20),
                   ('extra.md', 'sk-' + 'proj-' + 'z' * 32),
                   ('extra.md', 'private' + '@' + 'gmail.com'),
                   ('state.json', '{}'), ('reference.png', 'pixels')]
        for filename, content in poisons:
            with self.subTest(filename=filename, content_type=content[:3]), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / filename
                p.write_text(content)
                with self.assertRaises(AssertionError):
                    release.privacy(Path(tmp))

    def test_privacy_audit_rejects_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'safe.md').write_text('Safe')
            (p / 'alias.md').symlink_to(p / 'safe.md')
            with self.assertRaisesRegex(AssertionError, 'Symlink'):
                release.privacy(p)

    def test_frozen_lock_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'kit'
            shutil.copytree(ROOT, p, ignore=shutil.ignore_patterns('__pycache__', '.git'))
            lock = p / 'requirements-r2v.lock'
            lock.write_text(lock.read_text().replace('torch==2.14.1', 'torch==2.14.2'))
            with self.assertRaisesRegex(AssertionError, 'requirements-r2v.lock'):
                release.validate(p)


class ElapsedBudgetChecks(unittest.TestCase):
    setUp = ref_tests.Ref2VAChecks.setUp
    tearDown = ref_tests.Ref2VAChecks.tearDown

    def initialize(self, **updates):
        with patch.object(r.guard, 'require_cloud'), patch.object(r.time, 'time', return_value=10000.):
            return r.initialize_budget(dict(self.settings, **updates))

    def test_elapsed_time_is_deducted_from_both_deadlines(self):
        s = self.initialize(gpu_minutes_before_input=5.)
        self.assertEqual(s['session_started_epoch'], 9700.)
        self.assertAlmostEqual(s['deadline_epoch'], 9700 + 5 / 6.77 * 3600 * .85)
        self.assertAlmostEqual(s['shutdown_deadline_epoch'], 9700 + 5 / 6.77 * 3600)

    def test_reentry_and_lower_rate_cannot_extend_either_deadline(self):
        s = self.initialize(gpu_minutes_before_input=5.)
        later = self.initialize(gpu_minutes_before_input=0., cu_rate=3.)
        for key in ('session_started_epoch', 'deadline_epoch', 'shutdown_deadline_epoch'):
            self.assertEqual(s[key], later[key])

    def test_higher_rate_shortens_both_deadlines_using_original_start(self):
        s = self.initialize(gpu_minutes_before_input=5.)
        later = self.initialize(cu_rate=10.)
        self.assertEqual(later['session_started_epoch'], s['session_started_epoch'])
        self.assertLess(later['deadline_epoch'], s['deadline_epoch'])
        self.assertAlmostEqual(later['shutdown_deadline_epoch'], 9700 + 5 / 10 * 3600)

    def test_invalid_elapsed_and_changed_budget_preserve_the_old_timer(self):
        self.initialize()
        for value in (True, -1, float('nan'), float('inf'), '5'):
            old = Mock()
            with self.subTest(value=value), patch.object(r.guard, 'require_cloud'), patch.object(r.guard.threading, 'Timer') as timer:
                with self.assertRaises(r.WorkflowError):
                    r.arm_budget(dict(self.settings, gpu_minutes_before_input=value), previous_timer=old)
                old.cancel.assert_not_called()
                timer.assert_not_called()
        old = Mock()
        with patch.object(r.guard, 'require_cloud'):
            with self.assertRaisesRegex(r.WorkflowError, 'CU上限'):
                r.arm_budget(dict(self.settings, cu_budget=4.), previous_timer=old)
        old.cancel.assert_not_called()

    def test_expired_processing_still_arms_cleanup_without_running_a_model(self):
        settings = dict(self.settings, gpu_minutes_before_input=40.)
        old = Mock()
        with patch.object(r.guard, 'require_cloud'), patch.object(r.time, 'time', return_value=10000.), \
             patch.object(r.guard.threading, 'Timer') as timer:
            with self.assertRaises(r.guard.BudgetExceeded):
                r.initialize_budget(settings)
            r.arm_budget(settings, previous_timer=old)
            timer.return_value.start.assert_called_once()
            old.cancel.assert_called_once()
            self.assertGreater(timer.call_args.args[0], 0)

    def test_rearm_start_failure_keeps_old_cleanup(self):
        old = Mock()
        with patch.object(r.guard, 'require_cloud'), patch.object(r.time, 'time', return_value=10000.), \
             patch.object(r.guard.threading, 'Timer') as timer:
            timer.return_value.start.side_effect = RuntimeError('timer failed')
            with self.assertRaisesRegex(RuntimeError, 'timer failed'):
                r.arm_budget(self.settings, previous_timer=old)
            old.cancel.assert_not_called()

    def test_r2v_budget_does_not_overwrite_easy_budget(self):
        easy_state = self.root / 'easy-state.json'
        easy_state.write_text('{"cu_budget":3,"marker":"keep"}')
        with patch.object(easy, 'STATE', easy_state):
            self.initialize(gpu_minutes_before_input=5.)
            self.assertEqual(easy.read_state(), {'cu_budget': 3, 'marker': 'keep'})

    def test_reference_record_dimensions_must_match_actual_png(self):
        s = copy.deepcopy(self.settings)
        s['reference']['width'] = 512
        with self.assertRaisesRegex(r.WorkflowError, '寸法記録'):
            r.validate(s)

    def test_setup_uses_exact_lock_before_preflight_without_resolution(self):
        def phase(command, label, **kwargs):
            if label == 'sync_r2v_environment':
                self.assertEqual((self.root / 'requirements-r2v.lock').read_bytes(), (ROOT / 'requirements-r2v.lock').read_bytes())
                self.assertEqual(command[:3], ['uv', 'pip', 'sync'])
            if label == 'r2v_api_preflight':
                raise r.WorkflowError('stop after API check')
        with patch.object(r.guard, 'require_cloud'), patch.object(r, 'initialize_budget'), \
             patch.object(r.guard, 'inspect_resources', return_value={}), patch.object(r.guard, 'boot_id', return_value='offline'), \
             patch.object(r.guard, 'SCRATCH', self.root / 'cache'), patch.object(r.guard, 'run_waiting', side_effect=phase) as run:
            with self.assertRaisesRegex(r.WorkflowError, 'stop after API check'):
                r.setup(self.settings)
        labels = [c.args[1] for c in run.call_args_list]
        self.assertEqual(labels, ['install_uv', 'create_r2v_venv', 'sync_r2v_environment', 'r2v_api_preflight'])

    def test_new_upload_stops_even_with_stale_content_confirmation(self):
        nb = json.loads((ROOT / 'MiniMax_H3_R2V_720p_Colab.ipynb').read_text())
        source = 'settings=None\n' + release.source(nb['cells'][1]).split('candidate=', 1)[1]
        source = source.replace('settings=None\n', 'settings=None\ncandidate=', 1)
        flags = {name: True for name in ('GPU_TRIAL_APPROVED', 'SOURCE_EXECUTION_ACCEPTED', 'LICENSE_ACCEPTED',
                  'ELIGIBLE_USE_CONFIRMED', 'IMAGE_USE_CONFIRMED', 'IMAGE_CONTENT_CONFIRMED',
                  'HIGHER_RESOLUTION_RISK_ACCEPTED', 'UPLOAD_REFERENCE')}
        ns = dict(flags, r2v=r, PROFILE='near720_crop', PROMPT=self.settings['prompts'][0],
                  json=json, VIDEO_COUNT=1, PROMPTS_JSON='', NEW_RUN=False, RESUME_FAILED_QUEUE=False, SEED=42, CU_REMAINING=100., CU_PER_HOUR=6.77, CU_BUDGET=5., GPU_CONNECTED_MINUTES_AGO=0.)
        display = types.ModuleType('IPython.display')
        display.Image = lambda **kwargs: None
        display.display = lambda *args: None
        with patch.object(r, 'arm_budget'), patch.object(r, 'upload_reference', return_value=self.settings['reference']), \
             patch.dict(sys.modules, {'IPython.display': display}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, '参照画像を確認'):
                exec(compile(source, '<input-upload>', 'exec'), ns)
        self.assertIsNone(ns['settings'])
        self.assertFalse(ns['IMAGE_CONTENT_CONFIRMED'])
        self.assertFalse(ns['UPLOAD_REFERENCE'])


if __name__ == '__main__':
    unittest.main()
