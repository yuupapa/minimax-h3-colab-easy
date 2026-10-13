"""Finite, journalled cloud queues; no model imports or automatic retries."""
import hashlib
import json
import math
from pathlib import Path
import uuid
import zipfile

MAX_COUNT = 100


def install_budget_policy(guard):
    """Only this isolated route loses the legacy five-CU trial cap."""
    legacy = guard.validate_inputs

    def validate_budget(settings):
        for key in ('cu_remaining', 'cu_rate', 'cu_budget'):
            value = settings.get(key)
            try:
                good = not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
            except (ValueError, TypeError):
                good = False
            if not good:
                raise guard.WorkflowError('残CU・使用率・上限CUは正の有限値が必要です。')
        if float(settings['cu_budget']) > float(settings['cu_remaining']):
            raise guard.WorkflowError('CU上限が残量を超えています。追加購入せず停止してください。')
        # Keep all legacy scalar consent/prompt/seed checks, with its old cap
        # applied only to the validation copy. The real budget stays unchanged.
        legacy(dict(settings, cu_budget=min(5., float(settings['cu_budget']))))
        return dict(settings)

    guard.validate_inputs = validate_budget


def validate_count(settings, error):
    count = settings.get('count')
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
        raise error(f'本数は1〜{MAX_COUNT}の有限な整数です。')
    prompts = settings.get('prompts')
    if not isinstance(prompts, list) or len(prompts) not in (1, count) or any(
            not isinstance(p, str) or not p.strip() or len(p) > 4000 for p in prompts):
        raise error('文章は1件（同文を本数分）または本数と同じ件数を指定してください。')
    seed = settings.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1 - count:
        raise error('seedと本数の組合せが有効範囲を超えています。')
    for key in ('new_run', 'resume_failed_queue'):
        if key in settings and not isinstance(settings[key], bool):
            raise error(f'{key}は真偽値で指定してください。')


def single_settings(settings, index):
    prompts = settings['prompts']
    return dict(settings, count=1, prompts=[prompts[index] if len(prompts) > 1 else prompts[0]],
                seed=settings['seed'] + index, new_run=False, resume_failed_queue=False)


def request_fingerprint(settings, single_fingerprint):
    # Preserve scalar fingerprints so completed old runs can be migrated.
    if settings['count'] == 1:
        return single_fingerprint(settings)
    parts = [single_fingerprint(single_settings(settings, i)) for i in range(settings['count'])]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def select(guard, settings, fingerprint, *, mode, legacy_result=None):
    guard.require_cloud()
    guard.deadline_remaining()
    state = guard.read_state()
    queues = state.setdefault('generation_queues', {})
    current = queues.get(state.get('active_queue_id'))
    request = fingerprint(settings)
    fresh = settings.get('new_run', False)
    resume = settings.get('resume_failed_queue', False)
    if fresh and resume:
        raise guard.WorkflowError('NEW_RUNとRESUME_FAILED_QUEUEは同時に指定できません。')
    if current and current['fingerprint'] == request and not fresh:
        if current['status'] in ('failed', 'stopped', 'running'):
            if not resume:
                raise guard.WorkflowError('停止・失敗したキューです。完成分を保存し、RESUME_FAILED_QUEUEで明示的に再開してください。')
            for job in current['jobs']:
                if job['status'] != 'complete':
                    job['status'] = 'pending'
            current['status'] = 'pending'
            current['saved_confirmed'] = False
            current.pop('error', None)
            (guard.ROOT/'queue.stop').unlink(missing_ok=True)
        elif resume:
            raise guard.WorkflowError('停止・失敗した現在のキューだけを再開できます。')
    else:
        if resume:
            raise guard.WorkflowError('再開には停止時と同じ文章・画像・seed・本数を選んでください。')
        if current and current['status'] == 'running':
            raise guard.WorkflowError('実行中のキューを変更できません。先に実行を中断してください。')
        qid = mode + '_queue_' + uuid.uuid4().hex[:12]
        current = {'queue_id': qid, 'mode': mode, 'fingerprint': request,
                   'status': 'pending', 'saved_confirmed': False,
                   'jobs': [{'index': i, 'status': 'pending', 'attempts': []}
                            for i in range(settings['count'])]}
        if not queues and settings['count'] == 1 and not fresh and legacy_result and legacy_result.get('complete'):
            if legacy_result.get('fingerprint') == request:
                current['jobs'][0].update(status='complete', result=legacy_result)
                current['status'] = 'complete'
        queues[qid] = current
        state['active_queue_id'] = qid
    state['active_fingerprint'] = request
    guard.write_state(state)
    return dict(settings, new_run=False, resume_failed_queue=False)


def active(guard, settings, fingerprint):
    state = guard.read_state()
    queue = state.get('generation_queues', {}).get(state.get('active_queue_id'))
    if not queue or queue['fingerprint'] != fingerprint(settings):
        raise guard.WorkflowError('現在の入力キューと一致しません。入力セルを再実行してください。')
    jobs = queue.get('jobs')
    if (not isinstance(jobs, list) or len(jobs) != settings['count']
            or [j.get('index') for j in jobs] != list(range(settings['count']))):
        raise guard.WorkflowError('保存されたキューの本数・順序が入力と異なります。生成せず状態を確認してください。')
    return state, queue


def package(guard, queue):
    complete = [j for j in queue['jobs'] if j['status'] == 'complete']
    if not complete:
        return None
    if len(queue['jobs']) == 1:
        archive = Path(complete[0]['result']['archive_path'])
    else:
        manifest = {'queue_id': queue['queue_id'], 'mode': queue['mode'],
                    'status': queue['status'], 'requested_count': len(queue['jobs']),
                    'completed_count': len(complete),
                    'jobs': [{'index': j['index'], 'status': j['status'],
                              'run_id': j.get('result', {}).get('run_id')} for j in queue['jobs']]}
        revision = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
        archive = guard.ROOT / (queue['queue_id'] + '_' + revision + '.zip')
        if not archive.exists():
            temp = archive.with_name(archive.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
                    for job in complete:
                        source = Path(job['result']['archive_path'])
                        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(guard.ROOT.resolve()):
                            raise guard.WorkflowError('完成済み出力ZIPの保存先が不正です。再生成しません。')
                        with zipfile.ZipFile(source) as old:
                            for name in old.namelist():
                                if name in ('native.mp4', 'delivery_720p.mp4', 'metrics.json', 'AI_GENERATED.txt'):
                                    z.writestr(f"clip_{job['index'] + 1:03d}/" + name, old.read(name))
                    z.writestr('queue_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
                    lock = guard.ROOT / 'requirements-r2v.lock'
                    if lock.is_file():
                        z.write(lock, lock.name)
                temp.replace(archive)
            finally:
                temp.unlink(missing_ok=True)
    if not archive.is_file():
        raise guard.WorkflowError('完成済みZIPがありません。自動再生成せず保存先を確認してください。')
    return str(archive)


def current_output(guard, settings, fingerprint):
    guard.require_cloud()
    state, queue = active(guard, settings, fingerprint)
    archive = package(guard, queue)
    if not archive:
        raise guard.WorkflowError('このキューには保存できる完成動画がまだありません。')
    results = [j['result'] for j in queue['jobs'] if j['status'] == 'complete']
    last = results[-1]
    if not (Path(last['output_dir']) / 'delivery_720p.mp4').is_file():
        raise guard.WorkflowError('完成済みMP4がありません。自動再生成しません。')
    return dict(last, mode=queue['mode'], run_id=last['run_id'] if len(queue['jobs']) == 1 else queue['queue_id'],
                queue_id=queue['queue_id'], archive_path=archive, outputs=results,
                requested_count=len(queue['jobs']), completed_count=len(results),
                status=queue['status'], complete=queue['status'] == 'complete')


def request_stop(guard):
    guard.require_cloud()
    state = guard.read_state()
    (guard.ROOT/'queue.stop').write_text(state.get('active_queue_id') or '')


def run(guard, settings, fingerprint, generate_one):
    state, queue = active(guard, settings, fingerprint)
    if queue['status'] == 'complete':
        return current_output(guard, settings, fingerprint)['archive_path']
    if queue['status'] != 'pending':
        raise guard.WorkflowError('キューの停止・失敗後は入力セルで明示的に再開してください。')
    queue['status'] = 'running'; guard.write_state(state)
    try:
        for index in range(len(queue['jobs'])):
            state, queue = active(guard, settings, fingerprint)
            job = queue['jobs'][index]
            if job['status'] == 'complete':
                continue
            if (guard.ROOT/'queue.stop').is_file() and (guard.ROOT/'queue.stop').read_text()==queue['queue_id']:
                raise KeyboardInterrupt('キュー停止要求を受け取りました。')
            guard.deadline_remaining()
            job['status'] = 'running'; guard.write_state(state)
            try:
                result = generate_one(single_settings(settings, job['index']))
            except BaseException:
                job['status'] = 'failed'
                raise
            # Merge with model-parent/resource writes, preserving the queue journal.
            state = guard.read_state(); queue = state['generation_queues'][queue['queue_id']]
            job = queue['jobs'][job['index']]
            job.update(status='complete', result=result)
            state['active_fingerprint'] = fingerprint(settings)
            guard.write_state(state)
            print(f"QUEUE_PROGRESS: {sum(j['status'] == 'complete' for j in queue['jobs'])}/{len(queue['jobs'])}", flush=True)
        queue['status'] = 'complete'; guard.write_state(state)
        return current_output(guard, settings, fingerprint)['archive_path']
    except BaseException as exc:
        latest = guard.read_state()
        # A failing child may have written state after the job was marked running.
        saved_queue = latest['generation_queues'][queue['queue_id']]
        for j in saved_queue['jobs']:
            if j['status'] == 'running':
                j['status'] = 'failed'
                j['attempts'].append({'status': 'failed', 'error_type': type(exc).__name__})
        saved_queue['status'] = 'stopped' if isinstance(exc, (KeyboardInterrupt, guard.BudgetExceeded)) else 'failed'
        saved_queue['error'] = type(exc).__name__
        latest['active_fingerprint'] = fingerprint(settings)
        guard.write_state(latest)
        try:
            path = package(guard, saved_queue)
            if path:
                print('PARTIAL_QUEUE_SAVED:', path, flush=True)
        except Exception as archive_error:
            print('完成した個別ZIPは保持されています。まとめZIPの作成失敗:', type(archive_error).__name__, flush=True)
        raise
