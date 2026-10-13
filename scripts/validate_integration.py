"""Offline release checks. No installs, provider access or model imports."""
import ast
import hashlib
import json
from pathlib import Path
import re

import validate_distribution as baseline

ROOT = Path(__file__).resolve().parent.parent
PRESERVED = {
    'scripts/colab_workflow.py': '0ec1e2f6d3de22f2c55045c2e050da598a46bcb4a38debbf8cc3219b3f5e7a2d',
    'scripts/generate_native.py': 'e33f1173a702e0943a6b2f1b2b3f68d603cd48cd52df79bd86aec68fc6469dc6',
    'requirements-colab.txt': 'e469072a3dc29f744b1c82e0548de5754037392c06ced5038ead26fb13429956',
    'build-constraints.txt': 'e954edaf6d6072c23d815de88bea8d62c717b8636fa5320b1c40cfe8e948bde9',
    'scripts/r2v_native.py': '58c17569568e38104123ce280b0396f6e610438ea8d45808e210fb7ea8849b5b',
    'requirements-r2v.lock': '57d89223bdebf8f7ab0de2bdc8fc1c7ff8b52e9987aa477c2646184ea6b4670b',
    'vendor/modeltc/LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
    'vendor/modeltc/inference_minimax_h3.py': 'a77bc604b26a711efab66e804b517957bb10a00d1cb1e2a4a556ab4854c0dffd',
    'vendor/modeltc/minimax_h3_ref2va_pipeline.py': '7d9fcd0f9bb102b49c871f84939729e358573e0934e0b39eaef7f9c3d652e9d4',
    'vendor/modeltc/resolution_util.py': '91ed96b6745d68c41848ea436704fb160ef33198aaacc524f059e563538c6314',
    'vendor/modeltc/manifest.json': '1f9a3448ede2d73d58720c41f23b18e69d7a5ea53d2af71cf9c4d72826b00961',
}
EASY_CELLS = {
    2: 'faafc730fd1cfb5a55ebd248ce56f8d78a045f24f6178332ff9a011f9114e080',
    3: '7f8df9aa09c34f7a48358b288345fd61b48d81205b106201e8d787bca52c4ef8',
    4: '4136a7e80df56739ec0208aa5a5f50c10a4cc61d3c0cf3587cb2d459b5eb96e1',
    5: 'e85bf6206f25c1a23525683430fbad2956fcf7765657d62ee67bc573bf705b2b',
}


def source(cell):
    return ''.join(cell['source'])


def assignment(tree, name):
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def privacy(root):
    files = [p for p in root.rglob('*') if p.is_file() and not {'__pycache__', '.git'} & set(p.parts)]
    for p in files:
        rel = p.relative_to(root)
        assert not p.is_symlink(), f'Symlink in distribution: {rel}'
        assert p.suffix in ('.md', '.txt', '.py', '.json', '.ipynb', '.in', '.lock') or p.name in ('LICENSE', 'NOTICE', '.gitignore'), f'Unexpected artifact: {rel}'
        assert p.name not in ('state.json', 'metrics.json', 'download_report.json', '.env', '.netrc'), f'Runtime/private state: {rel}'
        text = p.read_text()
        patterns = [r'/User' + r's/[A-Za-z0-9_.-]+/', r'colab\.research\.google\.com/drive/[\w-]{8,}',
                    r'libfile_[a-f0-9]{8,}', r'(?i)(?:hf_|sk-proj-)[a-z0-9]{15,}',
                    r'data:image/[^;]+;base64,', r'[\w.+-]+@(?:gmail|icloud|outlook)\.com',
                    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
        for expression in patterns:
            assert not re.search(expression, text), f'Private material in {rel}'
    return len(files)


def validate(root=ROOT):
    count = privacy(root)
    previous = baseline.ROOT
    try:
        baseline.ROOT = root
        base = baseline.validate()
    finally:
        baseline.ROOT = previous
    for name, digest in PRESERVED.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, f'Preserved bytes changed: {name}'
    easy = json.loads((root / 'MiniMax_H3_Easy_Colab.ipynb').read_text())
    # Only the mode label and routing message changed in the Easy input cell.
    original_input = source(easy['cells'][1]).replace('Reference-to-Video (別ノートブック)', 'Reference-to-Video (準備中)')
    original_input = original_input.replace(
        'Reference-to-Videoは別のMiniMax_H3_R2V_720p_Colab.ipynbを使用してください。このノートブックはT2V/I2V専用です。',
        'Reference-to-Videoは別重みと実機検証が必要です。この候補ではまだ実行しません。')
    assert hashlib.sha256(original_input.encode()).hexdigest() == 'aa19ee061e3d6958b5926012bcc7cfa00249140c7decdab4094d3b82e1e20bcc', 'Easy input changed beyond R2V routing'
    for index, digest in EASY_CELLS.items():
        assert hashlib.sha256(source(easy['cells'][index]).encode()).hexdigest() == digest, f'Easy execution cell changed: {index}'
    ref = json.loads((root / 'MiniMax_H3_R2V_720p_Colab.ipynb').read_text())
    assert ref['nbformat'] == 4 and len(ref['cells']) == 6
    assert set(ref['metadata']) == {'colab', 'kernelspec'}
    assert ref['metadata']['colab'] == {'name': 'MiniMax_H3_R2V_720p_Colab.ipynb'}
    for cell in ref['cells']:
        assert cell['metadata'] == {}
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            ast.parse(source(cell))
    first = ast.parse(source(ref['cells'][1]))
    embedded = assignment(first, 'EMBEDDED_KIT')
    expected = {'scripts/colab_workflow.py', 'scripts/r2v_workflow.py', 'scripts/r2v_native.py', 'scripts/serial_queue.py',
                'requirements-r2v.in', 'requirements-r2v.lock', 'build-constraints.txt',
                'vendor/modeltc/manifest.json', 'vendor/modeltc/LICENSE', 'vendor/modeltc/inference_minimax_h3.py',
                'vendor/modeltc/minimax_h3_ref2va_pipeline.py', 'vendor/modeltc/resolution_util.py'}
    assert set(embedded) == expected
    for name, text in embedded.items():
        assert text == (root / name).read_text(), f'R2V embed stale: {name}'
    for name in ('GPU_TRIAL_APPROVED', 'SOURCE_EXECUTION_ACCEPTED', 'LICENSE_ACCEPTED', 'ELIGIBLE_USE_CONFIRMED',
                 'IMAGE_USE_CONFIRMED', 'IMAGE_CONTENT_CONFIRMED', 'HIGHER_RESOLUTION_RISK_ACCEPTED', 'UPLOAD_REFERENCE'):
        assert assignment(first, name) is False, f'Consent default: {name}'
    assert assignment(first, 'CU_REMAINING') == assignment(first, 'CU_PER_HOUR') == 0
    stop = ast.parse(source(ref['cells'][-1]))
    for name in ('TRY_UNVERIFIED_SHUTDOWN_HELPER', 'SAVED_TO_MY_DEVICE', 'STOP_AFTER_ERROR'):
        assert assignment(stop, name) is False
    pins = [line.strip() for line in (root / 'requirements-r2v.lock').read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    assert len(pins) == 67
    for line in pins:
        assert re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+', line) or line == 'diffusers @ git+https://github.com/huggingface/diffusers.git@85c9fa4caee33d1c5b473395784471d2d7957c1b', line
    for entry in json.loads((root / 'vendor/modeltc/manifest.json').read_text()):
        data = (root / 'vendor/modeltc' / entry['path']).read_bytes()
        assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
        assert '/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/' in entry['url']
    return {'files_checked': count, 'notebooks': 2, 'code_cells_checked': 10,
            't2v_i2v_pins': base['pinned_dependencies'], 'r2v_pins': len(pins),
            'outputs_in_distribution': 0, 'preserved_core_files': len(PRESERVED),
            'verification': 'offline only; no additional GPU or remote publication'}


if __name__ == '__main__':
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
