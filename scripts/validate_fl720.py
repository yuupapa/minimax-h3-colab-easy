"""Local candidate audit; does not import frameworks or allocate a GPU."""
import ast
import hashlib
import json
from pathlib import Path

import validate_integration as prior
import build_fl720_notebook as builder

ROOT = Path(__file__).resolve().parent.parent
BASELINE_MANIFEST_SHA = '417e4dcdd1a4d1407335c75849b934b4225dbd9faff94f6d6223e63bdcf8720f'
UPDATED_ENTRY_DOCS = {'README.md', 'AGENTS.md', 'MiniMax_H3_R2V_720p_Colab.ipynb', 'scripts/r2v_workflow.py', 'scripts/build_r2v_notebook.py', 'scripts/validate_integration.py', 'tests/test_r2v.py', 'tests/test_integration.py', 'docs/R2V_VALIDATION.md'}


def validate():
    previous = prior.validate(ROOT)
    manifest = ROOT / 'docs/FL720_BASELINE_MANIFEST.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == BASELINE_MANIFEST_SHA
    base = json.loads(manifest.read_text())
    assert base['commit'] == '2dd951cfc81fc030bf4edd6eb293e70baa0b9e6d' and len(base['files']) == 37
    for entry in base['files']:
        # Explicitly authorized entry-point documentation for the added route.
        # All other original files, including both old notebooks, stay exact.
        if entry['path'] in UPDATED_ENTRY_DOCS:
            continue
        data = (ROOT / entry['path']).read_bytes()
        assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
    readme = (ROOT / 'README.md').read_text()
    for name in ('MiniMax_H3_Easy_Colab.ipynb', 'MiniMax_H3_R2V_720p_Colab.ipynb',
                 'MiniMax_H3_T2V_I2V_720p_DEV_Colab.ipynb'):
        assert (ROOT / name).is_file()
        assert 'https://colab.research.google.com/github/yuupapa/minimax-h3-colab-easy/blob/main/' + name in readme
    assert 'offline mock regression evidence only' in (ROOT/'AGENTS.md').read_text()
    nb = json.loads((ROOT / 'MiniMax_H3_T2V_I2V_720p_DEV_Colab.ipynb').read_text())
    assert nb['nbformat'] == 4 and len(nb['cells']) == 6
    assert set(nb['metadata']) == {'colab', 'kernelspec'}
    assert nb['metadata']['colab'] == {'name': 'MiniMax_H3_T2V_I2V_720p_DEV_Colab.ipynb'}
    for cell in nb['cells']:
        assert cell['metadata'] == {}
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and cell['outputs'] == []
            ast.parse(prior.source(cell))
    first = ast.parse(prior.source(nb['cells'][1]))
    for name in ('GPU_TRIAL_APPROVED', 'T2V_TRIAL_APPROVED', 'I2V_TRIAL_APPROVED', 'MODE_SWITCH_CONFIRMED', 'SOURCE_EXECUTION_ACCEPTED', 'LICENSE_ACCEPTED', 'ELIGIBLE_USE_CONFIRMED',
                 'HIGHER_RESOLUTION_RISK_ACCEPTED', 'IMAGE_USE_CONFIRMED', 'IMAGE_CONTENT_CONFIRMED', 'UPLOAD_FIRST_FRAME'):
        assert prior.assignment(first, name) is False, name
    assert prior.assignment(first, 'CU_REMAINING') == prior.assignment(first, 'CU_PER_HOUR') == 0
    assert prior.assignment(first, 'CU_BUDGET') == 5.0
    assert prior.assignment(first, 'GENERATION_MODE') == 'Text-to-Video'
    embedded = prior.assignment(first, 'EMBEDDED_KIT')
    assert set(embedded) == set(builder.FILES)
    for name, text in embedded.items():
        assert text == (ROOT / name).read_text(), f'FL720 embed stale: {name}'
    flow = ast.parse((ROOT / 'scripts/fl720_workflow.py').read_text())
    assert prior.assignment(flow, 'BASE_REV') == '42ed227ee7df40d41602854ae760620d6eb651fe'
    assert prior.assignment(flow, 'MODELTC_REV') == '02e26d591f7a04d5d1a074c9566d5dd4f22f6225'
    assert prior.assignment(flow, 'DIFFUSERS_REV') == '85c9fa4caee33d1c5b473395784471d2d7957c1b'
    assert prior.assignment(flow, 'LORA_REV') == '3ec17a324ced54151364f24f8b5fb6bf7e26414f'
    assert prior.assignment(flow, 'LORA_FILE') == 'minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors'
    assert prior.assignment(flow, 'LORA_SHA') == '1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c'
    assert prior.assignment(flow, 'PROFILE') == {'megapixels': .9, 'native': [1280, 736], 'delivery': [1280, 720],
                                                'filter': 'crop=1280:720:0:8,setsar=1', 'transform': '8px top/bottom crop; no resize'}
    native = ast.parse((ROOT / 'scripts/fl720_native.py').read_text())
    args = next(n for n in ast.walk(native) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'argparse.Namespace')
    values = {k.arg: k.value for k in args.keywords}
    for key, value in (('video_shift', 6.), ('audio_shift', 3.), ('lora_alpha', 8), ('lora_scale', 1.),
                       ('fuse_lora', False), ('attention_backend', None), ('no_cpu_offload', False)):
        assert ast.literal_eval(values[key]) == value, key
    save = ast.parse(prior.source(nb['cells'][4]))
    assert prior.assignment(save,'SAVED_CURRENT_RUN') is False
    stop = ast.parse(prior.source(nb['cells'][-1]))
    assert all(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
               and n.value.func.id == 'print' for n in stop.body), 'Final cell must only explain menu deletion'
    return {'files_checked': previous['files_checked'], 'notebooks': 3, 'code_cells_checked': 15,
            'published_files_preserved_byte_for_byte': 28, 'updated_entry_docs': 2, 'existing_t2v_i2v_dependencies': 234,
            'shared_frozen_diffusers_dependencies': 67, 'outputs_in_distribution': 0,
            'new_fl720_status': 'T2V/I2V one core GPU trial each passed in separate cleanVMs; shared switch/default UI/expiry shutdown unverified',
            'proposal_cu_single_vm_total': 5.0, 'max_outputs_per_queue':100, 'dependency_setup_and_weight_downloads':1}


if __name__ == '__main__':
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
