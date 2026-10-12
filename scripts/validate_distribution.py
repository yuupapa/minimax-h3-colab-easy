"""Offline validation only: no model imports, installs, browser or network."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent.parent


def validate():
    required=['README.md','AGENTS.md','LICENSE','NOTICE','requirements-colab.txt','build-constraints.txt','MiniMax_H3_Easy_Colab.ipynb','scripts/colab_workflow.py','scripts/generate_native.py','scripts/build_notebook.py','docs/LICENSING.md']
    for name in required: assert (ROOT/name).is_file(),name
    all_files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts]
    forbidden=[r'/Users/[A-Za-z0-9_-]+',r'colab\.research\.google\.com/drive/[A-Za-z0-9_-]{8,}',r'drive\.google\.com/file/d/[A-Za-z0-9_-]{8,}',r'libfile_[a-f0-9]+',r'data:video/mp4;'+r'base64,',r'(?i)(?:hf_|sk-proj-)[A-Za-z0-9]{15,}']
    for p in all_files:
        assert p.suffix not in ('.mp4','.png','.jpg','.jpeg','.webp','.gif','.safetensors','.pt','.pth','.bin','.ckpt','.zip','.log','.pyc'),p.name
        text=p.read_text()
        for expression in forbidden: assert not re.search(expression,text),f'Private material pattern in {p.relative_to(ROOT)}'
        if p.suffix=='.py':compile(text,str(p.relative_to(ROOT)),'exec')
    nb=json.loads((ROOT/'MiniMax_H3_Easy_Colab.ipynb').read_text())
    assert nb['nbformat']==4
    assert not any(k in nb['metadata'] for k in ('widgets','authors','user'))
    for i,c in enumerate(nb['cells']):
        assert not any(k in c['metadata'] for k in ('executionInfo','outputId','base_uri'))
        if c['cell_type']=='code':
            assert c['execution_count'] is None and c['outputs']==[]
            compile(''.join(c['source']),f'notebook-cell-{i}','exec')
    first=''.join(nb['cells'][1]['source']);tree=ast.parse(first)
    embedded=next(ast.literal_eval(x.value) for x in ast.walk(tree) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EMBEDDED_KIT' for t in x.targets))
    for name,text in embedded.items():assert text==(ROOT/name).read_text(),f'Notebook embed stale: {name}'
    assert set(embedded)=={'scripts/colab_workflow.py','scripts/generate_native.py','requirements-colab.txt','build-constraints.txt'}
    assert 'LICENSE_ACCEPTED = False' in first and 'ELIGIBLE_USE_CONFIRMED = False' in first
    assert 'VIDEO_COUNT = 1' in first and 'DURATION_SECONDS = 5' in first and 'CU_BUDGET = 3.0' in first
    assert 'SAVED_TO_MY_DEVICE = False' in ''.join(nb['cells'][5]['source'])
    assert 'STOP_AFTER_ERROR = False' in ''.join(nb['cells'][5]['source'])
    assert '_h3_stop_gate_open=False' in first
    assert "previous_timer=globals().get('_h3_budget_timer')" in first
    assert '_h3_budget_timer.cancel()' not in first
    generation=''.join(nb['cells'][3]['source'])
    assert generation.index('NEW_RUN=False') < generation.index('h3.generate(')
    build_lines=[x for x in (ROOT/'build-constraints.txt').read_text().splitlines() if x and not x.startswith('#')]
    assert len(build_lines)==5 and all(re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+',x) for x in build_lines)
    requirements=(ROOT/'requirements-colab.txt').read_text()
    lines=[line for line in requirements.splitlines() if line and not line.startswith('#')]
    assert len(lines)==234
    for line in lines:assert re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+',line) or line=='sglang[diffusion] @ git+https://github.com/sgl-project/sglang.git@448759b2358eb38e2a244fbe7d5a293a69dd314c#subdirectory=python',line
    native=ast.parse((ROOT/'scripts/generate_native.py').read_text())
    loads=[x for x in ast.walk(native) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='from_pretrained']
    assert len(loads)==1
    for loop in (x for x in ast.walk(native) if isinstance(x,(ast.For,ast.While))):
        assert not any(x is loads[0] for x in ast.walk(loop)), 'Model load must be outside loops'
    config=next(x.value for x in ast.walk(native) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in x.targets))
    parsed={ast.literal_eval(k):ast.literal_eval(v) for k,v in zip(config.keys,config.values) if not isinstance(v,ast.Subscript)}
    expected={'model_path':'MiniMaxAI/MiniMax-H3','revision':'42ed227ee7df40d41602854ae760620d6eb651fe','model_variant':'fl2va','backend':'sglang','trust_remote_code':False,'num_gpus':1,'tp_size':1,'ulysses_degree':1,'performance_mode':'memory','layerwise_offload_components':['dit','text_encoder','vae'],'dit_layerwise_resident_layers':20,'layerwise_resident_layers':'video_vae=36','attention_backend':'torch_sdpa','enable_torch_compile':False,'warmup_mode':'off','lora_scale':1.0,'lora_alpha':8,'lora_merge_mode':'auto'}
    assert parsed==expected,'Native configuration differs from verified profile'
    assert hashlib.sha256((ROOT/'licenses/MINIMAX-H3-COMMUNITY.txt').read_bytes()).hexdigest()=='59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44'
    return {'files_checked':len(all_files),'code_cells_checked':sum(c['cell_type']=='code' for c in nb['cells']),'pinned_dependencies':len(lines),'build_constraints':len(build_lines),'model_load_calls':len(loads),'outputs_in_distribution':0,'gpu_test':'this offline validator does not execute GPU tests; see docs/VALIDATION.md'}


if __name__=='__main__': print(json.dumps(validate(),ensure_ascii=False,indent=2))
