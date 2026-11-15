"""Run in an approved FREE GPU runtime. No remote write or paid API.
Token comes from HF_TOKEN env or Colab Secrets, never from this file.
CircuitKIT runtime compatibility check for CohereLabs/tiny-aya-fire (cohere2):
loads via circuitkit's TransformerLens compat port and runs a tiny EAP
discovery on a stock synthetic task to prove the pipeline runs on this
architecture. This is a plumbing check on a synthetic task, NOT the safety
circuit and NOT a safety metric.
"""
import argparse, hashlib, json, os, platform, time
from pathlib import Path
MODEL='CohereLabs/tiny-aya-fire'
REVISION='73500c80fcb6c8bea3e830c21d5d2920249a0e1a'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='eval/circuitkit')
    p.add_argument('--num-examples',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    import torch
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report={'stage':'circuitkit_runtime_check','model':MODEL,'revision':REVISION,
      'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.cuda.is_available(),
      'status':'preflight','caveat':'Stock synthetic task: proves load+discovery runs on cohere2, not a safety result.'}
    if report['cuda']:
        report['gpu']=torch.cuda.get_device_name(0);report['total_gpu_bytes']=torch.cuda.get_device_properties(0).total_memory
    (out/'run_manifest.json').write_text(json.dumps(report,indent=2))
    if args.preflight_only:print(json.dumps(report,indent=2));return
    if not report['cuda']:raise RuntimeError('GPU required. Do not substitute paid inference.')
    if report['total_gpu_bytes']<12*1024**3:raise RuntimeError('Less than 12 GiB GPU. Stop and review memory plan, do not spend.')
    token=os.environ.get('HF_TOKEN')
    if not token:
        try:
            from google.colab import userdata
            token=userdata.get('HF_TOKEN')
        except Exception as e:raise RuntimeError(f'Colab secret read failed: {type(e).__name__}: {e}') from None
    if not token:raise RuntimeError('Missing HF_TOKEN. No token is generated or printed by this script.')
    from huggingface_hub import whoami
    identity=whoami(token=token)
    if identity.get('name')!='ha55an-hg':raise RuntimeError('Token is not ha55an-hg. Stop; do not use another account.')
    import circuitkit as ck, transformer_lens
    report['circuitkit']=ck.__version__ if hasattr(ck,'__version__') else 'unknown'
    report['transformer_lens']=__import__('importlib.metadata',fromlist=['metadata']).version('transformer_lens')
    t0=time.time()
    m=ck.load_model(MODEL,revision=REVISION,dtype='bfloat16',device='cuda')  # token via HF_TOKEN env: circuitkit's cohere2 port passes its own token= and would collide
    report['load_seconds']=round(time.time()-t0,1)
    cfg={'model':{'name':MODEL,'precision':'bfloat16'},
      'discovery':{'algorithm':'eap','task':'greater_than','level':'node',
        'data_params':{'num_examples':args.num_examples,'batch_size':args.batch_size},'seed':42},
      'pruning':{'target_sparsity':0.3,'scope':'both'}}
    t1=time.time()
    artifact=ck.discover_circuit(cfg,_model=m)
    report['discovery_seconds']=round(time.time()-t1,1)
    report['artifact_type']=type(artifact).__name__
    report['artifact_summary']=str(artifact)[:500]
    report['status']='runtime_check_passed'
    report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    (out/'run_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
