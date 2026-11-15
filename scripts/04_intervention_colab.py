"""Run in an approved FREE GPU runtime. No remote write or paid API.
Token comes from HF_TOKEN env or Colab Secrets, never from this file.
Stage 2 (intervention): extract a refusal direction with SafeTune
(Arditi et al. 2406.11717 selection sweep) from train-split harmful prompts
vs aya_dataset harmless prompts, then steer (mode='steer', positive strength
ADDS refusal) and regenerate the SAME frozen test prompts as
03_baseline_colab.py. Not a safety score: responses require review.
Matched baseline-vs-steered comparison is a separate unrun step.
"""
import argparse, hashlib, json, os, platform, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
from pathlib import Path
MODEL='CohereLabs/tiny-aya-fire'
REVISION='73500c80fcb6c8bea3e830c21d5d2920249a0e1a'

def select_test_rows(path,per_language):
    rows=[json.loads(x) for x in Path(path).open()]
    eligible=[r for r in rows if r['metadata']['split']=='test']
    result=[]
    for lang in ['en','hi','bn']:
        subset=[r for r in eligible if r['metadata']['language']==lang]
        subset.sort(key=lambda r:hashlib.sha256(('baseline-v1:'+r['id']).encode()).hexdigest())
        result.extend(subset[:per_language])
    return result

def select_harmful_train(path,per_language):
    rows=[json.loads(x) for x in Path(path).open()]
    eligible=[r for r in rows if r['metadata']['split']=='train']
    result=[]
    for lang in ['en','hi','bn']:
        subset=[r for r in eligible if r['metadata']['language']==lang]
        subset.sort(key=lambda r:hashlib.sha256(('intervene-v1:'+r['id']).encode()).hexdigest())
        result.extend(subset[:per_language])
    return result

def load_harmless(per_language):
    from datasets import load_dataset
    ds=load_dataset('CohereLabs/aya_dataset',split='train')
    wanted={'en':'English','hi':'Hindi','bn':'Bengali'}
    buckets={k:[] for k in wanted}
    for r in ds:
        for k,name in wanted.items():
            if r['language']==name:
                s=(r['inputs'] or '').strip()
                if s:buckets[k].append(s)
    out=[]
    for k in ['en','hi','bn']:
        subset=buckets[k]
        subset.sort(key=lambda s:hashlib.sha256(('harmless-v1:'+s).encode()).hexdigest())
        if len(subset)<per_language:
            raise RuntimeError(f'aya_dataset has only {len(subset)} usable {wanted[k]} prompts; expected >= {per_language}.')
        out.extend(subset[:per_language])
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='eval/curated.jsonl')
    p.add_argument('--out',default='eval/intervention')
    p.add_argument('--per-language',type=int,default=2)
    p.add_argument('--contrast-per-language',type=int,default=16)
    p.add_argument('--strength',type=float,default=1.0)
    p.add_argument('--layers',default=None)  # CSV override, e.g. '16,18,20'
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--skip-done',action='store_true')
    args=p.parse_args()
    import torch, transformers
    test_rows=select_test_rows(args.input,args.per_language)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report={'stage':'intervention_refusal_direction','model':MODEL,'revision':REVISION,
      'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
      'cuda':torch.cuda.is_available(),'selected_ids':[r['id'] for r in test_rows],
      'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
      'max_input_tokens':512,'max_new_tokens':128,'do_sample':False,
      'steer_mode':'steer','steer_strength':args.strength,
      'status':'preflight','caveat':'Not a safety metric: steered responses require review; baseline comparison is a separate step.'}
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
    import safetune
    from safetune.interventions.steer.refusal_direction import (
        RefusalDirectionConfig, RefusalDirectionModel, extract_refusal_direction)
    report['safetune']=safetune.__version__
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,token=token,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,token=token,trust_remote_code=False,
       dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,device_map={'':'cuda:0'},attn_implementation='eager')
    model.eval()
    _cache=Path('/content/direction_cache.pt'); _hit=args.skip_done and _cache.exists()
    harmful_rows=[] if _hit else select_harmful_train(args.input,args.contrast_per_language)
    harmful=[r['instruction'] for r in harmful_rows]
    harmless=[] if _hit else load_harmless(args.contrast_per_language)
    report['contrast']={'harmful_per_language':args.contrast_per_language,'harmless_per_language':args.contrast_per_language,
      'harmful_ids':[r['id'] for r in harmful_rows],'harmless_source':'CohereLabs/aya_dataset (apache-2.0, ungated)'}
    # T4 has ~2GB free after the 6.7GB model; the extractor's default batch_size=16
    # OOMs on 262k-vocab logits. Patch the dataclass default down to 2.
    from safetune.core.runtime.inference import vector_extraction as _ve
    _orig_ve_init=_ve.VectorExtractionConfig.__init__
    def _patched_ve_init(self,**kw):
        kw.setdefault('batch_size',2)
        _orig_ve_init(self,**kw)
    _ve.VectorExtractionConfig.__init__=_patched_ve_init
    cfg=RefusalDirectionConfig(target_layers=list(range(7,29)),n_val=8,max_new_tokens=24)
    t0=time.time()
    if _hit:
        _d=torch.load(_cache,map_location='cpu'); direction=_d['direction'].to('cuda' if torch.cuda.is_available() else 'cpu'); layer_idx=_d['layer_idx']
    else:
        direction,layer_idx,_=extract_refusal_direction(model,tokenizer,harmful,harmless,cfg)
        torch.save({'direction':direction.detach().cpu(),'layer_idx':int(layer_idx)},_cache)
    report['extraction']={'cache':'hit' if _hit else 'extracted','selected_layer':int(layer_idx),'direction_norm':float(direction.norm()),
      'sweep_seconds':round(time.time()-t0,1),'target_layers':'7..28','n_val':8}
    _layers=[int(x) for x in args.layers.split(',')] if args.layers else [int(layer_idx)]
    report['steer_layers']=_layers
    wrapper=RefusalDirectionModel(model,direction,mode='steer',strength=args.strength,layers=_layers).install()
    resp_file=out/'responses_steer.jsonl'
    results=[json.loads(x) for x in resp_file.open()] if (args.skip_done and resp_file.exists()) else []
    done={x['id'] for x in results}
    rows_to_run=[r for r in test_rows if r['id'] not in done]
    print(f'RESUME: {len(done)} done, {len(rows_to_run)} to run')
    try:
        for gi,r in enumerate(rows_to_run):
            text=tokenizer.apply_chat_template([{'role':'user','content':r['instruction']}],tokenize=False,add_generation_prompt=True)
            full=tokenizer(text,return_tensors='pt',add_special_tokens=False)
            original_tokens=full['input_ids'].shape[1]
            if original_tokens>512:
                results.append({'id':r['id'],'status':'skipped_input_too_long','input_tokens':original_tokens});continue
            inputs={k:v.to('cuda:0') for k,v in full.items()};start=time.time()
            with torch.inference_mode():
                tokens=model.generate(**inputs,max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id)
            response=tokenizer.decode(tokens[0,original_tokens:],skip_special_tokens=True)
            results.append({'id':r['id'],'language':r['metadata']['language'],'categories':r['metadata']['categories'],
             'prompt':r['instruction'],'response':response,'seconds':round(time.time()-start,3),
             'input_tokens':original_tokens,'output_tokens':tokens.shape[1]-original_tokens,'review_status':'UNREVIEWED',
             'possibly_length_limited':tokens.shape[1]-original_tokens==128})
            (out/'responses_steer.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results))
            if gi%10==9: torch.cuda.empty_cache()
    finally:
        wrapper.remove()
    report['status']='responses_generated_unreviewed';report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    report['responses']=sum('response' in x for x in results);report['skipped']=sum('response' not in x for x in results)
    (out/'run_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
