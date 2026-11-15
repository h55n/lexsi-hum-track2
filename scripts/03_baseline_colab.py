"""Run in an approved FREE GPU runtime. No remote write or paid API.
Token comes from HF_TOKEN env or Colab Secrets, never from this file.
Evaluates only frozen test inputs. Not a safety score: responses require review.
"""
import argparse, hashlib, json, os, platform, time
from pathlib import Path
MODEL='CohereLabs/tiny-aya-fire'
REVISION='73500c80fcb6c8bea3e830c21d5d2920249a0e1a'
def select_rows(path,per_language):
    rows=[json.loads(x) for x in Path(path).open()]
    eligible=[r for r in rows if r['metadata']['split']=='test']
    result=[]
    for lang in ['en','hi','bn']:
        subset=[r for r in eligible if r['metadata']['language']==lang]
        subset.sort(key=lambda r:hashlib.sha256(('baseline-v1:'+r['id']).encode()).hexdigest())
        result.extend(subset[:per_language])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='eval/curated.jsonl');p.add_argument('--out',default='eval/baseline');p.add_argument('--per-language',type=int,default=2);p.add_argument('--preflight-only',action='store_true');p.add_argument('--skip-done',action='store_true');args=p.parse_args()
    import torch, transformers
    rows=select_rows(args.input,args.per_language)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report={'model':MODEL,'revision':REVISION,'python':platform.python_version(),'torch':torch.__version__,
      'transformers':transformers.__version__,'cuda':torch.cuda.is_available(),'selected_ids':[r['id'] for r in rows],
      'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),'max_input_tokens':512,'max_new_tokens':128,
      'do_sample':False,'status':'preflight','caveat':'No safety metric without reviewed annotations; truncation recorded.'}
    if report['cuda']:
        report['gpu']=torch.cuda.get_device_name(0);report['total_gpu_bytes']=torch.cuda.get_device_properties(0).total_memory
    (out/'run_manifest.json').write_text(json.dumps(report,indent=2))
    if args.preflight_only:print(json.dumps(report,indent=2));return
    if not report['cuda']:raise RuntimeError('GPU required for this baseline plan. Do not substitute paid inference.')
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
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,token=token,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,token=token,trust_remote_code=False,
       dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,device_map={'':'cuda:0'},attn_implementation='eager')
    model.eval();report['status']='running';report['peak_cuda_bytes']=0
    resp_file=out/'responses.jsonl'
    results=[json.loads(x) for x in resp_file.open()] if (args.skip_done and resp_file.exists()) else []
    done={x['id'] for x in results}
    rows_to_run=[r for r in rows if r['id'] not in done]
    print(f'RESUME: {len(done)} done, {len(rows_to_run)} to run')
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
        (out/'responses.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results))
        if gi%10==9: torch.cuda.empty_cache()
    report['status']='responses_generated_unreviewed';report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    report['responses']=sum('response' in x for x in results);report['skipped']=sum('response' not in x for x in results)
    (out/'run_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
