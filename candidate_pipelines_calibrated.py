from pathlib import Path
import csv, json, statistics
import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from candidate_pipelines import (RAW_DEFAULT, REF_DEFAULT, OUT_DEFAULT, pair_files, split_pairs,
    PIPES, SOG_DEFAULT, pipe_a, fixed_pipeline, build_configs, metrics_of, panel, write_csv)

OUT = Path(__file__).resolve().parent / 'results' / 'candidate_pipelines_calibrated'
SEED=42
TUNE_N=150

def raw_calibration(tune_pairs):
    rows=[]
    for src, refp in tune_pairs:
        raw, ref=cv2.imread(str(src)),cv2.imread(str(refp))
        ref_lab=rgb2lab(cv2.cvtColor(ref,cv2.COLOR_BGR2RGB))
        m=metrics_of(raw,ref,ref_lab,with_uciqe=False)
        rows.append({k:m[k] for k in ('PSNR','SSIM','CIEDE2000','UIQM')})
    out={}
    for k in ('PSNR','SSIM','CIEDE2000','UIQM'):
        v=np.array([r[k] for r in rows],dtype=float)
        out[k]={'raw_tuning_mean':float(v.mean()),'raw_tuning_sd':float(v.std(ddof=1))}
    return out

def score(m,cal):
    z=[(m['PSNR']-cal['PSNR']['raw_tuning_mean'])/cal['PSNR']['raw_tuning_sd'],
       (m['SSIM']-cal['SSIM']['raw_tuning_mean'])/cal['SSIM']['raw_tuning_sd'],
       (cal['CIEDE2000']['raw_tuning_mean']-m['CIEDE2000'])/cal['CIEDE2000']['raw_tuning_sd'],
       (m['UIQM']-cal['UIQM']['raw_tuning_mean'])/cal['UIQM']['raw_tuning_sd']]
    return float(np.mean(z))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs,_=pair_files(RAW_DEFAULT,REF_DEFAULT)
    tune_pairs,test_pairs=split_pairs(pairs,.70,SEED)
    tune_pairs=tune_pairs[:TUNE_N]
    cal=raw_calibration(tune_pairs)
    (OUT/'normalization.json').write_text(json.dumps({'method':'Each metric expressed as signed standard-deviation improvement over raw-input distribution on tuning set; equal-weight mean. For CIEDE2000, lower is better.','n_tuning_images':len(tune_pairs),'seed':SEED,'metrics':cal,'note':'No test values used to calculate normalization anchors.'},indent=2),encoding='utf-8')
    configs=build_configs(); sums=[[] for _ in configs]
    for j,(src,refp) in enumerate(tune_pairs,1):
        img,ref=cv2.imread(str(src)),cv2.imread(str(refp)); ref_lab=rgb2lab(cv2.cvtColor(ref,cv2.COLOR_BGR2RGB))
        for k,(family,p) in enumerate(configs):
            m=metrics_of(PIPES[family](img,p),ref,ref_lab,with_uciqe=False)
            sums[k].append((p,m,score(m,cal)))
        if j%10==0 or j==len(tune_pairs): print(f'Tune {j}/{len(tune_pairs)}',flush=True)
    tune_rows=[]; best={}
    for (family,p),vals in zip(configs,sums):
        s=float(np.mean([v[2] for v in vals]))
        tune_rows.append({'family':family,'config':json.dumps(p,sort_keys=True),'raw_standardized_score_mean':s})
        if family not in best or s>best[family][1]: best[family]=(p,s)
    write_csv(OUT/'tuning.csv',tune_rows)
    (OUT/'selected_configs.json').write_text(json.dumps({'objective':'maximize equal-weight mean signed raw-tuning-set standard-deviation improvement across PSNR, SSIM, CIEDE2000 and UIQM; anchors calculated from raw tuning images only','winners':{k:{'params':v[0],'tuning_score':v[1]} for k,v in best.items()}},indent=2),encoding='utf-8')
    methods={'raw':lambda im:im,'sog_default':lambda im:pipe_a(im,SOG_DEFAULT),'fixed_pipeline':fixed_pipeline}
    for f,(p,s) in best.items(): methods[f'best_{f}']=lambda im,f=f,p=p:PIPES[f](im,p)
    rows=[]
    for j,(src,refp) in enumerate(test_pairs,1):
        img,ref=cv2.imread(str(src)),cv2.imread(str(refp)); ref_lab=rgb2lab(cv2.cvtColor(ref,cv2.COLOR_BGR2RGB))
        for name,fn in methods.items():
            m=metrics_of(fn(img),ref,ref_lab)
            rows.append({'image':src.name,'method':name,**m,'raw_standardized_score':score(m,cal)})
        if j%10==0 or j==len(test_pairs): print(f'Test {j}/{len(test_pairs)}',flush=True)
    write_csv(OUT/'test_per_image.csv',rows)
    summary=[]
    for name in methods:
        vals=[r for r in rows if r['method']==name]
        item={'method':name,'n_images':len(vals)}
        for k in ('PSNR','SSIM','CIEDE2000','UIQM','UCIQE','raw_standardized_score'):
            arr=[r[k] for r in vals];item[k+'_mean']=statistics.mean(arr);item[k+'_std']=statistics.stdev(arr)
        summary.append(item)
    summary.sort(key=lambda r:r['raw_standardized_score_mean'],reverse=True)
    write_csv(OUT/'test_summary.csv',summary)
    (OUT/'README.md').write_text('# Calibrated composite experiment\n\nNormalization anchors are the mean and sample standard deviation of each raw-image metric over the 150 tuning images only. Each method metric is converted to a signed standardized improvement versus that raw tuning distribution; CIEDE2000 is reversed because lower is better. The four standardized values receive equal weight. This score is not bounded to 0–1; zero means at the raw tuning mean and +1 means one raw-tuning standard deviation better on average across metric dimensions. Test results are exploratory because this test split was previously inspected under the earlier score. The new score must not be described as an independent confirmatory evaluation.\n',encoding='utf-8')
    print('RESULTS',json.dumps(summary,indent=2))
if __name__=='__main__': main()
