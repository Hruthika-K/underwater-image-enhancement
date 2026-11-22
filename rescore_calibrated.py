import csv,json,statistics
from pathlib import Path
cal=json.loads(Path('results/candidate_pipelines_calibrated/normalization.json').read_text())
# Normalization file was raw stats only, wrap to expected record structure if necessary.
if 'metrics' not in cal:
    cal={'n_tuning_images':150,'seed':42,'metrics':cal}
means=cal['metrics']
def score(row):
 return statistics.mean([
  (float(row['PSNR'])-means['PSNR']['raw_tuning_mean'])/means['PSNR']['raw_tuning_sd'],
  (float(row['SSIM'])-means['SSIM']['raw_tuning_mean'])/means['SSIM']['raw_tuning_sd'],
  (means['CIEDE2000']['raw_tuning_mean']-float(row['CIEDE2000']))/means['CIEDE2000']['raw_tuning_sd'],
  (float(row['UIQM'])-means['UIQM']['raw_tuning_mean'])/means['UIQM']['raw_tuning_sd']])
path=Path('results/candidate_pipelines/candidate_test_per_image.csv')
rows=list(csv.DictReader(path.open()))
summary={}
for r in rows:
 summary.setdefault(r['method'],[]).append((score(r),r))
out=[]
for m,vals in summary.items():
 item={'method':m,'n':len(vals),'raw_standardized_score_mean':statistics.mean(x[0] for x in vals)}
 for metric in ('PSNR','SSIM','CIEDE2000','UIQM','UCIQE'):
  item[metric+'_mean']=statistics.mean(float(x[1][metric]) for x in vals)
 out.append(item)
out.sort(key=lambda x:x['raw_standardized_score_mean'],reverse=True)
print('TEST METHODS RESCORED (exploratory; test was previously inspected)')
for x in out: print(x)
# Rerank candidate settings from saved tuning metrics (means; z-score is affine, so this is exact for the composite mean).
tune=list(csv.DictReader(Path('results/candidate_pipelines/candidate_tuning.csv').open()))
by={}
for r in tune:
 m={k:r[k+'_mean'] for k in ('PSNR','SSIM','CIEDE2000','UIQM')}; s=score(m); by.setdefault(r['family'],[]).append((s,r['config']))
print('TUNING CONFIG WINNERS under calibrated objective')
for f,vs in by.items():
 print(f, sorted(vs,reverse=True)[:3])
# Save an auditable rescore table and README.
base=Path('results/candidate_pipelines_calibrated')
with (base/'test_summary_rescored.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
with (base/'tuning_rescored.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['family','config','raw_standardized_score_mean'])
 for f,vs in by.items():
  for s,cfg in vs:w.writerow([f,cfg,f'{s:.9f}'])

