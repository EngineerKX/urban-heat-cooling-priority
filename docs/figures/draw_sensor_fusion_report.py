"""Generate the Section 5.5 diagram and its HTML preview."""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(16, 11))
ax.set(xlim=(0,16), ylim=(0,11)); ax.axis('off')
fig.patch.set_facecolor('white')
def box(x,y,w,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,1.62,boxstyle='round,pad=0.12',facecolor=color,edgecolor='#acc0d2',linewidth=1.2))
    ax.text(x+.14,y+1.40,title,fontsize=12.5,weight='bold',color='#17334f',va='top')
    ax.text(x+.14,y+1.02,body,fontsize=10.5,color='#33465a',va='top',linespacing=1.5)
def arrow(x1,x2,y):
    ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle='-|>',mutation_scale=16,color='#6b8298',linewidth=1.6))
ax.text(.35,10.45,'Sensor fusion approach',fontsize=25,weight='bold',color='#17334f')
ax.text(.35,10.02,'From aligned thermal and spectral observations to heat estimates and subzone planning',fontsize=12,color='#53677c')
for x,t in [(.4,'INPUTS'),(5.7,'INTEGRATION PROCESS'),(11.25,'OUTPUTS')]:
    ax.text(x,9.5,t,fontsize=11,weight='bold',color='#53677c')
rows=[
(7.48,'1  Spatial and temporal alignment','Landsat temperature + Sentinel-2\nDifferent observation dates and grids',
 'Prepare compatible observations','Cloud masking; matching seasonal windows\nReprojection and grid alignment\nAggregate Sentinel-2 indices to 30 m',
 'Aligned modelling inputs','Comparable spatial support for training\nPrepared spectral and thermal layers','#edf4fb'),
(5.30,'2  Thermal and spectral fusion','Landsat temperature target\nSentinel-2 NDVI, NDBI and NDWI',
 'Regression downscaling','Fit the relationship on the 30 m grid\nApply at 10 m; interpolate coarse residuals\nAdd residuals to fine-grid predictions',
 'Estimated 10 m heat layer','Thermal information from Landsat\nSpatial predictors from Sentinel-2','#fff3e5'),
(3.12,'3  Land cover and heat modelling','Sentinel-2 bands and indices\nDynamic World training labels',
 'U-Net classification and CNN heat model','U-Net produces four land-cover classes\nCNN: 9 spectral/index + 4 class channels\nTarget: resampled Landsat temperature',
 'Land-cover and CNN outputs','Four-class land-cover map\nModelled heat on a 10 m grid\nInputs for local greening scenarios','#edf6ec'),
(.94,'4  Subzone integration','Baseline Landsat heat + land-cover fractions\nSingStat demographics + URA boundaries',
 'Aggregate and join by subzone','Scoring: heat, sensitivity, greenery deficit\nXGBoost: land cover, indices, demographics\nXGBoost target: subzone Landsat heat',
 'Planning outputs','Cooling-priority ranking\nSubzone temperature estimates\nIndicative greening scenarios','#f2edf8')]
for y,it,ib,mt,mb,ot,ob,c in rows:
    box(.4,y,4.5,it,ib,c);box(5.7,y,4.75,mt,mb,c);box(11.25,y,4.3,ot,ob,c)
    arrow(5.03,5.56,y+.8);arrow(10.58,11.11,y+.8)
ax.text(.4,.35,'10 m refers to the model output grid. It does not establish independently observed 10 m thermal detail.',fontsize=10,color='#53677c')
fig.savefig(OUT/'sensor_fusion_report.png',dpi=220,bbox_inches='tight',facecolor='white')
fig.savefig(OUT/'sensor_fusion_report.svg',bbox_inches='tight',facecolor='white')
plt.close(fig)
shutil.copyfile(OUT/'sensor_fusion_report.png',OUT/'fusion_diagram.png')
(OUT/'fusion_diagram.html').write_text('''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Sensor Fusion Approach | Section 5.5</title>
<style>body{margin:0;padding:24px;background:#eef2f6;font-family:Arial,sans-serif}main{max-width:1600px;margin:auto;background:white}img{display:block;width:100%;height:auto}nav{padding:16px}a{color:#17334f;margin-right:24px}@media print{body{padding:0;background:white}nav{display:none}}</style></head>
<body><main><img src="sensor_fusion_report.svg" alt="Four stages of sensor fusion: spatial and temporal alignment, thermal and spectral fusion, land-cover and heat modelling, and subzone integration."><nav><a href="sensor_fusion_report.png">Download PNG</a><a href="sensor_fusion_report.svg">Download SVG</a></nav></main></body></html>''',encoding='utf-8')
