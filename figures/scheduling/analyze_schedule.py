import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

METHODS=['No-DPS','Early','Uniform','Late']
COLORS={'No-DPS':'#444444','Early':'#999999','Uniform':'#2b63d9','Late':'#e23b3b'}
SLOTS={'No-DPS':[],'Early':[0,4,8,12],'Uniform':[0,10,20,30],'Late':[16,20,24,28]}

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    data=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=read_csv(data/'metrics.csv')
    groups={m:{int(r['case']):r for r in rows if r['method']==m} for m in METHODS}
    cases=sorted(groups['No-DPS'])
    assert all(sorted(groups[m])==cases for m in METHODS)
    assert len(rows)==4*len(cases)
    assert a.allow_partial or len(cases)==32
    for c in cases:assert len({groups[m][c]['index'] for m in METHODS})==1
    for r in rows:assert int(r['physical_forward_backward_calls'])==(0 if r['method']=='No-DPS' else 4) and int(r['evaluation_forward_calls'])==1
    fields=['fdtd_mse','fdtd_mae','fdtd_corr','model_rmse','model_mae','seconds']
    arr={m:{f:np.array([float(groups[m][c][f]) for c in cases]) for f in fields} for m in METHODS}
    summary=[]
    for m in METHODS:
        summary.append(dict(method=m,cases=len(cases),indices=','.join(str(i) for i in SLOTS[m]),mean_mse=float(arr[m]['fdtd_mse'].mean()),median_mse=float(np.median(arr[m]['fdtd_mse'])),mean_mae=float(arr[m]['fdtd_mae'].mean()),mean_corr=float(arr[m]['fdtd_corr'].mean()),mean_model_rmse=float(arr[m]['model_rmse'].mean()),time_mean_seconds=float(arr[m]['seconds'].mean()),time_sd_seconds=float(arr[m]['seconds'].std(ddof=1)),physics_calls=len(SLOTS[m]),evaluation_calls=1))
    write_csv(out/'Table_guidance_schedule.csv',summary)
    rng=np.random.default_rng(20261028);comparisons=[]
    for base,method in [('No-DPS','Early'),('No-DPS','Uniform'),('No-DPS','Late'),('Early','Late'),('Uniform','Late')]:
        d=arr[base]['fdtd_mse']-arr[method]['fdtd_mse'];ids=rng.integers(0,len(cases),size=(10000,len(cases)))
        comparisons.append(dict(baseline=base,method=method,mean_mse_gain=float(d.mean()),relative_reduction_of_mean_mse_pct=float(100*d.mean()/arr[base]['fdtd_mse'].mean()),improved_cases=int((d>0).sum()),worsened_cases=int((d<0).sum()),paired_bootstrap_mean_gain_95ci=np.quantile(d[ids].mean(axis=1),[.025,.975]).tolist()))
    report=dict(cases=len(cases),K=1,steps=32,sampling_seeds=1,corrected_modes_physics_calls=4,matched_step_size=0.004472135988319589,summary=summary,comparisons=comparisons,bootstrap='10000 paired case resamples, seed 20261028; intervals describe across-case variability for one trained model and sampling seed.')
    (out/'schedule_summary.json').write_text(json.dumps(report,indent=2))
    paired=[]
    for j,c in enumerate(cases):
        r=dict(case=c,index=groups['No-DPS'][c]['index'])
        for m in METHODS:
            for f in fields:r[m.lower().replace('-','_')+'_'+f]=float(arr[m][f][j])
        paired.append(r)
    write_csv(out/'paired_schedule_data.csv',paired)

    mpl.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'stix',
        'font.size':8.,'axes.linewidth':.75,'axes.labelsize':8.4,'xtick.labelsize':7.4,'ytick.labelsize':7.4,
        'legend.fontsize':7.4,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})
    fig,axes=plt.subplots(1,2,figsize=(6.25,2.65))
    fig.subplots_adjust(left=.12,right=.985,bottom=.25,top=.84,wspace=.44)
    ax=axes[0]
    for k,m in enumerate(METHODS):
        y=3-k
        ax.hlines(y,0,31,color='.82',lw=1,zorder=1)
        if SLOTS[m]:ax.scatter(SLOTS[m],[y]*4,s=25,color=COLORS[m],zorder=3)
    ax.set_yticks([3,2,1,0],METHODS);ax.set_ylim(-.55,3.55);ax.set_xlim(-1,32)
    ax.set_xticks([0,8,16,24,31]);ax.set_xlabel('Reverse-sampling step')
    ax.set_title('Guidance placement',fontsize=8.4,pad=9)
    ax.grid(True,axis='x',color='.9',lw=.45,alpha=.8)
    ax=axes[1];names=METHODS[1:]
    values=[100*(arr['No-DPS']['fdtd_mse']-arr[m]['fdtd_mse'])/arr['No-DPS']['fdtd_mse'] for m in names]
    box=ax.boxplot(values,widths=.45,patch_artist=True,showfliers=False,medianprops=dict(color='black',lw=1.1),whiskerprops=dict(color='.25',lw=.8),capprops=dict(color='.25',lw=.8))
    for patch,m in zip(box['boxes'],names):patch.set_facecolor(COLORS[m]);patch.set_alpha(.20);patch.set_edgecolor(COLORS[m]);patch.set_linewidth(1.1)
    jitter=np.random.default_rng(11).normal(0,.045,size=len(cases))
    for j in range(len(cases)):ax.plot(np.arange(1,4)+jitter[j],[v[j] for v in values],color='.78',lw=.4,alpha=.32,zorder=1)
    for k,(m,v) in enumerate(zip(names,values),start=1):ax.scatter(k+jitter,v,s=9,color=COLORS[m],alpha=.55,linewidths=0,zorder=3)
    ax.axhline(0,color='.35',lw=.65,ls='--');ax.set_xticks([1,2,3],names)
    ax.set_ylabel('MSE reduction (%)');ax.set_title(f'Paired waveform gains ($n={len(cases)}$)',fontsize=8.4,pad=9)
    ax.yaxis.set_major_locator(MaxNLocator(5));ax.grid(True,axis='y',color='.88',lw=.45,alpha=.8)
    for k,ax in enumerate(axes):
        ax.text(-.20,1.065,f'({chr(97+k)})',transform=ax.transAxes,fontweight='bold',fontsize=8.8,ha='left',va='bottom')
        ax.tick_params(length=3,width=.7)
    fig.canvas.draw();rects=[]
    for ax in axes:
        r=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted());rects.append([v*72 for v in [r.x0,r.y0,r.width,r.height]])
    diffs={key:abs(rects[0][j]-rects[1][j]) for key,j in [('bottom',1),('width',2),('height',3)]}
    assert all(v<1.5 for v in diffs.values())
    (out/'Figure8_guidance_schedule.alignment.json').write_text(json.dumps(dict(rectangles_pt=rects,differences_pt=diffs,tolerance_pt=1.5,passed=True,check='Independent rendered Matplotlib geometry audit'),indent=2))
    for ext in ['pdf','svg','png']:fig.savefig(out/('Figure8_guidance_schedule.'+ext),dpi=600,facecolor='white')
    plt.close(fig)
    import fitz
    doc=fitz.open(out/'Figure8_guidance_schedule.pdf');spans=[];lines=[]
    for b in doc[0].get_text('dict')['blocks']:
        for l in b.get('lines',[]):
            if not l.get('spans'):continue
            lines.append(dict(text=''.join(s['text'] for s in l['spans']),bbox=l['bbox']));spans.extend(l['spans'])
    clipped=[s['text'] for s in spans if s['bbox'][0]<-.5 or s['bbox'][1]<-.5 or s['bbox'][2]>doc[0].rect.width+.5 or s['bbox'][3]>doc[0].rect.height+.5]
    small=[s['text'] for s in spans if s['text'].strip() and s['size']<4.99]
    collisions=[]
    for j,a0 in enumerate(lines):
        for b0 in lines[j+1:]:
            x=min(a0['bbox'][2],b0['bbox'][2])-max(a0['bbox'][0],b0['bbox'][0]);y=min(a0['bbox'][3],b0['bbox'][3])-max(a0['bbox'][1],b0['bbox'][1])
            if x>.8 and y>.8:collisions.append([a0['text'],b0['text']])
    qa=dict(fonts=sorted({s['font'] for s in spans}),minimum_glyph_pt=min(s['size'] for s in spans if s['text'].strip()),clipped_text=clipped,small_glyphs=small,text_line_overlap_candidates=collisions,rows_used=len(rows),excluded_rows=0,
        audit='Independent Matplotlib geometry and PyMuPDF text checks plus visual inspection. Bundled skill auditors remain unavailable due to OS file permissions.')
    (out/'Figure8_guidance_schedule.QA.json').write_text(json.dumps(qa,indent=2));assert not clipped and not small,qa
    tex=[r'\begin{table}[pos=htbp]',r'\centering',r'\tablesetup',r'\begin{threeparttable}',
        r'\caption{Fixed-budget guidance-schedule ablation on '+str(len(cases))+r' held-out cases with $K=1$.}',r'\label{tab:guidance_schedule}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l r r r r r r@{}}',r'\toprule',
        r'Schedule & \makecell{Mean\\MSE} & \makecell{Median\\MSE} & MAE & $r$ & $\mathrm{RMSE}_m$ & \makecell{Time\\(s/case)} \\',r'\midrule']
    for r in summary:
        tex.append(f"{r['method']} & {1000*r['mean_mse']:.3f} & {1000*r['median_mse']:.3f} & {r['mean_mae']:.4f} & {r['mean_corr']:.4f} & {r['mean_model_rmse']:.4f} & ${r['time_mean_seconds']:.2f}\\pm{r['time_sd_seconds']:.2f}$"+r' \\')
    tex += [r'\bottomrule',r'\end{tabular*}',r'\begin{tablenotes}[flushleft]',r'\footnotesize',
        r'\item MSE entries are in units of $10^{-3}$. $r$ and $\mathrm{RMSE}_m$ are mean waveform correlation and mean per-case permittivity RMSE. All cases follow the original fixed test ordering.',
        r'\item Each guided schedule uses four FDTD forward/backward corrections and one final forward evaluation. The zero-based correction indices are early $(0,4,8,12)$, uniform $(0,10,20,30)$, and late $(16,20,24,28)$. The late schedule is the default EM-DPS schedule.',
        r'\item Objective weights, gradient normalization, clipping, bounds and the four ordered update sizes are identical across guided schedules. Each case shares one initial noise field across all treatments; $K=1$ removes candidate-selection effects.',
        r'\item Times are mean $\pm$ sample standard deviation across all '+str(len(cases))+r' cases, measured with CUDA synchronization on an RTX 3090. Setup, warm-up and file I/O are excluded. Treatment execution order rotates across cases.',
        r'\end{tablenotes}',r'\end{threeparttable}',r'\end{table}']
    (out/'Table_guidance_schedule.tex').write_text('\n'.join(tex)+'\n')
    caption=r'''\begin{figure}[pos=htbp]
\centering
\includegraphics[width=0.96\linewidth]{Figure8_guidance_schedule.pdf}
\caption{Effect of guidance placement at a fixed physical budget. (a) Correction indices within the 32-step sampling trajectory. Each guided schedule uses four corrections with the same ordered update sizes. (b) Per-case waveform MSE reductions relative to unguided sampling for CASECOUNT held-out observations. Points show all cases; thin lines connect the same observation across schedules. Boxes show medians and interquartile ranges, with whiskers extending to the most extreme observations within 1.5 interquartile ranges. All treatments use a single candidate, so the comparison excludes multi-candidate ranking effects.}
\label{fig:guidance_schedule}
\end{figure}
'''.replace('CASECOUNT',str(len(cases)))
    (out/'Figure_guidance_schedule.tex').write_text(caption)
    print(json.dumps(report,indent=2));print('FIGURE_QA',json.dumps(qa))

if __name__=='__main__':main()
