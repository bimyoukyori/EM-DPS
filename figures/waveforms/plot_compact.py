import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import fitz

CASES=[10,24]
SLOTS=[0,16,28]
METHODS=['No-DPS','Early','Uniform','Late']
KEYS=['no_dps','early','uniform','late']
FIELD_EXTENT=[0,1.5,.5,0]

def style():
    mpl.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8.,'axes.linewidth':.65,
        'axes.labelsize':7.4,'xtick.labelsize':7.4,'ytick.labelsize':7.4,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})

def axis_style(ax):
    ax.tick_params(length=2.5,width=.65,pad=2)
    for sp in ax.spines.values():sp.set_linewidth(.65)

def pdf_check(path):
    doc=fitz.open(path);page=doc[0];spans=[];lines=[]
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines',[]):
            if l.get('spans'):
                spans.extend(l['spans']);lines.append(dict(text=''.join(s['text'] for s in l['spans']),bbox=l['bbox']))
    clipped=[s['text'] for s in spans if s['text'].strip() and not page.rect.contains(fitz.Rect(s['bbox']))]
    small=[s['text'] for s in spans if s['text'].strip() and s['size']<5]
    overlap=[]
    for i,a in enumerate(lines):
        for b in lines[i+1:]:
            x=min(a['bbox'][2],b['bbox'][2])-max(a['bbox'][0],b['bbox'][0])
            y=min(a['bbox'][3],b['bbox'][3])-max(a['bbox'][1],b['bbox'][1])
            if x>.8 and y>.8:overlap.append([a['text'],b['text']])
    qa=dict(minimum_glyph_pt=min(s['size'] for s in spans if s['text'].strip()),
            fonts=sorted({s['font'] for s in spans}),clipped_text=clipped,
            small_glyphs=small,text_line_overlap_candidates=overlap,
            audit='Independent PDF text geometry; bundled skill audit scripts unavailable under OS file permissions.')
    path.with_suffix('.QA.json').write_text(json.dumps(qa,indent=2))
    assert not clipped and not small and not overlap,qa

def save(fig,path,groups,reason):
    fig.canvas.draw();rects=[]
    axes=[a for group in groups for a in group]
    for ax in axes:
        b=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        rects.append([q*72 for q in [b.x0,b.y0,b.width,b.height]])
    group_checks=[];offset=0
    for group in groups:
        r=rects[offset:offset+len(group)];offset+=len(group)
        deviations={k:float(np.ptp([a[i] for a in r])) for k,i in [('bottom',1),('width',2),('height',3)]}
        if len(r)>2:
            gaps=[r[i+1][0]-r[i][0]-r[i][2] for i in range(len(r)-1)]
            deviations['gutter']=float(np.ptp(gaps))
        assert all(d<1.5 for d in deviations.values()),deviations
        group_checks.append(deviations)
    path.with_suffix('.alignment.json').write_text(json.dumps(dict(rectangles_pt=rects,
        groups=group_checks,tolerance_pt=1.5,passed=True,
        exemptions=reason,audit='Independent rendered Matplotlib plot-area geometry'),indent=2))
    for ext in ['pdf','svg','png']:fig.savefig(path.with_suffix('.'+ext),facecolor='white',dpi=600,bbox_inches='tight',pad_inches=.035)
    plt.close(fig);pdf_check(path.with_suffix('.pdf'))

def write_float(path,caption,label):
    path.with_suffix('.tex').write_text('\n'.join([r'\begin{figure}[pos=htbp]',r'\centering',
        r'\includegraphics[width=0.96\linewidth]{'+path.name+'.pdf}',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{figure}','']))

def mechanism(case,data,rows,out,limit):
    a=np.load(data/f'case{case:03d}_isolated.npz')
    records={int(r['slot']):r for r in rows if int(r['case'])==case}
    fig=plt.figure(figsize=(7.1,3.4));axes=[]
    
    left=.095;right=.835;gap=.045;width=(right-left-2*gap)/3
    height=width*7.1/(3*3.4)
    bottoms=[2.32/3.4,1.45/3.4,.53/3.4]
    stages=['Early: step 0','Middle: step 16','Late: step 28']
    keys=['state_delta','immediate_delta','final_delta']
    headings=['Applied state update','Immediate model response','Final model response']
    for row,(key,heading) in enumerate(zip(keys,headings)):
        group=[]
        y=bottoms[row]
        fig.text(left,y+height+.08/3.4,f'({chr(97+row)})  {heading}',fontsize=8.4,fontweight='bold',ha='left')
        for col,slot in enumerate(SLOTS):
            ax=fig.add_axes([left+col*(width+gap),y,width,height]);group.append(ax)
            im=ax.imshow(a[f'{key}_s{slot:02d}'],extent=FIELD_EXTENT,cmap='RdBu_r',
                vmin=-limit,vmax=limit,interpolation='nearest',aspect='equal')
            ax.set_xticks([0,.75,1.5]);ax.set_yticks([0,.25,.5])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
            if col==0:ax.set_ylabel('Depth (m)',labelpad=3)
            else:ax.set_yticklabels([])
            if row<2:ax.set_xticklabels([])
            if row==0:
                noise={0:4.2879195213,16:.1990345567,28:.00478642946}[slot]
                center=left+col*(width+gap)+width/2
                fig.text(center,3.28/3.4,stages[col],ha='center',va='center',fontsize=8.4)
                fig.text(center,3.105/3.4,f'Noise = {noise:.4g}',ha='center',va='center',fontsize=7.4)
            if row==1:
                gain=float(records[slot]['immediate_to_state_norm'])
                ax.text(.5,-.10,f'Response gain: {gain:.1%}',transform=ax.transAxes,ha='center',va='top',fontsize=7.4)
            if row==2:
                cosine=float(records[slot]['immediate_final_cosine'])
                ax.text(.5,-.43,f'Cosine: {cosine:.5f}',transform=ax.transAxes,ha='center',va='top',fontsize=7.4)
                center=left+col*(width+gap)+width/2
                fig.text(center,.09/3.4,'Distance (m)',ha='center',va='center',fontsize=7.4)
            axis_style(ax)
        axes.append(group)
    cax=fig.add_axes([.88,.53/3.4,.014,2.32/3.4+height-.53/3.4]);cb=fig.colorbar(im,cax=cax,orientation='vertical')
    cb.set_ticks([-limit,0,limit]);cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    cb.set_label('Change (relative-permittivity scale)',fontsize=7.4,labelpad=3)
    cb.ax.tick_params(length=2.5,width=.65,pad=2)
    path=out/f'Figure_update_transfer_case{case:03d}'
    save(fig,path,axes,['Colorbar excluded from comparison groups; all nine model maps have equal physical aspect and dimensions.'])
    caption=(r'Transfer of an isolated electromagnetic correction through the denoiser and the remaining sampling trajectory. '
        r'Columns show corrections at steps 0, 16 and 28 for held-out case '+str(case)+r'. '
        r'(a) Actual projected state changes, multiplied by the training standard deviation to express them on the relative-permittivity scale. '
        r'(b) Immediate changes in the denoised model, comparing denoising of the updated and original states at the same noise level. '
        r'(c) Changes in the final reconstructed model relative to the common unguided trajectory. '
        r'Each intervention uses one correction with the same coefficient and bounds; all subsequent sampling steps are unguided. '
        r'All nine maps share one symmetric colour scale, with no saturation or per-panel normalization. '
        r'Response gain is the Euclidean norm of the immediate model change divided by the norm of the physical-scaled state change. '
        r'Direction retention is the cosine similarity between the immediate and final model-change fields. '
        r'These interventions diagnose correction transfer; they are separate from the four-correction schedule comparison.')
    write_float(path,caption,f'fig:update_transfer_{case}')

def model_comparison(case,data,out):
    a=np.load(data/f'case{case:03d}_spatial.npz')
    fig=plt.figure(figsize=(7.1,2.95));groups=[]
    width=.365;height=width*7.1/(3*2.95)
    keys=['eps_true','eps_no_dps','eps_late',None]
    titles=['True permittivity','Unguided reconstruction','Guided reconstruction (EM-DPS)','Guided minus unguided']
    for row,y in enumerate([1.76/2.95,.43/2.95]):
        group=[]
        for col,x in enumerate([.07,.485]):
            k=row*2+col;ax=fig.add_axes([x,y,width,height]);group.append(ax)
            delta=keys[k] is None
            values=a['eps_late']-a['eps_no_dps'] if delta else a[keys[k]]
            im=ax.imshow(values,extent=FIELD_EXTENT,aspect='equal',interpolation='nearest',
                cmap='RdBu_r' if delta else 'turbo',vmin=-.32 if delta else 3,vmax=.32 if delta else 11)
            if delta:im_delta=im
            else:im_eps=im
            ax.set_xticks([0,.75,1.5]);ax.set_yticks([0,.25,.5])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
            if row==1:ax.set_xlabel('Distance (m)',labelpad=2)
            else:ax.set_xticklabels([])
            if col==0:ax.set_ylabel('Depth (m)',labelpad=2)
            else:ax.set_yticklabels([])
            ax.set_title(titles[k],fontsize=8.4,pad=9)
            ax.text(-.12,1.15,f'({chr(97+k)})',transform=ax.transAxes,fontsize=8.8,fontweight='bold')
            axis_style(ax)
        groups.append(group)
    for y,image,label,ticks in [(1.76/2.95,im_eps,'Relative permittivity',[3,7,11]),(.43/2.95,im_delta,'Permittivity change',[-.32,0,.32])]:
        cb=fig.colorbar(image,cax=fig.add_axes([.90,y,.014,height]),orientation='vertical')
        cb.set_ticks(ticks);cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        cb.set_label(label,fontsize=7.4,labelpad=3);cb.ax.tick_params(length=2.5,width=.65,pad=2)
    path=out/f'Figure_model_comparison_case{case:03d}'
    save(fig,path,groups,['Colorbars excluded; all four models retain equal dimensions and physical aspect.'])
    caption=(r'Model comparison for held-out case '+str(case)+r'. (a) True relative permittivity. '
        r'(b) Unguided conditional-diffusion reconstruction. (c) EM-DPS reconstruction with the default late guidance schedule. '
        r'(d) Guided-minus-unguided permittivity difference. Panels (a--c) share the same absolute-permittivity scale; '
        r'panel (d) uses the separate symmetric change scale. Reconstructions use the same initial noise and one candidate ($K=1$). '
        r'The arrays are the original schedule-ablation outputs, displayed without per-panel rescaling or colour saturation.')
    write_float(path,caption,f'fig:model_comparison_{case}')

def waveforms(case,data,out,full=False):
    a=np.load(data/f'case{case:03d}_spatial.npz')
    scores=json.loads((data/f'case{case:03d}_spatial_metrics.json').read_text())
    end=1697 if full else 850
    fig=plt.figure(figsize=(7.1,2.65));axes=[]
    left=.075;right=.975;gap=.019;width=(right-left-4*gap)/5
    for col in range(5):
        ax=fig.add_axes([left+col*(width+gap),.75/2.65,width,1.48/2.65]);axes.append(ax)
        values=a['obs_e'] if col==0 else a['res_'+KEYS[col-1]]
        im=ax.imshow(values[:end],origin='upper',interpolation='nearest',aspect='auto',
            extent=[-.5,42.5,end-.5,-.5],cmap='gray' if col==0 else 'RdBu_r',
            vmin=-1 if col==0 else -1.2,vmax=1 if col==0 else 1.2)
        if col==0:im_obs=im
        else:im_res=im
        ax.set_xticks([0,21,42]);ax.set_yticks([0,800,1600] if full else [0,400,800])
        ax.set_xlabel('Trace index',labelpad=2)
        if col==0:ax.set_ylabel('Time sample',labelpad=2)
        else:ax.set_yticklabels([])
        title='Observed' if col==0 else METHODS[col-1]+' residual'
        ax.set_title(title,fontsize=7.8,pad=17)
        if col>0:ax.text(.5,1.025,f'MSE: {1000*scores[col-1]["fdtd_mse"]:.3f}e-3',
            transform=ax.transAxes,ha='center',va='bottom',fontsize=7.4)
        ax.text(-.13,1.16,f'({chr(97+col)})',transform=ax.transAxes,fontsize=8.8,fontweight='bold')
        axis_style(ax)
    for pos,image,label,ticks in [([.086,.29/2.65,.13,.045/2.65],im_obs,'Amplitude',[-1,0,1]),
        ([.32,.29/2.65,.57,.045/2.65],im_res,'Simulated minus observed (shared scale)',[-1.2,0,1.2])]:
        cb=fig.colorbar(image,cax=fig.add_axes(pos),orientation='horizontal')
        cb.set_ticks(ticks);cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        cb.set_label(label,fontsize=7.4,labelpad=2);cb.ax.tick_params(length=2.5,width=.65,pad=2)
    path=out/(f'Figure_waveform_residuals_full_case{case:03d}' if full else f'Figure_waveform_residuals_case{case:03d}')
    save(fig,path,[axes],['Both bottom colourbars excluded; all five radargrams have equal dimensions.'])
    caption=(r'Waveform evidence for held-out case '+str(case)+r'. (a) Processed observed B-scan. '
        r'(b--e) Processed simulated-minus-observed B-scans for unguided, early, uniform and default late-guided sampling, respectively. '
        r'All residuals share one fixed symmetric colour scale, also fixed across the selected cases. All 43 traces are shown. '
        r'MSE labels are computed from the full 1697-sample records. ')
    if full:caption+=r'The complete acquisition window is displayed.'
    else:caption+=(r'The displayed 850-sample window follows the previously fixed common 99.9\% cumulative-energy cutoff '
        r'across both observations and their residuals, rounded up to 50 samples. Full-window plots are supplied separately.')
    write_float(path,caption,f'fig:waveform_residuals_{"full_" if full else ""}{case}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();data=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'manifest.json').read_text())
    assert all(hashlib.sha256((data/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
    rows=list(csv.DictReader((data/'state_response_metrics.csv').open()))
    style()
    for case in CASES:
        mechanism(case,data,rows,out,.13)
        model_comparison(case,data,out)
        waveforms(case,data,out)
        waveforms(case,data,out,full=True)
    (out/'revision_metadata.json').write_text(json.dumps(dict(cases=CASES,
        source_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
        data_changed=False,mechanism_colourbar='right, shared vertical',
        model_panels=['True','Unguided','Guided (EM-DPS, late)','Guided minus unguided'],
        waveforms='separate figure, unchanged bottom horizontal colourbars',
        fixed_limits=dict(response=[-.13,.13],permittivity=[3,11],difference=[-.32,.32],residual=[-1.2,1.2])),indent=2))
    print('DONE: 3 revised figure types for both cases; full-window waveform views retained.')

if __name__=='__main__':main()
