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
    for ext in ['pdf','svg','png']:fig.savefig(path.with_suffix('.'+ext),facecolor='white',dpi=600)
    plt.close(fig);pdf_check(path.with_suffix('.pdf'))

def write_float(path,caption,label):
    path.with_suffix('.tex').write_text('\n'.join([r'\begin{figure}[pos=htbp]',r'\centering',
        r'\includegraphics[width=0.96\linewidth]{'+path.name+'.pdf}',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{figure}','']))

def mechanism(case,data,rows,out,limit):
    a=np.load(data/f'case{case:03d}_isolated.npz')
    records={int(r['slot']):r for r in rows if int(r['case'])==case}
    fig=plt.figure(figsize=(7.1,5.2));axes=[]
    
    left=.095;right=.975;gap=.045;width=(right-left-2*gap)/3
    height=width*7.1/(3*5.2)
    bottoms=[.75,.50,.25]
    stages=['Early: step 0','Middle: step 16','Late: step 28']
    keys=['state_delta','immediate_delta','final_delta']
    headings=['Applied state update','Immediate model response','Final model response']
    for row,(key,heading) in enumerate(zip(keys,headings)):
        group=[]
        y=bottoms[row]
        fig.text(left,y+height+.028,f'({chr(97+row)})  {heading}',fontsize=8.4,fontweight='bold',ha='left')
        for col,slot in enumerate(SLOTS):
            ax=fig.add_axes([left+col*(width+gap),y,width,height]);group.append(ax)
            im=ax.imshow(a[f'{key}_s{slot:02d}'],extent=FIELD_EXTENT,cmap='RdBu_r',
                vmin=-limit,vmax=limit,interpolation='nearest',aspect='equal')
            ax.set_xticks([0,.75,1.5]);ax.set_yticks([0,.25,.5])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
            if col==0:ax.set_ylabel('Depth (m)',labelpad=3)
            else:ax.set_yticklabels([])
            if row==0:
                noise={0:4.2879195213,16:.1990345567,28:.00478642946}[slot]
                center=left+col*(width+gap)+width/2
                fig.text(center,.97,stages[col],ha='center',va='center',fontsize=8.4)
                fig.text(center,.935,f'Noise = {noise:.4g}',ha='center',va='center',fontsize=7.4)
            if row==1:
                gain=float(records[slot]['immediate_to_state_norm'])
                ax.text(.5,-.47,f'Response gain: {gain:.1%}',transform=ax.transAxes,ha='center',va='top',fontsize=7.4)
            if row==2:
                cosine=float(records[slot]['immediate_final_cosine'])
                ax.text(.5,-.49,f'Direction retention: {cosine:.5f}',transform=ax.transAxes,ha='center',va='top',fontsize=7.4)
                center=left+col*(width+gap)+width/2
                fig.text(center,.153,'Distance (m)',ha='center',va='center',fontsize=7.4)
            axis_style(ax)
        axes.append(group)
    cax=fig.add_axes([.30,.095,.49,.014]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    cb.set_ticks([-limit,0,limit]);cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
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

def spatial(case,data,out,eps_limits,res_limit,delta_limit,window_end=1697,full=False):
    a=np.load(data/f'case{case:03d}_spatial.npz')
    scores=json.loads((data/f'case{case:03d}_spatial_metrics.json').read_text())
    fig=plt.figure(figsize=(7.1,4.9));top=[];bottom=[]
    left=.075;right=.975;gap=.043;w=(right-left-2*gap)/3
    h=w*7.1/(3*4.9)
    for col,(key,title) in enumerate(zip(['eps_true','eps_no_dps',None],['True permittivity','Unguided permittivity','Late minus unguided'])):
        ax=fig.add_axes([left+col*(w+gap),.735,w,h]);top.append(ax)
        values=a[key] if key else a['eps_late']-a['eps_no_dps']
        
        im=ax.imshow(values,extent=FIELD_EXTENT,aspect='equal',interpolation='nearest',
            cmap='turbo' if key else 'RdBu_r',vmin=eps_limits[0] if key else -delta_limit,
            vmax=eps_limits[1] if key else delta_limit)
        if key:im_eps=im
        else:im_delta=im
        ax.set_xticks([0,.75,1.5]);ax.set_yticks([0,.25,.5])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_xlabel('Distance (m)',labelpad=2)
        if col==0:ax.set_ylabel('Depth (m)',labelpad=2)
        else:ax.set_yticklabels([])
        ax.set_title(title,fontsize=8.4,pad=10)
        ax.text(-.13,1.16,f'({chr(97+col)})',transform=ax.transAxes,fontsize=8.8,fontweight='bold')
        axis_style(ax)
    
    for pos,image,label,ticks in [([.15,.634,.41,.013],im_eps,'Relative permittivity',[eps_limits[0],sum(eps_limits)/2,eps_limits[1]]),
                                 ([.73,.634,.20,.013],im_delta,'Permittivity change',[-delta_limit,0,delta_limit])]:
        cb=fig.colorbar(image,cax=fig.add_axes(pos),orientation='horizontal')
        cb.set_ticks(ticks);cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        cb.set_label(label,fontsize=7.4,labelpad=2);cb.ax.tick_params(length=2.5,width=.65,pad=2)
    gap=.025;w=(right-left-4*gap)/5
    for col in range(5):
        ax=fig.add_axes([left+col*(w+gap),.155,w,.345]);bottom.append(ax)
        values=a['obs_e'] if col==0 else a['res_'+KEYS[col-1]]
        im=ax.imshow(values[:window_end],origin='upper',interpolation='nearest',aspect='auto',
            extent=[-.5,42.5,window_end-.5,-.5],cmap='gray' if col==0 else 'RdBu_r',
            vmin=-1 if col==0 else -res_limit,vmax=1 if col==0 else res_limit)
        if col==0:im_obs=im
        else:im_res=im
        ax.set_xticks([0,21,42]);ax.set_yticks([0,800,1600] if full else [0,400,800])
        ax.set_xlabel('Trace index',labelpad=2)
        if col==0:ax.set_ylabel('Time sample',labelpad=2)
        else:ax.set_yticklabels([])
        title='Observed' if col==0 else METHODS[col-1]+' residual'
        ax.set_title(title,fontsize=7.8,pad=17)
        if col>0:
            mse=scores[col-1]['fdtd_mse']
            ax.text(.5,1.025,f'MSE: {1000*mse:.3f}e-3',transform=ax.transAxes,ha='center',va='bottom',fontsize=7.4)
        ax.text(-.13,1.16,f'({chr(100+col)})',transform=ax.transAxes,fontsize=8.8,fontweight='bold')
        axis_style(ax)
    for pos,image,label,ticks in [([.086,.058,.13,.013],im_obs,'Amplitude',[-1,0,1]),
                                 ([.32,.058,.57,.013],im_res,'Simulated minus observed (shared scale)',[-res_limit,0,res_limit])]:
        cb=fig.colorbar(image,cax=fig.add_axes(pos),orientation='horizontal')
        cb.set_ticks(ticks);cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        cb.set_label(label,fontsize=7.4,labelpad=2);cb.ax.tick_params(length=2.5,width=.65,pad=2)
    path=out/(f'Figure_spatial_waveform_full_case{case:03d}' if full else f'Figure_spatial_waveform_case{case:03d}')
    save(fig,path,[top,bottom],['Three equal physical-aspect model panels and five equal radargram panels intentionally occupy unequal-span rows; colorbars are excluded.'])
    caption=(r'Spatial evidence of waveform correction for held-out case '+str(case)+r'. '
        r'(a,b) True and unguided relative-permittivity models. '
        r'(c) Permittivity difference between the default late-guided reconstruction and the unguided reconstruction. '
        r'(d) Observed B-scan after the common evaluation preprocessing. '
        r'(e--h) Processed simulated-minus-observed B-scans for unguided, early, uniform and late guidance, respectively. '
        r'The four residual maps share one symmetric colour scale; model-change maps use a separate symmetric scale. '
        r'All maps use fixed amplitude scales without saturation or per-panel rescaling. '
        r'The reconstructions are the original single-candidate schedule-ablation outputs; the additional simulations reproduce their recorded MSE values. '
        r'This case was selected by its unguided error rank, before examining the gains of the guided schemes. '
        r'All 43 traces are displayed. ')
    if full:
        caption+=r'The full 1697-sample acquisition window is displayed.'
    else:
        caption+=(r'The displayed window contains the first '+str(window_end)+r' time samples. '
                  r'Its endpoint is the common, rounded-up 99.9\% cumulative-energy cutoff across both selected observations and all their residual maps. '
                  r'The MSE labels use the full records; full-window views and arrays are supplied separately.')
    label=f'fig:spatial_waveform_full_{case}' if full else f'fig:spatial_waveform_{case}'
    write_float(path,caption,label)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();data=Path(args.data);out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((data/'manifest.json').read_text())
    assert all(hashlib.sha256((data/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
    rows=list(csv.DictReader((data/'state_response_metrics.csv').open()))
    style()
    
    limits=[];deltas=[];residuals=[];eps=[];cutoffs=[]
    for case in CASES:
        i=np.load(data/f'case{case:03d}_isolated.npz');a=np.load(data/f'case{case:03d}_spatial.npz')
        assert a['obs_e'].shape==(1697,43)
        for slot in SLOTS:
            for k in ['state_delta','immediate_delta','final_delta']:limits.append(np.max(np.abs(i[f'{k}_s{slot:02d}'])))
        deltas.append(np.max(np.abs(a['eps_late']-a['eps_no_dps'])))
        residuals.extend(np.max(np.abs(a['res_'+k])) for k in KEYS)
        for k in ['obs_e']+['res_'+k for k in KEYS]:
            energy=np.sum(a[k].astype(np.float64)**2,axis=1)
            cutoffs.append(int(np.searchsorted(np.cumsum(energy),.999*energy.sum())+1))
        eps.extend([a['eps_true'].min(),a['eps_true'].max(),a['eps_no_dps'].min(),a['eps_no_dps'].max()])
    response_limit=float(np.ceil(max(limits)*100)/100)
    delta_limit=float(np.ceil(max(deltas)*100)/100)
    res_limit=float(np.ceil(max(residuals)*10)/10)
    eps_limits=[float(np.floor(min(eps))),float(np.ceil(max(eps)))]
    window_end=min(1697,int(np.ceil(max(cutoffs)/50)*50))
    for case in CASES:
        mechanism(case,data,rows,out,response_limit)
        spatial(case,data,out,eps_limits,res_limit,delta_limit,window_end=window_end)
        spatial(case,data,out,eps_limits,res_limit,delta_limit,full=True)
    contract=dict(question='How faithfully does a physical state update enter the model, and where does waveform mismatch decrease?',
        evidence={'update_transfer':'controlled isolated intervention at three sampling states',
                  'spatial_waveform':'original reconstructions and full processed waveform residuals'},
        selected_cases=CASES,selection='No-DPS MSE ranks 16 and 24 of 32, one-based; no gain-based selection.',
        response_range=[-response_limit,response_limit],permittivity_range=eps_limits,
        correction_range=[-delta_limit,delta_limit],residual_range=[-res_limit,res_limit],
        all_data_retained=True,focus_window_samples=window_end,full_window_samples=1697,
        focus_cutoff='Common rounded-up 99.9% cumulative energy cutoff across both observations and all residuals',
        focus_pixels_outside_window_per_image=(1697-window_end)*43,
        color_saturated_pixels=0,interpolation='nearest',
        fixed_limits_across_cases=True,original_reconstructions_changed=False,
        note='Two illustrative counterfactual cases do not establish general optimality or replace the 32-case schedule ablation.')
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2))
    print(json.dumps(contract,indent=2))

if __name__=='__main__':main()
