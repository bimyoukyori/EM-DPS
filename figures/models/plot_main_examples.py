import argparse,csv,hashlib,json

from pathlib import Path

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

from matplotlib.ticker import FormatStrFormatter

import fitz

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

def model_comparison(case,data,out):
    a=np.load(data/f'case{case:03d}_spatial.npz')
    eps_min=float(np.floor(min(a[k].min() for k in ['eps_true','eps_no_dps','eps_late'])))
    eps_max=float(np.ceil(max(a[k].max() for k in ['eps_true','eps_no_dps','eps_late'])))
    change_limit=float(np.ceil(np.max(np.abs(a['eps_late']-a['eps_no_dps']))*100)/100)
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
                cmap='RdBu_r' if delta else 'turbo',vmin=-change_limit if delta else eps_min,vmax=change_limit if delta else eps_max)
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
    for y,image,label,ticks in [(1.76/2.95,im_eps,'Relative permittivity',[eps_min,(eps_min+eps_max)/2,eps_max]),(.43/2.95,im_delta,'Permittivity change',[-change_limit,0,change_limit])]:
        cb=fig.colorbar(image,cax=fig.add_axes([.90,y,.014,height]),orientation='vertical')
        cb.set_ticks(ticks);cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        cb.set_label(label,fontsize=7.4,labelpad=3);cb.ax.tick_params(length=2.5,width=.65,pad=2)
    path=out/f'Figure_model_comparison_case{case:03d}'
    save(fig,path,groups,['Colorbars excluded; all four models retain equal dimensions and physical aspect.'])
    caption=(r'Model comparison for held-out case '+str(case)+r'. (a) True relative permittivity. '
        r'(b) Unguided conditional-diffusion reconstruction. (c) EM-DPS reconstruction with the default late guidance schedule. '
        r'(d) Guided-minus-unguided permittivity difference. Panels (a--c) share the same absolute-permittivity scale; '
        r'panel (d) uses the separate symmetric change scale. Each method generates four candidates ($K=4$); the displayed candidate is selected by the original processed-waveform MSE ranking. The selected candidate indices can differ between methods. '
        r'The arrays are the original schedule-ablation outputs, displayed without per-panel rescaling or colour saturation.')
    write_float(path,caption,f'fig:model_comparison_{case}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();data=Path(args.data);out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((data/'manifest.json').read_text())
    assert all(hashlib.sha256((data/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
    style()
    for case in [14,35,49,126]:model_comparison(case,data,out)
    print('DONE four original K=4 examples')

if __name__=='__main__':main()
