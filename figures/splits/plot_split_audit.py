from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

ROOT=Path(__file__).resolve().parent
from plot_utils import require_matplotlib_panel_alignment

def rows(name):
    with (ROOT/'raw'/name).open(newline='') as f:return list(csv.DictReader(f))

def main():
    for name,sha in json.loads((ROOT/'raw/manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/'raw'/name).read_bytes()).hexdigest()==sha,name
    comp=rows('composition.csv'); nn=rows('nearest_training.csv')
    report=json.loads((ROOT/'raw/report.json').read_text())
    indices=np.load(ROOT/'raw/split_indices.npz')
    train=set(indices['train'].tolist())
    assert len(train)==14400
    for split in ['val','test']:
        cases=[r for r in nn if r['split']==split]
        assert len(cases)==1800 and set(int(r['h5_index']) for r in cases)==set(indices[split])
        assert not train.intersection(indices[split])
        assert all(int(r[key]) in train for r in cases for key in ['model_nearest_h5_index','data_nearest_h5_index'])
    assert not set(indices['val']).intersection(indices['test'])
    mpl.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':8.,'axes.linewidth':.65,
        'axes.labelsize':7.4,'xtick.labelsize':7.4,'ytick.labelsize':7.4,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,
        'path.simplify':False})
    fig=plt.figure(figsize=(7.1,4.45))
    axes=[]
    for y in [.585,.105]:
        for x in [.095,.585]:axes.append(fig.add_axes([x,y,.375,.31]))
    titles=['Dataset partitions','Source-subset composition',
            'Model-space nearest neighbours','Data-space nearest neighbours']
    for i,ax in enumerate(axes):
        ax.set_label(chr(97+i))
        ax.tick_params(length=2.5,width=.65,pad=2)
        ax.spines[['top','right']].set_visible(False)
        ax.text(-.18,1.24,f'({chr(97+i)})',transform=ax.transAxes,
                weight='bold',fontsize=8.8,va='bottom')
        ax.text(.5,1.24,titles[i],transform=ax.transAxes,
                fontsize=8.4,va='bottom',ha='center')
    labels=['Training','Validation','Test']; names=['train','val','test']
    counts=[report['sizes'][k] for k in names]
    axes[0].bar(range(3),counts,width=.58,color=['#2b63d9','#7f9fd7','#a6b7ce'])
    for i,n in enumerate(counts):axes[0].text(i,n+430,f'{n:,}',ha='center',va='bottom',fontsize=7.4)
    axes[0].set(xticks=range(3),xticklabels=labels,ylim=(0,16700),yticks=[0,4000,8000,12000,16000],ylabel='Number of cases')
    axes[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axes[0].text(.97,.79,'Test evaluations\nMain: 128\nRepeated FDTD: 64\n(nested within main)',
                 transform=axes[0].transAxes,ha='right',va='top',fontsize=7.0,linespacing=1.25)
    palette=['#315b91','#7798bd','#aabdd1','#c4b4b5','#bd8a79','#9a5b50']
    bottom=np.zeros(3)
    for s,color in enumerate(palette,1):
        values=np.array([next(int(r['count']) for r in comp if r['split']==k and int(r['subset'])==s) for k in names])
        values=values/np.array(counts)*100
        axes[1].bar(range(3),values,bottom=bottom,width=.58,color=color,edgecolor='white',linewidth=.25)
        bottom+=values
    assert np.allclose(bottom,100)
    axes[1].set(xticks=range(3),xticklabels=labels,ylim=(0,100),yticks=[0,25,50,75,100],ylabel='Fraction of partition (%)')
    axes[1].legend(handles=[Patch(facecolor=c,label=f'Subset {i+1}') for i,c in enumerate(palette)],
        loc='lower center',bbox_to_anchor=(.5,1.005),ncol=3,frameon=False,fontsize=6.5,
        handlelength=1.2,columnspacing=1.1,handletextpad=.4,labelspacing=.3,borderaxespad=0)
    summaries={}
    for ax,key,space,dupekey in zip(axes[2:],['model_nrmse','data_nrmse'],
            ['permittivity','processed B-scan'],['permittivity','complete_pairs']):
        maxv=0
        for split,color,linestyle in [('val','#2b63d9','-'),('test','#e23b3b','--')]:
            v=np.sort(np.array([float(r[key]) for r in nn if r['split']==split]))
            assert len(v)==1800 and np.isfinite(v).all() and (v>=0).all()
            ax.step(np.r_[0,v],np.r_[0,np.arange(1,len(v)+1)/len(v)],where='post',color=color,lw=1.05,ls=linestyle)
            summaries[f'{key}_{split}']={'min':float(v.min()),'median':float(np.median(v)),'max':float(v.max())}
            maxv=max(maxv,v.max())
        ax.set(xlim=(0,maxv*1.025),ylim=(0,1.02),yticks=[0,.25,.5,.75,1],
               xlabel=f'Nearest-training NRMSE ({space})',ylabel='Cumulative probability')
        handles=[Line2D([],[],color='#2b63d9',lw=1.05,label='Validation'),
                 Line2D([],[],color='#e23b3b',lw=1.05,ls='--',label='Test')]
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,
                  frameon=False,fontsize=7.0,handlelength=2,columnspacing=1.5,borderaxespad=0)
        n=report['duplicate_counts'][dupekey]['cross_split_pairs']
        ax.text(.97,.07,f'Exact cross-split duplicates: {n}',transform=ax.transAxes,
                fontsize=6.5,ha='right',va='bottom')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=ROOT/'panel_alignment.json',
        row_groups=[['a','b'],['c','d']],column_groups=[['a','c'],['b','d']])
    for filename in ['Figure_data_split_audit.pdf','Figure_data_split_audit.svg','Figure_data_split_audit.png']:
        fig.savefig(ROOT/filename,dpi=600,bbox_inches='tight',pad_inches=.035,facecolor='white')
    plt.close(fig)
    (ROOT/'numeric_summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
