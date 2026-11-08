"""Draw Figures 8, 9 and 11 from networks reconstructed by the workflow."""
from __future__ import annotations
import argparse
import logging
import json
import re
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .common import ROOT, path, intermediate, parameter_file
from .products import table_path, figure_path
from .engine import CHICAGO, MAINLING, ABLATIONS, NAMES
from .artifacts import network, table_dir, read_csv

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                     'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
RED='#bc3036';BLUE='#326993';GRAY='#9da6ad';ORANGE='#db9b4b'


def save(fig, number):
    for suffix in ('png','pdf','svg'):
        fig.savefig(figure_path(number,suffix),dpi=320,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('WROTE',figure_path(number),flush=True)


def chicago_tracks():
    tracks=[]
    with ZipFile(path('chicago_tracks')) as archive:
        for name in sorted(archive.namelist()):
            if re.fullmatch(r'chicago/trips/trip_\d+\.txt',name):
                values=np.loadtxt(archive.open(name))
                tracks.append(np.atleast_2d(values)[:,:2])
    return tracks


def chicago_truth():
    with ZipFile(path('chicago_map')) as archive:
        vn=next(n for n in archive.namelist() if n.endswith('chicago_vertices_osm.txt'))
        en=next(n for n in archive.namelist() if n.endswith('chicago_edges_osm.txt'))
        vertices={}
        for line in archive.read(vn).decode().splitlines():
            f=re.split(r'[,\s]+',line.strip())
            if len(f)>=3:vertices[f[0]]=(float(f[1]),float(f[2]))
        result=[]
        for line in archive.read(en).decode().splitlines():
            f=re.split(r'[,\s]+',line.strip())
            if len(f)>=3 and f[1] in vertices and f[2] in vertices:result.append([vertices[f[1]],vertices[f[2]]])
    return result


def regional_tracks():
    tracks=[]
    for filename in sorted(path('mainling_clean').glob('*.csv')):
        frame=pd.read_csv(filename,usecols=['x_m','y_m'])
        points=frame[['x_m','y_m']].dropna().to_numpy(dtype=float)
        if len(points)>1:tracks.append(points)
    if not tracks:raise FileNotFoundError('Run Mainling trajectory preprocessing first.')
    return tracks


def lines_from_network(filename):
    import geopandas as gpd
    result=[]
    for geometry in gpd.read_file(filename,layer='edges').geometry:
        if geometry.geom_type=='LineString':result.append(np.array(geometry.coords)[:,:2])
        elif geometry.geom_type=='MultiLineString':result.extend(np.array(g.coords)[:,:2] for g in geometry.geoms)
    return result


def add_lines(ax, lines, color=RED, width=.8, alpha=1):
    collection=LineCollection(lines,colors=color,linewidths=width,alpha=alpha)
    ax.add_collection(collection)


def scale_bar(ax, bounds, metres):
    x0,y0,x1,y1=bounds;x=x0+.04*(x1-x0);y=y0+.06*(y1-y0)
    ax.plot([x,x+metres],[y,y],color='black',lw=1.5)
    ax.text(x,y+.018*(y1-y0),f'{metres:g} m',fontsize=7,va='bottom',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))


def frame(ax, bounds):
    ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#88939c');spine.set_linewidth(.5)


def chicago_comparison():
    truth=chicago_truth();networks=[lines_from_network(network('chicago',m)) for m in CHICAGO]
    points=np.vstack([line for lines in networks for line in lines]);low=points.min(axis=0)-250;high=points.max(axis=0)+250
    bb=[*low,*high];fig,axes=plt.subplots(2,3,figsize=(10,6.8))
    for index,(method,lines,ax) in enumerate(zip(CHICAGO,networks,axes.flat)):
        add_lines(ax,truth,GRAY,.25,.6);add_lines(ax,lines,RED if method=='proposed' else BLUE,.7)
        frame(ax,bb);scale_bar(ax,bb,1000);ax.set_title(f'({chr(97+index)}) {NAMES[method]}',fontsize=11)
    fig.tight_layout();save(fig,8)


def regional_comparison():
    import geopandas as gpd
    scenes=json.loads(parameter_file('scenes.json').read_text(encoding='utf-8'))['scenes']
    tracks=regional_tracks();methods=('edelkamp','karagiorgou','biagioni','proposed')
    data={m:lines_from_network(network('mainling',m)) for m in methods}
    activity=intermediate('mainling')/'proposed/S3_Activity/activity_zones.gpkg'
    uturn=intermediate('mainling')/'proposed/S4_Gravity/uturn_consensus.gpkg'
    zones={}
    for index,file,layer in [(0,uturn,'uturn_protection_zones'),(1,activity,'activity_suppression_zones')]:
        if file.exists():zones[index]=gpd.read_file(file,layer=layer)
    fig,axes=plt.subplots(4,4,figsize=(10,10.8))
    for row,method in enumerate(methods):
        for col,scene in enumerate(scenes):
            ax=axes[row,col];bb=scene['bounds']
            if col in zones:
                subset=zones[col].cx[bb[0]:bb[2],bb[1]:bb[3]]
                if len(subset):subset.plot(ax=ax,color='#f4d47c',alpha=.4,edgecolor='none')
            visible=lambda line:not(line[:,0].max()<bb[0] or line[:,0].min()>bb[2] or line[:,1].max()<bb[1] or line[:,1].min()>bb[3])
            add_lines(ax,[t for t in tracks if visible(t)],GRAY,.35,.5)
            add_lines(ax,[t for t in data[method] if visible(t)],RED,.8)
            frame(ax,bb);scale_bar(ax,bb,20 if bb[2]-bb[0]<120 else 50)
            if row==0:ax.set_title(['Hairpin bends','Activity areas','Local connections','Single-trajectory route'][col],fontsize=10)
            if col==0:ax.set_ylabel(NAMES[method],fontsize=10)
    fig.tight_layout();save(fig,9)


def ablation_responses():
    rows=read_csv(table_path('ablation_metrics.csv'));full=next(r for r in rows if r['method']=='Full')
    keys=('length_km','edges','components','endpoints','junctions','cycle_rank')
    labels=('Road length','Edges','Components','Endpoints','Junctions','Cycle rank')
    fig,ax=plt.subplots(figsize=(8,4.8))
    for row,color in zip((r for r in rows if r['method']!='Full'),(BLUE,ORANGE,RED)):
        ax.scatter([float(row[k])/float(full[k]) for k in keys],np.arange(6),label=row['method'],s=35,color=color)
    ax.axvline(1,color='black',ls='--',lw=1,label='Full')
    ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xscale('log')
    ax.set_xlabel('Ratio to Full (log scale)');ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,fontsize=9)
    fig.tight_layout();save(fig,11)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('number',type=int)
    number=parser.parse_args(argv).number
    if number==8:chicago_comparison()
    elif number==9:regional_comparison()
    elif number==11:ablation_responses()
    else:raise ValueError('Choose Figure 8, 9 or 11; Figures 10 and 12 export map layers.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
