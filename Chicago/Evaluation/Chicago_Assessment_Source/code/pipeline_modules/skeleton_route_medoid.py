#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, math, json, sys, importlib.util
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set
import numpy as np, pandas as pd, rasterio
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.morphology import thin
import geopandas as gpd
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import unary_union

HERE=Path(__file__).resolve().parent
BASE_FILE=HERE / 'skeleton_multithreshold.py'
spec=importlib.util.spec_from_file_location('s3base',BASE_FILE);base=importlib.util.module_from_spec(spec);sys.modules['s3base']=base;spec.loader.exec_module(base)

@dataclass
class Config:
    event_extension_m: float=20.0
    member_buffer_m: float=7.0
    medoid_buffer_m: float=5.0
    interior_buffer_m: float=1.0
    min_candidate_cluster_points: int=4
    min_candidate_detour_ratio: float=1.20
    resample_points: int=101
    inserted_spacing_m: float=1.0
    inserted_smooth_sigma: float=0.75
    direction_query_radius_m: float=7.0
    direction_distance_sigma_m: float=3.0
    direction_min_coherence: float=0.45
    direction_max_deviation_deg: float=42.0
    direction_max_nearest_m: float=6.0
    max_track_link_m: float=25.0
    final_branch_prune_m: float=14.0
    final_branch_support_lt: float=1.5
    min_component_length_m: float=12.0
    simplify_tolerance_m: float=0.8


def cumdist(xy):
    if len(xy)<2:return np.zeros(len(xy))
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]

def resample_arclength(xy,n=None,spacing=None):
    xy=np.asarray(xy,float);d=cumdist(xy)
    if len(xy)<2 or d[-1]<=1e-9:return xy.copy()
    if spacing is not None:
        s=np.arange(0,d[-1],spacing)
        if len(s)==0 or s[-1]<d[-1]-1e-9:s=np.r_[s,d[-1]]
    else:s=np.linspace(0,d[-1],int(n))
    return np.c_[np.interp(s,d,xy[:,0]),np.interp(s,d,xy[:,1])]

def smooth_curve(xy,sigma):
    if len(xy)<5 or sigma<=0:return xy
    out=np.c_[gaussian_filter1d(xy[:,0],sigma,mode='nearest'),gaussian_filter1d(xy[:,1],sigma,mode='nearest')]
    out[0]=xy[0];out[-1]=xy[-1]
    return out

def sample_raster(arr,tr,xy):
    rr,cc=rasterio.transform.rowcol(tr,xy[:,0],xy[:,1]);rr=np.asarray(rr);cc=np.asarray(cc)
    valid=(rr>=0)&(rr<arr.shape[0])&(cc>=0)&(cc<arr.shape[1]);out=np.zeros(len(xy),float);out[valid]=arr[rr[valid],cc[valid]]
    return out

def extract_candidates(points:pd.DataFrame,cluster_id:int,consensus:LineString,cfg:Config):
    sub=points[points.gravity_uturn_cluster_id.fillna(0).astype(int).eq(cluster_id)]
    out=[]
    if sub.empty:return out
    c0=np.asarray(consensus.coords[0]);c1=np.asarray(consensus.coords[-1])
    for clean_id,marked in sub.groupby('clean_track_id'):
        full=points[points.clean_track_id.eq(clean_id)].sort_values('clean_point_index').reset_index(drop=True)
        if len(marked)<cfg.min_candidate_cluster_points or len(full)<2:continue
        marked_idx=set(marked.clean_point_index.astype(int).tolist())
        positions=np.flatnonzero(full.clean_point_index.astype(int).isin(marked_idx).to_numpy())
        if len(positions)<cfg.min_candidate_cluster_points:continue
        i0,i1=int(positions.min()),int(positions.max())
        event=full.iloc[i0:i1+1]
        exy=event[['x_m','y_m']].to_numpy(float)
        if len(exy)<3:continue
        # split safety
        if np.any(np.linalg.norm(np.diff(exy,axis=0),axis=1)>cfg.max_track_link_m):continue
        # orient to consensus
        direct=np.linalg.norm(exy[0]-c0)+np.linalg.norm(exy[-1]-c1)
        reverse=np.linalg.norm(exy[-1]-c0)+np.linalg.norm(exy[0]-c1)
        reversed_flag=reverse<direct
        if reversed_flag: exy=exy[::-1]
        chord=float(np.linalg.norm(exy[-1]-exy[0]));detour=float(LineString(exy).length/max(chord,1e-9))
        if detour<cfg.min_candidate_detour_ratio:continue
        # extend along actual parent track
        fxy=full[['x_m','y_m']].to_numpy(float);fd=cumdist(fxy)
        lo=max(0.,fd[i0]-cfg.event_extension_m);hi=min(fd[-1],fd[i1]+cfg.event_extension_m)
        j0=int(np.searchsorted(fd,lo,'left'));j1=int(np.searchsorted(fd,hi,'right'))
        ext=fxy[j0:j1]
        if reversed_flag: ext=ext[::-1]
        norm=resample_arclength(exy,n=cfg.resample_points)
        out.append({'clean_track_id':clean_id,'event_xy':exy,'extended_xy':ext,'normalized':norm,'detour':detour,'n':len(event)})
    return out

def choose_medoid(candidates,kde,support,tr):
    if not candidates:return None,{}
    A=np.stack([c['normalized'] for c in candidates])
    D=np.sqrt(np.mean(np.sum((A[:,None,:,:]-A[None,:,:,:])**2,axis=3),axis=2))
    med=np.median(D,axis=1);mad=np.median(np.abs(med-np.median(med)))
    robust=med<=np.median(med)+2.5*max(mad,0.5)
    scores=[]
    for i,c in enumerate(candidates):
        kv=sample_raster(kde,tr,c['normalized']).mean();sv=sample_raster(support,tr,c['normalized']).mean()
        # primarily medoid, density tie-breaker
        score=-med[i]+0.0025*kv+0.08*sv
        if not robust[i]:score-=100
        scores.append(score)
    idx=int(np.argmax(scores));c=candidates[idx]
    return c,{'candidate_count':len(candidates),'selected_track_id':c['clean_track_id'],'median_pairwise_rms_m':float(med[idx]),'selected_mean_kde':float(sample_raster(kde,tr,c['normalized']).mean()),'selected_mean_support':float(sample_raster(support,tr,c['normalized']).mean())}

def interior_polygon(line:LineString):
    coords=np.asarray(line.coords)
    if len(coords)<4:return None
    p=Polygon(coords)
    if not p.is_valid:p=p.buffer(0)
    if p.is_empty or p.area<20:return None
    return p

def raster_line(line,shape,tr):
    return rasterize([(mapping(line),1)],out_shape=shape,transform=tr,fill=0,dtype='uint8',all_touched=False).astype(bool)

def replace_clusters(global_sk,points,consensus_gdf,kde,support,tr,cfg):
    result=global_sk.copy(); inserted=np.zeros_like(result);removed=np.zeros_like(result);records=[];zones=[]
    for _,f in consensus_gdf.sort_values('track_support',ascending=False).iterrows():
        cid=int(f.cluster_id);cons=f.geometry
        cand=extract_candidates(points,cid,cons,cfg)
        selected,diag=choose_medoid(cand,kde,support,tr)
        if selected is None:continue
        event=resample_arclength(selected['event_xy'],spacing=cfg.inserted_spacing_m);event=smooth_curve(event,cfg.inserted_smooth_sigma)
        ext=resample_arclength(selected['extended_xy'],spacing=cfg.inserted_spacing_m);ext=smooth_curve(ext,cfg.inserted_smooth_sigma)
        event_line=LineString(event);ext_line=LineString(ext)
        member_lines=[LineString(c['event_xy']) for c in cand]
        pieces=[event_line.buffer(cfg.medoid_buffer_m,cap_style=2,join_style=2)]
        pieces += [l.buffer(cfg.member_buffer_m,cap_style=2,join_style=2) for l in member_lines]
        ip=interior_polygon(event_line)
        if ip is not None:pieces.append(ip.buffer(cfg.interior_buffer_m))
        zone_geom=unary_union(pieces)
        zone=rasterize([(mapping(zone_geom),1)],out_shape=result.shape,transform=tr,fill=0,dtype='uint8',all_touched=True).astype(bool)
        old=result&zone; result[zone]=False;removed|=old
        ins=raster_line(ext_line,result.shape,tr);result|=ins;inserted|=ins
        records.append({'cluster_id':cid,**diag,'event_length_m':event_line.length,'extended_length_m':ext_line.length,'removed_pixels':int(old.sum()),'inserted_pixels':int(ins.sum()),'geometry':ext_line})
        zones.append({'cluster_id':cid,'geometry':zone_geom})
    result=thin(result).astype(bool)
    return result,inserted,removed,records,zones

def build_samples(points,cfg):
    xy=[];ang=[];cid=[]
    for _,g in points.sort_values(['clean_track_id','clean_point_index']).groupby('clean_track_id'):
        p=g[['x_m','y_m']].to_numpy(float);con=(g.source_file.fillna('').astype(str)+'||'+g.track_id.fillna('').astype(str)).iloc[0]
        for a,b in zip(p[:-1],p[1:]):
            v=b-a;L=float(np.linalg.norm(v))
            if not np.isfinite(L) or L<=1e-6 or L>cfg.max_track_link_m:continue
            th=math.atan2(v[1],v[0]);n=max(1,int(math.ceil(L)))
            t=(np.arange(n)+.5)/n;q=a+t[:,None]*v
            xy.extend(q.tolist());ang.extend([th]*n);cid.extend([con]*n)
    return np.asarray(xy),np.asarray(ang),np.asarray(cid,dtype=object)

def directional_prune(sk,points,support,tr,cfg):
    sxy,sang,scid=build_samples(points,cfg);tree=cKDTree(sxy);pixels=set(map(tuple,np.argwhere(sk)));degree=base.degree_map(sk)
    rm=[];diag=[]
    for r,c in pixels:
        deg=int(degree[r,c]);neigh=base.pixel_neighbors((r,c),pixels)
        if deg not in (1,2) or len(neigh)!=deg:continue
        if deg==2:v=np.array([neigh[1][1]-neigh[0][1],-(neigh[1][0]-neigh[0][0])],float)
        else:v=np.array([c-neigh[0][1],-(r-neigh[0][0])],float)
        if np.linalg.norm(v)<1e-9:continue
        th=math.atan2(v[1],v[0]);x,y=tr*(c+.5,r+.5);ids=tree.query_ball_point([x,y],cfg.direction_query_radius_m)
        if not ids:continue
        best={}
        for i in ids:
            d=float(np.linalg.norm(sxy[i]-[x,y]));k=scid[i]
            if k not in best or d<best[k][0]:best[k]=(d,sang[i])
        z=0j;ws=0
        for d,a in best.values():
            w=math.exp(-(d/cfg.direction_distance_sigma_m)**2);z+=w*complex(math.cos(2*a),math.sin(2*a));ws+=w
        if ws<=0:continue
        coh=abs(z)/ws;dom=.5*math.atan2(z.imag,z.real);diff=abs(math.degrees(math.atan2(math.sin(2*(th-dom)),math.cos(2*(th-dom))))/2);near=min(x[0] for x in best.values())
        flag=coh>=cfg.direction_min_coherence and diff>=cfg.direction_max_deviation_deg and near<=cfg.direction_max_nearest_m
        if flag:rm.append((r,c))
        diag.append({'row':r,'col':c,'x':x,'y':y,'degree':deg,'track_count':len(best),'coherence':coh,'angle_diff_deg':diff,'nearest_m':near,'removed':flag,'support':int(support[r,c])})
    out=sk.copy()
    if rm:
        rr,cc=zip(*rm);out[np.asarray(rr),np.asarray(cc)]=False
    out=base.prune_short_low_support_branches(out,support,abs(tr.a),cfg.final_branch_prune_m,cfg.final_branch_support_lt)
    out=base.remove_short_components_by_length(out,abs(tr.a),cfg.min_component_length_m)
    out=thin(out).astype(bool)
    return out,pd.DataFrame(diag),len(rm)

def graph_neighbors_clean(pixel,pixels):
    r,c=pixel;out=[]
    for dr,dc in base.NEIGHBORS_8:
        q=(r+dr,c+dc)
        if q not in pixels:continue
        if dr!=0 and dc!=0:
            if (r+dr,c) in pixels or (r,c+dc) in pixels:continue
        out.append(q)
    return out

def trace_clean(sk):
    pixels=set(map(tuple,np.argwhere(sk)));deg={p:len(graph_neighbors_clean(p,pixels)) for p in pixels};nodes={p for p in pixels if deg[p]!=2};vis=set();paths=[]
    def ek(a,b):return (a,b) if a<=b else (b,a)
    for n in sorted(nodes):
        for nb in graph_neighbors_clean(n,pixels):
            if ek(n,nb) in vis:continue
            path=[n,nb];vis.add(ek(n,nb));prev,cur=n,nb
            while cur not in nodes:
                cs=[x for x in graph_neighbors_clean(cur,pixels) if x!=prev]
                if not cs:break
                nx=cs[0];vis.add(ek(cur,nx));path.append(nx);prev,cur=cur,nx
            paths.append(path)
    for st in sorted(pixels):
        un=[x for x in graph_neighbors_clean(st,pixels) if ek(st,x) not in vis]
        if not un:continue
        path=[st,un[0]];vis.add(ek(st,un[0]));prev,cur=st,un[0]
        while True:
            cs=[x for x in graph_neighbors_clean(cur,pixels) if x!=prev and ek(cur,x) not in vis]
            if not cs:break
            nx=cs[0];vis.add(ek(cur,nx));path.append(nx);prev,cur=cur,nx
            if cur==st:break
        if len(path)>=2:paths.append(path)
    return paths

def write_clean_gpkg(path,sk,tr,crs,support,cfg):
    rec=[]
    for i,pix in enumerate(trace_clean(sk),1):
        coords=[];sv=[]
        for r,c in pix:
            x,y=tr*(c+.5,r+.5);coords.append((x,y));sv.append(support[r,c])
        if len(coords)<2:continue
        geom=LineString(coords).simplify(cfg.simplify_tolerance_m,preserve_topology=True)
        rec.append({'segment_id':i,'length_m':geom.length,'mean_support':float(np.mean(sv)),'min_support':float(np.min(sv)),'geometry':geom})
    gpd.GeoDataFrame(rec,geometry='geometry',crs=crs).to_file(path,layer='road_skeleton_segments',driver='GPKG')

def run(kde_dir,gravity_points,uturn_gpkg,global_skeleton,output,cfg):
    output.mkdir(parents=True,exist_ok=True)
    pts=pd.read_csv(gravity_points,low_memory=False);cons=gpd.read_file(uturn_gpkg,layer='uturn_consensus_lines')
    kp=next(kde_dir.glob('kde_independent_support_sigma4m_adaptive_uturn2m_uint16_0_*.tif'))
    with rasterio.open(kp) as ks,rasterio.open(kde_dir/'independent_track_support.tif') as ss,rasterio.open(global_skeleton) as gs:
        kde=ks.read(1);sup=ss.read(1);glob=gs.read(1)>0;tr=ks.transform
        rep,ins,rem,recs,zones=replace_clusters(glob,pts,cons,kde,sup,tr,cfg)
        final,ddiag,nrm=directional_prune(rep,pts,sup,tr,cfg)
        base.write_raster(output/'skeleton_binary_route_medoid_directional_min4.tif',final.astype(np.uint8),ks,0)
        base.write_raster(output/'route_medoid_inserted.tif',ins.astype(np.uint8),ks,0)
        base.write_raster(output/'route_medoid_removed.tif',rem.astype(np.uint8),ks,0)
        write_clean_gpkg(output/'road_skeleton_route_medoid_directional.gpkg',final,tr,ks.crs,sup,cfg)
        if recs:gpd.GeoDataFrame(recs,geometry='geometry',crs=ks.crs).to_file(output/'route_medoid_diagnostics.gpkg',layer='selected_medoid_routes',driver='GPKG')
        if zones:gpd.GeoDataFrame(zones,geometry='geometry',crs=ks.crs).to_file(output/'route_medoid_diagnostics.gpkg',layer='clear_zones',driver='GPKG')
        pd.DataFrame([{k:v for k,v in r.items() if k!='geometry'} for r in recs]).to_csv(output/'route_medoid_cluster_diagnostics.csv',index=False,encoding='utf-8-sig')
        ddiag.to_csv(output/'direction_prune_diagnostics.csv',index=False,encoding='utf-8-sig')
        report={'config':asdict(cfg),'input_global_pixels':int(glob.sum()),'after_replacement_pixels':int(rep.sum()),'direction_removed_pixels':nrm,'final_pixels':int(final.sum()),'metrics':base.graph_metrics(final,sup,abs(tr.a))}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(report)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kde-dir',type=Path,required=True);ap.add_argument('--gravity-points',type=Path,required=True);ap.add_argument('--uturn-gpkg',type=Path,required=True);ap.add_argument('--global-skeleton',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();run(a.kde_dir,a.gravity_points,a.uturn_gpkg,a.global_skeleton,a.output,Config())
if __name__=='__main__':main()
