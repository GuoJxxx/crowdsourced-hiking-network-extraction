#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, sys, importlib.util, shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np, pandas as pd, rasterio
from rasterio.features import rasterize
from skimage.morphology import thin
import geopandas as gpd
from shapely.geometry import LineString, mapping
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rm', HERE / 'skeleton_route_medoid.py')
rm=importlib.util.module_from_spec(spec);sys.modules['rm']=rm;spec.loader.exec_module(rm)


def compress_positive(values):
    out=[]
    for v in values:
        v=int(v) if pd.notna(v) else 0
        if v>0 and (not out or v!=out[-1]):out.append(v)
    return out

def frequent_pairs(points,min_tracks=3):
    d={}
    points=points.copy();points['_contribution_id']=points.source_file.fillna('').astype(str)+'||'+points.track_id.fillna('').astype(str)
    for cid,gc in points.sort_values(['clean_track_id','clean_point_index']).groupby('_contribution_id'):
        # original contribution can contain multiple clean tracks; inspect each
        for clean_id,g in gc.groupby('clean_track_id'):
            seq=compress_positive(g.gravity_uturn_cluster_id)
            for a,b in zip(seq[:-1],seq[1:]):d.setdefault((a,b),set()).add(cid)
    return {k:v for k,v in d.items() if len(v)>=min_tracks}

def extract_pair_candidates(points,a,b,cfg):
    out=[];points=points.copy();points['_contribution_id']=points.source_file.fillna('').astype(str)+'||'+points.track_id.fillna('').astype(str)
    for cid,gc in points.groupby('_contribution_id'):
        best=None
        for clean_id,g in gc.groupby('clean_track_id'):
            g=g.sort_values('clean_point_index').reset_index(drop=True)
            cl=g.gravity_uturn_cluster_id.fillna(0).astype(int).to_numpy()
            ia=np.flatnonzero(cl==a);ib=np.flatnonzero(cl==b)
            if len(ia)==0 or len(ib)==0:continue
            # directed a -> b only; last a before first b
            candidates=[(i,j) for i in ia for j in ib if j>i]
            if not candidates:continue
            i,j=min(candidates,key=lambda z:z[1]-z[0])
            # include a few points inside both clusters / 8m margins
            xy=g[['x_m','y_m']].to_numpy(float);d=rm.cumdist(xy)
            lo=max(0,d[i]-8);hi=min(d[-1],d[j]+8);i0=int(np.searchsorted(d,lo,'left'));i1=int(np.searchsorted(d,hi,'right'))
            path=xy[i0:i1]
            if len(path)<3 or np.any(np.linalg.norm(np.diff(path,axis=0),axis=1)>cfg.max_track_link_m):continue
            length=LineString(path).length
            if length<8:continue
            norm=rm.resample_arclength(path,n=81)
            rec={'contribution_id':cid,'clean_track_id':clean_id,'xy':path,'normalized':norm,'length':length}
            if best is None or length>best['length']:best=rec
        if best:out.append(best)
    return out

def choose_pair_medoid(cands,kde,support,tr):
    if not cands:return None,{}
    A=np.stack([c['normalized'] for c in cands]);D=np.sqrt(np.mean(np.sum((A[:,None]-A[None,:])**2,axis=3),axis=2));med=np.median(D,axis=1)
    scores=[]
    for i,c in enumerate(cands):
        kv=rm.sample_raster(kde,tr,c['normalized']).mean();sv=rm.sample_raster(support,tr,c['normalized']).mean();scores.append(-med[i]+.0025*kv+.08*sv)
    i=int(np.argmax(scores));c=cands[i]
    return c,{'candidate_count':len(cands),'selected_track_id':c['clean_track_id'],'median_pairwise_rms_m':float(med[i]),'selected_mean_kde':float(rm.sample_raster(kde,tr,c['normalized']).mean()),'selected_mean_support':float(rm.sample_raster(support,tr,c['normalized']).mean())}

def replace_pairs(sk,points,kde,support,tr,cfg,min_tracks=3,buffer_m=6.0):
    result=sk.copy();inserted=np.zeros_like(sk);removed=np.zeros_like(sk);records=[];zones=[]
    pairs=frequent_pairs(points,min_tracks)
    # strongest pairs first
    for (a,b),contributors in sorted(pairs.items(),key=lambda kv:len(kv[1]),reverse=True):
        cands=extract_pair_candidates(points,a,b,cfg);sel,diag=choose_pair_medoid(cands,kde,support,tr)
        if sel is None:continue
        xy=rm.resample_arclength(sel['xy'],spacing=cfg.inserted_spacing_m);xy=rm.smooth_curve(xy,cfg.inserted_smooth_sigma);line=LineString(xy)
        member=[LineString(c['xy']).buffer(buffer_m,cap_style=2,join_style=2) for c in cands]
        zone_geom=unary_union(member+[line.buffer(4.0,cap_style=2,join_style=2)])
        zone=rasterize([(mapping(zone_geom),1)],out_shape=result.shape,transform=tr,fill=0,dtype='uint8',all_touched=True).astype(bool)
        old=result&zone;result[zone]=False;removed|=old
        ins=rm.raster_line(line,result.shape,tr);result|=ins;inserted|=ins
        records.append({'pair_a':a,'pair_b':b,'track_support':len(contributors),**diag,'length_m':line.length,'removed_pixels':int(old.sum()),'inserted_pixels':int(ins.sum()),'geometry':line})
        zones.append({'pair_a':a,'pair_b':b,'track_support':len(contributors),'geometry':zone_geom})
    return thin(result).astype(bool),inserted,removed,records,zones

def alignment_maps(sk,points,tr,cfg):
    sxy,sang,scid=rm.build_samples(points,cfg)
    from scipy.spatial import cKDTree
    tree=cKDTree(sxy);pix=set(map(tuple,np.argwhere(sk)));aligned={};allcount={}
    for p in pix:
        ns=rm.graph_neighbors_clean(p,pix)
        if len(ns)>=2:
            arr=np.array([[q[1]-p[1],-(q[0]-p[0])] for q in ns],float);C=arr.T@arr;ev,evec=np.linalg.eigh(C);v=evec[:,np.argmax(ev)]
        elif len(ns)==1:v=np.array([ns[0][1]-p[1],-(ns[0][0]-p[0])],float)
        else:v=np.array([1.,0.])
        th=math.atan2(v[1],v[0]);x,y=tr*(p[1]+.5,p[0]+.5);ids=tree.query_ball_point([x,y],cfg.direction_query_radius_m);best={}
        for i in ids:
            d=float(np.linalg.norm(sxy[i]-[x,y]));cid=scid[i];diff=abs(math.degrees(math.atan2(math.sin(2*(sang[i]-th)),math.cos(2*(sang[i]-th))))/2)
            if cid not in best or d<best[cid][0]:best[cid]=(d,diff)
        aligned[p]=sum(1 for d,diff in best.values() if d<=cfg.direction_max_nearest_m and diff<=30);allcount[p]=len(best)
    return aligned,allcount

def primary_auxiliary(sk,points,support,tr,cfg,mean_threshold=1.75,fraction_threshold=.45):
    aligned,allc=alignment_maps(sk,points,tr,cfg);paths=rm.trace_clean(sk);primary=np.zeros_like(sk);rows=[]
    for i,path in enumerate(paths,1):
        vals=np.asarray([aligned.get(p,0) for p in path]);length=sum(rm.base.step_length(a,b,abs(tr.a)) for a,b in zip(path[:-1],path[1:]));mean=float(vals.mean());frac=float(np.mean(vals>=2));is_primary=(mean>=mean_threshold or frac>=fraction_threshold or (length<=8 and mean>=1))
        if is_primary:
            for p in path:primary[p]=True
        rows.append({'segment_id':i,'length_m':length,'aligned_track_mean':mean,'aligned_ge2_fraction':frac,'primary':is_primary})
    primary=thin(primary).astype(bool);aux=sk&~primary
    return primary,aux,pd.DataFrame(rows)

def run(args):
    cfg=rm.Config()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    pts=pd.read_csv(args.gravity_points,low_memory=False)
    cons=gpd.read_file(args.uturn_gpkg,layer='uturn_consensus_lines')
    base_suffix=rm.base.sanitize_sigma_name(args.base_sigma_m) if hasattr(rm.base, 'sanitize_sigma_name') else f'{args.base_sigma_m:g}'.replace('.', 'p')
    uturn_suffix=rm.base.sanitize_sigma_name(args.uturn_sigma_m) if hasattr(rm.base, 'sanitize_sigma_name') else f'{args.uturn_sigma_m:g}'.replace('.', 'p')
    candidates=sorted(args.kde_dir.glob(f'kde_independent_support_sigma{base_suffix}m_adaptive_uturn{uturn_suffix}m_uint16_0_*.tif'))
    if not candidates:
        raise FileNotFoundError(f"""Adaptive KDE not found: sigma{base_suffix}m_adaptive_uturn{uturn_suffix}m_uint16, directory={args.kde_dir}""")
    kp=candidates[0]
    support_path=args.kde_dir/'independent_track_support.tif'
    union_path=args.kde_dir/'raw_trajectory_union.tif'
    distance_path=args.kde_dir/'distance_to_raw_trajectory_m.tif'
    if args.global_skeleton is None:
        global_dir=args.output/'global_multithreshold'
        base_cfg=rm.base.SkeletonConfig(
            max_exponent=9,
            min_exponent=args.min_exponent,
            support_corridor_m=args.support_corridor_m,
            write_stages=True,
            write_gpkg=True,
            clear_output=True,
        )
        rm.base.process_one(kp,support_path,union_path,distance_path,global_dir,base_cfg)
        global_path=global_dir/'skeleton_binary.tif'
    else:
        global_path=args.global_skeleton
    with rasterio.open(kp) as ks,rasterio.open(support_path) as ss,rasterio.open(global_path) as gs:
        kde=ks.read(1);sup=ss.read(1);glob=gs.read(1)>0;tr=ks.transform
        a,ins1,rem1,rec1,z1=rm.replace_clusters(glob,pts,cons,kde,sup,tr,cfg)
        b,ins2,rem2,rec2,z2=replace_pairs(a,pts,kde,sup,tr,cfg,args.min_pair_tracks,args.pair_buffer_m)
        all_ev,ddiag,nrm=rm.directional_prune(b,pts,sup,tr,cfg)
        primary,aux,segdiag=primary_auxiliary(all_ev,pts,sup,tr,cfg,args.primary_aligned_mean,args.primary_ge2_fraction)
        for name,arr in [('skeleton_all_evidence',all_ev),('skeleton_primary',primary),('skeleton_auxiliary_singletrack',aux),('cluster_routes_inserted',ins1),('sequence_routes_inserted',ins2),('cluster_routes_removed',rem1),('sequence_routes_removed',rem2)]:rm.base.write_raster(args.output/(name+'.tif'),arr.astype(np.uint8),ks,0)
        rm.write_clean_gpkg(args.output/'road_skeleton_all_evidence.gpkg',all_ev,tr,ks.crs,sup,cfg);rm.write_clean_gpkg(args.output/'road_skeleton_primary.gpkg',primary,tr,ks.crs,sup,cfg);rm.write_clean_gpkg(args.output/'road_skeleton_auxiliary_singletrack.gpkg',aux,tr,ks.crs,sup,cfg)
        if rec1:gpd.GeoDataFrame(rec1,geometry='geometry',crs=ks.crs).to_file(args.output/'route_diagnostics.gpkg',layer='uturn_medoid_routes',driver='GPKG')
        if rec2:gpd.GeoDataFrame(rec2,geometry='geometry',crs=ks.crs).to_file(args.output/'route_diagnostics.gpkg',layer='switchback_sequence_routes',driver='GPKG')
        if z1:gpd.GeoDataFrame(z1,geometry='geometry',crs=ks.crs).to_file(args.output/'route_diagnostics.gpkg',layer='uturn_clear_zones',driver='GPKG')
        if z2:gpd.GeoDataFrame(z2,geometry='geometry',crs=ks.crs).to_file(args.output/'route_diagnostics.gpkg',layer='sequence_clear_zones',driver='GPKG')
        pd.DataFrame([{k:v for k,v in r.items() if k!='geometry'} for r in rec1]).to_csv(args.output/'uturn_route_diagnostics.csv',index=False,encoding='utf-8-sig');pd.DataFrame([{k:v for k,v in r.items() if k!='geometry'} for r in rec2]).to_csv(args.output/'sequence_route_diagnostics.csv',index=False,encoding='utf-8-sig');segdiag.to_csv(args.output/'primary_auxiliary_segments.csv',index=False,encoding='utf-8-sig');ddiag.to_csv(args.output/'direction_prune_pixels.csv',index=False,encoding='utf-8-sig')
        report={'global_pixels':int(glob.sum()),'all_evidence_pixels':int(all_ev.sum()),'primary_pixels':int(primary.sum()),'auxiliary_pixels':int(aux.sum()),'direction_removed':nrm,'uturn_routes':len(rec1),'sequence_routes':len(rec2),'primary_metrics':rm.base.graph_metrics(primary,sup,abs(tr.a)),'all_metrics':rm.base.graph_metrics(all_ev,sup,abs(tr.a)),'config':vars(args)}
        report['config']={k:str(v) if isinstance(v,Path) else v for k,v in report['config'].items()};(args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(report)

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--kde-dir',type=Path,required=True)
    ap.add_argument('--gravity-points',type=Path,required=True)
    ap.add_argument('--uturn-gpkg',type=Path,required=True)
    ap.add_argument('--global-skeleton',type=Path,default=None,help='Optional; otherwise generate the lowest-threshold skeleton from adaptive KDE')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--base-sigma-m',type=float,default=4.0)
    ap.add_argument('--uturn-sigma-m',type=float,default=2.0)
    ap.add_argument('--min-exponent',type=int,default=2)
    ap.add_argument('--support-corridor-m',type=float,default=8.0)
    ap.add_argument('--min-pair-tracks',type=int,default=3)
    ap.add_argument('--pair-buffer-m',type=float,default=6)
    ap.add_argument('--primary-aligned-mean',type=float,default=1.75)
    ap.add_argument('--primary-ge2-fraction',type=float,default=.45)
    run(ap.parse_args())
if __name__=='__main__':main()
