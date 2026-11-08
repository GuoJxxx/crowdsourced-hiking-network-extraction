"""Compute manuscript measures and export tables and cartographic input layers."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from .common import ROOT, path, source, parameter_file, intermediate
from .products import network_file, evaluation_file, table_path, map_folder, stem
from .engine import CHICAGO, MAINLING, ABLATIONS, NAMES


def network(group, method):
    return network_file(group,method)


def graph_measures(filename):
    """Measure the normalized embedded graph; retain both active and total node counts."""
    import geopandas as gpd
    import networkx as nx
    nodes=gpd.read_file(filename,layer='nodes')
    edges=gpd.read_file(filename,layer='edges')
    if nodes.node_id.duplicated().any():raise ValueError('Duplicate node identifiers: '+str(filename))
    known=set(nodes.node_id)
    # Parallel edges and self-loops contribute to cycle rank and remain in the graph.
    graph=nx.MultiGraph()
    for row in edges.itertuples():
        if row.u not in known or row.v not in known:raise ValueError('Edge endpoint missing from nodes')
        graph.add_edge(row.u,row.v,length=float(row.geometry.length))
    # Components below contain edges. Isolated vertices are counted separately.
    components=list(nx.connected_components(graph))
    membership={node:i for i,component in enumerate(components) for node in component}
    lengths=[0.0]*len(components)
    for u,v,data in graph.edges(data=True):lengths[membership[u]]+=data['length']
    length=float(edges.geometry.length.sum())
    canonical=hashlib.sha256()
    for value in sorted(geometry.normalize().wkb for geometry in edges.geometry if geometry is not None):
        canonical.update(len(value).to_bytes(8,'little'));canonical.update(value)
    active=graph.number_of_nodes()
    return {'nodes':len(nodes),'active_nodes':active,'edges':len(edges),'length_m':length,'length_km':length/1000,
            'components':len(components),'components_all_nodes':len(components)+len(nodes)-active,
            'endpoints':sum(degree==1 for _,degree in graph.degree()),
            'junctions':sum(degree>=3 for _,degree in graph.degree()),
            'cycle_rank':len(edges)-active+len(components),
            'largest_component_length_share':max(lengths,default=0)/length if length else 0,
            'crs':str(edges.crs),'canonical_geometry_sha256':canonical.hexdigest()}


def table_dir():
    """Study-area table directory (individual filenames use table_path)."""
    return table_path('table_3.csv').parent


def write_csv(filename, records, columns=None):
    filename.parent.mkdir(parents=True,exist_ok=True)
    columns=columns or list(records[0])
    with filename.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)
    print('WROTE',filename,flush=True)


def read_csv(filename):
    with filename.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def group_measures(group):
    methods={'chicago':CHICAGO,'mainling':MAINLING,'ablation':ABLATIONS}[group]
    result=[];tolerance_rows=[]
    for method in methods:
        values={'method':NAMES[method],**graph_measures(network(group,method))}
        if group=='chicago':
            evaluation=evaluation_file(method)
            assessed=json.loads(evaluation.read_text(encoding='utf-8'))
            values.update(assessed['proximity']['15'])
            for tolerance,scores in sorted(assessed['proximity'].items(),key=lambda item:float(item[0])):
                tolerance_rows.append({'method':NAMES[method],'tolerance_m':float(tolerance),**scores})
        result.append(values)
    write_csv(table_path(f'{group}_metrics.csv'),result)
    if tolerance_rows:write_csv(table_path('chicago_tolerance.csv'),tolerance_rows)
    (table_path(f'{group}_metrics.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def table(number):
    if number=='1':
        rows=read_csv(table_path('chicago_metrics.csv'))
        write_csv(table_path('table_1.csv'),rows,['method','precision','recall','f1','length_km','cycle_rank'])
    elif number=='2':
        records=[]
        for mode in ABLATIONS:
            records.append({'configuration':NAMES[mode],
                            'activity_awareness':mode!='wo_activity',
                            'hairpin_preservation':mode!='wo_uturn',
                            'topology_optimization':mode!='wo_evidence_topology'})
        write_csv(table_path('table_2.csv'),records)
    elif number=='3':
        rows=read_csv(table_path('ablation_metrics.csv'))
        write_csv(table_path('table_3.csv'),rows,['method','length_km','edges','components','endpoints','junctions','cycle_rank','largest_component_length_share'])
        full=next(r for r in rows if r['method']=='Full')
        ratios=[]
        for row in rows:
            ratios.append({'method':row['method'],**{key:float(row[key])/float(full[key]) for key in
                           ('length_km','edges','components','endpoints','junctions','cycle_rank')}})
        write_csv(table_path('ablation_ratios.csv'),ratios)
    else:
        config=json.loads(parameter_file('mainling.json').read_text(encoding='utf-8-sig'))
        schema=json.loads((source('mainling_full')/'configs/parameter_schema.json').read_text(encoding='utf-8-sig'))
        rows=[]
        for stage,document in schema['stages'].items():
            for parameter in document['parameters']:
                value=config['stages'][stage].get(parameter['dest'])
                rows.append({'stage':stage,'parameter':parameter['dest'],'value':json.dumps(value,ensure_ascii=False),
                             'role':parameter.get('help',''),'option':parameter.get('option',parameter.get('positive_option',''))})
        write_csv(table_path('parameters_full.csv'),rows,['stage','parameter','value','role','option'])
        paper=json.loads(parameter_file('paper_table_A1.json').read_text(encoding='utf-8'))
        write_csv(table_path('table_A1.csv'),paper,['module','parameter','value','role'])


def summary():
    """Collect the measures calculated from all current reconstructed graphs."""
    from .figures import chicago_tracks
    import pandas as pd
    document={group:json.loads(table_path(f'{group}_metrics.json').read_text(encoding='utf-8'))
              for group in ('chicago','mainling','ablation')}
    tracks=chicago_tracks();clean=path('mainling_clean')
    regional={'observed_files':len(list(path('mainling_observed').glob('*.csv'))),
              'cleaned_segments':len(list(clean.glob('*.csv'))),
              'cleaned_points':sum(len(chunk) for chunk in pd.read_csv(clean.parent/'all_cleaned_points.csv',usecols=['x_m'],chunksize=100000)),
              'crs':'EPSG:32647'}
    chicago={'dataset':{'trajectories':len(tracks),'points':sum(len(t) for t in tracks),'crs':'EPSG:26916'},
             'methods':document['chicago']}
    full=next(r for r in document['ablation'] if r['method']=='Full')
    ratios=[{'method':r['method'],**{key:r[key]/full[key] for key in
             ('length_km','edges','components','endpoints','junctions','cycle_rank')}} for r in document['ablation']]
    study={'dataset':regional,'methods':document['mainling'],'ablation':document['ablation'],'ablation_ratios':ratios}
    for number in ('1','2','3','a1'):table(number)
    from .common import dataset_results
    target=dataset_results('chicago')/'Chicago_TablesAndFigures/Chicago_ReportedMetrics.json'
    target.write_text(json.dumps(chicago,indent=2),encoding='utf-8')
    table_path('reported_metrics.json').write_text(json.dumps(study,indent=2),encoding='utf-8')
    print('WROTE',target,flush=True)


def map_layers(number):
    directory=map_folder(number)
    directory.mkdir(parents=True,exist_ok=True)
    group,methods=('mainling',('proposed',)) if number==10 else ('ablation',ABLATIONS)
    records=[]
    for method in methods:
        source=network(group,method);target=directory/(stem(group,method)+'_Network.gpkg')
        shutil.copy2(source,target)
        records.append({'method':NAMES[method],'file':target.name,'layers':['nodes','edges'],'crs':'EPSG:32647'})
    records.append({'role':'cleaned trajectories','directory':Path(__import__('os').path.relpath(path('mainling_clean'),directory)).as_posix(),
                    'coordinate_columns':['x_m','y_m'],'crs':'EPSG:32647'})
    # Local maps consume model-derived semantic areas as well as the final edges.
    if number==12:
        for name,relative,layer in (
            ('activity_zones.gpkg','ablation/full/S3_Activity/activity_zones.gpkg','activity_suppression_zones'),
            ('hairpin_zones.gpkg','ablation/full/S4_Gravity/uturn_consensus.gpkg','uturn_protection_zones')):
            source=intermediate('ablation')/relative.removeprefix('ablation/')
            if source.is_file():
                shutil.copy2(source,directory/('StudyArea_Proposed_'+name))
                records.append({'role':name.removesuffix('.gpkg'),'file':'StudyArea_Proposed_'+name,'layers':[layer],'crs':'EPSG:32647'})
        scenes=json.loads(parameter_file('scenes.json').read_text(encoding='utf-8'))
        (directory/'StudyArea_Figure12_Scenes.json').write_text(json.dumps({'crs':scenes['crs'],'scenes':scenes['scenes'][:3]},indent=2),encoding='utf-8')
    records.append({'role':'study boundary','file':Path(__import__('os').path.relpath(path('boundary'),directory)).as_posix()})
    (directory/f'StudyArea_Figure{number}_Layers.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('WROTE',directory/f'StudyArea_Figure{number}_Layers.json',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('chicago-metrics','mainling-metrics','ablation-metrics','table','summary','map-layers'))
    parser.add_argument('number',nargs='?')
    args=parser.parse_args(argv)
    if args.command.endswith('-metrics'):group_measures(args.command.split('-')[0])
    elif args.command=='table':table(args.number)
    elif args.command=='summary':summary()
    else:map_layers(int(args.number))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
