"""Four Mainling methods using a shared computed clean trajectory collection."""
import argparse
from .common import ROOT,finish,module,path,py,tool,source,intermediate,RUNTIME
from .pipeline import mainling_pipeline

METHODS=('proposed','edelkamp','biagioni','karagiorgou')

def main(argv=None):
    parser=argparse.ArgumentParser(description='Mainling: FULL and three reference methods')
    parser.add_argument('--method',choices=(*METHODS,'all'),default='all')
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args(argv)
    methods=list(METHODS) if args.method=='all' else [args.method]
    out=intermediate('mainling')
    steps=[]
    if 'proposed' in methods:
        steps+=mainling_pipeline(out/'proposed')
    else:
        steps+=mainling_pipeline(out/'proposed')[:1]
    references=out/'references'
    legacy=[m for m in methods if m in ('edelkamp','biagioni')]
    runner=source('mainling_references')/'run_reference_experiments.py'
    original=ROOT/'StudyArea/Methods'
    if legacy or 'karagiorgou' in methods:
        steps.append(py('references_prepare',runner,'prepare','--root',references,'--original-root',original,'--edelkamp-source',ROOT/'StudyArea/Methods/Edelkamp/StudyArea_Edelkamp_Source','--biagioni-source',ROOT/'StudyArea/Methods/Biagioni/StudyArea_Biagioni_Source',
                        '--tracks',path('mainling_clean'),outputs=[references/'preparation_manifest.json']))
        steps[-1]['restart_targets']=[str(references)]
    if legacy:
        for method in legacy:
            database=references/('Edelkamp/raw/edelkamp_graph.db' if method=='edelkamp' else 'Biagioni/stages/08_final/skeleton_map_mm2.db')
            steps.append(py('reference_'+method,runner,'run-'+method,'--root',references,outputs=[database]))
            if method=='edelkamp':
                steps[-1]['restart_targets']=[str(database)+suffix for suffix in ('','-wal','-shm','-journal')]
            steps.append(py('normalize_'+method,runner,'normalize','--root',references,'--method',method.title(),
                            '--target-crs','EPSG:32647',outputs=[references/method.title()/'normalized/road_network.gpkg']))
    if 'karagiorgou' in methods:
        method_root=references/'Karagiorgou'
        root=method_root/'run'
        steps += [module('karagiorgou_prepare','mainling_references.karagiorgou','prepare',
                         '--tracks',method_root/'inputs/trips','--manifest',path('reference_manifest'),
                         '--root',root,outputs=[root/'identity.json']),
                  module('karagiorgou_run','mainling_references.karagiorgou','run',
                         '--root',root,'--matlab',tool('matlab'),'--disable-jit','--max-restarts','3',
                         outputs=[root/'native/completion.json',root/'native/tracebundle_vertices.txt',root/'native/tracebundle_edges.txt']),
                  py('karagiorgou_normalize',RUNTIME/'workflow/prepare.py','mainling-normalize','--root',root,
                     '--output',method_root/'normalized',
                     outputs=[method_root/'normalized/road_network.gpkg',method_root/'normalized/graph_metrics.json'])]
        next(item for item in steps if item['name']=='karagiorgou_prepare')['restart_targets']=[str(root)]
    result=finish({'group':'mainling','methods':methods,'steps':steps},args,
                  ['matlab'] if 'karagiorgou' in methods else [])
    if not args.plan:
        from .publish import publish
        publish('mainling',methods)
    return result


if __name__=='__main__':raise SystemExit(main())
