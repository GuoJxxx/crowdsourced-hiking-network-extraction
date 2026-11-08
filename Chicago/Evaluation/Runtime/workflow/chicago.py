"""Chicago reconstruction and evaluation with the manuscript parameter profile."""
import argparse
from .common import ROOT,finish,module,path,py,source,intermediate,RUNTIME

METHODS=('ahmed','biagioni','davies','edelkamp','karagiorgou','proposed')
REFERENCE='experiments.chicago_six_methods.reference_methods.run_reference_methods'


def proposed_steps(out):
    package=source('chicago_proposed')
    scripts=package/'code'
    observed=out/'S1_Projected'
    wgs=out/'S1_WGS84'
    s2,s3,s4,s5,s6=(out/name for name in ('S2_Prepare','S3_Activity','S4_Gravity','S5_KDE','S6_Skeleton'))
    s7=out/'S7_Network'
    steps=[py('proposed_S1_archive',RUNTIME/'workflow/prepare.py','chicago',
              '--archive',path('chicago_tracks'),'--output',observed,outputs=[observed/'summary.json']),
           py('proposed_S1_coordinates',scripts/'tools/prepare_chicago_s1_wgs84.py',
              '--input',observed,'--output',wgs,'--source-crs','EPSG:26916',outputs=[wgs/'coordinate_adapter_summary.json']),
           py('proposed_S2',scripts/'tools/prepare_chicago_s2_passthrough.py',
              '--input',wgs,'--output',s2,outputs=[s2/'all_cleaned_points.csv'])]
    steps += [py('proposed_S3',scripts/'S3_activity_zone_filter.py',
                 '--input',s2/'all_cleaned_points.csv','--output',s3,'--max-output-link-m','0','--clear-output',
                 outputs=[s3/'road_candidate_points.csv']),
              py('proposed_S4',scripts/'S4_uturn_consensus_gravity.py',
                 '--input',s3/'road_candidate_points.csv','--output',s4,outputs=[s4/'all_cleaned_points.csv',s4/'gravity_uturn_report.json']),
              py('proposed_S5',scripts/'S5_directional_kde.py',
                 '--input',s4/'all_cleaned_points.csv','--output',s5,'--gravity-report',s4/'gravity_uturn_report.json','--max-link-m','0',
                 outputs=[s5/'directional_kde_report.json'])]
    kde=s5/'kde_directional_along4m_cross1p5m_float32.tif'
    support=s5/'independent_track_support.tif'
    distance=s5/'distance_to_raw_trajectory_m.tif'
    raw_union=s5/'raw_trajectory_union.tif'
    gravity=s4/'all_cleaned_points.csv'
    semantics=['--uturn-gpkg',s4/'uturn_consensus.gpkg','--activity-zones',s3/'activity_zones.gpkg']
    steps.append(py('proposed_S6',scripts/'S6_route_locked_skeleton.py',
                    '--kde',kde,'--support',support,'--distance',distance,'--raw-union',raw_union,
                    '--gravity-points',gravity,*semantics,'--output',s6,'--thresholds','1024,512,256',
                    '--support-rescue-min-tracks','2',outputs=[s6/'skeleton_all_evidence_route_locked.tif']))
    arguments=['--primary',s6/'skeleton_all_evidence_route_locked.tif','--support',support,
               '--kde',kde,'--distance',distance,'--gravity-points',gravity,'--raw-tracks',wgs,
               *semantics,'--output',s7,'--isolated-component-hard-max-m','0',
               '--isolated-component-weak-max-m','120','--isolated-component-weak-max-tracks','2',
               '--isolated-component-weak-max-confidence','0.7','--mainroad-only-final-prune',
               '--mainroad-short-max-m','800','--mainroad-short-max-tracks','2',
               '--mainroad-short-prune-rounds','20','--raw-track-max-step-m','0','--raw-track-max-speed-mps','0']
    network=s7/'08_final_clean_network/road_network_final_clean.gpkg'
    steps.append(py('proposed_S7_first',scripts/'S7_network_cleanup.py',*arguments,outputs=[network]))
    steps.append(py('proposed_S7_final',scripts/'S7_network_cleanup.py',*arguments,
                    '--keep-output','--resume','--mainroad-force-leaf-max-m','150',outputs=[network]))
    # Use the same manuscript trajectory-coverage raster as the five reference methods.
    steps.append(module('proposed_evaluation_inputs',REFERENCE,'prepare','--profile','paper',
                        '--output-root',out/'assessment','--tracks-archive',path('chicago_tracks'),
                        '--map-archive',path('chicago_map'),'--resume',outputs=[out/'assessment/inputs/input_manifest.json']))
    steps.append(module('proposed_evaluation_support',REFERENCE,'support','--profile','paper',
                        '--output-root',out/'assessment','--resume',outputs=[out/'assessment/support/.chicago_paper_support.json']))
    steps.append(py('proposed_evaluate',RUNTIME/'workflow/prepare.py','chicago-evaluate',
                    '--network',network,'--assessment',out/'assessment','--output',out/'evaluation.json',
                    outputs=[out/'evaluation.json']))
    return steps


def main(argv=None):
    parser=argparse.ArgumentParser(description='Chicago: six-method reconstruction and evaluation')
    parser.add_argument('--method',choices=(*METHODS,'all'),default='all')
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args(argv)
    methods=list(METHODS) if args.method=='all' else [args.method]
    steps=[]
    references=[method for method in methods if method!='proposed']
    if references:
        output=intermediate('chicago')/'references'
        # Prepare observations, execute each native method,
        # normalizes geometry, rebuilds the assessment raster and evaluates.
        steps.append(module('references_'+args.method,REFERENCE,'all','--profile','paper',
                    '--method','All' if len(references)==5 else references[0].title(),
                    '--output-root',output,'--tracks-archive',path('chicago_tracks'),
                    '--map-archive',path('chicago_map'),'--resume',
                    outputs=[output/m/'normalized/road_network.gpkg' for m in references]+
                            [output/m/'evaluation.json' for m in references]))
    if 'proposed' in methods:
        steps+=proposed_steps(intermediate('chicago')/'proposed')
    tools=[]
    if 'ahmed' in methods: tools+=['java','javac']
    if 'karagiorgou' in methods: tools+=['matlab']
    result=finish({'group':'chicago','methods':methods,'steps':steps},args,tools)
    if not args.plan:
        from .publish import publish
        publish('chicago',methods)
    return result


if __name__=='__main__':raise SystemExit(main())
