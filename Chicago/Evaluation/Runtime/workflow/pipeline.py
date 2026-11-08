"""Issue the preserved S2--S7 commands from their parameter schema."""
import json
from .common import ROOT, path, py, source, parameter_file

STAGES = {'S2':'S2_clean_tracks.py','S3':'S3_activity_zone_filter.py',
          'S4':'S4_uturn_consensus_gravity.py','S5':'S5_directional_kde.py',
          'S6':'S6_route_locked_skeleton.py','S7':'S7_network_cleanup.py'}


def stage_arguments(schema, config, stage):
    result = []
    for parameter in schema['stages'][stage]['parameters']:
        value = config['stages'][stage].get(parameter['dest'])
        kind = parameter['kind']
        if kind == 'bool_optional':
            result.append(parameter['positive_option'] if bool(value) else parameter['negative_option'])
        elif kind == 'bool_flag':
            if bool(value) == bool(parameter['flag_value']):
                result.append(parameter['option'])
        elif value is not None and str(value).strip():
            result += [parameter['option'], str(value)]
    return result


def mainling_pipeline(output, *, ablation=None):
    package = source('mainling_ablation' if ablation else 'mainling_full')
    config = json.loads(parameter_file('ablation.json' if ablation else 'mainling.json').read_text(encoding='utf-8-sig'))
    schema = json.loads((package/'configs/parameter_schema.json').read_text(encoding='utf-8-sig'))
    dirs = {stage: output/config['paths'][stage.lower()+'_dir'] for stage in STAGES}
    raw = path('mainling_observed')
    code = package/'code'
    s2, s3, s4, s5, s6, s7 = (dirs[f'S{i}'] for i in range(2,8))
    kde = s5/'kde_directional_along4m_cross1p5m_float32.tif'
    support = s5/'independent_track_support.tif'
    distance = s5/'distance_to_raw_trajectory_m.tif'
    raw_union = s5/'raw_trajectory_union.tif'
    report = s5/'directional_kde_report.json'
    gravity = s4/'all_cleaned_points.csv'
    gravity_report = s4/'gravity_uturn_report.json'
    activity = s3/'activity_zones.gpkg'
    uturn = s4/'uturn_consensus.gpkg'
    prefix = f'{ablation or "proposed"}_'
    steps = []
    def add(stage, arguments, outputs):
        steps.append(py(prefix+stage,code/STAGES[stage],*arguments,
                        *stage_arguments(schema,config,stage),outputs=outputs))
        steps[-1]['optional_paths']=['--auxiliary','--rescued','--uturn-gpkg','--activity-zones']
    add('S2',['--input',raw,'--output',s2,'--boundary',path('boundary'),'--dem',path('dem')],
        [s2/'all_cleaned_points.csv',s2/'cleaned_hiking_tracks'])
    args = ['--input',s2/'all_cleaned_points.csv','--output',s3]
    if ablation=='wo_activity': args += ['--disable-activity-aware']
    add('S3',args,[s3/'road_candidate_points.csv',activity])
    args = ['--input',s3/'road_candidate_points.csv','--output',s4]
    if ablation=='wo_uturn': args += ['--disable-uturn-semantics']
    add('S4',args,[gravity,gravity_report])
    add('S5',['--input',gravity,'--output',s5,'--gravity-report',gravity_report],
        [report,kde,support,distance,raw_union])
    semantics = []
    if ablation!='wo_uturn': semantics += ['--uturn-gpkg',uturn]
    if ablation!='wo_activity': semantics += ['--activity-zones',activity]
    if ablation=='wo_evidence_topology': semantics += ['--disable-evidence-topology-optimization']
    primary = s6/'skeleton_primary_route_locked.tif'
    add('S6',['--kde',kde,'--support',support,'--distance',distance,'--raw-union',raw_union,
              '--gravity-points',gravity,'--output',s6,*semantics],[primary])
    add('S7',['--primary',primary,'--support',support,'--kde',kde,'--distance',distance,
              '--gravity-points',gravity,'--raw-tracks',raw,'--tile-report',report,'--output',s7,
              '--auxiliary',s6/'skeleton_auxiliary_singletrack_safe.tif',
              '--rescued',s6/'skeleton_support_rescued.tif',*semantics,'--dem',path('dem')],
        [s7/'08_final_clean_network/road_network_final_clean.gpkg'])
    return steps
