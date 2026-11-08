"""Method-qualified publication products and native intermediate locations."""
from .common import dataset_results, intermediate

METHODS={'ahmed':'Ahmed','biagioni':'Biagioni','davies':'Davies','edelkamp':'Edelkamp',
         'karagiorgou':'Karagiorgou','proposed':'Proposed','full':'Proposed_Full',
         'wo_activity':'Proposed_WithoutActivity','wo_uturn':'Proposed_WithoutHairpin',
         'wo_evidence_topology':'Proposed_WithoutTopology'}

def dataset(group):return 'Chicago' if group=='chicago' else 'StudyArea'
def stem(group,method):return dataset(group)+'_'+METHODS[method]
def result_folder(group,method):
    base=dataset_results(group)
    if group=='ablation':base=base/'StudyArea_Proposed_Ablation'
    return base/stem(group,method)
def network_file(group,method):return result_folder(group,method)/(stem(group,method)+'_Network.gpkg')
def evaluation_file(method):return result_folder('chicago',method)/(stem('chicago',method)+'_Evaluation.json')

def native_network(group,method):
    base=intermediate(group)
    if group=='chicago':
        return (base/'proposed/S7_Network/08_final_clean_network/road_network_final_clean.gpkg' if method=='proposed'
                else base/'references'/method/'normalized/road_network.gpkg')
    if group=='mainling':
        return (base/'proposed/S7_Network/08_final_clean_network/road_network_final_clean.gpkg' if method=='proposed'
                else base/'references'/METHODS[method]/'normalized/road_network.gpkg')
    return base/method/'S7_Network/08_final_clean_network/road_network_final_clean.gpkg'

def table_path(name):
    group='chicago' if name.startswith('chicago_') or name=='table_1.csv' else 'mainling'
    prefix=dataset(group)
    names={'chicago_metrics':'AccuracyAndStructure','chicago_tolerance':'ToleranceMetrics',
           'mainling_metrics':'GraphMetrics','ablation_metrics':'AblationMetrics',
           'ablation_ratios':'AblationRatios','table_1':'Table1','table_2':'Table2','table_3':'Table3',
           'table_A1':'TableA1','parameters_full':'Proposed_ParameterCatalogue',
           'reported_metrics':'ReportedMetrics'}
    key,suffix=name.rsplit('.',1)
    directory=dataset_results(group)/(prefix+'_TablesAndFigures')
    directory.mkdir(parents=True,exist_ok=True)
    return directory/(prefix+'_'+names.get(key,key)+'.'+suffix)

def figure_path(number,suffix='png'):
    group='chicago' if number==8 else 'mainling';prefix=dataset(group)
    directory=dataset_results(group)/(prefix+'_TablesAndFigures')
    directory.mkdir(parents=True,exist_ok=True)
    return directory/f'{prefix}_Figure{number}.{suffix}'

def map_folder(number):
    directory=dataset_results('mainling')/f'StudyArea_TablesAndFigures/StudyArea_Figure{number}'
    directory.mkdir(parents=True,exist_ok=True)
    return directory
