"""Data-format preparation and graph conversion used by the group launchers."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'settings.json').is_file())
for location in (ROOT,ROOT/'Chicago/Evaluation/Runtime',ROOT/'StudyArea/Evaluation/Runtime',ROOT/'StudyArea/Evaluation/Runtime/mainling_references'):
    sys.path.insert(0,str(location))

FIELDS=('track_id','source_file','document_name','placemark_name','segment_id','point_index',
        'raw_point_index','longitude','latitude','elevation_m','time_raw','time_utc','time_china','time_source')


def chicago_observed(archive,output):
    output.mkdir(parents=True,exist_ok=True)
    point_count=0
    with zipfile.ZipFile(archive) as source:
        members=sorted(n for n in source.namelist() if re.fullmatch(r'chicago/trips/trip_\d+\.txt',n))
        for member in members:
            source_name=Path(member).name
            stem=Path(source_name).stem
            with (output/(stem+'__seg001__gxtrack.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream,lineterminator='\r\n')
                writer.writerow(FIELDS)
                for index,line in enumerate(source.read(member).decode('utf-8').splitlines(),1):
                    if not line.strip(): continue
                    x,y,timestamp=(float(v) for v in line.split())
                    moment=datetime.fromtimestamp(timestamp,tz=timezone.utc)
                    writer.writerow((stem+'__seg001',source_name,'Chicago_trips',stem,1,index,index,x,y,'',str(timestamp),
                                     moment.isoformat(timespec='milliseconds').replace('+00:00','Z'),
                                     moment.astimezone(timezone(timedelta(hours=8))).isoformat(timespec='milliseconds'),
                                     'unix_timestamp'))
                    point_count+=1
    report={'files':len(members),'points':point_count,'source_crs':'EPSG:26916'}
    if report!={'files':889,'points':118360,'source_crs':'EPSG:26916'}:
        raise ValueError(f'Unexpected Chicago input counts: {report}')
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('chicago');p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('mainling-normalize');p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    p=sub.add_parser('chicago-evaluate');p.add_argument('--network',type=Path,required=True);p.add_argument('--assessment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='chicago':
        chicago_observed(args.archive,args.output)
    elif args.command=='mainling-normalize':
        from reference_tools import normalize_text_graph
        result=normalize_text_graph(args.root/'native/tracebundle_vertices.txt',args.root/'native/tracebundle_edges.txt',
                                    args.output or args.root/'normalized',method='Karagiorgou',source_crs='EPSG:32647',target_crs='EPSG:32647')
        print(json.dumps(result,ensure_ascii=False))
    else:
        tool=ROOT/'Chicago/Evaluation/Chicago_Assessment_Tools/evaluate_final_network.py'
        subprocess.run([sys.executable,'-B',str(tool),'--network',str(args.network),'--layer','edges',
                        '--raw-union',str(args.assessment/'support/S5_KDE/raw_trajectory_union.tif'),
                        '--vertices',str(args.assessment/'inputs/truth/chicago_vertices_osm.txt'),
                        '--edges',str(args.assessment/'inputs/truth/chicago_edges_osm.txt'),
                        '--coverage-m','15','--tolerances','4,6,8,10,15,20,25,30,35,40',
                        '--output',str(args.output)],check=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
