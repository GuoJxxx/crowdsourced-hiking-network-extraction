import argparse
from .common import finish,path,intermediate
from .pipeline import mainling_pipeline

MODES=('full','wo_activity','wo_uturn','wo_evidence_topology')

def main(argv=None):
    parser=argparse.ArgumentParser(description='Mainling mechanism ablations')
    parser.add_argument('--mode',choices=(*MODES,'all'),default='all')
    parser.add_argument('--plan',action='store_true',help='Print commands without running experiments')
    args=parser.parse_args(argv)
    modes=list(MODES) if args.mode=='all' else [args.mode]
    steps=[]
    for mode in modes:
        steps+=mainling_pipeline(intermediate('ablation')/mode,ablation=mode)
    result=finish({'group':'ablation','modes':modes,'steps':steps},args)
    if not args.plan:
        from .publish import publish
        publish('ablation',modes)
    return result


if __name__=='__main__':raise SystemExit(main())
