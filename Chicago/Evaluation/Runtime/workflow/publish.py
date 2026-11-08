"""Publish only the final products computed by the current method run."""
import shutil
from .common import intermediate
from .products import native_network, network_file, evaluation_file

def publish(group,methods):
    for method in methods:
        source=native_network(group,method)
        if not source.is_file():raise FileNotFoundError(source)
        target=network_file(group,method);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if group=='chicago':
            base=intermediate(group)
            assessment=(base/'proposed/evaluation.json' if method=='proposed' else base/'references'/method/'evaluation.json')
            shutil.copy2(assessment,evaluation_file(method))
        print('WROTE',target,flush=True)
