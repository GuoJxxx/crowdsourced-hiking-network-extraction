"""Add progress reporting to staged copies of the reference MATLAB scripts."""

from pathlib import Path
import re


HELPER = Path(__file__).with_suffix('.m')

# Line anchors identify outer loops in the bundled reference source. Each hook
# observes existing counters; the original statements and loop bounds stay intact.
RULES = {
    'intersection_nodes_extraction.m': (
        (79, 'for g=1:length(fnames(1,:))', "'Reading trajectories'", 'g-1', 'length(fnames(1,:))', 'items'),
        (216, 'for g=1:length(vehicles)', "'Preparing trajectory segments'", 'g-1', 'length(vehicles)', 'items'),
        (268, 'for g=1:length(my_corner_pts(:,1))', "'Analysing turns'", 'g-1', 'length(my_corner_pts(:,1))', 'items'),
        (472, 'while g<=length(my_corner_pts(:,1))', "'Clustering intersections'", 'max(0,length(my_corner_pts(:,1))-g+1)', 'length(my_corner_pts(:,1))', 'remaining'),
    ),
    'tracebundle.m': (
        (77, 'for g=1:length(vehicles)', "'Matching samples to intersections'", 'g-1', 'length(vehicles)', 'items'),
        (129, 'for g=1:length(cluster_to_tr(:,1))', "'Grouping link samples'", 'g-1', 'length(cluster_to_tr(:,1))', 'items'),
        (235, 'for g=1:length(cluster_to_tr(:,1))', "'Compacting link samples'", 'g-1', 'length(cluster_to_tr(:,1))', 'items'),
        (443, 'for g=1:length(cluster_to_tr(:,1))', "'Calculating link distances'", 'g-1', 'length(cluster_to_tr(:,1))', 'items'),
        (777, 'while (k<=rlen && netnodes(k,1)<=maxk)||k<=rlen', "sprintf('Merging links - pass %d of %d',scounter,gscounter)", 'k-1', 'rlen', 'items'),
        (14865, 'for g=1:length(cluster_to_tr(:,1))', "'Refining link geometry'", 'g-1', 'length(cluster_to_tr(:,1))', 'items'),
        (15386, 'for k=1:length(n_edges)', "'Writing edges'", 'k-1', 'length(n_edges)', 'items'),
        (15391, 'for k=1:length(n_nodes)', "'Writing vertices'", 'k-1', 'length(n_nodes)', 'items'),
    ),
}


def instrument_matlab_source(content: bytes, name: str) -> bytes:
    """Insert marked calls without changing any original source line."""
    lines = content.decode('utf-8').splitlines(keepends=True)
    newline = '\r\n' if b'\r\n' in content else '\n'
    before, after = {}, {}
    for number, anchor, phase, completed, total, mode in RULES[name]:
        index = number - 1
        if index >= len(lines) or lines[index].strip() != anchor:
            raise ValueError(f'Unexpected reference progress anchor: {name}:{number}')
        indent = lines[index][:len(lines[index])-len(lines[index].lstrip())]
        end = next((j for j in range(index+1, len(lines))
                    if re.fullmatch(re.escape(indent)+r'end\s*;?\s*', lines[j].rstrip('\r\n'))), None)
        if end is None:
            raise ValueError(f'Missing reference progress anchor end: {name}:{number}')

        def call(done, count, action, indentation):
            return (f"{indentation}Chicago_Karagiorgou_progress({phase},{done},{count},'{action}'); "
                    f'% TNR_PROGRESS{newline}')

        before.setdefault(index, []).append(call('0', total, 'start', indent))
        after.setdefault(index, []).append(call(completed, total, mode, indent+'    '))
        after.setdefault(end, []).append(call('0', '0', 'finish', indent))
    return ''.join(''.join(before.get(i, []))+line+''.join(after.get(i, []))
                   for i, line in enumerate(lines)).encode('utf-8')
