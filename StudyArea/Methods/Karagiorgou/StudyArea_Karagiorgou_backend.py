"""Auditable MATLAB derivatives with full-workspace loop checkpoints.

Scientific predicates and order-sensitive loops stay in MATLAB. Generated
files, inputs and helper files are hash-bound before a resume is allowed.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .StudyArea_Karagiorgou_source import stage_accelerated_source
from .StudyArea_Karagiorgou_runner import INTERMEDIATES
from experiments.chicago_six_methods.reference_methods.methods.karagiorgou_native import (
    _track_files, validate_native_outputs,
)

PACKAGE = Path(__file__).resolve().parent
PUBLIC_SOURCE = PACKAGE / "StudyArea_Karagiorgou_Source/tracebundle/matlab"
OPEN = re.compile(r"^\s*(for|parfor|while|if|switch|try)\b")
CLOSE = re.compile(r"^\s*end\s*;?\s*(%.*)?$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): digest(p)
            for p in sorted(root.rglob('*')) if p.is_file() and
            (p.suffix == '.m' or p.name in ('track_manifest.txt','track_directory.txt'))}


def validate_hashes(root: Path, expected: dict[str, str]) -> None:
    for name, expected_hash in expected.items():
        p = root / name
        if not p.is_file() or digest(p) != expected_hash:
            raise ValueError(f'Run source or input changed: {p}')


def validate_import_identity(previous: dict, hashes: dict, order: list[str]) -> None:
    if previous['input_hashes'] != hashes:
        raise ValueError('Imported first-stage inputs changed')
    if previous.get('input_order') != order:
        raise ValueError('Imported first-stage input order changed')


def validate_import_files(native: Path, expected: dict[str,str]) -> None:
    validate_hashes(native,expected)
    validate_hashes(native/'tracebundle_intermediates',expected)


def matlab_pids(native: Path) -> list[int]:
    """Detect MATLAB descendants even if their Python launcher has exited.

    Bind to the exact native directory in the command line, not a reusable PID.
    A failed process query must stop startup rather than risk double execution.
    """
    needle=str(native.resolve()).replace("'","''")
    command=("$ErrorActionPreference='Stop';$needle='"+needle+"';"
        "$p=@(Get-CimInstance Win32_Process -Filter \"Name='MATLAB.exe' OR Name='matlab.exe'\" | "
        "Where-Object {$_.CommandLine -and $_.CommandLine.Contains($needle)} | "
        "ForEach-Object {$_.ProcessId});ConvertTo-Json -InputObject $p -Compress")
    encoded=base64.b64encode(command.encode('utf-16le')).decode('ascii')
    result=subprocess.run(['powershell.exe','-NoProfile','-EncodedCommand',encoded],
                           capture_output=True,text=True,encoding='utf-8',
                           errors='backslashreplace',timeout=30)
    if result.returncode:
        raise RuntimeError('Cannot verify existing MATLAB process ownership: '+result.stderr[-1000:])
    return [int(value) for value in json.loads(result.stdout)]


def completed_outputs(root: Path, identity: dict) -> dict | None:
    """Accept only a final, identity-bound marker and a structurally valid graph."""
    native=root/'native'
    try:
        marker=json.loads((native/'completion.json').read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):
        return None
    if marker.get('run_id')!=identity.get('run_id') or marker.get('phase')!=identity.get('final_phase'):
        return None
    vertices,edges=validate_native_outputs(native/'tracebundle_vertices.txt',native/'tracebundle_edges.txt')
    if vertices!=marker['vertices'] or edges!=marker['edges']:
        raise ValueError('Final graph counts differ from completion marker')
    return {'vertices':vertices,'edges':edges,'native_output_sha256':{
        name:digest(native/name) for name in ('tracebundle_vertices.txt','tracebundle_edges.txt')}}


@contextmanager
def run_lock(root: Path):
    """OS-owned lock: a process exit releases it without deleting a run."""
    import msvcrt
    with (root / 'run.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('This run is already active') from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def split_blocks(source: str) -> list[str]:
    """Split top-level MATLAB constructs; retain nested control flow verbatim.

    The export tail is indivisible: numeric fopen handles cannot be resumed.
    Fail closed on unbalanced/inline constructs rather than corrupting code.
    """
    result, pending, nested = [], [], []
    depth = 0
    export = False
    for line in source.splitlines(keepends=True):
        if not depth and re.search(r'\bfopen\(', line) and not line.lstrip().startswith('%'):
            export = True
        if export:
            pending.append(line)
            continue
        if depth:
            nested.append(line)
            if OPEN.match(line):
                depth += 1
            if CLOSE.match(line):
                depth -= 1
                if depth == 0:
                    result.append(''.join(nested))
                    nested = []
        elif OPEN.match(line):
            if pending:
                result.append(''.join(pending))
                pending = []
            nested = [line]
            depth = 1
        else:
            pending.append(line)
    if depth:
        raise ValueError('Unbalanced MATLAB control blocks')
    if pending:
        result.append(''.join(pending))
    return result


def instrument(source: str, start: int) -> str:
    chunks = []
    for step, block in enumerate(split_blocks(source), start):
        header, *rest = block.splitlines(keepends=True)
        match = re.match(r'\s*for\s+(\w+)\s*=\s*(.*?)\s*;?\s*(?:%.*)?\n?$', header)
        if match:
            var, values = match.groups()
            body = ''.join(rest)
            # Remove only the closing end of the top-level loop.
            close = list(re.finditer(r'^\s*end\s*;?\s*(?:%[^\n]*)?$', body, re.M))[-1]
            body = body[:close.start()]
            emitted = (
                f'if tb_loop_active~={step}\n'
                f' tb_loop_values={values}; tb_loop_next=1; tb_loop_active={step};\nend\n'
                'for tb_loop_cursor=tb_loop_next:numel(tb_loop_values)\n'
                f'{var}=tb_loop_values(tb_loop_cursor);\n' + body +
                '\ntb_loop_next=tb_loop_cursor+1; tb_checkpoint(false);\nend\n'
            )
        elif re.match(r'\s*while\b', header):
            close = list(re.finditer(r'^\s*end\s*;?\s*(?:%[^\n]*)?$', block, re.M))[-1]
            emitted = block[:close.start()] + '\ntb_checkpoint(false);\n' + block[close.start():]
        else:
            emitted = block
        chunks.append(
            f'if tb_phase<={step}\ntb_phase={step};\n' + emitted +
            f'\ntb_phase={step + 1}; tb_loop_active=0; tb_checkpoint(true);\nend\n'
        )
    return '\n'.join(chunks)


def optimize_extraction(source: str) -> str:
    """Buffer outputs per track; keep scientific operations and their order."""
    source = source.replace('[fnames,path]=uigetfile(\'*.txt\',\'...select trajectory files!\',\'MultiSelect\',\'on\');',
                            '[fnames,path]=tracebundle_select_tracks();')
    source = re.sub(r'^clear;\s*\n', '', source, flags=re.M)
    begin = source.index('for g=1:length(fnames(1,:))')
    finish = source.index('\nslk=[];', begin)
    track_loop = source[begin:finish]
    # These arrays/counters have no consumer in either public script output.
    track_loop = '\n'.join(line for line in track_loop.split('\n')
                           if not any(token in line for token in (
                               'rectr=vertcat', 'angles=vertcat', 'xy=vertcat',
                               "dd=dd+polylineLength", 'total=total+length')))
    track_loop = track_loop.replace('pts_ln=vertcat(pts_ln,', 'tb_pts_track=vertcat(tb_pts_track,')
    track_loop = track_loop.replace('my_corner_pts=vertcat(my_corner_pts,',
                                    'tb_corner_track=vertcat(tb_corner_track,')
    track_loop = track_loop.replace('    mat=load', '    tb_pts_track=[]; tb_corner_track=[];\n    mat=load', 1)
    tail = track_loop.rfind('\nend')
    track_loop = (track_loop[:tail] +
                  '\n tb_pts_parts{g}=tb_pts_track; tb_corner_parts{g}=tb_corner_track;\n' +
                  track_loop[tail:])
    source = (source[:begin] +
              'tb_pts_parts=cell(1,length(fnames)); tb_corner_parts=cell(1,length(fnames));\n' +
              track_loop + '\npts_ln=vertcat(tb_pts_parts{:});\n'
              'my_corner_pts=vertcat(tb_corner_parts{:});\n'
              'clear tb_pts_parts tb_corner_parts tb_pts_track tb_corner_track;\n' + source[finish:])
    begin = source.index('for g=1:length(vehicles)')
    finish = source.index('% ----------------------------------------------------------', begin)
    loop = source[begin:finish]
    loop = loop.replace('    tr=find', '    tb_slk_track=[];\n    tr=find', 1)
    loop = loop.replace('slk=vertcat(slk,mllk);', 'tb_slk_track=vertcat(tb_slk_track,mllk);')
    tail = loop.rfind('\nend')
    loop = loop[:tail] + '\n tb_slk_parts{g}=tb_slk_track;\n' + loop[tail:]
    source = (source[:begin] + 'tb_slk_parts=cell(1,length(vehicles));\n' + loop +
            '\nslk=vertcat(tb_slk_parts{:});\nclear tb_slk_parts tb_slk_track;\n' + source[finish:])
    begin = source.index('        ipts=[aa(1,1) aa(1,2)];')
    finish = source.index('        aaa=centroid(ipts);', begin)
    return source[:begin] + '        ipts=tb_pair_intersections(mpts,aa,p1);\n\n' + source[finish:]


def repair_cluster_membership(source: str) -> str:
    """Use the membership recorded in column 12, not an ambiguous sample key.

    Extraction can record the same (vehicle, sample) in different clusters.
    Preserve both records and their ordering; do not choose the first cluster.
    """
    for suffix, row in [('1', 'h'), ('2', 'h+1')]:
        original = (f'rx{suffix}=find(inter_to_cluster(:,1)==pts_to_i({row},3) & '
                    f'inter_to_cluster(:,2)==pts_to_i({row},4));')
        fixed = original[:-2] + f' & inter_to_cluster(:,3)==pts_to_i({row},12));'
        if source.count(original) != 1:
            raise ValueError('Cannot identify unique public membership lookup')
        source = source.replace(original, fixed, 1)
    return source


def optimize_bundling(source: str) -> str:
    """Index immutable points, and capture complete per-link sweep state."""
    loaded = "pts_ln=load(lfile) ; % load the file"
    source = source.replace(loaded, loaded + "\ntb_point_grid=tracebundle_build_endpoint_grid(" +
        "[pts_ln.pts_ln(:,1:2) pts_ln.pts_ln(:,1:2)],200.0);")
    source = re.sub(r'inpolygon\(\s*pts_ln\.pts_ln\(:,1\),\s*pts_ln\.pts_ln\(:,2\),',
                    'tb_indexed_inpolygon(tb_point_grid,pts_ln.pts_ln(:,1),pts_ln.pts_ln(:,2),', source)
    source = re.sub(r'\binpolygon\((centre_pts\(:,1\)|netsections\(:,2\)|elk\(:,1\)|slk\(:,1\))',
                    r'tb_cached_inpolygon(\1', source)
    outer = 'while prevlen~=rlen && scounter<=gscounter'
    inner = '    while (k<=rlen && netnodes(k,1)<=maxk)||k<=rlen'
    start = source.index(outer)
    inner_start = source.index(inner, start)
    # Locate the matching ends in the preserved public control flow.
    lines = source.splitlines(keepends=True)
    offsets, cursor = [], 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    depth, inner_end, outer_end = 0, None, None
    for offset, line in zip(offsets, lines):
        if offset < start:
            continue
        if OPEN.match(line):
            depth += 1
        if CLOSE.match(line):
            if depth == 2 and offset > inner_start and inner_end is None:
                inner_end = offset
            depth -= 1
            if depth == 0:
                outer_end = offset
                break
    if inner_end is None or outer_end is None:
        raise ValueError('Cannot identify public merge sweep boundaries')
    # Apply edits backwards so original offsets remain valid.
    source = source[:outer_end] + 'tb_merge_active=false;\n' + source[outer_end:]
    source = source[:inner_end] + '        tb_checkpoint(false);\n' + source[inner_end:]
    source = source[:inner_start] + '    tb_merge_active=true;\n    end\n' + source[inner_start:]
    new_outer = ('tb_merge_active=false;\n'
                 'while tb_merge_active || (prevlen~=rlen && scounter<=gscounter)\n'
                 '    if ~tb_merge_active')
    source = source[:start] + source[start:].replace(outer, new_outer, 1)
    # Synthetic merge-only recovery tests have no trajectory association block.
    if 'rx1=find(inter_to_cluster' in source:
        source = repair_cluster_membership(source)
    return source


def prepare(tracks: Path, root: Path, manifest: Path | None, *, count: int | None = None,
            strict: bool = False, first_stage_from: Path | None = None,
            source_crs: str = 'EPSG:32647') -> Path:
    """Create a new owned run; never touch an earlier run directory."""
    files, order = _track_files(tracks, manifest)
    if count is not None:
        files, order = files[:count], order[:count]
    if not files:
        raise ValueError('No tracks selected')
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    selected = root / 'inputs'
    selected.mkdir()
    for p in files:
        shutil.copy2(p, selected / p.name)
    subset_manifest = root / 'input_manifest.json'
    subset_manifest.write_text(json.dumps({'track_order': order,
        'track_sha256': {p.name: digest(p) for p in files}}, indent=2), encoding='utf-8')
    native = stage_accelerated_source(PUBLIC_SOURCE, selected, root / 'native',
                                      input_manifest=subset_manifest)
    # Both generated scripts execute in native; input references survive relocation.
    (native / 'track_directory.txt').write_text('../inputs\n', encoding='utf-8')
    if strict:
        extraction = (PUBLIC_SOURCE / 'source/intersection_nodes_extraction.m').read_text(encoding='utf-8')
        extraction = extraction.replace("[fnames,path]=uigetfile('*.txt','...select trajectory files!','MultiSelect','on');",
                                        '[fnames,path]=tracebundle_select_tracks();')
        extraction = re.sub(r'^clear;\s*\n', '', extraction, flags=re.M)
    else:
        # Optimize the already indexed derivative.
        extraction = optimize_extraction((native / 'intersection_nodes_extraction.m').read_text(encoding='utf-8'))
    bundling = (native / 'tracebundle.m').read_text(encoding='utf-8')
    if not strict:
        bundling = optimize_bundling(bundling)
    # Keep originals: they are scientific outputs and recovery boundaries.
    bundling = re.sub(r"^delete\('[^']+\.mat'\);", '% Keep intermediate for audit and recovery.', bundling, flags=re.M)
    boundary = '\n'.join(f"copyfile('{n}',fullfile('tracebundle_intermediates','{n}'));" for n in INTERMEDIATES)
    extraction += "\nif exist('tracebundle_intermediates','dir')~=7\n mkdir('tracebundle_intermediates');\nend\n" + boundary + '\n'
    nblocks = len(split_blocks(extraction))
    (native / 'extraction_resume.m').write_text(instrument(extraction, 1), encoding='utf-8')
    (native / 'bundle_resume.m').write_text(instrument(bundling, nblocks + 1), encoding='utf-8')
    for name in ('tb_checkpoint.m', 'tb_restore.m', 'tb_entry.m',
                 'tb_pair_intersections.m', 'tb_indexed_inpolygon.m', 'tb_cached_inpolygon.m'):
        shutil.copy2(PACKAGE / 'StudyArea_Karagiorgou_MATLAB' / name, native / name)
    identity = {'schema_version': 1, 'execution_class': 'MATLAB implementation' if strict else
                'MATLAB implementation',
                'count': len(files), 'source_crs': source_crs,
                'parameters': {'cluster_distance_m': 40, 'angle_tolerance_deg': 12.5},
                'source_hashes': file_hashes(native),
                'input_hashes': {p.name: digest(p) for p in selected.iterdir()}}
    identity['input_order']=[p.name for p in files]
    if first_stage_from is not None:
        from scipy.io import savemat
        previous = json.loads((first_stage_from / 'identity.json').read_text(encoding='utf-8'))
        if 'input_order' not in previous:
            # Older locally generated validation runs recorded consumed order here.
            previous['input_order']=(first_stage_from/'native/track_manifest.txt').read_text(encoding='utf-8').splitlines()
        validate_import_identity(previous,identity['input_hashes'],identity['input_order'])
        origin = first_stage_from / 'native/tracebundle_intermediates'
        for name in INTERMEDIATES:
            if not (origin / name).is_file():
                raise ValueError(f'Incomplete imported first stage: {name}')
        (native / 'tracebundle_intermediates').mkdir()
        for name in INTERMEDIATES:
            shutil.copy2(origin / name, native / name)
            shutil.copy2(origin / name, native / 'tracebundle_intermediates' / name)
        identity['imported_first_stage'] = {name: digest(origin / name) for name in INTERMEDIATES}
    identity['final_phase']=nblocks+len(split_blocks(bundling))+1
    identity['run_id']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode('utf-8')).hexdigest()
    if first_stage_from is not None:
        savemat(native / 'checkpoint_current.mat', {'tb_phase': nblocks + 1,
            'tb_loop_next': 1, 'tb_loop_active': 0, 'tb_loop_values': [],
            'tb_run_id':identity['run_id']})
    (root / 'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    return root


def retry_allowed(exit_code: int, advanced: bool, attempt: int, limit: int) -> bool:
    return (exit_code & 0xffffffff)==0xc0000374 and advanced and attempt<limit


def run(root: Path, matlab: str, timeout: float | None = None, *, disable_jit: bool = False,
        max_restarts: int = 0) -> int:
    """Resume a provenance-bound run with streamed output and no shell quoting."""
    root = root.resolve()
    with run_lock(root):
        history_path=root/'supervisor_history.json'
        history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
        for attempt in range(max_restarts+1):
            checkpoint=root/'native/checkpoint_current.mat'
            before=checkpoint.stat().st_mtime_ns if checkpoint.exists() else 0
            code=_run_locked(root,matlab,timeout,disable_jit=disable_jit)
            after=checkpoint.stat().st_mtime_ns if checkpoint.exists() else 0
            history.append({'attempt':len(history)+1,'finished_unix':time.time(),
                            'launcher_exit_code':code,'checkpoint_advanced':after>before})
            history_path.write_text(json.dumps(history,indent=2),encoding='utf-8')
            if not retry_allowed(code,after>before,attempt,max_restarts):
                return code
            print('[resume] MATLAB heap crash; a newer checkpoint exists; recovering',flush=True)
        return 1


def _run_locked(root: Path, matlab: str, timeout: float | None, *, disable_jit: bool = False) -> int:
    identity = json.loads((root / 'identity.json').read_text(encoding='utf-8'))
    validate_hashes(root / 'native', identity['source_hashes'])
    validate_hashes(root / 'inputs', identity['input_hashes'])
    if 'imported_first_stage' in identity:
        validate_import_files(root/'native',identity['imported_first_stage'])
    native = root / 'native'
    active=matlab_pids(native)
    if active:
        raise RuntimeError(f'This run is already active in MATLAB: {active}')
    already=completed_outputs(root,identity)
    if already is not None:
        report_path=root/'report.json'
        previous={}
        if report_path.is_file():
            previous=json.loads(report_path.read_text(encoding='utf-8'))
            if previous.get('state')=='completed':
                validate_hashes(native,previous.get('native_output_sha256',{}))
        result={**previous,'state':'completed','reused_completed_outputs':True,**already}
        report_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
        return 0
    prefix = "feature('accel','off'); " if disable_jit else ''
    command = prefix + "cd('" + str(native).replace("'", "''") + "'); tb_entry;"
    started = time.monotonic()
    with (root / 'console.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen([matlab, '-batch', command], cwd=native,
                                   stdout=log, stderr=subprocess.STDOUT)
        (root / 'process.json').write_text(json.dumps({'pid': process.pid,
            'started_unix': time.time(), 'launcher_pid': os.getpid()}), encoding='utf-8')
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only bounded benchmarks may request timeout; terminate their own tree.
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True, check=False)
            code = 124
    report = {'exit_code': code, 'elapsed_seconds': time.monotonic() - started,
              'jit_disabled': disable_jit}
    finished=completed_outputs(root,identity)
    if finished is not None:
        report.update(finished)
        report['state'] = 'completed'
        if code:
            report['engine_warning']='MATLAB exited abnormally after its final completion marker; outputs validated'
    else:
        report['state'] = 'benchmark_timeout' if code == 124 else 'failed'
    (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    return 0 if report['state']=='completed' else code or 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--tracks', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--count', type=int)
    p.add_argument('--strict', action='store_true')
    p.add_argument('--first-stage-from', type=Path,
                   help='Reuse declared scientific intermediates only with identical input hashes')
    p.add_argument('--source-crs',default='EPSG:32647')
    p = sub.add_parser('run')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--matlab', default='matlab')
    p.add_argument('--benchmark-timeout', type=float)
    p.add_argument('--disable-jit', action='store_true',
                   help='Diagnostic MATLAB compiler setting; scientific code unchanged')
    p.add_argument('--max-restarts',type=int,default=0,
                   help='Recover this many heap crashes only when a newer checkpoint exists')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.tracks, args.root, args.manifest, count=args.count, strict=args.strict,
                      first_stage_from=args.first_stage_from,source_crs=args.source_crs))
        return 0
    if args.max_restarts<0:
        parser.error('--max-restarts must be nonnegative')
    return run(args.root, args.matlab, args.benchmark_timeout, disable_jit=args.disable_jit,
               max_restarts=args.max_restarts)


if __name__ == '__main__':
    sys.exit(main())
