"""Audited recovery for a sample assigned to multiple intersection clusters.

The parent run is immutable. Only the two ambiguous lookups and the identity
field of copied checkpoints change; all scientific workspace datasets remain
identical. Recovery is allowed only before any ambiguous vehicle was processed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy.io import loadmat

from .StudyArea_Karagiorgou_backend import (INTERMEDIATES, digest, file_hashes, matlab_pids,
                        repair_cluster_membership, run_lock, validate_hashes)


def workspace_digest(path: Path) -> str:
    """Hash every logical dataset except the explicitly migrated run ID."""
    checksum = hashlib.sha256()
    with h5py.File(path, 'r') as f:
        def visit(name, item):
            if not isinstance(item, h5py.Dataset) or name == 'tb_run_id':
                return
            checksum.update(name.encode('utf-8'))
            checksum.update(str((item.shape, item.dtype.str)).encode('ascii'))
            values = item[()]
            if h5py.check_dtype(ref=item.dtype) is not None:
                targets = [f[ref].name if ref else None for ref in values.ravel()]
                checksum.update(json.dumps(targets).encode('utf-8'))
            else:
                checksum.update(np.asarray(values).tobytes())
        f.visititems(visit)
    return checksum.hexdigest()


def checkpoint_position(path: Path, expected_id: str, count: int) -> int:
    with h5py.File(path, 'r') as f:
        actual = ''.join(chr(int(v)) for v in f['tb_run_id'][()].ravel())
        if actual != expected_id:
            raise ValueError('Checkpoint identity differs from parent run')
        phase = float(f['tb_phase'][()].item())
        active = float(f['tb_loop_active'][()].item())
        cursor = float(f['tb_loop_next'][()].item())
        if phase != 15 or active != 15 or not cursor.is_integer() or not 1 <= cursor <= count:
            raise ValueError('Repair requires a safe trajectory-association loop checkpoint')
        return int(cursor)


def rebind_checkpoint(path: Path, old_id: str, new_id: str, count: int) -> dict:
    cursor = checkpoint_position(path, old_id, count)
    before = workspace_digest(path)
    with h5py.File(path, 'r+') as f:
        dataset = f['tb_run_id']
        if dataset.size != len(new_id) or dataset.dtype != np.dtype('uint16'):
            raise ValueError('Unexpected MATLAB run identity representation')
        dataset[...] = np.array([ord(c) for c in new_id], dtype='uint16').reshape(dataset.shape)
    if workspace_digest(path) != before:
        raise ValueError('Scientific workspace changed during identity migration')
    return {'next_iteration': cursor, 'scientific_workspace_sha256': before}


def repair(source: Path, destination: Path) -> Path:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if source in destination.parents:
        raise ValueError('Recovery directory must not be inside parent run')
    with run_lock(source):
        native = source/'native'
        if matlab_pids(native):
            raise RuntimeError('Parent MATLAB is still active')
        identity = json.loads((source/'identity.json').read_text(encoding='utf-8'))
        validate_hashes(native, identity['source_hashes'])
        validate_hashes(source/'inputs', identity['input_hashes'])
        if (native/'completion.json').exists():
            raise ValueError('A completed run must not be repaired')
        script = (native/'bundle_resume.m').read_text(encoding='utf-8')
        fixed = repair_cluster_membership(script)
        # Prove the public column-12 membership and mapping table agree exactly.
        association = loadmat(native/'pts_to_inter.mat')['pts_to_inter']
        mapping = loadmat(native/'inter_to_cluster.mat')['inter_to_cluster']
        np.testing.assert_array_equal(association[:, [2, 3, 11]], mapping)
        vehicles = loadmat(native/'vehicles.mat')['vehicles'].ravel()
        keys, counts = np.unique(mapping[:, :2], axis=0, return_counts=True)
        duplicate_keys = keys[counts > 1]
        checkpoint_names = [name for name in ('checkpoint_current.mat','checkpoint_previous.mat')
                            if (native/name).is_file()]
        if 'checkpoint_current.mat' not in checkpoint_names:
            raise ValueError('A current checkpoint is required')
        positions = {}
        for name in checkpoint_names:
            cursor = checkpoint_position(native/name, identity['run_id'], identity['count'])
            if np.isin(duplicate_keys[:, 0], vehicles[:cursor-1]).any():
                raise ValueError('An ambiguous vehicle was already processed; cannot reuse its state')
            positions[name] = cursor
        intermediates = {name: digest(native/name) for name in INTERMEDIATES}
        validate_hashes(native/'tracebundle_intermediates', intermediates)
        parent_id = identity['run_id']
        provenance = {'parent_run_id': parent_id,
            'parent_identity_sha256': digest(source/'identity.json'),
            'rule': 'Disambiguate vehicle/sample lookup with its recorded pts_to_inter column-12 cluster ID',
            'duplicate_vehicle_sample_keys': duplicate_keys.tolist(),
            'parent_checkpoint_sha256': {name: digest(native/name) for name in checkpoint_names},
            'first_stage_sha256': intermediates}
        # New directory retains the parent logs; parent itself is never edited.
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns('run.lock'))
        child_native = destination/'native'
        (child_native/'bundle_resume.m').write_text(fixed, encoding='utf-8')
        identity.pop('run_id')
        identity['source_hashes'] = file_hashes(child_native)
        identity['mapping_repair'] = provenance
        identity['run_id'] = hashlib.sha256(json.dumps(identity,sort_keys=True).encode('utf-8')).hexdigest()
        for name in checkpoint_names:
            provenance_result = rebind_checkpoint(child_native/name, parent_id,
                                                  identity['run_id'], identity['count'])
            positions[name] = {**provenance_result, 'migrated_sha256': digest(child_native/name)}
        validate_hashes(child_native, intermediates)
        validate_hashes(child_native, identity['source_hashes'])
        (destination/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
        (destination/'mapping_repair.json').write_text(json.dumps({
            **provenance, 'run_id': identity['run_id'], 'checkpoints': positions,
            'parent_relative_path': '../'+source.name,
            'scientific_arrays_unchanged': True},indent=2),encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args = parser.parse_args()
    print(repair(args.source,args.destination))


if __name__ == '__main__':
    main()
