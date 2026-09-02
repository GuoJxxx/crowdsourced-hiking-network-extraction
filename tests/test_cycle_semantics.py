from __future__ import annotations

import sys
from pathlib import Path
import networkx as nx
from shapely.geometry import LineString, Point

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code'
if str(CODE) not in sys.path:
    sys.path.insert(0,str(CODE))

from pipeline_modules.network_cleanup.models import NetworkData,reindex_network
from pipeline_modules.network_cleanup.raw_track_topology import RawTrackChunk,RawTrackWitnessIndex
from pipeline_modules.network_cleanup.cycle_semantics import resolve_single_corridor_cycles


def _net():
    pts=[(-5,0),(0,0),(5,3),(5,-3),(10,0),(15,0)]
    nodes=[{'node_id':i,'x':x,'y':y,'geometry':Point(x,y)} for i,(x,y) in enumerate(pts)]
    specs=[(0,1,[(-5,0),(0,0)]),(1,2,[(0,0),(5,3)]),(2,4,[(5,3),(10,0)]),
           (1,3,[(0,0),(5,-3)]),(3,4,[(5,-3),(10,0)]),(4,5,[(10,0),(15,0)])]
    edges=[]
    for eid,(u,v,c) in enumerate(specs):
        g=LineString(c); edges.append({'edge_id':eid,'u':u,'v':v,'geometry':g,'length_m':g.length,
            'source_type':'primary','support_mean':1.0,'support_ge2_fraction':1.0,'confidence':0.9})
    return reindex_network(NetworkData(nodes,edges,'EPSG:3857',None,0,0,{}))


def _raw(n_each):
    chunks=[]
    for i in range(n_each):
        chunks.append(RawTrackChunk(f't{i}',f'top{i}','s',LineString([(-5,0),(0,0),(5,3+0.1*i),(10,0),(15,0)]),5))
        chunks.append(RawTrackChunk(f'b{i}',f'bot{i}','s',LineString([(-5,0),(0,0),(5,-3-0.1*i),(10,0),(15,0)]),5))
    return RawTrackWitnessIndex(chunks,'EPSG:3857')


def test_thin_duplicate_requires_strong_independent_route_families():
    cleaned,rows=resolve_single_corridor_cycles(
        _net(),_raw(2),raw_corridor_m=1.5,min_route_coverage=0.6,
        min_independent_tracks=1,thin_duplicate_min_independent_tracks=3,max_rounds=3)
    g=nx.Graph((int(e['u']),int(e['v'])) for e in cleaned.edges)
    assert nx.cycle_basis(g)==[]
    assert any(r.get('action')=='collapse_single_corridor_false_cycle' and r.get('thin_duplicate') for r in rows)


def test_thin_dual_route_survives_only_with_three_independent_tracks_per_route():
    cleaned,rows=resolve_single_corridor_cycles(
        _net(),_raw(3),raw_corridor_m=1.5,min_route_coverage=0.6,
        min_independent_tracks=1,thin_duplicate_min_independent_tracks=3,max_rounds=3)
    g=nx.Graph((int(e['u']),int(e['v'])) for e in cleaned.edges)
    assert len(nx.cycle_basis(g))==1
    assert any(r.get('action')=='preserve_independent_dual_route_cycle' for r in rows)


def test_two_gate_cycle_without_local_raw_route_evidence_is_opened_in_v05():
    far = RawTrackWitnessIndex([
        RawTrackChunk('far','far','s',LineString([(1000,1000),(1010,1000)]),2)
    ], 'EPSG:3857')
    cleaned, rows = resolve_single_corridor_cycles(
        _net(), far, raw_corridor_m=1.5, min_route_coverage=0.6,
        min_independent_tracks=1, thin_duplicate_min_independent_tracks=3, max_rounds=3,
    )
    g=nx.Graph((int(e['u']),int(e['v'])) for e in cleaned.edges)
    assert nx.cycle_basis(g)==[]
    assert any(r.get('collapse_reason') in {'no_raw_route_family_support','thin_duplicate_without_raw_route_evidence'} for r in rows)
