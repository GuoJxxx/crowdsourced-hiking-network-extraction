from __future__ import annotations

import sys
from pathlib import Path
import networkx as nx
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from pipeline_modules.network_cleanup.models import NetworkData, reindex_network
from pipeline_modules.network_cleanup.raw_track_topology import (
    RawTrackChunk, RawTrackWitnessIndex, repair_high_support_uncovered_corridors,
)


def _net(node_xy, edgespecs):
    nodes=[{'node_id':i,'x':float(x),'y':float(y),'geometry':Point(float(x),float(y))} for i,(x,y) in enumerate(node_xy)]
    edges=[]
    for eid,(u,v,coords) in enumerate(edgespecs):
        g=LineString(coords)
        edges.append({'edge_id':eid,'u':u,'v':v,'geometry':g,'length_m':g.length,
                      'source_type':'primary','confidence':0.9,'support_mean':2.0,'support_ge2_fraction':1.0})
    return reindex_network(NetworkData(nodes,edges,'EPSG:3857',None,0,0,{}))


def test_high_support_coverage_rescue_restores_long_curved_missing_corridor():
    net=_net(
        [(0,0),(8,0),(72,0),(80,0)],
        [(0,1,[(0,0),(8,0)]),(2,3,[(72,0),(80,0)])],
    )
    path1=LineString([(0,0),(8,0),(25,3),(40,5),(55,3),(72,0),(80,0)])
    path2=LineString([(0,0.6),(8,0.5),(25,3.5),(40,5.4),(55,3.4),(72,0.5),(80,0.6)])
    raw=RawTrackWitnessIndex([
        RawTrackChunk('r1','c1','s1',path1,7),
        RawTrackChunk('r2','c2','s2',path2,7),
    ],'EPSG:3857')
    fixed, rows=repair_high_support_uncovered_corridors(
        net, raw,
        max_endpoint_distance_m=90.0,
        max_track_path_m=140.0,
        endpoint_snap_m=2.0,
        max_angle_deg=35.0,
        min_witness_tracks=2,
        max_path_ratio=2.5,
    )
    g=nx.Graph((int(e['u']),int(e['v'])) for e in fixed.edges)
    assert nx.is_connected(g)
    assert len(rows)==1
    assert rows[0]['coverage_audit'] is True
    assert rows[0]['raw_witness_tracks'] >= 2
    assert rows[0]['cycle_closure'] is False


def test_coverage_rescue_never_closes_same_component_cycle():
    net=_net(
        [(0,0),(0,10),(10,10),(10,0)],
        [(0,1,[(0,0),(0,10)]),(1,2,[(0,10),(10,10)]),(2,3,[(10,10),(10,0)])],
    )
    raw=RawTrackWitnessIndex([
        RawTrackChunk('r1','c1','s1',LineString([(0,0),(5,0),(10,0)]),3),
        RawTrackChunk('r2','c2','s2',LineString([(0,0.4),(5,0.4),(10,0.4)]),3),
    ],'EPSG:3857')
    fixed, rows=repair_high_support_uncovered_corridors(
        net, raw,
        max_endpoint_distance_m=20.0,
        max_track_path_m=30.0,
        endpoint_snap_m=1.0,
        max_angle_deg=100.0,
        min_witness_tracks=2,
    )
    assert rows == []
    g=nx.Graph((int(e['u']),int(e['v'])) for e in fixed.edges)
    assert len(nx.cycle_basis(g)) == 0
