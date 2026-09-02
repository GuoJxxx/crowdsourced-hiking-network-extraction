import sys
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from pipeline_modules import s6_kde_topology_refine as refine
from pipeline_modules.network_cleanup.junction_cleanup import prune_short_isolated_components, prune_short_leaf_chains
from pipeline_modules.network_cleanup.models import NetworkData


def _net(nodes, edges):
    return NetworkData(nodes, edges, None, None, 0, 0, {})


def test_rescue_bridge_between_distinct_primary_components_is_kept():
    base = np.zeros((11, 21), dtype=bool)
    base[5, 1:5] = True
    base[5, 16:20] = True
    rescue = np.zeros_like(base)
    rescue[5, 5:16] = True
    kde = np.zeros_like(base, dtype=np.uint16); kde[5, :] = 255
    support = np.zeros_like(base, dtype=np.uint16); support[5, 5:16] = 3
    kept, info = refine.filter_rescue_by_ridge_and_attachments(
        rescue, base, kde, support, cell_size_m=1.0,
        min_ridge_ratio=0.0, max_primary_attachments=1,
        bridge_min_support=3, bridge_min_support_fraction=0.5,
        bridge_min_ridge_ratio=0.0,
    )
    assert kept[5, 10]
    assert info.get('rescue_distinct_component_bridges_kept', 0) >= 1


def test_short_isolated_component_with_strong_raw_evidence_is_not_pruned():
    nodes = [
        {'node_id':0,'x':0.0,'y':0.0}, {'node_id':1,'x':12.0,'y':0.0},
        {'node_id':2,'x':100.0,'y':0.0}, {'node_id':3,'x':200.0,'y':0.0},
    ]
    edges = [
        {'edge_id':0,'u':0,'v':1,'geometry':LineString([(0,0),(12,0)]),'distinct_tracks':5,'support_mean':4.0,'coverage_ratio':0.8,'confidence':0.9},
        {'edge_id':1,'u':2,'v':3,'geometry':LineString([(100,0),(200,0)]),'distinct_tracks':0,'support_mean':0.0,'coverage_ratio':0.0,'confidence':0.5},
    ]
    out, rows = prune_short_isolated_components(_net(nodes, edges), hard_max_length_m=25.0)
    assert any(int(e['edge_id']) == 0 for e in out.edges)
    assert not any(int(r['edge_id']) == 0 for r in rows)


def test_short_leaf_with_strong_raw_evidence_is_not_pruned():
    nodes = [
        {'node_id':0,'x':0.0,'y':0.0}, {'node_id':1,'x':5.0,'y':0.0},
        {'node_id':2,'x':5.0,'y':5.0}, {'node_id':3,'x':10.0,'y':0.0}, {'node_id':4,'x':5.0,'y':-5.0},
    ]
    edges = [
        {'edge_id':0,'u':0,'v':1,'geometry':LineString([(0,0),(5,0)]),'distinct_tracks':4,'support_mean':5.0,'coverage_ratio':0.9,'confidence':0.9},
        {'edge_id':1,'u':1,'v':2,'geometry':LineString([(5,0),(5,5)]),'distinct_tracks':0,'support_mean':0.0,'coverage_ratio':0.0,'confidence':0.5},
        {'edge_id':2,'u':1,'v':3,'geometry':LineString([(5,0),(10,0)]),'distinct_tracks':0,'support_mean':0.0,'coverage_ratio':0.0,'confidence':0.5},
        {'edge_id':3,'u':1,'v':4,'geometry':LineString([(5,0),(5,-5)]),'distinct_tracks':0,'support_mean':0.0,'coverage_ratio':0.0,'confidence':0.5},
    ]
    out, rows = prune_short_leaf_chains(_net(nodes, edges), hard_max_length_m=6.0, max_rounds=1)
    assert any(int(e['edge_id']) == 0 for e in out.edges)
    assert not any(int(r['edge_id']) == 0 for r in rows)

class _Args:
    dense_rescue_radius_m=2.0
    dense_rescue_support_sum_min=3.0
    dense_rescue_occupied_pixels_min=1.0
    dense_rescue_near_support_min=1
    support_rescue_min_tracks=3
    support_rescue_max_distance_m=20.0
    support_rescue_kde_min_level=1
    dense_rescue_closing_m=0.0
    support_rescue_dilation=0
    dense_rescue_connection_m=2.0
    support_rescue_min_component_pixels=1


def test_dense_support_rescue_is_allowed_inside_activity_zone():
    support=np.zeros((9,9),dtype=np.uint16); support[4,2:7]=3
    distance=np.zeros((9,9),dtype=float)
    scaled=np.zeros((9,9),dtype=np.uint16); scaled[4,2:7]=200
    activity=np.zeros((9,9),dtype=bool); activity[3:6,1:8]=True
    protected=np.zeros_like(activity); existing=np.zeros_like(activity); existing[4,1]=True
    # Call through S6 module because build_dense_support_rescue lives there.
    import S6_route_locked_skeleton as s6
    rescued,dense,info=s6.build_dense_support_rescue(support,distance,scaled,activity,protected,existing,1.0,_Args())
    assert rescued[4,4]
