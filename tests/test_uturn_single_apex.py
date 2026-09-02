from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from pipeline_modules import uturn_consensus as uc


def _event(curve, cid, *, activity=0.0):
    curve = uc.resample_curve(np.asarray(curve, dtype=float), 101)
    return {
        'track_id': cid,
        'original_contribution_id': cid,
        'curve': curve,
        'curve_fixed': curve,
        'score': 0.95,
        'apex_x': float(curve[len(curve)//2,0]),
        'apex_y': float(curve[len(curve)//2,1]),
        'path_efficiency': 0.25,
        'length_m': float(uc.cumulative_distance(curve)[-1]),
        'activity_zone_fraction': float(activity),
        'activity_through_corridor_fraction': 0.0,
    }


def test_single_apex_hairpin_passes_strict_structure():
    base = [(0,14),(0,10),(0,6),(0,2),(1,-1),(3,-3),(5,-1),(6,2),(6,6),(6,10),(6,14)]
    events = [_event(np.asarray(base)+np.array([dx,0.0]), f'h{i}') for i,dx in enumerate([-0.3,-0.1,0.1,0.3])]
    cfg = uc.UTurnConfig(
        min_distinct_original_tracks=3,
        min_leg_straightness=0.88,
        min_leg_longitudinal_overlap_ratio=0.55,
        min_apex_turn_concentration=0.58,
        max_secondary_turn_peak_ratio=0.60,
    )
    consensus = uc.build_consensus(events, cfg)['consensus']
    r = uc.validate_uturn_structure(events, consensus, cfg)
    assert r['structure_valid'] is True
    assert r['start_leg_straightness'] >= 0.88
    assert r['end_leg_straightness'] >= 0.88
    assert r['leg_longitudinal_overlap_ratio'] >= 0.55
    assert r['secondary_turn_peak_ratio'] <= 0.60


def test_distributed_round_turn_is_rejected_not_protected_as_uturn():
    # A broad rounded arc reverses direction, but the turn is distributed rather
    # than concentrated at one apex and its legs are not straight/overlapping.
    theta = np.linspace(np.pi, 0.0, 40)
    curve = np.c_[18*np.cos(theta), 14*np.sin(theta)]
    events = [_event(curve + np.array([0.0,dy]), f'r{i}', activity=0.8) for i,dy in enumerate([-0.4,0,0.4,0.8])]
    cfg = uc.UTurnConfig(
        min_distinct_original_tracks=3,
        min_leg_straightness=0.88,
        min_leg_longitudinal_overlap_ratio=0.50,
        min_apex_turn_concentration=0.58,
        max_secondary_turn_peak_ratio=0.55,
    )
    consensus = uc.build_consensus(events, cfg)['consensus']
    r = uc.validate_uturn_structure(events, consensus, cfg)
    assert r['structure_valid'] is False
    assert r['reject_reason'] in {
        'leg_not_straight', 'insufficient_leg_overlap', 'turn_not_single_apex',
        'turn_not_apex_concentrated', 'activity_zone_structure_too_weak',
        'legs_not_parallel', 'legs_not_opposite'
    }


def test_activity_zone_requires_stronger_single_apex_evidence():
    # Geometrically hairpin-like but with a broad two-corner bottom; ordinary mode
    # may be permissive, activity-zone mode must reject it.
    base = [(0,14),(0,8),(0,2),(2,-2),(4,-2),(7,1),(7,7),(7,14)]
    events = [_event(np.asarray(base)+np.array([dx,0.0]), f'a{i}', activity=0.9) for i,dx in enumerate([-0.3,-0.1,0.1,0.3])]
    cfg = uc.UTurnConfig(
        min_distinct_original_tracks=3,
        min_leg_straightness=0.82,
        min_leg_longitudinal_overlap_ratio=0.45,
        min_apex_turn_concentration=0.45,
        max_secondary_turn_peak_ratio=0.80,
        activity_min_distinct_original_tracks=4,
        activity_min_apex_turn_concentration=0.72,
        activity_max_secondary_turn_peak_ratio=0.45,
    )
    consensus = uc.build_consensus(events, cfg)['consensus']
    r = uc.validate_uturn_structure(events, consensus, cfg)
    assert r['structure_valid'] is False
    assert r['reject_reason'] == 'activity_zone_structure_too_weak'


def test_synthetic_activity_backbone_does_not_generate_uturn_event_support():
    rows=[]
    pts=[(0,12),(0,6),(0,1),(2,-2),(5,0),(6,6),(6,12)]
    for i,(x,y) in enumerate(pts,1):
        rows.append({
            'clean_track_id':'activity_backbone_z00001_b00001',
            'source_clean_track_id':'activity_backbone_z00001_b00001',
            'clean_point_index':i,'x_m':x,'y_m':y,
            'activity_backbone_synthetic':True,'activity_zone_id':1,
        })
    df=pd.DataFrame(rows)
    events=uc.detect_events(df, uc.UTurnConfig())
    assert events == []
