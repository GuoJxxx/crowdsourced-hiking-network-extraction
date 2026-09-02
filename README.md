# Crowdsourced Hiking Network Extraction

**Activity-Aware Pipeline — Smoothed Network Extraction**

A Python workflow for extracting hiking road/trail networks from crowdsourced GPS trajectories, with explicit handling of activity-area noise, U-turn/hairpin geometry, directional evidence, topology repair, and final network smoothing.

> This repository contains the research code for crowdsourced hiking-network extraction, including activity-aware filtering, U-turn/hairpin protection, directional evidence modeling, topology optimization, and final network smoothing.

## Method overview

The workflow contains seven main stages:

1. **S1 — KML trajectory extraction**  
   Extract observed `gx:Track` trajectories from KML and save one CSV per independent trajectory segment.
2. **S2 — Trajectory cleaning**  
   Clip to the study area, remove spatial/temporal outliers, filter non-hiking movement, repair elevation anomalies, and preserve switchback geometry.
3. **S3 — Activity-aware filtering**  
   Identify activity-like trajectory concentrations and retain coherent road/trail evidence using direction, efficiency, endpoint, speed, spatial elongation, and multi-track support information.
4. **S4 — U-turn-aware trajectory centralization**  
   Apply multi-trajectory consensus gravity centralization while protecting U-turn/hairpin legs from cross-wing attraction.
5. **S5 — Direction-constrained KDE**  
   Build directional trajectory evidence rasters for subsequent road extraction.
6. **S6 — Road skeleton extraction**  
   Extract evidence-supported road skeletons with route locking and multi-threshold refinement.
7. **S7 — Evidence-guided topology optimization and smoothing**  
   Repair supported gaps, suppress implausible loops/parallel artifacts, optimize topology, and smooth complete road chains while protecting junctions and U-turn geometry.

The overall map-inference workflow was **primarily inspired by Biagioni and Eriksson (2012)**, particularly the use of trajectory-density evidence, skeleton/centerline extraction, and trajectory-supported topology refinement. This repository is **not a direct reimplementation** of that method: the present pipeline introduces activity-aware evidence weighting, U-turn/switchback semantics, constrained multi-trajectory centralization, directional KDE, evidence-guided topology repair, and final geometry smoothing for crowdsourced hiking trajectories in mountainous canyon environments.

The principal final network is written by S7 under:

```text
08_final_clean_network/road_network_final_clean.*
```

## Repository structure

```text
.
├── code/                       # S1-S7 scripts and internal modules
│   ├── S1_extract_kml.py
│   ├── S2_clean_tracks.py
│   ├── S3_activity_zone_filter.py
│   ├── S4_uturn_consensus_gravity.py
│   ├── S5_directional_kde.py
│   ├── S6_route_locked_skeleton.py
│   ├── S7_network_cleanup.py
│   ├── pipeline_modules/
│   └── tools/
├── configs/                    # Default pipeline configuration
├── data/                       # Local input data (not committed by default)
├── outputs/                    # Pipeline outputs (not committed by default)
├── docs/                       # Parameter reference and running guide
├── tests/                      # Regression and topology tests
├── requirements.txt
├── run_S1_to_S7_windows.cmd
└── run_S2_to_S7_windows.cmd
```

## Installation

Python 3.10+ is recommended.

Create and activate a clean environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

For geospatial packages on Windows, a conda-forge environment may be more convenient:

```bash
conda create -n hiking-network -c conda-forge python=3.11 numpy pandas scipy pyproj geopandas shapely rasterio scikit-image pillow pyogrio networkx pytest
conda activate hiking-network
```

## Input data

The repository intentionally does **not** include raw trajectory data, study-area boundaries, DEM data, or generated GIS outputs.

Typical local layout:

```text
data/
├── raw_kml/                    # Original KML files containing gx:Track
├── observed_tracks/            # Optional existing S1 trajectory CSVs
├── boundary/
│   └── study_area.shp          # Optional study-area polygon
└── dem.tif                     # Optional DEM
```

You may use any local paths. The recommended approach is to set them in `configs/default_config.json` or select them in the Windows configuration launcher.

## Running the pipeline

### Windows graphical/configurable launcher

For a complete S1-S7 run:

```text
run_S1_to_S7_windows.cmd
```

If S1 trajectory extraction has already been completed:

```text
run_S2_to_S7_windows.cmd
```

The launcher opens the configuration interface implemented in:

```text
code/tools/run_pipeline_configurable.ps1
```

### Run individual stages

Each stage can also be called directly with command-line arguments. For example:

```bash
python code/S1_extract_kml.py data/raw_kml -o outputs/S1_TracksFromKML
```

```bash
python code/S2_clean_tracks.py \
  --input outputs/S1_TracksFromKML/observed_tracks \
  --boundary data/boundary/study_area.shp \
  --output outputs/S2_Clean
```

Use `-h` or `--help` on an individual stage to inspect its available parameters.

## Configuration

The central configuration file is:

```text
configs/default_config.json
```

All path entries are empty or repository-relative in the public release. Set the paths to match your own data environment before running the complete workflow.

Detailed parameter descriptions are available in:

- `docs/PARAMETER_REFERENCE.md`
- `docs/PARAMETER_REFERENCE.csv`

## Network smoothing behavior

The final S7 smoothing stage follows these principles:

- A full degree-2 chain between real junctions/endpoints is treated as one continuous road segment.
- Intermediate degree-2 nodes may move during smoothing; true junctions and endpoints remain fixed.
- Gap-repair edges participate in chain smoothing to avoid artificial corners at repaired gaps.
- Ordinary roads use strong smoothing constrained by a validated road-evidence corridor.
- U-turn/hairpin regions use a narrower protection corridor to avoid straightening the bend.
- Strong Douglas-Peucker simplification is disabled by default for the final smoothed network.

## Tests

Run the regression suite from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover connectivity, U-turn semantics, cycle handling, raw-track evidence, junction cleanup, smoothing behavior, runtime safety, and smoothing behavior.

## Reproducibility notes

Results depend on trajectory density, sampling characteristics, projected CRS, study-area geometry, and parameter settings. Preserve the configuration used for each experiment. The configurable runner supports saving the effective run configuration to the output directory.

## Methodological basis and references

The following publications provide methodological foundations for specific parts of this implementation. Biagioni and Eriksson (2012) is the **primary reference for the overall map-inference pipeline concept** used in this project; the other references correspond to particular algorithmic components.

1. **Biagioni, J., & Eriksson, J. (2012).** Map inference in the face of noise and disparity. *Proceedings of the 20th International Conference on Advances in Geographic Information Systems (SIGSPATIAL '12)*, 79–88. https://doi.org/10.1145/2424321.2424333  
   Primary methodological reference for trajectory-based map inference, including density-based representation, centerline/skeleton extraction, and trajectory-supported topology/geometric refinement. The present implementation substantially extends this framework for mountainous hiking trajectories.

2. **Newson, P., & Krumm, J. (2009).** Hidden Markov map matching through noise and sparseness. *Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, 336–343. https://doi.org/10.1145/1653771.1653818  
   Methodological basis for the HMM/Viterbi map-matching structure used in the S7 trajectory-evidence matcher. The implementation in this repository adds heading, evidence-source, and network-transition constraints and should therefore be regarded as an adapted implementation rather than a reproduction of the original algorithm.

3. **Guo, Z., & Hall, R. W. (1989).** Parallel thinning with two-subiteration algorithms. *Communications of the ACM*, 32(3), 359–373. https://doi.org/10.1145/62065.62074  
   Algorithmic basis of the binary thinning operation used through `skimage.morphology.thin` during skeleton extraction/refinement.

## Citation

If this code supports a publication, please cite the associated paper after its bibliographic information becomes available. A suggested software citation format is:

```text
Authors. (2026). Crowdsourced Hiking Network Extraction [Computer software]. GitHub.
```

Replace `Authors` and add the repository URL / DOI after the public repository and archival release are created.

## License

This project is released under the MIT License. See `LICENSE`.

---

## 中文说明

本仓库为众源徒步轨迹路网构建方法代码，包含 **S1–S7** 七个主要步骤：轨迹提取、轨迹清洗、活动区感知过滤、U 型弯语义约束的多轨迹引力集中、方向约束 KDE、道路骨架提取，以及证据驱动的拓扑优化与最终路网平滑。

整体 map-inference 流程主要参考 **Biagioni and Eriksson (2012)**，尤其是轨迹证据/密度表达、骨架中心线提取以及轨迹证据支持的拓扑细化思想。在此基础上，本项目针对高山峡谷区众源徒步轨迹进一步加入活动区感知、U 型弯/连续回弯语义保护、多轨迹约束集中、方向约束 KDE、证据驱动拓扑修补与最终路网平滑等机制。

代码已采用仓库相对路径组织默认输入输出。原始轨迹、研究区边界、DEM 和运行结果默认均不纳入 Git 版本控制。运行前请在 `configs/default_config.json` 或 Windows 配置界面中指定你自己的数据路径。

更详细的中文运行说明见 `docs/RUNNING_GUIDE_ZH.md`。
