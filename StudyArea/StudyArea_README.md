# Study-area experiments

The four methods are Edelkamp, Biagioni, Karagiorgou and Proposed.

1. Open the root `Start.cmd` and install/check the environment.
2. Retain the observations, boundary, DEM and reference input manifest under `Data`.
3. Select **StudyArea** for the four reconstructed networks. Each reference method first regenerates the common cleaned observations with the Proposed S2 cleaning stage. The subsequent reference methods use that same 1,543-segment collection.
4. Select **Ablation** for Full, Without activity awareness, Without hairpin preservation and Without topology optimization.
5. Run `StudyArea_RunAll.cmd` to compute the study-area methods, ablations, Tables 2-3/A1 and Figures 9-12.
6. Open `Results/StudyArea_<Method>/StudyArea_<Method>_Network.gpkg`. Ablations are under `Results/StudyArea_Proposed_Ablation` with one named folder per profile.
7. Tables, metrics and figure outputs are under `Results/StudyArea_TablesAndFigures`.

`Ablation/StudyArea_Proposed_Ablation_RunAll.py` runs all four ablation profiles. Individual profile entry names are Full, WithoutActivity, WithoutHairpin and WithoutTopology.

The observed files are already clipped to the shared study boundary. S2 performs the reproducible cleaning, coordinate conversion and segmentation used in the paper. The DEM retains its native cropped grid; the modeled networks are exported in EPSG:32647.
