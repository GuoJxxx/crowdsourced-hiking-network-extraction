# Chicago experiments

The six methods are Ahmed, Biagioni, Davies, Edelkamp, Karagiorgou and Proposed.

1. Open the root `Start.cmd` and install/check the environment.
2. Retain `Data/Chicago_Trajectories.zip` and `Data/Chicago_ReferenceMap.zip` in this folder. The trajectories contain 889 trips and 118,360 samples in EPSG:26916.
3. Select an individual method in the **Chicago** group, or run `Chicago_RunAll.cmd` for all six methods, Table 1 and Figure 8.
4. The runner extracts the original trip observations, prepares the method input format, executes reconstruction, converts the network to GeoPackage, rebuilds the trajectory coverage raster, and evaluates all ten spatial tolerances.
5. Open `Results/Chicago_<Method>/Chicago_<Method>_Network.gpkg` and `Chicago_<Method>_Evaluation.json`.
6. Read `Results/Chicago_TablesAndFigures/Chicago_Table1.csv`, `Chicago_AccuracyAndStructure.csv`, `Chicago_ToleranceMetrics.csv` and `Chicago_Figure8.png`.

Per-method entry/config names follow `Methods/<Method>/Chicago_<Method>_run.py` and `Chicago_<Method>_parameters.json`. Parameters used in native code are documented beside those entries.

Ahmed uses Java/JDK 17 or newer. Karagiorgou uses MATLAB R2021a or R2021b with Statistics and Machine Learning Toolbox. The root requirements file defines compatible dependency ranges.
