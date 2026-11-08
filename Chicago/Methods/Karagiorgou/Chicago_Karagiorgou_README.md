# Chicago: Karagiorgou

1. Open `Start.cmd` in the release root. Check the installation and input locations.
2. Select **Chicago > Karagiorgou** and click **Run selected**.
3. Open `Chicago/Results/Chicago_Karagiorgou/Chicago_Karagiorgou_Network.gpkg`. The GeoPackage contains `nodes` and `edges` layers.

The method entry is `Chicago_Karagiorgou_run.py`; the fixed paper parameters are listed in `Chicago_Karagiorgou_parameters.json`.

Selecting this method also prepares the shared trajectory inputs. MATLAB stage messages appear in the window as they are printed and are saved in `native/matlab.log`. **Check installation** records the complete MATLAB version, including its update level.
The MATLAB stages run with `-batch` and hidden figures. The JVM remains enabled for the reference source's graphics calls.
The implementation is in `Chicago_Karagiorgou_Source/` and `Chicago_Karagiorgou_backend.py`.
The launcher prepares the shared observations, executes reconstruction, and publishes the computed network.

Command-line execution from the release root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Chicago/Evaluation/Runtime/workflow/Configure.ps1 -Task chicago.karagiorgou
```

Add `-Action plan` to inspect the commands and inputs before computation. The command uses the Python installation saved in Configuration. The method entry also accepts `--plan` when called directly with that interpreter.

## Runtime

Karagiorgou runs the reference MATLAB implementation with its original sequential loops. Runtime depends on the number of trajectories, trajectory points and candidate intersections, and on CPU performance. A full-data run can take several hours or longer. Progress messages are printed at major stage boundaries and saved in `native/matlab.log`.

The window displays **Stage progress** with the current phase, processed item count and percentage from 0 to 100. Each phase starts a new counter. Trajectory reading, turn analysis, intersection clustering, link compaction, merging passes and network export report their own progress. The progress bar remains at its current value during **Pause** and continues on **Resume**.

The runtime notice reads: **Karagiorgou follows the reference source code. Processing can take several hours or longer. Please wait.**
