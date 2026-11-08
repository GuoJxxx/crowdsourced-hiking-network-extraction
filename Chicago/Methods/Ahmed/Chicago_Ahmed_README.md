# Chicago: Ahmed

1. Open `Start.cmd` in the release root. Check the installation and input locations.
2. Select **Chicago > Ahmed** and click **Run selected**.
3. Open `Chicago/Results/Chicago_Ahmed/Chicago_Ahmed_Network.gpkg`. The GeoPackage contains `nodes` and `edges` layers.

The method entry is `Chicago_Ahmed_run.py`; the fixed paper parameters are listed in `Chicago_Ahmed_parameters.json`.
The implementation is in `Chicago_Ahmed_Source/` and `Chicago_Ahmed_backend.py`.
The launcher prepares the shared observations, executes reconstruction, and publishes the computed network.

Trajectories are supplied in numeric prepared-file order (`trip_0.txt`, `trip_1.txt`, `trip_2.txt`, ...), matching the manuscript run. The Java entry adapter passes this ordered list to the original graph-construction functions. The generated order is saved in `native/track_order.txt`.

Command-line execution from the release root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Chicago/Evaluation/Runtime/workflow/Configure.ps1 -Task chicago.ahmed
```

Add `-Action plan` to inspect the commands and inputs before computation. The command uses the Python installation saved in Configuration. The method entry also accepts `--plan` when called directly with that interpreter.
