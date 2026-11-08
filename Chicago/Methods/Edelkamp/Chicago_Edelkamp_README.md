# Chicago: Edelkamp

1. Open `Start.cmd` in the release root. Check the installation and input locations.
2. Select **Chicago > Edelkamp** and click **Run selected**.
3. Open `Chicago/Results/Chicago_Edelkamp/Chicago_Edelkamp_Network.gpkg`. The GeoPackage contains `nodes` and `edges` layers.

The method entry is `Chicago_Edelkamp_run.py`; the fixed paper parameters are listed in `Chicago_Edelkamp_parameters.json`.

The loader processes trajectories in numeric prepared-file order, matching the manuscript's seed initialization. The generated order is saved in `native/track_order.txt`.
The implementation is in `Chicago_Edelkamp_Source/` and `Chicago_Edelkamp_backend.py`.
The launcher prepares the shared observations, executes reconstruction, and publishes the computed network.

Command-line execution from the release root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Chicago/Evaluation/Runtime/workflow/Configure.ps1 -Task chicago.edelkamp
```

Add `-Action plan` to inspect the commands and inputs before computation. The command uses the Python installation saved in Configuration. The method entry also accepts `--plan` when called directly with that interpreter.
