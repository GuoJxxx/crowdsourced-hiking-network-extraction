# StudyArea: Edelkamp

1. Open `Start.cmd` in the release root. Check the installation and input locations.
2. Select **StudyArea > Edelkamp** and click **Run selected**.
3. Open `StudyArea/Results/StudyArea_Edelkamp/StudyArea_Edelkamp_Network.gpkg`. The GeoPackage contains `nodes` and `edges` layers.

The method entry is `StudyArea_Edelkamp_run.py`; the fixed paper parameters are listed in `StudyArea_Edelkamp_parameters.json`.
The implementation is in `StudyArea_Edelkamp_Source/`.
The launcher prepares the shared observations, executes reconstruction, and publishes the computed network.

Command-line execution from the release root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Chicago/Evaluation/Runtime/workflow/Configure.ps1 -Task mainling.edelkamp
```

Add `-Action plan` to inspect the commands and inputs before computation. The command uses the Python installation saved in Configuration. The method entry also accepts `--plan` when called directly with that interpreter.
