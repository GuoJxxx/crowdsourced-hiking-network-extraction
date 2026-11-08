# Trail network reconstruction

Data, code and instructions for reproducing the experiments in **Activity-aware and hairpin-preserving trail network reconstruction from crowdsourced hiking trajectories in complex mountainous terrain**.

The package covers six Chicago methods, four study-area methods and four study-area ablation configurations. [REPRODUCE.md](REPRODUCE.md) gives the source-to-output procedure for **Tables 1-3 and A1, Figures 8-12, and the associated numerical measures**. These are the table numbers in the revised manuscript.

## 1. Software requirements

Use 64-bit Windows with the following software:

| Software | Requirement | Use |
| --- | --- | --- |
| Python | 3.11 or newer; packages in `requirements.txt` | Preparation, reconstruction, evaluation and export |
| Java JDK | 17 or newer, including `java` and `javac` | Ahmed |
| MATLAB | R2021a or R2021b with Statistics and Machine Learning Toolbox | Karagiorgou |

Check installation reports the complete software versions and Windows compatibility. Python 3.11-3.12 use `lib2to3` for the reference Python sources; Python 3.13 and newer use `fissix` as specified in `requirements.txt`.

## 2. Configure and run

1. Extract the complete ZIP to a writable folder. Open the extracted `TrailNetwork` folder and double-click **Start.cmd**. The folder can be renamed or moved.
2. Open **Configuration**. Use **Browse** to select the Python executable, MATLAB executable and JDK directory on this computer. The JDK selection supplies both Java and Javac.
3. Keep the supplied data locations and the default output folders, or select locations through **Browse**. Choose **Save configuration**.
4. Choose **Check installation**. Confirm `ready: true` in the log. If packages are missing, choose **Install Python packages**, then check again. Packages are installed into the selected Python environment; **Stop installation** cancels that installation.
5. Open **Reproduction**, select a method, table or figure, and choose **Run selected**. The task planner includes its preparation, reconstruction and assessment dependencies. **Run all findings** executes the complete set of manuscript tasks.
6. Choose **Open results** when the task finishes. Method networks contain `nodes` and `edges` layers in GeoPackage format. Tables, metrics and figure products are written to the dataset's `TablesAndFigures` folder.

The same window provides **Reproduction**, **Configuration**, **Parameters** and **Instructions** tabs. The **Parameters** tab displays the supplied experiment settings. Configuration selects installed software and input/output locations; source-code editing is unnecessary.

All supplied data and result entries in `settings.json` are relative to the package root. The configuration window creates local links for installations or data on another drive. Select installed software again when transferring the package to another computer. Default results remain inside the current package folder.

The [illustrated window guide](REPRODUCE.md#1-configure-and-run-with-the-illustrated-window-guide) uses eight current-interface screenshots to explain software selection, relative data/output locations, installation checking, parameter viewing, task planning, actual task completion and the Instructions page.

## 3. Experiments and outputs

| Dataset or experiment | Methods or configurations | Instructions | Default results |
| --- | --- | --- | --- |
| Chicago | Ahmed, Biagioni, Davies, Edelkamp, Karagiorgou, Proposed | [Chicago_README.md](Chicago/Chicago_README.md) | `Chicago/Results` |
| Study area | Edelkamp, Biagioni, Karagiorgou, Proposed | [StudyArea_README.md](StudyArea/StudyArea_README.md) | `StudyArea/Results` |
| Study-area ablation | Full, WithoutActivity, WithoutHairpin, WithoutTopology | [REPRODUCE.md](REPRODUCE.md#5-table-3--ablation-network-structural-measures) | `StudyArea/Results/StudyArea_Proposed_Ablation` |

For each method, `Methods/<Method>/` contains its run entry, parameter file, README and implementation. Final networks follow `<Dataset>/Results/<Dataset>_<Method>/<Dataset>_<Method>_Network.gpkg`. Chicago methods also export an evaluation JSON.

| Manuscript item | Window task ID | Main product |
| --- | --- | --- |
| Table 1 | `table.1` | `Chicago_Table1.csv` |
| Table 2 | `table.2` | `StudyArea_Table2.csv` |
| Table 3 | `table.3` | `StudyArea_Table3.csv` |
| Table A1 | `table.a1` | `StudyArea_TableA1.csv` |
| Figure 8 | `figure.8` | `Chicago_Figure8.png`, PDF and SVG |
| Figure 9 | `figure.9` | `StudyArea_Figure9.png`, PDF and SVG |
| Figure 10 | `figure.10` | Reconstructed network and layer-location JSON for the GIS overlay |
| Figure 11 | `figure.11` | `StudyArea_Figure11.png`, PDF and SVG |
| Figure 12 | `figure.12` | Four ablation networks, semantic layers and scene definitions for GIS composition |
| Reported numerical measures | `metrics.all` | `Chicago_ReportedMetrics.json` and `StudyArea_ReportedMetrics.json` |

Products are under `Chicago/Results/Chicago_TablesAndFigures` or `StudyArea/Results/StudyArea_TablesAndFigures`. Figure 10 and Figure 12 tasks export computed map layers for the author-composed GIS layouts. Figure 10's OSM comparison layer is a separate cartographic input. Each finding's inputs, full computational workflow, task and output paths are specified in [REPRODUCE.md](REPRODUCE.md).

## 4. Data and computational workflow

**Chicago.** `Chicago/Data/Chicago_Trajectories.zip` and `Chicago_ReferenceMap.zip` supply the benchmark observations and independent reference map. The pipeline extracts the archives, prepares each method's coordinate and text formats, constructs the six networks and converts them to EPSG:26916 GeoPackages. A separate assessment branch regenerates the manuscript's common trajectory-coverage raster and calculates precision, trajectory-covered recall, F1 and graph measures. Reference-method construction and common assessment are separate branches.

**Study area.** `StudyArea/Data` supplies anonymized trajectory observations cropped to the study boundary, boundary sidecars, the cropped DEM and reference input manifest. S2 regenerates the shared cleaned hiking segments. The three reference methods construct networks from those segments; Proposed applies activity awareness, hairpin preservation, directional density estimation, skeleton extraction and topology refinement. Final networks use EPSG:32647. The four ablation profiles remove one mechanism at a time while retaining the other experiment settings.

**Tables and figures.** Tasks depend on reconstructed networks and calculated measures. Selecting a table or figure executes these dependencies from the published inputs, then writes its table, plot or GIS layers. The code-navigation table in `REPRODUCE.md` identifies the stage files and equation annotations.

## 5. Run controls and progress

- **Pause** retains the current process and its memory. **Resume** continues that process.
- After a computer restart, **Resume** restores the saved task selection, reuses completed stages with matching inputs and settings, and restarts the unfinished stage.
- Closing an active window offers **Stop calculation**, **Keep running in background** and **Cancel**. Completed checkpoints remain on disk.
- **Stage progress** shows the current phase's processed items and percentage. Each phase has its own counter.

Karagiorgou follows the reference implementation's sequential loops. Full-data runs can take several hours or longer. The window displays its stage messages and progress as the calculation proceeds. Logs and software diagnostics use English.

## 6. Command line

Configure the package in the window first. The following PowerShell commands use the saved Python selection and the same planner:

```powershell
# Inspect the complete task plan.
powershell -NoProfile -ExecutionPolicy Bypass -File Chicago/Evaluation/Runtime/workflow/Configure.ps1 -Task all -Action plan

# Reproduce one manuscript item.
powershell -NoProfile -ExecutionPolicy Bypass -File Chicago/Evaluation/Runtime/workflow/Configure.ps1 -Task table.1

# Reproduce all manuscript tasks.
powershell -NoProfile -ExecutionPolicy Bypass -File Chicago/Evaluation/Runtime/workflow/Configure.ps1 -Task all
```

Run these commands from the extracted package root. The worker prints its log location; the window displays the same run status. Individual method commands and complete output locations are listed in the dataset and method READMEs.

## 7. Package contents

`Chicago` and `StudyArea` hold the dataset-specific inputs, methods, evaluation tools and instructions. `requirements.txt` specifies package compatibility ranges, and `RELEASE_MANIFEST.json` records payload checksums. Third-party licenses remain with their source directories.

Project repository: <https://github.com/GuoJxxx/crowdsourced-hiking-network-extraction>.
