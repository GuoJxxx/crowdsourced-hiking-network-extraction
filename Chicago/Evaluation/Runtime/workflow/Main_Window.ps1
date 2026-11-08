function Get-WindowTaskGroups {
    return [ordered]@{
        Chicago=[ordered]@{'chicago.ahmed'='Chicago - Ahmed';'chicago.biagioni'='Chicago - Biagioni';'chicago.davies'='Chicago - Davies';'chicago.edelkamp'='Chicago - Edelkamp';'chicago.karagiorgou'='Chicago - Karagiorgou';'chicago.proposed'='Chicago - Proposed'}
        StudyArea=[ordered]@{'mainling.proposed'='StudyArea - Proposed';'mainling.edelkamp'='StudyArea - Edelkamp';'mainling.biagioni'='StudyArea - Biagioni';'mainling.karagiorgou'='StudyArea - Karagiorgou'}
        Ablation=[ordered]@{'ablation.full'='Full';'ablation.wo_activity'='Without activity awareness';'ablation.wo_uturn'='Without hairpin preservation';'ablation.wo_evidence_topology'='Without topology optimization'}
        Figures=[ordered]@{'figure.8'='Figure 8 - Chicago networks';'figure.9'='Figure 9 - StudyArea local comparisons';'figure.10'='Figure 10 - StudyArea map layers';'figure.11'='Figure 11 - Ablation structural responses';'figure.12'='Figure 12 - Ablation map layers'}
        Tables=[ordered]@{'table.1'='Table 1';'table.2'='Table 2';'table.3'='Table 3';'table.a1'='Table A1'}
        Metrics=[ordered]@{'metrics.chicago'='Chicago accuracy and graph measures';'metrics.mainling'='StudyArea graph measures';'metrics.ablation'='Ablation graph measures';'metrics.all'='All reported numerical measures'}
    }
}

function Get-WindowRunState {
    param($State,[bool]$FreshWindow)
    if (-not $FreshWindow) { return $State }
    $resumable=($State.action -eq 'run' -and $State.targets -and $State.state -ne 'completed')
    if ($resumable) {
        return [pscustomobject]@{state='interrupted';current='Ready. Resume is available for the previous local run.';completed=0;total=0;action='run';targets=$State.targets}
    }
    return [pscustomobject]@{state='ready';current='Select a task to begin.';completed=0;total=0}
}

function New-RunCloseDialog {
    $dialog=New-Object Windows.Forms.Form
    $dialog.Text='Close Trail Network Reconstruction';$dialog.ClientSize=New-Object Drawing.Size(580,165)
    $dialog.StartPosition='CenterParent';$dialog.FormBorderStyle='FixedDialog';$dialog.MaximizeBox=$false;$dialog.MinimizeBox=$false
    $label=New-Object Windows.Forms.Label;$label.Location=New-Object Drawing.Point(18,18);$label.Size=New-Object Drawing.Size(544,75)
    $label.Text="A calculation is running or paused. Choose what happens when this window closes.`r`n`r`nStopping retains saved checkpoints; work since the last checkpoint may be repeated."
    $dialog.Controls.Add($label)
    $x=18
    foreach ($option in @(@('Stop calculation','Yes',155),@('Keep running in background','No',220),@('Cancel','Cancel',145))) {
        $button=New-Object Windows.Forms.Button;$button.Text=$option[0];$button.DialogResult=[Windows.Forms.DialogResult]$option[1]
        $button.Location=New-Object Drawing.Point($x,112);$button.Size=New-Object Drawing.Size($option[2],32)
        $dialog.Controls.Add($button);$x+=$option[2]+8
        if ($option[1] -eq 'Cancel') { $dialog.CancelButton=$button;$dialog.AcceptButton=$button }
    }
    return $dialog
}

function Show-RunCloseDialog {
    param($Owner)
    $dialog=New-RunCloseDialog
    try { return $dialog.ShowDialog($Owner) } finally { $dialog.Dispose() }
}

function Show-ReproductionWindow {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [Windows.Forms.Application]::EnableVisualStyles()
    $ui=@{Cfg=(Read-Configuration $script:Root);Selections=@{};SelectedFields=@{};DataFields=@{};Editable=(New-Object Collections.ArrayList);Dirty=$false;LogFile='';LogPosition=0;NativeLogFile='';NativeLogPosition=0}
    $initial=Read-RunState $script:Root $ui.Cfg
    $ui.InitialJob='';if (-not (Test-ActiveRun $script:Root $ui.Cfg)) { $ui.InitialJob=[string]$initial.job }
    $form=New-Object Windows.Forms.Form
    $form.Text='Trail Network Reconstruction'
    $form.ClientSize=New-Object Drawing.Size(1120,760)
    $form.MinimumSize=New-Object Drawing.Size(940,640)
    $form.StartPosition='CenterScreen';$form.Font=New-Object Drawing.Font('Segoe UI',9)
    $form.BackColor=[Drawing.Color]::FromArgb(244,246,248)
    $title=New-Object Windows.Forms.Label;$title.Text='Trail Network Reconstruction'
    $title.Font=New-Object Drawing.Font('Segoe UI',21,[Drawing.FontStyle]::Bold)
    $title.ForeColor=[Drawing.Color]::FromArgb(23,50,77);$title.Padding=New-Object Windows.Forms.Padding(18,14,0,0)
    $title.Dock='Top';$title.Height=65
    $toolbar=New-Object Windows.Forms.FlowLayoutPanel;$toolbar.Dock='Top';$toolbar.Height=46;$toolbar.Padding=New-Object Windows.Forms.Padding(18,3,0,3)
    $tabs=New-Object Windows.Forms.TabControl;$tabs.Dock='Fill';$tabs.Padding=New-Object Drawing.Point(12,5)
    $pages=@{}
    foreach ($name in @('Reproduction','Configuration','Parameters','Instructions')) {
        $page=New-Object Windows.Forms.TabPage;$page.Text=$name;$page.Padding=New-Object Windows.Forms.Padding(10)
        $tabs.TabPages.Add($page);$pages[$name]=$page
    }
    $footer=New-Object Windows.Forms.Panel;$footer.Dock='Bottom';$footer.Height=250;$footer.Padding=New-Object Windows.Forms.Padding(18,5,18,10)
    $summary=New-Object Windows.Forms.Label;$summary.Dock='Top';$summary.Height=48;$summary.Text='Ready';$summary.Font=New-Object Drawing.Font('Segoe UI',10,[Drawing.FontStyle]::Bold)
    $runtimeNotice=New-Object Windows.Forms.Label;$runtimeNotice.Dock='Top';$runtimeNotice.Height=32;$runtimeNotice.Visible=$false
    $runtimeNotice.ForeColor=[Drawing.Color]::FromArgb(23,50,77)
    $progressCaption=New-Object Windows.Forms.Label;$progressCaption.Dock='Top';$progressCaption.Height=23;$progressCaption.Text='Task completion: 0%'
    $progress=New-Object Windows.Forms.ProgressBar;$progress.Dock='Top';$progress.Height=16;$progress.Maximum=100
    $log=New-Object Windows.Forms.TextBox;$log.Multiline=$true;$log.ReadOnly=$true;$log.ScrollBars='Vertical';$log.Dock='Fill'
    $log.Font=New-Object Drawing.Font('Consolas',9);$log.BackColor=[Drawing.Color]::FromArgb(23,39,56);$log.ForeColor=[Drawing.Color]::FromArgb(224,231,238)
    $footer.Controls.Add($log);$footer.Controls.Add($progress);$footer.Controls.Add($progressCaption);$footer.Controls.Add($runtimeNotice);$footer.Controls.Add($summary)
    $form.Controls.Add($tabs);$form.Controls.Add($footer);$form.Controls.Add($toolbar);$form.Controls.Add($title)
    $tip=New-Object Windows.Forms.ToolTip
    $buttons=@{}
    foreach ($definition in @(@('Run selected','selected',122),@('Run all findings','all',145),@('Pause','pause',80),@('Resume','resume',85),@('Configuration','configure',135),@('Open results','results',125))) {
        $button=New-Object Windows.Forms.Button;$button.Text=$definition[0];$button.Tag=$definition[1]
        $button.Width=$definition[2];$button.Height=31;$toolbar.Controls.Add($button);$buttons[$definition[1]]=$button
    }
    $split=New-Object Windows.Forms.SplitContainer;$split.Dock='Fill';$split.Size=New-Object Drawing.Size(1050,440);$split.SplitterDistance=350
    $tree=New-Object Windows.Forms.TreeView;$tree.Dock='Fill';$tree.HideSelection=$false;$tree.ItemHeight=25
    $plan=New-Object Windows.Forms.TextBox;$plan.Multiline=$true;$plan.ReadOnly=$true;$plan.ScrollBars='Vertical';$plan.Dock='Fill'
    $plan.Font=New-Object Drawing.Font('Segoe UI',10);$plan.BackColor=[Drawing.Color]::White
    $plan.Text="Select an experiment, table or figure.`r`n`r`nUse Configuration to select software and data locations."
    $split.Panel1.Controls.Add($tree);$split.Panel2.Controls.Add($plan);$pages.Reproduction.Controls.Add($split)
    $groups=Get-WindowTaskGroups
    foreach ($group in $groups.Keys) {
        $node=$tree.Nodes.Add($group);$node.Tag=[string[]]@($groups[$group].Keys)
        foreach ($key in $groups[$group].Keys) { $child=$node.Nodes.Add($groups[$group][$key]);$child.Tag=[string[]]@($key) }
        if ($group -in @('Chicago','StudyArea','Ablation')) { $node.Expand() }
    }
    $tree.Add_AfterSelect({
        try {
            if ($ui.Dirty) { $plan.Text='Save the configuration to view the run plan.';return }
            $python=Get-SelectedPython $script:Root (Read-Configuration $script:Root)
            $arguments=@('-B',(Join-Path $script:Root 'Run_Workflow.py'),'plan')+@($tree.SelectedNode.Tag)
            $output=& $python @arguments 2>&1
            if ($LASTEXITCODE -ne 0) { throw ($output -join "`n") }
            $tasks=($output -join "`n") | ConvertFrom-Json
            $lines=@();$index=0
            foreach ($task in $tasks) { $index++;$lines+=('{0:00}  {1}' -f $index,$task.label) }
            $plan.Text=$lines -join "`r`n"
        } catch { $plan.Text=(Get-EnglishText $_.Exception.Message) }
    })
    $intro=New-Object Windows.Forms.Label;$intro.Dock='Top';$intro.Height=36
    $intro.Text='Select installed software and data locations.'
    $scroll=New-Object Windows.Forms.Panel;$scroll.Dock='Fill';$scroll.AutoScroll=$true
    $scroll.AutoScrollMinSize=New-Object Drawing.Size(1020,560)
    $actions=New-Object Windows.Forms.FlowLayoutPanel;$actions.Dock='Bottom';$actions.Height=42
    $configInfo=New-Object Windows.Forms.Label;$configInfo.Dock='Bottom';$configInfo.Height=27;$configInfo.Text='Python 3.11+ | JDK 17+ | MATLAB R2021a/R2021b with Statistics and Machine Learning Toolbox'
    $pages.Configuration.Controls.Add($scroll);$pages.Configuration.Controls.Add($configInfo);$pages.Configuration.Controls.Add($actions);$pages.Configuration.Controls.Add($intro)
    $software=New-Object Windows.Forms.GroupBox;$software.Text='Software';$software.SetBounds(0,3,1010,178);$scroll.Controls.Add($software)
    $label=New-Object Windows.Forms.Label;$label.Text='Selected installation';$label.SetBounds(201,19,700,20);$software.Controls.Add($label)
    $rows=@(@('python','Python executable'),@('matlab','MATLAB executable'),@('jdk','JDK directory'))
    for ($index=0;$index -lt $rows.Count;$index++) {
        $name=$rows[$index][0];$y=43+41*$index
        $label=New-Object Windows.Forms.Label;$label.Text=$rows[$index][1];$label.SetBounds(14,$y+4,185,25);$software.Controls.Add($label)
        $display=Get-SoftwareDisplay $script:Root $ui.Cfg $ui.Selections $name
        $selected=New-Object Windows.Forms.TextBox;$selected.ReadOnly=$true;$selected.Text=$display.Selected;$selected.SetBounds(201,$y,700,26);$software.Controls.Add($selected)
        $ui.SelectedFields[$name]=$selected;$tip.SetToolTip($selected,$display.Selected)
        $browse=New-Object Windows.Forms.Button;$browse.Text='Browse';$browse.Tag=$name;$browse.SetBounds(913,$y-1,82,29);$software.Controls.Add($browse);$ui.Editable.Add($browse) | Out-Null
        $browse.Add_Click({
            try {
                $key=$this.Tag
                if ($key -eq 'jdk') {
                    $picker=New-Object Windows.Forms.FolderBrowserDialog;$picker.Description='Select the JDK installation directory.'
                    if ($picker.ShowDialog($form) -ne 'OK') { return };$location=$picker.SelectedPath
                } else {
                    $picker=New-Object Windows.Forms.OpenFileDialog;$picker.Title='Select '+$key+'.exe';$picker.Filter='Executable files (*.exe)|*.exe';$picker.CheckFileExists=$true
                    if ($picker.ShowDialog($form) -ne 'OK') { return };$location=$picker.FileName
                }
                Set-SoftwareSelection $ui.Cfg $ui.Selections $key $location
                $display=Get-SoftwareDisplay $script:Root $ui.Cfg $ui.Selections $key
                $ui.SelectedFields[$key].Text=$display.Selected
                $tip.SetToolTip($ui.SelectedFields[$key],$display.Selected)
                $ui.Dirty=$true;$configInfo.Text='Selection changed. Choose Save configuration to apply it.'
                if ($key -eq 'python') { & $showTarget }
                $picker.Dispose()
            } catch { [Windows.Forms.MessageBox]::Show($form,(Get-EnglishText $_.Exception.Message),'Configuration') | Out-Null }
        })
    }
    $packageTarget=New-Object Windows.Forms.Label;$packageTarget.SetBounds(14,185,990,27);$scroll.Controls.Add($packageTarget)
    $showTarget={
        $display=Get-SoftwareDisplay $script:Root $ui.Cfg $ui.Selections 'python'
        if ($display.Selected) {
            $target=& $display.Selected -I -B -c "import json,sysconfig;print(json.dumps(sysconfig.get_path('purelib')))" 2>&1
            if ($LASTEXITCODE -eq 0) { $packageTarget.Text='Package destination: '+(($target -join '') | ConvertFrom-Json) }
            else { $packageTarget.Text='Check the selected Python installation.' }
        } else { $packageTarget.Text='Select Python to display its package destination.' }
    }
    & $showTarget
    $data=New-Object Windows.Forms.GroupBox;$data.Text='Data and results';$data.SetBounds(0,219,1010,337);$scroll.Controls.Add($data)
    $labels=@{chicago_tracks='Chicago trajectories';chicago_map='Chicago reference map';mainling_observed='StudyArea trajectories';reference_manifest='StudyArea trajectory manifest';boundary='StudyArea boundary';dem='StudyArea DEM';chicago_results='Chicago results';studyarea_results='StudyArea results'}
    $folders=@('mainling_observed','chicago_results','studyarea_results');$row=0
    foreach ($property in $ui.Cfg.paths.PSObject.Properties) {
        $name=$property.Name;$y=24+37*$row;$row++
        $label=New-Object Windows.Forms.Label;$label.Text=$labels[$name];$label.SetBounds(14,$y+4,185,25);$data.Controls.Add($label)
        $entry=New-Object Windows.Forms.TextBox;$entry.Text=$property.Value;$entry.SetBounds(201,$y,700,26);$data.Controls.Add($entry);$ui.DataFields[$name]=$entry;$ui.Editable.Add($entry) | Out-Null
        $tip.SetToolTip($entry,(Get-ConfiguredLocation $script:Root $property.Value))
        $entry.Add_TextChanged({$ui.Dirty=$true;$configInfo.Text='Locations changed. Choose Save configuration to apply them.'})
        $browse=New-Object Windows.Forms.Button;$browse.Text='Browse';$browse.Tag=$name;$browse.SetBounds(913,$y-1,82,29);$data.Controls.Add($browse);$ui.Editable.Add($browse) | Out-Null
        $browse.Add_Click({
            try {
                $key=$this.Tag
                if ($folders -contains $key) {
                    $picker=New-Object Windows.Forms.FolderBrowserDialog;$picker.Description=$labels[$key]
                    if ($picker.ShowDialog($form) -ne 'OK') { return };$location=$picker.SelectedPath
                } else {
                    $picker=New-Object Windows.Forms.OpenFileDialog;$picker.Title=$labels[$key];$picker.CheckFileExists=$true
                    if ($picker.ShowDialog($form) -ne 'OK') { return };$location=$picker.FileName
                }
                $value=Get-RelativeLocation $script:Root $location
                if ($null -eq $value) {
                    if ($key.EndsWith('_results')) { throw 'Select a results folder inside this package.' }
                    $value=$script:BindingBase+'/Locations/'+$key+'.lnk'
                    $ui.Selections[$key]=@{Relative=$value;Target=$location}
                } else { $ui.Selections.Remove($key) }
                $ui.DataFields[$key].Text=$value;$tip.SetToolTip($ui.DataFields[$key],$location)
                $ui.Dirty=$true;$configInfo.Text='Selected '+$labels[$key]+': '+$location
                $picker.Dispose()
            } catch { [Windows.Forms.MessageBox]::Show($form,(Get-EnglishText $_.Exception.Message),'Configuration') | Out-Null }
        })
    }
    $save={
        foreach ($property in $ui.Cfg.paths.PSObject.Properties) { $ui.Cfg.paths.($property.Name)=$ui.DataFields[$property.Name].Text.Trim() }
        Save-Configuration $script:Root $ui.Cfg $ui.Selections
        $ui.Selections.Clear();$ui.Dirty=$false
        $configInfo.Text='Configuration saved. The selected installations will be used.'
    }
    foreach ($definition in @(@('Save configuration','save',166),@('Install Python packages','install',185),@('Check installation','check',159),@('Stop installation','stop',153))) {
        $button=New-Object Windows.Forms.Button;$button.Text=$definition[0];$button.Tag=$definition[1];$button.Width=$definition[2];$button.Height=30;$actions.Controls.Add($button);$buttons[$definition[1]]=$button
        if ($definition[1] -ne 'stop') { $ui.Editable.Add($button) | Out-Null }
        $button.Add_Click({
            try {
                $action=$this.Tag
                if ($action -eq 'stop') { Stop-Installation $script:Root (Read-Configuration $script:Root);return }
                & $save
                if ($action -ne 'save') { Start-WorkflowTask $script:Root $ui.Cfg @() $action }
            } catch { [Windows.Forms.MessageBox]::Show($form,(Get-EnglishText $_.Exception.Message),'Configuration') | Out-Null }
        })
    }
    $buttons.configure.Add_Click({$tabs.SelectedTab=$pages.Configuration})
    $buttons.results.Add_Click({
        try {
            $cfg=Read-Configuration $script:Root
            foreach ($name in @('chicago_results','studyarea_results')) {
                $folder=Get-ConfiguredLocation $script:Root $cfg.paths.$name
                New-Item -ItemType Directory -Path $folder -Force | Out-Null
                Start-Process -FilePath $folder
            }
        } catch { [Windows.Forms.MessageBox]::Show($form,(Get-EnglishText $_.Exception.Message),'Results') | Out-Null }
    })
    $buttons.selected.Add_Click({
        try {
            if (-not $tree.SelectedNode) { throw 'Select an experiment, table, figure or metric.' }
            if ($ui.Dirty) { & $save }
            Start-WorkflowTask $script:Root (Read-Configuration $script:Root) @($tree.SelectedNode.Tag)
        } catch { [Windows.Forms.MessageBox]::Show($form,(Get-EnglishText $_.Exception.Message),'Run status') | Out-Null }
    })
    $buttons.all.Add_Click({
        try {
            if ($ui.Dirty) { & $save }
            Start-WorkflowTask $script:Root (Read-Configuration $script:Root) @('all')
        } catch { [Windows.Forms.MessageBox]::Show($form,(Get-EnglishText $_.Exception.Message),'Run status') | Out-Null }
    })
    foreach ($controlName in @('pause','resume')) {
        $buttons[$controlName].Add_Click({
            try {
                $this.Enabled=$false
                Set-WorkflowControl $script:Root (Read-Configuration $script:Root) ([string]$this.Tag)
            } catch { [Windows.Forms.MessageBox]::Show($form,(Get-EnglishText $_.Exception.Message),'Run status') | Out-Null }
        })
    }
    $tip.SetToolTip($buttons.pause,'Pause the current experiment and retain its in-memory state.')
    $tip.SetToolTip($buttons.resume,'Continue the paused run, or resume the previous selection from saved stages.')
    $parameterChoices=New-Object Windows.Forms.ComboBox;$parameterChoices.Dock='Top';$parameterChoices.DropDownStyle='DropDownList'
    $parameterText=New-Object Windows.Forms.TextBox;$parameterText.Dock='Fill';$parameterText.Multiline=$true;$parameterText.ReadOnly=$true;$parameterText.ScrollBars='Both';$parameterText.WordWrap=$false;$parameterText.Font=New-Object Drawing.Font('Consolas',9)
    $parameterFiles=[ordered]@{'StudyArea Proposed'='StudyArea/Methods/Proposed/StudyArea_Proposed_parameters.json';'Ablation'='StudyArea/Ablation/StudyArea_Proposed_Ablation_parameters.json';'Reference methods'='Chicago/Evaluation/Chicago_Reference_parameters.json';'Table A1'='StudyArea/Evaluation/StudyArea_TableA1_parameters.json';'Scenes'='StudyArea/Evaluation/StudyArea_Scenes.json'}
    foreach ($name in $parameterFiles.Keys) { $parameterChoices.Items.Add($name) | Out-Null }
    $parameterChoices.Add_SelectedIndexChanged({$parameterText.Text=[IO.File]::ReadAllText((Join-Path $script:Root $parameterFiles[$parameterChoices.SelectedItem]))})
    $pages.Parameters.Controls.Add($parameterText);$pages.Parameters.Controls.Add($parameterChoices);$parameterChoices.SelectedIndex=0
    $instructions=New-Object Windows.Forms.Label;$instructions.Dock='Top';$instructions.Height=310;$instructions.Font=New-Object Drawing.Font('Segoe UI',10)
    $instructions.Text="1. Open Configuration and select Python, MATLAB and the JDK directory.`r`n2. Review the selected installations and package destination.`r`n3. Save configuration and choose Check installation. The final ready value must be true.`r`n4. Install Python packages when dependencies are missing. Stop installation cancels only that task.`r`n5. In Reproduction, select a method or finding, review its plan and choose Run selected.`r`n6. Open results to access the computed networks, measures, tables and figures.`r`n`r`nPause retains the current calculation in memory. Resume continues the same run. After a computer restart, Resume reuses completed stages on disk.`r`n`r`nWhen closing an active run, choose Stop calculation, Keep running in background, or Cancel. Reopen Start.cmd to view a local background run. Another computer starts with an idle window."
    $guide=New-Object Windows.Forms.Button;$guide.Text='Open reproduction guide';$guide.Width=205;$guide.Height=32;$guide.Dock='Bottom'
    $guide.Add_Click({Start-Process -FilePath (Join-Path $script:Root 'REPRODUCE.md')})
    $pages.Instructions.Controls.Add($instructions);$pages.Instructions.Controls.Add($guide)
    $timer=New-Object Windows.Forms.Timer;$timer.Interval=1000
    $refresh={
        try {
            $cfg=Read-Configuration $script:Root;$state=Read-RunState $script:Root $cfg;$running=Test-ActiveRun $script:Root $cfg
            $state=Get-WindowRunState $state (-not $running -and [string]$state.job -eq $ui.InitialJob)
            $display=[string]$state.state
            if ($display -in @('running','starting','pausing','paused','resuming') -and -not $running) { $display='interrupted' }
            $summary.Text=$display.ToUpper()+'   '+(Get-EnglishText ([string]$state.current))+'   '+[int]$state.completed+' / '+[int]$state.total+' tasks'
            if ($state.stage) { $summary.Text+="`r`n"+(Get-EnglishText ([string]$state.stage)) }
            $runtimeNotice.Text=Get-EnglishText ([string]$state.runtime_note);$runtimeNotice.Visible=-not [string]::IsNullOrWhiteSpace($runtimeNotice.Text)
            if ($state.progress) {
                $progress.Value=[Math]::Max(0,[Math]::Min(100,[int]$state.progress.percent))
                $progressCaption.Text='Stage progress: '+[string]$state.progress.phase+' - '+$progress.Value+'% ('+[string]$state.progress.completed+' / '+[string]$state.progress.total+' items)'
            } else {
                $progress.Value=[Math]::Max(0,[Math]::Min(100,[int][Math]::Floor(100*[double]$state.completed/[Math]::Max(1,[int]$state.total))))
                $progressCaption.Text='Task completion: '+$progress.Value+'%'
            }
            $buttons.selected.Enabled=-not $running;$buttons.all.Enabled=-not $running
            $buttons.pause.Enabled=($running -and $state.action -eq 'run' -and $state.state -in @('running','starting'))
            $buttons.resume.Enabled=($state.action -eq 'run' -and $state.targets -and ($display -in @('paused','failed','interrupted','stopped')))
            foreach ($control in $ui.Editable) { $control.Enabled=-not $running }
            $buttons.stop.Enabled=($running -and $state.action -eq 'install')
            if ($state.log) {
                $filename=Get-ConfiguredLocation $script:Root $state.log
                if ($filename -ne $ui.LogFile) { $ui.LogFile=$filename;$ui.LogPosition=0;$log.Clear() }
                if (Test-Path -LiteralPath $filename) {
                    $stream=Open-SharedRead $filename
                    try {
                        if ($stream.Length -lt $ui.LogPosition) { $ui.LogPosition=0 }
                        $stream.Position=$ui.LogPosition;$length=[int][Math]::Min(65536,$stream.Length-$ui.LogPosition)
                        if ($length -gt 0) {
                            $buffer=New-Object byte[] $length;$count=$stream.Read($buffer,0,$length);$ui.LogPosition+=$count
                            $log.AppendText((Get-EnglishText ([Text.Encoding]::UTF8.GetString($buffer,0,$count))))
                            if ($log.TextLength -gt 180000) { $log.Text=$log.Text.Substring($log.TextLength-120000) }
                        }
                    } finally { $stream.Dispose() }
                }
            }
            if ($state.native_log) {
                $nativeFile=[string]$state.native_log
                if ($nativeFile -ne $ui.NativeLogFile) {
                    $ui.NativeLogFile=$nativeFile;$ui.NativeLogPosition=0
                    $log.AppendText("`r`n[RUNTIME] Following live MATLAB progress.`r`n")
                }
                $chunk=Read-LogChunk -Filename $nativeFile -Position $ui.NativeLogPosition -Tail
                $ui.NativeLogPosition=$chunk.Position
                if ($chunk.Text) {
                    $log.AppendText((Get-EnglishText $chunk.Text))
                    if ($log.TextLength -gt 180000) { $log.Text=$log.Text.Substring($log.TextLength-120000) }
                }
            } else { $ui.NativeLogFile='';$ui.NativeLogPosition=0 }
        } catch { $summary.Text=(Get-EnglishText $_.Exception.Message) }
    }
    $timer.Add_Tick($refresh)
    $form.Add_FormClosing({
        param($sender,$event)
        try {
            $cfg=Read-Configuration $script:Root
            if (Test-ActiveRun $script:Root $cfg) {
                $choice=Show-RunCloseDialog $form
                if ($choice -eq [Windows.Forms.DialogResult]::Cancel) { $event.Cancel=$true;return }
                if ($choice -eq [Windows.Forms.DialogResult]::Yes) {
                    $active=Read-RunState $script:Root $cfg
                    if ($active.action -eq 'install') { Stop-Installation $script:Root $cfg }
                    else { Set-WorkflowControl $script:Root $cfg 'stop' }
                }
            }
        } catch {
            $event.Cancel=$true
            [Windows.Forms.MessageBox]::Show($form,(Get-EnglishText $_.Exception.Message),'Close calculation') | Out-Null
        }
    })
    $form.Add_FormClosed({$timer.Stop();$timer.Dispose()})
    $form.Add_Shown({ & $refresh;$timer.Start() })
    if (-not $ui.Cfg.tools.python) { $tabs.SelectedTab=$pages.Configuration }
    $form.ShowDialog() | Out-Null
    $form.Dispose()
}
