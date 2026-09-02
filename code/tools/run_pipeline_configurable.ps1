param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('S2S7','S1S7')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$script:PackageRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$script:CodeDir = Join-Path $script:PackageRoot 'code'
$script:SchemaPath = Join-Path $script:PackageRoot 'configs\parameter_schema.json'
$script:DefaultConfigPath = Join-Path $script:PackageRoot 'configs\default_config.json'
$script:LastConfigPath = Join-Path $script:PackageRoot 'configs\last_config.json'
$script:Controls = @{}
$script:RunRequested = $false
$script:CurrentConfig = $null

function Read-JsonFile([string]$PathValue) {
    if (-not (Test-Path -LiteralPath $PathValue -PathType Leaf)) { throw "配置文件不存在: $PathValue" }
    return (Get-Content -LiteralPath $PathValue -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Write-JsonFile($Object, [string]$PathValue) {
    $dir = Split-Path -Parent $PathValue
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $json = $Object | ConvertTo-Json -Depth 30
    [System.IO.File]::WriteAllText($PathValue, $json, (New-Object System.Text.UTF8Encoding($true)))
}

function Find-Python {
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { $cmd = Get-Command python -ErrorAction SilentlyContinue }
    if ($null -eq $cmd) { throw '未找到 Python。请先在 CMD 中确认 python --version 可以运行。' }
    return $cmd.Source
}

function Show-Message([string]$Text, [string]$Title='Activity-Aware Pipeline', [System.Windows.Forms.MessageBoxIcon]$Icon=[System.Windows.Forms.MessageBoxIcon]::Information) {
    [System.Windows.Forms.MessageBox]::Show($Text,$Title,[System.Windows.Forms.MessageBoxButtons]::OK,$Icon) | Out-Null
}

function Get-PropValue($Obj, [string]$Name) {
    if ($null -eq $Obj) { return $null }
    $p = $Obj.PSObject.Properties[$Name]
    if ($null -eq $p) { return $null }
    return $p.Value
}

function Set-PropValue($Obj, [string]$Name, $Value) {
    $p = $Obj.PSObject.Properties[$Name]
    if ($null -eq $p) { $Obj | Add-Member -NotePropertyName $Name -NotePropertyValue $Value }
    else { $p.Value = $Value }
}

function Select-Folder([string]$Description, [string]$CurrentPath='') {
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = $Description
    $dialog.ShowNewFolderButton = $true
    if (-not [string]::IsNullOrWhiteSpace($CurrentPath)) {
        if (Test-Path -LiteralPath $CurrentPath -PathType Container) {
            $dialog.SelectedPath = $CurrentPath
        } else {
            $parent = Split-Path -Parent $CurrentPath
            if ($parent -and (Test-Path -LiteralPath $parent -PathType Container)) {
                $dialog.SelectedPath = $parent
            }
        }
    }
    $result = if ($null -ne $script:MainForm) { $dialog.ShowDialog($script:MainForm) } else { $dialog.ShowDialog() }
    if ($result -eq [System.Windows.Forms.DialogResult]::OK) { return $dialog.SelectedPath }
    return $null
}

function Select-File([string]$Title, [string]$Filter, [string]$CurrentPath='') {
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = $Title
    $dialog.Filter = $Filter
    $dialog.CheckFileExists = $true
    $dialog.CheckPathExists = $true
    if (-not [string]::IsNullOrWhiteSpace($CurrentPath)) {
        if (Test-Path -LiteralPath $CurrentPath -PathType Leaf) {
            $dialog.InitialDirectory = Split-Path -Parent $CurrentPath
            $dialog.FileName = Split-Path -Leaf $CurrentPath
        } elseif (Test-Path -LiteralPath $CurrentPath -PathType Container) {
            $dialog.InitialDirectory = $CurrentPath
        } else {
            $parent = Split-Path -Parent $CurrentPath
            if ($parent -and (Test-Path -LiteralPath $parent -PathType Container)) {
                $dialog.InitialDirectory = $parent
            }
        }
    }
    $result = if ($null -ne $script:MainForm) { $dialog.ShowDialog($script:MainForm) } else { $dialog.ShowDialog() }
    if ($result -eq [System.Windows.Forms.DialogResult]::OK) { return $dialog.FileName }
    return $null
}

function New-Label([string]$Text,[int]$Width=220) {
    $l = New-Object System.Windows.Forms.Label
    $l.Text = $Text
    $l.Width = $Width
    $l.Height = 24
    $l.TextAlign = [System.Drawing.ContentAlignment]::MiddleLeft
    return $l
}

function New-PathRow([string]$Key,[string]$Label,[string]$Kind,[string]$Help) {
    $row = New-Object System.Windows.Forms.TableLayoutPanel
    $row.Height = 38
    $row.Dock = [System.Windows.Forms.DockStyle]::Top
    $row.ColumnCount = 4
    $row.RowCount = 1
    $row.Padding = New-Object System.Windows.Forms.Padding(4,3,4,3)
    $row.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Absolute,220))) | Out-Null
    $row.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent,100))) | Out-Null
    $row.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Absolute,92))) | Out-Null
    $row.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Absolute,62))) | Out-Null

    $lab = New-Object System.Windows.Forms.Label
    $lab.Text = $Label
    $lab.Dock = [System.Windows.Forms.DockStyle]::Fill
    $lab.TextAlign = [System.Drawing.ContentAlignment]::MiddleLeft
    $row.Controls.Add($lab,0,0)
    $script:Controls['pathrow.'+$Key] = $row
    $script:Controls['pathlabel.'+$Key] = $lab

    $tb = New-Object System.Windows.Forms.TextBox
    $tb.Dock = [System.Windows.Forms.DockStyle]::Fill
    $tb.Margin = New-Object System.Windows.Forms.Padding(3,4,3,3)
    $row.Controls.Add($tb,1,0)
    $script:Controls['paths.'+$Key] = $tb

    if ($Kind -eq 'folder' -or $Kind -eq 'file') {
        $btn = New-Object System.Windows.Forms.Button
        $btn.Text = '浏览...'
        $btn.Dock = [System.Windows.Forms.DockStyle]::Fill
        $btn.Margin = New-Object System.Windows.Forms.Padding(3,1,3,1)
        $btn.Tag = [PSCustomObject]@{ Key=$Key; Kind=$Kind; Label=$Label }
        $btn.Add_Click({
            try {
                $info = $this.Tag
                $current = $script:Controls['paths.'+$info.Key].Text
                $picked = $null
                if ($info.Kind -eq 'folder') {
                    $picked = Select-Folder ('请选择：'+$info.Label) $current
                } else {
                    $filter = if ($info.Key -eq 'boundary') {
                        'Shapefile (*.shp)|*.shp|All files (*.*)|*.*'
                    } elseif ($info.Key -eq 'dem') {
                        'GeoTIFF (*.tif;*.tiff)|*.tif;*.tiff|All files (*.*)|*.*'
                    } else {
                        'All files (*.*)|*.*'
                    }
                    $picked = Select-File ('请选择：'+$info.Label) $filter $current
                }
                if ($picked) {
                    $script:Controls['paths.'+$info.Key].Text = $picked
                }
            } catch {
                Show-Message $_.Exception.Message '路径选择失败' ([System.Windows.Forms.MessageBoxIcon]::Error)
            }
        })
        $row.Controls.Add($btn,2,0)
        $script:Controls['pathbrowse.'+$Key] = $btn

        $clear = New-Object System.Windows.Forms.Button
        $clear.Text = '清空'
        $clear.Dock = [System.Windows.Forms.DockStyle]::Fill
        $clear.Margin = New-Object System.Windows.Forms.Padding(3,1,3,1)
        $clear.Tag = $Key
        $clear.Add_Click({
            $script:Controls['paths.'+[string]$this.Tag].Text = ''
        })
        $row.Controls.Add($clear,3,0)
        $script:Controls['pathclear.'+$Key] = $clear
    } else {
        $spacer = New-Object System.Windows.Forms.Label
        $spacer.Text = ''
        $row.Controls.Add($spacer,2,0)
        $spacer2 = New-Object System.Windows.Forms.Label
        $spacer2.Text = ''
        $row.Controls.Add($spacer2,3,0)
    }

    if (-not [string]::IsNullOrWhiteSpace($Help)) {
        $tip = New-Object System.Windows.Forms.ToolTip
        $tip.SetToolTip($lab,$Help)
        $tip.SetToolTip($tb,$Help)
    }
    return $row
}

function Add-PathRows($Parent) {
    $rows = @(
        @('s1_raw_kml','S1 原始 KML 根目录','folder','S1→S7 模式使用；S2→S7 模式可留空。'),
        @('s1_output','S1 输出根目录','folder','S1→S7 模式使用，S1 将在这里创建 observed_tracks。'),
        @('s1_observed_tracks','S1 observed_tracks','folder','S2→S7 模式必填，应包含 *__gxtrack.csv。'),
        @('boundary','研究区边界 .shp','file','S2 使用；若 S2 勾选 --no-boundary 可留空。'),
        @('dem','DEM .tif/.tiff','file','可选。S2 高程质控与 S7 DEM 属性使用。'),
        @('output_root','S2-S7 输出根目录','folder','所有阶段目录均在该目录下创建。'),
        @('s2_dir','S2 子目录名','text','默认 S2_Clean，可修改。'),
        @('s3_dir','S3 子目录名','text','默认 S3_Activity，可修改。'),
        @('s4_dir','S4 子目录名','text','默认 S4_Gravity，可修改。'),
        @('s5_dir','S5 子目录名','text','默认 S5_KDE，可修改。'),
        @('s6_dir','S6 子目录名','text','默认 S6_Skeleton，可修改。'),
        @('s7_dir','S7 子目录名','text','默认 S7_Network，可修改。')
    )
    # Dock=Top reverses insertion order, so add bottom-to-top.
    for ($i=$rows.Count-1; $i -ge 0; $i--) {
        $r=$rows[$i]
        $Parent.Controls.Add((New-PathRow $r[0] $r[1] $r[2] $r[3]))
    }
}


function Set-PathEnabled([string]$Key,[bool]$Enabled,[string]$Reason='') {
    $tb=$script:Controls['paths.'+$Key]
    $lab=$script:Controls['pathlabel.'+$Key]
    $browse=$script:Controls['pathbrowse.'+$Key]
    $clear=$script:Controls['pathclear.'+$Key]
    if($null -ne $tb){
        $tb.Enabled=$Enabled
        $tb.BackColor=if($Enabled){[System.Drawing.SystemColors]::Window}else{[System.Drawing.SystemColors]::Control}
    }
    if($null -ne $browse){$browse.Enabled=$Enabled}
    if($null -ne $clear){$clear.Enabled=$Enabled}
    if($null -ne $lab){
        $lab.ForeColor=if($Enabled){[System.Drawing.SystemColors]::ControlText}else{[System.Drawing.SystemColors]::GrayText}
        if(-not [string]::IsNullOrWhiteSpace($Reason)){
            $tip=New-Object System.Windows.Forms.ToolTip
            $tip.SetToolTip($lab,$Reason)
            if($null -ne $tb){$tip.SetToolTip($tb,$Reason)}
        }
    }
}

function Apply-ModeAwareUI() {
    if($Mode -eq 'S2S7'){
        Set-PathEnabled 's1_raw_kml' $false 'S2→S7 模式不运行 S1，因此无需设置原始 KML 路径。'
        Set-PathEnabled 's1_output' $false 'S2→S7 模式不运行 S1，因此无需设置 S1 输出目录。'
        Set-PathEnabled 's1_observed_tracks' $true
    } else {
        Set-PathEnabled 's1_raw_kml' $true
        Set-PathEnabled 's1_output' $true
        Set-PathEnabled 's1_observed_tracks' $false 'S1→S7 模式会自动使用“S1 输出根目录\observed_tracks”，无需单独填写。'
    }
}

function Param-Key([string]$Stage,[string]$Dest) { return ($Stage+'.'+$Dest) }

function Format-ParamValue($Meta,$Value) {
    if ($Meta.kind -eq 'bool_flag' -or $Meta.kind -eq 'bool_optional') {
        if ([bool]$Value) { return 'true' } else { return 'false' }
    }
    if ($null -eq $Value) { return '' }
    if ($Value -is [double] -or $Value -is [single] -or $Value -is [decimal]) {
        return ([Convert]::ToString($Value,[System.Globalization.CultureInfo]::InvariantCulture))
    }
    return [string]$Value
}

function Parse-BoolText([string]$Text,[string]$Label) {
    $t=$Text.Trim().ToLowerInvariant()
    if ($t -in @('true','1','yes','y','on','是')) { return $true }
    if ($t -in @('false','0','no','n','off','否')) { return $false }
    throw ($Label+' 必须填写 true/false（也可用 1/0、yes/no）。当前值: '+$Text)
}

function Parse-ParamText($Meta,[string]$Text,[string]$Stage) {
    $label=$Stage+' '+[string]$Meta.option
    if ($Meta.kind -eq 'bool_flag' -or $Meta.kind -eq 'bool_optional') { return Parse-BoolText $Text $label }
    $text2=$Text.Trim()
    if ($text2 -eq '') { return $null }
    if ($Meta.kind -eq 'choice') {
        $allowed=@($Meta.choices | ForEach-Object {[string]$_})
        if ($allowed -notcontains $text2) { throw ($label+' 只能是: '+($allowed -join ', ')+'。当前值: '+$text2) }
        return $text2
    }
    if ($Meta.value_type -eq 'int') {
        $iv=0L
        if (-not [Int64]::TryParse($text2,[System.Globalization.NumberStyles]::Integer,[System.Globalization.CultureInfo]::InvariantCulture,[ref]$iv)) { throw ($label+' 必须是整数: '+$text2) }
        return $iv
    }
    if ($Meta.value_type -eq 'float') {
        $dv=0.0
        if (-not [Double]::TryParse($text2,[System.Globalization.NumberStyles]::Float,[System.Globalization.CultureInfo]::InvariantCulture,[ref]$dv)) { throw ($label+' 必须是数值: '+$text2) }
        return $dv
    }
    return $text2
}

function New-ParameterGrid($Schema,$Cfg) {
    $grid=New-Object System.Windows.Forms.DataGridView
    $grid.Dock=[System.Windows.Forms.DockStyle]::Fill
    $grid.AllowUserToAddRows=$false
    $grid.AllowUserToDeleteRows=$false
    $grid.AllowUserToResizeRows=$false
    $grid.RowHeadersVisible=$false
    $grid.AutoGenerateColumns=$false
    $grid.SelectionMode=[System.Windows.Forms.DataGridViewSelectionMode]::CellSelect
    $grid.MultiSelect=$false
    $grid.EditMode=[System.Windows.Forms.DataGridViewEditMode]::EditOnEnter
    $grid.BackgroundColor=[System.Drawing.SystemColors]::Window
    $grid.AutoSizeRowsMode=[System.Windows.Forms.DataGridViewAutoSizeRowsMode]::None
    $grid.RowTemplate.Height=25

    $cols=@(
        @('Stage','阶段',58,$true),
        @('Option','参数',250,$true),
        @('Value','当前值（可编辑）',210,$false),
        @('Type','类型',75,$true),
        @('Default','默认值',130,$true),
        @('Help','说明 / 可选值',620,$true)
    )
    foreach($c in $cols){
        $col=New-Object System.Windows.Forms.DataGridViewTextBoxColumn
        $col.Name=$c[0];$col.HeaderText=$c[1];$col.Width=[int]$c[2];$col.ReadOnly=[bool]$c[3]
        if($c[0] -eq 'Help'){$col.AutoSizeMode=[System.Windows.Forms.DataGridViewAutoSizeColumnMode]::Fill;$col.MinimumWidth=300}
        [void]$grid.Columns.Add($col)
    }

    foreach($stageProp in $Schema.stages.PSObject.Properties){
        $stage=$stageProp.Name
        if($Mode -eq 'S2S7' -and $stage -eq 'S1'){ continue }
        $stageCfg=Get-PropValue $Cfg.stages $stage
        foreach($p in $stageProp.Value.parameters){
            $v=Get-PropValue $stageCfg $p.dest
            $kind=if($p.kind -eq 'bool_flag' -or $p.kind -eq 'bool_optional'){'bool'}elseif($p.kind -eq 'choice'){'choice'}else{[string]$p.value_type}
            $def=Format-ParamValue $p $p.default
            $help=[string]$p.help
            if($p.kind -eq 'choice'){$help=($help+'  可选: '+(@($p.choices) -join ', ')).Trim()}
            $idx=$grid.Rows.Add($stage,[string]$p.option,(Format-ParamValue $p $v),$kind,$def,$help)
            $row=$grid.Rows[$idx]
            $row.Tag=[PSCustomObject]@{Stage=$stage;Dest=[string]$p.dest;Meta=$p}
        }
    }
    return $grid
}

function Apply-ConfigToGrid($Grid,$Cfg) {
    foreach($row in $Grid.Rows){
        $tag=$row.Tag
        if($null -eq $tag){continue}
        $stageCfg=Get-PropValue $Cfg.stages $tag.Stage
        $v=Get-PropValue $stageCfg $tag.Dest
        $row.Cells['Value'].Value=Format-ParamValue $tag.Meta $v
    }
}

function Collect-GridToConfig($Grid,$Cfg) {
    # Commit the active editor first, then read the actual DataGridView cell values.
    # Previous launcher logic had the null check reversed, which converted every non-null visible value
    # (including 'false') into an empty string during validation.
    if($Grid.IsCurrentCellInEditMode){
        [void]$Grid.CommitEdit([System.Windows.Forms.DataGridViewDataErrorContexts]::Commit)
    }
    [void]$Grid.EndEdit()

    foreach($row in $Grid.Rows){
        $tag=$row.Tag
        if($null -eq $tag){continue}
        $cellValue=$row.Cells['Value'].Value
        $text=if($null -ne $cellValue){[string]$cellValue}else{''}
        $v=Parse-ParamText $tag.Meta $text $tag.Stage
        $stageCfg=Get-PropValue $Cfg.stages $tag.Stage
        Set-PropValue $stageCfg $tag.Dest $v
    }
    return $Cfg
}

function Apply-PathConfig($Cfg) {
    foreach ($name in @('s1_raw_kml','s1_output','s1_observed_tracks','boundary','dem','output_root','s2_dir','s3_dir','s4_dir','s5_dir','s6_dir','s7_dir')) {
        $c=$script:Controls['paths.'+$name]
        if($null -ne $c){$v=Get-PropValue $Cfg.paths $name;$c.Text=if($null -eq $v){''}else{[string]$v}}
    }
    $script:Controls['pipeline.auto_align_kde_to_gravity'].Checked=[bool](Get-PropValue $Cfg.pipeline 'auto_align_kde_to_gravity')
    $script:Controls['pipeline.save_run_config_to_output'].Checked=[bool](Get-PropValue $Cfg.pipeline 'save_run_config_to_output')
    $script:Controls['runtime.gdal_cachemax_mb'].Text=[string](Get-PropValue $Cfg.runtime 'gdal_cachemax_mb')
    $script:Controls['runtime.python_unbuffered'].Checked=[bool](Get-PropValue $Cfg.runtime 'python_unbuffered')
    $start=[string](Get-PropValue $Cfg.runtime 'start_stage')
    if([string]::IsNullOrWhiteSpace($start)){$start='auto'}
    $combo=$script:Controls['runtime.start_stage']
    if($null -ne $combo){
        $idx=$combo.Items.IndexOf($start)
        if($idx -lt 0){$idx=0}
        $combo.SelectedIndex=$idx
    }
    $resume=Get-PropValue $Cfg.runtime 'resume_completed_stages'
    $script:Controls['runtime.resume_completed_stages'].Checked=if($null -eq $resume){$true}else{[bool]$resume}
    $vr=Get-PropValue $Cfg.runtime 'validate_s5_rasters'
    $script:Controls['runtime.validate_s5_rasters'].Checked=if($null -eq $vr){$true}else{[bool]$vr}
    $adopt=Get-PropValue $Cfg.runtime 'adopt_existing_outputs'
    $script:Controls['runtime.adopt_existing_outputs'].Checked=if($null -eq $adopt){$true}else{[bool]$adopt}
    $rc=Get-PropValue $Cfg.runtime 'io_retry_count'
    $script:Controls['runtime.io_retry_count'].Text=if($null -eq $rc){'2'}else{[string]$rc}
    $rw=Get-PropValue $Cfg.runtime 'io_retry_wait_s'
    $script:Controls['runtime.io_retry_wait_s'].Text=if($null -eq $rw){'3'}else{[string]$rw}
}

function Collect-PathConfig($Cfg) {
    $pathNames=@('boundary','dem','output_root','s2_dir','s3_dir','s4_dir','s5_dir','s6_dir','s7_dir')
    if($Mode -eq 'S2S7'){
        $pathNames=@('s1_observed_tracks')+$pathNames
    } else {
        $pathNames=@('s1_raw_kml','s1_output')+$pathNames
    }
    foreach ($name in $pathNames) {
        Set-PropValue $Cfg.paths $name $script:Controls['paths.'+$name].Text.Trim()
    }
    Set-PropValue $Cfg.pipeline 'auto_align_kde_to_gravity' ([bool]$script:Controls['pipeline.auto_align_kde_to_gravity'].Checked)
    Set-PropValue $Cfg.pipeline 'save_run_config_to_output' ([bool]$script:Controls['pipeline.save_run_config_to_output'].Checked)
    $cacheText=$script:Controls['runtime.gdal_cachemax_mb'].Text.Trim();$cache=0
    if(-not [Int32]::TryParse($cacheText,[ref]$cache) -or $cache -lt 0){throw 'GDAL_CACHEMAX 必须是非负整数(MB)。'}
    Set-PropValue $Cfg.runtime 'gdal_cachemax_mb' $cache
    Set-PropValue $Cfg.runtime 'python_unbuffered' ([bool]$script:Controls['runtime.python_unbuffered'].Checked)

    $start=[string]$script:Controls['runtime.start_stage'].SelectedItem
    if([string]::IsNullOrWhiteSpace($start)){$start='auto'}
    Set-PropValue $Cfg.runtime 'start_stage' $start
    Set-PropValue $Cfg.runtime 'resume_completed_stages' ([bool]$script:Controls['runtime.resume_completed_stages'].Checked)
    Set-PropValue $Cfg.runtime 'validate_s5_rasters' ([bool]$script:Controls['runtime.validate_s5_rasters'].Checked)
    Set-PropValue $Cfg.runtime 'adopt_existing_outputs' ([bool]$script:Controls['runtime.adopt_existing_outputs'].Checked)

    $retryText=$script:Controls['runtime.io_retry_count'].Text.Trim();$retry=0
    if(-not [Int32]::TryParse($retryText,[ref]$retry) -or $retry -lt 0 -or $retry -gt 10){
        throw 'I/O 自动重试次数必须是 0-10 的整数。'
    }
    Set-PropValue $Cfg.runtime 'io_retry_count' $retry

    $waitText=$script:Controls['runtime.io_retry_wait_s'].Text.Trim();$wait=0
    if(-not [Int32]::TryParse($waitText,[ref]$wait) -or $wait -lt 0 -or $wait -gt 300){
        throw 'I/O 重试等待秒数必须是 0-300 的整数。'
    }
    Set-PropValue $Cfg.runtime 'io_retry_wait_s' $wait
    return $Cfg
}

function Update-GridFilter($Grid,$StageCombo,$SearchBox) {
    $Grid.CurrentCell=$null
    $stage=[string]$StageCombo.SelectedItem
    $q=$SearchBox.Text.Trim().ToLowerInvariant()
    foreach($row in $Grid.Rows){
        $stageOk=($stage -eq '全部') -or ([string]$row.Cells['Stage'].Value -eq $stage)
        $text=(([string]$row.Cells['Option'].Value)+' '+([string]$row.Cells['Help'].Value)).ToLowerInvariant()
        $qOk=($q -eq '') -or $text.Contains($q)
        $row.Visible=$stageOk -and $qOk
    }
}

function Show-Configurator($Schema,$Cfg) {
    Write-Host 'Opening parameter window...' -ForegroundColor Cyan
    $form=New-Object System.Windows.Forms.Form
    $script:MainForm=$form
    $form.Text=if($Mode -eq 'S2S7'){'Activity-Aware Pipeline - S2→S7 智能续跑'}else{'Activity-Aware Pipeline - S1→S7 智能续跑'}
    $form.Size=New-Object System.Drawing.Size(1450,900)
    $form.StartPosition=[System.Windows.Forms.FormStartPosition]::CenterScreen
    $form.MinimumSize=New-Object System.Drawing.Size(1100,700)

    $tabs=New-Object System.Windows.Forms.TabControl
    $tabs.Dock=[System.Windows.Forms.DockStyle]::Fill
    $form.Controls.Add($tabs)

    # Path tab uses only 12 lightweight rows and appears immediately.
    $pathTab=New-Object System.Windows.Forms.TabPage
    $pathTab.Text='路径 / 输出'
    $pathPanel=New-Object System.Windows.Forms.Panel
    $pathPanel.Dock=[System.Windows.Forms.DockStyle]::Fill
    $pathPanel.AutoScroll=$true
    $pathTab.Controls.Add($pathPanel)
    Add-PathRows $pathPanel
    $pathNote = New-Object System.Windows.Forms.Label
    $pathNote.Text = if($Mode -eq 'S2S7'){
        '当前为 S2→S7 模式：只需选择 S1 observed_tracks、研究区边界、DEM 和输出根目录；S1 原始 KML / S1 输出目录已自动禁用，S1 参数不会校验也不会运行。'
    }else{
        '当前为 S1→S7 模式：请选择原始 KML、S1 输出根目录、研究区边界、DEM 和 S2-S7 输出根目录；observed_tracks 将由 S1 输出目录自动推导。'
    }
    $pathNote.Dock = [System.Windows.Forms.DockStyle]::Top
    $pathNote.Height = 42
    $pathNote.Padding = New-Object System.Windows.Forms.Padding(8,8,8,4)
    $pathNote.ForeColor = [System.Drawing.Color]::DarkBlue
    $pathPanel.Controls.Add($pathNote)
    $pathNote.BringToFront()
    $tabs.TabPages.Add($pathTab)|Out-Null

    # All 190 parameters live in one DataGridView instead of 190 nested WinForms controls.
    $paramTab=New-Object System.Windows.Forms.TabPage
    $paramTab.Text=if($Mode -eq 'S2S7'){'S2-S7 参数'}else{'S1-S7 参数'}
    $top=New-Object System.Windows.Forms.FlowLayoutPanel
    $top.Dock=[System.Windows.Forms.DockStyle]::Top
    $top.Height=42
    $top.WrapContents=$false
    $top.Padding=New-Object System.Windows.Forms.Padding(6)
    $stageLabel=New-Object System.Windows.Forms.Label;$stageLabel.Text='阶段:';$stageLabel.AutoSize=$true;$stageLabel.Margin=New-Object System.Windows.Forms.Padding(4,8,4,0)
    $stageCombo=New-Object System.Windows.Forms.ComboBox;$stageCombo.DropDownStyle=[System.Windows.Forms.ComboBoxStyle]::DropDownList;$stageCombo.Width=90
    $stageItems=if($Mode -eq 'S2S7'){@('全部','S2','S3','S4','S5','S6','S7')}else{@('全部','S1','S2','S3','S4','S5','S6','S7')}
    foreach($x in $stageItems){[void]$stageCombo.Items.Add($x)};$stageCombo.SelectedIndex=0
    $searchLabel=New-Object System.Windows.Forms.Label;$searchLabel.Text='搜索:';$searchLabel.AutoSize=$true;$searchLabel.Margin=New-Object System.Windows.Forms.Padding(18,8,4,0)
    $searchBox=New-Object System.Windows.Forms.TextBox;$searchBox.Width=300
    $resetStage=New-Object System.Windows.Forms.Button;$resetStage.Text='当前阶段恢复默认';$resetStage.Width=130
    foreach($ctl in @($stageLabel,$stageCombo,$searchLabel,$searchBox,$resetStage)){[void]$top.Controls.Add($ctl)}
    $paramTab.Controls.Add($top)
    $grid=New-ParameterGrid $Schema $Cfg
    $paramTab.Controls.Add($grid)
    $grid.BringToFront()
    $tabs.TabPages.Add($paramTab)|Out-Null

    $stageCombo.Add_SelectedIndexChanged({Update-GridFilter $grid $stageCombo $searchBox})
    $searchBox.Add_TextChanged({Update-GridFilter $grid $stageCombo $searchBox})
    $resetStage.Add_Click({
        try{
            $defaults=Read-JsonFile $script:DefaultConfigPath
            $sel=[string]$stageCombo.SelectedItem
            foreach($row in $grid.Rows){
                $tag=$row.Tag;if($null -eq $tag){continue}
                if($sel -ne '全部' -and $tag.Stage -ne $sel){continue}
                $stageCfg=Get-PropValue $defaults.stages $tag.Stage
                $v=Get-PropValue $stageCfg $tag.Dest
                $row.Cells['Value'].Value=Format-ParamValue $tag.Meta $v
            }
        }catch{Show-Message $_.Exception.Message '恢复默认失败' ([System.Windows.Forms.MessageBoxIcon]::Error)}
    })

    $runTab=New-Object System.Windows.Forms.TabPage
    $runTab.Text='运行设置'
    $runPanel=New-Object System.Windows.Forms.Panel;$runPanel.Dock=[System.Windows.Forms.DockStyle]::Fill;$runTab.Controls.Add($runPanel)
    $cbAlign=New-Object System.Windows.Forms.CheckBox;$cbAlign.Text='KDE 自动继承 S4 瓦片网格';$cbAlign.AutoSize=$true;$cbAlign.Location=New-Object System.Drawing.Point(30,30);$script:Controls['pipeline.auto_align_kde_to_gravity']=$cbAlign;$runPanel.Controls.Add($cbAlign)
    $cbSave=New-Object System.Windows.Forms.CheckBox;$cbSave.Text='输出目录保存本次配置 pipeline_run_config.json';$cbSave.AutoSize=$true;$cbSave.Location=New-Object System.Drawing.Point(30,65);$script:Controls['pipeline.save_run_config_to_output']=$cbSave;$runPanel.Controls.Add($cbSave)

    $labStart=New-Label '开始阶段' 180;$labStart.Location=New-Object System.Drawing.Point(30,105);$runPanel.Controls.Add($labStart)
    $comboStart=New-Object System.Windows.Forms.ComboBox;$comboStart.DropDownStyle=[System.Windows.Forms.ComboBoxStyle]::DropDownList;$comboStart.Location=New-Object System.Drawing.Point(215,105);$comboStart.Width=180
    $startItems=if($Mode -eq 'S2S7'){@('auto','S2','S3','S4','S5','S6','S7')}else{@('auto','S1','S2','S3','S4','S5','S6','S7')}
    foreach($x in $startItems){[void]$comboStart.Items.Add($x)}
    $script:Controls['runtime.start_stage']=$comboStart;$runPanel.Controls.Add($comboStart)
    $labStartHelp=New-Object System.Windows.Forms.Label;$labStartHelp.Text='auto = 推荐：按成功标记自动续跑；当前这次 S6 失败可临时选择 S6。';$labStartHelp.AutoSize=$true;$labStartHelp.Location=New-Object System.Drawing.Point(420,110);$runPanel.Controls.Add($labStartHelp)

    $cbResume=New-Object System.Windows.Forms.CheckBox;$cbResume.Text='自动跳过“参数一致且已成功”的阶段（断点续跑）';$cbResume.AutoSize=$true;$cbResume.Location=New-Object System.Drawing.Point(30,145);$script:Controls['runtime.resume_completed_stages']=$cbResume;$runPanel.Controls.Add($cbResume)
    $cbValidate=New-Object System.Windows.Forms.CheckBox;$cbValidate.Text='S5 完成后、进入 S6/S7 前完整逐块校验 4 个核心栅格';$cbValidate.AutoSize=$true;$cbValidate.Location=New-Object System.Drawing.Point(30,180);$script:Controls['runtime.validate_s5_rasters']=$cbValidate;$runPanel.Controls.Add($cbValidate)
    $cbAdopt=New-Object System.Windows.Forms.CheckBox;$cbAdopt.Text='复用已有结果：检测 S3/S4/S5 输出并询问是否直接续跑';$cbAdopt.AutoSize=$true;$cbAdopt.Location=New-Object System.Drawing.Point(30,215);$script:Controls['runtime.adopt_existing_outputs']=$cbAdopt;$runPanel.Controls.Add($cbAdopt)

    $labRetry=New-Label 'S6/S7 I/O 自动重试次数' 180;$labRetry.Location=New-Object System.Drawing.Point(30,255);$runPanel.Controls.Add($labRetry)
    $tbRetry=New-Object System.Windows.Forms.TextBox;$tbRetry.Location=New-Object System.Drawing.Point(215,255);$tbRetry.Width=80;$script:Controls['runtime.io_retry_count']=$tbRetry;$runPanel.Controls.Add($tbRetry)
    $labWait=New-Label '重试等待 (秒)' 130;$labWait.Location=New-Object System.Drawing.Point(330,255);$runPanel.Controls.Add($labWait)
    $tbWait=New-Object System.Windows.Forms.TextBox;$tbWait.Location=New-Object System.Drawing.Point(465,255);$tbWait.Width=80;$script:Controls['runtime.io_retry_wait_s']=$tbWait;$runPanel.Controls.Add($tbWait)

    $labCache=New-Label 'GDAL_CACHEMAX (MB)' 180;$labCache.Location=New-Object System.Drawing.Point(30,300);$runPanel.Controls.Add($labCache)
    $tbCache=New-Object System.Windows.Forms.TextBox;$tbCache.Location=New-Object System.Drawing.Point(215,300);$tbCache.Width=140;$script:Controls['runtime.gdal_cachemax_mb']=$tbCache;$runPanel.Controls.Add($tbCache)
    $cbUnbuf=New-Object System.Windows.Forms.CheckBox;$cbUnbuf.Text='PYTHONUNBUFFERED（实时输出进度）';$cbUnbuf.AutoSize=$true;$cbUnbuf.Location=New-Object System.Drawing.Point(30,340);$script:Controls['runtime.python_unbuffered']=$cbUnbuf;$runPanel.Controls.Add($cbUnbuf)

    $note=New-Object System.Windows.Forms.Label;$note.AutoSize=$true;$note.MaximumSize=New-Object System.Drawing.Size(1120,0);$note.Location=New-Object System.Drawing.Point(30,390);$note.Text=if($Mode -eq 'S2S7'){'推荐：开始阶段=auto，并启用自动续跑、S5 栅格校验和已有结果复用。检测到完整的 S5 输出但缺少成功标记时，会询问是否从 S6 继续。'}else{'推荐：开始阶段=auto，并启用自动续跑和 S5 栅格校验。'};$runPanel.Controls.Add($note)
    $tabs.TabPages.Add($runTab)|Out-Null

    $bottom=New-Object System.Windows.Forms.FlowLayoutPanel
    $bottom.Dock=[System.Windows.Forms.DockStyle]::Bottom
    $bottom.Height=55
    $bottom.FlowDirection=[System.Windows.Forms.FlowDirection]::RightToLeft
    $bottom.Padding=New-Object System.Windows.Forms.Padding(8)
    $form.Controls.Add($bottom)

    $btnRun=New-Object System.Windows.Forms.Button;$btnRun.Text='保存并运行';$btnRun.Width=120
    $btnCancel=New-Object System.Windows.Forms.Button;$btnCancel.Text='取消';$btnCancel.Width=90
    $btnSave=New-Object System.Windows.Forms.Button;$btnSave.Text='另存配置';$btnSave.Width=100
    $btnLoad=New-Object System.Windows.Forms.Button;$btnLoad.Text='载入配置';$btnLoad.Width=100
    $btnReset=New-Object System.Windows.Forms.Button;$btnReset.Text='全部恢复默认';$btnReset.Width=110
    foreach($b in @($btnRun,$btnCancel,$btnSave,$btnLoad,$btnReset)){$bottom.Controls.Add($b)|Out-Null}

    $btnCancel.Add_Click({$script:RunRequested=$false;$form.Close()})
    $btnReset.Add_Click({
        try{$script:CurrentConfig=Read-JsonFile $script:DefaultConfigPath;Apply-PathConfig $script:CurrentConfig;Apply-ModeAwareUI;Apply-ConfigToGrid $grid $script:CurrentConfig}
        catch{Show-Message $_.Exception.Message '恢复默认失败' ([System.Windows.Forms.MessageBoxIcon]::Error)}
    })
    $btnLoad.Add_Click({
        $d=New-Object System.Windows.Forms.OpenFileDialog;$d.Filter='JSON config (*.json)|*.json|All files (*.*)|*.*';$d.Title='载入参数配置'
        if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){
            try{$script:CurrentConfig=Read-JsonFile $d.FileName;Apply-PathConfig $script:CurrentConfig;Apply-ModeAwareUI;Apply-ConfigToGrid $grid $script:CurrentConfig}
            catch{Show-Message $_.Exception.Message '载入失败' ([System.Windows.Forms.MessageBoxIcon]::Error)}
        }
    })
    $btnSave.Add_Click({
        try{
            $script:CurrentConfig=Collect-PathConfig $script:CurrentConfig
            $script:CurrentConfig=Collect-GridToConfig $grid $script:CurrentConfig
            $d=New-Object System.Windows.Forms.SaveFileDialog;$d.Filter='JSON config (*.json)|*.json';$d.Title='保存参数配置';$d.FileName='pipeline_config.json'
            if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){Write-JsonFile $script:CurrentConfig $d.FileName;Show-Message '配置已保存。'}
        }catch{Show-Message $_.Exception.Message '参数错误' ([System.Windows.Forms.MessageBoxIcon]::Error)}
    })
    $btnRun.Add_Click({
        try{
            $script:CurrentConfig=Collect-PathConfig $script:CurrentConfig
            $script:CurrentConfig=Collect-GridToConfig $grid $script:CurrentConfig
            Write-JsonFile $script:CurrentConfig $script:LastConfigPath
            $script:RunRequested=$true;$form.Close()
        }catch{Show-Message $_.Exception.Message '参数错误' ([System.Windows.Forms.MessageBoxIcon]::Error)}
    })

    Apply-PathConfig $Cfg
    Apply-ModeAwareUI
    $form.Add_Shown({$form.Activate();$form.TopMost=$true;$form.TopMost=$false})
    [void]$form.ShowDialog()
    return $script:RunRequested
}

function Assert-Folder([string]$PathValue,[string]$Label) {
    if ([string]::IsNullOrWhiteSpace($PathValue) -or -not (Test-Path -LiteralPath $PathValue -PathType Container)) { throw ($Label+'不存在: '+$PathValue) }
}
function Assert-File([string]$PathValue,[string]$Label) {
    if ([string]::IsNullOrWhiteSpace($PathValue) -or -not (Test-Path -LiteralPath $PathValue -PathType Leaf)) { throw ($Label+'不存在: '+$PathValue) }
}

function Get-StageArgs([string]$Stage,$Schema,$Cfg) {
    $out = New-Object System.Collections.Generic.List[string]
    $meta=Get-PropValue $Schema.stages $Stage
    $stageCfg=Get-PropValue $Cfg.stages $Stage
    foreach ($p in $meta.parameters) {
        $v=Get-PropValue $stageCfg $p.dest
        if ($p.kind -eq 'bool_optional') {
            if ([bool]$v) { $out.Add([string]$p.positive_option) } else { $out.Add([string]$p.negative_option) }
        } elseif ($p.kind -eq 'bool_flag') {
            $target=[bool]$p.flag_value
            if ([bool]$v -eq $target) { $out.Add([string]$p.option) }
        } else {
            if ($null -eq $v) { continue }
            $s=[string]$v
            if ([string]::IsNullOrWhiteSpace($s)) { continue }
            $out.Add([string]$p.option); $out.Add($s)
        }
    }
    return $out.ToArray()
}


function Get-TextSha256([string]$TextValue) {
    $sha=[System.Security.Cryptography.SHA256]::Create()
    try{
        $bytes=[System.Text.Encoding]::UTF8.GetBytes($TextValue)
        $hash=$sha.ComputeHash($bytes)
        return (($hash | ForEach-Object {$_.ToString('x2')}) -join '')
    } finally {$sha.Dispose()}
}

function Get-FileSignature([string]$PathValue) {
    if([string]::IsNullOrWhiteSpace($PathValue) -or -not (Test-Path -LiteralPath $PathValue -PathType Leaf)){return $null}
    $f=Get-Item -LiteralPath $PathValue
    return [ordered]@{path=$f.FullName;size=[int64]$f.Length;mtime_utc=$f.LastWriteTimeUtc.ToString('o')}
}

function Get-ShapefileSignature([string]$ShpPath) {
    if([string]::IsNullOrWhiteSpace($ShpPath)){return $null}
    $dir=Split-Path -Parent $ShpPath
    $base=[System.IO.Path]::GetFileNameWithoutExtension($ShpPath)
    $items=@()
    foreach($ext in @('.shp','.shx','.dbf','.prj','.cpg')){
        $p=Join-Path $dir ($base+$ext)
        $sig=Get-FileSignature $p
        if($null -ne $sig){$items+=$sig}
    }
    return $items
}

function Get-TrackFolderSignature([string]$Folder) {
    if([string]::IsNullOrWhiteSpace($Folder) -or -not (Test-Path -LiteralPath $Folder -PathType Container)){return $null}
    $files=Get-ChildItem -LiteralPath $Folder -Filter '*__gxtrack.csv' -File -ErrorAction SilentlyContinue
    if($null -eq $files){$files=@()}
    $count=@($files).Count
    $total=[int64]0
    $maxTicks=[int64]0
    foreach($f in $files){
        $total += [int64]$f.Length
        if($f.LastWriteTimeUtc.Ticks -gt $maxTicks){$maxTicks=$f.LastWriteTimeUtc.Ticks}
    }
    return [ordered]@{path=(Resolve-Path -LiteralPath $Folder).Path;count=$count;total_size=$total;max_mtime_ticks=$maxTicks}
}

function Get-StageFingerprints($Cfg,[string]$S1Tracks) {
    $out=@{}
    $upstream=''
    if($Mode -eq 'S1S7'){
        $s1Obj=[ordered]@{
            stage='S1'
            raw_kml=[string]$Cfg.paths.s1_raw_kml
            output=[string]$Cfg.paths.s1_output
            params=$Cfg.stages.S1
        }
        $upstream=Get-TextSha256 (($s1Obj | ConvertTo-Json -Depth 30 -Compress))
        $out['S1']=$upstream
    } else {
        $seed=[ordered]@{observed_tracks=(Get-TrackFolderSignature $S1Tracks)}
        $upstream=Get-TextSha256 (($seed | ConvertTo-Json -Depth 30 -Compress))
    }

    foreach($stage in @('S2','S3','S4','S5','S6','S7')){
        $obj=[ordered]@{stage=$stage;upstream=$upstream;params=(Get-PropValue $Cfg.stages $stage)}
        if($stage -eq 'S2'){
            $obj['tracks']=Get-TrackFolderSignature $S1Tracks
            if(-not [bool](Get-PropValue $Cfg.stages.S2 'no_boundary')){$obj['boundary']=Get-ShapefileSignature ([string]$Cfg.paths.boundary)}
            if(-not [string]::IsNullOrWhiteSpace([string]$Cfg.paths.dem)){$obj['dem']=Get-FileSignature ([string]$Cfg.paths.dem)}
        }
        if($stage -eq 'S5'){$obj['auto_align_kde_to_gravity']=[bool]$Cfg.pipeline.auto_align_kde_to_gravity}
        if($stage -eq 'S6'){$obj['implementation_version']='S6_v0.5_evidence_first_activity_support'}
        if($stage -eq 'S7'){
            $obj['implementation_version']='S7_v0.5_smooth_network'
            if(-not [string]::IsNullOrWhiteSpace([string]$Cfg.paths.dem)){$obj['dem']=Get-FileSignature ([string]$Cfg.paths.dem)}
        }
        $upstream=Get-TextSha256 (($obj | ConvertTo-Json -Depth 30 -Compress))
        $out[$stage]=$upstream
    }
    return $out
}

function Get-StateDir([string]$Root) {
    $d=Join-Path $Root '.pipeline_state'
    if(-not (Test-Path -LiteralPath $d)){New-Item -ItemType Directory -Path $d -Force | Out-Null}
    return $d
}

function Get-StageMarker([string]$Root,[string]$Stage) {
    return (Join-Path (Get-StateDir $Root) ($Stage+'.success.json'))
}

function Test-StageMarker([string]$Root,[string]$Stage,[string]$Fingerprint,[string[]]$Outputs) {
    $m=Get-StageMarker $Root $Stage
    if(-not (Test-Path -LiteralPath $m -PathType Leaf)){return $false}
    try{$o=Read-JsonFile $m}catch{return $false}
    if([string]$o.fingerprint -ne $Fingerprint){return $false}
    foreach($p in $Outputs){
        if([string]::IsNullOrWhiteSpace($p) -or -not (Test-Path -LiteralPath $p)){return $false}
    }
    return $true
}

function Write-StageMarker([string]$Root,[string]$Stage,[string]$Fingerprint,[string[]]$Outputs) {
    $o=[ordered]@{
        stage=$Stage
        fingerprint=$Fingerprint
        completed_at=(Get-Date).ToUniversalTime().ToString('o')
        outputs=$Outputs
    }
    Write-JsonFile $o (Get-StageMarker $Root $Stage)
}

function Get-StartStageNumber($Cfg) {
    $s=[string](Get-PropValue $Cfg.runtime 'start_stage')
    if([string]::IsNullOrWhiteSpace($s) -or $s -eq 'auto'){return 0}
    if($s -match '^S([1-7])$'){return [int]$Matches[1]}
    return 0
}

function Should-RunStage([string]$Stage,[int]$Number,[string]$Fingerprint,[string[]]$Outputs,$Cfg,[string]$Root) {
    $adoptStart=if($null -eq $script:AdoptedStartStage){0}else{[int]$script:AdoptedStartStage}
    if($adoptStart -gt 0 -and $Number -lt $adoptStart){
        Write-Host ("[SKIP] {0}: 已确认复用已有结果，本次从 S{1} 继续。" -f $Stage,$adoptStart) -ForegroundColor Yellow
        return $false
    }

    $start=Get-StartStageNumber $Cfg
    if($start -gt 0){
        if($Number -lt $start){
            Write-Host ("[SKIP] {0}: 手动开始阶段={1}" -f $Stage,[string]$Cfg.runtime.start_stage) -ForegroundColor Yellow
            return $false
        }
        return $true
    }

    if(-not [bool](Get-PropValue $Cfg.runtime 'resume_completed_stages')){return $true}
    if($script:ForceDownstream){return $true}

    if(Test-StageMarker $Root $Stage $Fingerprint $Outputs){
        Write-Host ("[RESUME] {0}: 参数一致且已有成功标记，跳过。" -f $Stage) -ForegroundColor Green
        return $false
    }

    $script:ForceDownstream=$true
    return $true
}

function Test-Rasters([string[]]$Paths,[int]$Retries,[int]$WaitSeconds) {
    $args=New-Object System.Collections.Generic.List[string]
    $args.Add('tools\validate_rasters.py')
    $args.Add('--retries');$args.Add([string][Math]::Max(0,$Retries))
    $args.Add('--wait-s');$args.Add([string][Math]::Max(0,$WaitSeconds))
    foreach($p in $Paths){$args.Add($p)}
    Write-Host ''
    Write-Host '========== S5 栅格完整性检查 ==========' -ForegroundColor Cyan
    & $script:PythonExe @($args.ToArray())
    return ($LASTEXITCODE -eq 0)
}

function Assert-S5Rasters([string[]]$Paths,$Cfg) {
    if(-not [bool](Get-PropValue $Cfg.runtime 'validate_s5_rasters')){return}
    $retry=[int](Get-PropValue $Cfg.runtime 'io_retry_count')
    $wait=[int](Get-PropValue $Cfg.runtime 'io_retry_wait_s')
    if(-not (Test-Rasters $Paths 1 $wait)){
        throw 'S5 栅格完整读取校验失败。请从 S5 重新运行；S6/S7 未启动。'
    }
}

function Test-IsRasterIoFailure($OutputLines) {
    if($null -eq $OutputLines){return $false}
    $text=(@($OutputLines) -join "`n")
    if([string]::IsNullOrWhiteSpace($text)){return $false}
    $patterns=@(
        'RasterioIOError',
        'IReadBlock',
        'TIFFRead',
        'ReadBlock',
        'Error reading block',
        'Input/output error',
        'I/O error',
        'failed to read raster',
        'failed to read.*tif',
        'CPLE_AppDefinedError.*read'
    )
    foreach($pattern in $patterns){
        if($text -match $pattern){return $true}
    }
    return $false
}

function Run-PythonWithRasterRetry([string]$Stage,[string[]]$Arguments,[string[]]$RasterPaths,$Cfg) {
    $maxRetry=[int](Get-PropValue $Cfg.runtime 'io_retry_count')
    $wait=[int](Get-PropValue $Cfg.runtime 'io_retry_wait_s')
    for($attempt=0;$attempt -le $maxRetry;$attempt++){
        try{
            Run-Python $Stage $Arguments
            return
        }catch{
            $isRasterIo=Test-IsRasterIoFailure $script:LastPythonOutput
            if(-not $isRasterIo){
                Write-Host ("[NO-RETRY] {0} 为确定性程序/拓扑错误，不执行栅格 I/O 重试。" -f $Stage) -ForegroundColor Yellow
                throw
            }
            if($attempt -ge $maxRetry){throw}
            Write-Host ("[RETRY] {0} 检测到栅格 I/O 读取异常，第 {1} 次复检输入栅格..." -f $Stage,($attempt+1)) -ForegroundColor Yellow
            if([bool](Get-PropValue $Cfg.runtime 'validate_s5_rasters')){
                if(-not (Test-Rasters $RasterPaths 1 $wait)){
                    throw ($Stage+' 失败，同时 S5 输入栅格复检失败。请从 S5 重新运行。')
                }
            }
            if($wait -gt 0){Write-Host ("[RETRY] 等待 {0} 秒后重新运行 {1}..." -f $wait,$Stage) -ForegroundColor Yellow;Start-Sleep -Seconds $wait}
        }
    }
}


function Test-HasAnySuccessMarker([string]$Root) {
    $d=Join-Path $Root '.pipeline_state'
    if(-not (Test-Path -LiteralPath $d -PathType Container)){return $false}
    $m=Get-ChildItem -LiteralPath $d -Filter 'S*.success.json' -File -ErrorAction SilentlyContinue
    return ($null -ne $m -and @($m).Count -gt 0)
}

function Get-ExistingResumeCandidate($Cfg,[string]$S3,[string]$S4,[string]$S5) {
    $candidate=[ordered]@{deepest=0;stage='none';details=@();s5_rasters=@()}

    $s3Main=Join-Path $S3 'road_candidate_points.csv'
    $activity=Join-Path $S3 'activity_zones.gpkg'
    if((Test-Path -LiteralPath $s3Main -PathType Leaf) -and (Test-Path -LiteralPath $activity -PathType Leaf)){
        $candidate.deepest=3;$candidate.stage='S3';$candidate.details+=('S3: '+$s3Main)
    }

    $s4Main=Join-Path $S4 'all_cleaned_points.csv'
    $s4Report=Join-Path $S4 'gravity_uturn_report.json'
    if((Test-Path -LiteralPath $s4Main -PathType Leaf) -and (Test-Path -LiteralPath $s4Report -PathType Leaf)){
        $candidate.deepest=4;$candidate.stage='S4';$candidate.details+=('S4: '+$s4Main)
    }

    $kdeReport=Join-Path $S5 'directional_kde_report.json'
    if(Test-Path -LiteralPath $kdeReport -PathType Leaf){
        try{
            $rep=Read-JsonFile $kdeReport
            $rasters=@([string]$rep.float32,[string]$rep.support,[string]$rep.distance,[string]$rep.raw_union)
            $allFiles=$true
            foreach($p in $rasters){if([string]::IsNullOrWhiteSpace($p) -or -not (Test-Path -LiteralPath $p -PathType Leaf)){$allFiles=$false;break}}
            if($allFiles){
                $candidate.deepest=5;$candidate.stage='S5';$candidate.details+=('S5: '+$kdeReport);$candidate.s5_rasters=$rasters
            }
        }catch{}
    }
    return [PSCustomObject]$candidate
}

function Confirm-AdoptExistingResults($Candidate) {
    if($Candidate.deepest -lt 3){return $false}
    $next=[Math]::Min(7,[int]$Candidate.deepest+1)
    $msg="检测到已有到 $($Candidate.stage) 的结果，但没有当前代码的成功标记。`r`n`r`n"
    $msg+="建议直接从 S$next 继续，避免重新计算 S2-S5。`r`n"
    if($Candidate.deepest -ge 5){$msg+="S5 四个核心栅格将在确认后用当前 Python/Rasterio 完整逐块校验。`r`n"}
    $msg+="`r`n【是】复用现有结果并从 S$next 继续`r`n【否】从 S2 重新计算`r`n【取消】退出"
    $r=[System.Windows.Forms.MessageBox]::Show(
        $msg,'检测到已有结果',
        [System.Windows.Forms.MessageBoxButtons]::YesNoCancel,
        [System.Windows.Forms.MessageBoxIcon]::Question
    )
    if($r -eq [System.Windows.Forms.DialogResult]::Cancel){throw '用户取消运行。'}
    return ($r -eq [System.Windows.Forms.DialogResult]::Yes)
}

function Run-Python([string]$Stage,[string[]]$Arguments) {
    Write-Host ''
    Write-Host ('========== '+$Stage+' ==========') -ForegroundColor Cyan
    Write-Host ($script:PythonExe+' '+(($Arguments | ForEach-Object { if ($_ -match '\s') {'"'+$_+'"'} else {$_} }) -join ' '))
    $captured=New-Object System.Collections.Generic.List[string]
    # Windows PowerShell 5.1 represents native stderr merged by 2>&1 as ErrorRecord.
    # Keep the script-wide Stop policy, but do not let a Python RuntimeWarning on
    # stderr become a terminating PowerShell exception. Native success/failure is
    # decided only from LASTEXITCODE immediately after the Python process exits.
    $nativeErrorActionPreference=$ErrorActionPreference
    try {
        $ErrorActionPreference='Continue'
        & $script:PythonExe @Arguments 2>&1 | ForEach-Object {
            $line=[string]$_
            $captured.Add($line)
            Write-Host $line
        }
        $exitCode=$LASTEXITCODE
    } finally {
        $ErrorActionPreference=$nativeErrorActionPreference
    }
    $script:LastPythonOutput=$captured.ToArray()
    if ($exitCode -ne 0) {
        $tail=@($script:LastPythonOutput | Select-Object -Last 20) -join "`n"
        throw ($Stage+' 失败，退出码 '+$exitCode+"`n"+$tail)
    }
}

function Join-Out($Cfg,[string]$Name) { return (Join-Path ([string]$Cfg.paths.output_root) ([string](Get-PropValue $Cfg.paths $Name))) }

function Run-Pipeline($Schema,$Cfg) {
    $script:PythonExe=Find-Python
    $root=[string]$Cfg.paths.output_root
    if ([string]::IsNullOrWhiteSpace($root)) { throw '必须选择 S2-S7 输出根目录。' }
    if (-not (Test-Path -LiteralPath $root)) { New-Item -ItemType Directory -Path $root -Force | Out-Null }

    # Save before execution too, so even a failed run has a reproducible config.
    if ([bool]$Cfg.pipeline.save_run_config_to_output) { Write-JsonFile $Cfg (Join-Path $root 'pipeline_run_config.json') }

    $env:GDAL_CACHEMAX=[string]$Cfg.runtime.gdal_cachemax_mb
    if ([bool]$Cfg.runtime.python_unbuffered) { $env:PYTHONUNBUFFERED='1' } else { $env:PYTHONUNBUFFERED='0' }

    $script:ForceDownstream=$false
    $s1Tracks=''

    Push-Location $script:CodeDir
    try {
        # --------------------------------------------------------------
        # S1 / existing observed_tracks
        # --------------------------------------------------------------
        if ($Mode -eq 'S1S7') {
            Assert-Folder ([string]$Cfg.paths.s1_raw_kml) 'S1 原始 KML 目录'
            $s1Out=[string]$Cfg.paths.s1_output
            if ([string]::IsNullOrWhiteSpace($s1Out)) { throw '必须填写 S1 输出目录。' }
            if (-not (Test-Path -LiteralPath $s1Out)) { New-Item -ItemType Directory -Path $s1Out -Force | Out-Null }
            $s1Tracks=Join-Path $s1Out 'observed_tracks'

            $tmpFp=Get-TextSha256 (([ordered]@{
                stage='S1';raw_kml=[string]$Cfg.paths.s1_raw_kml;output=$s1Out;params=$Cfg.stages.S1
            } | ConvertTo-Json -Depth 30 -Compress))

            if(Should-RunStage 'S1' 1 $tmpFp @($s1Tracks) $Cfg $root){
                $a=New-Object System.Collections.Generic.List[string]
                $a.Add('S1_extract_kml.py');$a.Add([string]$Cfg.paths.s1_raw_kml);$a.Add('-o');$a.Add($s1Out)
                foreach($x in (Get-StageArgs 'S1' $Schema $Cfg)){ $a.Add($x) }
                Run-Python 'S1/7 KML轨迹提取' $a.ToArray()
                Assert-Folder $s1Tracks 'S1 observed_tracks'
                Write-StageMarker $root 'S1' $tmpFp @($s1Tracks)
            }
        } else {
            $s1Tracks=[string]$Cfg.paths.s1_observed_tracks
        }

        Assert-Folder $s1Tracks 'S1 observed_tracks'
        $tracks=Get-ChildItem -LiteralPath $s1Tracks -Filter '*__gxtrack.csv' -File -ErrorAction SilentlyContinue
        if ($null -eq $tracks -or $tracks.Count -eq 0) { throw ('未找到 *__gxtrack.csv: '+$s1Tracks) }

        $fps=Get-StageFingerprints $Cfg $s1Tracks
        $s2=Join-Out $Cfg 's2_dir';$s3=Join-Out $Cfg 's3_dir';$s4=Join-Out $Cfg 's4_dir';$s5=Join-Out $Cfg 's5_dir';$s6=Join-Out $Cfg 's6_dir';$s7=Join-Out $Cfg 's7_dir'
        foreach($stageOut in @($s2,$s3,$s4,$s5,$s6,$s7)){
            if(-not (Test-Path -LiteralPath $stageOut)){ New-Item -ItemType Directory -Path $stageOut -Force | Out-Null }
        }

        $manualStart=Get-StartStageNumber $Cfg
        Write-Host ''
        Write-Host ('[RUN SETTINGS] start_stage='+[string]$Cfg.runtime.start_stage+
                    ', resume='+[string]$Cfg.runtime.resume_completed_stages+
                    ', adopt_existing='+[string]$Cfg.runtime.adopt_existing_outputs) -ForegroundColor Cyan

        $script:AdoptedStartStage=0
        if($manualStart -eq 0 -and
           [bool](Get-PropValue $Cfg.runtime 'adopt_existing_outputs') -and
           -not (Test-HasAnySuccessMarker $root)){
            $cand=Get-ExistingResumeCandidate $Cfg $s3 $s4 $s5
            if($cand.deepest -ge 3 -and (Confirm-AdoptExistingResults $cand)){
                if($cand.deepest -ge 5 -and [bool](Get-PropValue $Cfg.runtime 'validate_s5_rasters')){
                    Write-Host '[RESUME] 正在校验已有 S5 四个核心栅格...' -ForegroundColor Cyan
                    if(-not (Test-Rasters @($cand.s5_rasters) 1 ([int]$Cfg.runtime.io_retry_wait_s))){
                        throw '已有 S5 栅格校验失败，不能直接续跑。请从 S5 或 S2 重新计算。'
                    }
                }
                $script:AdoptedStartStage=[Math]::Min(7,[int]$cand.deepest+1)
                Write-Host ("[MIGRATION] 已确认复用到 {0}，本次从 S{1} 开始。" -f $cand.stage,$script:AdoptedStartStage) -ForegroundColor Green
            }
        }

        # --------------------------------------------------------------
        # S2
        # --------------------------------------------------------------
        $s2Main=Join-Path $s2 'all_cleaned_points.csv'
        if(Should-RunStage 'S2' 2 $fps['S2'] @($s2Main) $Cfg $root){
            $a=New-Object System.Collections.Generic.List[string]
            $a.Add('S2_clean_tracks.py');$a.Add('--input');$a.Add($s1Tracks);$a.Add('--output');$a.Add($s2)
            $noBoundary=[bool](Get-PropValue $Cfg.stages.S2 'no_boundary')
            if (-not $noBoundary) { Assert-File ([string]$Cfg.paths.boundary) '研究区边界';$a.Add('--boundary');$a.Add([string]$Cfg.paths.boundary) }
            if (-not [string]::IsNullOrWhiteSpace([string]$Cfg.paths.dem)) { Assert-File ([string]$Cfg.paths.dem) 'DEM';$a.Add('--dem');$a.Add([string]$Cfg.paths.dem) }
            foreach($x in (Get-StageArgs 'S2' $Schema $Cfg)){ $a.Add($x) }
            Run-Python 'S2/7 轨迹清洗' $a.ToArray()
            Assert-File $s2Main 'S2 all_cleaned_points.csv'
            Write-StageMarker $root 'S2' $fps['S2'] @($s2Main)
        }

        # --------------------------------------------------------------
        # S3
        # --------------------------------------------------------------
        $s3Main=Join-Path $s3 'road_candidate_points.csv'
        if(Should-RunStage 'S3' 3 $fps['S3'] @($s3Main) $Cfg $root){
            Assert-File $s2Main 'S2 all_cleaned_points.csv'
            $a=New-Object System.Collections.Generic.List[string]
            $a.Add('S3_activity_zone_filter.py');$a.Add('--input');$a.Add($s2Main);$a.Add('--output');$a.Add($s3)
            foreach($x in (Get-StageArgs 'S3' $Schema $Cfg)){ $a.Add($x) }
            Run-Python 'S3/7 活动区识别' $a.ToArray()
            Assert-File $s3Main 'S3 road_candidate_points.csv'
            Write-StageMarker $root 'S3' $fps['S3'] @($s3Main)
        }

        # --------------------------------------------------------------
        # S4
        # --------------------------------------------------------------
        $s4Main=Join-Path $s4 'all_cleaned_points.csv'
        $gravityReport=Join-Path $s4 'gravity_uturn_report.json'
        if(Should-RunStage 'S4' 4 $fps['S4'] @($s4Main,$gravityReport) $Cfg $root){
            Assert-File $s3Main 'S3 road_candidate_points.csv'
            $a=New-Object System.Collections.Generic.List[string]
            $a.Add('S4_uturn_consensus_gravity.py');$a.Add('--input');$a.Add($s3Main);$a.Add('--output');$a.Add($s4)
            foreach($x in (Get-StageArgs 'S4' $Schema $Cfg)){ $a.Add($x) }
            Run-Python 'S4/7 U形弯共识引力' $a.ToArray()
            Assert-File $s4Main 'S4 all_cleaned_points.csv'
            Assert-File $gravityReport 'S4 gravity report'
            Write-StageMarker $root 'S4' $fps['S4'] @($s4Main,$gravityReport)
        }

        # --------------------------------------------------------------
        # S5
        # --------------------------------------------------------------
        $kdeReportPath=Join-Path $s5 'directional_kde_report.json'
        $runS5=Should-RunStage 'S5' 5 $fps['S5'] @($kdeReportPath) $Cfg $root
        if($runS5){
            Assert-File $s4Main 'S4 all_cleaned_points.csv'
            $a=New-Object System.Collections.Generic.List[string]
            $a.Add('S5_directional_kde.py');$a.Add('--input');$a.Add($s4Main);$a.Add('--output');$a.Add($s5)
            if ([bool]$Cfg.pipeline.auto_align_kde_to_gravity) { Assert-File $gravityReport 'S4 gravity report';$a.Add('--gravity-report');$a.Add($gravityReport) }
            foreach($x in (Get-StageArgs 'S5' $Schema $Cfg)){ $a.Add($x) }
            Run-Python 'S5/7 方向各向异性KDE' $a.ToArray()
        }

        Assert-File $kdeReportPath 'S5 KDE report'
        $krep=Read-JsonFile $kdeReportPath
        $kdeFloat=[string]$krep.float32;$support=[string]$krep.support;$distance=[string]$krep.distance;$rawUnion=[string]$krep.raw_union
        Assert-File $kdeFloat 'KDE float32';Assert-File $support 'support';Assert-File $distance 'distance';Assert-File $rawUnion 'raw_union'
        $s5Rasters=@($kdeFloat,$support,$distance,$rawUnion)

        # Full read validation is intentionally placed before S6.
        if([bool](Get-PropValue $Cfg.runtime 'validate_s5_rasters')){
            if(-not (Test-Rasters $s5Rasters 1 ([int]$Cfg.runtime.io_retry_wait_s))){
                if(-not $runS5 -and (Get-StartStageNumber $Cfg) -eq 0){
                    Write-Host '[RESUME] 已有 S5 栅格校验失败，自动从 S5 重新生成。' -ForegroundColor Yellow
                    $script:ForceDownstream=$true
                    $a=New-Object System.Collections.Generic.List[string]
                    $a.Add('S5_directional_kde.py');$a.Add('--input');$a.Add($s4Main);$a.Add('--output');$a.Add($s5)
                    if ([bool]$Cfg.pipeline.auto_align_kde_to_gravity) {$a.Add('--gravity-report');$a.Add($gravityReport)}
                    foreach($x in (Get-StageArgs 'S5' $Schema $Cfg)){ $a.Add($x) }
                    Run-Python 'S5/7 方向各向异性KDE（校验失败后重建）' $a.ToArray()
                    $krep=Read-JsonFile $kdeReportPath
                    $kdeFloat=[string]$krep.float32;$support=[string]$krep.support;$distance=[string]$krep.distance;$rawUnion=[string]$krep.raw_union
                    $s5Rasters=@($kdeFloat,$support,$distance,$rawUnion)
                    if(-not (Test-Rasters $s5Rasters 1 ([int]$Cfg.runtime.io_retry_wait_s))){throw 'S5 重建后栅格仍无法完整读取。请检查磁盘/文件系统。'}
                    $runS5=$true
                } else {
                    throw 'S5 栅格无法完整读取。当前手动开始阶段不允许自动回退，请将“开始阶段”改为 S5 或 auto 后重试。'
                }
            }
        }
        if($runS5){Write-StageMarker $root 'S5' $fps['S5'] @($kdeReportPath,$kdeFloat,$support,$distance,$rawUnion)}

        # --------------------------------------------------------------
        # S6
        # --------------------------------------------------------------
        $primary=Join-Path $s6 'skeleton_primary_route_locked.tif'
        if(Should-RunStage 'S6' 6 $fps['S6'] @($primary) $Cfg $root){
            Assert-File $s4Main 'S4 all_cleaned_points.csv'
            $a=New-Object System.Collections.Generic.List[string]
            $a.Add('S6_route_locked_skeleton.py');$a.Add('--kde');$a.Add($kdeFloat);$a.Add('--support');$a.Add($support);$a.Add('--distance');$a.Add($distance);$a.Add('--raw-union');$a.Add($rawUnion);$a.Add('--gravity-points');$a.Add($s4Main);$a.Add('--output');$a.Add($s6)
            $uturn=Join-Path $s4 'uturn_consensus.gpkg';if(Test-Path -LiteralPath $uturn){$a.Add('--uturn-gpkg');$a.Add($uturn)}
            $activity=Join-Path $s3 'activity_zones.gpkg';if(Test-Path -LiteralPath $activity){$a.Add('--activity-zones');$a.Add($activity)}
            foreach($x in (Get-StageArgs 'S6' $Schema $Cfg)){ $a.Add($x) }
            Run-PythonWithRasterRetry 'S6/7 KDE脊线约束路径锁定骨架' $a.ToArray() $s5Rasters $Cfg
            Assert-File $primary 'S6 primary skeleton'
            Write-StageMarker $root 'S6' $fps['S6'] @($primary)
        }

        # --------------------------------------------------------------
        # S7
        # --------------------------------------------------------------
        $final=Join-Path $s7 '08_final_clean_network\road_network_final_clean.gpkg'
        if(Should-RunStage 'S7' 7 $fps['S7'] @($final) $Cfg $root){
            Assert-File $primary 'S6 primary skeleton'
            if([bool](Get-PropValue $Cfg.runtime 'validate_s5_rasters')){
                if(-not (Test-Rasters $s5Rasters 1 ([int]$Cfg.runtime.io_retry_wait_s))){throw '进入 S7 前 S5 栅格复检失败。'}
            }
            $a=New-Object System.Collections.Generic.List[string]
            $a.Add('-u');$a.Add('S7_network_cleanup.py');$a.Add('--primary');$a.Add($primary);$a.Add('--support');$a.Add($support);$a.Add('--kde');$a.Add($kdeFloat);$a.Add('--distance');$a.Add($distance);$a.Add('--gravity-points');$a.Add($s4Main);$a.Add('--raw-tracks');$a.Add($s1Tracks);$a.Add('--tile-report');$a.Add($kdeReportPath);$a.Add('--output');$a.Add($s7)
            $aux=Join-Path $s6 'skeleton_auxiliary_singletrack_safe.tif';if(Test-Path -LiteralPath $aux){$a.Add('--auxiliary');$a.Add($aux)}
            $resc=Join-Path $s6 'skeleton_support_rescued.tif';if(Test-Path -LiteralPath $resc){$a.Add('--rescued');$a.Add($resc)}
            $uturn=Join-Path $s4 'uturn_consensus.gpkg';if(Test-Path -LiteralPath $uturn){$a.Add('--uturn-gpkg');$a.Add($uturn)}
            $activity=Join-Path $s3 'activity_zones.gpkg';if(Test-Path -LiteralPath $activity){$a.Add('--activity-zones');$a.Add($activity)}
            if(-not [string]::IsNullOrWhiteSpace([string]$Cfg.paths.dem)){$a.Add('--dem');$a.Add([string]$Cfg.paths.dem)}
            foreach($x in (Get-StageArgs 'S7' $Schema $Cfg)){ $a.Add($x) }
            Run-PythonWithRasterRetry 'S7/7 路网清理与矢量化' $a.ToArray() $s5Rasters $Cfg
            Assert-File $final 'S7 final network'
            Write-StageMarker $root 'S7' $fps['S7'] @($final)
        }

        if ([bool]$Cfg.pipeline.save_run_config_to_output) { Write-JsonFile $Cfg (Join-Path $root 'pipeline_run_config.json') }

        Write-Host ''
        Write-Host '============================================================' -ForegroundColor Green
        $completeText = if($Mode -eq 'S2S7'){'S2-S7 全流程完成。'}else{'S1-S7 全流程完成。'}
        Write-Host $completeText -ForegroundColor Green
        Write-Host ('最终路网: '+$final) -ForegroundColor Green
        Write-Host ('续跑状态目录: '+(Get-StateDir $root))
        Write-Host ('本次参数: '+(Join-Path $root 'pipeline_run_config.json'))
        Write-Host '============================================================' -ForegroundColor Green
    } finally { Pop-Location }
}

try {
    $schema=Read-JsonFile $script:SchemaPath
    $cfg=if(Test-Path -LiteralPath $script:LastConfigPath){Read-JsonFile $script:LastConfigPath}else{Read-JsonFile $script:DefaultConfigPath}
    $script:CurrentConfig=$cfg
    if(-not (Show-Configurator $schema $cfg)){exit 0}
    Run-Pipeline $schema $script:CurrentConfig
    exit 0
} catch {
    $msg=$_.Exception.Message
    Write-Host '';Write-Host ('ERROR: '+$msg) -ForegroundColor Red
    Show-Message $msg '运行失败' ([System.Windows.Forms.MessageBoxIcon]::Error)
    exit 1
}
