param([switch]$Launch, [switch]$Library, [string]$Task, [ValidateSet('run','plan')] [string]$Action='run')
$ErrorActionPreference = 'Stop'
[Threading.Thread]::CurrentThread.CurrentUICulture = [Globalization.CultureInfo]::GetCultureInfo('en-US')
[Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::GetCultureInfo('en-US')
if (-not ('TrailRuntimeUiLanguage' -as [type])) {
    Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class TrailRuntimeUiLanguage { [DllImport("kernel32.dll")] public static extern ushort SetThreadUILanguage(ushort language); }'
}
[TrailRuntimeUiLanguage]::SetThreadUILanguage(0x0409) | Out-Null
$script:Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
$script:BindingBase = 'Chicago/Evaluation/Runtime'

function Get-EnglishText {
    param([AllowEmptyString()][string]$Text)
    $clean=[regex]::Replace($Text,'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','')
    return [regex]::Replace($clean,'[^\x00-\x7F]',{param($match) '\u'+([int][char]$match.Value).ToString('x4')})
}

function Open-SharedRead {
    param([string]$Filename)
    # MATLAB and Python publish files by replacing them. Readers must also
    # share DELETE access on Windows, including while a stream remains open.
    return [IO.File]::Open($Filename,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
}

function Read-SharedText {
    param([string]$Filename)
    $stream=Open-SharedRead $Filename
    $reader=$null
    try {
        $reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$true)
        return $reader.ReadToEnd()
    } finally {
        if ($reader) { $reader.Dispose() } else { $stream.Dispose() }
    }
}

function Read-Configuration {
    param([string]$Root)
    $cfg = Read-SharedText (Join-Path $Root 'settings.json') | ConvertFrom-Json
    foreach ($name in @('python','matlab','java','javac')) {
        if (-not $cfg.tools.PSObject.Properties[$name]) {
            $cfg.tools | Add-Member -NotePropertyName $name -NotePropertyValue ''
        }
    }
    return $cfg
}

function Get-ConfiguredLocation {
    param([string]$Root, [string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    if ([IO.Path]::IsPathRooted($Value) -or $Value -match '^[A-Za-z]:') {
        throw 'Enter a relative location, or select a location with Browse.'
    }
    $full=[IO.Path]::GetFullPath((Join-Path $Root $Value))
    if ([IO.Path]::GetExtension($full) -ieq '.lnk' -and (Test-Path -LiteralPath $full -PathType Leaf)) {
        $shell=New-Object -ComObject WScript.Shell
        return $shell.CreateShortcut($full).TargetPath
    }
    return $full
}

function Get-RelativeLocation {
    param([string]$Root, [string]$Selected)
    $rootPath=[IO.Path]::GetFullPath($Root).TrimEnd('\','/')
    $selectedPath=[IO.Path]::GetFullPath($Selected).TrimEnd('\','/')
    if ($rootPath -ieq $selectedPath) { return '.' }
    if (-not $selectedPath.StartsWith($rootPath+'\',[StringComparison]::OrdinalIgnoreCase)) { return $null }
    $base=[Uri]($rootPath+'\')
    return [Uri]::UnescapeDataString($base.MakeRelativeUri([Uri]$selectedPath).ToString())
}

function Set-LocalShortcut {
    param([string]$Root, [string]$Relative, [string]$Target)
    $rootPath=[IO.Path]::GetFullPath($Root).TrimEnd('\','/')
    $link=[IO.Path]::GetFullPath((Join-Path $rootPath $Relative))
    if (-not $link.StartsWith($rootPath+'\',[StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetExtension($link) -ine '.lnk') { throw 'The local entry must be a shortcut inside the package.' }
    if (-not (Test-Path -LiteralPath $Target)) { throw 'Select an existing file or directory.' }
    New-Item -ItemType Directory -Path (Split-Path $link -Parent) -Force | Out-Null
    $shell=New-Object -ComObject WScript.Shell
    $shortcut=$shell.CreateShortcut($link)
    $shortcut.TargetPath=[IO.Path]::GetFullPath($Target)
    if (Test-Path -LiteralPath $Target -PathType Container) { $shortcut.WorkingDirectory=$Target }
    else { $shortcut.WorkingDirectory=Split-Path $Target -Parent }
    $shortcut.Save()
}

function Get-RuntimeHostId {
    $base=[Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine,[Microsoft.Win32.RegistryView]::Registry64)
    try {
        $key=$base.OpenSubKey('SOFTWARE\Microsoft\Cryptography')
        try { $machine=[string]$key.GetValue('MachineGuid') } finally { $key.Dispose() }
    } finally { $base.Dispose() }
    $identity=($env:COMPUTERNAME+'|'+$env:USERNAME+'|'+$machine).ToLowerInvariant()
    $sha=[Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($identity)))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Test-ActiveRun {
    param([string]$Root, $Configuration)
    $results = Get-ConfiguredLocation $Root $Configuration.paths.chicago_results
    if (-not $results) { return $false }
    $statusFile = Join-Path $results 'Chicago_Runtime/.workflow/status.json'
    if (-not (Test-Path -LiteralPath $statusFile)) { return $false }
    $state = Read-SharedText $statusFile | ConvertFrom-Json
    if ($state.host_id -ne (Get-RuntimeHostId)) { return $false }
    if ($state.state -notin @('running','starting','pausing','paused','resuming') -or -not $state.pid) { return $false }
    $process=Get-Process -Id $state.pid -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    $expected=$state.process_created;$tolerance=0.02
    if ($null -eq $expected) { $expected=$state.started;$tolerance=60 }
    if ($null -ne $expected) {
        try {
            $created=([DateTimeOffset]$process.StartTime).ToUnixTimeMilliseconds()/1000.0
            return ([Math]::Abs($created-[double]$expected) -le $tolerance)
        } catch { return $false }
    }
    return $true
}

function Save-Configuration {
    param([string]$Root, $Configuration, [hashtable]$Selections = @{})
    if (Test-ActiveRun $Root (Read-Configuration $Root)) {
        throw 'Configuration is available after the active run finishes.'
    }
    foreach ($property in $Configuration.paths.PSObject.Properties) {
        if ([string]::IsNullOrWhiteSpace($property.Value)) { throw "Select $($property.Name)." }
        $resolved=Get-ConfiguredLocation $Root $property.Value
        if ($property.Name.EndsWith('_results') -and $null -eq (Get-RelativeLocation $Root $resolved)) {
            throw 'Select a results folder inside this package.'
        }
    }
    foreach ($property in $Configuration.tools.PSObject.Properties) {
        $null = Get-ConfiguredLocation $Root $property.Value
    }
    foreach ($name in $Selections.Keys) {
        $entry = $Selections[$name]
        if (-not (Test-Path -LiteralPath $entry.Target)) { throw "Directory not found: $name" }
    }
    foreach ($name in $Selections.Keys) {
        $entry = $Selections[$name]
        Set-LocalShortcut $Root $entry.Relative $entry.Target
    }
    foreach ($property in $Configuration.tools.PSObject.Properties) {
        if ($property.Value) {
            $exe = Get-ConfiguredLocation $Root $property.Value
            if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Select $($property.Name) with Browse." }
        }
    }
    $destination = Join-Path $Root 'settings.json'
    $temporary = $destination + '.writing'
    [IO.File]::WriteAllText($temporary, ($Configuration | ConvertTo-Json -Depth 8) + "`n", (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $destination -Force
}

function Get-SelectedPython {
    param([string]$Root, $Configuration)
    $exe = Get-ConfiguredLocation $Root $Configuration.tools.python
    if (-not $exe -or -not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw 'Select Python 3.11 or newer with Browse.'
    }
    return $exe
}


function Set-SoftwareSelection {
    param($Configuration, [hashtable]$Selections, [ValidateSet('python','matlab','jdk')] [string]$Name, [string]$Selected, [string]$Root=$script:Root)
    if ($Name -eq 'jdk') {
        $directory=[IO.Path]::GetFullPath($Selected)
        if ((Split-Path $directory -Leaf) -ieq 'bin') { $directory=Split-Path $directory -Parent }
        foreach ($engine in @('java','javac')) {
            $target=Join-Path $directory ('bin/'+$engine+'.exe')
            if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw 'Select a JDK directory containing bin/java.exe and bin/javac.exe.' }
        }
        foreach ($engine in @('java','javac')) {
            $target=Join-Path $directory ('bin/'+$engine+'.exe')
            $entry=Get-RelativeLocation $Root $target
            if ($null -eq $entry) {
                $entry=$script:BindingBase+'/Engines/JDK/'+$engine+'.lnk'
                $Selections[$engine]=@{Relative=$entry;Target=$target}
            } else { $Selections.Remove($engine) }
            $Configuration.tools.$engine=$entry
        }
    } else {
        if (-not (Test-Path -LiteralPath $Selected -PathType Leaf) -or [IO.Path]::GetFileName($Selected) -ine ($Name+'.exe')) { throw ('Select '+$Name+'.exe.') }
        if ($Name -eq 'matlab') {
            $installation=Split-Path (Split-Path $Selected -Parent) -Parent
            if (-not (Test-Path -LiteralPath (Join-Path $installation 'bin/matlab.exe'))) { throw 'Select MATLAB bin/matlab.exe.' }
            $entry=$script:BindingBase+'/Engines/MATLAB.lnk'
        } else { $entry=$script:BindingBase+'/Engines/Python.lnk' }
        $internal=Get-RelativeLocation $Root $Selected
        if ($null -ne $internal) { $entry=$internal;$Selections.Remove($Name) }
        else { $Selections[$Name]=@{Relative=$entry;Target=[IO.Path]::GetFullPath($Selected)} }
        $Configuration.tools.$Name=$entry
    }
}

function Get-SoftwareDisplay {
    param([string]$Root, $Configuration, [hashtable]$Selections, [string]$Name)
    # Display the selected installation resolved from its saved local entry.
    $engine=$Name; if ($Name -eq 'jdk') { $engine='java' }
    $entry=[string]$Configuration.tools.$engine
    $target=''
    if ($Selections.ContainsKey($engine)) { $target=$Selections[$engine].Target }
    elseif ($entry) {
        $resolved=Get-ConfiguredLocation $Root $entry
        if (Test-Path -LiteralPath $resolved -PathType Leaf) { $target=$resolved }
    }
    if ($Name -eq 'jdk') {
        if ($entry) { $entry=(Split-Path $entry -Parent).Replace('\','/') }
        if ($target) {
            $target=Split-Path $target -Parent
            if ((Split-Path $target -Leaf) -ieq 'bin') { $target=Split-Path $target -Parent }
        }
    }
    return [pscustomobject]@{Selected=$target;Entry=$entry}
}

function Read-RunState {
    param([string]$Root, $Configuration)
    $results=Get-ConfiguredLocation $Root $Configuration.paths.chicago_results
    $file=Join-Path $results 'Chicago_Runtime/.workflow/status.json'
    if (-not (Test-Path -LiteralPath $file)) { return [pscustomobject]@{state='ready';completed=0;total=0} }
    try { $state=Read-SharedText $file | ConvertFrom-Json }
    catch { return [pscustomobject]@{state='reading';completed=0;total=0} }
    if ($state.host_id -ne (Get-RuntimeHostId)) { return [pscustomobject]@{state='ready';completed=0;total=0} }
    # Native MATLAB output is written independently of the Python pipe. Poll it
    # even when the calculation was paused or started before this window opened.
    if ($state.task -eq 'mainling.karagiorgou' -and $state.state -in @('running','pausing','paused','resuming')) {
        $studyResults=Get-ConfiguredLocation $Root $Configuration.paths.studyarea_results
        $native=Join-Path $studyResults 'StudyArea_Intermediate/references/Karagiorgou/run/native'
        $nativeLog=Join-Path $native 'matlab_live.log'
        if (Test-Path -LiteralPath $nativeLog) {
            $state | Add-Member -NotePropertyName native_log -NotePropertyValue $nativeLog -Force
        }
        try {
            $record=Read-SharedText (Join-Path $native 'progress.json') | ConvertFrom-Json
            if ($record.phase -is [ValueType] -and $record.next_iteration -is [ValueType] -and $record.loop_total -is [ValueType] -and $record.loop_total -ge 0) {
                $total=[long]$record.loop_total
                $completed=[Math]::Min($total,[Math]::Max(0,[long]$record.next_iteration-1))
                $percent=100;if ($total -gt 0) { $percent=[int][Math]::Floor(100.0*$completed/$total) }
                $state | Add-Member -NotePropertyName progress -NotePropertyValue ([pscustomobject]@{
                    phase=('Karagiorgou phase '+$record.phase);completed=$completed;total=$total;percent=$percent
                }) -Force
            }
        } catch { } # An atomic progress-file replacement must not hide run controls.
    }
    return $state
}

function Read-LogChunk {
    param([string]$Filename, [long]$Position=0, [switch]$Tail)
    $stream=Open-SharedRead $Filename
    try {
        if ($Position -gt $stream.Length) { $Position=0 }
        if ($Tail -and $Position -eq 0) { $Position=[Math]::Max(0,$stream.Length-65536) }
        $stream.Position=$Position
        $length=[int][Math]::Min(65536,$stream.Length-$Position)
        $buffer=New-Object byte[] $length
        $count=$stream.Read($buffer,0,$length)
        return [pscustomobject]@{Text=[Text.Encoding]::UTF8.GetString($buffer,0,$count);Position=($Position+$count)}
    } finally { $stream.Dispose() }
}

function Start-WorkflowTask {
    param([string]$Root, $Configuration, [string[]]$Targets=@(), [ValidateSet('run','install','check')] [string]$Action='run')
    $python=Get-SelectedPython $Root $Configuration
    if (Test-ActiveRun $Root $Configuration) { throw 'A task is running. Its progress is shown below.' }
    foreach ($name in $Targets) { if ($name -notmatch '^[A-Za-z0-9_.]+$') { throw 'Select a named task.' } }
    $arguments='-B "'+(Join-Path $Root 'Run_Workflow.py')+'" run --action '+$Action+' '+($Targets -join ' ')
    Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $Root -WindowStyle Hidden
}

function Set-WorkflowControl {
    param([string]$Root, $Configuration, [ValidateSet('pause','resume','stop')] [string]$Command)
    $python=Get-SelectedPython $Root $Configuration
    $output=& $python -B (Join-Path $Root 'Run_Workflow.py') $Command 2>&1
    if ($LASTEXITCODE -ne 0) { throw ($output -join "`n") }
}

function Stop-Installation {
    param([string]$Root, $Configuration)
    $state=Read-RunState $Root $Configuration
    if ($state.action -ne 'install' -or $state.state -notin @('running','starting')) { throw 'No dependency installation is running.' }
    $worker=Get-CimInstance Win32_Process -Filter ("ProcessId="+[int]$state.pid) -ErrorAction SilentlyContinue
    if (-not $worker) { return }
    if ($worker.CommandLine -notmatch '-m\s+workflow\.engine\s+worker.*--action\s+install\b') { throw 'The recorded process is not an installation task.' }
    $created=([DateTimeOffset]$worker.CreationDate).ToUnixTimeSeconds()
    if ([Math]::Abs($created-[double]$state.started) -gt 60) { throw 'The recorded installation process has changed.' }
    & (Join-Path $env:SystemRoot 'System32/taskkill.exe') /PID $worker.ProcessId /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'The installation has already finished or could not be stopped.' }
    $python=Get-SelectedPython $Root $Configuration
    & $python -B (Join-Path $Root 'Run_Workflow.py') cleanup-installation ([string]$state.pid)
    if ($LASTEXITCODE -ne 0) { throw 'Installation stopped. Check the temporary installation folder shown in the log.' }
    $state.state='interrupted'; $state.current='Dependency installation stopped'
    $state | Add-Member -NotePropertyName finished -NotePropertyValue ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()) -Force
    $results=Get-ConfiguredLocation $Root $Configuration.paths.chicago_results
    $file=Join-Path $results 'Chicago_Runtime/.workflow/status.json'
    [IO.File]::WriteAllText($file,($state | ConvertTo-Json -Depth 10),(New-Object Text.UTF8Encoding($false)))
}

if ($Library) { return }
$env:PYTHONUTF8='1';$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONNOUSERSITE='1'
if ($Task) {
    try {
        $configuration=Read-Configuration $script:Root
        $python=Get-SelectedPython $script:Root $configuration
        & $python -B (Join-Path $script:Root 'Run_Workflow.py') $Action $Task
        exit $LASTEXITCODE
    } catch {
        [Console]::Error.WriteLine('ERROR: '+(Get-EnglishText $_.Exception.Message))
        exit 1
    }
}
. (Join-Path $PSScriptRoot 'Main_Window.ps1')
try { Show-ReproductionWindow }
catch {
    Add-Type -AssemblyName System.Windows.Forms
    [Windows.Forms.MessageBox]::Show((Get-EnglishText $_.Exception.Message),'Trail Network Reconstruction') | Out-Null
    exit 1
}
