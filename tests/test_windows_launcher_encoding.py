from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_cmd_launchers_are_utf8_without_bom_and_start_with_echo_off():
    launchers = sorted(ROOT.glob("run_*_windows.cmd"))
    assert launchers, "No Windows CMD launchers found"

    for launcher in launchers:
        data = launcher.read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf"), f"{launcher.name} contains UTF-8 BOM"
        assert data.startswith(b"@echo off"), f"{launcher.name} must begin exactly with '@echo off'"
        assert b"\r\n" in data, f"{launcher.name} must use Windows CRLF line endings"


def test_windows_powershell_launcher_is_utf8_with_bom_for_windows_powershell_51():
    launcher = ROOT / "code" / "tools" / "run_pipeline_configurable.ps1"
    data = launcher.read_bytes()

    assert data.startswith(b"\xef\xbb\xbf"), (
        "run_pipeline_configurable.ps1 contains Chinese UI strings and must be UTF-8 with BOM "
        "so Windows PowerShell 5.1 does not parse it using the legacy ANSI code page"
    )
    text = data.decode("utf-8-sig")
    assert "Activity-Aware Pipeline" in text
    assert "未找到 Python" in text
    assert "请选择" in text
