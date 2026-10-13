@echo off
REM ==============================================================================
REM Prototypical Head Ablation Study Windows Batch Runner
REM Usage:
REM   run_all_ablations.bat          -> Runs all suites
REM   run_all_ablations.bat distance -> Runs distance suite
REM   run_all_ablations.bat kshot    -> Runs k-shot sweep
REM ==============================================================================

set SUITE=%1
if "%SUITE%"=="" set SUITE=all

echo ==============================================================================
echo Running Prototypical Head Ablation Suite: %SUITE%
echo ==============================================================================

cd ..\..\..
python -u scripts\run_proto_ablations.py --suite %SUITE%
pause
