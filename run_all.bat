@echo off
REM One-click launcher for the autonomous experiment pipeline.
REM Safe to re-run: resumes from the last completed step via state file.
cd /d "%~dp0copdph-gcn-repo"
python auto_pipeline.py %*
pause
