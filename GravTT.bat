@echo off
title GravTT
cd /d "%~dp0"
python GravTT.py
if errorlevel 1 pause
