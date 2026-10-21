@echo off
title Compilar GravTT.exe
cd /d "%~dp0"
echo ============================================================
echo   Compilando ejecutable nativo GravTT.exe con logo embebido
echo ============================================================
pyinstaller --noconfirm --onedir --windowed --icon=assets\icon.ico --add-data "assets;assets" --name=GravTT GravTT.py
echo.
echo ============================================================
echo   Compilacion finalizada con exito!
echo   El ejecutable con icono se encuentra en: dist\GravTT\GravTT.exe
echo ============================================================
pause
