@echo off
title Subir GravTT a GitHub
cd /d "%~dp0"
echo ========================================================
echo          SUBIR GRAVTT A TU REPOSITORIO DE GITHUB
echo ========================================================
echo.
set /p REPO_URL="Pega aqui la URL de tu repositorio de GitHub (ej: https://github.com/usuario/GravTT.git): "
if "%REPO_URL%"=="" (
    echo No ingresaste ninguna URL. Operacion cancelada.
    pause
    exit /b
)

echo.
echo Enlazando con %REPO_URL%...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo Subiendo archivos a GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo Ocurrio un error al subir. Verifica tus credenciales o permisos de GitHub.
) else (
    echo.
    echo ========================================================
    echo   FELICITACIONES! GravTT fue subido exitosamente a GitHub!
    echo ========================================================
)
pause
