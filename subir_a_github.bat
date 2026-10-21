@echo off
title Subir GravTT a GitHub
cd /d "%~dp0"
echo ========================================================
echo          SUBIENDO GRAVTT A TU REPOSITORIO GITHUB
echo ========================================================
echo.
echo Repositorio destino: https://github.com/tomastosello03-hub/GravTT.git
echo.
git push -u origin main
echo.
if errorlevel 1 (
    echo ========================================================
    echo  Si es la primera vez que subes a GitHub desde esta PC,
    echo  inicia sesion en la ventana emergente o pega tu Token.
    echo ========================================================
) else (
    echo ========================================================
    echo   FELICITACIONES! GravTT fue subido con exito a GitHub!
    echo ========================================================
)
echo.
pause
