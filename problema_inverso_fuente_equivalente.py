# -*- coding: utf-8 -*-
"""
================================================================================
MÓDULO DE PROBLEMA INVERSO: FUENTE EQUIVALENTE (EQUIVALENT SOURCE)
================================================================================
Implementa la formulación rigurosa de fuentes equivalentes sobre topografía real
a partir de dos rasters acoplados:
  - Raster 1: Mediciones gravimétricas observadas G(x, y)
  - Raster 2: Modelo Digital de Terreno MDT Z(x, y)

Formulación matemática:
  Matrices: X, Y, Z, G de tamaño R x C.
  Vectorización: N = R * C
    Xobs = vec(X), Yobs = vec(Y), Zobs = vec(Z), Gobs = vec(G)

  Matriz de sensibilidad Psi (N x N):
    Psi_ii = 2 * pi * t * gamma
    Psi_ij = gamma * t * dx * dy * (Zobs_i - Zobs_j) / [ (Zobs_i - Zobs_j)^2 + (Yobs_i - Yobs_j)^2 + (Xobs_i - Xobs_j)^2 ]^(3/2)

  Inversión de densidad de capa:
    rho_obs = Psi^(-1) * Gobs

  Continuación a plano horizontal Z0 (Z0 > max(Zobs)):
    Psi_plano_ij = gamma * t * dx * dy * (Z0_i - Zobs_j) / [ (Z0_i - Zobs_j)^2 + (Yobs_i - Yobs_j)^2 + (Xobs_i - Xobs_j)^2 ]^(3/2)

  Gravedad resultante en el plano Z0:
    Gobs_plano = Psi_plano * rho_obs
    G_plano = reshape(Gobs_plano, (R, C))
================================================================================
"""

import sys
import os
import re
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
import scipy.linalg as sla

from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QGroupBox, QSplitter, QTabWidget,
    QProgressBar, QScrollArea, QGridLayout, QFrame, QRadioButton,
    QButtonGroup, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QColor, QPixmap

# Constantes Gravitacionales
GAMMA_DEFAULT_MGAL = 6.6743e-6   # mGal * m^2 / kg (G_SI * 1e5)
T_DEFAULT_METERS = 5.0           # Espesor por defecto solicitado por el usuario: 5 m


# ==============================================================================
# LECTURA Y RESAMPLEO DE RASTERS (GEOTIFF, ASCII GRID, SURFER, CSV)
# ==============================================================================
def read_raster_file(filepath):
    """
    Lee archivos raster en formatos GeoTIFF (.tif, .tiff), ESRI ASCII (.asc),
    Surfer ASCII (.grd) o tablas CSV/XYZ, retornando (Z_grid, x_vec, y_vec, meta).
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.tif', '.tiff']:
        img = Image.open(filepath)
        nx, ny = img.size
        tags = getattr(img, 'tag_v2', {})
        scale_tag = tags.get(33550, None)
        tie_tag = tags.get(33922, None)

        if scale_tag and tie_tag:
            dx = float(scale_tag[0])
            dy = float(scale_tag[1])
            x0 = float(tie_tag[3])
            y0 = float(tie_tag[4])
        else:
            dx = 50.0
            dy = 50.0
            x0 = 0.0
            y0 = ny * dy

        raw_arr = np.array(img, dtype=np.float64)
        mask_nodata = (raw_arr < -9000.0) | np.isnan(raw_arr)
        if np.any(mask_nodata):
            val = float(np.nanmean(raw_arr[~mask_nodata])) if np.any(~mask_nodata) else 0.0
            raw_arr[mask_nodata] = val

        x_vec = x0 + (np.arange(nx) + 0.5) * dx
        y_vec = y0 - (np.arange(ny) + 0.5) * dy
        if y_vec[0] > y_vec[-1]:
            raw_arr = np.flipud(raw_arr)
            y_vec = y_vec[::-1]

        return raw_arr, x_vec, y_vec, {'type': 'geotiff', 'dx': dx, 'dy': dy, 'unit': 'm'}

    elif ext == '.asc':
        header = {}
        with open(filepath, 'r') as f:
            for _ in range(6):
                line = f.readline().strip().split()
                if len(line) == 2:
                    header[line[0].lower()] = float(line[1])
            ncols = int(header.get('ncols', 0))
            nrows = int(header.get('nrows', 0))
            xll = header.get('xllcorner', header.get('xllcenter', 0.0))
            yll = header.get('yllcorner', header.get('yllcenter', 0.0))
            cellsize = header.get('cellsize', 1.0)
            nodata = header.get('nodata_value', -9999.0)
            data = np.loadtxt(f)
            data = np.flipud(data)
            if nodata is not None:
                mask = np.isclose(data, nodata) | (data < -9000)
                if np.any(mask):
                    data[mask] = np.nanmean(data[~mask])
            x_vec = xll + (np.arange(ncols) + 0.5) * cellsize
            y_vec = yll + (np.arange(nrows) + 0.5) * cellsize
            return data, x_vec, y_vec, {'type': 'esri_asc', 'dx': cellsize, 'dy': cellsize, 'unit': 'm'}

    elif ext == '.grd':
        with open(filepath, 'r') as f:
            tag = f.readline().strip()
            if tag == 'DSAA':
                nx, ny = map(int, f.readline().split())
                xmin, xmax = map(float, f.readline().split())
                ymin, ymax = map(float, f.readline().split())
                zmin, zmax = map(float, f.readline().split())
                data = np.fromstring(f.read(), sep=' ').reshape((ny, nx))
                x_vec = np.linspace(xmin, xmax, nx)
                y_vec = np.linspace(ymin, ymax, ny)
                dx = (xmax - xmin) / max(1, nx - 1)
                dy = (ymax - ymin) / max(1, ny - 1)
                return data, x_vec, y_vec, {'type': 'surfer_grd', 'dx': dx, 'dy': dy, 'unit': 'm'}

    elif ext in ['.csv', '.txt', '.dat', '.xyz']:
        df = pd.read_csv(filepath, sep=None, engine='python')
        cols = df.columns
        if len(cols) >= 3:
            x_raw = df.iloc[:, 0].values.astype(float)
            y_raw = df.iloc[:, 1].values.astype(float)
            z_raw = df.iloc[:, 2].values.astype(float)
            ux = np.unique(x_raw)
            uy = np.unique(y_raw)
            if len(ux) * len(uy) == len(df):
                ux.sort(); uy.sort()
                grid_z = z_raw.reshape((len(uy), len(ux)))
                dx = float(ux[1] - ux[0]) if len(ux) > 1 else 1.0
                dy = float(uy[1] - uy[0]) if len(uy) > 1 else 1.0
                return grid_z, ux, uy, {'type': 'xyz_table', 'dx': dx, 'dy': dy, 'unit': 'm'}
            else:
                raise ValueError("La tabla XYZ no corresponde a una cuadrícula regular completa.")
        else:
            # Matriz directa
            mat = df.values.astype(float)
            ny, nx = mat.shape
            x_vec = np.arange(nx, dtype=float) * 100.0
            y_vec = np.arange(ny, dtype=float) * 100.0
            return mat, x_vec, y_vec, {'type': 'matrix_csv', 'dx': 100.0, 'dy': 100.0, 'unit': 'm'}

    raise ValueError(f"Formato de archivo '{ext}' no soportado para raster.")


# ==============================================================================
# DIÁLOGO DE CONFIGURACIÓN Y PARÁMETROS PARA RASTER TIFF / GEOTIFF
# ==============================================================================
class DialogoParametrosRasterTIFF(QDialog):
    """
    Diálogo interactivo para importar y parametrizar imágenes raster TIFF / GeoTIFF
    con mediciones gravimétricas o cotas topográficas MDT.
    Permite:
      - Inspección de dimensiones, bandas y valores de píxel.
      - Detección automática de tags GeoTIFF (PixelScale y Tiepoint).
      - Configuración manual o ajustada de origen (X0, Y0), espaciado (dX, dY) o límites extremos.
      - Asignación de unidades físicas de coordenadas (m o km) y atributo Z (mGal, µGal, g.u., m/s², m).
      - Tratamiento y sustitución de valores NoData / nulos.
      - Inversión vertical (Flip Y) para congruencia cartesiana.
    """
    def __init__(self, filepath, is_gravity=True, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.is_gravity = is_gravity
        self.result_data = None

        self.setWindowTitle(
            "Configurar Raster TIFF: Mediciones de Gravedad" if is_gravity
            else "Configurar Raster TIFF: Topografia MDT"
        )
        self.resize(620, 640)
        self.setMinimumWidth(560)

        # Leer archivo con PIL
        self.img = Image.open(filepath)
        self.nx, self.ny = self.img.size
        raw = np.array(self.img, dtype=np.float64)
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        self.raw_arr = raw

        # Leer tags GeoTIFF si existen
        tags = getattr(self.img, 'tag_v2', {})
        scale_tag = tags.get(33550, None)
        tie_tag = tags.get(33922, None)
        nodata_tag = tags.get(42113, None)

        self.has_geotiff_tags = bool(scale_tag and tie_tag)
        if self.has_geotiff_tags:
            self.init_dx = float(scale_tag[0])
            self.init_dy = float(scale_tag[1])
            self.init_x0 = float(tie_tag[3])
            self.init_y0 = float(tie_tag[4])
        else:
            self.init_dx = 50.0
            self.init_dy = 50.0
            self.init_x0 = 0.0
            self.init_y0 = float(self.ny * self.init_dy)

        if nodata_tag is not None:
            try:
                self.init_nodata = float(nodata_tag)
            except Exception:
                self.init_nodata = -9999.0
        else:
            self.init_nodata = -9999.0

        # Estadísticas de píxeles preliminares
        mask_nodata = (self.raw_arr <= -9000.0) | np.isnan(self.raw_arr)
        if self.init_nodata is not None:
            mask_nodata |= np.isclose(self.raw_arr, self.init_nodata)
        val_clean = self.raw_arr[~mask_nodata]
        if len(val_clean) > 0:
            self.raw_min = float(np.min(val_clean))
            self.raw_max = float(np.max(val_clean))
            self.raw_mean = float(np.mean(val_clean))
        else:
            self.raw_min = float(np.nanmin(self.raw_arr))
            self.raw_max = float(np.nanmax(self.raw_arr))
            self.raw_mean = float(np.nanmean(self.raw_arr))

        self.init_ui()
        self.update_summary()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 1. Información del Raster
        grp_info = QGroupBox("1. Informacion del Archivo Raster TIFF")
        grid_info = QGridLayout(grp_info)
        grid_info.setSpacing(4)
        grid_info.addWidget(QLabel("<b>Archivo:</b>"), 0, 0)
        lbl_fn = QLabel(os.path.basename(self.filepath))
        lbl_fn.setStyleSheet("color: #003366; font-weight: bold;")
        grid_info.addWidget(lbl_fn, 0, 1)

        grid_info.addWidget(QLabel("<b>Dimensiones:</b>"), 1, 0)
        grid_info.addWidget(QLabel(f"{self.nx} columnas (Nx) x {self.ny} filas (Ny) | Total: {self.nx*self.ny:,} celdas"), 1, 1)

        grid_info.addWidget(QLabel("<b>Valores Pixel:</b>"), 2, 0)
        grid_info.addWidget(QLabel(f"Min: {self.raw_min:.2f} | Max: {self.raw_max:.2f} | Media: {self.raw_mean:.2f}"), 2, 1)

        grid_info.addWidget(QLabel("<b>Metadatos:</b>"), 3, 0)
        if self.has_geotiff_tags:
            lbl_tag = QLabel(f"GeoTIFF detectado (PixelScale dx={self.init_dx:g}, dy={self.init_dy:g}; Tiepoint x0={self.init_x0:g}, y0={self.init_y0:g})")
            lbl_tag.setStyleSheet("color: #006600; font-weight: bold;")
        else:
            lbl_tag = QLabel("TIFF estandar sin georreferenciacion interna embebida. Ajusta los parametros abajo.")
            lbl_tag.setStyleSheet("color: #b35900; font-weight: bold;")
        grid_info.addWidget(lbl_tag, 3, 1)
        layout.addWidget(grp_info)

        # 2. Georreferenciación y Extensión Espacial
        grp_geo = QGroupBox("2. Coordenadas y Georreferenciacion (X, Y)")
        grid_geo = QGridLayout(grp_geo)
        grid_geo.setSpacing(4)

        self.rb_origin_step = QRadioButton("Definir por Origen (X0, Y0) y Espaciado (dX, dY)")
        self.rb_origin_step.setChecked(True)
        self.rb_extents = QRadioButton("Definir por Limites Extremos [Xmin, Xmax, Ymin, Ymax]")
        self.bg_mode = QButtonGroup(self)
        self.bg_mode.addButton(self.rb_origin_step)
        self.bg_mode.addButton(self.rb_extents)
        self.rb_origin_step.toggled.connect(self.on_mode_toggled)

        grid_geo.addWidget(self.rb_origin_step, 0, 0, 1, 2)
        grid_geo.addWidget(self.rb_extents, 0, 2, 1, 2)

        # Origen y Paso
        grid_geo.addWidget(QLabel("Origen X (X0 / Oeste):"), 1, 0)
        self.spin_x0 = QDoubleSpinBox()
        self.spin_x0.setRange(-1e12, 1e12); self.spin_x0.setDecimals(3)
        self.spin_x0.setValue(self.init_x0)
        self.spin_x0.valueChanged.connect(self.on_param_changed)
        grid_geo.addWidget(self.spin_x0, 1, 1)

        grid_geo.addWidget(QLabel("Origen Y (Y0 / Sur):"), 1, 2)
        self.spin_y0 = QDoubleSpinBox()
        self.spin_y0.setRange(-1e12, 1e12); self.spin_y0.setDecimals(3)
        if self.has_geotiff_tags and self.init_y0 > 0:
            calc_ymin = self.init_y0 - (self.ny - 1) * self.init_dy
            self.spin_y0.setValue(calc_ymin)
        else:
            self.spin_y0.setValue(self.init_y0)
        self.spin_y0.valueChanged.connect(self.on_param_changed)
        grid_geo.addWidget(self.spin_y0, 1, 3)

        grid_geo.addWidget(QLabel("Espaciado dX (Paso X):"), 2, 0)
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(0.0001, 1e9); self.spin_dx.setDecimals(4)
        self.spin_dx.setValue(self.init_dx)
        self.spin_dx.valueChanged.connect(self.on_param_changed)
        grid_geo.addWidget(self.spin_dx, 2, 1)

        grid_geo.addWidget(QLabel("Espaciado dY (Paso Y):"), 2, 2)
        self.spin_dy = QDoubleSpinBox()
        self.spin_dy.setRange(0.0001, 1e9); self.spin_dy.setDecimals(4)
        self.spin_dy.setValue(self.init_dy)
        self.spin_dy.valueChanged.connect(self.on_param_changed)
        grid_geo.addWidget(self.spin_dy, 2, 3)

        # Límites
        grid_geo.addWidget(QLabel("X Minimo (Oeste):"), 3, 0)
        self.spin_xmin = QDoubleSpinBox(); self.spin_xmin.setRange(-1e12, 1e12); self.spin_xmin.setDecimals(3)
        self.spin_xmin.valueChanged.connect(self.on_extents_changed)
        grid_geo.addWidget(self.spin_xmin, 3, 1)

        grid_geo.addWidget(QLabel("X Maximo (Este):"), 3, 2)
        self.spin_xmax = QDoubleSpinBox(); self.spin_xmax.setRange(-1e12, 1e12); self.spin_xmax.setDecimals(3)
        self.spin_xmax.valueChanged.connect(self.on_extents_changed)
        grid_geo.addWidget(self.spin_xmax, 3, 3)

        grid_geo.addWidget(QLabel("Y Minimo (Sur):"), 4, 0)
        self.spin_ymin = QDoubleSpinBox(); self.spin_ymin.setRange(-1e12, 1e12); self.spin_ymin.setDecimals(3)
        self.spin_ymin.valueChanged.connect(self.on_extents_changed)
        grid_geo.addWidget(self.spin_ymin, 4, 1)

        grid_geo.addWidget(QLabel("Y Maximo (Norte):"), 4, 2)
        self.spin_ymax = QDoubleSpinBox(); self.spin_ymax.setRange(-1e12, 1e12); self.spin_ymax.setDecimals(3)
        self.spin_ymax.valueChanged.connect(self.on_extents_changed)
        grid_geo.addWidget(self.spin_ymax, 4, 3)

        # Unidad
        grid_geo.addWidget(QLabel("<b>Unidad de Coordenadas:</b>"), 5, 0)
        self.combo_unit_xy = QComboBox()
        self.combo_unit_xy.addItem("Metros [m]", "m")
        self.combo_unit_xy.addItem("Kilometros [km]", "km")
        self.combo_unit_xy.currentIndexChanged.connect(self.update_summary)
        grid_geo.addWidget(self.combo_unit_xy, 5, 1)

        self.btn_reset_tags = QPushButton("Restablecer a Metadatos GeoTIFF")
        self.btn_reset_tags.setEnabled(self.has_geotiff_tags)
        self.btn_reset_tags.clicked.connect(self.reset_to_geotiff_tags)
        grid_geo.addWidget(self.btn_reset_tags, 5, 2, 1, 2)

        layout.addWidget(grp_geo)

        # 3. Parámetros Físicos del Atributo Z
        grp_z = QGroupBox("3. Parametros Fisicos del Atributo Z (Gravedad / Elevacion)")
        grid_z = QGridLayout(grp_z)
        grid_z.setSpacing(4)

        grid_z.addWidget(QLabel("<b>Unidad en el Archivo:</b>"), 0, 0)
        self.combo_unit_z = QComboBox()
        if self.is_gravity:
            self.combo_unit_z.addItem("mGal (MiliGal) [factor x 1.0]", 1.0)
            self.combo_unit_z.addItem("uGal (MicroGal) [factor x 0.001]", 0.001)
            self.combo_unit_z.addItem("g.u. (Unidades Gravimetricas) [factor x 0.1]", 0.1)
            self.combo_unit_z.addItem("m/s2 (SI) [factor x 100,000]", 100000.0)
            self.combo_unit_z.addItem("cm/s2 (Gal) [factor x 1,000]", 1000.0)
            self.combo_unit_z.addItem("Personalizado...", "custom")
        else:
            self.combo_unit_z.addItem("Metros [m] [factor x 1.0]", 1.0)
            self.combo_unit_z.addItem("Kilometros [km] [factor x 1,000]", 1000.0)
            self.combo_unit_z.addItem("Pies [ft] [factor x 0.3048]", 0.3048)
            self.combo_unit_z.addItem("Personalizado...", "custom")
        self.combo_unit_z.currentIndexChanged.connect(self.on_unit_z_changed)
        grid_z.addWidget(self.combo_unit_z, 0, 1)

        grid_z.addWidget(QLabel("Factor Multiplicador Manual:"), 0, 2)
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(-1e9, 1e9); self.spin_scale.setDecimals(6)
        self.spin_scale.setValue(1.0)
        self.spin_scale.valueChanged.connect(self.update_summary)
        grid_z.addWidget(self.spin_scale, 0, 3)

        grid_z.addWidget(QLabel("<b>Filtro NoData / Nulo:</b>"), 1, 0)
        self.chk_has_nodata = QCheckBox("Tratar valores como NoData:")
        self.chk_has_nodata.setChecked(True)
        self.chk_has_nodata.stateChanged.connect(self.update_summary)
        grid_z.addWidget(self.chk_has_nodata, 1, 1)

        self.spin_nodata = QDoubleSpinBox()
        self.spin_nodata.setRange(-1e12, 1e12); self.spin_nodata.setDecimals(2)
        self.spin_nodata.setValue(self.init_nodata)
        self.spin_nodata.valueChanged.connect(self.update_summary)
        grid_z.addWidget(self.spin_nodata, 1, 2)

        self.combo_nodata_action = QComboBox()
        self.combo_nodata_action.addItem("Rellenar con promedio de validos", "mean")
        self.combo_nodata_action.addItem("Rellenar con 0.0", "zero")
        self.combo_nodata_action.addItem("Rellenar con valor minimo", "min")
        grid_z.addWidget(self.combo_nodata_action, 1, 3)

        self.chk_flip_y = QCheckBox("Invertir orientacion vertical (Fila 0 = Norte / Esquina Superior)")
        self.chk_flip_y.setChecked(True)
        self.chk_flip_y.setToolTip("En imagenes TIFF estandar la fila 0 esta arriba (Norte). Marca esta casilla para alinear con el eje Y cartesiano.")
        grid_z.addWidget(self.chk_flip_y, 2, 0, 1, 4)

        layout.addWidget(grp_z)

        # 4. Resumen
        grp_sum = QGroupBox("4. Resumen del Raster Resultante")
        v_sum = QVBoxLayout(grp_sum)
        self.lbl_summary = QLabel("Calculando resumen...")
        self.lbl_summary.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; background-color: #fafafa; border: 1px solid #d0d0d0; padding: 6px;")
        v_sum.addWidget(self.lbl_summary)
        layout.addWidget(grp_sum)

        # Botones
        row_btn = QHBoxLayout()
        row_btn.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        row_btn.addWidget(btn_cancel)

        btn_ok = QPushButton("Aceptar y Cargar Raster" if not self.is_gravity else "Aceptar y Cargar Raster de Gravedad")
        btn_ok.setStyleSheet("background-color: #006600; color: #ffffff; font-weight: bold; font-size: 12px; padding: 6px 14px;")
        btn_ok.clicked.connect(self.accept)
        row_btn.addWidget(btn_ok)
        layout.addLayout(row_btn)

        self.on_mode_toggled()
        self.sync_from_origin_step()

    def on_mode_toggled(self):
        use_origin = self.rb_origin_step.isChecked()
        self.spin_x0.setEnabled(use_origin)
        self.spin_y0.setEnabled(use_origin)
        self.spin_dx.setEnabled(use_origin)
        self.spin_dy.setEnabled(use_origin)

        self.spin_xmin.setEnabled(not use_origin)
        self.spin_xmax.setEnabled(not use_origin)
        self.spin_ymin.setEnabled(not use_origin)
        self.spin_ymax.setEnabled(not use_origin)

    def sync_from_origin_step(self):
        x0 = self.spin_x0.value()
        y0 = self.spin_y0.value()
        dx = self.spin_dx.value()
        dy = self.spin_dy.value()

        xmin = x0
        xmax = x0 + (self.nx - 1) * dx
        ymin = y0
        ymax = y0 + (self.ny - 1) * dy

        self.spin_xmin.blockSignals(True)
        self.spin_xmax.blockSignals(True)
        self.spin_ymin.blockSignals(True)
        self.spin_ymax.blockSignals(True)
        self.spin_xmin.setValue(xmin)
        self.spin_xmax.setValue(xmax)
        self.spin_ymin.setValue(ymin)
        self.spin_ymax.setValue(ymax)
        self.spin_xmin.blockSignals(False)
        self.spin_xmax.blockSignals(False)
        self.spin_ymin.blockSignals(False)
        self.spin_ymax.blockSignals(False)

    def on_param_changed(self):
        if self.rb_origin_step.isChecked():
            self.sync_from_origin_step()
        self.update_summary()

    def on_extents_changed(self):
        if self.rb_extents.isChecked():
            xmin = self.spin_xmin.value()
            xmax = self.spin_xmax.value()
            ymin = self.spin_ymin.value()
            ymax = self.spin_ymax.value()

            dx = abs(xmax - xmin) / max(1, self.nx - 1)
            dy = abs(ymax - ymin) / max(1, self.ny - 1)

            self.spin_x0.blockSignals(True)
            self.spin_y0.blockSignals(True)
            self.spin_dx.blockSignals(True)
            self.spin_dy.blockSignals(True)
            self.spin_x0.setValue(xmin)
            self.spin_y0.setValue(ymin)
            self.spin_dx.setValue(dx)
            self.spin_dy.setValue(dy)
            self.spin_x0.blockSignals(False)
            self.spin_y0.blockSignals(False)
            self.spin_dx.blockSignals(False)
            self.spin_dy.blockSignals(False)

        self.update_summary()

    def on_unit_z_changed(self):
        val = self.combo_unit_z.currentData()
        if val == 'custom':
            self.spin_scale.setEnabled(True)
        else:
            self.spin_scale.setValue(float(val))
        self.update_summary()

    def reset_to_geotiff_tags(self):
        self.spin_x0.setValue(self.init_x0)
        calc_ymin = self.init_y0 - (self.ny - 1) * self.init_dy if self.init_y0 > 0 else self.init_y0
        self.spin_y0.setValue(calc_ymin)
        self.spin_dx.setValue(self.init_dx)
        self.spin_dy.setValue(self.init_dy)
        self.sync_from_origin_step()
        self.update_summary()

    def get_conversion_factor(self):
        c_val = self.combo_unit_z.currentData()
        if c_val == 'custom':
            return self.spin_scale.value()
        else:
            return float(c_val) * self.spin_scale.value()

    def get_computed_extents(self):
        xmin = self.spin_xmin.value()
        xmax = self.spin_xmax.value()
        ymin = self.spin_ymin.value()
        ymax = self.spin_ymax.value()
        dx = self.spin_dx.value()
        dy = self.spin_dy.value()
        return xmin, xmax, ymin, ymax, dx, dy

    def update_summary(self):
        xmin, xmax, ymin, ymax, dx, dy = self.get_computed_extents()
        u = self.combo_unit_xy.currentData()
        factor = self.get_conversion_factor()
        z_unit = "mGal" if self.is_gravity else "m"

        z_min_conv = self.raw_min * factor
        z_max_conv = self.raw_max * factor

        txt = (
            f"Malla: {self.nx} cols x {self.ny} filas ({self.nx*self.ny:,} nodos) | Paso: dX={dx:.3f} {u}, dY={dy:.3f} {u}\n"
            f"Extension X: [{xmin:,.2f}, {xmax:,.2f}] {u}  (Ancho: {abs(xmax-xmin):,.2f} {u})\n"
            f"Extension Y: [{ymin:,.2f}, {ymax:,.2f}] {u}  (Alto: {abs(ymax-ymin):,.2f} {u})\n"
            f"Rango Z estimado ({z_unit}): [{z_min_conv:.2f}, {z_max_conv:.2f}] {z_unit}  (Factor: x{factor:g})"
        )
        self.lbl_summary.setText(txt)

    def get_result(self):
        arr = self.raw_arr.copy()
        factor = self.get_conversion_factor()

        # NoData
        if self.chk_has_nodata.isChecked():
            nd_val = self.spin_nodata.value()
            mask_nd = np.isclose(arr, nd_val) | (arr <= -9000.0) | np.isnan(arr)
            action = self.combo_nodata_action.currentData()
            if np.any(mask_nd):
                if action == 'mean':
                    val_sub = float(np.nanmean(arr[~mask_nd])) if np.any(~mask_nd) else 0.0
                    arr[mask_nd] = val_sub
                elif action == 'zero':
                    arr[mask_nd] = 0.0
                elif action == 'min':
                    val_sub = float(np.nanmin(arr[~mask_nd])) if np.any(~mask_nd) else 0.0
                    arr[mask_nd] = val_sub

        arr = arr * factor

        if self.chk_flip_y.isChecked():
            arr = np.flipud(arr)

        xmin, xmax, ymin, ymax, dx, dy = self.get_computed_extents()
        x_vec = np.linspace(xmin, xmax, self.nx)
        y_vec = np.linspace(ymin, ymax, self.ny)

        unit_xy = self.combo_unit_xy.currentData()
        unit_z = "mGal" if self.is_gravity else "m"

        meta = {
            'type': 'geotiff',
            'dx': dx,
            'dy': dy,
            'x0': xmin,
            'y0': ymin,
            'unit_xy': unit_xy,
            'unit_z': unit_z,
            'nx': self.nx,
            'ny': self.ny,
            'filepath': self.filepath
        }
        return arr, x_vec, y_vec, meta


def extract_mdt_at_gravity_pixels(grav_x, grav_y, mdt_Z=None, mdt_x=None, mdt_y=None, mdt_obj=None):
    """
    Extrae la coordenada Z del MDT sobrepuesto en el raster de gravedad,
    evaluando la cota exactamente en el centro de cada píxel de gravedad.

    La matriz de gravedad rige SIEMPRE el proceso matricial:
      - R = len(grav_y) filas
      - C = len(grav_x) columnas
      - Coordenadas de los centros de píxel: (grav_x[c], grav_y[r])
      - Retorno: matriz Z_matched de dimensiones estrictamente (R, C).
    """
    R = len(grav_y)
    C = len(grav_x)
    Xc, Yc = np.meshgrid(grav_x, grav_y)

    # 1. Caso Objeto GeoTiffMDT georreferenciado
    if mdt_obj is not None and getattr(mdt_obj, 'is_loaded', False) and mdt_obj.grid_z is not None:
        x_min = min(mdt_obj.x0, mdt_obj.x1)
        x_max = max(mdt_obj.x0, mdt_obj.x1)
        y_min = min(mdt_obj.y0, mdt_obj.y1)
        y_max = max(mdt_obj.y0, mdt_obj.y1)

        gx_min, gx_max = float(np.min(grav_x)), float(np.max(grav_x))
        gy_min, gy_max = float(np.min(grav_y)), float(np.max(grav_y))

        overlap_x = (gx_min <= x_max) and (gx_max >= x_min)
        overlap_y = (gy_min <= y_max) and (gy_max >= y_min)

        if overlap_x and overlap_y:
            # Extracción georreferenciada en el centro de cada celda del raster de gravedad
            col = np.int64(np.floor((Xc - mdt_obj.x0) / mdt_obj.dx))
            row = np.int64(np.floor((mdt_obj.y0 - Yc) / mdt_obj.dy))
            col_clip = np.clip(col, 0, mdt_obj.nx - 1)
            row_clip = np.clip(row, 0, mdt_obj.ny - 1)
            return mdt_obj.grid_z[row_clip, col_clip].astype(np.float64)
        else:
            # Coordenadas locales o sintéticas: mapeo proporcional normalizado [0, 1]
            x_norm = (Xc - gx_min) / max(1e-9, gx_max - gx_min)
            y_norm = (Yc - gy_min) / max(1e-9, gy_max - gy_min)
            c_idx = np.clip(np.int64(np.floor(x_norm * mdt_obj.nx)), 0, mdt_obj.nx - 1)
            r_idx = np.clip(np.int64(np.floor((1.0 - y_norm) * mdt_obj.ny)), 0, mdt_obj.ny - 1)
            return mdt_obj.grid_z[r_idx, c_idx].astype(np.float64)

    # 2. Caso matriz numérica mdt_Z con vectores de coordenadas
    if mdt_Z is not None:
        ny_m, nx_m = mdt_Z.shape
        if mdt_x is None or len(mdt_x) != nx_m:
            mdt_x = np.linspace(float(np.min(grav_x)), float(np.max(grav_x)), nx_m)
        if mdt_y is None or len(mdt_y) != ny_m:
            mdt_y = np.linspace(float(np.min(grav_y)), float(np.max(grav_y)), ny_m)

        mdt_x = np.asarray(mdt_x, dtype=float)
        mdt_y = np.asarray(mdt_y, dtype=float)

        sort_x = np.argsort(mdt_x)
        sort_y = np.argsort(mdt_y)
        sx = mdt_x[sort_x]
        sy = mdt_y[sort_y]
        sZ = np.asarray(mdt_Z, dtype=float)[np.ix_(sort_y, sort_x)]

        gx_min, gx_max = float(np.min(grav_x)), float(np.max(grav_x))
        gy_min, gy_max = float(np.min(grav_y)), float(np.max(grav_y))

        overlap_x = (gx_min <= float(np.max(sx))) and (gx_max >= float(np.min(sx)))
        overlap_y = (gy_min <= float(np.max(sy))) and (gy_max >= float(np.min(sy)))

        if overlap_x and overlap_y:
            # Interpolación RegularGrid con method='nearest': valor del MDT en el centro de pixel
            rgi = RegularGridInterpolator((sy, sx), sZ, method='nearest', bounds_error=False, fill_value=None)
            pts = np.column_stack([Yc.ravel(), Xc.ravel()])
            return rgi(pts).reshape((R, C))
        else:
            x_norm = (Xc - gx_min) / max(1e-9, gx_max - gx_min)
            y_norm = (Yc - gy_min) / max(1e-9, gy_max - gy_min)
            col = np.clip(np.int64(np.floor(x_norm * len(sx))), 0, len(sx) - 1)
            row = np.clip(np.int64(np.floor(y_norm * len(sy))), 0, len(sy) - 1)
            return sZ[row, col].astype(np.float64)

    return np.zeros((R, C), dtype=float)


def resample_grid_to_match(src_Z, src_x, src_y, tgt_x, tgt_y):
    """
    Interpola la cuadrícula fuente sobre los nodos exactos de la cuadrícula objetivo
    garantizando coincidencia estricta dimensión por dimensión R x C.
    """
    return extract_mdt_at_gravity_pixels(tgt_x, tgt_y, mdt_Z=src_Z, mdt_x=src_x, mdt_y=src_y)


# ==============================================================================
# HILO DE CÁLCULO EN SEGUNDO PLANO (WORKER FUENTE EQUIVALENTE)
# ==============================================================================
class WorkerFuenteEquivalente(QThread):
    sig_progress = pyqtSignal(int, str)
    sig_finished = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(self, X_m, Y_m, Z_m, G_m, Z0_m, t_m, gamma_val, dx_m, dy_m, damping=0.0, decimation_step='auto', interp_full=True):
        super().__init__()
        self.X_m = X_m
        self.Y_m = Y_m
        self.Z_m = Z_m
        self.G_m = G_m
        self.Z0_m = float(Z0_m)
        self.t_m = float(t_m)
        self.gamma_val = float(gamma_val)
        self.dx_m = float(dx_m)
        self.dy_m = float(dy_m)
        self.damping = float(damping)
        self.decimation_step = decimation_step
        self.interp_full = bool(interp_full)

    def run(self):
        try:
            orig_ny, orig_nx = self.X_m.shape
            N_total = orig_ny * orig_nx

            # Verificación física fundamental de cota Z0
            z_max_obs = float(np.max(self.Z_m))
            if self.Z0_m <= z_max_obs:
                raise ValueError(
                    f"La altura del plano Z0 ({self.Z0_m:.1f} m) debe ser estrictamente mayor "
                    f"que la cota máxima del terreno ({z_max_obs:.1f} m)."
                )

            # ==================================================================
            # MODO 1: PROCESAMIENTO POR TESELAS / BLOQUES (Para mallas gigantes)
            # ==================================================================
            if str(self.decimation_step).lower() == 'tiled':
                tile_size = 40
                overlap = 12
                step_shift = tile_size - overlap

                G_accum = np.zeros((orig_ny, orig_nx), dtype=np.float64)
                rho_accum = np.zeros((orig_ny, orig_nx), dtype=np.float64)
                weight_accum = np.zeros((orig_ny, orig_nx), dtype=np.float64)

                r_starts = list(range(0, orig_ny, step_shift))
                c_starts = list(range(0, orig_nx, step_shift))
                total_tiles = len(r_starts) * len(c_starts)
                tile_count = 0

                self.sig_progress.emit(10, f"Procesando {total_tiles} teselas en bloques locales de {tile_size}×{tile_size}...")

                for r0 in r_starts:
                    r1 = min(r0 + tile_size, orig_ny)
                    for c0 in c_starts:
                        c1 = min(c0 + tile_size, orig_nx)
                        tile_count += 1

                        sub_X = self.X_m[r0:r1, c0:c1].ravel().astype(np.float32)
                        sub_Y = self.Y_m[r0:r1, c0:c1].ravel().astype(np.float32)
                        sub_Z = self.Z_m[r0:r1, c0:c1].ravel().astype(np.float32)
                        sub_G = self.G_m[r0:r1, c0:c1].ravel().astype(np.float32)

                        Nt = len(sub_X)
                        if Nt < 4:
                            continue

                        dX_t = sub_X[:, None] - sub_X[None, :]
                        dY_t = sub_Y[:, None] - sub_Y[None, :]
                        dZ_t = sub_Z[:, None] - sub_Z[None, :]
                        R2_t = dX_t**2 + dY_t**2 + dZ_t**2
                        np.fill_diagonal(R2_t, 1.0)
                        R3_t = R2_t * np.sqrt(R2_t)

                        factor_t = np.float32(self.gamma_val * self.t_m * self.dx_m * self.dy_m)
                        Psi_t = factor_t * (dZ_t / R3_t)
                        diag_t = np.float32(2.0 * np.pi * self.t_m * self.gamma_val * (1.0 + self.damping))
                        np.fill_diagonal(Psi_t, diag_t)

                        # Inversión rápida LAPACK
                        rho_t = sla.solve(Psi_t, sub_G, assume_a='gen', overwrite_a=True)

                        # Sensibilidad hacia plano Z0 local
                        dZ0_t = np.float32(self.Z0_m) - sub_Z[None, :]
                        R2_0t = dX_t**2 + dY_t**2 + (dZ0_t ** 2)
                        R3_0t = R2_0t * np.sqrt(R2_0t)
                        Psi_plano_t = factor_t * (dZ0_t / R3_0t)
                        # Diagonal fisica exacta del prisma/celda directamente debajo
                        Rc_t = np.float32(np.sqrt(self.dx_m * self.dy_m / np.pi))
                        ht = np.float32(self.Z0_m) - sub_Z
                        diag_plano_t = np.float32(2.0 * np.pi * self.gamma_val * self.t_m) * (1.0 - ht / np.sqrt(Rc_t**2 + ht**2))
                        np.fill_diagonal(Psi_plano_t, diag_plano_t)
                        G_plano_t = Psi_plano_t @ rho_t

                        tile_ny = r1 - r0
                        tile_nx = c1 - c0
                        G_mat_t = G_plano_t.reshape((tile_ny, tile_nx))
                        rho_mat_t = rho_t.reshape((tile_ny, tile_nx))

                        # Ventana de Hann 2D para solapamiento suave sin artefactos
                        wy = np.hanning(tile_ny + 2)[1:-1] if tile_ny > 2 else np.ones(tile_ny)
                        wx = np.hanning(tile_nx + 2)[1:-1] if tile_nx > 2 else np.ones(tile_nx)
                        W_tile = np.outer(wy, wx)

                        G_accum[r0:r1, c0:c1] += G_mat_t * W_tile
                        rho_accum[r0:r1, c0:c1] += rho_mat_t * W_tile
                        weight_accum[r0:r1, c0:c1] += W_tile

                        if tile_count % max(1, total_tiles // 10) == 0:
                            pct = 10 + int(78 * (tile_count / total_tiles))
                            self.sig_progress.emit(pct, f"Procesando teselas: {tile_count}/{total_tiles}...")

                valid_w = weight_accum > 1e-12
                G_plano_final = np.zeros_like(G_accum)
                rho_final = np.zeros_like(rho_accum)
                G_plano_final[valid_w] = G_accum[valid_w] / weight_accum[valid_w]
                rho_final[valid_w] = rho_accum[valid_w] / weight_accum[valid_w]

                final_R, final_C = orig_ny, orig_nx
                R, C = orig_ny, orig_nx
                N = orig_ny * orig_nx
                step = 1

            # ==================================================================
            # MODO 2: ADAPTATIVO INTELIGENTE / DIRECTO CON LAPACK
            # ==================================================================
            else:
                self.sig_progress.emit(5, "Optimizando resolución de la fuente equivalente...")

                # Resolver paso de muestreo
                if isinstance(self.decimation_step, (int, float)) and self.decimation_step < 0:
                    target_dim = abs(int(self.decimation_step))
                    step = max(1, int(round(max(orig_ny, orig_nx) / target_dim)))
                elif str(self.decimation_step).lower() in ['auto', '-50']:
                    step = max(1, int(round(max(orig_ny, orig_nx) / 50.0)))
                else:
                    try:
                        step = int(self.decimation_step)
                    except Exception:
                        step = 1

                # Protección estricta contra sobrecarga de memoria RAM (> 4000 fuentes densas)
                if step == 1 and N_total > 4000:
                    step = max(1, int(np.ceil(np.sqrt(N_total / 3600.0))))
                    self.sig_progress.emit(8, f" Protección de memoria: Malla de {N_total:,} nodos adaptada a paso {step}x...")

                if step > 1:
                    r_sub = max(2, int(np.ceil(orig_ny / step)))
                    c_sub = max(2, int(np.ceil(orig_nx / step)))
                    # Muestreo con extremos incluidos para que coincida exactamente con las coordenadas límite
                    r_idx = np.round(np.linspace(0, orig_ny - 1, r_sub)).astype(int)
                    c_idx = np.round(np.linspace(0, orig_nx - 1, c_sub)).astype(int)

                    X_calc = self.X_m[np.ix_(r_idx, c_idx)].astype(np.float32)
                    Y_calc = self.Y_m[np.ix_(r_idx, c_idx)].astype(np.float32)
                    Z_calc = self.Z_m[np.ix_(r_idx, c_idx)].astype(np.float32)
                    G_calc = self.G_m[np.ix_(r_idx, c_idx)].astype(np.float32)
                    dx_c = np.float32(self.dx_m * step)
                    dy_c = np.float32(self.dy_m * step)
                else:
                    X_calc = self.X_m.astype(np.float32)
                    Y_calc = self.Y_m.astype(np.float32)
                    Z_calc = self.Z_m.astype(np.float32)
                    G_calc = self.G_m.astype(np.float32)
                    dx_c = np.float32(self.dx_m)
                    dy_c = np.float32(self.dy_m)

                R, C = X_calc.shape
                N = R * C

                self.sig_progress.emit(15, f"Vectorizando {N:,} fuentes equivalentes (R={R}, C={C})...")
                Xobs = X_calc.ravel()
                Yobs = Y_calc.ravel()
                Zobs = Z_calc.ravel()
                Gobs = G_calc.ravel()

                ram_mb = (N * N * 4) / (1024.0 ** 2)
                self.sig_progress.emit(25, f"Construyendo matriz de sensibilidad Ψ ({N}×{N}, ~{ram_mb:.1f} MB)...")

                dX = Xobs[:, None] - Xobs[None, :]
                dY = Yobs[:, None] - Yobs[None, :]
                dZ = Zobs[:, None] - Zobs[None, :]

                R2 = dX**2 + dY**2 + dZ**2
                np.fill_diagonal(R2, 1.0)
                R3 = R2 * np.sqrt(R2)

                factor_off = np.float32(self.gamma_val * self.t_m * dx_c * dy_c)
                Psi = factor_off * (dZ / R3)

                # Diagonal regularizada con Damping de Tikhonov normalizado
                diag_val = np.float32(2.0 * np.pi * self.t_m * self.gamma_val)
                diag_eff = diag_val * np.float32(1.0 + self.damping)
                np.fill_diagonal(Psi, diag_eff)

                self.sig_progress.emit(52, f"Resolviendo inversión con LAPACK scipy.linalg ({N:,} fuentes)...")
                rho_obs = sla.solve(Psi, Gobs, assume_a='gen', overwrite_a=True)

                self.sig_progress.emit(75, f"Proyectando campo gravitatorio al plano Z0 = {self.Z0_m:.1f} m...")
                dZ0 = np.float32(self.Z0_m) - Zobs[None, :]
                R2_0 = dX**2 + dY**2 + (dZ0 ** 2)
                R3_0 = R2_0 * np.sqrt(R2_0)
                Psi_plano = factor_off * (dZ0 / R3_0)

                # Diagonal fisica exacta del prisma/celda directamente debajo
                Rc = np.float32(np.sqrt(dx_c * dy_c / np.pi))
                h_i = np.float32(self.Z0_m) - Zobs
                diag_plano = np.float32(2.0 * np.pi * self.gamma_val * self.t_m) * (1.0 - h_i / np.sqrt(Rc**2 + h_i**2))
                np.fill_diagonal(Psi_plano, diag_plano)

                Gobs_plano = Psi_plano @ rho_obs

                G_plano_sub = Gobs_plano.reshape((R, C))
                rho_sub = rho_obs.reshape((R, C))

                if step > 1 and self.interp_full:
                    self.sig_progress.emit(90, f"Reconstruyendo solución a resolución completa ({orig_nx}×{orig_ny}) con spline bicúbico...")
                    x_sub = X_calc[0, :]
                    y_sub = Y_calc[:, 0]
                    x_orig = self.X_m[0, :]
                    y_orig = self.Y_m[:, 0]

                    rgi_g = RegularGridInterpolator((y_sub, x_sub), G_plano_sub, method='cubic', bounds_error=False, fill_value=None)
                    rgi_r = RegularGridInterpolator((y_sub, x_sub), rho_sub, method='cubic', bounds_error=False, fill_value=None)

                    X_full, Y_full = np.meshgrid(x_orig, y_orig)
                    pts_full = np.column_stack([Y_full.ravel(), X_full.ravel()])
                    G_plano_final = rgi_g(pts_full).reshape((orig_ny, orig_nx))
                    rho_final = rgi_r(pts_full).reshape((orig_ny, orig_nx))
                    final_R, final_C = orig_ny, orig_nx
                else:
                    G_plano_final = G_plano_sub
                    rho_final = rho_sub
                    final_R, final_C = R, C

            self.sig_progress.emit(100, "¡Cálculo de fuente equivalente completado con éxito!")

            results = {
                'G_plano': G_plano_final,
                'rho_grid': rho_final,
                'G_obs': self.G_m,
                'Z_obs': self.Z_m,
                'X_m': self.X_m,
                'Y_m': self.Y_m,
                'Z0': self.Z0_m,
                't': self.t_m,
                'gamma': self.gamma_val,
                'damping': self.damping,
                'dx': self.dx_m,
                'dy': self.dy_m,
                'N_calc': N,
                'R_calc': R,
                'C_calc': C,
                'final_R': final_R,
                'final_C': final_C,
                'step': step,
                'rho_min': float(np.min(rho_final)),
                'rho_max': float(np.max(rho_final)),
                'rho_mean': float(np.mean(rho_final)),
                'G_plano_min': float(np.min(G_plano_final)),
                'G_plano_max': float(np.max(G_plano_final)),
                'G_plano_mean': float(np.mean(G_plano_final)),
            }
            self.sig_finished.emit(results)

        except Exception as e:
            self.sig_error.emit(str(e))


CONTOUR_COLOR_HEX = {
    "Negro": "#000000",
    "Blanco": "#ffffff",
    "Gris Oscuro": "#222222",
    "Gris Claro": "#e0e0e0",
    "Rojo": "#cc0000",
    "Azul": "#0033cc",
    "Amarillo": "#ffd700",
    "Cian": "#00e5ff",
    "Magenta": "#ff00e5"
}


# ==============================================================================
# CLASE PRINCIPAL: VENTANA DEL PROBLEMA INVERSO (FUENTE EQUIVALENTE)
# ==============================================================================
class VentanaProblemaInverso(QDialog):
    """
    Ventana independiente dedicada para resolver el Problema Inverso de Fuente Equivalente
    y continuar campos gravimétricos desde topografía irregular a un plano horizontal Z0.
    """
    def __init__(self, main_window=None, parent=None):
        parent_widget = parent if isinstance(parent, QWidget) else (main_window if isinstance(main_window, QWidget) else None)
        super().__init__(parent_widget)
        self.main_window = main_window

        self.setWindowTitle("GravTT - Problema Inverso: Fuente Equivalente (Continuación a Plano Z0 con MDT)")
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.resize(1300, 840)
        self.setMinimumSize(950, 600)

        # Estados y Matrices (Gravedad rige siempre el tamaño de malla R x C)
        self.raster_G = None       # Matriz G obs (R x C)
        self.raster_Z = None       # Matriz Z MDT en centros de píxel de Gravedad (R x C)
        self.raster_x = None       # Vector X de centros de columnas de G (C)
        self.raster_y = None       # Vector Y de centros de filas de G (R)
        self.raw_mdt_Z = None      # MDT original sin remuestrear
        self.raw_mdt_x = None      # X original de MDT
        self.raw_mdt_y = None      # Y original de MDT
        self.raw_mdt_obj = None    # Objeto GeoTiffMDT si está disponible
        self.unit_xy = "m"
        self.unit_z = "mGal"

        # Configuración visual independiente por cada tipo de gráfico / capa
        self.chart_settings = {
            'gobs': { # 1. Anomalía Observada
                'cmap': 'turbo',
                'invert': False,
                'contours': False,
                'n_contours': 12,
                'contour_color': 'Negro'
            },
            'gz0': { # 2. Anomalía Procesada Z0
                'cmap': 'turbo',
                'invert': False,
                'contours': False,
                'n_contours': 12,
                'contour_color': 'Negro'
            },
            'mdt': { # 3. Topografía MDT
                'cmap': 'terrain',
                'invert': False,
                'contours': False,
                'n_contours': 12,
                'contour_color': 'Blanco'
            },
            'rho': { # 4. Densidad de Capa rho
                'cmap': 'seismic',
                'invert': False,
                'contours': False,
                'n_contours': 12,
                'contour_color': 'Negro'
            }
        }
        self._updating_vis_controls = False

        # Resultados del cálculo
        self.last_results = None
        self.worker = None

        self.init_ui()
        self.sync_from_main_window()
        self.showMaximized()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(0)

        # Splitter Principal (Izquierda: Configuración compacta, Derecha: Visualización Máxima)
        splitter = QSplitter(Qt.Horizontal)

        # ======================================================================
        # PANEL IZQUIERDO: CONTROLES Y PARÁMETROS (COMPACTO)
        # ======================================================================
        left_widget = QWidget()
        left_widget.setMinimumWidth(390)
        v_left = QVBoxLayout(left_widget)
        v_left.setContentsMargins(0, 0, 2, 0)
        v_left.setSpacing(4)

        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left.setFrameShape(QFrame.NoFrame)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        form_v = QVBoxLayout(scroll_content)
        form_v.setContentsMargins(4, 4, 4, 4)
        form_v.setSpacing(6)

        # 1. RASTERS DE ENTRADA (GRAVEDAD Y MDT)
        grp_rasters = QGroupBox("1. Rasters de Entrada (Mediciones G & Topografía MDT)")
        grid_r = QGridLayout(grp_rasters)
        grid_r.setSpacing(6)

        # Raster 1: Mediciones G
        grid_r.addWidget(QLabel("<b>Raster 1: Gravedad (G_obs):</b>"), 0, 0, 1, 2)
        self.combo_source_g = QComboBox()
        self.combo_source_g.addItem("Malla Activa del Proyecto Principal", "main_active")
        self.combo_source_g.addItem("Malla Filtrada Fourier 2D", "main_filt")
        self.combo_source_g.addItem("Cargar Raster TIFF con Parámetros...", "file_g_tiff")
        self.combo_source_g.addItem("Cargar Archivo Raster (ASC, GRD, CSV)...", "file_g")
        self.combo_source_g.currentIndexChanged.connect(self.on_source_g_changed)
        grid_r.addWidget(self.combo_source_g, 1, 0)

        btn_browse_g = QPushButton("Abrir...")
        btn_browse_g.clicked.connect(self.browse_raster_g)
        grid_r.addWidget(btn_browse_g, 1, 1)

        self.lbl_info_g = QLabel("G: Sin datos.")
        self.lbl_info_g.setStyleSheet("font-size: 10px; color: #555555; background-color: #fafafa; border: 1px solid #d0d0d0; padding: 3px;")
        grid_r.addWidget(self.lbl_info_g, 2, 0, 1, 2)

        # Raster 2: MDT
        grid_r.addWidget(QLabel("<b>Raster 2: Topografía (MDT Z_obs):</b>"), 3, 0, 1, 2)
        self.combo_source_z = QComboBox()
        self.combo_source_z.addItem("MDT GeoTIFF del Proyecto Principal", "main_mdt")
        self.combo_source_z.addItem("Cargar Raster TIFF MDT con Parámetros...", "file_z_tiff")
        self.combo_source_z.addItem("Cargar Archivo MDT (ASC, GRD, CSV)...", "file_z")
        self.combo_source_z.currentIndexChanged.connect(self.on_source_z_changed)
        grid_r.addWidget(self.combo_source_z, 4, 0)

        btn_browse_z = QPushButton("Abrir MDT...")
        btn_browse_z.clicked.connect(self.browse_raster_z)
        grid_r.addWidget(btn_browse_z, 4, 1)

        self.lbl_info_z = QLabel("MDT: Sin datos.")
        self.lbl_info_z.setStyleSheet("font-size: 10px; color: #555555; background-color: #fafafa; border: 1px solid #d0d0d0; padding: 3px;")
        grid_r.addWidget(self.lbl_info_z, 5, 0, 1, 2)

        # Selector de Unidades de Coordenadas
        row_unit = QHBoxLayout()
        row_unit.addWidget(QLabel("Unidad Coordenadas:"))
        self.combo_unit_xy = QComboBox()
        self.combo_unit_xy.addItem("Metros [m]", "m")
        self.combo_unit_xy.addItem("Kilómetros [km]", "km")
        self.combo_unit_xy.currentIndexChanged.connect(self.update_dimensions_preview)
        row_unit.addWidget(self.combo_unit_xy)
        grid_r.addLayout(row_unit, 6, 0, 1, 2)

        form_v.addWidget(grp_rasters)

        # 2. PARÁMETROS DE FUENTE EQUIVALENTE
        grp_params = QGroupBox("2. Parámetros de la Fuente Equivalente")
        grid_p = QGridLayout(grp_params)
        grid_p.setSpacing(4)

        # Altura del plano Z0
        grid_p.addWidget(QLabel("<b>Cota Plano Z0 [m]:</b>"), 0, 0)
        self.spin_z0 = QDoubleSpinBox()
        self.spin_z0.setRange(-100000.0, 100000.0)
        self.spin_z0.setDecimals(1)
        self.spin_z0.setValue(2000.0)
        self.spin_z0.valueChanged.connect(self.on_z0_changed)
        grid_p.addWidget(self.spin_z0, 0, 1)

        self.lbl_z0_validation = QLabel("Z0 debe ser > max(MDT)")
        self.lbl_z0_validation.setStyleSheet("font-size: 10px; color: #888888;")
        grid_p.addWidget(self.lbl_z0_validation, 1, 0, 1, 2)

        # Botones rápidos de altura de plano
        row_quick_z0 = QHBoxLayout()
        btn_z0_p100 = QPushButton("+100 m")
        btn_z0_p100.clicked.connect(lambda: self.adjust_z0_relative(100.0))
        btn_z0_p250 = QPushButton("+250 m")
        btn_z0_p250.clicked.connect(lambda: self.adjust_z0_relative(250.0))
        btn_z0_p500 = QPushButton("+500 m")
        btn_z0_p500.clicked.connect(lambda: self.adjust_z0_relative(500.0))
        btn_z0_p1000 = QPushButton("+1000 m")
        btn_z0_p1000.clicked.connect(lambda: self.adjust_z0_relative(1000.0))
        row_quick_z0.addWidget(btn_z0_p100)
        row_quick_z0.addWidget(btn_z0_p250)
        row_quick_z0.addWidget(btn_z0_p500)
        row_quick_z0.addWidget(btn_z0_p1000)
        grid_p.addLayout(row_quick_z0, 2, 0, 1, 2)

        # Espesor de la capa t (m)
        grid_p.addWidget(QLabel("Espesor capa t [m]:"), 3, 0)
        self.spin_t = QDoubleSpinBox()
        self.spin_t.setRange(0.1, 10000.0); self.spin_t.setDecimals(2)
        self.spin_t.setValue(T_DEFAULT_METERS)
        self.spin_t.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.spin_t, 3, 1)

        # Constante gravitacional gamma
        grid_p.addWidget(QLabel("Constante γ [mGal·m²/kg]:"), 4, 0)
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(1e-12, 1.0); self.spin_gamma.setDecimals(10)
        self.spin_gamma.setValue(GAMMA_DEFAULT_MGAL)
        self.spin_gamma.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.spin_gamma, 4, 1)

        # Regularización Damping
        grid_p.addWidget(QLabel("Damping Tikhonov (λ):"), 5, 0)
        self.spin_damping = QDoubleSpinBox()
        self.spin_damping.setRange(0.0, 10.0); self.spin_damping.setDecimals(6)
        self.spin_damping.setValue(0.0)
        self.spin_damping.setToolTip("Amortiguamiento / Suavizado para estabilizar el cálculo (λ >= 0). 0 = Sin freno.")
        self.spin_damping.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.spin_damping, 5, 1)

        form_v.addWidget(grp_params)

        # 3. RESOLUCIÓN Y RENDIMIENTO DE CÁLCULO
        grp_perf = QGroupBox("3. Resolución & Rendimiento")
        v_perf = QVBoxLayout(grp_perf)
        v_perf.setSpacing(4)

        v_perf.addWidget(QLabel("Modo de Resolución para Matriz Ψ:"))
        self.combo_decimation = QComboBox()
        self.combo_decimation.addItem("Automático Inteligente (~50x50, Spline HD)", -50)
        self.combo_decimation.addItem("Alta Precisión (~80x80, Spline HD)", -80)
        self.combo_decimation.addItem("Inversión por Bloques / Teselas (Malla Gigante)", "tiled")
        self.combo_decimation.addItem("Decimar paso 2 (1/4 de nodos)", 2)
        self.combo_decimation.addItem("Decimar paso 3 (1/9 de nodos)", 3)
        self.combo_decimation.addItem("Decimar paso 4 (1/16 de nodos)", 4)
        self.combo_decimation.addItem("Resolución Completa Directa (Paso 1)", 1)
        self.combo_decimation.currentIndexChanged.connect(self.on_param_changed)
        v_perf.addWidget(self.combo_decimation)

        self.chk_interp_full = QCheckBox("Re-interpolar a resolución nativa original")
        self.chk_interp_full.setChecked(True)
        v_perf.addWidget(self.chk_interp_full)

        self.lbl_perf_stats = QLabel("Nodos: - | Memoria: -")
        self.lbl_perf_stats.setStyleSheet("font-family: 'Consolas', monospace; font-size: 10px; background-color: #fafafa; border: 1px solid #d0d0d0; padding: 4px;")
        v_perf.addWidget(self.lbl_perf_stats)

        form_v.addWidget(grp_perf)

        # BOTÓN DE CÁLCULO (Sin auto-recalcular)
        self.btn_run_inv = QPushButton("Calcular Fuente Equivalente y Continuar a Z0")
        self.btn_run_inv.setStyleSheet("background-color: #cc0000; color: #ffffff; font-weight: bold; font-size: 12px; padding: 8px;")
        self.btn_run_inv.clicked.connect(lambda: self.run_inversion(is_auto=False))
        form_v.addWidget(self.btn_run_inv)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        form_v.addWidget(self.progress_bar)

        self.lbl_calc_status = QLabel("Listo para iniciar cálculo.")
        self.lbl_calc_status.setStyleSheet("font-size: 10px; color: #333333;")
        form_v.addWidget(self.lbl_calc_status)

        # 4. ACCIONES DE INTEGRACIÓN Y EXPORTACIÓN
        grp_export = QGroupBox("4. Exportación & Transferencia")
        v_exp = QVBoxLayout(grp_export)
        v_exp.setSpacing(4)

        self.btn_inject_main = QPushButton("Enviar Malla Z0 al Proyecto Principal (Filtros y Mapas)")
        self.btn_inject_main.setStyleSheet("background-color: #006600; color: #ffffff; font-weight: bold; padding: 6px;")
        self.btn_inject_main.clicked.connect(self.inject_into_main_project)
        self.btn_inject_main.setEnabled(False)
        v_exp.addWidget(self.btn_inject_main)

        self.chk_set_as_active = QCheckBox("Fijar como Malla Activa Principal")
        self.chk_set_as_active.setChecked(True)
        v_exp.addWidget(self.chk_set_as_active)

        btn_export_csv = QPushButton("Guardar Malla Z0 en CSV (XYZ)...")
        btn_export_csv.clicked.connect(self.export_csv_xyz)
        v_exp.addWidget(btn_export_csv)

        btn_export_grd = QPushButton("Guardar Malla Z0 en Surfer Grid (.grd)...")
        btn_export_grd.clicked.connect(self.export_surfer_grid)
        v_exp.addWidget(btn_export_grd)

        form_v.addWidget(grp_export)

        scroll_left.setWidget(scroll_content)
        v_left.addWidget(scroll_left)
        left_widget.setMinimumWidth(440)
        left_widget.setMaximumWidth(600)
        splitter.addWidget(left_widget)

        # ======================================================================
        # PANEL DERECHO: VISUALIZADOR MULTICAPA MATPLOTLIB (MÁXIMO TAMAÑO)
        # ======================================================================
        right_widget = QWidget()
        v_right = QVBoxLayout(right_widget)
        v_right.setContentsMargins(2, 0, 0, 0)
        v_right.setSpacing(2)

        # Barra de visualización superior compacta con controles independientes por capa
        vis_bar = QHBoxLayout()
        vis_bar.setContentsMargins(0, 0, 0, 2)
        vis_bar.setSpacing(4)

        vis_bar.addWidget(QLabel("<b>Capa:</b>"))
        self.combo_target_layer = QComboBox()
        self.combo_target_layer.addItem("Anomalia Observada (G_obs)", "gobs")
        self.combo_target_layer.addItem("Anomalia Procesada (Z0)", "gz0")
        self.combo_target_layer.addItem("Topografia MDT", "mdt")
        self.combo_target_layer.addItem("Densidad Capa (rho)", "rho")
        self.combo_target_layer.currentIndexChanged.connect(self.on_target_layer_changed)
        vis_bar.addWidget(self.combo_target_layer)

        vis_bar.addWidget(QLabel("<b>Paleta:</b>"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems([
            "turbo", "jet", "viridis", "terrain", "coolwarm", "seismic",
            "plasma", "magma", "bwr", "Spectral", "gist_earth", "gray"
        ])
        self.combo_cmap.currentIndexChanged.connect(self.on_vis_control_changed)
        vis_bar.addWidget(self.combo_cmap)

        self.chk_cmap_r = QCheckBox("Invertir (_r)")
        self.chk_cmap_r.stateChanged.connect(self.on_vis_control_changed)
        vis_bar.addWidget(self.chk_cmap_r)

        self.chk_contours = QCheckBox("Curvas de Nivel")
        self.chk_contours.setChecked(False)
        self.chk_contours.stateChanged.connect(self.on_vis_control_changed)
        vis_bar.addWidget(self.chk_contours)

        self.spin_contours_n = QSpinBox()
        self.spin_contours_n.setRange(4, 60); self.spin_contours_n.setValue(12)
        self.spin_contours_n.valueChanged.connect(self.on_vis_control_changed)
        vis_bar.addWidget(self.spin_contours_n)

        vis_bar.addWidget(QLabel("<b>Color Curvas:</b>"))
        self.combo_contour_color = QComboBox()
        for c_name, hex_code in CONTOUR_COLOR_HEX.items():
            pm = QPixmap(14, 14)
            pm.fill(QColor(hex_code))
            self.combo_contour_color.addItem(QIcon(pm), c_name)
        self.combo_contour_color.currentIndexChanged.connect(self.on_vis_control_changed)
        vis_bar.addWidget(self.combo_contour_color)

        vis_bar.addStretch()

        self.lbl_head_badge = QLabel("Listo para configurar rasters")
        self.lbl_head_badge.setStyleSheet("font-size: 11px; font-weight: 600; color: #404040; background-color: #e8e8e8; padding: 2px 8px; border: 1px solid #c0c0c0; border-radius: 2px;")
        self.lbl_head_badge.setFixedHeight(22)
        vis_bar.addWidget(self.lbl_head_badge)

        v_right.addLayout(vis_bar)

        # Pestañas de Visualización Multicapa (Orden: Anomalía - Anomalía Procesada - Topografía - Densidad - Vista 2x2)
        self.tabs_view = QTabWidget()
        self.tabs_view.currentChanged.connect(self.on_tab_changed)

        # Tab 0: Gravedad Observada Gobs (Anomalía)
        self.canvas_gobs = self._create_canvas()
        self.tabs_view.addTab(self.canvas_gobs['widget'], "Anomalia Observada G_obs")

        # Tab 1: Gravedad en Plano Z0 (Anomalía Procesada)
        self.canvas_res = self._create_canvas()
        self.tabs_view.addTab(self.canvas_res['widget'], "Anomalia Procesada Z0")

        # Tab 2: MDT Topografía
        self.canvas_mdt = self._create_canvas()
        self.tabs_view.addTab(self.canvas_mdt['widget'], "Topografia MDT")

        # Tab 3: Densidad de Fuentes rho
        self.canvas_rho = self._create_canvas()
        self.tabs_view.addTab(self.canvas_rho['widget'], "Densidad de Capa rho")

        # Tab 4: Cuadrantes 2x2
        self.canvas_quad = self._create_canvas()
        self.tabs_view.addTab(self.canvas_quad['widget'], "Vista Cuadruple 2x2")

        v_right.addWidget(self.tabs_view)

        # Barra inferior de coordenadas
        self.lbl_coord_readout = QLabel("Coordenadas: (X: -, Y: -) | Valor: -")
        self.lbl_coord_readout.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; background-color: #f0f0f0; border: 1px solid #c0c0c0; padding: 2px 4px;")
        v_right.addWidget(self.lbl_coord_readout)

        splitter.addWidget(right_widget)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 1050])
        root_layout.addWidget(splitter, 1)
        self.sync_vis_controls_to_layer('gobs')

    def _create_canvas(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        fig = Figure(figsize=(7, 6), dpi=100)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, w)
        toolbar.setStyleSheet("background-color: #f0f0f0; border: 1px solid #b0b0b0;")
        v.addWidget(toolbar)
        v.addWidget(canvas)
        canvas.mpl_connect('motion_notify_event', self.on_canvas_mouse_move)
        return {'widget': w, 'fig': fig, 'canvas': canvas, 'toolbar': toolbar}

    # ==========================================================================
    # SINCRONIZACIÓN CON EL PROYECTO PRINCIPAL
    # ==========================================================================
    def sync_from_main_window(self):
        """Toma las mallas activas en memoria desde la ventana principal si existen."""
        if self.main_window is None:
            return

        # Sincronizar unidad
        if hasattr(self.main_window, 'unit_xy'):
            u = self.main_window.unit_xy.lower()
            idx = 1 if 'km' in u else 0
            self.combo_unit_xy.setCurrentIndex(idx)
            self.unit_xy = "km" if 'km' in u else "m"

        # 1. Cargar Malla de Gravedad del proyecto si existe (Gobierna el proceso)
        if getattr(self.main_window, 'grid_Z', None) is not None and getattr(self.main_window, 'grid_x_vec', None) is not None:
            self.raster_G = self.main_window.grid_Z.copy()
            self.raster_x = self.main_window.grid_x_vec.copy()
            self.raster_y = self.main_window.grid_y_vec.copy()
            ny, nx = self.raster_G.shape
            self.lbl_info_g.setText(
                f" Gravedad: Proyecto [{nx} cols × {ny} filas] (Rige Matriz) | "
                f"Rango: [{np.min(self.raster_G):.2f}, {np.max(self.raster_G):.2f}] mGal"
            )

        # 2. Cargar MDT del proyecto si existe
        if getattr(self.main_window, 'grid_mdt', None) is not None:
            self.raw_mdt_Z = self.main_window.grid_mdt.copy()
            self.raw_mdt_x = getattr(self.main_window, 'mdt_x_vec', self.main_window.grid_x_vec)
            self.raw_mdt_y = getattr(self.main_window, 'mdt_y_vec', self.main_window.grid_y_vec)
            if self.raw_mdt_x is not None:
                self.raw_mdt_x = self.raw_mdt_x.copy()
            if self.raw_mdt_y is not None:
                self.raw_mdt_y = self.raw_mdt_y.copy()
        else:
            mdt_obj = getattr(self.main_window, 'mdt_obj', None)
            if mdt_obj is not None and getattr(mdt_obj, 'is_loaded', False) and mdt_obj.grid_z is not None:
                self.raw_mdt_obj = mdt_obj

        # Sincronizar y acoplar MDT a los centros de píxel de Gravedad
        self.match_mdt_to_gravity()
        self.update_dimensions_preview()
        self.replot_active_tab()

    def load_mdt_from_obj(self, mdt_obj):
        """Carga y acopla el objeto GeoTiffMDT muestreándolo en los centros de celda de gravedad."""
        self.raw_mdt_obj = mdt_obj
        if self.raster_x is not None and self.raster_y is not None and self.raster_G is not None:
            self.match_mdt_to_gravity()
        else:
            self.raster_Z = mdt_obj.grid_z.copy()
            self.raster_x = mdt_obj.x_vec.copy()
            self.raster_y = mdt_obj.y_vec.copy()
            z_max = float(np.max(self.raster_Z))
            self.update_default_z0(z_max)

    def match_mdt_to_gravity(self):
        """
        Regla de oro: La matriz de gravedad rige SIEMPRE el proceso matricial.
        Toma la coordenada Z del MDT sobrepuesto en el raster de gravedad,
        evaluando el valor del MDT en el centro de cada píxel de gravedad.
        Garantiza que raster_Z tenga dimensiones idénticas (R, C) a raster_G.
        """
        if self.raster_G is None or self.raster_x is None or self.raster_y is None:
            return

        R, C = self.raster_G.shape

        if self.raw_mdt_obj is not None and getattr(self.raw_mdt_obj, 'is_loaded', False):
            self.raster_Z = extract_mdt_at_gravity_pixels(
                self.raster_x, self.raster_y, mdt_obj=self.raw_mdt_obj
            )
        elif self.raw_mdt_Z is not None:
            self.raster_Z = extract_mdt_at_gravity_pixels(
                self.raster_x, self.raster_y,
                mdt_Z=self.raw_mdt_Z, mdt_x=self.raw_mdt_x, mdt_y=self.raw_mdt_y
            )
        elif self.raster_Z is not None and self.raster_Z.shape != self.raster_G.shape:
            ny_z, nx_z = self.raster_Z.shape
            x_z_tmp = np.linspace(float(self.raster_x[0]), float(self.raster_x[-1]), nx_z)
            y_z_tmp = np.linspace(float(self.raster_y[0]), float(self.raster_y[-1]), ny_z)
            self.raster_Z = extract_mdt_at_gravity_pixels(
                self.raster_x, self.raster_y,
                mdt_Z=self.raster_Z, mdt_x=x_z_tmp, mdt_y=y_z_tmp
            )

        if self.raster_Z is not None:
            assert self.raster_Z.shape == self.raster_G.shape, (
                f"Error dimensional: MDT ({self.raster_Z.shape}) != Gravedad ({self.raster_G.shape})"
            )
            z_min = float(np.min(self.raster_Z))
            z_max = float(np.max(self.raster_Z))
            self.lbl_info_z.setText(
                f" MDT: Centros de Píxel de Gravedad [{C} cols × {R} filas] | Cotas: [{z_min:.0f}, {z_max:.0f}] m"
            )
            self.update_default_z0(z_max)

    def update_default_z0(self, z_max):
        """Actualiza la cota Z0 sugerida automáticamente superando el pico más alto."""
        suggested = float(np.ceil((z_max + 250.0) / 50.0) * 50.0)
        self.spin_z0.blockSignals(True)
        self.spin_z0.setValue(suggested)
        self.spin_z0.blockSignals(False)
        self.on_z0_changed()

    def on_z0_changed(self):
        z0 = self.spin_z0.value()
        z_max = float(np.max(self.raster_Z)) if self.raster_Z is not None else 0.0
        if z0 <= z_max:
            self.lbl_z0_validation.setText(f" INVÁLIDO: Z0 ({z0:.1f}m) debe ser mayor que el pico del MDT ({z_max:.1f}m)")
            self.lbl_z0_validation.setStyleSheet("font-size: 10px; font-weight: bold; color: #cc0000;")
            self.btn_run_inv.setEnabled(False)
        else:
            diff = z0 - z_max
            self.lbl_z0_validation.setText(f" Válido: Z0 supera la cota máxima del MDT por +{diff:.1f} metros.")
            self.lbl_z0_validation.setStyleSheet("font-size: 10px; font-weight: bold; color: #006600;")
            self.btn_run_inv.setEnabled(True)
            self.on_param_changed()

    def adjust_z0_relative(self, delta):
        z_max = float(np.max(self.raster_Z)) if self.raster_Z is not None else 0.0
        self.spin_z0.setValue(z_max + delta)

    def on_param_changed(self):
        """Disparado cuando cambia cualquier parámetro: Z0, Damping, espesor t, gamma o resolución."""
        self.update_dimensions_preview()
        if self.last_results is not None:
            self.lbl_head_badge.setText("Parámetros modificados (haz clic en 'Calcular' para refrescar)")
            self.lbl_head_badge.setStyleSheet("font-size: 11px; font-weight: 600; color: #b35900; background-color: #fff3e0; padding: 2px 8px; border: 1px solid #ffb74d; border-radius: 2px;")

    def on_tab_changed(self, tab_idx):
        tab_to_key = {
            0: 'gobs',  # Anomalía Observada G_obs
            1: 'gz0',   # Anomalía Procesada Z0
            2: 'mdt',   # Topografía MDT
            3: 'rho',   # Densidad de Capa rho
        }
        if tab_idx in tab_to_key:
            target_key = tab_to_key[tab_idx]
            idx = self.combo_target_layer.findData(target_key)
            if idx >= 0 and self.combo_target_layer.currentIndex() != idx:
                self.combo_target_layer.blockSignals(True)
                self.combo_target_layer.setCurrentIndex(idx)
                self.combo_target_layer.blockSignals(False)
                self.sync_vis_controls_to_layer(target_key)
        self.replot_active_tab()

    def on_target_layer_changed(self):
        target_key = self.combo_target_layer.currentData()
        if target_key:
            self.sync_vis_controls_to_layer(target_key)
            self.replot_active_tab()

    def sync_vis_controls_to_layer(self, layer_key):
        if layer_key not in self.chart_settings:
            return
        cfg = self.chart_settings[layer_key]
        self._updating_vis_controls = True
        try:
            c_idx = self.combo_cmap.findText(cfg['cmap'])
            if c_idx >= 0:
                self.combo_cmap.setCurrentIndex(c_idx)
            self.chk_cmap_r.setChecked(cfg['invert'])
            self.chk_contours.setChecked(cfg['contours'])
            self.spin_contours_n.setValue(cfg['n_contours'])
            col_idx = self.combo_contour_color.findText(cfg['contour_color'])
            if col_idx >= 0:
                self.combo_contour_color.setCurrentIndex(col_idx)
        finally:
            self._updating_vis_controls = False

    def on_vis_control_changed(self):
        if getattr(self, '_updating_vis_controls', False):
            return
        layer_key = self.combo_target_layer.currentData()
        if not layer_key or layer_key not in self.chart_settings:
            return
        self.chart_settings[layer_key]['cmap'] = self.combo_cmap.currentText()
        self.chart_settings[layer_key]['invert'] = self.chk_cmap_r.isChecked()
        self.chart_settings[layer_key]['contours'] = self.chk_contours.isChecked()
        self.chart_settings[layer_key]['n_contours'] = self.spin_contours_n.value()
        self.chart_settings[layer_key]['contour_color'] = self.combo_contour_color.currentText()
        self.replot_active_tab()

    def _get_active_cmap(self):
        target_key = self.combo_target_layer.currentData()
        return self._get_layer_cmap_name(target_key if target_key else 'gz0')

    def _get_layer_cmap_name(self, layer_key):
        if layer_key in self.chart_settings:
            cfg = self.chart_settings[layer_key]
            cmap = cfg['cmap']
            if cfg['invert']:
                cmap += "_r"
            return cmap
        return 'turbo'

    def _draw_contours_for_layer(self, ax, layer_key, data, custom_levels=None):
        if layer_key not in self.chart_settings:
            return
        cfg = self.chart_settings[layer_key]
        if cfg['contours'] and self.raster_x is not None and self.raster_y is not None:
            try:
                gx, gy = np.meshgrid(self.raster_x, self.raster_y)
                n_c = cfg['n_contours']
                c_name = cfg['contour_color']
                hex_col = CONTOUR_COLOR_HEX.get(c_name, '#000000')
                levels = custom_levels if custom_levels is not None else n_c
                ax.contour(gx, gy, data, levels=levels, colors=hex_col, alpha=0.8, linewidths=0.75)
            except Exception:
                pass

    # ==========================================================================
    # MANEJO DE FUENTES Y CARGA DE ARCHIVOS EXTERNOS
    # ==========================================================================
    def on_source_g_changed(self):
        key = self.combo_source_g.currentData()
        if key == 'main_active':
            if getattr(self.main_window, 'grid_Z', None) is not None:
                self.raster_G = self.main_window.grid_Z.copy()
                self.raster_x = self.main_window.grid_x_vec.copy()
                self.raster_y = self.main_window.grid_y_vec.copy()
                ny, nx = self.raster_G.shape
                self.lbl_info_g.setText(f" Gravedad: Original [{nx}×{ny}] (Rige Matriz)")
        elif key == 'main_filt':
            if getattr(self.main_window, 'grid_Z_filt', None) is not None:
                self.raster_G = self.main_window.grid_Z_filt.copy()
                self.raster_x = self.main_window.grid_x_vec.copy()
                self.raster_y = self.main_window.grid_y_vec.copy()
                ny, nx = self.raster_G.shape
                self.lbl_info_g.setText(f" Gravedad: Filtrada [{nx}×{ny}] (Rige Matriz)")
            else:
                QMessageBox.warning(self, "Sin Filtro", "Aún no se ha calculado la malla filtrada Fourier en la Pestaña 4.")
                self.combo_source_g.setCurrentIndex(0)
        elif key == 'file_g_tiff':
            self.browse_raster_g(force_tiff=True)
        elif key == 'file_g':
            self.browse_raster_g(force_tiff=False)

        self.match_mdt_to_gravity()
        self.update_dimensions_preview()
        self.replot_active_tab()

    def on_source_z_changed(self):
        key = self.combo_source_z.currentData()
        if key == 'main_mdt':
            mdt = getattr(self.main_window, 'mdt_obj', None)
            if mdt is not None and getattr(mdt, 'is_loaded', False):
                self.load_mdt_from_obj(mdt)
            else:
                QMessageBox.warning(self, "Sin MDT", "No hay MDT GeoTIFF cargado en el proyecto principal.")
        elif key == 'file_z_tiff':
            self.browse_raster_z(force_tiff=True)
        elif key == 'file_z':
            self.browse_raster_z(force_tiff=False)

        self.match_mdt_to_gravity()
        self.update_dimensions_preview()
        self.replot_active_tab()

    def browse_raster_g(self, force_tiff=False):
        filt = "Archivos TIFF (*.tif *.tiff)" if force_tiff else "Archivos Raster (*.tif *.tiff *.asc *.grd *.csv *.xyz);;GeoTIFF (*.tif *.tiff);;ESRI ASCII (*.asc);;Surfer Grid (*.grd);;CSV/XYZ (*.csv *.xyz);;Todos (*.*)"
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Cargar Raster de Mediciones Gravimétricas", os.getcwd(), filt
        )
        if fpath:
            try:
                ext = os.path.splitext(fpath)[1].lower()
                if ext in ['.tif', '.tiff'] or force_tiff:
                    dlg = DialogoParametrosRasterTIFF(fpath, is_gravity=True, parent=self)
                    if dlg.exec_() != QDialog.Accepted:
                        return
                    Z, x_vec, y_vec, meta = dlg.get_result()
                else:
                    Z, x_vec, y_vec, meta = read_raster_file(fpath)

                self.raster_G = Z
                self.raster_x = x_vec
                self.raster_y = y_vec
                if meta.get('unit_xy') == 'km':
                    self.combo_unit_xy.setCurrentIndex(1)
                else:
                    self.combo_unit_xy.setCurrentIndex(0)

                ny, nx = Z.shape
                fname = os.path.basename(fpath)
                self.lbl_info_g.setText(f"G: {fname} [{nx}×{ny}] (Rige Matriz) | Cotas: [{np.min(Z):.2f}, {np.max(Z):.2f}] mGal")
                self.match_mdt_to_gravity()
                self.update_dimensions_preview()
                self.replot_active_tab()
            except Exception as e:
                QMessageBox.critical(self, "Error al Cargar Raster G", f"No se pudo leer el archivo:\n{e}")

    def browse_raster_z(self, force_tiff=False):
        filt = "Archivos TIFF (*.tif *.tiff)" if force_tiff else "Archivos Raster (*.tif *.tiff *.asc *.grd *.csv *.xyz);;GeoTIFF (*.tif *.tiff);;ESRI ASCII (*.asc);;Surfer Grid (*.grd);;CSV/XYZ (*.csv *.xyz);;Todos (*.*)"
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Cargar Modelo Digital de Terreno (MDT)", os.getcwd(), filt
        )
        if fpath:
            try:
                ext = os.path.splitext(fpath)[1].lower()
                if ext in ['.tif', '.tiff'] or force_tiff:
                    dlg = DialogoParametrosRasterTIFF(fpath, is_gravity=False, parent=self)
                    if dlg.exec_() != QDialog.Accepted:
                        return
                    Z, x_vec, y_vec, meta = dlg.get_result()
                else:
                    Z, x_vec, y_vec, meta = read_raster_file(fpath)

                fname = os.path.basename(fpath)
                self.raw_mdt_Z = Z
                self.raw_mdt_x = x_vec
                self.raw_mdt_y = y_vec
                self.raw_mdt_obj = None

                if self.raster_G is not None and self.raster_x is not None and self.raster_y is not None:
                    self.match_mdt_to_gravity()
                else:
                    self.raster_Z = Z
                    self.raster_x = x_vec
                    self.raster_y = y_vec
                    z_max = float(np.max(self.raster_Z))
                    self.lbl_info_z.setText(f"MDT: {fname} [{len(x_vec)}×{len(y_vec)}] | Cotas: [{np.min(Z):.0f}, {z_max:.0f}] m")
                    self.update_default_z0(z_max)

                self.update_dimensions_preview()
                self.replot_active_tab()
            except Exception as e:
                QMessageBox.critical(self, "Error al Cargar MDT", f"No se pudo leer el archivo:\n{e}")

    def check_and_match_rasters(self):
        """Verifica que ambos rasters existan y garantiza coincidencia estricta R x C regida por Gravedad."""
        self.match_mdt_to_gravity()

    def update_dimensions_preview(self):
        """Calcula y muestra en tiempo real las dimensiones y uso de RAM para la matriz Psi."""
        if self.raster_G is None or self.raster_Z is None:
            self.lbl_perf_stats.setText("Esperando carga de ambos rasters (G y MDT)...")
            return

        ny, nx = self.raster_G.shape
        step_val = self.combo_decimation.currentData()

        if str(step_val).lower() == 'tiled':
            tile_sz = 40
            n_tile = tile_sz * tile_sz
            ram_tile_mb = (n_tile * n_tile * 4.0) / (1024.0 ** 2)
            n_calc = ny * nx
            self.lbl_perf_stats.setText(
                f"Malla original: {nx}×{ny} ({n_calc:,} nodos)\n"
                f"Modo Teselas: Inversión en bloques de 40×40 con ventana Hann 2D.\n"
                f"Memoria RAM máxima por bloque: ~{ram_tile_mb:.1f} MB (Seguro para mallas gigantes)"
            )
            self.lbl_perf_stats.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; background-color: #e8f5e9; border: 1px solid #81c784; color: #1b5e20; padding: 5px;")
            return

        if isinstance(step_val, (int, float)) and step_val < 0:
            target_dim = abs(int(step_val))
            step_calc = max(1, int(round(max(nx, ny) / target_dim)))
        else:
            try:
                step_calc = int(step_val)
            except Exception:
                step_calc = 1

        r_calc = max(2, int(np.ceil(ny / step_calc)))
        c_calc = max(2, int(np.ceil(nx / step_calc)))
        n_calc = r_calc * c_calc
        ram_mb = (n_calc * n_calc * 4.0) / (1024.0 ** 2)

        u_mult = 1000.0 if self.combo_unit_xy.currentData() == 'km' else 1.0
        dx_m = (abs(self.raster_x[1] - self.raster_x[0]) if len(self.raster_x) > 1 else 100.0) * u_mult
        dy_m = (abs(self.raster_y[1] - self.raster_y[0]) if len(self.raster_y) > 1 else 100.0) * u_mult

        txt_warn = ""
        if n_calc > 10000:
            txt_warn = "  Matriz densa. Se recomienda Automático o Teselas."

        self.lbl_perf_stats.setText(
            f"Malla original: {nx}×{ny} nodos (Δx={dx_m:.1f}m, Δy={dy_m:.1f}m)\n"
            f"Cálculo: R={r_calc} × C={c_calc} (Paso={step_calc}) -> N={n_calc:,} fuentes\n"
            f"Memoria Matriz Ψ ({n_calc:,}×{n_calc:,}): ~{ram_mb:.1f} MB RAM (Float32 LAPACK){txt_warn}"
        )

        if n_calc > 12000:
            self.lbl_perf_stats.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; background-color: #fff0f0; border: 1px solid #cc0000; color: #cc0000; padding: 5px;")
        else:
            self.lbl_perf_stats.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; background-color: #fafafa; border: 1px solid #d0d0d0; color: #000000; padding: 5px;")

    # ==========================================================================
    # EJECUCIÓN DEL CÁLCULO
    # ==========================================================================
    def run_inversion(self, is_auto=False):
        """Valida e inicia el proceso de fuente equivalente en un hilo desacoplado con LAPACK."""
        if self.worker is not None and self.worker.isRunning():
            if is_auto:
                return  # Si ya hay un cálculo en curso reactivo, no solapar
            else:
                self.worker.terminate()
                self.worker.wait(100)

        if self.raster_G is None or self.raster_Z is None:
            if not is_auto:
                QMessageBox.warning(self, "Datos Incompletos", "Debes cargar o vincular ambos rasters: Gravedad (G) y Topografía (MDT).")
            return

        self.check_and_match_rasters()

        z0 = self.spin_z0.value()
        z_max = float(np.max(self.raster_Z))
        if z0 <= z_max:
            if not is_auto:
                QMessageBox.critical(self, "Cota Z0 Inválida", f"La altura del plano Z0 ({z0:.1f} m) debe ser mayor que el pico más alto del MDT ({z_max:.1f} m).")
            return

        u_mult = 1000.0 if self.combo_unit_xy.currentData() == 'km' else 1.0
        X_mat, Y_mat = np.meshgrid(self.raster_x * u_mult, self.raster_y * u_mult)
        dx_m = float(abs(self.raster_x[1] - self.raster_x[0])) * u_mult if len(self.raster_x) > 1 else 100.0
        dy_m = float(abs(self.raster_y[1] - self.raster_y[0])) * u_mult if len(self.raster_y) > 1 else 100.0

        step_val = self.combo_decimation.currentData()
        if str(step_val).lower() == 'tiled':
            step_calc = 'tiled'
        elif isinstance(step_val, (int, float)) and step_val < 0:
            target_dim = abs(int(step_val))
            ny, nx = self.raster_G.shape
            step_calc = max(1, int(round(max(nx, ny) / target_dim)))
        else:
            try:
                step_calc = int(step_val)
            except Exception:
                step_calc = 1

        t_val = self.spin_t.value()
        gamma_val = self.spin_gamma.value()
        damping_val = self.spin_damping.value()
        interp_full = self.chk_interp_full.isChecked()

        self._is_auto_running = is_auto
        self.btn_run_inv.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_calc_status.setText("Iniciando inversión con LAPACK...")

        self.worker = WorkerFuenteEquivalente(
            X_mat, Y_mat, self.raster_Z, self.raster_G,
            z0, t_val, gamma_val, dx_m, dy_m, damping_val, step_calc, interp_full
        )
        self.worker.sig_progress.connect(self.on_worker_progress)
        self.worker.sig_finished.connect(self.on_worker_finished)
        self.worker.sig_error.connect(self.on_worker_error)
        self.worker.start()

    def on_worker_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.lbl_calc_status.setText(message)

    def on_worker_finished(self, results):
        self.btn_run_inv.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.last_results = results
        self.btn_inject_main.setEnabled(True)

        n = results['N_calc']
        z0 = results['Z0']
        g_min = results['G_plano_min']
        g_max = results['G_plano_max']
        rho_min = results['rho_min']
        rho_max = results['rho_max']

        self.lbl_head_badge.setText(f"Fuente Equivalente Z0={z0:.0f}m ({results['final_C']}×{results['final_R']} nodos)")
        self.lbl_head_badge.setStyleSheet("font-size: 11px; font-weight: 600; color: #006600; background-color: #e8f5e9; padding: 2px 8px; border: 1px solid #81c784; border-radius: 2px;")
        self.lbl_calc_status.setText(
            f"Actualizado | {n:,} fuentes | G_Z0: [{g_min:.2f}, {g_max:.2f}] mGal | ρ: [{rho_min:.1f}, {rho_max:.1f}] kg/m³"
        )

        # Forzar reploteo inmediato de la pestaña activa para actualizar la visualización
        self.replot_active_tab()

    def on_worker_error(self, err_msg):
        self.btn_run_inv.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_calc_status.setText("Error durante el cálculo.")
        self.lbl_head_badge.setText("Error en Inversión")
        self.lbl_head_badge.setStyleSheet("font-size: 11px; font-weight: 600; color: #cc0000; background-color: #ffebee; padding: 2px 8px; border: 1px solid #ef9a9a; border-radius: 2px;")
        QMessageBox.critical(self, "Error en Fuente Equivalente", f"Falló el cálculo:\n{err_msg}")

    # ==========================================================================
    # GRAFICACIÓN Y RENDERIZADO DE PESTAÑAS
    # ==========================================================================
    def replot_active_tab(self):
        tab_idx = self.tabs_view.currentIndex()
        if tab_idx == 0:
            self.plot_tab_gobs()
        elif tab_idx == 1:
            self.plot_tab_result()
        elif tab_idx == 2:
            self.plot_tab_mdt()
        elif tab_idx == 3:
            self.plot_tab_rho()
        elif tab_idx in [4, 5]:
            self.plot_tab_quad()

    def _get_extent(self):
        if self.raster_x is None or self.raster_y is None:
            return [0, 1, 0, 1]
        return [float(self.raster_x[0]), float(self.raster_x[-1]), float(self.raster_y[0]), float(self.raster_y[-1])]

    def _get_shared_gravity_limits(self):
        """Calcula el rango común [vmin, vmax] entre la anomalía observada y la procesada a Z0 para igualar sus escalas."""
        vals = []
        if self.raster_G is not None:
            clean_g = self.raster_G[np.isfinite(self.raster_G)]
            if len(clean_g) > 0:
                vals.append(clean_g)
        if self.last_results is not None and 'G_plano' in self.last_results:
            clean_z = self.last_results['G_plano'][np.isfinite(self.last_results['G_plano'])]
            if len(clean_z) > 0:
                vals.append(clean_z)
        if vals:
            v_min = float(min(np.min(v) for v in vals))
            v_max = float(max(np.max(v) for v in vals))
            return v_min, v_max
        return None, None

    def _setup_axes_style(self, ax, title, cbar_label, data, extent, layer_key, vmin=None, vmax=None, custom_levels=None):
        if title:
            ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        u = self.unit_xy
        ax.set_xlabel(f"Coordenada Este X [{u}]", fontsize=8)
        ax.set_ylabel(f"Coordenada Norte Y [{u}]", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.ticklabel_format(style='plain', useOffset=False)
        cmap = self._get_layer_cmap_name(layer_key)
        im = ax.imshow(data, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal', adjustable='box')
        self._draw_contours_for_layer(ax, layer_key, data, custom_levels=custom_levels)
        return im

    def plot_tab_result(self):
        fig = self.canvas_res['fig']
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)

        if self.last_results is not None and 'G_plano' in self.last_results:
            data = self.last_results['G_plano']
            z0 = self.last_results['Z0']
            extent = self._get_extent()
            g_min, g_max = self._get_shared_gravity_limits()
            levels = np.linspace(g_min, g_max, self.chart_settings['gz0']['n_contours']) if (g_min is not None and g_max is not None) else None
            im = self._setup_axes_style(ax, f"Gravedad fuente equivalente a Z0 = {z0:.1f} m", "Gravedad [mGal]", data, extent, layer_key='gz0', vmin=g_min, vmax=g_max, custom_levels=levels)
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("Gravedad Continuada Z0 [mGal]", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Aún no se ha calculado la gravedad continuada a Z0.\nPresiona el botón 'Calcular Fuente Equivalente'.", ha='center', va='center', fontsize=11, color='#666666')
            ax.set_axis_off()

        fig.tight_layout()
        self.canvas_res['canvas'].draw()

    def plot_tab_mdt(self):
        fig = self.canvas_mdt['fig']
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)

        if self.raster_Z is not None:
            extent = self._get_extent()
            im = self._setup_axes_style(ax, "Modelo Digital de Terreno (MDT Z_obs)", "Cota [m]", self.raster_Z, extent, layer_key='mdt')
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("Elevación Topográfica Z [m]", fontsize=8)
        else:
            ax.text(0.5, 0.5, "MDT no cargado.", ha='center', va='center', fontsize=11, color='#666666')
            ax.set_axis_off()

        fig.tight_layout()
        self.canvas_mdt['canvas'].draw()

    def plot_tab_gobs(self):
        fig = self.canvas_gobs['fig']
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)

        if self.raster_G is not None:
            extent = self._get_extent()
            g_min, g_max = self._get_shared_gravity_limits()
            levels = np.linspace(g_min, g_max, self.chart_settings['gobs']['n_contours']) if (g_min is not None and g_max is not None) else None
            im = self._setup_axes_style(ax, "Mediciones Gravimétricas Observadas (G_obs)", "Gravedad [mGal]", self.raster_G, extent, layer_key='gobs', vmin=g_min, vmax=g_max, custom_levels=levels)
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("Gravedad Observada [mGal]", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Gravedad observada no cargada.", ha='center', va='center', fontsize=11, color='#666666')
            ax.set_axis_off()

        fig.tight_layout()
        self.canvas_gobs['canvas'].draw()

    def plot_tab_rho(self):
        fig = self.canvas_rho['fig']
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)

        if self.last_results is not None and 'rho_grid' in self.last_results:
            data = self.last_results['rho_grid']
            extent = self._get_extent()
            im = self._setup_axes_style(ax, "Densidad de la Capa de Fuentes Equivalentes (rho_obs)", "Densidad [kg/m³]", data, extent, layer_key='rho')
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("Densidad Equivalente rho [kg/m³]", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Densidad de capa aún no calculada.\nRequiere ejecutar la inversión.", ha='center', va='center', fontsize=11, color='#666666')
            ax.set_axis_off()

        fig.tight_layout()
        self.canvas_rho['canvas'].draw()

    def plot_tab_quad(self):
        fig = self.canvas_quad['fig']
        fig.clf()

        if self.last_results is not None and self.raster_G is not None and self.raster_Z is not None:
            extent = self._get_extent()
            dx = abs(extent[1] - extent[0])
            dy = abs(extent[3] - extent[2])
            ar = (dx / dy) if dy > 0 else 1.0

            # Tamaño real del canvas en pulgadas
            w_fig, h_fig = fig.get_size_inches()

            # Cálculo de proporciones reales para agrupar las columnas centradas y sin huecos desproporcionados
            h_row_in = (h_fig * 0.94) / 2.0
            map_w_in = h_row_in * ar
            cbar_w_in = 0.50
            col_w_in = map_w_in + cbar_w_in
            gap_in = 0.60
            total_w_in = 2.0 * col_w_in + gap_in

            # Centrado balanceado: las 2 columnas quedan centradas simétricamente
            # con exactamente el mismo margen a izquierda y derecha, eliminando el desvío a la derecha.
            if total_w_in < w_fig:
                left_in = (w_fig - total_w_in) / 2.0
                left_frac = max(0.045, left_in / w_fig)
                right_frac = min(0.98, (left_in + total_w_in) / w_fig)
            else:
                left_frac = 0.04
                right_frac = 0.98

            gs = gridspec.GridSpec(2, 2, figure=fig, left=left_frac, right=right_frac, top=0.985, bottom=0.035, hspace=0.06, wspace=0.18)
            ax1 = fig.add_subplot(gs[0, 0]) # 1. Top-Left: Anomalia observada (G_obs)
            ax2 = fig.add_subplot(gs[0, 1]) # 2. Top-Right: Anomalia procesada (G_Z0)
            ax3 = fig.add_subplot(gs[1, 0]) # 3. Bottom-Left: Topografia MDT (Z_obs)
            ax4 = fig.add_subplot(gs[1, 1]) # 4. Bottom-Right: Densidad capa equivalente (rho)

            self.ax_quad = {'gobs': ax1, 'gz0': ax2, 'mdt': ax3, 'rho': ax4}

            # Escala unificada compartida para Anomalía Observada y Anomalía Procesada
            g_min, g_max = self._get_shared_gravity_limits()
            levels_gobs = np.linspace(g_min, g_max, self.chart_settings['gobs']['n_contours']) if (g_min is not None and g_max is not None) else None
            levels_gz0 = np.linspace(g_min, g_max, self.chart_settings['gz0']['n_contours']) if (g_min is not None and g_max is not None) else None

            # 1. Top-Left: Anomalia Observada (G_obs) [Escala unificada]
            cmap1 = self._get_layer_cmap_name('gobs')
            im1 = ax1.imshow(self.raster_G, origin='lower', extent=extent, cmap=cmap1, vmin=g_min, vmax=g_max)
            self._draw_contours_for_layer(ax1, 'gobs', self.raster_G, custom_levels=levels_gobs)
            cb1 = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.012)
            cb1.set_label("Anomalia G_obs [mGal]", fontsize=8)
            cb1.ax.tick_params(labelsize=7)

            # 2. Top-Right: Anomalia Procesada (G_Z0) [Misma escala unificada exactamente]
            cmap2 = self._get_layer_cmap_name('gz0')
            im2 = ax2.imshow(self.last_results['G_plano'], origin='lower', extent=extent, cmap=cmap2, vmin=g_min, vmax=g_max)
            self._draw_contours_for_layer(ax2, 'gz0', self.last_results['G_plano'], custom_levels=levels_gz0)
            z0 = self.last_results['Z0']
            cb2 = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.012)
            cb2.set_label(f"Anomalia Procesada Z0={z0:.0f}m [mGal]", fontsize=8)
            cb2.ax.tick_params(labelsize=7)

            # 3. Bottom-Left: Topografia MDT (Z_obs)
            cmap3 = self._get_layer_cmap_name('mdt')
            im3 = ax3.imshow(self.raster_Z, origin='lower', extent=extent, cmap=cmap3)
            self._draw_contours_for_layer(ax3, 'mdt', self.raster_Z)
            cb3 = fig.colorbar(im3, ax=ax3, fraction=0.035, pad=0.012)
            cb3.set_label("Topografia MDT [m]", fontsize=8)
            cb3.ax.tick_params(labelsize=7)

            # 4. Bottom-Right: Densidad Capa Equivalente (rho)
            cmap4 = self._get_layer_cmap_name('rho')
            im4 = ax4.imshow(self.last_results['rho_grid'], origin='lower', extent=extent, cmap=cmap4)
            self._draw_contours_for_layer(ax4, 'rho', self.last_results['rho_grid'])
            cb4 = fig.colorbar(im4, ax=ax4, fraction=0.035, pad=0.012)
            cb4.set_label("Densidad Capa rho [kg/m³]", fontsize=8)
            cb4.ax.tick_params(labelsize=7)

            # Formato y estilo: sin títulos, aspecto cuadrado, ticks limpios
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_aspect('equal', adjustable='box')
                ax.tick_params(labelsize=7, pad=2)
                ax.ticklabel_format(style='plain', useOffset=False)

        else:
            self.ax_quad = {}
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "Vista cuadruple disponible tras calcular la fuente equivalente.", ha='center', va='center', fontsize=11, color='#666666')
            ax.set_axis_off()

        self.canvas_quad['canvas'].draw()

    def on_canvas_mouse_move(self, event):
        """Muestra las coordenadas y el valor en tiempo real al pasar el mouse sobre cualquier mapa."""
        if event.inaxes and event.xdata is not None and event.ydata is not None:
            x = event.xdata
            y = event.ydata
            u = self.unit_xy
            txt_val = ""
            if self.raster_x is not None and self.raster_y is not None:
                ix = int(np.clip(np.searchsorted(self.raster_x, x), 0, len(self.raster_x) - 1))
                iy = int(np.clip(np.searchsorted(self.raster_y, y), 0, len(self.raster_y) - 1))
                tab_idx = self.tabs_view.currentIndex()
                if tab_idx == 0 and self.raster_G is not None:
                    v = self.raster_G[iy, ix]
                    txt_val = f" | G_obs: {v:.3f} mGal"
                elif tab_idx == 1 and self.last_results is not None:
                    v = self.last_results['G_plano'][iy, ix]
                    txt_val = f" | G_Z0: {v:.3f} mGal"
                elif tab_idx == 2 and self.raster_Z is not None:
                    v = self.raster_Z[iy, ix]
                    txt_val = f" | MDT: {v:.1f} m"
                elif tab_idx == 3 and self.last_results is not None:
                    v = self.last_results['rho_grid'][iy, ix]
                    txt_val = f" | ρ: {v:.2f} kg/m³"
                elif tab_idx == 4 and getattr(self, 'ax_quad', None):
                    if event.inaxes == self.ax_quad.get('gobs') and self.raster_G is not None:
                        v = self.raster_G[iy, ix]
                        txt_val = f" | [Anomalía G_obs: {v:.3f} mGal]"
                    elif event.inaxes == self.ax_quad.get('gz0') and self.last_results is not None:
                        v = self.last_results['G_plano'][iy, ix]
                        txt_val = f" | [Anomalía Procesada Z0: {v:.3f} mGal]"
                    elif event.inaxes == self.ax_quad.get('mdt') and self.raster_Z is not None:
                        v = self.raster_Z[iy, ix]
                        txt_val = f" | [Topografía MDT: {v:.1f} m]"
                    elif event.inaxes == self.ax_quad.get('rho') and self.last_results is not None:
                        v = self.last_results['rho_grid'][iy, ix]
                        txt_val = f" | [Densidad rho: {v:.2f} kg/m³]"

            self.lbl_coord_readout.setText(f"Coordenadas: (X: {x:.2f} {u}, Y: {y:.2f} {u}){txt_val}")

    # ==========================================================================
    # TRANSFERENCIA AL PROYECTO PRINCIPAL & EXPORTACIÓN
    # ==========================================================================
    def inject_into_main_project(self):
        """Inyecta la malla resultante G_plano al proyecto principal para ser filtrada y visualizada."""
        if self.last_results is None or self.main_window is None:
            QMessageBox.warning(self, "Sin Resultados", "Primero debes calcular la fuente equivalente a Z0.")
            return

        G_plano = self.last_results['G_plano']
        Z0 = self.last_results['Z0']

        # 1. Almacenar como capa dedicada de fuente equivalente en la ventana principal
        self.main_window.grid_eq_source_z0 = G_plano.copy()
        self.main_window.grid_eq_source_z0_elev = Z0
        self.main_window.grid_eq_source_x = self.raster_x.copy()
        self.main_window.grid_eq_source_y = self.raster_y.copy()

        # 2. Agregar a la lista de capas del Visualizador Maestro si no existe
        combo = getattr(self.main_window, 'combo_master_layer', None)
        if combo is not None:
            idx = combo.findData('eq_source_z0')
            if idx == -1:
                combo.addItem(f" Gravedad Fuente Equivalente a Z0 ({Z0:.0f}m)", "eq_source_z0")
                idx = combo.findData('eq_source_z0')
            idx_filt = combo.findData('eq_source_z0_filt')
            if idx_filt == -1:
                combo.addItem(f" Gravedad Fuente Equivalente Filtrada", "eq_source_z0_filt")

        # 3. Opción: Establecerla como la malla activa de trabajo (grid_Z) para filtros Fourier 2D
        if self.chk_set_as_active.isChecked():
            if getattr(self.main_window, 'grid_Z_orig', None) is None and getattr(self.main_window, 'grid_Z', None) is not None:
                self.main_window.grid_Z_orig = self.main_window.grid_Z.copy()
            self.main_window.grid_Z = G_plano.copy()
            self.main_window.grid_x_vec = self.raster_x.copy()
            self.main_window.grid_y_vec = self.raster_y.copy()
            self.main_window.z_col = f"G_EqSource_Z0_{Z0:.0f}m"
            self.main_window.unit_z = "mGal"

            # Recomputar Fourier 2D sobre la nueva malla horizontal plana
            self.main_window.recompute_fourier_2d()
            self.main_window.plot_master_viewer()
            self.main_window.plot_fourier_dashboard()

            # Notificar al modelador sintético
            if hasattr(self.main_window, 'widget_synth') and hasattr(self.main_window.widget_synth, 'refresh_available_obs_grids'):
                self.main_window.widget_synth.refresh_available_obs_grids()

            # Cambiar combo_master_layer al resultado
            if combo is not None and idx != -1:
                combo.setCurrentIndex(idx)

            if hasattr(self.main_window, 'tabs'):
                self.main_window.tabs.setCurrentIndex(2) # Visualizador Maestro

            if hasattr(self.main_window, 'status'):
                self.main_window.status.showMessage(
                    f" Malla de Gravedad Fuente Equivalente (Z0={Z0:.1f} m) inyectada y cargada para análisis Fourier 2D."
                )

            QMessageBox.information(
                self, "Inyección Exitosa",
                f"La malla continuada 'Gravedad fuente equivalente a Z0' ({Z0:.1f} m) "
                f"ha sido transferida e inyectada con éxito en el proyecto principal.\n\n"
                f"Ya está seleccionada como malla activa en:\n"
                f"• Pestaña 3: Visualizador Maestro 2D\n"
                f"• Pestaña 4: Filtros & Análisis Fourier 2D"
            )
        else:
            if combo is not None and idx != -1:
                combo.setCurrentIndex(idx)
            self.main_window.plot_master_viewer()
            QMessageBox.information(
                self, "Capa Agregada",
                f"La capa 'Gravedad fuente equivalente a Z0' ha sido registrada en el Visualizador Maestro 2D."
            )

    def export_csv_xyz(self):
        if self.last_results is None:
            QMessageBox.warning(self, "Sin Datos", "No hay malla calculada a Z0 para exportar.")
            return

        fpath, _ = QFileDialog.getSaveFileName(
            self, "Guardar Gravedad Z0 (CSV XYZ)", os.getcwd(), "CSV (*.csv)"
        )
        if fpath:
            G_p = self.last_results['G_plano']
            Xi, Yi = np.meshgrid(self.raster_x, self.raster_y)
            df = pd.DataFrame({
                f"X_{self.unit_xy}": Xi.ravel(),
                f"Y_{self.unit_xy}": Yi.ravel(),
                "G_Z0_mGal": G_p.ravel(),
                "Z0_meters": np.full_like(G_p.ravel(), self.last_results['Z0'])
            })
            df.to_csv(fpath, index=False)
            QMessageBox.information(self, "Exportación Exitosa", f"Malla CSV XYZ guardada en:\n{fpath}")

    def export_surfer_grid(self):
        if self.last_results is None:
            QMessageBox.warning(self, "Sin Datos", "No hay malla calculada a Z0 para exportar.")
            return

        fpath, _ = QFileDialog.getSaveFileName(
            self, "Guardar Gravedad Z0 (Surfer ASCII Grid .grd)", os.getcwd(), "Surfer Grid (*.grd)"
        )
        if fpath:
            G_p = self.last_results['G_plano']
            ny, nx = G_p.shape
            xmin, xmax = float(self.raster_x[0]), float(self.raster_x[-1])
            ymin, ymax = float(self.raster_y[0]), float(self.raster_y[-1])
            zmin, zmax = float(np.min(G_p)), float(np.max(G_p))

            with open(fpath, 'w') as f:
                f.write("DSAA\n")
                f.write(f"{nx} {ny}\n")
                f.write(f"{xmin:.4f} {xmax:.4f}\n")
                f.write(f"{ymin:.4f} {ymax:.4f}\n")
                f.write(f"{zmin:.4f} {zmax:.4f}\n")
                for r in range(ny):
                    row_vals = G_p[r, :]
                    f.write(" ".join(f"{v:.4f}" for v in row_vals) + "\n")

            QMessageBox.information(self, "Exportación Exitosa", f"Malla Surfer .grd guardada en:\n{fpath}")
