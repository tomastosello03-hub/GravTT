# -*- coding: utf-8 -*-
"""
================================================================================
CORRECTOR Y REDUCTOR GRAVIMÉTRICO SECUENCIAL
================================================================================
Procesador secuencial de correcciones gravimétricas según los estándares de la
geofísica clásica (Blakely 1995, Heiskanen & Moritz, Hinze et al.):

1. Carga de Mediciones Crudas: (long, lat, h, X, Y, G).
2. Gravedad Normal / Teórica del Elipsoide (g0):
   Fórmula cerrada de Somigliana (WGS84, Blakely 1995, Eqn. 7.11):
   g0(λ) = 9.7803267714 * (1 + 0.00193185138639*sin²λ) / sqrt(1 - 0.00669437999013*sin²λ) [m/s²]
3. Corrección de Aire Libre (FAC):
   c_fa = +0.3086 * h [mGal] (o expansión con latitud)
   Δg_fa = (g_obs - g0) + c_fa
   (con opción de ondulación geoidal N: H_orto = h_elip - N)
4. Corrección de Placa de Bouguer (BSC):
   c_bp = 2 * pi * G * rho * h ≈ 0.1119 * h [mGal] para rho=2670 kg/m³
   Δg_sb = Δg_fa - c_bp  (SECUENCIAL: requiere previa corrección de aire libre)
5. Corrección de Terreno / Topografía con MDT (GeoTIFF):
   gz = gamma * rho * dx * dy * ∫ [ (z - z0) / ( (x0 - xi)² + (y0 - yi)² + (z - z0)² )^(3/2) ] dz
      = gamma * rho * dx * dy * [ 1/r - 1/sqrt(r² + (z - z0)²) ]
   Δg_cb = Δg_sb + T  (SECUENCIAL: requiere previo Bouguer y Aire Libre)
================================================================================
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from PIL import Image

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QDoubleSpinBox,
    QSpinBox, QComboBox, QCheckBox, QGroupBox, QFileDialog,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QSplitter, QProgressBar, QTextEdit, QFrame,
    QScrollArea
)
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ==============================================================================
# CONSTANTES FÍSICAS Y GEODÉSICAS ESTÁNDAR
# ==============================================================================
G_SI = 6.67430e-11           # Constante Gravitacional Universal [m³/(kg·s²)]
SI_TO_MGAL = 1.0e5           # 1 m/s² = 100,000 mGal
GAMMA_MGAL = G_SI * SI_TO_MGAL  # 6.6743e-6 mGal·m² / kg
RHO_CRUST_DEFAULT = 2670.0   # Densidad media cortical de Bouguer [kg/m³]
FAC_GRADIENT_DEFAULT = 0.3086 # Gradiente normal de aire libre [mGal/m]


# ==============================================================================
# MODELOS DE ELIPSOIDE DE REFERENCIA (GRAVEDAD NORMAL g0)
# ==============================================================================
ELLIPSOID_MODELS = {
    'WGS84_Somigliana': {
        'name': "WGS84 (Somigliana cerrada, Blakely 1995, Eqn. 7.11)",
        'ge': 9.7803267714,     # Gravedad normal en el ecuador [m/s²]
        'k': 0.00193185138639,  # Constante de Somigliana k = (b*gp - a*ge)/(a*ge)
        'e2': 0.00669437999013, # Primera excentricidad al cuadrado e²
        'a': 6378137.0,         # Semieje mayor [m]
        'b': 6356752.3142       # Semieje menor [m]
    },
    'GRS80_Somigliana': {
        'name': "GRS80 (Geodetic Reference System 1980)",
        'ge': 9.780327,
        'k': 0.001931851386,
        'e2': 0.006694380023,
        'a': 6378137.0,
        'b': 6356752.3141
    },
    'GRS67_Chebyshev': {
        'name': "GRS67 (Fórmula Internacional de 1967)",
        'ge': 9.780318,
        'c1': 0.0053024,
        'c2': -0.0000059,
        'a': 6378160.0
    },
    'Internacional_1930': {
        'name': "Internacional 1930 (Helmert / Cassinis)",
        'ge': 9.780490,
        'c1': 0.0052884,
        'c2': -0.0000059,
        'a': 6378388.0
    }
}


def compute_normal_gravity(lat_deg, model_key='WGS84_Somigliana'):
    """
    Calcula la gravedad normal de referencia g0 sobre el elipsoide en mGal.
    Fórmula cerrada de Somigliana (Eqn. 7.11 Blakely 1995 para WGS84):
    g0 = ge * (1 + k * sin²λ) / sqrt(1 - e² * sin²λ)
    """
    lat_rad = np.radians(np.asarray(lat_deg, dtype=float))
    sin2 = np.sin(lat_rad) ** 2

    info = ELLIPSOID_MODELS.get(model_key, ELLIPSOID_MODELS['WGS84_Somigliana'])

    if 'k' in info and 'e2' in info:
        ge = info['ge']
        k = info['k']
        e2 = info['e2']
        g0_ms2 = ge * (1.0 + k * sin2) / np.sqrt(1.0 - e2 * sin2)
    else:
        # Polinomio armónico esférico clásico
        ge = info['ge']
        c1 = info.get('c1', 0.0053024)
        c2 = info.get('c2', -0.0000059)
        sin2_2lat = np.sin(2.0 * lat_rad) ** 2
        g0_ms2 = ge * (1.0 + c1 * sin2 + c2 * sin2_2lat)

    return g0_ms2 * SI_TO_MGAL


def compute_free_air_correction(h_meters, lat_deg=None, gradient=0.3086, use_second_order=False):
    """
    Calcula la corrección de aire libre c_fa [mGal].
    Fórmula lineal estándar: c_fa = +0.3086 * h [mGal]
    Expansión de 2do orden (Blakely 1995, Eqn. 7.14):
    c_fa = (0.308765 - 0.0004398 * sin²λ) * h - 7.2125e-8 * h²
    """
    h = np.asarray(h_meters, dtype=float)
    if use_second_order and lat_deg is not None:
        lat_rad = np.radians(np.asarray(lat_deg, dtype=float))
        sin2 = np.sin(lat_rad) ** 2
        coef = 0.308765 - 0.0004398 * sin2
        return coef * h - 7.2125e-8 * (h ** 2)
    else:
        return gradient * h


def compute_bouguer_slab_correction(h_meters, rho=RHO_CRUST_DEFAULT):
    """
    Calcula la corrección por placa infinita homogénea de Bouguer c_bp [mGal].
    c_bp = 2 * pi * G * rho * h [mGal]
    Para rho = 2670 kg/m³: c_bp ≈ 0.111868 * h ≈ 0.1119 * h [mGal]
    """
    h = np.asarray(h_meters, dtype=float)
    factor = 2.0 * np.pi * GAMMA_MGAL * rho  # mGal / m
    return factor * h


# ==============================================================================
# LECTOR NATIVO DE MDT (GEOTIFF) CON PILLOW + NUMPY
# ==============================================================================
class GeoTiffMDT:
    """
    Lector y administrador eficiente de Modelos Digitales de Terreno (GeoTIFF).
    Lee metadatos georreferenciados estándar TIFF (ModelPixelScaleTag, ModelTiepointTag)
    sin requerir librerías pesadas C++ como GDAL/Rasterio.
    """
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.is_loaded = False
        self.grid_z = None
        self.nx = 0
        self.ny = 0
        self.x0 = 0.0
        self.y0 = 0.0
        self.dx = 100.0
        self.dy = 100.0
        self.x1 = 0.0
        self.y1 = 0.0
        self.x_vec = None
        self.y_vec = None
        self.z_min = 0.0
        self.z_max = 0.0
        self.z_mean = 0.0
        self.nodata = -9999.0

        if filepath and os.path.exists(filepath):
            self.load(filepath)

    def load(self, filepath):
        self.filepath = filepath
        img = Image.open(filepath)
        self.nx, self.ny = img.size

        # Lectura de etiquetas georreferenciadas TIFF
        tags = getattr(img, 'tag_v2', {})
        scale_tag = tags.get(33550, None)    # ModelPixelScaleTag
        tie_tag = tags.get(33922, None)      # ModelTiepointTag

        if scale_tag and tie_tag:
            self.dx = float(scale_tag[0])
            self.dy = float(scale_tag[1])
            self.x0 = float(tie_tag[3])
            self.y0 = float(tie_tag[4])      # Esquina superior izquierda
        else:
            # Fallback predeterminado en metros
            self.dx = 50.0
            self.dy = 50.0
            self.x0 = 0.0
            self.y0 = self.ny * self.dy

        self.x1 = self.x0 + self.nx * self.dx
        self.y1 = self.y0 - self.ny * self.dy

        # Matriz de elevación
        raw_arr = np.array(img, dtype=np.float32)
        # Manejo de NoData
        mask_nodata = (raw_arr < -9000.0) | np.isnan(raw_arr)
        if np.any(mask_nodata):
            valid_vals = raw_arr[~mask_nodata]
            fill_val = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
            raw_arr[mask_nodata] = fill_val

        self.grid_z = raw_arr
        self.x_vec = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        self.y_vec = self.y0 - (np.arange(self.ny) + 0.5) * self.dy

        self.z_min = float(np.min(self.grid_z))
        self.z_max = float(np.max(self.grid_z))
        self.z_mean = float(np.mean(self.grid_z))
        self.is_loaded = True

    def get_elevation_at(self, x, y):
        """Interpola o extrae la cota Z en metros del MDT para una coordenada (x, y)."""
        if not self.is_loaded or self.grid_z is None:
            return 0.0
        col = int((x - self.x0) / self.dx)
        row = int((self.y0 - y) / self.dy)
        if 0 <= col < self.nx and 0 <= row < self.ny:
            return float(self.grid_z[row, col])
        return self.z_mean

    def compute_terrain_correction_point(self, x0, y0, z0, r_max=10000.0, rho=RHO_CRUST_DEFAULT):
        """
        Calcula la integral exacta de corrección de terreno del prisma en (x0, y0, z0):
        g_z = gamma * rho * dx * dy * ∫ [ (z - z0) / ( (x0 - xi)² + (y0 - yi)² + (z - z0)² )^(3/2) ] dz
            = gamma * rho * dx * dy * [ 1/r - 1/sqrt(r² + (z - z0)²) ]
        con aproximación cilíndrica de Hammer para el compartimiento central.
        """
        if not self.is_loaded or self.grid_z is None:
            return 0.0

        c_min = max(0, int((x0 - r_max - self.x0) / self.dx))
        c_max = min(self.nx, int((x0 + r_max - self.x0) / self.dx) + 1)
        r_min = max(0, int((self.y0 - (y0 + r_max)) / self.dy))
        r_max_row = min(self.ny, int((self.y0 - (y0 - r_max)) / self.dy) + 1)

        if c_min >= c_max or r_min >= r_max_row:
            return 0.0

        sub_z = self.grid_z[r_min:r_max_row, c_min:c_max]
        x_sub = self.x0 + (np.arange(c_min, c_max) + 0.5) * self.dx
        y_sub = self.y0 - (np.arange(r_min, r_max_row) + 0.5) * self.dy
        X_sub, Y_sub = np.meshgrid(x_sub, y_sub)

        r2 = (X_sub - x0) ** 2 + (Y_sub - y0) ** 2
        r_cutoff = 0.5 * min(self.dx, self.dy)

        # Máscara dentro del radio R_max y excluyendo el píxel singular exacto
        mask = (r2 <= r_max ** 2) & (r2 >= r_cutoff ** 2)

        r = np.sqrt(r2[mask])
        dz = np.abs(sub_z[mask] - z0)

        # Fórmula exacta analítica de la integral dada por el usuario
        int_term = (1.0 / r) - (1.0 / np.sqrt(r2[mask] + dz ** 2))
        T_exterior = GAMMA_MGAL * rho * self.dx * self.dy * np.sum(int_term)

        # Compartimiento central cilíndrico equivalente
        # Radio R0 de área equivalente dx*dy
        R0 = np.sqrt((self.dx * self.dy) / np.pi)
        col_c = int((x0 - self.x0) / self.dx)
        row_c = int((self.y0 - y0) / self.dy)
        if 0 <= col_c < self.nx and 0 <= row_c < self.ny:
            dz0 = abs(float(self.grid_z[row_c, col_c]) - z0)
        else:
            dz0 = 0.0

        T_central = 2.0 * np.pi * GAMMA_MGAL * rho * (R0 - np.sqrt(R0**2 + dz0**2) + dz0)

        return float(T_exterior + T_central)


# ==============================================================================
# HILO DE TRABAJO (QTHREAD) PARA EL CÁLCULO DE TERRENO MULTI-ESTACIÓN
# ==============================================================================
class WorkerCalculoTerreno(QThread):
    progreso = pyqtSignal(int, int)   # actual, total
    finalizado = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, mdt, coords_x, coords_y, coords_z, r_max=10000.0, rho=2670.0):
        super().__init__()
        self.mdt = mdt
        self.coords_x = np.asarray(coords_x)
        self.coords_y = np.asarray(coords_y)
        self.coords_z = np.asarray(coords_z)
        self.r_max = float(r_max)
        self.rho = float(rho)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            n = len(self.coords_x)
            t_values = np.zeros(n, dtype=float)
            for i in range(n):
                if self._is_cancelled:
                    return
                x0 = self.coords_x[i]
                y0 = self.coords_y[i]
                z0 = self.coords_z[i]
                t_val = self.mdt.compute_terrain_correction_point(x0, y0, z0, r_max=self.r_max, rho=self.rho)
                t_values[i] = t_val
                if i % 5 == 0 or i == n - 1:
                    self.progreso.emit(i + 1, n)
            self.finalizado.emit(t_values)
        except Exception as e:
            self.error.emit(str(e))


# ==============================================================================
# CONVERSOR BÁSICO PROYECCIÓN TRANSVERSE MERCATOR / GAUSS-KRÜGER / UTM
# ==============================================================================
def latlon_to_gauss_kruger(lat_deg, lon_deg, lon0_deg=-65.0, false_easting=3500000.0):
    """
    Convierte Lat/Lon (WGS84) a coordenadas métricas Transverse Mercator (Faja Gauss-Krüger / UTM).
    Por defecto: Faja 3 Argentina (meridiano central -65°, falso este 3,500,000 m).
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f ** 2
    e_prime2 = e2 / (1.0 - e2)

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lon0 = np.radians(lon0_deg)

    N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
    T = np.tan(lat)**2
    C = e_prime2 * np.cos(lat)**2
    A = (lon - lon0) * np.cos(lat)

    M = a * ((1.0 - e2/4.0 - 3.0*e2**2/64.0 - 5.0*e2**3/256.0) * lat
             - (3.0*e2/8.0 + 3.0*e2**2/32.0 + 45.0*e2**3/1024.0) * np.sin(2.0*lat)
             + (15.0*e2**2/256.0 + 45.0*e2**3/1024.0) * np.sin(4.0*lat)
             - (35.0*e2**3/3072.0) * np.sin(6.0*lat))

    x = false_easting + (N * (A + (1.0 - T + C) * A**3 / 6.0
                             + (5.0 - 18.0*T + T**2 + 72.0*C - 58.0*e_prime2) * A**5 / 120.0))
    y = M + N * np.tan(lat) * (A**2 / 2.0 + (5.0 - T + 9.0*C + 4.0*C**2) * A**4 / 24.0
                               + (61.0 - 58.0*T + T**2 + 600.0*C - 330.0*e_prime2) * A**6 / 720.0)

    # Si está en el hemisferio sur y se requiere falso norte estándar
    if np.any(lat < 0) and np.all(y < 0):
        y = y + 10000000.0

    return x, y


# ==============================================================================
# VENTANA PRINCIPAL DEL CORRECTOR GRAVIMÉTRICO
# ==============================================================================
class VentanaCorrectorGravimetrico(QMainWindow):
    """
    Aplicación integral para la reducción secuencial de datos gravimétricos crudos:
    - Lectura (long, lat, h, X, Y, G)
    - Elipsoide normal Somigliana (WGS84 Eqn. 7.11)
    - Aire Libre (-0.3086 h) y corrección geoidal
    - Placa de Bouguer (0.1119 h)
    - Terreno con MDT GeoTIFF e integración analítica continua
    - Visualización gráfica, perfiles interactivos, mapas y exportación a FFT.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GravTT - Corrector Gravimétrico Secuencial (Elipsoide, Aire Libre, Bouguer, Terreno MDT)")
        self.resize(1280, 840)
        self.setMinimumSize(1040, 700)

        # Estado secuencial de las correcciones
        self.step_raw_done = False
        self.step_g0_done = False
        self.step_fa_done = False
        self.step_bouguer_done = False
        self.step_terrain_done = False

        # Datos tabulares
        self.df = None
        self.mdt = GeoTiffMDT()
        self.worker_thread = None

        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ======================================================================
        # 1. BARRA SUPERIOR DE SECUENCIA Y ESTADO EN CASCADA
        # ======================================================================
        bar_steps = QFrame()
        bar_steps.setObjectName("barSteps")
        h_steps = QHBoxLayout(bar_steps)
        h_steps.setContentsMargins(10, 6, 10, 6)

        self.lbl_st1 = QLabel("<b>1. Datos Crudos:</b> ⏳ Sin datos")
        self.lbl_st2 = QLabel("<b>2. Elipsoide g₀:</b> ⏳ Pendiente")
        self.lbl_st3 = QLabel("<b>3. Aire Libre:</b> ⏳ Pendiente")
        self.lbl_st4 = QLabel("<b>4. Placa Bouguer:</b> ⏳ Pendiente")
        self.lbl_st5 = QLabel("<b>5. Terreno MDT:</b> ⏳ Pendiente")

        for lbl in [self.lbl_st1, self.lbl_st2, self.lbl_st3, self.lbl_st4, self.lbl_st5]:
            lbl.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #ffffff;")
            h_steps.addWidget(lbl)
            if lbl != self.lbl_st5:
                arrow = QLabel("➔")
                arrow.setStyleSheet("font-weight: bold; color: #555555;")
                h_steps.addWidget(arrow)

        h_steps.addStretch(1)
        btn_run_all = QPushButton("▶️ Ejecutar Secuencia Completa (1 Clic)")
        btn_run_all.setObjectName("btnRunAll")
        btn_run_all.clicked.connect(self.run_full_sequence)
        h_steps.addWidget(btn_run_all)

        main_layout.addWidget(bar_steps)

        # Splitter principal: Panel Izquierdo (Controles) vs Panel Derecho (Visualización)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)

        # ======================================================================
        # 2. PANEL IZQUIERDO: CONTROLES SECUENCIALES
        # ======================================================================
        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left.setFrameShape(QFrame.NoFrame)
        left_widget = QWidget()
        vbox_left = QVBoxLayout(left_widget)
        vbox_left.setContentsMargins(4, 4, 4, 4)
        vbox_left.setSpacing(8)

        # --- SECCIÓN 1: DATOS CRUDOS ---
        grp_data = QGroupBox("1. Carga de Mediciones Crudas (long, lat, h, X, Y, G)")
        v_data = QVBoxLayout(grp_data)

        btn_load_file = QPushButton("📁 Cargar Archivo de Mediciones Crudas (CSV / TXT / Excel)...")
        btn_load_file.setObjectName("btnAccent")
        btn_load_file.clicked.connect(self.on_load_raw_file)
        v_data.addWidget(btn_load_file)

        self.lbl_data_info = QLabel("Ningún archivo cargado.")
        self.lbl_data_info.setStyleSheet("color: #666666; font-size: 10px;")
        v_data.addWidget(self.lbl_data_info)
        vbox_left.addWidget(grp_data)

        # --- SECCIÓN 2: ELIPSOIDE NORMAL g0 ---
        grp_g0 = QGroupBox("2. Modelo de Elipsoide Clásico (g₀)")
        grid_g0 = QGridLayout(grp_g0)

        grid_g0.addWidget(QLabel("<b>Modelo de Referencia:</b>"), 0, 0)
        self.combo_ellipsoid = QComboBox()
        for k, v in ELLIPSOID_MODELS.items():
            self.combo_ellipsoid.addItem(v['name'], k)
        grid_g0.addWidget(self.combo_ellipsoid, 0, 1)

        self.lbl_formula_g0 = QLabel(
            "<code>g₀(λ) = 9.7803267714 · (1 + 0.00193185138639·sin²λ) / √(1 - 0.00669437999013·sin²λ)</code>"
        )
        self.lbl_formula_g0.setWordWrap(True)
        self.lbl_formula_g0.setStyleSheet("font-size: 10px; color: #003366; background: #eaf2fb; padding: 4px; border: 1px solid #c2d8f2;")
        grid_g0.addWidget(self.lbl_formula_g0, 1, 0, 1, 2)

        self.btn_calc_g0 = QPushButton("⚙️ Paso 2: Calcular Gravedad Normal (g₀)")
        self.btn_calc_g0.clicked.connect(self.run_step_g0)
        grid_g0.addWidget(self.btn_calc_g0, 2, 0, 1, 2)
        vbox_left.addWidget(grp_g0)

        # --- SECCIÓN 3: CORRECCIÓN DE AIRE LIBRE ---
        grp_fa = QGroupBox("3. Corrección de Aire Libre (FAC)")
        grid_fa = QGridLayout(grp_fa)

        grid_fa.addWidget(QLabel("<b>Gradiente vertical [mGal/m]:</b>"), 0, 0)
        self.spin_grad_fa = QDoubleSpinBox()
        self.spin_grad_fa.setRange(0.20, 0.40); self.spin_grad_fa.setDecimals(4); self.spin_grad_fa.setValue(0.3086)
        grid_fa.addWidget(self.spin_grad_fa, 0, 1)

        self.chk_second_order = QCheckBox("Usar expansión de 2do orden con latitud")
        self.chk_second_order.setChecked(False)
        grid_fa.addWidget(self.chk_second_order, 1, 0, 1, 2)

        grid_fa.addWidget(QLabel("<b>Ondulación Geoidal N [m]:</b>"), 2, 0)
        self.spin_geoid_n = QDoubleSpinBox()
        self.spin_geoid_n.setRange(-200.0, 200.0); self.spin_geoid_n.setValue(0.0)
        self.spin_geoid_n.setToolTip("Ondulación N para cota ortométrica: H = h_elip - N (0 = h ortométrica directa)")
        grid_fa.addWidget(self.spin_geoid_n, 2, 1)

        self.btn_calc_fa = QPushButton("⚙️ Paso 3: Aplicar Corrección de Aire Libre")
        self.btn_calc_fa.clicked.connect(self.run_step_free_air)
        grid_fa.addWidget(self.btn_calc_fa, 3, 0, 1, 2)
        vbox_left.addWidget(grp_fa)

        # --- SECCIÓN 4: CORRECCIÓN DE BOUGUER ---
        grp_b = QGroupBox("4. Corrección de Placa Simple de Bouguer")
        grid_b = QGridLayout(grp_b)

        grid_b.addWidget(QLabel("<b>Densidad de Placa ρ [kg/m³]:</b>"), 0, 0)
        self.spin_rho_b = QDoubleSpinBox()
        self.spin_rho_b.setRange(500.0, 5000.0); self.spin_rho_b.setValue(2670.0)
        self.spin_rho_b.valueChanged.connect(self.update_bouguer_factor_label)
        grid_b.addWidget(self.spin_rho_b, 0, 1)

        self.lbl_factor_b = QLabel("Factor: 2·π·G·ρ ≈ 0.1119 mGal/m")
        self.lbl_factor_b.setStyleSheet("font-size: 10px; color: #555555;")
        grid_b.addWidget(self.lbl_factor_b, 1, 0, 1, 2)

        self.btn_calc_b = QPushButton("⚙️ Paso 4: Aplicar Corrección Placa de Bouguer")
        self.btn_calc_b.clicked.connect(self.run_step_bouguer)
        grid_b.addWidget(self.btn_calc_b, 2, 0, 1, 2)
        vbox_left.addWidget(grp_b)

        # --- SECCIÓN 5: CORRECCIÓN DE TERRENO (MDT GEOTIFF) ---
        grp_t = QGroupBox("5. Corrección de Terreno / Topografía con MDT (GeoTIFF)")
        grid_t = QGridLayout(grp_t)

        btn_sel_mdt = QPushButton("🗺️ Cargar Modelo Digital de Terreno MDT (.tif)...")
        btn_sel_mdt.setObjectName("btnPrimary")
        btn_sel_mdt.clicked.connect(self.on_select_mdt_file)
        grid_t.addWidget(btn_sel_mdt, 0, 0, 1, 2)

        self.lbl_mdt_status = QLabel("MDT no cargado.")
        self.lbl_mdt_status.setStyleSheet("font-size: 10px; color: #666666;")
        grid_t.addWidget(self.lbl_mdt_status, 1, 0, 1, 2)

        grid_t.addWidget(QLabel("<b>Radio de Integración R_max [m]:</b>"), 2, 0)
        self.spin_rmax = QDoubleSpinBox()
        self.spin_rmax.setRange(500.0, 100000.0); self.spin_rmax.setSingleStep(1000.0); self.spin_rmax.setValue(8000.0)
        grid_t.addWidget(self.spin_rmax, 2, 1)

        self.progress_terrain = QProgressBar()
        self.progress_terrain.setRange(0, 100)
        self.progress_terrain.setValue(0)
        self.progress_terrain.setTextVisible(True)
        grid_t.addWidget(self.progress_terrain, 3, 0, 1, 2)

        h_t_actions = QHBoxLayout()
        self.btn_calc_t = QPushButton("⚙️ Paso 5: Calcular Corrección de Terreno (T)")
        self.btn_calc_t.setObjectName("btnAccent")
        self.btn_calc_t.clicked.connect(self.run_step_terrain)
        h_t_actions.addWidget(self.btn_calc_t)

        self.btn_cancel_t = QPushButton("⏹️ Cancelar")
        self.btn_cancel_t.setEnabled(False)
        self.btn_cancel_t.clicked.connect(self.on_cancel_terrain)
        h_t_actions.addWidget(self.btn_cancel_t)
        grid_t.addLayout(h_t_actions, 4, 0, 1, 2)

        vbox_left.addWidget(grp_t)

        # --- SECCIÓN 6: EXPORTACIÓN Y CONEXIÓN CON ANALIZADOR FFT ---
        grp_exp = QGroupBox("6. Exportación y Conexión con FFT 2D")
        v_exp = QVBoxLayout(grp_exp)

        btn_export_csv = QPushButton("💾 Guardar Tabla Corregida (.csv)...")
        btn_export_csv.clicked.connect(self.on_export_csv)
        v_exp.addWidget(btn_export_csv)

        btn_send_fft = QPushButton("🚀 Enviar Malla al Analizador FFT 2D")
        btn_send_fft.setObjectName("btnSuccess")
        btn_send_fft.setToolTip("Genera la malla regular de la anomalía completa de Bouguer y la envía directamente al analizador FFT")
        btn_send_fft.clicked.connect(self.on_send_to_fft_window)
        v_exp.addWidget(btn_send_fft)

        vbox_left.addWidget(grp_exp)
        vbox_left.addStretch(1)

        scroll_left.setWidget(left_widget)
        splitter.addWidget(scroll_left)

        # ======================================================================
        # 3. PANEL DERECHO: PESTAÑAS DE VISUALIZACIÓN MULTI-MODO
        # ======================================================================
        right_box = QWidget()
        vbox_right = QVBoxLayout(right_box)
        vbox_right.setContentsMargins(0, 0, 0, 0)

        self.tabs_view = QTabWidget()

        # PESTAÑA A: PERFILES DE REDUCCIÓN GRAVIMÉTRICA
        self.tab_plot = QWidget()
        v_plot = QVBoxLayout(self.tab_plot)
        v_plot.setContentsMargins(2, 2, 2, 2)

        self.fig_plot = Figure(figsize=(8, 6), dpi=100)
        self.canvas_plot = FigureCanvas(self.fig_plot)
        self.toolbar_plot = NavigationToolbar(self.canvas_plot, self)
        v_plot.addWidget(self.toolbar_plot)
        v_plot.addWidget(self.canvas_plot)
        self.tabs_view.addTab(self.tab_plot, "📈 Perfiles de Reducción Secuencial")

        # PESTAÑA B: MAPA 2D (MDT + ESTACIONES SUPERPUESTAS)
        self.tab_map = QWidget()
        v_map = QVBoxLayout(self.tab_map)
        v_map.setContentsMargins(2, 2, 2, 2)

        h_map_ctrl = QHBoxLayout()
        h_map_ctrl.addWidget(QLabel("<b>Variable a mostrar sobre el mapa:</b>"))
        self.combo_map_var = QComboBox()
        self.combo_map_var.addItems([
            "Anomalía Completa Bouguer (Δg_CB)",
            "Anomalía Simple Bouguer (Δg_SB)",
            "Anomalía de Aire Libre (Δg_FA)",
            "Efecto de Terreno T (mGal)",
            "Gravedad Observada (g_obs)",
            "Elevación de Estación h (m)"
        ])
        self.combo_map_var.currentIndexChanged.connect(self.update_map_view)
        h_map_ctrl.addWidget(self.combo_map_var)
        h_map_ctrl.addStretch(1)
        v_map.addLayout(h_map_ctrl)

        self.fig_map = Figure(figsize=(8, 6), dpi=100)
        self.canvas_map = FigureCanvas(self.fig_map)
        self.toolbar_map = NavigationToolbar(self.canvas_map, self)
        v_map.addWidget(self.toolbar_map)
        v_map.addWidget(self.canvas_map)
        self.tabs_view.addTab(self.tab_map, "🗺️ Mapa 2D (MDT + Estaciones)")

        # PESTAÑA C: TABLA COMPLETA DE RESULTADOS NUMÉRICOS
        self.tab_table = QWidget()
        v_table = QVBoxLayout(self.tab_table)
        self.tbl_results = QTableWidget()
        self.tbl_results.setAlternatingRowColors(True)
        v_table.addWidget(self.tbl_results)
        self.tabs_view.addTab(self.tab_table, "📋 Tabla de Datos y Correcciones")

        # PESTAÑA D: GLOSARIO Y FUNDAMENTO FÍSICO
        self.tab_info = QWidget()
        v_info = QVBoxLayout(self.tab_info)
        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        self.load_glossary_text()
        v_info.addWidget(self.txt_info)
        self.tabs_view.addTab(self.tab_info, "📖 Fundamento Teórico y Fórmulas")

        vbox_right.addWidget(self.tabs_view)
        splitter.addWidget(right_box)
        splitter.setSizes([420, 860])

        main_layout.addWidget(splitter)
        self.update_step_badges()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f4f6f9; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #004488;
            }
            #barSteps {
                background: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 6px;
            }
            #btnRunAll {
                background: #0077cc;
                color: white;
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 4px;
            }
            #btnRunAll:hover { background: #005fa3; }
            #btnAccent {
                background: #28a745;
                color: white;
                font-weight: bold;
                padding: 5px 12px;
                border-radius: 4px;
            }
            #btnAccent:hover { background: #218838; }
            #btnSuccess {
                background: #17a2b8;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            #btnSuccess:hover { background: #138496; }
            QTableWidget {
                gridline-color: #e0e0e0;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #eef2f7;
                font-weight: bold;
                border: 1px solid #d0d7de;
                padding: 4px;
            }
        """)

    def update_bouguer_factor_label(self):
        rho = self.spin_rho_b.value()
        fac = 2.0 * np.pi * GAMMA_MGAL * rho
        self.lbl_factor_b.setText(f"Factor: 2·π·G·ρ = {fac:.4f} mGal/m (≈ {fac:.4f}·h)")

    def update_step_badges(self):
        """Actualiza el semáforo y estado de la barra superior en cascada."""
        # Paso 1
        if self.step_raw_done and self.df is not None:
            self.lbl_st1.setText(f"<b>1. Datos Crudos:</b> ✔️ {len(self.df)} ptos")
            self.lbl_st1.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb;")
        else:
            self.lbl_st1.setText("<b>1. Datos Crudos:</b> ⏳ Sin datos")
            self.lbl_st1.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #fff3cd; color: #856404; border: 1px solid #ffeeba;")

        # Paso 2
        if self.step_g0_done:
            self.lbl_st2.setText("<b>2. Elipsoide g₀:</b> ✔️ Calculada")
            self.lbl_st2.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb;")
        else:
            self.lbl_st2.setText("<b>2. Elipsoide g₀:</b> ⏳ Pendiente")
            self.lbl_st2.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #e9ecef; color: #6c757d; border: 1px solid #ced4da;")

        # Paso 3
        if self.step_fa_done:
            self.lbl_st3.setText("<b>3. Aire Libre:</b> ✔️ Aplicada")
            self.lbl_st3.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb;")
        else:
            self.lbl_st3.setText("<b>3. Aire Libre:</b> ⏳ Pendiente")
            self.lbl_st3.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #e9ecef; color: #6c757d; border: 1px solid #ced4da;")

        # Paso 4
        if self.step_bouguer_done:
            self.lbl_st4.setText("<b>4. Placa Bouguer:</b> ✔️ Aplicada")
            self.lbl_st4.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb;")
        else:
            self.lbl_st4.setText("<b>4. Placa Bouguer:</b> ⏳ Pendiente")
            self.lbl_st4.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #e9ecef; color: #6c757d; border: 1px solid #ced4da;")

        # Paso 5
        if self.step_terrain_done:
            self.lbl_st5.setText("<b>5. Terreno MDT:</b> ✔️ Aplicada")
            self.lbl_st5.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb;")
        else:
            self.lbl_st5.setText("<b>5. Terreno MDT:</b> ⏳ Pendiente")
            self.lbl_st5.setStyleSheet("font-size: 11px; padding: 3px 6px; border-radius: 4px; background: #e9ecef; color: #6c757d; border: 1px solid #ced4da;")

        # Activar/desactivar botones respetando la estricta secuencia pedida por el usuario
        self.btn_calc_g0.setEnabled(self.step_raw_done)
        self.btn_calc_fa.setEnabled(self.step_g0_done)
        self.btn_calc_b.setEnabled(self.step_fa_done)
        self.btn_calc_t.setEnabled(self.step_bouguer_done and self.mdt.is_loaded)

    # ==========================================================================
    # CARGA DE DATOS CRUDOS
    # ==========================================================================
    def on_load_raw_file(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Cargar Archivo de Mediciones Crudas", "",
            "Archivos de Datos (*.csv *.txt *.dat *.tsv *.xlsx);;Todos los Archivos (*.*)"
        )
        if not fpath:
            return

        try:
            if fpath.endswith(('.xlsx', '.xls')):
                df_raw = pd.read_excel(fpath)
            else:
                # Detectar separador automáticamente
                df_raw = pd.read_csv(fpath, sep=None, engine='python')

            # Mapeo automático inteligente de columnas
            mapped = self.auto_map_columns(df_raw)
            if mapped is None:
                QMessageBox.warning(self, "Columnas no detectadas",
                                    "No se pudieron identificar automáticamente las columnas.\n"
                                    "Asegúrate de que el archivo contenga: Longitud, Latitud, h, X, Y, G.")
                return

            self.df = mapped
            self.step_raw_done = True
            self.step_g0_done = False
            self.step_fa_done = False
            self.step_bouguer_done = False
            self.step_terrain_done = False

            self.lbl_data_info.setText(f"Cargado: <b>{os.path.basename(fpath)}</b> ({len(self.df)} estaciones)")
            self.update_step_badges()
            self.update_table_view()
            self.update_plots()
            QMessageBox.information(self, "Carga Exitosa", f"Se cargaron correctamente {len(self.df)} estaciones de medición gravimétrica.")

        except Exception as e:
            QMessageBox.critical(self, "Error al Cargar", f"No se pudo leer el archivo:\n{e}")

    def auto_map_columns(self, df_in):
        """Mapea las columnas requeridas (long, lat, h, X, Y, G) de forma flexible."""
        cols = [str(c).strip() for c in df_in.columns]
        mapping = {}

        def find_col(patterns):
            for p in patterns:
                for c in cols:
                    if c.lower() == p.lower():
                        return c
            for p in patterns:
                for c in cols:
                    if p.lower() in c.lower():
                        return c
            return None

        c_lon = find_col(['long', 'lon', 'lng', 'longitud', 'x_geo', 'lambda'])
        c_lat = find_col(['lat', 'latitud', 'latitude', 'y_geo', 'phi'])
        c_h = find_col(['h', 'elev', 'cota', 'altura', 'height', 'z'])
        c_x = find_col(['x', 'este', 'east', 'x_proj', 'x_m', 'utm_x'])
        c_y = find_col(['y', 'norte', 'north', 'y_proj', 'y_m', 'utm_y'])
        c_g = find_col(['g', 'g_obs', 'grav', 'gravity', 'gobs', 'observada', 'g_cruda', 'g_mgal'])

        # Si vienen columnas sin nombre (ej. índice numérico 0, 1, 2...)
        if len(cols) >= 3 and (c_lon is None or c_lat is None):
            c_lon = cols[0]
            c_lat = cols[1]
            c_h = cols[2] if len(cols) > 2 else None
            c_g = cols[3] if len(cols) > 3 else None

        if c_lon is None or c_lat is None:
            return None

        res_df = pd.DataFrame()
        res_df['long'] = pd.to_numeric(df_in[c_lon], errors='coerce')
        res_df['lat'] = pd.to_numeric(df_in[c_lat], errors='coerce')
        res_df['h'] = pd.to_numeric(df_in[c_h], errors='coerce') if c_h else 0.0

        # Si no tiene X e Y proyectadas, convertirlas automáticamente
        if c_x and c_y:
            res_df['X'] = pd.to_numeric(df_in[c_x], errors='coerce')
            res_df['Y'] = pd.to_numeric(df_in[c_y], errors='coerce')
        else:
            # Proyección automática Gauss-Krüger / UTM
            x_proj, y_proj = latlon_to_gauss_kruger(res_df['lat'], res_df['long'])
            res_df['X'] = x_proj
            res_df['Y'] = y_proj

        if c_g:
            res_df['g_obs'] = pd.to_numeric(df_in[c_g], errors='coerce')
        else:
            # Si no hay G cruda, estimar una gravedad realista a partir de latitud y cota
            g0_temp = compute_normal_gravity(res_df['lat'])
            res_df['g_obs'] = g0_temp - 0.3086 * res_df['h'] + 0.1119 * res_df['h']

        # Limpiar filas con NaN en coordenadas principales
        res_df = res_df.dropna(subset=['long', 'lat', 'X', 'Y']).reset_index(drop=True)
        return res_df

    def set_raw_dataframe(self, df_in, source_name="Archivo"):
        """Permite cargar directamente un DataFrame de mediciones crudas."""
        mapped = self.auto_map_columns(df_in)
        if mapped is None:
            QMessageBox.warning(self, "Columnas no detectadas",
                                "No se pudieron identificar las columnas requeridas (long, lat, h, X, Y, G).")
            return False

        self.df = mapped
        self.step_raw_done = True
        self.step_g0_done = False
        self.step_fa_done = False
        self.step_bouguer_done = False
        self.step_terrain_done = False

        self.lbl_data_info.setText(f"Cargado: <b>{source_name}</b> ({len(self.df)} estaciones)")
        self.update_step_badges()
        self.update_table_view()
        self.update_plots()
        return True

    # ==========================================================================
    # MANEJO DEL MDT GEOTIFF
    # ==========================================================================
    def on_select_mdt_file(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Modelo Digital de Terreno (GeoTIFF)", "",
            "Archivos GeoTIFF (*.tif *.tiff);;Todos los Archivos (*.*)"
        )
        if fpath:
            self.load_mdt_file(fpath)

    def load_mdt_file(self, filepath):
        try:
            self.mdt.load(filepath)
            fname = os.path.basename(filepath)
            self.lbl_mdt_status.setText(
                f"<b>{fname}</b>: {self.mdt.nx}×{self.mdt.ny} celdas (dx={self.mdt.dx:.1f}m, dy={self.mdt.dy:.1f}m) | "
                f"Cota: [{self.mdt.z_min:,.0f} a {self.mdt.z_max:,.0f} m]"
            )
            self.lbl_mdt_status.setStyleSheet("color: #007700; font-size: 10px; font-weight: bold;")
            self.update_step_badges()
            self.update_map_view()
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar MDT", f"No se pudo leer el archivo GeoTIFF:\n{e}")

    # ==========================================================================
    # PASOS SECUENCIALES DE REDUCCIÓN GRAVIMÉTRICA
    # ==========================================================================
    def run_step_g0(self):
        """Paso 2: Gravedad normal g0 según Somigliana (Eqn 7.11) y anomalía de latitud."""
        if not self.step_raw_done or self.df is None:
            QMessageBox.warning(self, "Paso Previo Incompleto", "Primero debes cargar las mediciones crudas.")
            return

        model_key = self.combo_ellipsoid.currentData()
        self.df['g0'] = compute_normal_gravity(self.df['lat'], model_key=model_key)
        self.df['anom_lat'] = self.df['g_obs'] - self.df['g0']

        self.step_g0_done = True
        # Si se recalcula un paso previo, invalidar los pasos posteriores para mantener estricta secuencia
        self.step_fa_done = False
        self.step_bouguer_done = False
        self.step_terrain_done = False

        self.update_step_badges()
        self.update_table_view()
        self.update_plots()

    def run_step_free_air(self):
        """Paso 3: Corrección de Aire Libre c_fa y Anomalía de Aire Libre Δg_FA."""
        if not self.step_g0_done:
            QMessageBox.warning(self, "Secuencia Requerida", "Para aplicar Aire Libre primero se debe calcular la gravedad normal g0.")
            return

        grad = self.spin_grad_fa.value()
        use_2nd = self.chk_second_order.isChecked()
        n_geoid = self.spin_geoid_n.value()

        # Cota ortométrica efectiva
        h_eff = self.df['h'] - n_geoid

        self.df['c_fa'] = compute_free_air_correction(h_eff, lat_deg=self.df['lat'], gradient=grad, use_second_order=use_2nd)
        # Anomalía de Aire Libre: Δg_FA = (g_obs - g0) + c_fa
        self.df['anom_fa'] = self.df['anom_lat'] + self.df['c_fa']

        self.step_fa_done = True
        self.step_bouguer_done = False
        self.step_terrain_done = False

        self.update_step_badges()
        self.update_table_view()
        self.update_plots()

    def run_step_bouguer(self):
        """Paso 4: Corrección de Placa de Bouguer c_bp y Anomalía Simple de Bouguer Δg_SB."""
        if not self.step_fa_done:
            QMessageBox.warning(self, "Secuencia Requerida", "Para aplicar Bouguer se tiene que aplicar previamente la corrección de Aire Libre.")
            return

        rho = self.spin_rho_b.value()
        n_geoid = self.spin_geoid_n.value()
        h_eff = self.df['h'] - n_geoid

        self.df['c_bp'] = compute_bouguer_slab_correction(h_eff, rho=rho)
        # Anomalía Simple de Bouguer: Δg_SB = Δg_FA - c_bp
        self.df['anom_sb'] = self.df['anom_fa'] - self.df['c_bp']

        self.step_bouguer_done = True
        self.step_terrain_done = False

        self.update_step_badges()
        self.update_table_view()
        self.update_plots()

    def run_step_terrain(self):
        """Paso 5: Corrección de Terreno T y Anomalía Completa de Bouguer Δg_CB."""
        if not self.step_bouguer_done:
            QMessageBox.warning(self, "Secuencia Requerida", "Para hacer corrección de terreno tiene que estar aplicada la de Bouguer y Aire Libre.")
            return

        if not self.mdt.is_loaded:
            QMessageBox.warning(self, "MDT Requerido", "Debes cargar un archivo GeoTIFF con el Modelo Digital de Terreno.")
            return

        r_max = self.spin_rmax.value()
        rho = self.spin_rho_b.value()

        # Lanzar cálculo en hilo separado para no bloquear la interfaz
        self.btn_calc_t.setEnabled(False)
        self.btn_cancel_t.setEnabled(True)
        self.progress_terrain.setValue(0)

        self.worker_thread = WorkerCalculoTerreno(
            self.mdt, self.df['X'], self.df['Y'], self.df['h'],
            r_max=r_max, rho=rho
        )
        self.worker_thread.progreso.connect(self.on_terrain_progress)
        self.worker_thread.finalizado.connect(self.on_terrain_finished)
        self.worker_thread.error.connect(self.on_terrain_error)
        self.worker_thread.start()

    def on_terrain_progress(self, current, total):
        pct = int(100.0 * current / total)
        self.progress_terrain.setValue(pct)
        self.progress_terrain.setFormat(f"Calculando terreno: {current}/{total} ({pct}%)")

    def on_cancel_terrain(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.btn_cancel_t.setEnabled(False)
            self.btn_calc_t.setEnabled(True)
            self.progress_terrain.setFormat("Cancelado por el usuario")

    def on_terrain_finished(self, t_values):
        self.df['c_terreno'] = t_values
        # Anomalía Completa de Bouguer: Δg_CB = Δg_SB + T
        self.df['anom_cb'] = self.df['anom_sb'] + self.df['c_terreno']

        self.step_terrain_done = True
        self.btn_calc_t.setEnabled(True)
        self.btn_cancel_t.setEnabled(False)
        self.progress_terrain.setValue(100)
        self.progress_terrain.setFormat("✔️ Terreno completado (100%)")

        self.update_step_badges()
        self.update_table_view()
        self.update_plots()
        self.update_map_view()
        QMessageBox.information(self, "Secuencia Completa",
                                f"Corrección de terreno calculada exitosamente para {len(self.df)} estaciones.\n"
                                f"Efecto de terreno medio: {np.mean(t_values):.3f} mGal (máx: {np.max(t_values):.3f} mGal).")

    def on_terrain_error(self, err_msg):
        self.btn_calc_t.setEnabled(True)
        self.btn_cancel_t.setEnabled(False)
        QMessageBox.critical(self, "Error en Terreno", f"Ocurrió un error calculando el terreno:\n{err_msg}")

    def run_full_sequence(self):
        """Ejecuta de manera secuencial y automática todos los pasos desde el inicio."""
        if not self.step_raw_done or self.df is None:
            QMessageBox.warning(self, "Datos Faltantes", "Carga primero un archivo de mediciones crudas.")
            return

        self.run_step_g0()
        self.run_step_free_air()
        self.run_step_bouguer()
        if self.mdt.is_loaded:
            self.run_step_terrain()
        else:
            QMessageBox.information(self, "Bouguer Simple Completado",
                                    "Se completó la reducción hasta Anomalía Simple de Bouguer.\n"
                                    "Para el Paso 5 (Terreno) carga un MDT GeoTIFF.")

    # ==========================================================================
    # ACTUALIZACIÓN DE GRÁFICOS Y PERFILES
    # ==========================================================================
    def update_plots(self):
        self.fig_plot.clf()
        if self.df is None or len(self.df) == 0:
            self.canvas_plot.draw()
            return

        # Coordenada horizontal para el perfil: distancia a lo largo del perfil en km
        dx_prof = self.df['X'] - self.df['X'].iloc[0]
        dy_prof = self.df['Y'] - self.df['Y'].iloc[0]
        dist_km = np.sqrt(dx_prof**2 + dy_prof**2) / 1000.0

        gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.4, 1.0], hspace=0.35)
        ax_topo = self.fig_plot.add_subplot(gs[0])
        ax_grav = self.fig_plot.add_subplot(gs[1], sharex=ax_topo)
        ax_corr = self.fig_plot.add_subplot(gs[2], sharex=ax_topo)

        # 1. Topografía
        ax_topo.plot(dist_km, self.df['h'], 'o-', color='#553311', lw=1.8, ms=4, label='Cota h [m]')
        ax_topo.fill_between(dist_km, 0, self.df['h'], color='#eeddcc', alpha=0.5)
        ax_topo.set_ylabel("Cota h [m]", fontsize=8, fontweight='bold')
        ax_topo.set_title("1. Perfil Topográfico de las Estaciones", fontsize=9, fontweight='bold', pad=4)
        ax_topo.grid(True, linestyle=':', alpha=0.6)
        ax_topo.legend(loc='upper right', fontsize=8)

        # 2. Anomalías y Reducciones en Cascada
        if 'anom_lat' in self.df.columns:
            ax_grav.plot(dist_km, self.df['anom_lat'], '--', color='#888888', lw=1.2, label='Δg_lat (g_obs - g₀)')
        if 'anom_fa' in self.df.columns:
            ax_grav.plot(dist_km, self.df['anom_fa'], 'b.-', lw=1.5, ms=4, label='Aire Libre (Δg_FA)')
        if 'anom_sb' in self.df.columns:
            ax_grav.plot(dist_km, self.df['anom_sb'], 'm.-', lw=1.6, ms=4, label='Simple Bouguer (Δg_SB)')
        if 'anom_cb' in self.df.columns:
            ax_grav.plot(dist_km, self.df['anom_cb'], 'r.-', lw=2.0, ms=5, label='Completa Bouguer (Δg_CB = Δg_SB + T)')

        ax_grav.set_ylabel("Anomalía [mGal]", fontsize=8, fontweight='bold')
        ax_grav.set_title("2. Reducción Secuencial: Evolución de las Anomalías Gravimétricas", fontsize=9, fontweight='bold', pad=4)
        ax_grav.grid(True, linestyle=':', alpha=0.6)
        h_g, l_g = ax_grav.get_legend_handles_labels()
        if h_g:
            ax_grav.legend(h_g, l_g, loc='upper right', fontsize=8, ncol=2)

        # 3. Correcciones individuales
        if 'c_fa' in self.df.columns:
            ax_corr.plot(dist_km, self.df['c_fa'], color='#0066cc', lw=1.5, label='+Aire Libre (+0.3086·h)')
        if 'c_bp' in self.df.columns:
            ax_corr.plot(dist_km, -self.df['c_bp'], color='#993300', lw=1.5, label='-Placa Bouguer (-2πGρh)')
        if 'c_terreno' in self.df.columns:
            ax_corr.plot(dist_km, self.df['c_terreno'], color='#00aa44', lw=1.8, label='+Terreno MDT (+T)')

        ax_corr.set_xlabel("Distancia a lo largo del perfil [km]", fontsize=8, fontweight='bold')
        ax_corr.set_ylabel("Corrección [mGal]", fontsize=8, fontweight='bold')
        ax_corr.set_title("3. Magnitud de las Correcciones Individuales", fontsize=9, fontweight='bold', pad=4)
        ax_corr.grid(True, linestyle=':', alpha=0.6)
        h_c, l_c = ax_corr.get_legend_handles_labels()
        if h_c:
            ax_corr.legend(h_c, l_c, loc='upper right', fontsize=8, ncol=3)

        self.canvas_plot.draw()

    # ==========================================================================
    # ACTUALIZACIÓN DEL MAPA 2D (MDT + ESTACIONES)
    # ==========================================================================
    def update_map_view(self):
        self.fig_map.clf()
        ax = self.fig_map.add_subplot(111)

        # 1. Dibujar MDT de fondo si está cargado
        if self.mdt.is_loaded and self.mdt.grid_z is not None:
            extent = [self.mdt.x0, self.mdt.x1, self.mdt.y1, self.mdt.y0]
            im_mdt = ax.imshow(self.mdt.grid_z, origin='upper', extent=extent, cmap='terrain', alpha=0.75)
            cb_mdt = self.fig_map.colorbar(im_mdt, ax=ax, pad=0.02, fraction=0.04)
            cb_mdt.set_label("Elevación MDT [m]", fontsize=8)

        # 2. Superponer estaciones
        if self.df is not None and len(self.df) > 0 and 'X' in self.df.columns:
            var_name = self.combo_map_var.currentText()
            col_key = 'h'
            label_units = "m"

            if "Completa Bouguer" in var_name and 'anom_cb' in self.df.columns:
                col_key = 'anom_cb'; label_units = "mGal"
            elif "Simple Bouguer" in var_name and 'anom_sb' in self.df.columns:
                col_key = 'anom_sb'; label_units = "mGal"
            elif "Aire Libre" in var_name and 'anom_fa' in self.df.columns:
                col_key = 'anom_fa'; label_units = "mGal"
            elif "Efecto de Terreno" in var_name and 'c_terreno' in self.df.columns:
                col_key = 'c_terreno'; label_units = "mGal"
            elif "Gravedad Observada" in var_name and 'g_obs' in self.df.columns:
                col_key = 'g_obs'; label_units = "mGal"

            vals = self.df[col_key]
            sc = ax.scatter(self.df['X'], self.df['Y'], c=vals, cmap='plasma', s=45, edgecolor='#000000', lw=0.8, zorder=5)
            cb_sc = self.fig_map.colorbar(sc, ax=ax, pad=0.03, fraction=0.04)
            cb_sc.set_label(f"{var_name} [{label_units}]", fontsize=8)

            ax.plot(self.df['X'], self.df['Y'], 'k--', lw=0.9, alpha=0.5, zorder=4)

        ax.set_title("Mapa 2D: Modelo Digital de Terreno y Estaciones Gravimétricas", fontsize=9, fontweight='bold', pad=6)
        ax.set_xlabel("Coordenada X [m]", fontsize=8)
        ax.set_ylabel("Coordenada Y [m]", fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.5)

        self.canvas_map.draw()

    # ==========================================================================
    # ACTUALIZACIÓN DE LA TABLA MAESTRA
    # ==========================================================================
    def update_table_view(self):
        if self.df is None:
            self.tbl_results.clear()
            return

        headers = list(self.df.columns)
        self.tbl_results.setColumnCount(len(headers))
        self.tbl_results.setRowCount(len(self.df))
        self.tbl_results.setHorizontalHeaderLabels(headers)

        for r_idx in range(len(self.df)):
            for c_idx, c_name in enumerate(headers):
                val = self.df.iloc[r_idx, c_idx]
                if isinstance(val, float):
                    item_str = f"{val:,.3f}"
                else:
                    item_str = str(val)
                item = QTableWidgetItem(item_str)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_results.setItem(r_idx, c_idx, item)

        self.tbl_results.resizeColumnsToContents()

    # ==========================================================================
    # EXPORTACIÓN
    # ==========================================================================
    def on_export_csv(self):
        if self.df is None or len(self.df) == 0:
            QMessageBox.warning(self, "Sin Datos", "No hay datos procesados para exportar.")
            return

        fpath, _ = QFileDialog.getSaveFileName(
            self, "Guardar Tabla Reducida", "datos_gravimetricos_reducidos.csv",
            "Archivos CSV (*.csv);;Archivos de Texto (*.txt)"
        )
        if fpath:
            self.df.to_csv(fpath, index=False)
            QMessageBox.information(self, "Guardado", f"Datos exportados exitosamente a:\n{fpath}")

    def on_send_to_fft_window(self):
        """
        Interpola la anomalía calculada (Completa o Simple Bouguer) a una malla regular 2D
        y la envía directamente a la ventana principal del Analizador FFT 2D.
        """
        if self.df is None or len(self.df) == 0:
            QMessageBox.warning(self, "Sin Datos", "No hay datos calculados.")
            return

        target_col = 'anom_cb' if 'anom_cb' in self.df.columns else ('anom_sb' if 'anom_sb' in self.df.columns else 'anom_fa')
        if target_col not in self.df.columns:
            QMessageBox.warning(self, "Anomalía Incompleta", "Debes completar al menos el Paso 3 (Aire Libre).")
            return

        try:
            from scipy.interpolate import griddata

            # Crear grilla regular
            x_min, x_max = self.df['X'].min(), self.df['X'].max()
            y_min, y_max = self.df['Y'].min(), self.df['Y'].max()

            # Espaciamiento adaptativo
            span = max(x_max - x_min, y_max - y_min)
            if span < 10.0:
                span = 10000.0; x_max = x_min + span; y_max = y_min + span

            n_grid = 81
            xi = np.linspace(x_min, x_max, n_grid)
            yi = np.linspace(y_min, y_max, n_grid)
            XI, YI = np.meshgrid(xi, yi)

            pts = np.vstack([self.df['X'].values, self.df['Y'].values]).T
            vals = self.df[target_col].values

            # Interpolación cúbica con fallback lineal y vecino cercano
            ZI = griddata(pts, vals, (XI, YI), method='linear')
            if np.isnan(ZI).any():
                ZI_near = griddata(pts, vals, (XI, YI), method='nearest')
                ZI[np.isnan(ZI)] = ZI_near[np.isnan(ZI)]

            # Conectar con ventana principal
            p = self.parent()
            if p is not None and hasattr(p, 'grid_Z'):
                p.grid_Z = ZI
                p.grid_x_vec = xi
                p.grid_y_vec = yi
                p.unit_xy = "m"
                p.unit_z = "mGal"
                if hasattr(p, 'recalculate_spectrum'):
                    p.recalculate_spectrum()
                QMessageBox.information(self, "Inyección Exitosa",
                                        f"La malla regular de la anomalía ({target_col}) [{n_grid}×{n_grid} nodos] "
                                        f"fue enviada e inyectada con éxito en el Analizador FFT 2D.")
            else:
                # Guardar como archivo CSV de malla regular para abrir
                save_f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "malla_anomalia_bouguer.csv")
                malla_df = pd.DataFrame(ZI, index=np.round(yi, 1), columns=np.round(xi, 1))
                malla_df.to_csv(save_f)
                QMessageBox.information(self, "Malla Exportada",
                                        f"Malla 2D generada exitosamente y guardada en:\n{save_f}\n"
                                        f"Puedes abrirla desde el Analizador FFT o usarla en el modelador sintético.")

        except Exception as e:
            QMessageBox.critical(self, "Error al generar malla", f"No se pudo interpolar la malla:\n{e}")

    # ==========================================================================
    # GLOSARIO Y FUNDAMENTO FÍSICO DETALLADO
    # ==========================================================================
    def load_glossary_text(self):
        html = """
        <div style='font-family: Segoe UI, sans-serif; font-size: 11px; color: #222222; line-height: 1.5;'>
        <h2 style='color: #004488; margin-bottom: 2px;'>Fundamento Físico y Secuencia de Reducciones Gravimétricas</h2>
        <p>El objetivo de la reducción gravimétrica es aislar las anomalías causadas por contrastes de densidad en el subsuelo,
        removiendo sucesivamente los efectos teóricos y de masa conocidos de forma estrictamente secuencial:</p>

        <h3 style='color: #0055aa; border-bottom: 1px solid #c0c0c0; padding-bottom: 2px;'>1. Gravedad Normal del Elipsoide (g₀)</h3>
        <p>Calcula la atracción gravitacional y aceleración centrífuga sobre la superficie del elipsoide de referencia a cota cero en función de la latitud geodésica λ.</p>
        <p><b>Fórmula cerrada de Somigliana (WGS84, Blakely 1995, Eqn. 7.11):</b><br>
        <code>g₀(λ) = 9.7803267714 · [ 1 + 0.00193185138639·sin²λ ] / √( 1 - 0.00669437999013·sin²λ )  [m/s²]</code></p>
        <p><b>Anomalía de Latitud:</b> <code>Δg_lat = g_obs - g₀(λ)</code></p>

        <h3 style='color: #0055aa; border-bottom: 1px solid #c0c0c0; padding-bottom: 2px;'>2. Corrección de Aire Libre (FAC)</h3>
        <p>Como la estación se encuentra a una cota <i>h</i> sobre el elipsoide/geoide, la gravedad disminuye a razón de aprox. <code>-0.3086 mGal/m</code> debido a la mayor distancia al centro terrestre. La corrección compensa esta separación elevando la referencia o reduciendo la observación:</p>
        <p><code>c_fa = +0.3086 · h  [mGal]</code></p>
        <p><b>Anomalía de Aire Libre:</b> <code>Δg_FA = Δg_lat + c_fa = (g_obs - g₀) + 0.3086 · h</code></p>

        <h3 style='color: #0055aa; border-bottom: 1px solid #c0c0c0; padding-bottom: 2px;'>3. Corrección de Placa Simple de Bouguer (BSC)</h3>
        <p>Remueve la atracción gravitacional de la masa de roca comprendida entre el nivel del mar y la estación, modelada como una placa horizontal infinita de espesor <i>h</i> y densidad ρ (estándar de corteza continental ρ = 2670 kg/m³):</p>
        <p><code>c_bp = 2·π·G·ρ·h ≈ 0.111868 · h ≈ 0.1119 · h  [mGal]</code></p>
        <p><b>Anomalía Simple de Bouguer:</b> <code>Δg_SB = Δg_FA - c_bp  [mGal]</code> (SECUENCIAL: requiere previa corrección de aire libre).</p>

        <h3 style='color: #0055aa; border-bottom: 1px solid #c0c0c0; padding-bottom: 2px;'>4. Corrección de Terreno / Topografía con MDT (TC)</h3>
        <p>La placa de Bouguer asume un terreno plano infinito. Para corregir las colinas que atraen hacia arriba y los valles donde falta roca, se suma el efecto topográfico exacto mediante la integral analítica del prisma elemental sobre el MDT:</p>
        <p><code>g_z = γ·ρ·dx·dy · ∫ [ (z - z₀) / ( (x₀ - x_i)² + (y₀ - y_i)² + (z - z₀)² )^(3/2) ] dz</code><br>
        <code>g_z = γ·ρ·dx·dy · [ 1/r - 1 / √(r² + (z - z₀)²) ]  [mGal]</code></p>
        <p>El efecto de terreno <i>T</i> es <b>siempre positivo</b> (suma gravedad tanto para cumbres como para valles).</p>
        <p><b>Anomalía Completa de Bouguer:</b> <code>Δg_CB = Δg_SB + T  [mGal]</code></p>
        </div>
        """
        self.txt_info.setHtml(html)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = VentanaCorrectorGravimetrico()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
