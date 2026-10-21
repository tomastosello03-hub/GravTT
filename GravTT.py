# -*- coding: utf-8 -*-
"""
GravTT - Procesamiento, Interpolación 2D y Análisis Fourier
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from PIL import Image, TiffImagePlugin
from modelador_sintetico_blakely import (
    CuerpoSintetico, VentanaModeladorSintetico, VentanaComparacionModelos, BLAKELY_MODELS_INFO
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import cKDTree

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QGroupBox, QSplitter, QTabWidget,
    QScrollArea, QFrame, QGridLayout, QStatusBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QSlider, QDialog, QMenu, QAction, QRadioButton, QButtonGroup,
    QAbstractItemView, QSplashScreen
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QFont, QCursor, QIcon, QPixmap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESCARGAS_DIR = os.path.join(BASE_DIR, 'descargas')
os.makedirs(DESCARGAS_DIR, exist_ok=True)

from corrector_gravimetrico import (
    G_SI, SI_TO_MGAL, GAMMA_MGAL, RHO_CRUST_DEFAULT, FAC_GRADIENT_DEFAULT,
    ELLIPSOID_MODELS, compute_normal_gravity, compute_free_air_correction,
    compute_bouguer_slab_correction, GeoTiffMDT, WorkerCalculoTerreno,
    latlon_to_gauss_kruger
)
from problema_inverso_fuente_equivalente import VentanaProblemaInverso, DialogoParametrosRasterTIFF

# ==============================================================================
# ESTILO VISUAL: TÉCNICO RETRO WINDOWS VISTA / WIN64 (BORDES RECTOS, FONDO BLANCO)
# ==============================================================================
WIN_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f0f0f0;
    color: #000000;
    font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
    font-size: 12px;
}

QLabel {
    color: #000000;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #707070;
    border-radius: 0px;
    padding: 2px 5px;
    min-height: 24px;
    color: #000000;
    selection-background-color: #cc0000;
    selection-color: #ffffff;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #cc0000;
}

QComboBox {
    padding-left: 6px;
    padding-right: 22px;
    min-height: 24px;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #a0a0a0;
    background: #e8e8e8;
}

QComboBox::down-arrow {
    width: 0px;
    height: 0px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #333333;
}

QSpinBox, QDoubleSpinBox {
    padding: 2px 4px;
    min-height: 24px;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fefefe, stop:0.5 #ececec, stop:1 #d8d8d8);
    color: #000000;
    border: 1px solid #707070;
    border-radius: 0px;
    padding: 5px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:0.5 #f4f4f4, stop:1 #e4e4e4);
    border: 1px solid #cc0000;
    color: #cc0000;
}

QPushButton:pressed {
    background: #c8c8c8;
    border: 1px solid #000000;
}

QPushButton#btnPrimary {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d32f2f, stop:1 #991b1b);
    border: 1px solid #000000;
    color: #ffffff;
    font-size: 13px;
    padding: 7px 15px;
    font-weight: bold;
}

QPushButton#btnPrimary:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e53e3e, stop:1 #b91c1c);
    color: #ffffff;
}

QPushButton#btnPrimary:pressed {
    background: #7f1d1d;
}

QPushButton#btnReset {
    background: #e4e4e4;
    border: 1px solid #707070;
    border-radius: 0px;
    color: #000000;
}

QPushButton#btnMini {
    background: #e8e8e8;
    border: 1px solid #808080;
    border-radius: 0px;
    padding: 3px 6px;
    font-size: 11px;
}

QPushButton#accordionHeader {
    background-color: #e4e4e4;
    color: #000000;
    text-align: left;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: bold;
    border: 1px solid #707070;
    border-radius: 0px;
}

QPushButton#accordionHeader:hover {
    background-color: #d8d8d8;
    color: #cc0000;
    border: 1px solid #cc0000;
}

QPushButton#accordionHeader:checked {
    border-left: 3px solid #cc0000;
}

QWidget#accordionContent {
    background-color: #ffffff;
    border: 1px solid #707070;
    border-top: none;
    border-radius: 0px;
}

QTabWidget::pane {
    border: 1px solid #707070;
    border-radius: 0px;
    background-color: #ffffff;
}

QTabBar::tab {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f0f0f0, stop:1 #dcdcdc);
    color: #333333;
    padding: 7px 16px;
    border: 1px solid #808080;
    border-bottom: none;
    border-radius: 0px;
    margin-right: 2px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #cc0000;
    border: 1px solid #707070;
    border-top: 2px solid #cc0000;
    border-bottom: 1px solid #ffffff;
}

QTableWidget {
    background-color: #ffffff;
    border: 1px solid #707070;
    border-radius: 0px;
    color: #000000;
    gridline-color: #d0d0d0;
    selection-background-color: #cc0000;
    selection-color: #ffffff;
}

QTableWidget QHeaderView::section {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f8f8, stop:1 #e4e4e4);
    color: #000000;
    font-weight: 600;
    padding: 4px;
    border: 1px solid #808080;
    border-radius: 0px;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #707070;
    border-radius: 0px;
    margin-top: 8px;
    padding-top: 10px;
    background-color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #000000;
}

QStatusBar {
    background-color: #e4e4e4;
    color: #000000;
    border-top: 1px solid #b0b0b0;
}

QProgressBar {
    border: 1px solid #707070;
    border-radius: 0px;
    text-align: center;
    background-color: #ffffff;
    color: #000000;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #cc0000;
}

QLineEdit#codeEditor {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    background-color: #fafafa;
    border: 1px solid #707070;
    color: #000000;
    font-weight: bold;
}
"""

class CollapsibleSection(QWidget):
    def __init__(self, title="", parent=None, is_expanded=True):
        super().__init__(parent)
        self.is_expanded = is_expanded

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 4)
        self.main_layout.setSpacing(0)

        self.toggle_btn = QPushButton(f"{'▼' if is_expanded else '▶'}  {title}")
        self.toggle_btn.setObjectName("accordionHeader")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(is_expanded)
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.main_layout.addWidget(self.toggle_btn)

        self.content_widget = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(6)
        self.content_widget.setObjectName("accordionContent")
        self.main_layout.addWidget(self.content_widget)
        if not is_expanded:
            self.content_widget.setVisible(False)

    def on_toggle(self):
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        title = self.toggle_btn.text()[3:]
        arrow = "▼" if self.is_expanded else "▶"
        self.toggle_btn.setText(f"{arrow}  {title}")
        self.toggle_btn.setChecked(self.is_expanded)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#ffffff')
        super().__init__(self.fig)
        self.setParent(parent)


def setup_ax_style(ax, title, xlabel, ylabel):
    ax.set_facecolor('#ffffff')
    ax.set_title(title, color='#000000', fontsize=10, fontweight='bold', pad=7)
    ax.set_xlabel(xlabel, color='#000000', fontsize=9, labelpad=5)
    ax.set_ylabel(ylabel, color='#000000', fontsize=9, labelpad=5)
    ax.tick_params(colors='#000000', labelsize=8)
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.grid(True, linestyle=':', color='#d0d0d0', alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color('#000000')
        spine.set_linewidth(1.0)


def setup_dual_frequency_wavelength_axis(ax, unit_xy):
    """
    Formatea el pie del eje cartesiano mostrando directamente en el eje X:
    - Línea 1 (en negro): valores numéricos de frecuencia f [ciclos/u]
    - Línea 2 (en rojo): valores numéricos de longitud de onda λ [u] = 1/f
    - Extremo derecho: 'f  (c/u)' en negro y 'λ  (u)' en rojo.
    """
    ticks = ax.get_xticks()
    x_min, x_max = ax.get_xlim()
    valid_ticks = [float(t) for t in ticks if x_min <= t <= x_max]
    if len(valid_ticks) < 3:
        valid_ticks = list(np.linspace(x_min, x_max, 5))

    # Fijar explícitamente los ticks ANTES de set_xticklabels para perfecta correspondencia 1:1
    ax.set_xticks(valid_ticks)
    labels = []
    for t in valid_ticks:
        if abs(t) < 1e-12:
            labels.append("0")
        elif abs(t) >= 0.01:
            labels.append(f"{t:.3g}")
        else:
            labels.append(f"{t:.4g}")
    ax.set_xticklabels(labels, color='#000000', fontsize=8)

    # 2. Debajo de cada tick, en ROJO, la longitud de onda
    for t in valid_ticks:
        if abs(t) <= 1e-11:
            lam_str = "∞"
        else:
            lam = 1.0 / abs(t)
            if lam >= 1000.0:
                lam_str = f"{lam:,.0f}"
            elif lam >= 10.0:
                lam_str = f"{lam:.1f}"
            else:
                lam_str = f"{lam:.2f}"

        ax.annotate(lam_str, xy=(t, 0), xytext=(0, -18),
                    xycoords=('data', 'axes fraction'), textcoords='offset points',
                    ha='center', va='top', color='#cc0000', fontsize=8, fontweight='bold')

    # 3. Rótulos en el extremo derecho del pie del gráfico
    ax.annotate(f"f  (c/{unit_xy})", xy=(1.0, 0), xytext=(8, -7),
                xycoords='axes fraction', textcoords='offset points',
                ha='left', va='center', color='#000000', fontsize=8, fontweight='bold')
    ax.annotate(f"λ  ({unit_xy})", xy=(1.0, 0), xytext=(8, -21),
                xycoords='axes fraction', textcoords='offset points',
                ha='left', va='center', color='#cc0000', fontsize=8, fontweight='bold')


def _calc_tail_factor_2d(r, tail_type):
    r_clamped = np.clip(r, 0.0, 1.0)
    if tail_type == 'abrupt':
        return np.ones_like(r_clamped)
    elif tail_type == 'linear':
        return r_clamped
    elif tail_type == 'sinu':
        return 3.0 * (r_clamped ** 2) - 2.0 * (r_clamped ** 3)
    else:
        return 0.5 * (1.0 - np.cos(np.pi * r_clamped))


def eval_math_filter_2d(expr, u_grid, v_grid, k_rad, theta):
    expr = str(expr).strip()
    if not expr:
        return np.ones_like(k_rad, dtype=complex), None
    expr = expr.replace('^', '**')
    expr = re.sub(r'\bln\b', 'log', expr)
    # Soporte natural para unidades pegadas o escritas en la fórmula (ej: 500m, 500 metros, 2km)
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*(metros|metro|km|cm|mm|m)\b', r'\1*\2', expr, flags=re.IGNORECASE)

    safe_env = {
        'u': u_grid, 'U': u_grid,
        'v': v_grid, 'V': v_grid,
        'k': k_rad, 'K': k_rad, 'kr': k_rad, 'r': k_rad,
        'k_nyq': float(np.max(k_rad)) if len(k_rad) > 0 else 0.05,
        'k_nyquist': float(np.max(k_rad)) if len(k_rad) > 0 else 0.05,
        'k_max': float(np.max(k_rad)) if len(k_rad) > 0 else 0.05,
        'theta': theta, 'theta_deg': np.degrees(theta),
        'np': np,
        'm': 1.0, 'metro': 1.0, 'metros': 1.0,
        'km': 1000.0, 'cm': 0.01, 'mm': 0.001,
        'mGal': 1.0, 'mgal': 1.0, 'uGal': 1e-3, 'nT': 1.0,
        'log': lambda x: np.log(np.maximum(1e-12, np.asarray(x, dtype=float))),
        'ln': lambda x: np.log(np.maximum(1e-12, np.asarray(x, dtype=float))),
        'log10': lambda x: np.log10(np.maximum(1e-12, np.asarray(x, dtype=float))),
        'exp': lambda x: np.exp(np.clip(np.asarray(x, dtype=float), -100.0, 80.0)),
        'sqrt': lambda x: np.sqrt(np.maximum(0.0, np.asarray(x, dtype=float))),
        'abs': np.abs,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'pi': np.pi, 'PI': np.pi,
        'e': np.e, 'E': np.e,
        '1j': 1j, 'j': 1j, 'I': 1j, 'i': 1j,
        'max': lambda a, b: np.maximum(a, b),
        'min': lambda a, b: np.minimum(a, b),
    }

    try:
        res = eval(expr, {"__builtins__": {}}, safe_env)
        if np.isscalar(res):
            res_arr = np.full_like(k_rad, complex(res))
        else:
            res_arr = np.asarray(res, dtype=complex)
            if res_arr.shape != k_rad.shape:
                res_arr = np.broadcast_to(res_arr, k_rad.shape)
        res_arr = np.nan_to_num(res_arr, nan=0.0, posinf=100.0, neginf=-100.0)
        return res_arr, None
    except Exception as e:
        err_msg = str(e)
        if err_msg == "'__import__'":
            err_msg = "Sintaxis no válida o token desconocido"
        return None, err_msg


def apply_2d_edge_taper(Z, target_mode='mean', width_pct=10.0, taper_type='cosine'):
    """
    Acondiciona suavemente los extremos de una malla 2D hacia un valor de referencia
    (media de la malla, mínimo o cero) para suprimir la fuga espectral (efecto de Gibbs)
    y discontinuidades de borde en la Transformada de Fourier 2D.
    """
    if Z is None:
        return None
    ny, nx = Z.shape
    if nx < 4 or ny < 4:
        return Z.copy()

    # 1. Determinar el valor de referencia de destino en el perímetro
    if target_mode == 'mean':
        ref = float(np.nanmean(Z))
    elif target_mode == 'min':
        ref = float(np.nanmin(Z))
    else:  # 'zero'
        ref = 0.0

    # 2. Ancho del margen de borde (en celdas)
    pct = np.clip(float(width_pct), 1.0, 45.0) / 100.0
    mx = max(1, int(round(nx * pct)))
    my = max(1, int(round(ny * pct)))

    # 3. Ponderaciones unidimensionales
    def calc_axis_weights(n, m, t_type):
        w = np.ones(n, dtype=float)
        if m <= 0:
            return w
        idx_l = np.arange(m, dtype=float)
        r_l = idx_l / float(m)
        idx_r = (n - 1 - np.arange(n - m, n, dtype=float)) / float(m)
        fac_l = _calc_tail_factor_2d(r_l, t_type)
        fac_r = _calc_tail_factor_2d(idx_r, t_type)
        w[:m] = fac_l
        w[n - m:] = fac_r
        return w

    wx = calc_axis_weights(nx, mx, taper_type)
    wy = calc_axis_weights(ny, my, taper_type)

    # 4. Máscara perimetral 2D (1 en el centro interior, 0 en el perímetro exacto)
    W2D = np.outer(wy, wx)

    # 5. Malla con extremos llevados suavemente al valor objetivo
    Z_tapered = ref + (Z - ref) * W2D
    return Z_tapered


FILTER_THEORY_DATABASE = {
    'upward': {
        'title': "Upward Continuation (Continuación hacia Arriba: exp(-2·π·h·k))",
        'theory': (
            "<b>Fundamento Matemático:</b> Solución de Dirichlet a la ecuación de Laplace para campos potenciales en el espacio libre (∇²Z = 0). "
            "El factor de transferencia analítico es <i>H(k) = exp(-2·π·h·k)</i>, donde <i>h</i> es la altura en unidades espaciales.<br>"
            "<b>Comportamiento Espectral:</b> Actúa como un filtro pasabajos de decaimiento exponencial natural sin oscilaciones ni sobreimpulsos. "
            "Atenúa drásticamente las longitudes de onda cortas.<br>"
            "<b>Interpretación Geofísica:</b> Simula medir el campo a una mayor altura <i>h</i> sobre la superficie. "
            "Atenúa las fuentes someras y el ruido superficial, destacando las grandes anomalías regionales profundas (ej: basamento y corteza profunda)."
        )
    },
    'downward': {
        'title': "Downward Continuation (Continuación hacia Abajo: exp(2·π·h·k))",
        'theory': (
            "<b>Fundamento Matemático:</b> Proyección analítica hacia una profundidad <i>h</i> cercana a los cuerpos fuente con factor <i>exp(+2·π·h·k)</i> "
            "(idéntico al Upward Continuation pero con signo positivo en el exponente, sin signo menos).<br>"
            "<b>Comportamiento Espectral:</b> Amplifica exponencialmente las componentes de mayor frecuencia espacial a medida que <i>k</i> crece.<br>"
            "<b>Interpretación Geofísica:</b> Aumenta la resolución lateral y separa anomalías de fuentes geológicas contiguas. "
            "Advertencia física: Al tener exponente positivo, las altas frecuencias y el ruido instrumental crecen exponencialmente si la profundidad <i>h</i> es grande."
        )
    },
    '1vd': {
        'title': "Primera Derivada Vertical (1VD: ∂Z/∂z = 2·π·k)",
        'theory': (
            "<b>Fundamento Matemático:</b> Derivada vertical calculada en Fourier mediante el operador lineal <i>2·π·k</i> (Laplace en campos potenciales).<br>"
            "<b>Comportamiento Espectral:</b> Filtro pasaltos con ganancia nula en el origen (<i>k = 0</i>), suprimiendo la componente continua (DC).<br>"
            "<b>Interpretación Geofísica:</b> Enfatiza contactos geológicos, fallas y bordes litológicos someros eliminando el fondo regional continuo."
        )
    },
    '2vd': {
        'title': "Segunda Derivada Vertical (2VD: ∂²Z/∂z² = (2·π·k)²)",
        'theory': (
            "<b>Fundamento Matemático:</b> Operador laplaciano inverso de segundo orden en Z, con ganancia cuadrática <i>(2·π·k)²</i>.<br>"
            "<b>Comportamiento Espectral:</b> Pasaltos de segundo orden altamente selectivo para fuentes someras.<br>"
            "<b>Interpretación Geofísica:</b> Delimitación estructural precisa: el contorno cero (<i>2VD = 0</i>) traza geométricamente el perímetro vertical de las fuentes causantes."
        )
    },
    'dx': {
        'title': "Derivada Horizontal en X (∂/∂x = 2·π·i·u)",
        'theory': (
            "<b>Fundamento Matemático:</b> Diferenciador espacial direccional a lo largo de X con rotación de fase imaginaria (+90°).<br>"
            "<b>Comportamiento Espectral:</b> Resalta gradientes en la dirección horizontal U.<br>"
            "<b>Interpretación Geofísica:</b> Mapea fallas, lineamientos y contactos geológicos con orientación Norte-Sur (perpendiculares a X)."
        )
    },
    'dy': {
        'title': "Derivada Horizontal en Y (∂/∂y = 2·π·i·v)",
        'theory': (
            "<b>Fundamento Matemático:</b> Diferenciador espacial a lo largo del eje Y con rotación imaginaria (+90°).<br>"
            "<b>Comportamiento Espectral:</b> Resalta gradientes en la dirección horizontal V.<br>"
            "<b>Interpretación Geofísica:</b> Mapea fallas, lineamientos y contactos con orientación Este-Oeste (perpendiculares a Y)."
        )
    },
    'butterworth_low': {
        'title': "Paso Bajo Butterworth 2D (1 / [1 + (k/kc)⁸])",
        'theory': (
            "<b>Fundamento Matemático:</b> Filtro con respuesta máxima plana en la banda de paso y orden de caída 8.<br>"
            "<b>Comportamiento Espectral:</b> Pasa intactas las frecuencias menores a <i>kc</i> y atenúa suavemente las superiores.<br>"
            "<b>Interpretación Geofísica:</b> Elimina ruido de alta frecuencia y artefactos de grilla sin generar ondulaciones espaciales de Gibbs."
        )
    },
    'butterworth_high': {
        'title': "Paso Alto Butterworth 2D ((k/kc)⁸ / [1 + (k/kc)⁸])",
        'theory': (
            "<b>Fundamento Matemático:</b> Filtro complementario que bloquea frecuencias inferiores a <i>kc</i> y transmite las altas.<br>"
            "<b>Comportamiento Espectral:</b> Ganancia cero en DC (k=0) y unitaria en altas frecuencias.<br>"
            "<b>Interpretación Geofísica:</b> Aísla anomalías residuales locales para exploración de detalle, eliminando el fondo regional profundo."
        )
    },
    'gaussian': {
        'title': "Filtro Gaussiano 2D (exp(-(k/kc)²))",
        'theory': (
            "<b>Fundamento Matemático:</b> Función gaussiana cuya transformada espacial y espectral son ambas campanas gaussianas continuas.<br>"
            "<b>Comportamiento Espectral:</b> Atenuación perfectamente suave y sin ningún sobreimpulso.<br>"
            "<b>Interpretación Geofísica:</b> Suavizado óptimo previo a diferenciaciones o modelado de fuentes."
        )
    },
    'notch': {
        'title': "Filtro Notch / Direccional Azimutal",
        'theory': (
            "<b>Fundamento Matemático:</b> Filtro azimutal dependiente de <i>theta = atan2(v, u)</i> que atenúa una franja angular específica.<br>"
            "<b>Comportamiento Espectral:</b> Filtro anisotrópico direccional.<br>"
            "<b>Interpretación Geofísica:</b> Remueve ruido de líneas de vuelo aerogeofísicas o resalta rumbos tectónicos particulares."
        )
    },
    'taper': {
        'title': "Atenuación Lineal (Taper Espectral)",
        'theory': (
            "<b>Fundamento Matemático:</b> Ponderación lineal que decae desde ganancia 1.0 en DC hasta 0.0 en el radio de corte.<br>"
            "<b>Comportamiento Espectral:</b> Reducción gradual y progresiva de armónicos.<br>"
            "<b>Interpretación Geofísica:</b> Atenuación suave de frecuencias límite de Nyquist."
        )
    },
    'window': {
        'title': "Ventana Espectral con Colas de Transición",
        'theory': (
            "<b>Fundamento Matemático:</b> Delimita una banda radial de frecuencias entre <i>k_min</i> y <i>k_max</i> con colas de transición personalizables.<br>"
            "<b>Comportamiento Espectral:</b> Banda de paso radial simétrica.<br>"
            "<b>Interpretación Geofísica:</b> Aísla anomalías geológicas asociadas a un rango de profundidades determinado."
        )
    }
}


def get_filter_theory_explanation(name, formula, filter_type):
    """Devuelve un resumen teórico y físico para el filtro dado."""
    text_check = f"{name} {formula}".lower()
    if 'upward' in text_check or 'exp(-2' in formula:
        return FILTER_THEORY_DATABASE['upward']
    elif 'downward' in text_check or 'exp(2' in formula:
        return FILTER_THEORY_DATABASE['downward']
    elif '1vd' in text_check or 'primera derivada' in text_check or '2*pi*k' in formula:
        return FILTER_THEORY_DATABASE['1vd']
    elif '2vd' in text_check or 'segunda derivada' in text_check or '(2*pi*k)**2' in formula:
        return FILTER_THEORY_DATABASE['2vd']
    elif 'd/dx' in text_check or '1j*u' in formula or 'j*u' in formula:
        return FILTER_THEORY_DATABASE['dx']
    elif 'd/dy' in text_check or '1j*v' in formula or 'j*v' in formula:
        return FILTER_THEORY_DATABASE['dy']
    elif 'butterworth' in text_check and ('paso alto' in text_check or 'high' in text_check):
        return FILTER_THEORY_DATABASE['butterworth_high']
    elif 'butterworth' in text_check or '1 / (1 +' in formula:
        return FILTER_THEORY_DATABASE['butterworth_low']
    elif 'gauss' in text_check or 'exp(-(k' in formula:
        return FILTER_THEORY_DATABASE['gaussian']
    elif 'notch' in text_check or 'theta' in formula:
        return FILTER_THEORY_DATABASE['notch']
    elif 'taper' in text_check:
        return FILTER_THEORY_DATABASE['taper']
    elif filter_type == 'window':
        return FILTER_THEORY_DATABASE['window']
    else:
        return {
            'title': f"Filtro Personalizado: {name}",
            'theory': (
                f"<b>Fórmula Matemática:</b> <code>{formula}</code><br>"
                "<b>Evaluación Espectral:</b> Este filtro calcula analíticamente la ganancia |H(u,v)| y la fase sobre cada celda de la malla de frecuencias en función de u, v, k y theta.<br>"
                "<b>Aplicación Geofísica:</b> Permite diseñar operadores a medida combinando funciones elementales, exponenciales, trigonométricas o potencias para enfatizar o suprimir rangos de longitud de onda específicos."
            )
        }


# ==============================================================================
# MODELO: FILTRO 2D (FÓRMULA MATEMÁTICA ANALÍTICA O VENTANA ESPECTRAL)
# ==============================================================================
class Filtro2D:
    def __init__(self, name="Filtro", filter_type="math",
                 formula_str="exp(-2 * pi * 500 * k)",
                 math_apply_band=False,
                 math_band_mode="bandpass",
                 k_low=0.0, k_high=0.05, g_in=1.0, g_out=0.0,
                 tail_low_type="smooth", tail_low_w=0.0,
                 tail_high_type="smooth", tail_high_w=0.005,
                 math_invert_band=False, math_smooth_w=0.0,
                 is_enabled=True, color="#cc0000"):
        self.name = name
        self.filter_type = filter_type  # 'math' o 'window'
        self.formula_str = str(formula_str).strip() or "exp(-2 * pi * 500 * k)"
        self.math_apply_band = bool(math_apply_band)
        self.math_band_mode = str(math_band_mode)
        self.math_invert_band = bool(math_invert_band)
        self.math_smooth_w = float(math_smooth_w)

        # Ventana radial y límites de corte
        self.k_low = float(k_low)
        self.k_high = float(k_high)
        self.g_in = float(g_in)
        self.g_out = float(g_out)
        self.tail_low_type = tail_low_type
        self.tail_low_w = float(tail_low_w)
        self.tail_high_type = tail_high_type
        self.tail_high_w = float(tail_high_w)

        self.is_enabled = is_enabled
        self.color = color

    def evaluate_2d(self, U, V, K, Theta):
        shape = K.shape
        if not self.is_enabled:
            return np.ones(shape, dtype=complex)

        if self.filter_type == 'math':
            H, err = eval_math_filter_2d(self.formula_str, U, V, K, Theta)
            if H is None:
                return np.ones(shape, dtype=complex)
            if self.math_apply_band:
                k1 = min(self.k_low, self.k_high)
                k2 = max(self.k_low, self.k_high)
                invert = getattr(self, 'math_invert_band', False)
                eps = 1e-11 * max(1.0, k2)

                if invert:
                    mask = (K < k1 - eps) | (K > k2 + eps)
                else:
                    mask = (K >= k1 - eps) & (K <= k2 + eps)

                H_band = np.full(shape, self.g_out, dtype=complex)
                H_band[mask] = H[mask]

                smooth_w = getattr(self, 'math_smooth_w', 0.0)
                if smooth_w > 0 and not invert:
                    if k1 > 0:
                        m_l = (K >= k1) & (K < k1 + smooth_w)
                        if np.any(m_l):
                            factor_l = 0.5 * (1.0 - np.cos(np.pi * (K[m_l] - k1) / smooth_w))
                            H_band[m_l] = self.g_out + (H[m_l] - self.g_out) * factor_l
                    if k2 > k1:
                        m_h = (K > k2 - smooth_w) & (K <= k2)
                        if np.any(m_h):
                            factor_h = 0.5 * (1.0 - np.cos(np.pi * (k2 - K[m_h]) / smooth_w))
                            H_band[m_h] = self.g_out + (H[m_h] - self.g_out) * factor_h

                return H_band
            return H

        elif self.filter_type == 'window':
            H = np.full(shape, self.g_out, dtype=float)
            k1 = min(self.k_low, self.k_high)
            k2 = max(self.k_low, self.k_high)
            bw = k2 - k1
            if bw <= 0:
                return H.astype(complex)

            w_low = 0.0 if self.tail_low_type == 'abrupt' else max(0.0, self.tail_low_w)
            w_high = 0.0 if self.tail_high_type == 'abrupt' else max(0.0, self.tail_high_w)
            sum_w = w_low + w_high
            if sum_w > bw and sum_w > 0:
                scale = bw / sum_w
                w_low *= scale
                w_high *= scale

            p1 = k1 + w_low
            p2 = k2 - w_high

            if p2 >= p1:
                mask_c = (K >= p1) & (K <= p2)
                H[mask_c] = self.g_in

            if w_low > 0:
                mask_low = (K >= k1) & (K < p1)
                if np.any(mask_low):
                    r = (K[mask_low] - k1) / w_low
                    factor = _calc_tail_factor_2d(r, self.tail_low_type)
                    H[mask_low] = self.g_out + (self.g_in - self.g_out) * factor

            if w_high > 0:
                mask_high = (K > p2) & (K <= k2)
                if np.any(mask_high):
                    r = (k2 - K[mask_high]) / w_high
                    factor = _calc_tail_factor_2d(r, self.tail_high_type)
                    H[mask_high] = self.g_out + (self.g_in - self.g_out) * factor

            if w_low == 0 and w_high == 0:
                mask_box = (K >= k1) & (K <= k2)
                H[mask_box] = self.g_in

            return H.astype(complex)

        return np.ones(shape, dtype=complex)

    def get_summary_text(self):
        if self.filter_type == 'math':
            if self.math_apply_band:
                k1 = min(self.k_low, self.k_high)
                k2 = max(self.k_low, self.k_high)
                invert = getattr(self, 'math_invert_band', False)
                l1_str = f"λ≥{1.0/k2:,.1f}" if k2 > 0 else "λ=∞"
                l2_str = f"λ≤{1.0/k1:,.1f}" if k1 > 0 else "λ=∞"
                if invert:
                    b_str = f" [Rechaza Dominio k∈[{k1:.4g}, {k2:.4g}]]"
                elif k1 == 0:
                    b_str = f" [Dominio k≤{k2:.4g} ({l1_str})]"
                else:
                    b_str = f" [Dominio k∈[{k1:.4g}, {k2:.4g}] (λ∈[{1.0/k2:,.1f}, {1.0/k1:,.1f}])]"
                return f"Fórmula: {self.formula_str}{b_str}"
            return f"Fórmula: {self.formula_str}"
        elif self.filter_type == 'window':
            return f"Ventana [{self.k_low:g}, {self.k_high:g}] (G_in={self.g_in:g}, G_out={self.g_out:g})"
        return "Filtro 2D"


# ==============================================================================
# DIÁLOGO DE EDICIÓN DE FILTROS 2D (CON PRESETS ÚNICOS Y UNIDADES FÍSICAS)
# ==============================================================================
class Filtro2DDialog(QDialog):
    """
    Ventana modal donde los filtros son construidos explícitamente mediante fórmulas
    o ventanas espectrales, con UN SOLO EJEMPLO por familia de filtro, unidades explícitas,
    resumen teórico de cada filtro y acotamiento espectral simétrico en frecuencias positivas.
    """
    def __init__(self, filtro, parent=None, max_kr_hint=0.2, unit_xy="m", unit_z="mGal"):
        super().__init__(parent)
        self.filtro = filtro
        self.max_kr_hint = max(1e-7, float(max_kr_hint))
        self.unit_xy = str(unit_xy)
        self.unit_z = str(unit_z)

        self.setWindowTitle(f"Configuración de Filtro 2D - {filtro.name}")
        self.resize(1080, 680)
        self.setMinimumSize(940, 560)
        self.setStyleSheet(WIN_STYLESHEET)

        # Grilla para previsualización en vivo (resolución suficiente para tiempo real fluido)
        n_p = 64
        self.prev_u = np.linspace(-self.max_kr_hint, self.max_kr_hint, n_p)
        self.prev_v = np.linspace(-self.max_kr_hint, self.max_kr_hint, n_p)
        self.prev_U, self.prev_V = np.meshgrid(self.prev_u, self.prev_v)
        self.prev_K = np.sqrt(self.prev_U**2 + self.prev_V**2)
        self.prev_Theta = np.arctan2(self.prev_V, self.prev_U)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # PANEL IZQUIERDO: CONTROLES
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        lbl_hdr = QLabel(f"<b>Diseño Matemático del Filtro 2D [Unidades: {self.unit_xy}]</b>")
        lbl_hdr.setStyleSheet("font-size: 13px; font-weight: bold; color: #cc0000;")
        left_layout.addWidget(lbl_hdr)

        # Nombre
        row_n = QHBoxLayout()
        row_n.addWidget(QLabel("<b>Nombre:</b>"))
        self.txt_name = QLineEdit(self.filtro.name)
        self.txt_name.textChanged.connect(self.update_preview)
        self.txt_name.textChanged.connect(self.update_filter_theory)
        row_n.addWidget(self.txt_name)
        left_layout.addLayout(row_n)

        # Modalidad: Fórmula Matemática vs Ventana Espectral
        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("<b>Modalidad del Filtro:</b>"))
        self.combo_type = QComboBox()
        self.combo_type.addItem("Filtro Matemático Analítico (Fórmula H(u,v))", "math")
        self.combo_type.addItem(f"Ventana de Frecuencia Radial (Corte [ciclos/{self.unit_xy}] & Colas)", "window")
        idx_t = self.combo_type.findData(self.filtro.filter_type)
        if idx_t >= 0:
            self.combo_type.setCurrentIndex(idx_t)
        self.combo_type.currentIndexChanged.connect(self.on_filter_type_changed)
        self.combo_type.currentIndexChanged.connect(self.update_filter_theory)
        row_t.addWidget(self.combo_type)
        left_layout.addLayout(row_t)

        # --- MODALIDAD 1: FILTRO MATEMÁTICO (FÓRMULA) ---
        self.panel_math = QWidget()
        math_vbox = QVBoxLayout(self.panel_math)
        math_vbox.setContentsMargins(0, 0, 0, 0)
        math_vbox.setSpacing(6)

        grp_formula = QGroupBox("Expresión Matemática H(u, v)")
        grp_form_l = QVBoxLayout(grp_formula)
        grp_form_l.setSpacing(6)

        grp_form_l.addWidget(QLabel(f"<b>Casilla de Fórmula (Ecuación con u, v, k en ciclos/{self.unit_xy}):</b>"))
        self.txt_formula = QLineEdit(self.filtro.formula_str)
        self.txt_formula.setObjectName("codeEditor")
        self.txt_formula.setPlaceholderText(f"Ej: exp(-2 * pi * 500 * k)  o  (2 * pi * k)**2")
        self.txt_formula.textChanged.connect(self.update_preview)
        self.txt_formula.textChanged.connect(self.update_filter_theory)
        grp_form_l.addWidget(self.txt_formula)

        # Plantillas clásicas adaptadas a la frecuencia Nyquist y escala de la malla activa
        kc_lp = round(self.max_kr_hint * 0.30, 6)
        kc_hp = round(self.max_kr_hint * 0.15, 6)
        kc_gauss = round(self.max_kr_hint * 0.35, 6)
        kc_taper = round(self.max_kr_hint * 0.75, 6)
        d_cell_approx = 0.5 / max(1e-7, self.max_kr_hint)
        h_up = round(max(5.0, 2.0 * d_cell_approx), 1)
        h_down = round(max(2.0, 0.5 * d_cell_approx), 1)

        row_temp = QHBoxLayout()
        row_temp.addWidget(QLabel("<b>Cargar Ejemplo Único:</b>"))
        self.combo_templates = QComboBox()
        self.combo_templates.addItem("— Selecciona un filtro conocido (1 ejemplo por tipo) —", "")
        self.combo_templates.addItem(f"Upward Continuation (h={h_up:g} {self.unit_xy}): exp(-2*pi*{h_up:g}*k)", f"exp(-2*pi*{h_up:g}*k)")
        self.combo_templates.addItem(f"Downward Continuation (h={h_down:g} {self.unit_xy}): exp(2*pi*{h_down:g}*k)", f"exp(2*pi*{h_down:g}*k)")
        self.combo_templates.addItem("Primera Derivada Vertical (1VD): 2*pi*k", "2*pi*k")
        self.combo_templates.addItem("Segunda Derivada Vertical (2VD): (2*pi*k)**2", "(2*pi*k)**2")
        self.combo_templates.addItem("Derivada Horizontal X (d/dx): 2*pi*1j*u", "2*pi*1j*u")
        self.combo_templates.addItem("Derivada Horizontal Y (d/dy): 2*pi*1j*v", "2*pi*1j*v")
        self.combo_templates.addItem(f"Paso Bajo Butterworth 2D: 1 / (1 + (k / {kc_lp:g})**8)", f"1 / (1 + (k / {kc_lp:g})**8)")
        self.combo_templates.addItem(f"Paso Alto Butterworth 2D: (k / {kc_hp:g})**8 / (1 + (k / {kc_hp:g})**8)", f"(k / {kc_hp:g})**8 / (1 + (k / {kc_hp:g})**8)")
        self.combo_templates.addItem(f"Filtro Gaussiano: exp(-(k / {kc_gauss:g})**2)", f"exp(-(k / {kc_gauss:g})**2)")
        self.combo_templates.addItem("Filtro Notch Direccional 45°: 1 - exp(-((abs(theta - pi/4)) / (15*pi/180))**2)", "1 - exp(-((abs(theta - pi/4)) / (15*pi/180))**2)")
        self.combo_templates.addItem(f"Atenuación Lineal (Taper): max(0, 1 - k / {kc_taper:g})", f"max(0, 1 - k / {kc_taper:g})")
        self.combo_templates.currentIndexChanged.connect(self.on_template_selected)
        row_temp.addWidget(self.combo_templates)
        grp_form_l.addLayout(row_temp)

        lbl_guide = QLabel(
            f"<b>Variables y Coherencia Física de Unidades:</b><br>"
            f"• <b>u, v</b>: Frecuencias espaciales en X e Y <b>[ciclos/{self.unit_xy}]</b>.<br>"
            f"• <b>k</b> o <b>r</b>: Frecuencia radial = √(u² + v²) <b>[ciclos/{self.unit_xy}]</b>.<br>"
            f"• <b>Continuación hacia Arriba:</b> exp(-2·π·h·k). La altura <b>h</b> debe estar en <b>[{self.unit_xy}]</b>.<br>"
            f"• <b>Continuación hacia Abajo:</b> exp(2·π·h·k). Idéntico a Upward pero con signo positivo (+) en el exponente. La profundidad <b>h</b> en <b>[{self.unit_xy}]</b>.<br>"
            f"• <b>theta</b>: Ángulo azimutal en radianes = atan2(v, u).<br>"
            f"• <b>1j</b>: Unidad imaginaria (para derivadas de fase).<br>"
            f"• <b>Operaciones:</b> +, -, *, /, ** o ^, exp, log, ln, sqrt, sin, cos, abs, pi, e."
        )
        lbl_guide.setStyleSheet("background-color: #fafafa; border: 1px solid #c0c0c0; padding: 6px; font-size: 11px; color: #222;")
        grp_form_l.addWidget(lbl_guide)

        self.lbl_math_status = QLabel("Fórmula válida")
        self.lbl_math_status.setStyleSheet("color: #007700; font-weight: bold; font-size: 11px;")
        grp_form_l.addWidget(self.lbl_math_status)
        math_vbox.addWidget(grp_formula)

        # Dominio Radial de la Fórmula (Recorte Espectral)
        dec = 7 if self.max_kr_hint < 0.01 else (6 if self.max_kr_hint < 1.0 else 4)
        step_val = max(1e-7, self.max_kr_hint / 50.0)
        max_val = max(100.0, self.max_kr_hint * 5.0)

        k_low_init = self.filtro.k_low
        k_high_init = self.filtro.k_high
        if k_high_init <= 0 or k_high_init > self.max_kr_hint * 3.0:
            k_high_init = self.max_kr_hint * 0.65
        if k_low_init < 0 or k_low_init > self.max_kr_hint * 3.0:
            k_low_init = 0.0

        l_min_init = (1.0 / k_high_init) if k_high_init > 0 else (1.0 / self.max_kr_hint)
        l_max_init = (1.0 / k_low_init) if k_low_init > 0 else (50.0 * l_min_init)

        grp_mb = QGroupBox(f"Dominio Radial de la Fórmula (Recorte Espectral)")
        grp_mb_l = QVBoxLayout(grp_mb)
        grp_mb_l.setSpacing(6)

        self.chk_math_band = QCheckBox("Limitar fórmula al dominio radial [k_min, k_max] (G_out fuera)")
        self.chk_math_band.setChecked(self.filtro.math_apply_band)
        self.chk_math_band.stateChanged.connect(self.update_preview)
        grp_mb_l.addWidget(self.chk_math_band)

        # Selector de modo de edición del dominio: Frecuencia k vs Longitud de Onda λ
        row_domain_mode = QHBoxLayout()
        row_domain_mode.addWidget(QLabel("<b>Definir Límites por:</b>"))
        self.combo_domain_units = QComboBox()
        self.combo_domain_units.addItem(f"Frecuencia Radial k [ciclos/{self.unit_xy}]", "freq")
        self.combo_domain_units.addItem(f"Longitud de Onda λ [{self.unit_xy}]", "lambda")
        self.combo_domain_units.currentIndexChanged.connect(self.on_domain_units_changed)
        row_domain_mode.addWidget(self.combo_domain_units)
        grp_mb_l.addLayout(row_domain_mode)

        # Panel 1: Ingreso por Frecuencia k
        self.panel_dom_freq = QWidget()
        grid_df = QGridLayout(self.panel_dom_freq)
        grid_df.setContentsMargins(0, 0, 0, 0)
        grid_df.addWidget(QLabel("<b>Radio Mínimo k_min:</b>"), 0, 0)
        self.spin_math_kmin = QDoubleSpinBox()
        self.spin_math_kmin.setRange(0.0, max_val)
        self.spin_math_kmin.setDecimals(dec)
        self.spin_math_kmin.setSingleStep(step_val)
        self.spin_math_kmin.setValue(k_low_init)
        self.spin_math_kmin.valueChanged.connect(self.sync_from_freq_spins)
        grid_df.addWidget(self.spin_math_kmin, 0, 1)
        self.lbl_sync_lmax = QLabel("↔ λ_max: ∞" if k_low_init <= 0 else f"↔ λ_max: {1.0/k_low_init:,.1f} {self.unit_xy}")
        self.lbl_sync_lmax.setStyleSheet("font-size: 11px; color: #0055aa; font-weight: bold;")
        grid_df.addWidget(self.lbl_sync_lmax, 0, 2)

        grid_df.addWidget(QLabel("<b>Radio Máximo k_max:</b>"), 1, 0)
        self.spin_math_kmax = QDoubleSpinBox()
        self.spin_math_kmax.setRange(0.0, max_val)
        self.spin_math_kmax.setDecimals(dec)
        self.spin_math_kmax.setSingleStep(step_val)
        self.spin_math_kmax.setValue(k_high_init)
        self.spin_math_kmax.valueChanged.connect(self.sync_from_freq_spins)
        grid_df.addWidget(self.spin_math_kmax, 1, 1)
        self.lbl_sync_lmin = QLabel(f"↔ λ_min = {l_min_init:,.1f} {self.unit_xy}")
        self.lbl_sync_lmin.setStyleSheet("font-size: 11px; color: #0055aa; font-weight: bold;")
        grid_df.addWidget(self.lbl_sync_lmin, 1, 2)
        grp_mb_l.addWidget(self.panel_dom_freq)

        # Panel 2: Ingreso por Longitud de Onda λ
        self.panel_dom_lambda = QWidget()
        grid_dl = QGridLayout(self.panel_dom_lambda)
        grid_dl.setContentsMargins(0, 0, 0, 0)
        grid_dl.addWidget(QLabel("<b>Longitud Mínima λ_min:</b>"), 0, 0)
        self.spin_math_lmin = QDoubleSpinBox()
        self.spin_math_lmin.setRange(0.01, 1e9)
        self.spin_math_lmin.setDecimals(1)
        self.spin_math_lmin.setSingleStep(max(1.0, l_min_init / 10.0))
        self.spin_math_lmin.setValue(l_min_init)
        self.spin_math_lmin.setSuffix(f" {self.unit_xy}")
        self.spin_math_lmin.valueChanged.connect(self.sync_from_lambda_spins)
        grid_dl.addWidget(self.spin_math_lmin, 0, 1)
        self.lbl_sync_kmax = QLabel(f"↔ k_max: {k_high_init:.6g}")
        self.lbl_sync_kmax.setStyleSheet("font-size: 11px; color: #007700; font-weight: bold;")
        grid_dl.addWidget(self.lbl_sync_kmax, 0, 2)

        grid_dl.addWidget(QLabel("<b>Longitud Máxima λ_max:</b>"), 1, 0)
        self.spin_math_lmax = QDoubleSpinBox()
        self.spin_math_lmax.setRange(0.01, 1e9)
        self.spin_math_lmax.setDecimals(1)
        self.spin_math_lmax.setSingleStep(max(1.0, l_min_init / 5.0))
        self.spin_math_lmax.setValue(l_max_init)
        self.spin_math_lmax.setSuffix(f" {self.unit_xy}")
        self.spin_math_lmax.valueChanged.connect(self.sync_from_lambda_spins)
        grid_dl.addWidget(self.spin_math_lmax, 1, 1)
        self.lbl_sync_kmin = QLabel("↔ k_min: 0.00000" if k_low_init <= 0 else f"↔ k_min: {k_low_init:.6g}")
        self.lbl_sync_kmin.setStyleSheet("font-size: 11px; color: #007700; font-weight: bold;")
        grid_dl.addWidget(self.lbl_sync_kmin, 1, 2)
        self.panel_dom_lambda.setVisible(False)
        grp_mb_l.addWidget(self.panel_dom_lambda)

        grid_mb_bottom = QGridLayout()
        grid_mb_bottom.addWidget(QLabel("<b>G_out (Fuera del Dominio):</b>"), 0, 0)
        self.spin_math_gout = QDoubleSpinBox()
        self.spin_math_gout.setRange(-1e5, 1e5)
        self.spin_math_gout.setDecimals(4)
        self.spin_math_gout.setValue(self.filtro.g_out)
        self.spin_math_gout.setToolTip("Ganancia asignada fuera del dominio radial (0.0 = supresión total)")
        self.spin_math_gout.valueChanged.connect(self.update_preview)
        grid_mb_bottom.addWidget(self.spin_math_gout, 0, 1)

        self.chk_math_invert = QCheckBox("Invertir dominio (Rechazar [k_min, k_max] y conservar fuera)")
        self.chk_math_invert.setChecked(getattr(self.filtro, 'math_invert_band', False))
        self.chk_math_invert.stateChanged.connect(self.update_preview)
        grid_mb_bottom.addWidget(self.chk_math_invert, 1, 0, 1, 2)

        row_smooth = QHBoxLayout()
        self.chk_math_smooth = QCheckBox("Suavizar bordes del corte (Coseno)")
        self.chk_math_smooth.setChecked(getattr(self.filtro, 'math_smooth_w', 0.0) > 0)
        self.chk_math_smooth.stateChanged.connect(self.on_smooth_toggled)
        row_smooth.addWidget(self.chk_math_smooth)
        self.spin_math_smooth_w = QDoubleSpinBox()
        self.spin_math_smooth_w.setRange(0.0, max_val)
        self.spin_math_smooth_w.setDecimals(dec)
        self.spin_math_smooth_w.setSingleStep(step_val)
        self.spin_math_smooth_w.setValue(getattr(self.filtro, 'math_smooth_w', 0.0) or step_val * 2.0)
        self.spin_math_smooth_w.setEnabled(self.chk_math_smooth.isChecked())
        self.spin_math_smooth_w.valueChanged.connect(self.update_preview)
        row_smooth.addWidget(self.spin_math_smooth_w)
        grid_mb_bottom.addLayout(row_smooth, 2, 0, 1, 2)

        lbl_sym_note = QLabel("<b>Dominio Radial Simétrico:</b> La fórmula actúa en K = √(u² + v²) dentro de [k_min, k_max] (360°). Fuera vale G_out.")
        lbl_sym_note.setStyleSheet("font-size: 10px; color: #333333;")
        grid_mb_bottom.addWidget(lbl_sym_note, 3, 0, 1, 2)
        grp_mb_l.addLayout(grid_mb_bottom)

        math_vbox.addWidget(grp_mb)
        left_layout.addWidget(self.panel_math)

        # --- MODALIDAD 2: FILTRO DE VENTANA RADIAL CON COLAS ---
        self.panel_window = QWidget()
        win_vbox = QVBoxLayout(self.panel_window)
        win_vbox.setContentsMargins(0, 0, 0, 0)
        win_vbox.setSpacing(6)

        grp_win_params = QGroupBox(f"Parámetros de Ventana [ciclos/{self.unit_xy}]")
        grp_wp_l = QGridLayout(grp_win_params)
        grp_wp_l.addWidget(QLabel("<b>k_min:</b>"), 0, 0)
        self.spin_kmin = QDoubleSpinBox()
        self.spin_kmin.setRange(0.0, max_val)
        self.spin_kmin.setDecimals(dec)
        self.spin_kmin.setSingleStep(step_val)
        self.spin_kmin.setValue(k_low_init)
        self.spin_kmin.valueChanged.connect(self.update_preview)
        grp_wp_l.addWidget(self.spin_kmin, 0, 1)

        grp_wp_l.addWidget(QLabel("<b>k_max:</b>"), 0, 2)
        self.spin_kmax = QDoubleSpinBox()
        self.spin_kmax.setRange(0.0, max_val)
        self.spin_kmax.setDecimals(dec)
        self.spin_kmax.setSingleStep(step_val)
        self.spin_kmax.setValue(k_high_init)
        self.spin_kmax.valueChanged.connect(self.update_preview)
        grp_wp_l.addWidget(self.spin_kmax, 0, 3)

        grp_wp_l.addWidget(QLabel("<b>G_in (Dentro):</b>"), 1, 0)
        self.spin_gin = QDoubleSpinBox()
        self.spin_gin.setRange(0.0, 1000.0)
        self.spin_gin.setValue(self.filtro.g_in)
        self.spin_gin.valueChanged.connect(self.update_preview)
        grp_wp_l.addWidget(self.spin_gin, 1, 1)

        grp_wp_l.addWidget(QLabel("<b>G_out (Fuera):</b>"), 1, 2)
        self.spin_gout = QDoubleSpinBox()
        self.spin_gout.setRange(0.0, 1000.0)
        self.spin_gout.setValue(self.filtro.g_out)
        self.spin_gout.valueChanged.connect(self.update_preview)
        grp_wp_l.addWidget(self.spin_gout, 1, 3)
        win_vbox.addWidget(grp_win_params)

        grp_low = QGroupBox(f"Cola Inferior Flanco k_min [ciclos/{self.unit_xy}]")
        grp_low_l = QGridLayout(grp_low)
        grp_low_l.addWidget(QLabel("Tipo de cola:"), 0, 0)
        self.combo_tail_low = QComboBox()
        self.combo_tail_low.addItem("Suave (Hann / Coseno)", "smooth")
        self.combo_tail_low.addItem("Sinuosa (Sigmoide / S-curve)", "sinu")
        self.combo_tail_low.addItem("Lineal (Rampa)", "linear")
        self.combo_tail_low.addItem("Abrupta (Escalón)", "abrupt")
        idx_tl = self.combo_tail_low.findData(self.filtro.tail_low_type)
        if idx_tl >= 0: self.combo_tail_low.setCurrentIndex(idx_tl)
        self.combo_tail_low.currentIndexChanged.connect(self.on_tail_low_type_changed)
        grp_low_l.addWidget(self.combo_tail_low, 0, 1)

        grp_low_l.addWidget(QLabel("Ancho Δk_inf:"), 1, 0)
        self.spin_tail_low_w = QDoubleSpinBox()
        self.spin_tail_low_w.setRange(0.0, max_val)
        self.spin_tail_low_w.setDecimals(dec)
        self.spin_tail_low_w.setSingleStep(step_val)
        val_tl = getattr(self.filtro, 'tail_low_w', None)
        if val_tl is not None and val_tl >= 0.0:
            self.spin_tail_low_w.setValue(float(val_tl))
        else:
            self.spin_tail_low_w.setValue(0.0)
        self.spin_tail_low_w.valueChanged.connect(self.update_preview)
        grp_low_l.addWidget(self.spin_tail_low_w, 1, 1)
        win_vbox.addWidget(grp_low)

        grp_high = QGroupBox(f"Cola Superior Flanco k_max [ciclos/{self.unit_xy}]")
        grp_high_l = QGridLayout(grp_high)
        grp_high_l.addWidget(QLabel("Tipo de cola:"), 0, 0)
        self.combo_tail_high = QComboBox()
        self.combo_tail_high.addItem("Suave (Hann / Coseno)", "smooth")
        self.combo_tail_high.addItem("Sinuosa (Sigmoide / S-curve)", "sinu")
        self.combo_tail_high.addItem("Lineal (Rampa)", "linear")
        self.combo_tail_high.addItem("Abrupta (Escalón)", "abrupt")
        idx_th = self.combo_tail_high.findData(self.filtro.tail_high_type)
        if idx_th >= 0: self.combo_tail_high.setCurrentIndex(idx_th)
        self.combo_tail_high.currentIndexChanged.connect(self.on_tail_high_type_changed)
        grp_high_l.addWidget(self.combo_tail_high, 0, 1)

        grp_high_l.addWidget(QLabel("Ancho Δk_sup:"), 1, 0)
        self.spin_tail_high_w = QDoubleSpinBox()
        self.spin_tail_high_w.setRange(0.0, max_val)
        self.spin_tail_high_w.setDecimals(dec)
        self.spin_tail_high_w.setSingleStep(step_val)
        val_th = getattr(self.filtro, 'tail_high_w', None)
        if val_th is not None and val_th >= 0.0:
            self.spin_tail_high_w.setValue(float(val_th))
        else:
            self.spin_tail_high_w.setValue(0.0)
        self.spin_tail_high_w.valueChanged.connect(self.update_preview)
        grp_high_l.addWidget(self.spin_tail_high_w, 1, 1)
        win_vbox.addWidget(grp_high)
        left_layout.addWidget(self.panel_window)

        # --- RESUMEN TEÓRICO Y FÍSICO DEL FILTRO (ABAJO DEL ACOTAMIENTO) ---
        self.grp_theory = QGroupBox("Resumen Teórico y Físico del Filtro")
        grp_th_l = QVBoxLayout(self.grp_theory)
        grp_th_l.setSpacing(4)
        self.lbl_theory_title = QLabel()
        self.lbl_theory_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #cc0000;")
        grp_th_l.addWidget(self.lbl_theory_title)

        self.lbl_theory_body = QLabel()
        self.lbl_theory_body.setWordWrap(True)
        self.lbl_theory_body.setStyleSheet("background-color: #fafafa; border: 1px solid #c0c0c0; padding: 6px; font-size: 10px; color: #222222; line-height: 130%;")
        grp_th_l.addWidget(self.lbl_theory_body)
        left_layout.addWidget(self.grp_theory)

        left_layout.addStretch()

        # Botones inferiores
        row_btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setObjectName("btnReset")
        btn_cancel.clicked.connect(self.reject)
        row_btns.addWidget(btn_cancel)

        btn_save = QPushButton("Guardar Filtro")
        btn_save.setObjectName("btnPrimary")
        btn_save.clicked.connect(self.on_save)
        row_btns.addWidget(btn_save)
        left_layout.addLayout(row_btns)

        left_widget.setMinimumWidth(440)
        left_widget.setMaximumWidth(500)
        main_layout.addWidget(left_widget)

        # PANEL DERECHO: PREVISUALIZACIÓN EN VIVO CON SELECTOR ARRIBA A LA DERECHA
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # BARRA SUPERIOR DERECHA: CAMBIAR ENTRE FRECUENCIA (f / k) Y LONGITUD DE ONDA (λ)
        row_top_prev = QHBoxLayout()
        row_top_prev.addWidget(QLabel("<b>Previsualización Espectral en Vivo:</b>"))
        row_top_prev.addStretch()
        lbl_v_mode = QLabel("<b>Ver Gráficos en:</b>")
        self.combo_disp_mode = QComboBox()
        self.combo_disp_mode.addItem(f"Frecuencia Espacial f / k [ciclos/{self.unit_xy}]", "freq")
        self.combo_disp_mode.addItem(f"Longitud de Onda λ [{self.unit_xy}]", "lambda")
        self.combo_disp_mode.setStyleSheet("font-weight: bold; padding: 3px 8px; border: 1px solid #cc0000;")
        self.combo_disp_mode.currentIndexChanged.connect(self.update_preview)
        row_top_prev.addWidget(lbl_v_mode)
        row_top_prev.addWidget(self.combo_disp_mode)
        right_layout.addLayout(row_top_prev)

        self.canvas_prev = MplCanvas(self, width=6, height=7)
        right_layout.addWidget(self.canvas_prev)
        main_layout.addWidget(right_widget)

        self.on_filter_type_changed()
        self.update_filter_theory()
        self.update_preview()

    def on_filter_type_changed(self):
        ft = self.combo_type.currentData()
        self.panel_math.setVisible(ft == 'math')
        self.panel_window.setVisible(ft == 'window')
        self.update_filter_theory()
        self.update_preview()

    def on_tail_low_type_changed(self):
        if self.combo_tail_low.currentData() == 'abrupt':
            self.spin_tail_low_w.setValue(0.0)
        self.update_preview()

    def on_tail_high_type_changed(self):
        if self.combo_tail_high.currentData() == 'abrupt':
            self.spin_tail_high_w.setValue(0.0)
        self.update_preview()

    def on_domain_units_changed(self):
        is_freq = (self.combo_domain_units.currentData() == 'freq')
        self.panel_dom_freq.setVisible(is_freq)
        self.panel_dom_lambda.setVisible(not is_freq)

    def on_smooth_toggled(self):
        self.spin_math_smooth_w.setEnabled(self.chk_math_smooth.isChecked())
        self.update_preview()

    def sync_from_freq_spins(self):
        if getattr(self, '_syncing_spins', False):
            return
        self._syncing_spins = True
        try:
            kmin = self.spin_math_kmin.value()
            kmax = self.spin_math_kmax.value()
            if kmax > 0:
                lmin = 1.0 / kmax
                self.spin_math_lmin.setValue(lmin)
                self.lbl_sync_lmin.setText(f"↔ λ_min = {lmin:,.1f} {self.unit_xy}")
                self.lbl_sync_kmax.setText(f"↔ k_max: {kmax:.6g}")
            else:
                self.lbl_sync_lmin.setText("↔ λ_min = ∞")
                self.lbl_sync_kmax.setText("↔ k_max: 0.0")

            if kmin > 0:
                lmax = 1.0 / kmin
                self.spin_math_lmax.setValue(lmax)
                self.lbl_sync_lmax.setText(f"↔ λ_max = {lmax:,.1f} {self.unit_xy}")
                self.lbl_sync_kmin.setText(f"↔ k_min: {kmin:.6g}")
            else:
                self.lbl_sync_lmax.setText("↔ λ_max = ∞ (Centro DC)")
                self.lbl_sync_kmin.setText("↔ k_min: 0.00000")
        finally:
            self._syncing_spins = False
        self.update_preview()

    def sync_from_lambda_spins(self):
        if getattr(self, '_syncing_spins', False):
            return
        self._syncing_spins = True
        try:
            lmin = self.spin_math_lmin.value()
            lmax = self.spin_math_lmax.value()
            if lmin > 0:
                kmax = 1.0 / lmin
                self.spin_math_kmax.setValue(kmax)
                self.lbl_sync_kmax.setText(f"↔ k_max: {kmax:.6g}")
                self.lbl_sync_lmin.setText(f"↔ λ_min = {lmin:,.1f} {self.unit_xy}")
            if lmax > 0 and lmax < 1e8:
                kmin = 1.0 / lmax
                self.spin_math_kmin.setValue(kmin)
                self.lbl_sync_kmin.setText(f"↔ k_min: {kmin:.6g}")
                self.lbl_sync_lmax.setText(f"↔ λ_max = {lmax:,.1f} {self.unit_xy}")
            else:
                self.spin_math_kmin.setValue(0.0)
                self.lbl_sync_kmin.setText("↔ k_min: 0.00000")
                self.lbl_sync_lmax.setText("↔ λ_max = ∞ (Centro DC)")
        finally:
            self._syncing_spins = False
        self.update_preview()

    def update_filter_theory(self):
        ft = self.combo_type.currentData()
        name = self.txt_name.text().strip()
        formula = self.txt_formula.text().strip()
        info = get_filter_theory_explanation(name, formula, ft)
        self.lbl_theory_title.setText(info.get('title', 'Filtro 2D'))
        self.lbl_theory_body.setText(info.get('theory', ''))

    def on_template_selected(self):
        form = self.combo_templates.currentData()
        if form:
            self.txt_formula.setText(form)
            t_txt = self.combo_templates.currentText()
            clean_name = t_txt.split(":")[0].strip()
            clean_name = re.sub(r'^[^\w]+', '', clean_name).strip()
            if clean_name:
                self.txt_name.setText(clean_name)
            self.update_filter_theory()

    def build_temp_filtro(self):
        ft = self.combo_type.currentData()
        name = self.txt_name.text().strip() or "Filtro 2D"
        if ft == 'math':
            k_low = min(self.spin_math_kmin.value(), self.spin_math_kmax.value())
            k_high = max(self.spin_math_kmin.value(), self.spin_math_kmax.value())
            g_out = self.spin_math_gout.value()
            invert = self.chk_math_invert.isChecked() if hasattr(self, 'chk_math_invert') else False
            smooth_w = self.spin_math_smooth_w.value() if hasattr(self, 'chk_math_smooth') and self.chk_math_smooth.isChecked() else 0.0
        else:
            k_low = self.spin_kmin.value()
            k_high = self.spin_kmax.value()
            g_out = self.spin_gout.value()
            invert = False
            smooth_w = 0.0

        f = Filtro2D(
            name=name, filter_type=ft,
            formula_str=self.txt_formula.text().strip(),
            math_apply_band=self.chk_math_band.isChecked() if ft == 'math' else False,
            math_band_mode='bandpass',
            k_low=k_low, k_high=k_high,
            g_in=self.spin_gin.value(),
            g_out=g_out,
            tail_low_type=self.combo_tail_low.currentData(), tail_low_w=self.spin_tail_low_w.value(),
            tail_high_type=self.combo_tail_high.currentData(), tail_high_w=self.spin_tail_high_w.value(),
            math_invert_band=invert, math_smooth_w=smooth_w,
            is_enabled=True, color="#cc0000"
        )
        return f

    def update_preview(self):
        fig = self.canvas_prev.fig
        fig.clf()

        f_temp = self.build_temp_filtro()
        if f_temp.filter_type == 'math':
            H_test, err = eval_math_filter_2d(f_temp.formula_str, self.prev_U, self.prev_V, self.prev_K, self.prev_Theta)
            if err:
                self.lbl_math_status.setText(f"Error sintaxis: {err[:40]}")
                self.lbl_math_status.setStyleSheet("color:#cc0000; font-weight:bold; font-size:11px;")
            else:
                self.lbl_math_status.setText("Fórmula válida")
                self.lbl_math_status.setStyleSheet("color:#007700; font-weight:bold; font-size:11px;")

        H = f_temp.evaluate_2d(self.prev_U, self.prev_V, self.prev_K, self.prev_Theta)
        H_abs = np.abs(H)

        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.2, 0.8])
        ax_2d = fig.add_subplot(gs[0, 0])
        ax_1d = fig.add_subplot(gs[1, 0])

        disp_mode = self.combo_disp_mode.currentData() if hasattr(self, 'combo_disp_mode') else 'freq'

        if disp_mode == 'lambda':
            # --- VISTA 2D CON ISOLÍNEAS DE LONGITUD DE ONDA λ ---
            u_label = f"Frecuencia Espacial U [ciclos/{self.unit_xy}]"
            v_label = f"Frecuencia Espacial V [ciclos/{self.unit_xy}]"
            setup_ax_style(ax_2d, f"Respuesta 2D [Isolíneas de λ en {self.unit_xy}] - [{f_temp.name}]", u_label, v_label)
            extent = [self.prev_u[0], self.prev_u[-1], self.prev_v[0], self.prev_v[-1]]
            im = ax_2d.imshow(H_abs, origin='lower', extent=extent, cmap='jet')
            cb = fig.colorbar(im, ax=ax_2d, pad=0.03)
            cb.set_label("Ganancia |H|", color='#000000', fontsize=8)

            # Círculos concéntricos de isolongitud de onda representativas
            base_l = 1.0 / self.max_kr_hint
            for mult in [1.5, 3.0, 7.0, 15.0]:
                val_l = base_l * mult
                r_l = 1.0 / val_l
                if self.max_kr_hint * 0.05 <= r_l <= self.max_kr_hint * 0.95:
                    circle = patches.Circle((0, 0), r_l, fill=False, edgecolor='#ffffff', linestyle='--', linewidth=1.1, alpha=0.85)
                    ax_2d.add_patch(circle)
                    ax_2d.text(0, r_l, f"λ={val_l:,.0f} {self.unit_xy}", color='#ffffff', fontsize=7, fontweight='bold',
                               ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.1', facecolor='#222222', alpha=0.6, edgecolor='none'))

            # --- VISTA 1D EN LONGITUD DE ONDA λ ---
            l_min_chart = max(1e-6, 1.0 / self.max_kr_hint)
            l_max_chart = 50.0 * l_min_chart
            lambda_vec = np.geomspace(l_min_chart, l_max_chart, 200)
            k_vec = 1.0 / lambda_vec

            zeros_k = np.zeros_like(k_vec)
            H_rad = f_temp.evaluate_2d(k_vec, zeros_k, k_vec, zeros_k)
            H_rad_abs = np.abs(H_rad)

            setup_ax_style(ax_1d, f"Perfil de Ganancia vs Longitud de Onda: |H(λ)|", f"Longitud de Onda Radial λ [{self.unit_xy}]", "Ganancia |H|")
            ax_1d.plot(lambda_vec, H_rad_abs, color='#cc0000', linewidth=2.2, label="|H(λ)|")
            ax_1d.set_xscale('log')

            if f_temp.filter_type == 'math' and f_temp.math_apply_band:
                k1 = min(f_temp.k_low, f_temp.k_high)
                k2 = max(f_temp.k_low, f_temp.k_high)
                l_dom_min = 1.0 / k2 if k2 > 0 else 0.0
                l_dom_max = 1.0 / k1 if k1 > 0 else float('inf')
                inv = getattr(f_temp, 'math_invert_band', False)

                if not inv:
                    if l_dom_min > 0:
                        ax_1d.axvline(l_dom_min, color='#0055aa', linestyle='--', linewidth=1.5, label=f"λ_min={l_dom_min:,.1f} {self.unit_xy}")
                    if k1 > 0:
                        ax_1d.axvline(l_dom_max, color='#00aa55', linestyle='--', linewidth=1.5, label=f"λ_max={l_dom_max:,.1f} {self.unit_xy}")
                        ax_1d.axvspan(l_dom_min, min(l_max_chart, l_dom_max), color='#0055aa', alpha=0.1, label="Dominio Activo")
                    else:
                        ax_1d.axvspan(l_dom_min, l_max_chart, color='#0055aa', alpha=0.1, label="Dominio Activo")
                else:
                    if l_dom_min > 0: ax_1d.axvline(l_dom_min, color='#cc0000', linestyle='--', linewidth=1.5)
                    if k1 > 0:
                        ax_1d.axvline(l_dom_max, color='#cc0000', linestyle='--', linewidth=1.5)
                        ax_1d.axvspan(l_dom_min, min(l_max_chart, l_dom_max), color='#cc0000', alpha=0.1, label="Banda Rechazada")

            ax_1d.legend(facecolor='#ffffff', edgecolor='#808080', labelcolor='#000000', loc='best', fontsize=8)

        else:
            # --- VISTA 2D Y 1D ESTÁNDAR EN FRECUENCIA ESPACIAL f / k ---
            u_label = f"Frecuencia Espacial U [ciclos/{self.unit_xy}]"
            v_label = f"Frecuencia Espacial V [ciclos/{self.unit_xy}]"
            setup_ax_style(ax_2d, f"Respuesta en Frecuencia 2D: |H(u,v)| - [{f_temp.name}]", u_label, v_label)
            extent = [self.prev_u[0], self.prev_u[-1], self.prev_v[0], self.prev_v[-1]]
            im = ax_2d.imshow(H_abs, origin='lower', extent=extent, cmap='jet')
            cb = fig.colorbar(im, ax=ax_2d, pad=0.03)
            cb.set_label("Ganancia |H|", color='#000000', fontsize=8)

            kr_label = f"Frecuencia Espacial Simétrica U [ciclos/{self.unit_xy}]"
            setup_ax_style(ax_1d, "Perfil Espectral Simétrico 1D: |H(u, v=0)|", kr_label, "Ganancia |H|")
            mid_row = H_abs[len(self.prev_v)//2, :]
            u_axis = self.prev_u
            ax_1d.plot(u_axis, mid_row, color='#cc0000', linewidth=2.0, label="|H(u, 0)|")
            ax_1d.axvline(0.0, color='#888888', linestyle=':', linewidth=0.8, alpha=0.7)

            if f_temp.filter_type == 'math' and f_temp.math_apply_band:
                k1 = min(f_temp.k_low, f_temp.k_high)
                k2 = max(f_temp.k_low, f_temp.k_high)
                inv = getattr(f_temp, 'math_invert_band', False)
                if not inv:
                    if k1 > 0:
                        ax_1d.axvline(-k2, color='#0055aa', linestyle='--', linewidth=1.2)
                        ax_1d.axvline(-k1, color='#00aa55', linestyle='--', linewidth=1.2)
                        ax_1d.axvline(k1, color='#00aa55', linestyle='--', linewidth=1.2, label=f"k_min ±{k1:.4g}")
                        ax_1d.axvline(k2, color='#0055aa', linestyle='--', linewidth=1.2, label=f"k_max ±{k2:.4g}")
                        ax_1d.axvspan(-k2, -k1, color='#0055aa', alpha=0.08)
                        ax_1d.axvspan(k1, k2, color='#0055aa', alpha=0.08, label="Dominio Activo")
                    else:
                        ax_1d.axvline(-k2, color='#0055aa', linestyle='--', linewidth=1.5, label=f"k_corte ±{k2:.4g}")
                        ax_1d.axvline(k2, color='#0055aa', linestyle='--', linewidth=1.5)
                        ax_1d.axvspan(-k2, k2, color='#0055aa', alpha=0.08, label="Dominio Activo")
                else:
                    ax_1d.axvline(-k2, color='#cc0000', linestyle='--', linewidth=1.2)
                    ax_1d.axvline(-k1, color='#cc0000', linestyle='--', linewidth=1.2)
                    ax_1d.axvline(k1, color='#cc0000', linestyle='--', linewidth=1.2)
                    ax_1d.axvline(k2, color='#cc0000', linestyle='--', linewidth=1.2, label=f"Rechazo ±[{k1:.3g}, {k2:.3g}]")
                    ax_1d.axvspan(-k2, -k1, color='#cc0000', alpha=0.1)
                    ax_1d.axvspan(k1, k2, color='#cc0000', alpha=0.1)

            ax_1d.legend(facecolor='#ffffff', edgecolor='#808080', labelcolor='#000000', loc='best', fontsize=8)

        fig.tight_layout()
        self.canvas_prev.draw()

    def on_save(self):
        f_temp = self.build_temp_filtro()
        self.filtro.name = f_temp.name
        self.filtro.filter_type = f_temp.filter_type
        self.filtro.formula_str = f_temp.formula_str
        self.filtro.math_apply_band = f_temp.math_apply_band
        self.filtro.math_band_mode = 'bandpass'
        self.filtro.math_invert_band = getattr(f_temp, 'math_invert_band', False)
        self.filtro.math_smooth_w = getattr(f_temp, 'math_smooth_w', 0.0)
        self.filtro.k_low = f_temp.k_low
        self.filtro.k_high = f_temp.k_high
        self.filtro.g_in = f_temp.g_in
        self.filtro.g_out = f_temp.g_out
        self.filtro.tail_low_type = f_temp.tail_low_type
        self.filtro.tail_low_w = f_temp.tail_low_w
        self.filtro.tail_high_type = f_temp.tail_high_type
        self.filtro.tail_high_w = f_temp.tail_high_w
        self.accept()


# ==============================================================================
# MOTOR DE INTERPOLACIÓN (TIN, KRIGING, RBF, IDW, CUBIC, NEAREST)
# ==============================================================================
def interpolate_kriging_ordinary(x, y, z, xi, yi, k_neighbors=16, model='spherical', nugget=0.0, sill=None, range_a=None):
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    grid_pts = np.column_stack([xi.ravel(), yi.ravel()])
    M = len(grid_pts)

    if range_a is None or range_a <= 0:
        span_diag = float(np.hypot(np.ptp(x), np.ptp(y)))
        range_a = max(1e-6, span_diag / 3.0)

    data_var = float(np.var(z))
    if sill is None or sill <= 0:
        sill = max(1e-6, data_var if data_var > 1e-9 else 1.0)

    c0 = float(nugget)
    c = max(1e-6, float(sill) - c0)
    a = float(range_a)

    def semivar(h):
        h = np.maximum(0.0, h)
        if model == 'spherical':
            ratio = np.clip(h / a, 0.0, 1.0)
            return np.where(h <= a, c0 + c * (1.5 * ratio - 0.5 * (ratio ** 3)), c0 + c)
        elif model == 'exponential':
            return c0 + c * (1.0 - np.exp(-3.0 * h / a))
        elif model == 'gaussian':
            return c0 + c * (1.0 - np.exp(-3.0 * (h / a) ** 2))
        else:
            ratio = np.clip(h / a, 0.0, 1.0)
            return np.where(h <= a, c0 + c * ratio, c0 + c)

    k = min(len(pts), max(4, k_neighbors))
    z_est = np.empty(M, dtype=float)
    var_est = np.empty(M, dtype=float)
    chunk_size = 4000

    for start_idx in range(0, M, chunk_size):
        end_idx = min(M, start_idx + chunk_size)
        sub_grid = grid_pts[start_idx:end_idx]
        cur_m = len(sub_grid)

        dists, idxs = tree.query(sub_grid, k=k)
        is_exact = dists[:, 0] < 1e-12

        p_subs = pts[idxs]
        diff = p_subs[:, :, None, :] - p_subs[:, None, :, :]
        D_mat = np.hypot(diff[:, :, :, 0], diff[:, :, :, 1])
        Gamma = semivar(D_mat)

        K_mat = np.empty((cur_m, k + 1, k + 1), dtype=float)
        K_mat[:, :k, :k] = Gamma
        diag_idx = np.arange(k)
        K_mat[:, diag_idx, diag_idx] += 1e-6 * max(1e-6, c)
        K_mat[:, :k, k] = 1.0
        K_mat[:, k, :k] = 1.0
        K_mat[:, k, k] = 0.0

        gamma_vec = np.empty((cur_m, k + 1, 1), dtype=float)
        gamma_vec[:, :k, 0] = semivar(dists)
        gamma_vec[:, k, 0] = 1.0

        try:
            w = np.linalg.solve(K_mat, gamma_vec)
            w_k = w[:, :k, 0]
            z_sub = z[idxs]
            est_chunk = np.sum(w_k * z_sub, axis=1)
            var_chunk = np.maximum(0.0, np.sum(w[:, :, 0] * gamma_vec[:, :, 0], axis=1))
        except np.linalg.LinAlgError:
            w_idw = 1.0 / np.maximum(dists, 1e-12)
            w_idw /= np.sum(w_idw, axis=1, keepdims=True)
            est_chunk = np.sum(w_idw * z[idxs], axis=1)
            var_chunk = np.var(z[idxs], axis=1)

        if np.any(is_exact):
            exact_indices = np.where(is_exact)[0]
            est_chunk[exact_indices] = z[idxs[exact_indices, 0]]
            var_chunk[exact_indices] = 0.0

        z_est[start_idx:end_idx] = est_chunk
        var_est[start_idx:end_idx] = var_chunk

    return z_est.reshape(xi.shape), var_est.reshape(xi.shape)


def interpolate_idw(x, y, z, xi, yi, power=2.0, k_neighbors=16):
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    grid_pts = np.column_stack([xi.ravel(), yi.ravel()])
    k = min(len(pts), max(4, k_neighbors))
    dists, idxs = tree.query(grid_pts, k=k)

    exact = dists[:, 0] < 1e-12
    weights = 1.0 / np.maximum(dists, 1e-12) ** power
    w_sums = np.sum(weights, axis=1, keepdims=True)
    norm_w = weights / w_sums

    z_vals = z[idxs]
    z_est = np.sum(norm_w * z_vals, axis=1)
    z_est[exact] = z[idxs[exact, 0]]
    return z_est.reshape(xi.shape)


def run_interpolation(x, y, z, x_grid, y_grid, method='tin', params=None):
    if params is None:
        params = {}
    xi, yi = np.meshgrid(x_grid, y_grid)
    pts = np.column_stack([x, y])
    var_grid = None

    if method == 'tin':
        Z_grid = griddata(pts, z, (xi, yi), method='linear')
        if np.any(np.isnan(Z_grid)):
            mask_nan = np.isnan(Z_grid)
            z_nearest = griddata(pts, z, (xi, yi), method='nearest')
            Z_grid[mask_nan] = z_nearest[mask_nan]

    elif method == 'kriging':
        k_nb = params.get('k_neighbors', 16)
        mod = params.get('model', 'spherical')
        Z_grid, var_grid = interpolate_kriging_ordinary(x, y, z, xi, yi, k_neighbors=k_nb, model=mod)

    elif method == 'rbf':
        kernel = params.get('kernel', 'thin_plate_spline')
        if len(x) > 4000:
            step = int(np.ceil(len(x) / 3500.0))
            sub_x, sub_y, sub_z = x[::step], y[::step], z[::step]
            sub_pts = np.column_stack([sub_x, sub_y])
        else:
            sub_pts, sub_z = pts, z
        rbf = RBFInterpolator(sub_pts, sub_z, kernel=kernel)
        eval_pts = np.column_stack([xi.ravel(), yi.ravel()])
        Z_grid = rbf(eval_pts).reshape(xi.shape)

    elif method == 'idw':
        p = params.get('power', 2.0)
        k_nb = params.get('k_neighbors', 16)
        Z_grid = interpolate_idw(x, y, z, xi, yi, power=p, k_neighbors=k_nb)

    elif method == 'cubic':
        Z_grid = griddata(pts, z, (xi, yi), method='cubic')
        if np.any(np.isnan(Z_grid)):
            mask_nan = np.isnan(Z_grid)
            z_nearest = griddata(pts, z, (xi, yi), method='nearest')
            Z_grid[mask_nan] = z_nearest[mask_nan]

    elif method == 'nearest':
        Z_grid = griddata(pts, z, (xi, yi), method='nearest')

    else:
        Z_grid = griddata(pts, z, (xi, yi), method='linear')

    # Matriz de pesos P (inverso a la distancia al punto de medición más cercano)
    try:
        tree = cKDTree(pts)
        grid_pts = np.column_stack([xi.ravel(), yi.ravel()])
        dists, _ = tree.query(grid_pts, k=1)
        dists = dists.reshape(xi.shape)
        dx_approx = abs(x_grid[1] - x_grid[0]) if len(x_grid) > 1 else 100.0
        dy_approx = abs(y_grid[1] - y_grid[0]) if len(y_grid) > 1 else 100.0
        d0 = 0.5 * min(dx_approx, dy_approx)
        weights_grid = 1.0 / np.maximum(dists, d0)
    except Exception:
        weights_grid = np.ones_like(Z_grid, dtype=np.float64)

    return Z_grid, var_grid, weights_grid


class InterpolationThread(QThread):
    finished = pyqtSignal(object, object, object, str)

    def __init__(self, x, y, z, x_grid, y_grid, method, params):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.method = method
        self.params = params

    def run(self):
        try:
            Z_grid, var_grid, weights_grid = run_interpolation(
                self.x, self.y, self.z,
                self.x_grid, self.y_grid,
                method=self.method, params=self.params
            )
            self.finished.emit(Z_grid, var_grid, weights_grid, "")
        except Exception as e:
            self.finished.emit(None, None, None, str(e))


# ==============================================================================
# VENTANA FLOTANTE DESACOPLABLE (MULTI-MONITOR PARA VISUALIZADOR MAESTRO)
# ==============================================================================
class DetachedSubWindow(QMainWindow):
    """Ventana independiente para permitir separar pestañas en múltiples monitores."""
    def __init__(self, key, title, widget, main_window, original_index):
        super().__init__()
        self.key = key
        self.original_title = title
        self.widget = widget
        self.main_window = main_window
        self.original_index = original_index

        self.setWindowTitle(f"{title} - GravTT")
        self.resize(1150, 780)
        self.setStyleSheet(WIN_STYLESHEET)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(4)

        header = QFrame()
        header.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:0.35 #ebebeb, stop:1 #d8d8d8); border: 1px solid #707070; padding: 3px;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 2, 8, 2)

        lbl = QLabel(f"<span style='color: #cc0000;'></span> <b>{title}</b> (Ventana Independiente Flotante / Multi-Monitor)")
        lbl.setStyleSheet("color: #000000; font-size: 12px; font-weight: bold;")
        h_layout.addWidget(lbl)
        h_layout.addStretch()

        btn_dock = QPushButton("Reacoplar a la Ventana Principal")
        btn_dock.setCursor(Qt.PointingHandCursor)
        btn_dock.clicked.connect(self.dock_back)
        h_layout.addWidget(btn_dock)

        vbox.addWidget(header)
        vbox.addWidget(widget)
        self.setCentralWidget(container)

    def dock_back(self):
        self.close()

    def closeEvent(self, event):
        if self.key in self.main_window.detached_windows:
            del self.main_window.detached_windows[self.key]
        if self.widget is not None and self.main_window.tabs is not None:
            self.centralWidget().layout().removeWidget(self.widget)
            idx = min(self.original_index, self.main_window.tabs.count())
            self.main_window.tabs.insertTab(idx, self.widget, self.original_title)
            self.main_window.tabs.setCurrentIndex(idx)
            for c in self.widget.findChildren(FigureCanvas):
                c.draw()
        event.accept()


# ==============================================================================
# APLICACIÓN PRINCIPAL: GENERADOR DE MALLAS 2D Y ANÁLISIS FOURIER
# ==============================================================================
class GeneradorMallas2DWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GravTT")
        icon_path = os.path.join(BASE_DIR, 'assets', 'icon.ico')
        if not os.path.exists(icon_path):
            icon_path = os.path.join(BASE_DIR, 'assets', 'logo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1280, 850)
        self.setMinimumSize(1040, 680)
        self.setStyleSheet(WIN_STYLESHEET)

        # Unidades físicas configurables
        self.unit_xy = "m"       # Unidad horizontal / espacial
        self.unit_z = "mGal"    # Unidad del atributo Z

        # Estado de datos crudos
        self.raw_filepath = None
        self.df_raw = None
        self.pts_x = None
        self.pts_y = None
        self.pts_z = None
        self.x_col = "X"
        self.y_col = "Y"
        self.z_col = "Z"

        # Variables del Reductor Gravimétrico Secuencial Integrado (Sin ejemplos precargados)
        self.raw_grav_df = None
        self.df_raw_source = None
        self.mdt_obj = GeoTiffMDT()
        self.terrain_worker = None
        self.step_g0_done = False
        self.step_fa_done = False
        self.step_sb_done = False
        self.step_tc_done = False

        # Bounding box y espaciado de celda
        self.x_min_sel = 0.0
        self.x_max_sel = 100.0
        self.y_min_sel = 0.0
        self.y_max_sel = 100.0
        self.dx_val = 1.0
        self.dy_val = 1.0

        # Malla regular espacial
        self.grid_x_vec = None
        self.grid_y_vec = None
        self.grid_Z = None
        self.grid_Z_orig = None
        self.grid_var = None
        self.grid_pts_sub_x = None
        self.grid_pts_sub_y = None
        self.grid_pts_sub_z = None
        self.grid_eq_source_z0 = None
        self.grid_eq_source_z0_elev = 0.0
        self.grid_eq_source_x = None
        self.grid_eq_source_y = None
        self.grid_eq_source_z0_filt = None
        self.dlg_problema_inverso = None
        self.interp_thread = None

        # Fourier 2D
        self.F_shift_2d = None
        self.U_grid = None
        self.V_grid = None
        self.K_grid = None
        self.Theta_grid = None
        self.u_vec = None
        self.v_vec = None
        self.rect_selector = None

        # Pila de Filtros 2D: INICIA COMPLETAMENTE VACÍA (Sin filtros precargados)
        self.filtros_2d = []
        self.grid_Z_filt = None
        self.grid_Z_tapered = None
        self.F_shift_filt = None
        self.H_total_2d = None

        # Modelado Espectral Sintético (Blakely 1995) y Comparación de Modelos
        self.cuerpos_sinteticos = []
        self.grid_Z_model = None
        self.model_x_vec = None
        self.model_y_vec = None
        self.dlg_synthetic_modeler = None
        self.dlg_model_comparison = None

        # MDT Topografía y Matriz de Pesos P (Convivencia de Datos)
        self.grid_mdt = None
        self.mdt_x_vec = None
        self.mdt_y_vec = None
        self.mdt_filepath = None
        self.grid_weights = None

        # Ventanas flotantes multi-monitor
        self.detached_windows = {}

        # Estado del Visualizador Maestro
        self.master_active_data = None
        self.master_active_extent = None
        self.master_active_is_freq = False
        self.master_active_unit = ""
        self.master_active_title = ""

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Barra superior
        top_bar = QHBoxLayout()
        logo_png = os.path.join(BASE_DIR, 'assets', 'logo.png')
        if os.path.exists(logo_png):
            lbl_logo = QLabel()
            lbl_logo.setPixmap(QPixmap(logo_png).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            top_bar.addWidget(lbl_logo)
        lbl_app = QLabel("<b>GravTT</b>")
        lbl_app.setStyleSheet("color: #cc0000; font-size: 16px; font-weight: bold; letter-spacing: 1px;")
        top_bar.addWidget(lbl_app)

        top_bar.addStretch()

        self.lbl_file_status = QLabel("Ningún archivo cargado")
        self.lbl_file_status.setStyleSheet("color: #404040; font-weight: bold; font-size: 11px;")
        top_bar.addWidget(self.lbl_file_status)
        main_layout.addLayout(top_bar)

        # Menú superior
        menu_bar = self.menuBar()
        menu_file = menu_bar.addMenu("Archivo")
        act_open_pts = menu_file.addAction("Cargar Puntos de Observaciones (CSV / Excel / TXT)...")
        act_open_pts.triggered.connect(self.browse_file)
        act_open_tiff_g = menu_file.addAction("Cargar Raster TIFF de Gravedad con Parámetros...")
        act_open_tiff_g.triggered.connect(self.browse_tiff_gravity_mesh)
        menu_file.addSeparator()
        ico_file = os.path.join(BASE_DIR, 'assets', 'icon.ico')
        ico_obj = QIcon(ico_file) if os.path.exists(ico_file) else QIcon()
        act_about = menu_file.addAction(ico_obj, "Acerca de GravTT...")
        act_about.triggered.connect(self.show_about_dialog)

        menu_inv = menu_bar.addMenu("Problema Inverso")
        act_inv = menu_inv.addAction("Fuente Equivalente (Continuación a Plano Z0 con MDT)...")
        act_inv.triggered.connect(self.open_fuente_equivalente_window)

        menu_synth = menu_bar.addMenu("Modelado y Comparación")
        act_m_synth = menu_synth.addAction("Modelador de Anomalías Sintéticas (Blakely 2D)...")
        act_m_synth.triggered.connect(self.open_synthetic_modeler)
        act_m_comp = menu_synth.addAction("Comparación de Modelos (Obs vs Mod Álgebra)...")
        act_m_comp.triggered.connect(self.open_model_comparison)

        menu_help = menu_bar.addMenu("Ayuda")
        act_help_about = menu_help.addAction(ico_obj, "Acerca de GravTT...")
        act_help_about.triggered.connect(self.show_about_dialog)

        # Pestañas principales
        self.tabs = QTabWidget()
        self.tab_data = QWidget()
        self.tab_synth = QWidget()
        self.tab_view = QWidget()
        self.tab_fourier = QWidget()
        self.tab_table = QWidget()

        self.tabs.addTab(self.tab_data, "1. Datos e Interpolación (TIN / Kriging)")
        self.tabs.addTab(self.tab_synth, "2. Modelado")
        self.tabs.addTab(self.tab_view, "3. Visualizador Maestro 2D")
        self.tabs.addTab(self.tab_fourier, "4. Filtros y Análisis Fourier 2D")
        self.tabs.addTab(self.tab_table, "5. Tabla y Exportación")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tabs)

        self.build_tab_data(self.tab_data)
        self.build_tab_synth(self.tab_synth)
        self.build_tab_master_viewer(self.tab_view)
        self.build_tab_fourier(self.tab_fourier)
        self.build_tab_table(self.tab_table)

        # Barra de progreso y estado
        bottom_bar = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0); self.progress_bar.setFixedHeight(16); self.progress_bar.setVisible(False)
        bottom_bar.addWidget(self.progress_bar)

        self.status = QStatusBar()
        self.status.showMessage("Listo para importar tabla de puntos dispersos (CSV, Excel, JSON, TXT).")
        bottom_bar.addWidget(self.status)
        main_layout.addLayout(bottom_bar)

    def on_tab_changed(self, idx):
        if idx == 1 and hasattr(self, 'widget_synth') and self.widget_synth is not None:
            try:
                has_spatial = (
                    (self.grid_x_vec is not None and self.grid_y_vec is not None and len(self.grid_x_vec) > 1) or
                    (getattr(self, 'x_min_sel', None) is not None and getattr(self, 'x_max_sel', None) is not None and getattr(self, 'y_min_sel', None) is not None) or
                    (self.pts_x is not None and len(self.pts_x) > 0)
                )
                self.widget_synth.rad_grid_auto.setEnabled(has_spatial)
                if has_spatial and self.widget_synth.rad_grid_auto.isChecked():
                    self.widget_synth.init_grid_coordinates()
                self.widget_synth.recalculate_model()
                total_w = self.widget_synth.width()
                if total_w > 800:
                    left_w = min(420, int(total_w * 0.35))
                    right_w = total_w - left_w
                    self.widget_synth.splitter_main.setSizes([left_w, right_w])
                self.widget_synth.canvas.draw_idle()
            except Exception as e:
                print(f"Error sincronizando pestaña sintética: {e}")

    # ==========================================================================
    # PESTAÑA 1: CARGA DE DATOS, ZONA & INTERPOLACIÓN (TIN, KRIGING, IDW, ETC.)
    # ==========================================================================
    def build_tab_data(self, parent):
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Panel izquierdo dentro de un ScrollArea para navegación cómoda y completa
        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left.setFrameShape(QFrame.NoFrame)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        panel_left = QWidget()
        vbox_left = QVBoxLayout(panel_left)
        vbox_left.setContentsMargins(4, 4, 4, 4)
        vbox_left.setSpacing(8)

        # 1. Carga de Archivo, Atributos y Unidades Físicas
        sec_col = CollapsibleSection("1. Modalidad y Carga de Observaciones", is_expanded=True)

        sec_col.addWidget(QLabel("<b>Tipo de Observaciones a Cargar:</b>"))
        self.combo_obs_mode = QComboBox()
        self.combo_obs_mode.addItem("Observaciones Corregidas (Anomalía Directa)", "corregidas")
        self.combo_obs_mode.addItem("Observaciones Crudas (Reducción Gravimétrica Secuencial)", "crudas")
        self.combo_obs_mode.currentIndexChanged.connect(self.on_obs_mode_changed)
        sec_col.addWidget(self.combo_obs_mode)

        # Panel para Observaciones Corregidas (Carga directa habitual)
        self.box_obs_corregidas = QWidget()
        v_corr = QVBoxLayout(self.box_obs_corregidas)
        v_corr.setContentsMargins(0, 4, 0, 4)

        btn_open_inline = QPushButton("Cargar Archivo de Puntos (CSV / Excel / TXT)...")
        btn_open_inline.clicked.connect(self.browse_file)
        v_corr.addWidget(btn_open_inline)

        btn_open_tiff_g = QPushButton("Cargar Malla / Raster TIFF de Gravedad con Parámetros...")
        btn_open_tiff_g.setStyleSheet("background-color: #0d47a1; color: #ffffff; font-weight: bold; font-size: 11px; padding: 6px; margin-top: 3px; border-radius: 3px;")
        btn_open_tiff_g.clicked.connect(self.browse_tiff_gravity_mesh)
        v_corr.addWidget(btn_open_tiff_g)

        btn_open_tiff_mdt = QPushButton("⛰️ Cargar Raster Topografía MDT (TIFF / GeoTIFF)...")
        btn_open_tiff_mdt.setStyleSheet("background-color: #2e7d32; color: #ffffff; font-weight: bold; font-size: 11px; padding: 6px; margin-top: 3px; border-radius: 3px;")
        btn_open_tiff_mdt.clicked.connect(self.browse_tiff_mdt)
        v_corr.addWidget(btn_open_tiff_mdt)

        grid_c = QGridLayout()
        grid_c.addWidget(QLabel("<b>Coordenada X (Este / Lon):</b>"), 0, 0)
        self.combo_col_x = QComboBox()
        self.combo_col_x.currentIndexChanged.connect(self.on_columns_selected)
        grid_c.addWidget(self.combo_col_x, 0, 1)

        grid_c.addWidget(QLabel("<b>Coordenada Y (Norte / Lat):</b>"), 1, 0)
        self.combo_col_y = QComboBox()
        self.combo_col_y.currentIndexChanged.connect(self.on_columns_selected)
        grid_c.addWidget(self.combo_col_y, 1, 1)

        grid_c.addWidget(QLabel("<b>Valor Z (Anomalía / Elev):</b>"), 2, 0)
        self.combo_col_z = QComboBox()
        self.combo_col_z.currentIndexChanged.connect(self.on_columns_selected)
        grid_c.addWidget(self.combo_col_z, 2, 1)

        # SELECTORES DE UNIDADES FÍSICAS
        grid_c.addWidget(QLabel("<b>Unidad Coordenadas XY:</b>"), 3, 0)
        self.combo_unit_xy = QComboBox()
        self.combo_unit_xy.setEditable(True)
        self.combo_unit_xy.addItem("Metros [m]", "m")
        self.combo_unit_xy.addItem("Kilómetros [km]", "km")
        self.combo_unit_xy.addItem("Pies [ft]", "ft")
        self.combo_unit_xy.addItem("Grados [°]", "°")
        self.combo_unit_xy.addItem("Centímetros [cm]", "cm")
        self.combo_unit_xy.addItem("Milímetros [mm]", "mm")
        self.combo_unit_xy.addItem("Unidades [u]", "u")
        self.combo_unit_xy.currentIndexChanged.connect(self.on_units_changed)
        if self.combo_unit_xy.lineEdit():
            self.combo_unit_xy.lineEdit().editingFinished.connect(self.on_units_changed)
        grid_c.addWidget(self.combo_unit_xy, 3, 1)

        grid_c.addWidget(QLabel("<b>Unidad Atributo Z:</b>"), 4, 0)
        self.combo_unit_z = QComboBox()
        self.combo_unit_z.setEditable(True)
        self.combo_unit_z.addItem("mGal (Gravedad)", "mGal")
        self.combo_unit_z.addItem("µGal (Microgravedad)", "µGal")
        self.combo_unit_z.addItem("nT (Magnetometría)", "nT")
        self.combo_unit_z.addItem("m (Metros / Cota)", "m")
        self.combo_unit_z.addItem("km (Kilómetros)", "km")
        self.combo_unit_z.addItem("g.u. (Gravity units)", "g.u.")
        self.combo_unit_z.addItem("Eötvös (Gradiente)", "Eötvös")
        self.combo_unit_z.addItem("u.a. (Adimensional)", "u.a.")
        self.combo_unit_z.currentIndexChanged.connect(self.on_units_changed)
        if self.combo_unit_z.lineEdit():
            self.combo_unit_z.lineEdit().editingFinished.connect(self.on_units_changed)
        grid_c.addWidget(self.combo_unit_z, 4, 1)
        v_corr.addLayout(grid_c)

        btn_apply_cols = QPushButton("Aplicar Atributos y Unidades")
        btn_apply_cols.setObjectName("btnPrimary")
        btn_apply_cols.clicked.connect(self.apply_column_selection)
        v_corr.addWidget(btn_apply_cols)
        sec_col.addWidget(self.box_obs_corregidas)

        # Panel para Observaciones Crudas (Reducción Gravimétrica Secuencial Integrada)
        self.box_obs_crudas = QWidget()
        v_crud = QVBoxLayout(self.box_obs_crudas)
        v_crud.setContentsMargins(0, 4, 0, 4)
        v_crud.setSpacing(6)

        # 1. Carga de Tabla y Asignación de Campos (Idéntico a Observaciones Corregidas)
        grp_data = QGroupBox("1. Tabla de Mediciones y Asignación de Campos")
        v_data = QVBoxLayout(grp_data)

        btn_load_raw_table = QPushButton("Cargar Tabla de Mediciones Crudas (CSV / Excel / TXT)...")
        btn_load_raw_table.setObjectName("btnAccent")
        btn_load_raw_table.clicked.connect(self.on_load_raw_measurements_file)
        v_data.addWidget(btn_load_raw_table)

        self.lbl_raw_file_info = QLabel("Ningún archivo cargado.")
        self.lbl_raw_file_info.setStyleSheet("font-size: 10px; color: #666666;")
        v_data.addWidget(self.lbl_raw_file_info)

        grid_raw_fields = QGridLayout()

        grid_raw_fields.addWidget(QLabel("<b>Coordenada X (Este / Lon):</b>"), 0, 0)
        self.combo_raw_col_x = QComboBox()
        grid_raw_fields.addWidget(self.combo_raw_col_x, 0, 1)

        grid_raw_fields.addWidget(QLabel("<b>Coordenada Y (Norte / Lat):</b>"), 1, 0)
        self.combo_raw_col_y = QComboBox()
        grid_raw_fields.addWidget(self.combo_raw_col_y, 1, 1)

        grid_raw_fields.addWidget(QLabel("<b>Cota / Altura h [m]:</b>"), 2, 0)
        self.combo_raw_col_h = QComboBox()
        grid_raw_fields.addWidget(self.combo_raw_col_h, 2, 1)

        grid_raw_fields.addWidget(QLabel("<b>Latitud λ [°] (para g₀):</b>"), 3, 0)
        self.combo_raw_col_lat = QComboBox()
        grid_raw_fields.addWidget(self.combo_raw_col_lat, 3, 1)

        grid_raw_fields.addWidget(QLabel("<b>Longitud [°] (opcional):</b>"), 4, 0)
        self.combo_raw_col_lon = QComboBox()
        grid_raw_fields.addWidget(self.combo_raw_col_lon, 4, 1)

        grid_raw_fields.addWidget(QLabel("<b>Gravedad Obs. G [mGal]:</b>"), 5, 0)
        self.combo_raw_col_g = QComboBox()
        grid_raw_fields.addWidget(self.combo_raw_col_g, 5, 1)

        v_data.addLayout(grid_raw_fields)

        btn_apply_raw = QPushButton("Aplicar Campos y Graficar en Mapa 2D")
        btn_apply_raw.setObjectName("btnPrimary")
        btn_apply_raw.clicked.connect(self.apply_raw_column_selection)
        v_data.addWidget(btn_apply_raw)
        v_crud.addWidget(grp_data)

        # 2. Elipsoide Normal g0 (Somigliana)
        grp_g0 = QGroupBox("2. Modelo de Elipsoide Normal (g₀ Somigliana)")
        v_g0 = QVBoxLayout(grp_g0)
        h_elip = QHBoxLayout()
        h_elip.addWidget(QLabel("<b>Modelo:</b>"))
        self.combo_ellipsoid = QComboBox()
        for k, v in ELLIPSOID_MODELS.items():
            self.combo_ellipsoid.addItem(v['name'], k)
        self.combo_ellipsoid.currentIndexChanged.connect(self.on_raw_ellipsoid_changed)
        h_elip.addWidget(self.combo_ellipsoid)
        v_g0.addLayout(h_elip)

        self.lbl_formula_g0 = QLabel(
            "<code>g₀(λ) = γ_a · (1 + k·sin²λ) / √(1 - e²·sin²λ)</code>"
        )
        self.lbl_formula_g0.setWordWrap(True)
        self.lbl_formula_g0.setStyleSheet("font-size: 9px; color: #003366; background: #eaf2fb; padding: 3px; border: 1px solid #c2d8f2;")
        v_g0.addWidget(self.lbl_formula_g0)
        v_crud.addWidget(grp_g0)

        # 3. Corrección de Aire Libre (FAC)
        grp_fa = QGroupBox("3. Corrección de Aire Libre (FAC)")
        grid_fa = QGridLayout(grp_fa)
        grid_fa.addWidget(QLabel("<b>Gradiente [mGal/m]:</b>"), 0, 0)
        self.spin_raw_grad_fa = QDoubleSpinBox()
        self.spin_raw_grad_fa.setRange(0.20, 0.40); self.spin_raw_grad_fa.setDecimals(4); self.spin_raw_grad_fa.setValue(0.3086)
        grid_fa.addWidget(self.spin_raw_grad_fa, 0, 1)

        grid_fa.addWidget(QLabel("<b>Ondulación Geoidal N [m]:</b>"), 1, 0)
        self.spin_raw_geoid_n = QDoubleSpinBox()
        self.spin_raw_geoid_n.setRange(-200.0, 200.0); self.spin_raw_geoid_n.setValue(0.0)
        grid_fa.addWidget(self.spin_raw_geoid_n, 1, 1)

        self.btn_apply_fa = QPushButton("Aplicar Aire Libre (+0.3086·h)")
        self.btn_apply_fa.setEnabled(False)
        self.btn_apply_fa.clicked.connect(self.run_raw_step_free_air)
        grid_fa.addWidget(self.btn_apply_fa, 2, 0, 1, 2)
        v_crud.addWidget(grp_fa)

        # 4. Corrección de Bouguer
        grp_b = QGroupBox("4. Corrección de Placa de Bouguer")
        grid_b = QGridLayout(grp_b)
        grid_b.addWidget(QLabel("<b>Densidad ρ [kg/m³]:</b>"), 0, 0)
        self.spin_raw_rho_b = QDoubleSpinBox()
        self.spin_raw_rho_b.setRange(500.0, 5000.0); self.spin_raw_rho_b.setValue(2670.0)
        self.spin_raw_rho_b.valueChanged.connect(self.update_raw_bouguer_factor_label)
        grid_b.addWidget(self.spin_raw_rho_b, 0, 1)

        self.lbl_raw_factor_b = QLabel("Factor: 2·π·G·ρ ≈ 0.1119 mGal/m")
        self.lbl_raw_factor_b.setStyleSheet("font-size: 10px; color: #555555;")
        grid_b.addWidget(self.lbl_raw_factor_b, 1, 0, 1, 2)

        self.btn_apply_b = QPushButton("Aplicar Bouguer (-0.1119·h)")
        self.btn_apply_b.setEnabled(False)
        self.btn_apply_b.clicked.connect(self.run_raw_step_bouguer)
        grid_b.addWidget(self.btn_apply_b, 2, 0, 1, 2)
        v_crud.addWidget(grp_b)

        # 5. Corrección de Topografía con MDT
        grp_t = QGroupBox("5. Corrección de Topografía con MDT (.tif)")
        grid_t = QGridLayout(grp_t)
        btn_sel_mdt = QPushButton("Cargar Modelo Digital MDT (.tif)...")
        btn_sel_mdt.setObjectName("btnPrimary")
        btn_sel_mdt.clicked.connect(self.on_select_raw_mdt_file)
        grid_t.addWidget(btn_sel_mdt, 0, 0, 1, 2)

        self.lbl_raw_mdt_status = QLabel("MDT no cargado.")
        self.lbl_raw_mdt_status.setStyleSheet("font-size: 10px; color: #666666;")
        grid_t.addWidget(self.lbl_raw_mdt_status, 1, 0, 1, 2)

        grid_t.addWidget(QLabel("<b>Radio R_max [m]:</b>"), 2, 0)
        self.spin_raw_rmax = QDoubleSpinBox()
        self.spin_raw_rmax.setRange(500.0, 100000.0); self.spin_raw_rmax.setSingleStep(1000.0); self.spin_raw_rmax.setValue(8000.0)
        grid_t.addWidget(self.spin_raw_rmax, 2, 1)

        self.progress_raw_terrain = QProgressBar()
        self.progress_raw_terrain.setRange(0, 100)
        self.progress_raw_terrain.setValue(0)
        grid_t.addWidget(self.progress_raw_terrain, 3, 0, 1, 2)

        h_t_actions = QHBoxLayout()
        self.btn_apply_t = QPushButton("Aplicar Topografía (+T)")
        self.btn_apply_t.setObjectName("btnAccent")
        self.btn_apply_t.setEnabled(False)
        self.btn_apply_t.clicked.connect(self.run_raw_step_terrain)
        h_t_actions.addWidget(self.btn_apply_t)

        self.btn_cancel_raw_t = QPushButton("Cancelar")
        self.btn_cancel_raw_t.setEnabled(False)
        self.btn_cancel_raw_t.clicked.connect(self.on_cancel_raw_terrain)
        h_t_actions.addWidget(self.btn_cancel_raw_t)
        grid_t.addLayout(h_t_actions, 4, 0, 1, 2)
        v_crud.addWidget(grp_t)

        # Exportar CSV reducido (opcional)
        btn_exp_raw = QPushButton("Guardar Tabla Reducida a CSV...")
        btn_exp_raw.clicked.connect(self.on_export_raw_reduced_csv)
        v_crud.addWidget(btn_exp_raw)

        sec_col.addWidget(self.box_obs_crudas)
        self.box_obs_crudas.setVisible(False)
        vbox_left.addWidget(sec_col)

        # 2. Delimitación de Zona (Rectángulo)
        sec_zone = CollapsibleSection("2. Delimitación de Zona (Rectángulo)", is_expanded=True)
        sec_zone.addWidget(QLabel("Ingresa coordenadas o arrastra en el mapa derecho:"))

        grid_z = QGridLayout()
        self.lbl_xmin = QLabel(f"X Mín [{self.unit_xy}]:")
        grid_z.addWidget(self.lbl_xmin, 0, 0)
        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setRange(-1e9, 1e9); self.spin_xmin.setDecimals(5)
        self.spin_xmin.valueChanged.connect(self.on_zone_spins_changed)
        grid_z.addWidget(self.spin_xmin, 0, 1)

        self.lbl_xmax = QLabel(f"X Máx [{self.unit_xy}]:")
        grid_z.addWidget(self.lbl_xmax, 0, 2)
        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setRange(-1e9, 1e9); self.spin_xmax.setDecimals(5)
        self.spin_xmax.valueChanged.connect(self.on_zone_spins_changed)
        grid_z.addWidget(self.spin_xmax, 0, 3)

        self.lbl_ymin = QLabel(f"Y Mín [{self.unit_xy}]:")
        grid_z.addWidget(self.lbl_ymin, 1, 0)
        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-1e9, 1e9); self.spin_ymin.setDecimals(5)
        self.spin_ymin.valueChanged.connect(self.on_zone_spins_changed)
        grid_z.addWidget(self.spin_ymin, 1, 1)

        self.lbl_ymax = QLabel(f"Y Máx [{self.unit_xy}]:")
        grid_z.addWidget(self.lbl_ymax, 1, 2)
        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-1e9, 1e9); self.spin_ymax.setDecimals(5)
        self.spin_ymax.valueChanged.connect(self.on_zone_spins_changed)
        grid_z.addWidget(self.spin_ymax, 1, 3)
        sec_zone.addLayout(grid_z)

        row_zoom = QHBoxLayout()
        btn_full_ext = QPushButton("Toda la Extensión")
        btn_full_ext.setObjectName("btnReset")
        btn_full_ext.clicked.connect(self.reset_zone_to_full)
        row_zoom.addWidget(btn_full_ext)

        btn_zoom_all = QPushButton("Ajustar a Datos")
        btn_zoom_all.setObjectName("btnReset")
        btn_zoom_all.clicked.connect(self.zoom_zone_all)
        row_zoom.addWidget(btn_zoom_all)

        btn_zoom_zone = QPushButton("Zoom a Selección")
        btn_zoom_zone.setObjectName("btnReset")
        btn_zoom_zone.clicked.connect(self.zoom_zone_selected)
        row_zoom.addWidget(btn_zoom_zone)
        sec_zone.addLayout(row_zoom)
        vbox_left.addWidget(sec_zone)

        # 3. Espaciado de Malla Regular con Unidades
        sec_res = CollapsibleSection("3. Espaciado de Malla Regular (deltaX, deltaY)", is_expanded=True)
        grid_s = QGridLayout()
        self.lbl_dx = QLabel(f"<b>deltaX (Paso X) [{self.unit_xy}]:</b>")
        grid_s.addWidget(self.lbl_dx, 0, 0)
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(1e-6, 1e8); self.spin_dx.setDecimals(5); self.spin_dx.setValue(1.0)
        self.spin_dx.valueChanged.connect(self.on_spacing_changed)
        grid_s.addWidget(self.spin_dx, 0, 1)

        self.lbl_dy = QLabel(f"<b>deltaY (Paso Y) [{self.unit_xy}]:</b>")
        grid_s.addWidget(self.lbl_dy, 1, 0)
        self.spin_dy = QDoubleSpinBox()
        self.spin_dy.setRange(1e-6, 1e8); self.spin_dy.setDecimals(5); self.spin_dy.setValue(1.0)
        self.spin_dy.valueChanged.connect(self.on_spacing_changed)
        grid_s.addWidget(self.spin_dy, 1, 1)
        sec_res.addLayout(grid_s)

        self.chk_square_cells = QCheckBox("Forzar celdas cuadradas (deltaX = deltaY)")
        self.chk_square_cells.setChecked(True)
        self.chk_square_cells.stateChanged.connect(self.on_square_cells_toggled)
        sec_res.addWidget(self.chk_square_cells)

        btn_sugg = QPushButton("Sugerir espaciado óptimo")
        btn_sugg.setObjectName("btnMini")
        btn_sugg.clicked.connect(self.suggest_optimal_spacing)
        sec_res.addWidget(btn_sugg)

        self.lbl_grid_dims = QLabel("Malla resultante: 0 cols (X) × 0 filas (Y) = 0 nodos.")
        self.lbl_grid_dims.setStyleSheet("font-size: 11px; font-weight: bold; color: #cc0000; padding-top: 4px;")
        sec_res.addWidget(self.lbl_grid_dims)
        vbox_left.addWidget(sec_res)

        # 4. Metodología de Interpolación
        sec_method = CollapsibleSection("4. Metodología de Interpolación (TIN, Kriging, etc.)", is_expanded=True)
        self.combo_method = QComboBox()
        self.combo_method.addItem("TIN / Red Delaunay (Lineal + Nearest)", "tin")
        self.combo_method.addItem("Kriging Ordinario (Con Semivariograma)", "kriging")
        self.combo_method.addItem("RBF / Splines Radiales (Multiquadric / TPS)", "rbf")
        self.combo_method.addItem("IDW (Ponderación Inversa a la Distancia)", "idw")
        self.combo_method.addItem("Spline Cúbico 2D (Clough-Tocher)", "cubic")
        self.combo_method.addItem("Vecino Más Cercano (Nearest Neighbor)", "nearest")
        self.combo_method.currentIndexChanged.connect(self.on_method_changed)
        sec_method.addWidget(self.combo_method)

        self.panel_krig_params = QWidget()
        k_lay = QGridLayout(self.panel_krig_params)
        k_lay.setContentsMargins(0, 4, 0, 0)
        k_lay.addWidget(QLabel("Modelo Semivariograma:"), 0, 0)
        self.combo_krig_model = QComboBox()
        self.combo_krig_model.addItems(["spherical", "exponential", "gaussian"])
        k_lay.addWidget(self.combo_krig_model, 0, 1)

        k_lay.addWidget(QLabel("Vecinos (k):"), 1, 0)
        self.spin_krig_k = QSpinBox(); self.spin_krig_k.setRange(4, 64); self.spin_krig_k.setValue(16)
        k_lay.addWidget(self.spin_krig_k, 1, 1)
        sec_method.addWidget(self.panel_krig_params)
        self.panel_krig_params.setVisible(False)

        self.panel_idw_params = QWidget()
        idw_lay = QGridLayout(self.panel_idw_params)
        idw_lay.setContentsMargins(0, 4, 0, 0)
        idw_lay.addWidget(QLabel("Exponente de potencia (p):"), 0, 0)
        self.spin_idw_power = QDoubleSpinBox(); self.spin_idw_power.setRange(0.5, 6.0); self.spin_idw_power.setValue(2.0)
        idw_lay.addWidget(self.spin_idw_power, 0, 1)
        idw_lay.addWidget(QLabel("Vecinos (k):"), 1, 0)
        self.spin_idw_k = QSpinBox(); self.spin_idw_k.setRange(4, 64); self.spin_idw_k.setValue(16)
        idw_lay.addWidget(self.spin_idw_k, 1, 1)
        sec_method.addWidget(self.panel_idw_params)
        self.panel_idw_params.setVisible(False)

        self.panel_rbf_params = QWidget()
        rbf_lay = QGridLayout(self.panel_rbf_params)
        rbf_lay.setContentsMargins(0, 4, 0, 0)
        rbf_lay.addWidget(QLabel("Kernel RBF:"), 0, 0)
        self.combo_rbf_kernel = QComboBox()
        self.combo_rbf_kernel.addItems(["thin_plate_spline", "multiquadric", "gaussian", "linear", "cubic"])
        rbf_lay.addWidget(self.combo_rbf_kernel, 0, 1)
        sec_method.addWidget(self.panel_rbf_params)
        self.panel_rbf_params.setVisible(False)

        self.btn_run_grid = QPushButton("GENERAR MALLA REGULAR 2D")
        self.btn_run_grid.setObjectName("btnPrimary")
        self.btn_run_grid.setStyleSheet("font-weight: bold; font-size: 12px; padding: 8px;")
        self.btn_run_grid.clicked.connect(self.start_interpolation)
        sec_method.addWidget(self.btn_run_grid)

        self.btn_open_eq_source = QPushButton("Problema Inverso: Fuente Equivalente (Continuación a Plano Z0 con MDT)")
        self.btn_open_eq_source.setStyleSheet("background-color: #990000; color: #ffffff; font-weight: bold; font-size: 11px; padding: 7px; margin-top: 5px; border: 1px solid #660000; border-radius: 3px;")
        self.btn_open_eq_source.clicked.connect(self.open_fuente_equivalente_window)
        sec_method.addWidget(self.btn_open_eq_source)
        vbox_left.addWidget(sec_method)

        # 5. Estadísticas descriptivas XYZ con unidades y Vista Previa (colapsable)
        sec_stats = CollapsibleSection("5. Estadísticas Descriptivas XYZ y Vista Previa", is_expanded=False)
        self.lbl_stats = QLabel("No hay datos cargados.")
        self.lbl_stats.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; background-color: #fafafa; border: 1px solid #c0c0c0; padding: 6px;")
        sec_stats.addWidget(self.lbl_stats)

        self.table_preview = QTableWidget(6, 4)
        self.table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_preview.setMaximumHeight(170)
        sec_stats.addWidget(self.table_preview)
        vbox_left.addWidget(sec_stats)

        vbox_left.addStretch()

        panel_left.setMinimumWidth(380)
        panel_left.setMaximumWidth(460)
        scroll_left.setWidget(panel_left)
        splitter.addWidget(scroll_left)

        # Panel Derecho: Mapa 2D de Puntos y Tabla de Mediciones
        self.tabs_data_views = QTabWidget()

        # 1. Pestaña: Mapa 2D de Puntos y Delimitación de Malla
        tab_scatter = QWidget()
        v_sc = QVBoxLayout(tab_scatter)
        v_sc.setContentsMargins(0, 0, 0, 0)
        self.canvas_scatter = MplCanvas(self, width=8, height=6)
        self.toolbar_scatter = NavigationToolbar(self.canvas_scatter, self)
        self.toolbar_scatter.setStyleSheet("background-color: #f0f0f0; border: 1px solid #b0b0b0;")
        v_sc.addWidget(self.toolbar_scatter)
        v_sc.addWidget(self.canvas_scatter)
        self.tabs_data_views.addTab(tab_scatter, "Mapa 2D de Puntos & Delimitación")

        # 2. Pestaña: Tabla de Mediciones Crudas & Reducciones
        tab_tbl = QWidget()
        v_tbl = QVBoxLayout(tab_tbl)
        v_tbl.setContentsMargins(4, 4, 4, 4)
        self.table_raw_reductions = QTableWidget()
        self.table_raw_reductions.setAlternatingRowColors(True)
        v_tbl.addWidget(self.table_raw_reductions)
        self.tabs_data_views.addTab(tab_tbl, "Tabla de Mediciones & Reducciones")

        splitter.addWidget(self.tabs_data_views)

        # Alias de compatibilidad total
        self.canvas_zone = self.canvas_scatter
        self.toolbar_zone = self.toolbar_scatter

    def on_units_changed(self):
        if getattr(self, '_updating_units', False):
            return
        self._updating_units = True
        try:
            # 1. Resolver unidad XY
            txt_xy = self.combo_unit_xy.currentText().strip()
            data_xy = self.combo_unit_xy.currentData()
            norm_xy = txt_xy.lower()
            if any(k in norm_xy for k in ['metro', '[m]', 'metros']) and 'km' not in norm_xy and 'kilo' not in norm_xy:
                u_xy = "m"
            elif any(k in norm_xy for k in ['km', 'kilo']):
                u_xy = "km"
            elif any(k in norm_xy for k in ['grado', '°', 'deg']):
                u_xy = "°"
            elif any(k in norm_xy for k in ['pie', 'ft', 'feet']):
                u_xy = "ft"
            elif any(k in norm_xy for k in ['cent', 'cm']):
                u_xy = "cm"
            elif any(k in norm_xy for k in ['mili', 'mm']):
                u_xy = "mm"
            elif data_xy:
                u_xy = str(data_xy)
            else:
                u_xy = txt_xy or "m"

            # 2. Resolver unidad Z
            txt_z = self.combo_unit_z.currentText().strip()
            data_z = self.combo_unit_z.currentData()
            norm_z = txt_z.lower()
            if any(k in norm_z for k in ['metro', 'metros', 'cota', 'altitud', 'elevacion']) and 'km' not in norm_z and 'kilo' not in norm_z:
                u_z = "m"
            elif any(k in norm_z for k in ['km', 'kilo']):
                u_z = "km"
            elif 'micro' in norm_z or 'µgal' in norm_z or 'ugal' in norm_z:
                u_z = "µGal"
            elif 'mgal' in norm_z or 'grav' in norm_z:
                u_z = "mGal"
            elif 'nt' in norm_z or 'nano' in norm_z or 'mag' in norm_z:
                u_z = "nT"
            elif 'eötvös' in norm_z or 'eotvos' in norm_z:
                u_z = "Eötvös"
            elif data_z and txt_z == self.combo_unit_z.itemText(self.combo_unit_z.currentIndex()):
                u_z = str(data_z)
            else:
                if "(" in txt_z:
                    u_z = txt_z.split("(")[0].strip()
                elif "[" in txt_z:
                    u_z = txt_z.split("[")[0].strip()
                else:
                    u_z = txt_z or "mGal"

            self.unit_xy = str(u_xy) if u_xy else "m"
            self.unit_z = str(u_z) if u_z else "mGal"

            # 3. Actualizar textos de UI en vivo
            self.update_unit_labels_in_ui()

            # 4. Actualización instantánea ligera de ejes de gráficos sin re-cálculos masivos
            self._update_axes_units_lightweight()
        finally:
            self._updating_units = False

    def _update_axes_units_lightweight(self):
        u = self.unit_xy
        z = self.unit_z

        # Canvas Scatter (Pestaña 1)
        if hasattr(self, 'canvas_scatter') and self.canvas_scatter.fig.axes:
            axes = self.canvas_scatter.fig.axes
            if len(axes) > 0 and hasattr(self, 'x_col') and hasattr(self, 'y_col'):
                axes[0].set_xlabel(f"X ({self.x_col}) [{u}]", color='#000000', fontsize=9)
                axes[0].set_ylabel(f"Y ({self.y_col}) [{u}]", color='#000000', fontsize=9)
            if len(axes) > 1 and hasattr(self, 'z_col'):
                axes[1].set_ylabel(f"Z ({self.z_col}) [{z}]", color='#000000', fontsize=9)
            self.canvas_scatter.draw_idle()

        # Canvas Zona (Pestaña 2)
        if hasattr(self, 'canvas_zone') and self.canvas_zone.fig.axes:
            axes = self.canvas_zone.fig.axes
            if len(axes) > 0 and hasattr(self, 'x_col') and hasattr(self, 'y_col'):
                axes[0].set_xlabel(f"X ({self.x_col}) [{u}]", color='#000000', fontsize=9)
                axes[0].set_ylabel(f"Y ({self.y_col}) [{u}]", color='#000000', fontsize=9)
            self.canvas_zone.draw_idle()

        # Canvas Master Visualizer (Pestaña 3)
        if hasattr(self, 'canvas_view') and self.canvas_view.fig.axes:
            axes = self.canvas_view.fig.axes
            if len(axes) > 0 and hasattr(self, 'x_col') and hasattr(self, 'y_col'):
                axes[0].set_xlabel(f"X ({self.x_col}) [{u}]", color='#000000', fontsize=9)
                axes[0].set_ylabel(f"Y ({self.y_col}) [{u}]", color='#000000', fontsize=9)
            self.canvas_view.draw_idle()

        # Canvas Fourier (Pestaña 4)
        if hasattr(self, 'canvas_fourier') and self.canvas_fourier.fig.axes:
            axes = self.canvas_fourier.fig.axes
            for ax in axes:
                xl = ax.get_xlabel()
                yl = ax.get_ylabel()
                if "[" in xl:
                    base_xl = xl.split("[")[0].strip()
                    if "ciclos" in xl:
                        ax.set_xlabel(f"{base_xl} [ciclos/{u}]", color='#000000', fontsize=9)
                    else:
                        ax.set_xlabel(f"{base_xl} [{u}]", color='#000000', fontsize=9)
                if "[" in yl:
                    base_yl = yl.split("[")[0].strip()
                    if "ciclos" in yl:
                        ax.set_ylabel(f"{base_yl} [ciclos/{u}]", color='#000000', fontsize=9)
                    else:
                        ax.set_ylabel(f"{base_yl} [{u}]", color='#000000', fontsize=9)
            self.canvas_fourier.draw_idle()

        # Encabezados de tabla de exportación (Pestaña 5)
        if hasattr(self, 'table_mesh_nodes'):
            cz = getattr(self, 'z_col', 'Z')
            self.table_mesh_nodes.setHorizontalHeaderLabels([
                f"Coordenada X [{u}]", f"Coordenada Y [{u}]", f"Valor {cz} [{z}]"
            ])

    def update_unit_labels_in_ui(self):
        u = self.unit_xy
        z = self.unit_z

        # Pestaña 2: Delimitación y paso
        if hasattr(self, 'lbl_xmin'): self.lbl_xmin.setText(f"X Mín [{u}]:")
        if hasattr(self, 'lbl_xmax'): self.lbl_xmax.setText(f"X Máx [{u}]:")
        if hasattr(self, 'lbl_ymin'): self.lbl_ymin.setText(f"Y Mín [{u}]:")
        if hasattr(self, 'lbl_ymax'): self.lbl_ymax.setText(f"Y Máx [{u}]:")
        if hasattr(self, 'lbl_dx'): self.lbl_dx.setText(f"<b>deltaX (Paso X) [{u}]:</b>")
        if hasattr(self, 'lbl_dy'): self.lbl_dy.setText(f"<b>deltaY (Paso Y) [{u}]:</b>")

        # Pestaña 4: Fourier info y selector de dominio
        if hasattr(self, 'lbl_fourier_units_badge'):
            self.lbl_fourier_units_badge.setText(
                f"<b>Coherencia Física:</b> Espacio en [{u}] | Frecuencias u, v, k en [ciclos/{u}].<br>"
                f"Continuación Upward/Downward <b>exp(∓2·π·h·k)</b>: La altura <b>'h'</b> debe expresarse en <b>[{u}]</b>."
            )
        if hasattr(self, 'chk_show_wavelength'):
            self.chk_show_wavelength.setText(f"Mostrar Caracterización Espacial (λ en [{u}] en rojo)")

        self.update_grid_dimension_labels()

    # ==========================================================================
    # PESTAÑA 2: MODELADO (BLAKELY 1995)
    # ==========================================================================
    def build_tab_synth(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        self.widget_synth = VentanaModeladorSintetico(self, parent=parent, is_embedded=True)
        layout.addWidget(self.widget_synth)

    # ==========================================================================
    # PESTAÑA 3: VISUALIZADOR MAESTRO DE TODO (UNIVERSAL 2D VIEWER + MULTI-MONITOR)
    # ==========================================================================
    def build_tab_master_viewer(self, parent):
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        panel_left = QWidget()
        vbox_left = QVBoxLayout(panel_left)
        vbox_left.setContentsMargins(4, 4, 4, 4)
        vbox_left.setSpacing(8)

        # 1. Selector de Capa a Visualizar (15 capas completas)
        sec_layer = CollapsibleSection("1. Capa a Visualizar (Visualizador de Todo)", is_expanded=True)
        sec_layer.addWidget(QLabel("<b>Selecciona la Capa / Espectro / Derivada:</b>"))
        self.combo_master_layer = QComboBox()
        self.combo_master_layer.addItem("Malla Original Z_orig(x,y)", "grid_orig")
        self.combo_master_layer.addItem("Malla Acondicionada (Taper Bordes Fourier)", "grid_taper")
        self.combo_master_layer.addItem("Malla Filtrada Fourier Z_filt(x,y)", "grid_filt")
        self.combo_master_layer.addItem("Diferencia Residual (Z_orig - Z_filt)", "grid_diff")
        self.combo_master_layer.addItem("Gradiente Horizontal Total (THDR)", "thdr")
        self.combo_master_layer.addItem("Primera Derivada Vertical Espacial (1VD)", "vd1")
        self.combo_master_layer.addItem("Espectro 2D Amplitud Log10 Original |F(u,v)|", "spec_orig")
        self.combo_master_layer.addItem("Espectro 2D Amplitud Log10 Filtrado |F_filt(u,v)|", "spec_filt")
        self.combo_master_layer.addItem("Filtro Aplicado Total |H_total(u,v)|", "filter_total")
        self.combo_master_layer.addItem("Espectro - Parte Real Re(F)", "spec_real")
        self.combo_master_layer.addItem("Espectro - Parte Imaginaria Im(F)", "spec_imag")
        self.combo_master_layer.addItem("Espectro de Fase ∠F(u,v) [rad]", "spec_phase")
        self.combo_master_layer.addItem("Varianza de Interpolación (Kriging)", "krig_var")
        self.combo_master_layer.addItem("Matriz de Pesos P (Inverso Distancia) [1/m]", "weights_p")
        self.combo_master_layer.addItem("Topografía / Elevación MDT [m]", "mdt")
        self.combo_master_layer.addItem("Gravedad Fuente Equivalente a Z0", "eq_source_z0")
        self.combo_master_layer.addItem("Gravedad Fuente Equivalente Filtrada", "eq_source_z0_filt")
        self.combo_master_layer.addItem("Nube de Puntos Crudos (Scatter Irregular)", "scatter_raw")
        self.combo_master_layer.currentIndexChanged.connect(self.plot_master_viewer)
        sec_layer.addWidget(self.combo_master_layer)

        btn_inv_shortcut = QPushButton("Abrir Problema Inverso (Fuente Equivalente Z0)...")
        btn_inv_shortcut.setStyleSheet("background-color: #990000; color: #ffffff; font-weight: bold; padding: 4px; margin-top: 2px;")
        btn_inv_shortcut.clicked.connect(self.open_fuente_equivalente_window)
        sec_layer.addWidget(btn_inv_shortcut)

        # Ficha estadística de la capa
        self.lbl_master_stat = QLabel("Sin datos generados aún.")
        self.lbl_master_stat.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; background-color: #fafafa; border: 1px solid #c0c0c0; padding: 5px;")
        sec_layer.addWidget(self.lbl_master_stat)
        vbox_left.addWidget(sec_layer)

        # 2. Renderizado, Paleta y Contornos
        sec_render = CollapsibleSection("2. Paleta de Colores & Superposiciones", is_expanded=True)
        grid_r = QGridLayout()
        grid_r.addWidget(QLabel("Paleta de Color:"), 0, 0)
        self.combo_master_cmap = QComboBox()
        self.combo_master_cmap.addItems(["jet", "viridis", "plasma", "inferno", "magma", "cividis", "turbo", "terrain", "seismic", "coolwarm", "rainbow", "gray"])
        self.combo_master_cmap.currentIndexChanged.connect(self.plot_master_viewer)
        grid_r.addWidget(self.combo_master_cmap, 0, 1)

        self.chk_master_cmap_r = QCheckBox("Invertir paleta (_r)")
        self.chk_master_cmap_r.stateChanged.connect(self.plot_master_viewer)
        grid_r.addWidget(self.chk_master_cmap_r, 1, 0, 1, 2)
        sec_render.addLayout(grid_r)

        # Curvas de nivel
        row_cnt = QHBoxLayout()
        self.chk_master_contours = QCheckBox("Superponer Contornos (Curvas de Nivel)")
        self.chk_master_contours.setChecked(False)
        self.chk_master_contours.stateChanged.connect(self.plot_master_viewer)
        row_cnt.addWidget(self.chk_master_contours)

        self.spin_master_contours_n = QSpinBox()
        self.spin_master_contours_n.setRange(5, 80); self.spin_master_contours_n.setValue(15)
        self.spin_master_contours_n.valueChanged.connect(self.plot_master_viewer)
        row_cnt.addWidget(self.spin_master_contours_n)
        sec_render.addLayout(row_cnt)

        self.chk_master_clabels = QCheckBox("Etiquetar valores en curvas de nivel")
        self.chk_master_clabels.setChecked(True)
        self.chk_master_clabels.stateChanged.connect(self.plot_master_viewer)
        sec_render.addWidget(self.chk_master_clabels)

        self.chk_master_overlay_pts = QCheckBox("Superponer puntos dentro de la malla")
        self.chk_master_overlay_pts.setToolTip("Muestra únicamente los puntos de observación situados dentro de la extensión geográfica donde se generó la malla.")
        self.chk_master_overlay_pts.setChecked(False)
        self.chk_master_overlay_pts.stateChanged.connect(self.plot_master_viewer)
        sec_render.addWidget(self.chk_master_overlay_pts)
        vbox_left.addWidget(sec_render)

        # 3. Escala y Rango de Visualización
        sec_scale = CollapsibleSection("3. Escala y Rango de Visualización", is_expanded=True)
        grid_sc = QGridLayout()
        grid_sc.addWidget(QLabel("Modo de Escala:"), 0, 0)
        self.combo_master_scale_mode = QComboBox()
        self.combo_master_scale_mode.addItem("Automático Completo (Mín - Máx)", "auto")
        self.combo_master_scale_mode.addItem("Percentiles Robustos (2% - 98%)", "robust")
        self.combo_master_scale_mode.addItem("Percentiles Estrictos (1% - 99%)", "strict")
        self.combo_master_scale_mode.addItem("Manual (Z Mín / Z Máx fijos)", "manual")
        self.combo_master_scale_mode.currentIndexChanged.connect(self.plot_master_viewer)
        grid_sc.addWidget(self.combo_master_scale_mode, 0, 1)

        grid_sc.addWidget(QLabel("Z Mín / Escala Inf:"), 1, 0)
        self.spin_master_vmin = QDoubleSpinBox()
        self.spin_master_vmin.setRange(-1e12, 1e12); self.spin_master_vmin.setDecimals(4)
        self.spin_master_vmin.valueChanged.connect(self.plot_master_viewer)
        grid_sc.addWidget(self.spin_master_vmin, 1, 1)

        grid_sc.addWidget(QLabel("Z Máx / Escala Sup:"), 2, 0)
        self.spin_master_vmax = QDoubleSpinBox()
        self.spin_master_vmax.setRange(-1e12, 1e12); self.spin_master_vmax.setDecimals(4)
        self.spin_master_vmax.valueChanged.connect(self.plot_master_viewer)
        grid_sc.addWidget(self.spin_master_vmax, 2, 1)
        sec_scale.addLayout(grid_sc)

        btn_reset_scale = QPushButton("Restablecer Rango Automático")
        btn_reset_scale.setObjectName("btnMini")
        btn_reset_scale.clicked.connect(self.reset_master_scale)
        sec_scale.addWidget(btn_reset_scale)
        vbox_left.addWidget(sec_scale)

        # 4. Ventana y Exportación
        sec_win = CollapsibleSection("4. Modo de Ventana & Exportación Gráfica", is_expanded=True)
        btn_detach_tab = QPushButton("Desacoplar a Ventana Flotante (Multi-Monitor)")
        btn_detach_tab.setObjectName("btnReset")
        btn_detach_tab.clicked.connect(self.detach_viewer)
        sec_win.addWidget(btn_detach_tab)

        btn_exp_img = QPushButton("Guardar Imagen de Alta Resolución (300 DPI)")
        btn_exp_img.setObjectName("btnPrimary")
        btn_exp_img.clicked.connect(self.export_master_image)
        sec_win.addWidget(btn_exp_img)
        vbox_left.addWidget(sec_win)
        vbox_left.addStretch()

        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left.setFrameShape(QFrame.NoFrame)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_left.setWidget(panel_left)
        scroll_left.setMinimumWidth(380)
        scroll_left.setMaximumWidth(480)
        splitter.addWidget(scroll_left)

        # Canvas interactivo derecho del Visualizador Maestro
        panel_right = QWidget()
        vbox_right = QVBoxLayout(panel_right)
        vbox_right.setContentsMargins(0, 0, 0, 0)
        self.canvas_view = MplCanvas(self, width=8, height=8)
        row_tool_master = QHBoxLayout()
        self.toolbar_view = NavigationToolbar(self.canvas_view, self)
        self.toolbar_view.setStyleSheet("background-color: #f0f0f0; border: 1px solid #b0b0b0;")
        row_tool_master.addWidget(self.toolbar_view)

        self.chk_master_aspect_equal = QCheckBox("Escala Isótropa 1:1 (X=Y)")
        self.chk_master_aspect_equal.setChecked(True)
        self.chk_master_aspect_equal.setToolTip("Mantiene proporción física 1:1 en metros sin distorsionar formas ni agregar marcos vacíos. Desmárcalo para expandir y llenar la pantalla con aspect='auto'.")
        self.chk_master_aspect_equal.stateChanged.connect(self.plot_master_viewer)
        row_tool_master.addWidget(self.chk_master_aspect_equal)
        row_tool_master.addStretch()

        vbox_right.addLayout(row_tool_master)
        vbox_right.addWidget(self.canvas_view)

        # Lectura bajo el cursor en vivo
        self.lbl_master_cursor = QLabel("Cursor: Mueve el ratón sobre el mapa para inspeccionar coordenadas y valores.")
        self.lbl_master_cursor.setStyleSheet("background-color: #fafafa; border: 1px solid #b0b0b0; padding: 4px; font-weight: bold; font-size: 11px;")
        vbox_right.addWidget(self.lbl_master_cursor)
        splitter.addWidget(panel_right)

        # Conectar movimiento de ratón
        self.canvas_view.mpl_connect('motion_notify_event', self.on_master_mouse_move)

    def reset_master_scale(self):
        self.combo_master_scale_mode.setCurrentIndex(0)
        self.plot_master_viewer()

    def detach_viewer(self):
        if "master_viewer" in self.detached_windows:
            self.detached_windows["master_viewer"].activateWindow()
            self.detached_windows["master_viewer"].raise_()
            return

        idx = self.tabs.indexOf(self.tab_view)
        if idx >= 0:
            self.tabs.removeTab(idx)
            sub_win = DetachedSubWindow(
                key="master_viewer",
                title="Visualizador Maestro de Todo",
                widget=self.tab_view,
                main_window=self,
                original_index=2
            )
            self.detached_windows["master_viewer"] = sub_win
            sub_win.show()

    def export_master_image(self):
        if self.master_active_data is None and self.grid_Z is None:
            QMessageBox.warning(self, "Sin Gráfico", "Aún no hay capas generadas para exportar.")
            return

        layer_tag = self.combo_master_layer.currentData()
        fpath, _ = QFileDialog.getSaveFileName(
            self, "Exportar Imagen de Alta Resolución",
            os.path.join(DESCARGAS_DIR, f"visualizador_{layer_tag}_300dpi.png"),
            "Imagen PNG (*.png);;Imagen PDF (*.pdf);;Vectorial SVG (*.svg);;Imagen JPEG (*.jpg)"
        )
        if fpath:
            self.canvas_view.fig.savefig(fpath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Imagen Exportada", f"Gráfico exportado con éxito a 300 DPI en:\n{fpath}")

    def on_master_mouse_move(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        x = event.xdata
        y = event.ydata
        u = self.unit_xy
        z = self.unit_z

        if self.master_active_is_freq:
            kr = np.hypot(x, y)
            val_str = ""
            if self.master_active_data is not None and self.u_vec is not None and self.v_vec is not None:
                ix = int(np.clip(np.searchsorted(self.u_vec, x), 0, len(self.u_vec)-1))
                iy = int(np.clip(np.searchsorted(self.v_vec, y), 0, len(self.v_vec)-1))
                val = self.master_active_data[iy, ix]
                val_str = f"  |  Valor: {val:.4g} [{self.master_active_unit}]"
            self.lbl_master_cursor.setText(
                f"Frecuencia: U = {x:.5g} [ciclos/{u}]  |  V = {y:.5g} [ciclos/{u}]  |  k_r = {kr:.5g} [ciclos/{u}]{val_str}"
            )
        else:
            val_str = ""
            if self.master_active_data is not None and self.grid_x_vec is not None and self.grid_y_vec is not None:
                ix = int(np.clip(np.searchsorted(self.grid_x_vec, x), 0, len(self.grid_x_vec)-1))
                iy = int(np.clip(np.searchsorted(self.grid_y_vec, y), 0, len(self.grid_y_vec)-1))
                val = self.master_active_data[iy, ix]
                val_str = f"  |  Valor {self.z_col}: {val:.4g} [{self.master_active_unit}]"
            self.lbl_master_cursor.setText(
                f"Coordenadas: X = {x:.3f} [{u}]  |  Y = {y:.3f} [{u}]{val_str}"
            )

    # ==========================================================================
    # PESTAÑA 4: FILTROS FOURIER 2D (FÓRMULAS MATEMÁTICAS & VENTANAS)
    # ==========================================================================
    def build_tab_fourier(self, parent):
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Panel Izquierdo: Gestión de Filtros
        panel_left = QWidget()
        vbox_left = QVBoxLayout(panel_left)
        vbox_left.setContentsMargins(4, 4, 4, 4)
        vbox_left.setSpacing(8)

        # 1. Parámetros Base Fourier & Coherencia Física
        sec_fopt = CollapsibleSection("1. Parámetros Fourier 2D & Unidades Físicas", is_expanded=True)
        self.lbl_fourier_units_badge = QLabel(
            f"<b>Coherencia Física:</b> Espacio en [{self.unit_xy}] | Frecuencias en [ciclos/{self.unit_xy}].<br>"
            f"Continuación Upward/Downward <b>exp(∓2·π·h·k)</b>: La altura <b>'h'</b> debe expresarse en <b>[{self.unit_xy}]</b>."
        )
        self.lbl_fourier_units_badge.setStyleSheet("background-color: #fafafa; border: 1px solid #c0c0c0; padding: 6px; font-size: 11px;")
        sec_fopt.addWidget(self.lbl_fourier_units_badge)

        self.chk_fourier_omit_dc = QCheckBox("Atenuar componente continua central (DC)")
        self.chk_fourier_omit_dc.setChecked(True)
        self.chk_fourier_omit_dc.stateChanged.connect(self.plot_fourier_dashboard)
        sec_fopt.addWidget(self.chk_fourier_omit_dc)

        self.chk_fourier_sub_mean = QCheckBox("Restar media espacial previa")
        self.chk_fourier_sub_mean.setChecked(True)
        self.chk_fourier_sub_mean.stateChanged.connect(self.recompute_fourier_2d)
        sec_fopt.addWidget(self.chk_fourier_sub_mean)

        # --- ACONDICIONAMIENTO DE BORDES 2D (TAPER PERIMETRAL PREVIO A FFT) ---
        grp_taper = QGroupBox("Acondicionamiento de Bordes (Tapering Previo a FFT)")
        grp_taper_l = QVBoxLayout(grp_taper)
        grp_taper_l.setSpacing(5)

        self.chk_fourier_taper = QCheckBox("Llevar extremos suavemente a valor de referencia")
        self.chk_fourier_taper.setChecked(True)
        self.chk_fourier_taper.stateChanged.connect(self.recompute_fourier_2d)
        grp_taper_l.addWidget(self.chk_fourier_taper)

        row_t_target = QHBoxLayout()
        row_t_target.addWidget(QLabel("<b>Extremos a:</b>"))
        self.combo_fourier_taper_target = QComboBox()
        self.combo_fourier_taper_target.addItem("Valor Medio (Media de la malla)", "mean")
        self.combo_fourier_taper_target.addItem("Valor Mínimo de la malla", "min")
        self.combo_fourier_taper_target.addItem("⭕ Cero Estricto (0.0)", "zero")
        self.combo_fourier_taper_target.currentIndexChanged.connect(self.recompute_fourier_2d)
        row_t_target.addWidget(self.combo_fourier_taper_target)
        grp_taper_l.addLayout(row_t_target)

        row_t_type = QHBoxLayout()
        row_t_type.addWidget(QLabel("<b>Transición:</b>"))
        self.combo_fourier_taper_type = QComboBox()
        self.combo_fourier_taper_type.addItem("Coseno Suave (Tukey / Hann)", "cosine")
        self.combo_fourier_taper_type.addItem("Sinusoidal (Hermite S-curve)", "sinu")
        self.combo_fourier_taper_type.addItem("Rampa Lineal", "linear")
        self.combo_fourier_taper_type.currentIndexChanged.connect(self.recompute_fourier_2d)
        row_t_type.addWidget(self.combo_fourier_taper_type)
        grp_taper_l.addLayout(row_t_type)

        row_t_pct = QHBoxLayout()
        row_t_pct.addWidget(QLabel("<b>Margen:</b>"))
        self.spin_fourier_taper_pct = QDoubleSpinBox()
        self.spin_fourier_taper_pct.setRange(1.0, 45.0)
        self.spin_fourier_taper_pct.setSingleStep(1.0)
        self.spin_fourier_taper_pct.setValue(10.0)
        self.spin_fourier_taper_pct.setSuffix(" % de borde")
        self.spin_fourier_taper_pct.valueChanged.connect(self.recompute_fourier_2d)
        row_t_pct.addWidget(self.spin_fourier_taper_pct)
        grp_taper_l.addLayout(row_t_pct)

        sec_fopt.addWidget(grp_taper)

        btn_recalc_f2d = QPushButton("Recalcular FFT 2D Base")
        btn_recalc_f2d.setObjectName("btnReset")
        btn_recalc_f2d.clicked.connect(self.recompute_fourier_2d)
        sec_fopt.addWidget(btn_recalc_f2d)
        vbox_left.addWidget(sec_fopt)

        # 2. Pila de Filtros Superpuestos (Inicia completamente vacía)
        sec_filters = CollapsibleSection("2. Pila de Filtros Fourier 2D", is_expanded=True)

        row_f_btns = QHBoxLayout()
        btn_add_f = QPushButton("Agregar Filtro...")
        btn_add_f.setObjectName("btnPrimary")
        btn_add_f.clicked.connect(self.on_add_filter_clicked)
        row_f_btns.addWidget(btn_add_f)

        btn_presets = QPushButton("Filtros Clásicos (1 Ejemplo) ▼")
        btn_presets.setObjectName("btnReset")
        btn_presets.clicked.connect(self.show_presets_menu)
        row_f_btns.addWidget(btn_presets)
        sec_filters.addLayout(row_f_btns)

        # Tabla de filtros
        self.tbl_filters = QTableWidget(0, 3)
        self.tbl_filters.setHorizontalHeaderLabels(["Activo", "Nombre del Filtro", "Fórmula / Ventana"])
        self.tbl_filters.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_filters.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_filters.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_filters.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_filters.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_filters.customContextMenuRequested.connect(self.on_filter_table_context_menu)
        self.tbl_filters.itemDoubleClicked.connect(self.on_filter_double_clicked)
        self.tbl_filters.itemChanged.connect(self.on_filter_table_item_changed)
        self.tbl_filters.setMaximumHeight(190)
        sec_filters.addWidget(self.tbl_filters)

        # Botones de gestión de filas
        row_row_btns = QHBoxLayout()
        btn_edit_f = QPushButton("Editar")
        btn_edit_f.setObjectName("btnMini")
        btn_edit_f.clicked.connect(self.edit_selected_filter)
        row_row_btns.addWidget(btn_edit_f)

        btn_dup_f = QPushButton("Duplicar")
        btn_dup_f.setObjectName("btnMini")
        btn_dup_f.clicked.connect(self.duplicate_selected_filter)
        row_row_btns.addWidget(btn_dup_f)

        btn_del_f = QPushButton("Eliminar")
        btn_del_f.setObjectName("btnMini")
        btn_del_f.clicked.connect(self.delete_selected_filter)
        row_row_btns.addWidget(btn_del_f)

        btn_up_f = QPushButton("Subir")
        btn_up_f.setObjectName("btnMini")
        btn_up_f.setFixedWidth(28)
        btn_up_f.clicked.connect(self.move_selected_filter_up)
        row_row_btns.addWidget(btn_up_f)

        btn_down_f = QPushButton("Bajar")
        btn_down_f.setObjectName("btnMini")
        btn_down_f.setFixedWidth(28)
        btn_down_f.clicked.connect(self.move_selected_filter_down)
        row_row_btns.addWidget(btn_down_f)

        btn_clear_f = QPushButton("Limpiar")
        btn_clear_f.setObjectName("btnMini")
        btn_clear_f.clicked.connect(self.clear_all_filters)
        row_row_btns.addWidget(btn_clear_f)
        sec_filters.addLayout(row_row_btns)

        self.chk_cascade_mult = QCheckBox("Multiplicar filtros en cascada (H_total = ∏ H_i)")
        self.chk_cascade_mult.setChecked(True)
        self.chk_cascade_mult.stateChanged.connect(self.plot_fourier_dashboard)
        sec_filters.addWidget(self.chk_cascade_mult)
        vbox_left.addWidget(sec_filters)

        # 3. Opciones de Visualización 2D en Fourier
        sec_fview = CollapsibleSection("3. Opciones de Visualización en Fourier 2D", is_expanded=True)
        grid_fv = QGridLayout()
        grid_fv.addWidget(QLabel("Capa Espacial Der:"), 0, 0)
        self.combo_fourier_map_layer = QComboBox()
        self.combo_fourier_map_layer.addItem("Malla Filtrada (Z_filt)", "filt")
        self.combo_fourier_map_layer.addItem("Malla Original (Z_orig)", "orig")
        self.combo_fourier_map_layer.addItem("Malla con Taper Bordes (Z_taper)", "taper")
        self.combo_fourier_map_layer.addItem("Diferencia (Z_orig - Z_filt)", "diff")
        self.combo_fourier_map_layer.currentIndexChanged.connect(self.plot_fourier_dashboard)
        grid_fv.addWidget(self.combo_fourier_map_layer, 0, 1)

        grid_fv.addWidget(QLabel("Gráfico Espectral Der:"), 1, 0)
        self.combo_spec_plot_mode = QComboBox()
        self.combo_spec_plot_mode.addItem("RAPS Comparativo", "raps")
        self.combo_spec_plot_mode.addItem("Espectro 2D Log10 Filtrado", "spec2d")
        self.combo_spec_plot_mode.currentIndexChanged.connect(self.plot_fourier_dashboard)
        grid_fv.addWidget(self.combo_spec_plot_mode, 1, 1)

        grid_fv.addWidget(QLabel("Escala RAPS Potencia:"), 2, 0)
        self.combo_raps_scale = QComboBox()
        self.combo_raps_scale.addItem("Logarítmica: ln(P)", "ln")
        self.combo_raps_scale.addItem("Lineal: P(k)", "linear")
        self.combo_raps_scale.addItem("Ganancia Espectral: |H(k)|² = P_filt / P_orig", "ratio")
        self.combo_raps_scale.currentIndexChanged.connect(self.plot_fourier_dashboard)
        grid_fv.addWidget(self.combo_raps_scale, 2, 1)
        sec_fview.addLayout(grid_fv)

        row_f_exp = QHBoxLayout()
        btn_exp_filt_grid = QPushButton("Exportar Malla Filtrada...")
        btn_exp_filt_grid.setObjectName("btnPrimary")
        btn_exp_filt_grid.clicked.connect(self.export_filtered_grid_quick)
        row_f_exp.addWidget(btn_exp_filt_grid)
        sec_fview.addLayout(row_f_exp)
        vbox_left.addWidget(sec_fview)

        # 4. Modelado Sintético y Comparación de Modelos (Blakely 1995)
        sec_model_ops = CollapsibleSection("4. Modelado y Comparación (Blakely 1995)", is_expanded=True)
        btn_tab4_synth = QPushButton("Modelador de Anomalías Sintéticas...")
        btn_tab4_synth.setToolTip("Diseñar cuerpos geológicos y anomalías sintéticas según Blakely 1995")
        btn_tab4_synth.clicked.connect(self.open_synthetic_modeler)
        sec_model_ops.addWidget(btn_tab4_synth)

        btn_tab4_comp = QPushButton("Comparador Obs vs Modelo (Álgebra)...")
        btn_tab4_comp.setObjectName("btnAccent")
        btn_tab4_comp.setToolTip("Contrastar las observaciones filtradas con el modelo y evaluar f(Obs, Mod) píxel a píxel")
        btn_tab4_comp.clicked.connect(self.open_model_comparison)
        sec_model_ops.addWidget(btn_tab4_comp)
        vbox_left.addWidget(sec_model_ops)

        vbox_left.addStretch()

        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left.setFrameShape(QFrame.NoFrame)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_left.setWidget(panel_left)
        scroll_left.setMinimumWidth(380)
        scroll_left.setMaximumWidth(480)
        splitter.addWidget(scroll_left)

        # Panel Derecho: 4 Cuadrantes Fourier 2D con Unidades Explícitas
        panel_right = QWidget()
        vbox_right = QVBoxLayout(panel_right)
        vbox_right.setContentsMargins(0, 0, 0, 0)

        # Barra superior con toolbar y controles de visualización física
        row_top_fourier = QHBoxLayout()
        self.canvas_fourier = MplCanvas(self, width=8, height=8)
        self.toolbar_fourier = NavigationToolbar(self.canvas_fourier, self)
        self.toolbar_fourier.setStyleSheet("background-color: #f0f0f0; border: 1px solid #b0b0b0;")
        row_top_fourier.addWidget(self.toolbar_fourier)
        row_top_fourier.addStretch()

        self.chk_fourier_aspect_equal = QCheckBox("Escala Isótropa 1:1 (X=Y)")
        self.chk_fourier_aspect_equal.setChecked(True)
        self.chk_fourier_aspect_equal.setToolTip("Mantiene proporción física 1:1 sin distorsionar formas. Desmárcalo para mallas alargadas que llenen todo el espacio (aspect='auto').")
        self.chk_fourier_aspect_equal.stateChanged.connect(self.plot_fourier_dashboard)
        row_top_fourier.addWidget(self.chk_fourier_aspect_equal)

        self.chk_show_wavelength = QCheckBox(f"Mostrar Caracterización Espacial (λ [{self.unit_xy}] en rojo)")
        self.chk_show_wavelength.setChecked(True)
        self.chk_show_wavelength.setToolTip("Muestra debajo de cada frecuencia anotada su longitud de onda λ = 1/f en rojo, sin alterar los ejes naturales de frecuencia.")
        self.chk_show_wavelength.stateChanged.connect(self.plot_fourier_dashboard)
        row_top_fourier.addWidget(self.chk_show_wavelength)

        vbox_right.addLayout(row_top_fourier)
        vbox_right.addWidget(self.canvas_fourier)
        splitter.addWidget(panel_right)

        self.refresh_filter_table()

    # ==========================================================================
    # PESTAÑA 5: TABLA DE MALLA Y EXPORTACIÓN
    # ==========================================================================
    def build_tab_table(self, parent):
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        panel_left = QWidget()
        vbox_left = QVBoxLayout(panel_left)
        vbox_left.setContentsMargins(4, 4, 4, 4)
        vbox_left.setSpacing(8)

        sec_src = CollapsibleSection("1. Fuente de Datos para Exportación", is_expanded=True)
        self.rad_exp_orig = QRadioButton("Malla Original Interpolada")
        self.rad_exp_taper = QRadioButton("Malla Acondicionada (Taper Bordes)")
        self.rad_exp_filt = QRadioButton("Malla Filtrada (Fourier 2D)")
        self.rad_exp_filt.setChecked(True)
        sec_src.addWidget(self.rad_exp_orig)
        sec_src.addWidget(self.rad_exp_taper)
        sec_src.addWidget(self.rad_exp_filt)
        self.rad_exp_orig.toggled.connect(self.update_mesh_table)
        self.rad_exp_taper.toggled.connect(self.update_mesh_table)
        vbox_left.addWidget(sec_src)

        sec_exp = CollapsibleSection("2. Formatos de Exportación", is_expanded=True)
        btn_csv_xyz = QPushButton("Exportar CSV XYZ (X, Y, Z)")
        btn_csv_xyz.clicked.connect(self.export_csv_xyz)
        sec_exp.addWidget(btn_csv_xyz)

        btn_csv_mat = QPushButton("Exportar Matriz 2D CSV (Ny × Nx)")
        btn_csv_mat.clicked.connect(self.export_csv_matrix)
        sec_exp.addWidget(btn_csv_mat)

        btn_blakely = QPushButton("Exportar CSV Fourier Blakely (U, V, Real, Imag)")
        btn_blakely.clicked.connect(self.export_csv_blakely)
        sec_exp.addWidget(btn_blakely)

        btn_surfer = QPushButton("Exportar Surfer ASCII Grid (.grd)")
        btn_surfer.clicked.connect(self.export_surfer_grid)
        sec_exp.addWidget(btn_surfer)

        btn_tiff = QPushButton("Exportar Raster TIFF (.tif)")
        btn_tiff.clicked.connect(self.export_tiff_grid)
        sec_exp.addWidget(btn_tiff)

        btn_json = QPushButton("Exportar JSON Estructurado")
        btn_json.clicked.connect(self.export_json_grid)
        sec_exp.addWidget(btn_json)
        vbox_left.addWidget(sec_exp)
        vbox_left.addStretch()

        panel_left.setMinimumWidth(320)
        panel_left.setMaximumWidth(400)
        splitter.addWidget(panel_left)

        panel_right = QWidget()
        vbox_right = QVBoxLayout(panel_right)
        vbox_right.setContentsMargins(0, 0, 0, 0)
        self.lbl_table_info = QLabel("Mostrando primeros nodos de la malla generada:")
        self.lbl_table_info.setStyleSheet("font-weight: bold; padding: 4px;")
        vbox_right.addWidget(self.lbl_table_info)

        self.table_mesh_nodes = QTableWidget(100, 3)
        self.table_mesh_nodes.setHorizontalHeaderLabels([f"Coordenada X [{self.unit_xy}]", f"Coordenada Y [{self.unit_xy}]", f"Valor Z [{self.unit_z}]"])
        self.table_mesh_nodes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        vbox_right.addWidget(self.table_mesh_nodes)
        splitter.addWidget(panel_right)

    def activate_raw_reduction_mode(self):
        """Activa la modalidad de observaciones crudas y el reductor integrado."""
        self.combo_obs_mode.setCurrentIndex(1)
        self.tabs.setCurrentIndex(0)
        if hasattr(self, 'tabs_data_views'):
            self.tabs_data_views.setCurrentIndex(0)

    def on_obs_mode_changed(self):
        """Cambia entre Observaciones Corregidas y Observaciones Crudas."""
        mode = self.combo_obs_mode.currentData()
        is_crudas = (mode == "crudas")
        self.box_obs_corregidas.setVisible(not is_crudas)
        self.box_obs_crudas.setVisible(is_crudas)
        if hasattr(self, 'tabs_data_views'):
            self.tabs_data_views.setCurrentIndex(0)
        if is_crudas:
            self.status.showMessage("Modalidad: Observaciones Crudas (Asignación de Campos -> Somigliana -> Aire Libre -> Bouguer -> Topografía)")
        else:
            self.status.showMessage("Modalidad: Observaciones Corregidas (Anomalía Directa)")

    def update_raw_bouguer_factor_label(self):
        rho = self.spin_raw_rho_b.value()
        factor = 2.0 * np.pi * GAMMA_MGAL * rho
        self.lbl_raw_factor_b.setText(f"Factor: 2·π·G·ρ ≈ {factor:.4f} mGal/m")

    def on_raw_ellipsoid_changed(self):
        """Actualiza la fórmula de Somigliana al cambiar de elipsoide de referencia."""
        model_key = self.combo_ellipsoid.currentData()
        info = ELLIPSOID_MODELS.get(model_key, ELLIPSOID_MODELS['WGS84_Somigliana'])
        k_val = info.get('k', 0.00193185)
        e2_val = info.get('e2', 0.00669438)
        ga_val = info.get('gamma_a', 9.780327)
        self.lbl_formula_g0.setText(
            f"<code>g₀(λ) = {ga_val:.7f} · (1 + {k_val:.8f}·sin²λ) / √(1 - {e2_val:.8f}·sin²λ)</code>"
        )
        if self.raw_grav_df is not None and 'lat' in self.raw_grav_df.columns:
            g0 = compute_normal_gravity(self.raw_grav_df['lat'], model_key=model_key)
            self.raw_grav_df['g0'] = g0
            self.raw_grav_df['anom_g0'] = self.raw_grav_df['G'] - g0
            self.update_raw_table()
            if self.step_fa_done:
                self.run_raw_step_free_air()
            elif self.step_sb_done:
                self.run_raw_step_bouguer()

    def on_load_raw_measurements_file(self):
        """Carga archivo de tabla de mediciones crudas y puebla los desplegables de campos."""
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Cargar Tabla de Mediciones Crudas", BASE_DIR,
            "Archivos Soportados (*.csv *.xlsx *.xls *.txt *.dat *.tsv *.json);;CSV (*.csv);;Excel (*.xlsx *.xls);;Todos (*.*)"
        )
        if not fpath:
            return
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fpath)[1].lower()
        try:
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(fpath)
            elif ext == '.json':
                try:
                    df = pd.read_json(fpath)
                except ValueError:
                    df = pd.read_json(fpath, lines=True)
            else:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    sample = [f.readline() for _ in range(12)]
                sample_text = "".join(sample)
                counts = {',': sample_text.count(','), ';': sample_text.count(';'), '\t': sample_text.count('\t'), ' ': sample_text.count(' ')}
                detected_sep = max(counts, key=counts.get)
                if detected_sep == ' ' and counts[' '] > 0:
                    try:
                        df = pd.read_csv(fpath, sep=r'\s+', engine='python')
                    except Exception:
                        df = pd.read_csv(fpath, sep=detected_sep)
                else:
                    df = pd.read_csv(fpath, sep=detected_sep)

            cols = [str(c).strip() for c in df.columns]
            df.columns = cols
            self.df_raw_source = df

            # Llenar comboboxes de asignación de campos
            for cb in [self.combo_raw_col_x, self.combo_raw_col_y, self.combo_raw_col_h,
                       self.combo_raw_col_lat, self.combo_raw_col_lon, self.combo_raw_col_g]:
                cb.blockSignals(True)
                cb.clear()
                cb.addItems(cols)
                cb.blockSignals(False)

            def match_col(names, priority_words):
                for pw in priority_words:
                    for i, name in enumerate(names):
                        nl = name.lower()
                        if nl == pw or nl.startswith(pw):
                            return i
                return -1

            idx_x = match_col(cols, ['x', 'este', 'easting', 'lon', 'longitude'])
            idx_y = match_col(cols, ['y', 'norte', 'northing', 'lat', 'latitude'])
            idx_h = match_col(cols, ['h', 'cota', 'alt', 'altura', 'elev', 'elevation', 'hs'])
            idx_lat = match_col(cols, ['lat', 'latitude', 'latitud', 'y'])
            idx_lon = match_col(cols, ['lon', 'long', 'longitud', 'longitude', 'x'])
            idx_g = match_col(cols, ['g_obs', 'g', 'gravedad', 'grav', 'obs', 'gravity'])

            if idx_x >= 0: self.combo_raw_col_x.setCurrentIndex(idx_x)
            elif len(cols) > 0: self.combo_raw_col_x.setCurrentIndex(0)

            if idx_y >= 0: self.combo_raw_col_y.setCurrentIndex(idx_y)
            elif len(cols) > 1: self.combo_raw_col_y.setCurrentIndex(1)

            if idx_h >= 0: self.combo_raw_col_h.setCurrentIndex(idx_h)
            elif len(cols) > 2: self.combo_raw_col_h.setCurrentIndex(2)

            if idx_lat >= 0: self.combo_raw_col_lat.setCurrentIndex(idx_lat)
            elif idx_y >= 0: self.combo_raw_col_lat.setCurrentIndex(idx_y)

            if idx_lon >= 0: self.combo_raw_col_lon.setCurrentIndex(idx_lon)
            elif idx_x >= 0: self.combo_raw_col_lon.setCurrentIndex(idx_x)

            if idx_g >= 0: self.combo_raw_col_g.setCurrentIndex(idx_g)
            elif len(cols) > 3: self.combo_raw_col_g.setCurrentIndex(3)

            self.lbl_raw_file_info.setText(f"{fname} ({len(df):,} filas, {len(cols)} columnas)")
            self.lbl_raw_file_info.setStyleSheet("color: #006600; font-weight: bold; font-size: 10px;")

            # Mostrar tabla previa en la sección 5
            self.df_raw = df
            self.populate_preview_table()

            # Aplicar asignación inicial automáticamente para que el mapa muestre inmediatamente los puntos
            self.apply_raw_column_selection()

        except Exception as e:
            QMessageBox.critical(self, "Error al Cargar Archivo", f"No se pudo leer el archivo:\n{e}")

    def apply_raw_column_selection(self):
        """Aplica la asignación de campos elegida por el usuario y grafica los puntos coloreados en el Mapa 2D."""
        if self.df_raw_source is None:
            QMessageBox.warning(self, "Sin Archivo", "Primero debes cargar un archivo de observaciones crudas.")
            return

        cx = self.combo_raw_col_x.currentText()
        cy = self.combo_raw_col_y.currentText()
        ch = self.combo_raw_col_h.currentText()
        clat = self.combo_raw_col_lat.currentText()
        clon = self.combo_raw_col_lon.currentText()
        cg = self.combo_raw_col_g.currentText()

        try:
            x_raw = pd.to_numeric(self.df_raw_source[cx], errors='coerce')
            y_raw = pd.to_numeric(self.df_raw_source[cy], errors='coerce')
            h_raw = pd.to_numeric(self.df_raw_source[ch], errors='coerce').fillna(0.0) if ch in self.df_raw_source.columns else pd.Series(0.0, index=self.df_raw_source.index)
            g_raw = pd.to_numeric(self.df_raw_source[cg], errors='coerce')

            if clat in self.df_raw_source.columns:
                lat_raw = pd.to_numeric(self.df_raw_source[clat], errors='coerce')
            else:
                if y_raw.abs().max() <= 90.0:
                    lat_raw = y_raw
                else:
                    lat_raw = pd.Series(-26.8, index=self.df_raw_source.index)

            if clon in self.df_raw_source.columns:
                lon_raw = pd.to_numeric(self.df_raw_source[clon], errors='coerce')
            else:
                if x_raw.abs().max() <= 360.0:
                    lon_raw = x_raw
                else:
                    lon_raw = pd.Series(-65.2, index=self.df_raw_source.index)

            # Proyección métrica si son coordenadas angulares
            if x_raw.abs().max() <= 360.0 and y_raw.abs().max() <= 90.0:
                gx, gy = latlon_to_gauss_kruger(lat_raw, lon_raw)
                x_metric = gx
                y_metric = gy
            else:
                x_metric = x_raw
                y_metric = y_raw

            valid = ~(x_metric.isna() | y_metric.isna() | g_raw.isna() | lat_raw.isna())
            if valid.sum() < 4:
                QMessageBox.warning(self, "Datos Insuficientes", "Se requieren al menos 4 puntos válidos.")
                return

            df = pd.DataFrame({
                'id': [f"EST_{i+1:03d}" for i in range(valid.sum())],
                'X': x_metric[valid].to_numpy(dtype=float),
                'Y': y_metric[valid].to_numpy(dtype=float),
                'h': h_raw[valid].to_numpy(dtype=float),
                'lat': lat_raw[valid].to_numpy(dtype=float),
                'long': lon_raw[valid].to_numpy(dtype=float),
                'G': g_raw[valid].to_numpy(dtype=float)
            })

            # Gravedad normal de Somigliana (Eqn 7.11)
            model_key = self.combo_ellipsoid.currentData()
            g0 = compute_normal_gravity(df['lat'], model_key=model_key)
            df['g0'] = g0
            df['anom_g0'] = df['G'] - g0

            self.raw_grav_df = df
            self.step_g0_done = True
            self.step_fa_done = False
            self.step_sb_done = False
            self.step_tc_done = False

            # Inyectar directamente a la nube de puntos activa para el Mapa 2D
            self.pts_x = df['X'].to_numpy(dtype=float)
            self.pts_y = df['Y'].to_numpy(dtype=float)
            self.pts_z = df['G'].to_numpy(dtype=float)
            self.x_col = cx; self.y_col = cy; self.z_col = "G_obs"
            self.unit_xy = "m"; self.unit_z = "mGal"

            # Actualizar límites de zona
            x_min, x_max = float(np.min(self.pts_x)), float(np.max(self.pts_x))
            y_min, y_max = float(np.min(self.pts_y)), float(np.max(self.pts_y))
            z_min, z_max = float(np.min(self.pts_z)), float(np.max(self.pts_z))
            n = len(self.pts_x)
            area = max(1e-12, (x_max - x_min) * (y_max - y_min))
            density = n / area

            self.lbl_stats.setText(
                f"Estaciones Válidas: {n:,}\n"
                f"X ({cx}): [{x_min:.1f} m , {x_max:.1f} m]\n"
                f"Y ({cy}): [{y_min:.1f} m , {y_max:.1f} m]\n"
                f"G_obs: [{z_min:.2f} mGal , {z_max:.2f} mGal] (Media: {np.mean(self.pts_z):.2f} mGal)\n"
                f"g₀ (Somigliana): [{df['g0'].min():.2f}, {df['g0'].max():.2f}] mGal\n"
                f"Densidad: {density:.2e} pts/m²"
            )

            prev_x = getattr(self, 'x_col', None)
            prev_y = getattr(self, 'y_col', None)
            has_bounds = hasattr(self, 'x_min_sel') and self.x_min_sel is not None
            same_xy = (prev_x == cx and prev_y == cy and has_bounds)

            if not same_xy:
                self.spin_xmin.blockSignals(True); self.spin_xmax.blockSignals(True)
                self.spin_ymin.blockSignals(True); self.spin_ymax.blockSignals(True)
                self.spin_xmin.setValue(x_min); self.spin_xmax.setValue(x_max)
                self.spin_ymin.setValue(y_min); self.spin_ymax.setValue(y_max)
                self.x_min_sel = x_min; self.x_max_sel = x_max
                self.y_min_sel = y_min; self.y_max_sel = y_max
                self.spin_xmin.blockSignals(False); self.spin_xmax.blockSignals(False)
                self.spin_ymin.blockSignals(False); self.spin_ymax.blockSignals(False)

                self.suggest_optimal_spacing()
                self.plot_data_and_zone(preserve_view=False)  # ¡MAPA 2D COLOREADO INMEDIATAMENTE!
            else:
                self.update_grid_dimension_labels()
                self.plot_data_and_zone(preserve_view=True)

            self.update_raw_table()

            # Habilitar secuencialmente Aire Libre
            self.btn_apply_fa.setEnabled(True)
            self.btn_apply_b.setEnabled(False)
            self.btn_apply_t.setEnabled(False)

            self.tabs_data_views.setCurrentIndex(0)
            self.status.showMessage(f"{n:,} estaciones cargadas. g₀ calculada. Mapa 2D coloreado por G_obs.")

        except Exception as e:
            QMessageBox.critical(self, "Error al Mapear Columnas Crudas", f"Error procesando columnas:\n{e}")

    def set_raw_dataframe(self, df_in, source_name="Archivo"):
        """Permite inyección directa de DataFrame para pruebas o scripts."""
        self.df_raw_source = df_in
        cols = list(df_in.columns)
        for cb in [self.combo_raw_col_x, self.combo_raw_col_y, self.combo_raw_col_h,
                   self.combo_raw_col_lat, self.combo_raw_col_lon, self.combo_raw_col_g]:
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(cols)
            cb.blockSignals(False)

        def match_col(names, priority_words):
            for pw in priority_words:
                for i, name in enumerate(names):
                    nl = name.lower()
                    if nl == pw or nl.startswith(pw):
                        return i
            return -1

        idx_x = match_col(cols, ['x', 'este', 'easting', 'lon'])
        idx_y = match_col(cols, ['y', 'norte', 'northing', 'lat'])
        idx_h = match_col(cols, ['h', 'cota', 'alt', 'elev'])
        idx_lat = match_col(cols, ['lat', 'latitude', 'y'])
        idx_lon = match_col(cols, ['lon', 'long', 'x'])
        idx_g = match_col(cols, ['g_obs', 'g', 'grav', 'obs'])

        if idx_x >= 0: self.combo_raw_col_x.setCurrentIndex(idx_x)
        if idx_y >= 0: self.combo_raw_col_y.setCurrentIndex(idx_y)
        if idx_h >= 0: self.combo_raw_col_h.setCurrentIndex(idx_h)
        if idx_lat >= 0: self.combo_raw_col_lat.setCurrentIndex(idx_lat)
        if idx_lon >= 0: self.combo_raw_col_lon.setCurrentIndex(idx_lon)
        if idx_g >= 0: self.combo_raw_col_g.setCurrentIndex(idx_g)

        self.apply_raw_column_selection()

    def run_raw_step_free_air(self):
        """Paso: Aplicar Aire Libre (+0.3086·h)."""
        if self.raw_grav_df is None or not self.step_g0_done:
            QMessageBox.warning(self, "Sin Datos", "Primero aplica la asignación de campos para calcular g₀.")
            return

        grad = self.spin_raw_grad_fa.value()
        geoid_n = self.spin_raw_geoid_n.value()
        h_eff = self.raw_grav_df['h'] - geoid_n

        fac = compute_free_air_correction(h_eff, lat_deg=None, gradient=grad, use_second_order=False)
        self.raw_grav_df['c_fa'] = fac
        self.raw_grav_df['anom_fa'] = self.raw_grav_df['anom_g0'] + fac
        self.step_fa_done = True

        # Inyectar anomalía a los puntos activos y actualizar el MAPA 2D
        self.pts_z = self.raw_grav_df['anom_fa'].to_numpy(dtype=float)
        self.z_col = "Δg_AireLibre"

        z_min, z_max = float(np.min(self.pts_z)), float(np.max(self.pts_z))
        self.lbl_stats.setText(
            f"Estaciones: {len(self.pts_z):,}\n"
            f"Anomalía Aire Libre: [{z_min:.2f} mGal , {z_max:.2f} mGal] (Media: {np.mean(self.pts_z):.2f} mGal)\n"
            f"Corrección FAC media: {np.mean(fac):.2f} mGal"
        )

        self.plot_data_and_zone(preserve_view=True)  # ¡MAPA 2D ACTUALIZADO CON COLORES DE AIRE LIBRE!
        self.update_raw_table()

        # Habilitar Bouguer
        self.btn_apply_b.setEnabled(True)
        self.tabs_data_views.setCurrentIndex(0)
        self.status.showMessage("Corrección de Aire Libre aplicada (+0.3086·h). Mapa 2D actualizado.")

    def run_raw_step_bouguer(self):
        """Paso: Aplicar Placa de Bouguer (-0.1119·h)."""
        if not self.step_fa_done:
            QMessageBox.warning(self, "Paso previo requerido",
                                "La corrección es SECUENCIAL. Primero debes aplicar la corrección de Aire Libre.")
            return

        rho = self.spin_raw_rho_b.value()
        geoid_n = self.spin_raw_geoid_n.value()
        h_eff = self.raw_grav_df['h'] - geoid_n
        c_bp = compute_bouguer_slab_correction(h_eff, rho=rho)

        self.raw_grav_df['c_bp'] = c_bp
        self.raw_grav_df['anom_sb'] = self.raw_grav_df['anom_fa'] - c_bp
        self.step_sb_done = True

        # Inyectar anomalía a los puntos activos y actualizar el MAPA 2D
        self.pts_z = self.raw_grav_df['anom_sb'].to_numpy(dtype=float)
        self.z_col = "Δg_Bouguer"

        z_min, z_max = float(np.min(self.pts_z)), float(np.max(self.pts_z))
        self.lbl_stats.setText(
            f"Estaciones: {len(self.pts_z):,}\n"
            f"Anomalía Bouguer: [{z_min:.2f} mGal , {z_max:.2f} mGal] (Media: {np.mean(self.pts_z):.2f} mGal)\n"
            f"Placa Bouguer media: -{np.mean(c_bp):.2f} mGal (ρ={rho:.0f} kg/m³)"
        )

        self.plot_data_and_zone(preserve_view=True)  # ¡MAPA 2D ACTUALIZADO CON COLORES DE BOUGUER!
        self.update_raw_table()

        # Habilitar Topografía
        self.btn_apply_t.setEnabled(True)
        self.tabs_data_views.setCurrentIndex(0)
        self.status.showMessage(f"Placa de Bouguer aplicada (-0.1119·h, ρ={rho:.0f} kg/m³). Mapa 2D actualizado.")

    def on_select_raw_mdt_file(self):
        """Carga archivo GeoTIFF de MDT seleccionado por el usuario (sin pre-ejemplos)."""
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Cargar Modelo Digital de Terreno (MDT GeoTIFF)", BASE_DIR,
            "Archivos GeoTIFF (*.tif *.tiff);;Todos (*.*)"
        )
        if not fpath:
            return
        try:
            self.mdt_obj = GeoTiffMDT(fpath)
            if self.mdt_obj.is_loaded:
                self.lbl_raw_mdt_status.setText(
                    f"MDT: {os.path.basename(fpath)} [{self.mdt_obj.nx}x{self.mdt_obj.ny}]\n"
                    f"Δx={self.mdt_obj.dx:.0f}m, Δy={self.mdt_obj.dy:.0f}m | Cotas: [{self.mdt_obj.z_min:.0f}, {self.mdt_obj.z_max:.0f}] m"
                )
                self.lbl_raw_mdt_status.setStyleSheet("color: #006600; font-size: 10px;")
        except Exception as e:
            QMessageBox.critical(self, "Error al Cargar MDT", f"No se pudo leer el archivo MDT:\n{e}")

    def run_raw_step_terrain(self):
        """Paso: Aplicar Corrección de Topografía con MDT GeoTIFF."""
        if not self.step_sb_done:
            QMessageBox.warning(self, "Paso previo requerido",
                                "La corrección es SECUENCIAL. Para aplicar topografía debe estar aplicada Bouguer y Aire Libre.")
            return
        if not self.mdt_obj.is_loaded:
            QMessageBox.warning(self, "MDT requerido",
                                "Primero debes cargar un archivo MDT GeoTIFF (.tif) para calcular la topografía.")
            return

        r_max = self.spin_raw_rmax.value()
        rho = self.spin_raw_rho_b.value()

        self.btn_apply_t.setEnabled(False)
        self.btn_cancel_raw_t.setEnabled(True)
        self.progress_raw_terrain.setValue(0)

        self.terrain_worker = WorkerCalculoTerreno(
            self.mdt_obj, self.raw_grav_df['X'], self.raw_grav_df['Y'], self.raw_grav_df['h'],
            r_max=r_max, rho=rho
        )
        self.terrain_worker.progreso.connect(self.on_raw_terrain_progress)
        self.terrain_worker.finalizado.connect(self.on_raw_terrain_finished)
        self.terrain_worker.error.connect(self.on_raw_terrain_error)
        self.terrain_worker.start()

    def on_raw_terrain_progress(self, current, total):
        pct = int(100.0 * current / total)
        self.progress_raw_terrain.setValue(pct)

    def on_raw_terrain_finished(self, t_vals):
        self.btn_apply_t.setEnabled(True)
        self.btn_cancel_raw_t.setEnabled(False)
        self.progress_raw_terrain.setValue(100)

        self.raw_grav_df['tc'] = t_vals
        self.raw_grav_df['anom_cb'] = self.raw_grav_df['anom_sb'] + t_vals
        self.step_tc_done = True

        # Inyectar anomalía a los puntos activos y actualizar el MAPA 2D
        self.pts_z = self.raw_grav_df['anom_cb'].to_numpy(dtype=float)
        self.z_col = "Δg_Final(Topografía)"

        z_min, z_max = float(np.min(self.pts_z)), float(np.max(self.pts_z))
        self.lbl_stats.setText(
            f"Estaciones: {len(self.pts_z):,}\n"
            f"Anomalía con Topografía: [{z_min:.2f} mGal , {z_max:.2f} mGal] (Media: {np.mean(self.pts_z):.2f} mGal)\n"
            f"Corrección Terreno T media: +{np.mean(t_vals):.2f} mGal"
        )

        self.plot_data_and_zone(preserve_view=True)  # ¡MAPA 2D ACTUALIZADO CON COLORES DE TOPOGRAFÍA!
        self.update_raw_table()
        self.tabs_data_views.setCurrentIndex(0)
        self.status.showMessage("Corrección de Topografía (+T) aplicada. Mapa 2D actualizado con la anomalía final.")

    def on_raw_terrain_error(self, err_msg):
        self.btn_apply_t.setEnabled(True)
        self.btn_cancel_raw_t.setEnabled(False)
        QMessageBox.critical(self, "Error en Cálculo de Terreno", f"Ocurrió un error en el cálculo:\n{err_msg}")

    def on_cancel_raw_terrain(self):
        if self.terrain_worker and self.terrain_worker.isRunning():
            self.terrain_worker.cancel()
            self.btn_apply_t.setEnabled(True)
            self.btn_cancel_raw_t.setEnabled(False)
            self.status.showMessage("Cálculo de terreno cancelado.")

    def plot_raw_reductions_profile(self):
        """Método de compatibilidad interna (sin perfiles en UI)."""
        pass

    def update_raw_table(self):
        """Actualiza la tabla interactiva de estaciones y reducciones."""
        if self.raw_grav_df is None:
            self.table_raw_reductions.setRowCount(0)
            return
        cols = ['id', 'long', 'lat', 'h', 'X', 'Y', 'G', 'g0', 'c_fa', 'anom_fa', 'c_bp', 'anom_sb', 'tc', 'anom_cb']
        avail_cols = [c for c in cols if c in self.raw_grav_df.columns]
        self.table_raw_reductions.setColumnCount(len(avail_cols))
        self.table_raw_reductions.setHorizontalHeaderLabels(avail_cols)
        self.table_raw_reductions.setRowCount(len(self.raw_grav_df))
        for r_idx, row in self.raw_grav_df.iterrows():
            for c_idx, col in enumerate(avail_cols):
                val = row[col]
                txt = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                self.table_raw_reductions.setItem(r_idx, c_idx, item)
        self.table_raw_reductions.resizeColumnsToContents()

    def on_export_raw_reduced_csv(self):
        """Exporta la tabla completa de reducciones a CSV."""
        if self.raw_grav_df is None or len(self.raw_grav_df) == 0:
            QMessageBox.warning(self, "Sin datos", "No hay datos para exportar.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Tabla de Mediciones y Reducciones", "Mediciones_Reducidas.csv",
            "Archivos CSV (*.csv);;Todos (*.*)"
        )
        if not save_path:
            return
        try:
            self.raw_grav_df.to_csv(save_path, index=False)
            QMessageBox.information(self, "Guardado exitoso", f"Archivo guardado en:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", f"No se pudo guardar el archivo:\n{e}")

    def show_about_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Acerca de GravTT")
        dlg.setFixedSize(460, 340)
        l = QVBoxLayout(dlg)
        l.setContentsMargins(20, 20, 20, 20)
        l.setSpacing(10)

        logo_path = os.path.join(BASE_DIR, 'assets', 'logo.png')
        if os.path.exists(logo_path):
            lbl_img = QLabel()
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setPixmap(QPixmap(logo_path).scaled(110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            l.addWidget(lbl_img)

        lbl_t = QLabel("<b style='font-size: 16px; color: #cc0000;'>GravTT v1.0</b>")
        lbl_t.setAlignment(Qt.AlignCenter)
        l.addWidget(lbl_t)

        lbl_d = QLabel(
            "<p align='center' style='font-size: 11px; color: #333333; margin: 0;'>"
            "Suite Geofísica Integral de Procesamiento Gravimétrico,<br>"
            "Interpolación Espacial 2D, Inversión por Fuentes Equivalentes<br>"
            "y Análisis Espectral 2D por Transformada Rápida de Fourier (FFT)."
            "</p>"
            "<p align='center' style='font-size: 10px; color: #666666; margin: 4px 0;'>"
            "Modelado Analítico Blakely (1995) | Reducciones Somigliana / Bouguer<br>"
            "Licencia MIT (2026)"
            "</p>"
        )
        lbl_d.setAlignment(Qt.AlignCenter)
        l.addWidget(lbl_d)

        btn_ok = QPushButton("Aceptar")
        btn_ok.setObjectName("btnPrimary")
        btn_ok.setFixedWidth(110)
        btn_ok.clicked.connect(dlg.accept)
        row_btn = QHBoxLayout()
        row_btn.addStretch()
        row_btn.addWidget(btn_ok)
        row_btn.addStretch()
        l.addLayout(row_btn)
        dlg.exec_()

    # ==========================================================================
    # CARGA DE ARCHIVOS Y ASIGNACIÓN DE COLUMNAS / UNIDADES
    # ==========================================================================
    def browse_file(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Abrir Tabla de Datos Irregulares", BASE_DIR,
            "Archivos Soportados (*.csv *.xlsx *.xls *.json *.txt *.tsv *.xyz *.dat);;CSV (*.csv);;Excel (*.xlsx *.xls);;JSON (*.json);;Todos (*.*)"
        )
        if fpath:
            self.load_file(fpath)

    def load_file(self, filepath):
        self.raw_filepath = filepath
        fname = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext in ['.xlsx', '.xls']:
                self.df_raw = pd.read_excel(filepath)
            elif ext == '.json':
                try:
                    self.df_raw = pd.read_json(filepath)
                except ValueError:
                    self.df_raw = pd.read_json(filepath, lines=True)
            else:
                detected_sep = ','
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    sample = [f.readline() for _ in range(12)]
                sample_text = "".join(sample)

                counts = {
                    ',': sample_text.count(','),
                    ';': sample_text.count(';'),
                    '\t': sample_text.count('\t'),
                    ' ': sample_text.count(' ')
                }
                detected_sep = max(counts, key=counts.get)
                if detected_sep == ' ' and counts[' '] > 0:
                    try:
                        self.df_raw = pd.read_csv(filepath, sep=r'\s+', engine='python')
                    except Exception:
                        self.df_raw = pd.read_csv(filepath, sep=detected_sep)
                else:
                    self.df_raw = pd.read_csv(filepath, sep=detected_sep)

            cols = [str(c).strip() for c in self.df_raw.columns]
            self.df_raw.columns = cols

            self.combo_col_x.blockSignals(True)
            self.combo_col_y.blockSignals(True)
            self.combo_col_z.blockSignals(True)
            self.combo_col_x.clear()
            self.combo_col_y.clear()
            self.combo_col_z.clear()
            self.combo_col_x.addItems(cols)
            self.combo_col_y.addItems(cols)
            self.combo_col_z.addItems(cols)

            def match_col(names, priority_words):
                for pw in priority_words:
                    for i, name in enumerate(names):
                        nl = name.lower()
                        if nl == pw or nl.startswith(pw):
                            return i
                return -1

            idx_x = match_col(cols, ['x', 'lon', 'longitude', 'longitud', 'este', 'easting'])
            idx_y = match_col(cols, ['y', 'lat', 'latitude', 'latitud', 'norte', 'northing'])
            idx_z = match_col(cols, ['z', 'g', 'elev', 'elevation', 'hs', 'alt', 'anomalia', 'gravity', 'mag', 'val'])

            if idx_x >= 0: self.combo_col_x.setCurrentIndex(idx_x)
            elif len(cols) > 0: self.combo_col_x.setCurrentIndex(0)

            if idx_y >= 0: self.combo_col_y.setCurrentIndex(idx_y)
            elif len(cols) > 1: self.combo_col_y.setCurrentIndex(1)

            if idx_z >= 0: self.combo_col_z.setCurrentIndex(idx_z)
            elif len(cols) > 2: self.combo_col_z.setCurrentIndex(2)

            self.combo_col_x.blockSignals(False)
            self.combo_col_y.blockSignals(False)
            self.combo_col_z.blockSignals(False)

            # Auto-inferencia inteligente de unidades físicas
            z_cand = self.combo_col_z.currentText().lower()
            if any(w in z_cand for w in ['elev', 'alt', 'hs', 'cota', 'topo', 'height']):
                idx_u = self.combo_unit_z.findData('m')
                if idx_u >= 0: self.combo_unit_z.setCurrentIndex(idx_u)
            elif any(w in z_cand for w in ['mag', 'tmi', 'nano', 'campo']):
                idx_u = self.combo_unit_z.findData('nT')
                if idx_u >= 0: self.combo_unit_z.setCurrentIndex(idx_u)
            elif any(w in z_cand for w in ['g', 'grav', 'anom', 'bouguer', 'faa', 'mgal']):
                idx_u = self.combo_unit_z.findData('mGal')
                if idx_u >= 0: self.combo_unit_z.setCurrentIndex(idx_u)

            x_cand = self.combo_col_x.currentText().lower()
            if any(w in x_cand for w in ['lon', 'lat', 'deg', '°']):
                idx_u = self.combo_unit_xy.findData('°')
                if idx_u >= 0: self.combo_unit_xy.setCurrentIndex(idx_u)

            self.on_units_changed()

            self.lbl_file_status.setText(f"{fname} ({len(self.df_raw):,} filas)")
            self.populate_preview_table()
            self.apply_column_selection()

        except Exception as e:
            QMessageBox.critical(self, "Error al Cargar Archivo", f"No se pudo leer el archivo:\n{e}")

    def populate_preview_table(self):
        if self.df_raw is None:
            return
        n_rows = min(8, len(self.df_raw))
        n_cols = len(self.df_raw.columns)
        self.table_preview.setRowCount(n_rows)
        self.table_preview.setColumnCount(n_cols)
        self.table_preview.setHorizontalHeaderLabels([str(c) for c in self.df_raw.columns])

        for r in range(n_rows):
            for c in range(n_cols):
                val = str(self.df_raw.iloc[r, c])
                item = QTableWidgetItem(val)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table_preview.setItem(r, c, item)

    def on_columns_selected(self):
        pass

    def apply_column_selection(self):
        if self.df_raw is None:
            return
        cx = self.combo_col_x.currentText()
        cy = self.combo_col_y.currentText()
        cz = self.combo_col_z.currentText()

        try:
            sub = self.df_raw[[cx, cy, cz]].dropna()
            self.pts_x = pd.to_numeric(sub[cx], errors='coerce').to_numpy()
            self.pts_y = pd.to_numeric(sub[cy], errors='coerce').to_numpy()
            self.pts_z = pd.to_numeric(sub[cz], errors='coerce').to_numpy()

            valid = ~(np.isnan(self.pts_x) | np.isnan(self.pts_y) | np.isnan(self.pts_z))
            self.pts_x = self.pts_x[valid]
            self.pts_y = self.pts_y[valid]
            self.pts_z = self.pts_z[valid]

            n = len(self.pts_x)
            if n < 4:
                QMessageBox.warning(self, "Datos Insuficientes", "Se requieren al menos 4 puntos válidos.")
                return

            prev_x = getattr(self, 'x_col', None)
            prev_y = getattr(self, 'y_col', None)
            has_bounds = hasattr(self, 'x_min_sel') and self.x_min_sel is not None
            same_xy = (prev_x == cx and prev_y == cy and has_bounds)

            self.x_col = cx; self.y_col = cy; self.z_col = cz

            x_min, x_max = float(np.min(self.pts_x)), float(np.max(self.pts_x))
            y_min, y_max = float(np.min(self.pts_y)), float(np.max(self.pts_y))
            z_min, z_max = float(np.min(self.pts_z)), float(np.max(self.pts_z))
            area = max(1e-12, (x_max - x_min) * (y_max - y_min))
            density = n / area
            u = self.unit_xy
            z = self.unit_z

            self.lbl_stats.setText(
                f"Puntos Válidos: {n:,}\n"
                f"X ({cx}): [{x_min:.4f} {u} , {x_max:.4f} {u}] (ΔX={x_max-x_min:.4f} {u})\n"
                f"Y ({cy}): [{y_min:.4f} {u} , {y_max:.4f} {u}] (ΔY={y_max-y_min:.4f} {u})\n"
                f"Z ({cz}): [{z_min:.4f} {z} , {z_max:.4f} {z}] (Media: {np.mean(self.pts_z):.2f} {z})\n"
                f"Densidad: {density:.2e} pts/{u}²"
            )

            if not same_xy:
                self.spin_xmin.blockSignals(True); self.spin_xmax.blockSignals(True)
                self.spin_ymin.blockSignals(True); self.spin_ymax.blockSignals(True)

                self.spin_xmin.setValue(x_min); self.spin_xmax.setValue(x_max)
                self.spin_ymin.setValue(y_min); self.spin_ymax.setValue(y_max)

                self.x_min_sel = x_min; self.x_max_sel = x_max
                self.y_min_sel = y_min; self.y_max_sel = y_max

                self.spin_xmin.blockSignals(False); self.spin_xmax.blockSignals(False)
                self.spin_ymin.blockSignals(False); self.spin_ymax.blockSignals(False)

                self.suggest_optimal_spacing()
                self.plot_data_and_zone(preserve_view=False)
            else:
                self.update_grid_dimension_labels()
                self.plot_data_and_zone(preserve_view=True)

            self.status.showMessage(f"{n:,} puntos cargados correctamente para X='{cx}' [{u}], Y='{cy}' [{u}], Z='{cz}' [{z}].")

        except Exception as e:
            QMessageBox.critical(self, "Error al Mapear Columnas", f"Error procesando columnas XYZ:\n{e}")

    # ==========================================================================
    # GRAFICACIÓN: DISPERSIÓN DE PUNTOS COLOREADOS POR Z CON DELIMITACIÓN Y RECORTE
    # ==========================================================================
    def plot_data_and_zone(self, preserve_view=False):
        fig = self.canvas_scatter.fig

        cur_xlim = None
        cur_ylim = None
        if preserve_view and fig.axes:
            try:
                ax_old = fig.axes[0]
                cur_xlim = ax_old.get_xlim()
                cur_ylim = ax_old.get_ylim()
            except Exception:
                pass

        if hasattr(self, 'rect_selector') and self.rect_selector is not None:
            try:
                self.rect_selector.set_active(False)
            except Exception:
                pass
            self.rect_selector = None

        fig.clf()
        self.zone_rect_patch = None

        if self.pts_x is None or len(self.pts_x) == 0:
            self.canvas_scatter.draw()
            return

        ax = fig.add_subplot(1, 1, 1)
        u = self.unit_xy
        z = self.unit_z

        n = len(self.pts_x)
        if n > 60000:
            step = int(np.ceil(n / 40000.0))
            px = self.pts_x[::step]; py = self.pts_y[::step]; pz = self.pts_z[::step]
            title_text = f"Nube de Puntos Irregulares Coloreados: {self.z_col} [Muestreo {len(px):,} de {n:,} pts]"
        else:
            px = self.pts_x; py = self.pts_y; pz = self.pts_z
            title_text = f"Nube de Puntos Irregulares Coloreados: {self.z_col} [{n:,} pts]"

        setup_ax_style(ax, title_text, f"X ({self.x_col}) [{u}]", f"Y ({self.y_col}) [{u}]")

        sc = ax.scatter(px, py, c=pz, cmap='turbo', s=12, edgecolors='none', alpha=0.85)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.12)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(f"Z ({self.z_col}) [{z}]", color='#000000', fontsize=9)
        cbar.ax.tick_params(colors='#000000', labelsize=8)

        x1, x2 = min(self.x_min_sel, self.x_max_sel), max(self.x_min_sel, self.x_max_sel)
        y1, y2 = min(self.y_min_sel, self.y_max_sel), max(self.y_min_sel, self.y_max_sel)
        self.zone_rect_patch = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.0, edgecolor='#cc0000', facecolor='#cc0000', alpha=0.18, linestyle='--',
            label="Zona a Interpolar (Recorte)"
        )
        ax.add_patch(self.zone_rect_patch)
        ax.legend(facecolor='#ffffff', edgecolor='#808080', labelcolor='#000000', loc='upper right', fontsize=8)

        x_min_data = float(np.min(self.pts_x))
        x_max_data = float(np.max(self.pts_x))
        y_min_data = float(np.min(self.pts_y))
        y_max_data = float(np.max(self.pts_y))
        span_x = max(1e-9, x_max_data - x_min_data)
        span_y = max(1e-9, y_max_data - y_min_data)
        pad_x, pad_y = span_x * 0.05, span_y * 0.05

        if cur_xlim is not None and cur_ylim is not None:
            ax.set_xlim(cur_xlim)
            ax.set_ylim(cur_ylim)
        else:
            ax.set_xlim(x_min_data - pad_x, x_max_data + pad_x)
            ax.set_ylim(y_min_data - pad_y, y_max_data + pad_y)

        ax.set_aspect('equal', adjustable='box')
        ax.set_anchor('C')

        def on_select_rect(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return
            self.spin_xmin.blockSignals(True); self.spin_xmax.blockSignals(True)
            self.spin_ymin.blockSignals(True); self.spin_ymax.blockSignals(True)
            self.spin_xmin.setValue(min(eclick.xdata, erelease.xdata))
            self.spin_xmax.setValue(max(eclick.xdata, erelease.xdata))
            self.spin_ymin.setValue(min(eclick.ydata, erelease.ydata))
            self.spin_ymax.setValue(max(eclick.ydata, erelease.ydata))
            self.spin_xmin.blockSignals(False); self.spin_xmax.blockSignals(False)
            self.spin_ymin.blockSignals(False); self.spin_ymax.blockSignals(False)
            self.x_min_sel = min(eclick.xdata, erelease.xdata)
            self.x_max_sel = max(eclick.xdata, erelease.xdata)
            self.y_min_sel = min(eclick.ydata, erelease.ydata)
            self.y_max_sel = max(eclick.ydata, erelease.ydata)
            self.update_grid_dimension_labels()
            self._update_zone_rect_patch()

        self.rect_selector = RectangleSelector(
            ax, on_select_rect, useblit=True,
            button=[1], minspanx=5, minspany=5, spancoords='data',
            interactive=True,
            props=dict(facecolor='#cc0000', edgecolor='#cc0000', alpha=0.22, fill=True)
        )

        fig.tight_layout()
        self.canvas_scatter.draw()

    def _update_zone_rect_patch(self):
        if getattr(self, 'zone_rect_patch', None) is not None and self.canvas_scatter.fig.axes:
            x1, x2 = min(self.x_min_sel, self.x_max_sel), max(self.x_min_sel, self.x_max_sel)
            y1, y2 = min(self.y_min_sel, self.y_max_sel), max(self.y_min_sel, self.y_max_sel)
            self.zone_rect_patch.set_xy((x1, y1))
            self.zone_rect_patch.set_width(x2 - x1)
            self.zone_rect_patch.set_height(y2 - y1)
            self.canvas_scatter.draw_idle()
        else:
            self.plot_data_and_zone()

    def plot_scatter_points(self):
        self.plot_data_and_zone()

    def plot_zone_selection(self):
        self._update_zone_rect_patch()

    # ==========================================================================
    # DELIMITACIÓN DE ZONA, ESPACIADO & ZOOM
    # ==========================================================================
    def reset_zone_to_full(self):
        if self.pts_x is None:
            return
        self.spin_xmin.setValue(float(np.min(self.pts_x)))
        self.spin_xmax.setValue(float(np.max(self.pts_x)))
        self.spin_ymin.setValue(float(np.min(self.pts_y)))
        self.spin_ymax.setValue(float(np.max(self.pts_y)))
        self.on_zone_spins_changed()
        self.zoom_zone_all()

    def zoom_zone_all(self):
        if self.pts_x is None or len(self.pts_x) == 0:
            return
        x_min_data, x_max_data = float(np.min(self.pts_x)), float(np.max(self.pts_x))
        y_min_data, y_max_data = float(np.min(self.pts_y)), float(np.max(self.pts_y))
        span_x = max(1e-9, x_max_data - x_min_data)
        span_y = max(1e-9, y_max_data - y_min_data)
        pad_x, pad_y = span_x * 0.05, span_y * 0.05
        if self.canvas_scatter.fig.axes:
            ax = self.canvas_scatter.fig.axes[0]
            ax.set_xlim(x_min_data - pad_x, x_max_data + pad_x)
            ax.set_ylim(y_min_data - pad_y, y_max_data + pad_y)
            self.canvas_scatter.draw_idle()

    def zoom_zone_selected(self):
        if self.pts_x is None:
            return
        x1, x2 = min(self.x_min_sel, self.x_max_sel), max(self.x_min_sel, self.x_max_sel)
        y1, y2 = min(self.y_min_sel, self.y_max_sel), max(self.y_min_sel, self.y_max_sel)
        span_x = max(1e-9, x2 - x1)
        span_y = max(1e-9, y2 - y1)
        pad_x, pad_y = span_x * 0.08, span_y * 0.08
        if self.canvas_scatter.fig.axes:
            ax = self.canvas_scatter.fig.axes[0]
            ax.set_xlim(x1 - pad_x, x2 + pad_x)
            ax.set_ylim(y1 - pad_y, y2 + pad_y)
            self.canvas_scatter.draw_idle()

    def on_zone_spins_changed(self):
        self.x_min_sel = min(self.spin_xmin.value(), self.spin_xmax.value())
        self.x_max_sel = max(self.spin_xmin.value(), self.spin_xmax.value())
        self.y_min_sel = min(self.spin_ymin.value(), self.spin_ymax.value())
        self.y_max_sel = max(self.spin_ymin.value(), self.spin_ymax.value())
        self.update_grid_dimension_labels()
        self._update_zone_rect_patch()

    def on_spacing_changed(self):
        if self.chk_square_cells.isChecked() and self.sender() == self.spin_dx:
            self.spin_dy.blockSignals(True)
            self.spin_dy.setValue(self.spin_dx.value())
            self.spin_dy.blockSignals(False)
        self.dx_val = max(1e-6, self.spin_dx.value())
        self.dy_val = max(1e-6, self.spin_dy.value())
        self.update_grid_dimension_labels()

    def on_square_cells_toggled(self):
        if self.chk_square_cells.isChecked():
            self.spin_dy.setValue(self.spin_dx.value())

    def suggest_optimal_spacing(self):
        if self.pts_x is None or len(self.pts_x) == 0:
            return
        span_x = self.x_max_sel - self.x_min_sel
        span_y = self.y_max_sel - self.y_min_sel
        area = max(1e-12, span_x * span_y)
        sugg = float(np.sqrt(area / max(100, len(self.pts_x))))
        sugg = float(f"{sugg:.4g}")
        self.spin_dx.setValue(sugg)
        self.spin_dy.setValue(sugg)
        self.on_spacing_changed()

    def update_grid_dimension_labels(self):
        span_x = max(0.0, self.x_max_sel - self.x_min_sel)
        span_y = max(0.0, self.y_max_sel - self.y_min_sel)
        dx = max(1e-6, self.spin_dx.value())
        dy = max(1e-6, self.spin_dy.value())

        nx = int(np.floor(span_x / dx)) + 1
        ny = int(np.floor(span_y / dy)) + 1
        total = nx * ny
        u = self.unit_xy
        k_nyq = 0.5 / dx

        if total > 2000000:
            self.lbl_grid_dims.setText(
                f"RESOLUCIÓN EXCESIVA ({nx:,} cols × {ny:,} filas = {total:,} nodos). ¡Aumenta ΔX, ΔY [{u}] para evitar congelamiento!"
            )
            self.lbl_grid_dims.setStyleSheet("font-size: 11px; font-weight: bold; color: #ff0000; background-color: #fff0f0; border: 1px solid #ff9999; padding: 4px;")
        elif nx < 2 or ny < 2:
            self.lbl_grid_dims.setText(
                f"RESOLUCIÓN INSUFICIENTE: {nx} cols × {ny} filas ({total} nodos). Paso {dx:g} {u} supera extensión {span_x:g} {u}."
            )
            self.lbl_grid_dims.setStyleSheet("font-size: 11px; font-weight: bold; color: #cc6600; background-color: #fff8f0; border: 1px solid #ffcc99; padding: 4px;")
        else:
            self.lbl_grid_dims.setText(
                f"Malla: {nx:,} cols × {ny:,} filas = {total:,} nodos | Celda: {dx:g}×{dy:g} {u} | Nyquist: {k_nyq:.5g} [ciclos/{u}]"
            )
            self.lbl_grid_dims.setStyleSheet("font-size: 11px; font-weight: bold; color: #007700; padding-top: 4px;")


    def on_method_changed(self):
        m = self.combo_method.currentData()
        self.panel_krig_params.setVisible(m == 'kriging')
        self.panel_idw_params.setVisible(m == 'idw')
        self.panel_rbf_params.setVisible(m == 'rbf')

    def start_interpolation(self):
        if self.pts_x is None or len(self.pts_x) == 0:
            QMessageBox.warning(self, "Sin Datos", "Carga una tabla de datos primero.")
            return

        x1 = min(self.spin_xmin.value(), self.spin_xmax.value())
        x2 = max(self.spin_xmin.value(), self.spin_xmax.value())
        y1 = min(self.spin_ymin.value(), self.spin_ymax.value())
        y2 = max(self.spin_ymin.value(), self.spin_ymax.value())

        dx = max(1e-6, self.spin_dx.value())
        dy = max(1e-6, self.spin_dy.value())

        span_x = abs(x2 - x1)
        span_y = abs(y2 - y1)
        nx_est = int(np.floor(span_x / dx)) + 1
        ny_est = int(np.floor(span_y / dy)) + 1
        tot_est = nx_est * ny_est

        if nx_est < 2 or ny_est < 2:
            QMessageBox.warning(
                self, "Resolución Insuficiente",
                f"El paso seleccionado (ΔX={dx:g} {self.unit_xy}, ΔY={dy:g} {self.unit_xy}) produce solo {nx_est}×{ny_est} nodos.\n\n"
                f"La extensión de la zona es {span_x:g} × {span_y:g} [{self.unit_xy}].\n"
                f"Si tus datos están en grados, define un paso menor (ej: 0.001°).\n"
                f"Si están en metros, ajusta ΔX para tener una malla representativa."
            )
            return

        if tot_est > 3500000:
            QMessageBox.critical(
                self, "Resolución Excesiva / Riesgo de Congelamiento",
                f"La resolución solicitada produciría {tot_est:,} nodos ({nx_est:,} cols × {ny_est:,} filas).\n\n"
                f"Esto saturaría la memoria RAM y bloquearía la aplicación.\n"
                f"Por favor, aumenta el espaciado ΔX y ΔY para generar una malla de tamaño moderado (< 1,500,000 nodos)."
            )
            return

        if tot_est > 1200000:
            reply = QMessageBox.question(
                self, "Malla Muy Densa",
                f"La resolución elegida creará {tot_est:,} nodos ({nx_est:,} cols × {ny_est:,} filas).\n¿Deseas continuar?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        mask = (self.pts_x >= x1) & (self.pts_x <= x2) & (self.pts_y >= y1) & (self.pts_y <= y2)
        sub_x = self.pts_x[mask]
        sub_y = self.pts_y[mask]
        sub_z = self.pts_z[mask]

        if len(sub_x) < 4:
            QMessageBox.warning(self, "Zona Vacía", "Hay menos de 4 puntos dentro de la zona seleccionada.")
            return

        x_grid = np.arange(x1, x2 + 0.5 * dx, dx)
        y_grid = np.arange(y1, y2 + 0.5 * dy, dy)
        nx = len(x_grid)
        ny = len(y_grid)

        method = self.combo_method.currentData()
        params = {}
        if method == 'kriging':
            params['k_neighbors'] = self.spin_krig_k.value()
            params['model'] = self.combo_krig_model.currentText()
        elif method == 'idw':
            params['power'] = self.spin_idw_power.value()
            params['k_neighbors'] = self.spin_idw_k.value()
        elif method == 'rbf':
            params['kernel'] = self.combo_rbf_kernel.currentText()

        self.btn_run_grid.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status.showMessage(f"Interpolando malla regular {nx}×{ny} con metodología '{method.upper()}'...")

        self.grid_x_vec = x_grid
        self.grid_y_vec = y_grid
        self.grid_pts_sub_x = sub_x
        self.grid_pts_sub_y = sub_y
        self.grid_pts_sub_z = sub_z

        self.interp_thread = InterpolationThread(sub_x, sub_y, sub_z, x_grid, y_grid, method, params)
        self.interp_thread.finished.connect(self.on_interpolation_finished)
        self.interp_thread.start()

    def on_interpolation_finished(self, Z_grid, var_grid, weights_grid, err_msg):
        self.btn_run_grid.setEnabled(True)
        self.progress_bar.setVisible(False)

        if err_msg or Z_grid is None:
            QMessageBox.critical(self, "Error de Interpolación", f"Falló la interpolación:\n{err_msg}")
            self.status.showMessage("Error durante la interpolación.")
            return

        self.grid_Z = Z_grid
        self.grid_Z_orig = Z_grid.copy()
        self.grid_var = var_grid
        self.grid_weights = weights_grid

        ny, nx = Z_grid.shape
        self.status.showMessage(f"Malla regular 2D generada con éxito: [{nx}×{ny} nodos].")

        self.recompute_fourier_2d()
        self.plot_master_viewer()
        self.update_mesh_table()
        self.tabs.setCurrentIndex(2)

    # ==========================================================================
    # VISUALIZADOR MAESTRO DE TODO (TAB 3): RENDERIZADO UNIVERSAL
    # ==========================================================================
    def plot_mesh_map(self):
        self.plot_master_viewer()

    def plot_master_viewer(self):
        fig = self.canvas_view.fig
        fig.clf()
        if self.grid_Z is None or self.grid_x_vec is None or self.grid_y_vec is None:
            self.canvas_view.draw()
            return

        ax = fig.add_subplot(1, 1, 1)

        x_grid = self.grid_x_vec
        y_grid = self.grid_y_vec
        ny, nx = self.grid_Z.shape
        u = self.unit_xy
        z = self.unit_z

        layer_key = self.combo_master_layer.currentData()
        cmap_name = self.combo_master_cmap.currentText()
        if self.chk_master_cmap_r.isChecked():
            cmap_name = cmap_name + "_r"

        extent_space = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
        data_to_plot = None
        is_freq = False
        unit_cbar = z
        title_str = ""

        # Selección de capa
        if layer_key == 'grid_orig':
            data_to_plot = self.grid_Z
            title_str = f"Malla Original Interpolada: {self.z_col}"
            unit_cbar = z

        elif layer_key == 'grid_taper':
            data_to_plot = self.grid_Z_tapered if self.grid_Z_tapered is not None else self.grid_Z
            raw_t_txt = self.combo_fourier_taper_target.currentText() if hasattr(self, 'combo_fourier_taper_target') else "Media"
            t_tgt = re.sub(r'^[^\w]+', '', raw_t_txt).split("(")[0].strip()
            title_str = f"Malla con Acondicionamiento Perimetral ({t_tgt}): {self.z_col}"
            unit_cbar = z

        elif layer_key == 'grid_filt':
            data_to_plot = self.grid_Z_filt if self.grid_Z_filt is not None else self.grid_Z
            title_str = f"Malla Filtrada Fourier 2D: {self.z_col}"
            unit_cbar = z

        elif layer_key == 'grid_diff':
            if self.grid_Z_filt is not None:
                data_to_plot = self.grid_Z - self.grid_Z_filt
            else:
                data_to_plot = np.zeros_like(self.grid_Z)
            title_str = f"Diferencia Residual (Z_orig - Z_filt): {self.z_col}"
            unit_cbar = z

        elif layer_key == 'eq_source_z0':
            if getattr(self, 'grid_eq_source_z0', None) is not None:
                data_to_plot = self.grid_eq_source_z0
                z0_val = getattr(self, 'grid_eq_source_z0_elev', 0.0)
                title_str = f"Gravedad Fuente Equivalente a Z0 = {z0_val:.1f} m"
                unit_cbar = "mGal"
            else:
                data_to_plot = self.grid_Z
                title_str = "Gravedad Fuente Equivalente (Sin datos calculados aún)"
                unit_cbar = z

        elif layer_key == 'eq_source_z0_filt':
            self.update_equivalent_source_filtered()
            if getattr(self, 'grid_eq_source_z0_filt', None) is not None:
                data_to_plot = self.grid_eq_source_z0_filt
                z0_val = getattr(self, 'grid_eq_source_z0_elev', 0.0)
                title_str = f"Gravedad Fuente Equivalente a Z0 ({z0_val:.1f} m) Filtrada Fourier 2D"
                unit_cbar = "mGal"
            else:
                data_to_plot = self.grid_Z
                title_str = "Gravedad Fuente Equivalente Filtrada (Sin datos calculados aún)"
                unit_cbar = z

        elif layer_key == 'thdr':
            # Gradiente Horizontal Total
            dz_dx = np.gradient(self.grid_Z, self.dx_val, axis=1)
            dz_dy = np.gradient(self.grid_Z, self.dy_val, axis=0)
            data_to_plot = np.sqrt(dz_dx**2 + dz_dy**2)
            title_str = f"Gradiente Horizontal Total THDR = √((∂Z/∂x)² + (∂Z/∂y)²)"
            unit_cbar = f"{z}/{u}"

        elif layer_key == 'vd1':
            # 1ra Derivada Vertical Espacial
            if self.F_shift_2d is not None and self.K_grid is not None:
                F_1vd = self.F_shift_2d * (2.0 * np.pi * self.K_grid)
                vd1 = np.real(np.fft.ifft2(np.fft.ifftshift(F_1vd)))
                data_to_plot = vd1
            else:
                data_to_plot = np.zeros_like(self.grid_Z)
            title_str = f"Primera Derivada Vertical Espacial (1VD: ∂Z/∂z)"
            unit_cbar = f"{z}/{u}"

        elif layer_key == 'spec_orig':
            if self.F_shift_2d is not None:
                data_to_plot = np.log10(np.abs(self.F_shift_2d) + 1.0)
            else:
                data_to_plot = np.zeros_like(self.grid_Z)
            is_freq = True
            title_str = "Espectro 2D de Amplitud Log10 (Original)"
            unit_cbar = f"log10(|F|)"

        elif layer_key == 'spec_filt':
            F_target = self.F_shift_filt if self.F_shift_filt is not None else self.F_shift_2d
            if F_target is not None:
                data_to_plot = np.log10(np.abs(F_target) + 1.0)
            else:
                data_to_plot = np.zeros_like(self.grid_Z)
            is_freq = True
            title_str = "Espectro 2D de Amplitud Log10 (Filtrado)"
            unit_cbar = f"log10(|F_filt|)"

        elif layer_key == 'filter_total':
            if self.H_total_2d is not None:
                data_to_plot = np.abs(self.H_total_2d)
            else:
                data_to_plot = np.ones_like(self.grid_Z)
            is_freq = True
            title_str = "Filtro Aplicado Total |H_total(u, v)|"
            unit_cbar = "Ganancia [adimensional]"

        elif layer_key == 'spec_real':
            if self.F_shift_2d is not None:
                data_to_plot = np.real(self.F_shift_2d)
            else:
                data_to_plot = np.zeros_like(self.grid_Z)
            is_freq = True
            title_str = "Espectro de Fourier - Parte Real Re(F)"
            unit_cbar = f"Re(F) [{z}·{u}²]"

        elif layer_key == 'spec_imag':
            if self.F_shift_2d is not None:
                data_to_plot = np.imag(self.F_shift_2d)
            else:
                data_to_plot = np.zeros_like(self.grid_Z)
            is_freq = True
            title_str = "Espectro de Fourier - Parte Imaginaria Im(F)"
            unit_cbar = f"Im(F) [{z}·{u}²]"

        elif layer_key == 'spec_phase':
            if self.F_shift_2d is not None:
                data_to_plot = np.angle(self.F_shift_2d)
            else:
                data_to_plot = np.zeros_like(self.grid_Z)
            is_freq = True
            title_str = "Espectro de Fase 2D ∠F(u, v)"
            unit_cbar = "Fase [radianes]"

        elif layer_key == 'krig_var':
            if self.grid_var is not None:
                data_to_plot = self.grid_var
            else:
                data_to_plot = np.zeros_like(self.grid_Z)
            title_str = "Varianza de Estimación (Kriging Ordinario)"
            unit_cbar = f"Varianza [{z}²]"

        elif layer_key == 'weights_p':
            if getattr(self, 'grid_weights', None) is not None:
                data_to_plot = self.grid_weights
            else:
                data_to_plot = np.ones_like(self.grid_Z)
            title_str = "Matriz de Pesos P (Inverso a la Distancia al Dato)"
            unit_cbar = "Peso P [1/m]"

        elif layer_key == 'mdt':
            if getattr(self, 'grid_mdt', None) is not None:
                if self.grid_mdt.shape == self.grid_Z.shape:
                    data_to_plot = self.grid_mdt
                else:
                    from scipy.interpolate import RegularGridInterpolator
                    mx = getattr(self, 'mdt_x_vec', self.grid_x_vec)
                    my = getattr(self, 'mdt_y_vec', self.grid_y_vec)
                    interp = RegularGridInterpolator((my, mx), self.grid_mdt, bounds_error=False, fill_value=None)
                    xi, yi = np.meshgrid(self.grid_x_vec, self.grid_y_vec)
                    pts_eval = np.column_stack([yi.ravel(), xi.ravel()])
                    data_to_plot = interp(pts_eval).reshape(self.grid_Z.shape)
                title_str = "Topografía / Elevación MDT (Z_terreno)"
                unit_cbar = "m"
            else:
                data_to_plot = np.zeros_like(self.grid_Z)
                title_str = "MDT Topografía (Sin archivo MDT cargado aún)"
                unit_cbar = "m"

        elif layer_key == 'scatter_raw':
            setup_ax_style(ax, f"Nube de Puntos Irregulares: {self.z_col}", f"X ({self.x_col}) [{u}]", f"Y ({self.y_col}) [{u}]")
            if getattr(self, 'chk_master_aspect_equal', None) and self.chk_master_aspect_equal.isChecked():
                ax.set_aspect('equal', adjustable='box')
            else:
                ax.set_aspect('auto')

            # Si el usuario activó "Superponer puntos dentro de la malla", restringir a los puntos de la malla
            if self.chk_master_overlay_pts.isChecked():
                x_min_mesh = min(float(x_grid[0]), float(x_grid[-1]))
                x_max_mesh = max(float(x_grid[0]), float(x_grid[-1]))
                y_min_mesh = min(float(y_grid[0]), float(y_grid[-1]))
                y_max_mesh = max(float(y_grid[0]), float(y_grid[-1]))
                mask_in_mesh = (
                    (self.pts_x >= x_min_mesh) & (self.pts_x <= x_max_mesh) &
                    (self.pts_y >= y_min_mesh) & (self.pts_y <= y_max_mesh)
                )
                sc_x = self.pts_x[mask_in_mesh]
                sc_y = self.pts_y[mask_in_mesh]
                sc_z = self.pts_z[mask_in_mesh] if self.pts_z is not None else None
                ax.set_xlim(extent_space[0], extent_space[1])
                ax.set_ylim(extent_space[2], extent_space[3])
                lbl_tag = f"Puntos dentro de la Malla: {len(sc_x):,}"
            else:
                sc_x = self.pts_x
                sc_y = self.pts_y
                sc_z = self.pts_z
                lbl_tag = f"Puntos Crudos Totales: {len(self.pts_x):,}"

            if sc_x is not None and len(sc_x) > 0:
                sc = ax.scatter(sc_x, sc_y, c=sc_z, cmap=cmap_name, s=14, edgecolors='none', alpha=0.9)
                ax.set_anchor('C')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3.5%", pad=0.12)
                cb = fig.colorbar(sc, cax=cax)
                cb.set_label(f"Z ({self.z_col}) [{z}]", color='#000000', fontsize=8)
                z_min_disp = float(np.min(sc_z)) if sc_z is not None else 0.0
                z_max_disp = float(np.max(sc_z)) if sc_z is not None else 0.0
                self.lbl_master_stat.setText(f"{lbl_tag} | Mín: {z_min_disp:.2f} {z} | Máx: {z_max_disp:.2f} {z}")
            else:
                self.lbl_master_stat.setText("No hay puntos dentro del sector de la malla.")
            self.master_active_data = None
            self.master_active_is_freq = False
            self.master_active_unit = z
            fig.tight_layout()
            self.canvas_view.draw()
            return

        if data_to_plot is None:
            data_to_plot = self.grid_Z

        # Estadísticas de la capa
        vmin_calc = float(np.nanmin(data_to_plot))
        vmax_calc = float(np.nanmax(data_to_plot))
        vmean_calc = float(np.nanmean(data_to_plot))
        vstd_calc = float(np.nanstd(data_to_plot))
        self.lbl_master_stat.setText(
            f"Capa: {nx}×{ny} nodos | Mín: {vmin_calc:.4g} {unit_cbar} | Máx: {vmax_calc:.4g} {unit_cbar}\n"
            f"Media: {vmean_calc:.4g} {unit_cbar} | Desv: {vstd_calc:.4g} {unit_cbar}"
        )

        # Rango de visualización (Escala)
        smode = self.combo_master_scale_mode.currentData()
        if smode == 'robust':
            vmin = float(np.nanpercentile(data_to_plot, 2))
            vmax = float(np.nanpercentile(data_to_plot, 98))
        elif smode == 'strict':
            vmin = float(np.nanpercentile(data_to_plot, 1))
            vmax = float(np.nanpercentile(data_to_plot, 99))
        elif smode == 'manual':
            vmin = self.spin_master_vmin.value()
            vmax = self.spin_master_vmax.value()
            if vmin >= vmax: vmax = vmin + 1e-4
        else:
            vmin, vmax = vmin_calc, vmax_calc

        # Actualizar spinboxes manuales sin activar bucle
        self.spin_master_vmin.blockSignals(True); self.spin_master_vmax.blockSignals(True)
        self.spin_master_vmin.setValue(vmin); self.spin_master_vmax.setValue(vmax)
        self.spin_master_vmin.blockSignals(False); self.spin_master_vmax.blockSignals(False)

        # Configurar ejes con unidades
        if is_freq and self.u_vec is not None and self.v_vec is not None:
            extent = [self.u_vec[0], self.u_vec[-1], self.v_vec[0], self.v_vec[-1]]
            setup_ax_style(ax, title_str, f"Frecuencia Espacial U [ciclos/{u}]", f"Frecuencia Espacial V [ciclos/{u}]")
        else:
            extent = extent_space
            setup_ax_style(ax, f"{title_str} [{nx}×{ny}]", f"X ({self.x_col}) [{u}]", f"Y ({self.y_col}) [{u}]")

        if is_freq:
            ax.set_aspect('equal', adjustable='box')
        elif getattr(self, 'chk_master_aspect_equal', None) and self.chk_master_aspect_equal.isChecked():
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_aspect('auto')
        ax.set_anchor('C')

        im = ax.imshow(data_to_plot, origin='lower', extent=extent, cmap=cmap_name, vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.12)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"{unit_cbar}", color='#000000', fontsize=8)

        # Superponer Curvas de Nivel
        if self.chk_master_contours.isChecked():
            n_c = self.spin_master_contours_n.value()
            if is_freq and self.u_vec is not None:
                gx, gy = self.U_grid, self.V_grid
            else:
                gx, gy = np.meshgrid(x_grid, y_grid)
            try:
                cs = ax.contour(gx, gy, data_to_plot, levels=n_c, colors='#000000', alpha=0.55, linewidths=0.9)
                if self.chk_master_clabels.isChecked():
                    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
            except Exception:
                pass

        # Superponer puntos crudos únicamente dentro del sector donde se generó la malla
        if not is_freq and self.chk_master_overlay_pts.isChecked() and self.pts_x is not None and self.pts_y is not None:
            x_min_mesh = min(float(x_grid[0]), float(x_grid[-1]))
            x_max_mesh = max(float(x_grid[0]), float(x_grid[-1]))
            y_min_mesh = min(float(y_grid[0]), float(y_grid[-1]))
            y_max_mesh = max(float(y_grid[0]), float(y_grid[-1]))

            mask_in_mesh = (
                (self.pts_x >= x_min_mesh) & (self.pts_x <= x_max_mesh) &
                (self.pts_y >= y_min_mesh) & (self.pts_y <= y_max_mesh)
            )
            pts_mesh_x = self.pts_x[mask_in_mesh]
            pts_mesh_y = self.pts_y[mask_in_mesh]

            if len(pts_mesh_x) > 0:
                step = max(1, len(pts_mesh_x) // 10000)
                ax.scatter(
                    pts_mesh_x[::step], pts_mesh_y[::step],
                    c='#000000', edgecolors='#ffffff', linewidths=0.5,
                    s=14, alpha=0.85, zorder=6, label=f"Puntos en Malla ({len(pts_mesh_x):,})"
                )

        # Clamping estricto de ejes al bounding box de la malla o frecuencias
        if not is_freq:
            ax.set_xlim(extent_space[0], extent_space[1])
            ax.set_ylim(extent_space[2], extent_space[3])
        elif is_freq and self.u_vec is not None and self.v_vec is not None:
            ax.set_xlim(self.u_vec[0], self.u_vec[-1])
            ax.set_ylim(self.v_vec[0], self.v_vec[-1])

        self.master_active_data = data_to_plot
        self.master_active_extent = extent
        self.master_active_is_freq = is_freq
        self.master_active_unit = unit_cbar
        self.master_active_title = title_str

        fig.tight_layout()
        self.canvas_view.draw()

    # ==========================================================================
    # CÁLCULO FOURIER 2D & PILA DE FILTROS (CON UNIDADES EN TODOS LOS EJES)
    # ==========================================================================
    def recompute_fourier_2d(self):
        if self.grid_Z is None or self.grid_x_vec is None or self.grid_y_vec is None:
            return

        ny, nx = self.grid_Z.shape
        dx = self.dx_val
        dy = self.dy_val

        # Acondicionamiento de bordes perimetrales (Taper 2D previo a Fourier)
        if hasattr(self, 'chk_fourier_taper') and self.chk_fourier_taper.isChecked():
            t_target = self.combo_fourier_taper_target.currentData()
            t_type = self.combo_fourier_taper_type.currentData()
            t_pct = self.spin_fourier_taper_pct.value()
            z_space = apply_2d_edge_taper(self.grid_Z, target_mode=t_target, width_pct=t_pct, taper_type=t_type)
            self.grid_Z_tapered = z_space.copy()
        else:
            z_space = self.grid_Z.copy()
            self.grid_Z_tapered = None

        if self.chk_fourier_sub_mean.isChecked():
            z_space -= np.mean(z_space)

        self.F_2d = np.fft.fft2(z_space)
        self.F_shift_2d = np.fft.fftshift(self.F_2d)

        self.u_vec = (np.arange(nx) - (nx // 2)) * (1.0 / (nx * dx))
        self.v_vec = (np.arange(ny) - (ny // 2)) * (1.0 / (ny * dy))
        self.U_grid, self.V_grid = np.meshgrid(self.u_vec, self.v_vec)
        self.K_grid = np.sqrt(self.U_grid**2 + self.V_grid**2)
        self.Theta_grid = np.arctan2(self.V_grid, self.U_grid)

        self.plot_fourier_dashboard()

    def plot_fourier_dashboard(self):
        fig = self.canvas_fourier.fig
        fig.clf()
        if self.F_shift_2d is None or self.grid_Z is None:
            self.canvas_fourier.draw()
            return

        ny, nx = self.grid_Z.shape
        U, V = self.U_grid, self.V_grid
        K, Theta = self.K_grid, self.Theta_grid
        u = self.unit_xy
        z = self.unit_z

        # 1. EVALUAR PILA DE FILTROS SUPERPUESTOS
        active_filters = [f for f in self.filtros_2d if f.is_enabled]
        if self.chk_cascade_mult.isChecked():
            H_comp = np.ones((ny, nx), dtype=complex)
            for f in active_filters:
                H_comp *= f.evaluate_2d(U, V, K, Theta)
        else:
            H_comp = np.zeros((ny, nx), dtype=complex)
            if active_filters:
                for f in active_filters:
                    H_comp += f.evaluate_2d(U, V, K, Theta)
                H_comp /= len(active_filters)
            else:
                H_comp = np.ones((ny, nx), dtype=complex)

        self.H_total_2d = H_comp

        # Reconstrucción espacial Z_filt
        self.F_shift_filt = self.F_shift_2d * H_comp
        Z_recon = np.real(np.fft.ifft2(np.fft.ifftshift(self.F_shift_filt)))
        if self.chk_fourier_sub_mean.isChecked():
            base_mean = np.mean(self.grid_Z_tapered) if self.grid_Z_tapered is not None else np.mean(self.grid_Z)
            Z_recon += base_mean
        self.grid_Z_filt = Z_recon

        # Sincronizar filtro con Fuente Equivalente y Modelador si están presentes
        self.update_equivalent_source_filtered()
        if hasattr(self, 'widget_synth') and hasattr(self.widget_synth, 'refresh_available_obs_grids'):
            self.widget_synth.refresh_available_obs_grids()

        # 2. CONFIGURAR LOS 4 CUADRANTES DE VISUALIZACIÓN 2D CON UNIDADES
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_filt = fig.add_subplot(gs[0, 1])
        ax_h = fig.add_subplot(gs[1, 0])
        ax_spec = fig.add_subplot(gs[1, 1])

        extent_space = [self.grid_x_vec[0], self.grid_x_vec[-1], self.grid_y_vec[0], self.grid_y_vec[-1]]
        extent_freq = [self.u_vec[0], self.u_vec[-1], self.v_vec[0], self.v_vec[-1]]

        # --- CUADRANTE 1 (Arriba Izquierda): MALLA DE ENTRADA ESPACIAL (CON O SIN TAPER) ---
        if self.grid_Z_tapered is not None:
            raw_t_txt = self.combo_fourier_taper_target.currentText()
            t_mode_txt = re.sub(r'^[^\w]+', '', raw_t_txt).split("(")[0].strip()
            setup_ax_style(ax_orig, f"Malla Entrada FFT ({t_mode_txt}): {self.z_col}", f"X [{u}]", f"Y [{u}]")
            im_o = ax_orig.imshow(self.grid_Z_tapered, origin='lower', extent=extent_space, cmap='jet')
        else:
            setup_ax_style(ax_orig, f"Malla Original: {self.z_col} [Sin Filtros]", f"X [{u}]", f"Y [{u}]")
            im_o = ax_orig.imshow(self.grid_Z, origin='lower', extent=extent_space, cmap='jet')
        if getattr(self, 'chk_fourier_aspect_equal', None) and self.chk_fourier_aspect_equal.isChecked():
            ax_orig.set_aspect('equal', adjustable='box')
            ax_orig.set_anchor('C')
            div_o = make_axes_locatable(ax_orig)
            cax_o = div_o.append_axes("right", size="3.5%", pad=0.1)
            cb_o = fig.colorbar(im_o, cax=cax_o)
        else:
            ax_orig.set_aspect('auto')
            cb_o = fig.colorbar(im_o, ax=ax_orig, pad=0.02)
        cb_o.set_label(f"{self.z_col} [{z}]", color='#000000', fontsize=8)

        # --- CUADRANTE 2 (Arriba Derecha): MALLA FILTRADA / RESIDUAL / TAPER CON UNIDADES ---
        layer_mode = self.combo_fourier_map_layer.currentData()
        if layer_mode == 'orig':
            data_map = self.grid_Z
            map_title = "Malla Original (Z_orig)"
        elif layer_mode == 'taper':
            data_map = self.grid_Z_tapered if self.grid_Z_tapered is not None else self.grid_Z
            raw_t_txt = self.combo_fourier_taper_target.currentText() if hasattr(self, 'combo_fourier_taper_target') else ""
            t_tgt = re.sub(r'^[^\w]+', '', raw_t_txt).split("(")[0].strip()
            map_title = f"Malla Acondicionada ({t_tgt}): Z_taper"
        elif layer_mode == 'diff':
            data_map = self.grid_Z - self.grid_Z_filt
            map_title = "Diferencia Residual (Z_orig - Z_filt)"
        else:
            data_map = self.grid_Z_filt
            n_act = len(active_filters)
            status_act = f"[{n_act} filtros activos]" if n_act > 0 else "[Sin filtros]"
            map_title = f"Malla Filtrada 2D {status_act}"

        setup_ax_style(ax_filt, map_title, f"X [{u}]", f"Y [{u}]")
        im_f = ax_filt.imshow(data_map, origin='lower', extent=extent_space, cmap='jet')
        if getattr(self, 'chk_fourier_aspect_equal', None) and self.chk_fourier_aspect_equal.isChecked():
            ax_filt.set_aspect('equal', adjustable='box')
            ax_filt.set_anchor('C')
            div_f = make_axes_locatable(ax_filt)
            cax_f = div_f.append_axes("right", size="3.5%", pad=0.1)
            cb_f = fig.colorbar(im_f, cax=cax_f)
        else:
            ax_filt.set_aspect('auto')
            cb_f = fig.colorbar(im_f, ax=ax_filt, pad=0.02)
        cb_f.set_label(f"{self.z_col} [{z}]", color='#000000', fontsize=8)

        # --- CUADRANTE 3 (Abajo Izquierda): FILTRO TOTAL RESULTANTE |H_total(u,v)| CON UNIDADES ---
        H_abs = np.abs(H_comp)
        show_wavelength = getattr(self, 'chk_show_wavelength', None) and self.chk_show_wavelength.isChecked()

        setup_ax_style(ax_h, "Filtro Aplicado Total: |H_total(u,v)|", f"Frecuencia Espacial U [ciclos/{u}]", f"Frecuencia Espacial V [ciclos/{u}]")
        ax_h.set_aspect('equal', adjustable='box')
        ax_h.set_anchor('C')
        im_h = ax_h.imshow(H_abs, origin='lower', extent=extent_freq, cmap='jet')
        div_h = make_axes_locatable(ax_h)
        cax_h = div_h.append_axes("right", size="3.5%", pad=0.1)
        cb_h = fig.colorbar(im_h, cax=cax_h)
        cb_h.set_label("Ganancia |H| [adimensional]", color='#000000', fontsize=8)

        # Frecuencias de corte características de filtros activos
        cutoffs = []
        for f in active_filters:
            if getattr(f, 'filter_type', '') == 'window' or getattr(f, 'math_apply_band', False):
                if getattr(f, 'k_low', 0.0) > 1e-9: cutoffs.append((float(f.k_low), f.name))
                if getattr(f, 'k_high', 0.0) > 1e-9: cutoffs.append((float(f.k_high), f.name))
            elif getattr(f, 'filter_type', '') == 'math':
                if getattr(f, 'k_high', 0.0) > 1e-9: cutoffs.append((float(f.k_high), f.name))
                elif getattr(f, 'k_low', 0.0) > 1e-9: cutoffs.append((float(f.k_low), f.name))

        max_freq = min(float(np.max(np.abs(self.u_vec))), float(np.max(np.abs(self.v_vec))))
        if not cutoffs and max_freq > 0 and show_wavelength:
            for mult in [0.25, 0.50, 0.75]:
                cutoffs.append((max_freq * mult, f"Ref {int(mult*100)}%"))

        # Dibujar anillos de frecuencia de corte en H(u,v) (líneas limpias sin cajas flotantes)
        for kc, fname in cutoffs[:4]:
            if 1e-9 < kc <= max_freq:
                c = patches.Circle((0, 0), kc, fill=False, edgecolor='#ffffff', linestyle='--', linewidth=1.0, alpha=0.65)
                ax_h.add_patch(c)

        if show_wavelength:
            setup_dual_frequency_wavelength_axis(ax_h, u)
            ax_h.set_xlabel(f"Frecuencia Espacial U [ciclos/{u}]", color='#000000', fontsize=8, labelpad=22)

        # --- CUADRANTE 4 (Abajo Derecha): ESPECTRO 2D O RAPS COMPARATIVO CON UNIDADES ---
        spec_mode = self.combo_spec_plot_mode.currentData()
        if spec_mode == 'spec2d':
            espectro_log = np.log10(np.abs(self.F_shift_filt) + 1.0)
            if self.chk_fourier_omit_dc.isChecked():
                espectro_vis = espectro_log.copy()
                espectro_vis[ny//2, nx//2] = np.median(espectro_log)
            else:
                espectro_vis = espectro_log
            setup_ax_style(ax_spec, "Espectro de Amplitud 2D Log10 (Filtrado)", f"Frecuencia Espacial U [ciclos/{u}]", f"Frecuencia Espacial V [ciclos/{u}]")
            ax_spec.set_aspect('equal', adjustable='box')
            ax_spec.set_anchor('C')
            im_s = ax_spec.imshow(espectro_vis, origin='lower', extent=extent_freq, cmap='jet')
            div_s = make_axes_locatable(ax_spec)
            cax_s = div_s.append_axes("right", size="3.5%", pad=0.1)
            cb_s = fig.colorbar(im_s, cax=cax_s)
            cb_s.set_label(f"log10(|F|) [{z}·{u}²]", color='#000000', fontsize=8)
            if show_wavelength:
                setup_dual_frequency_wavelength_axis(ax_spec, u)
                ax_spec.set_xlabel(f"Frecuencia Espacial U [ciclos/{u}]", color='#000000', fontsize=8, labelpad=22)
        else:
            # RAPS Comparativo con ejes SIEMPRE en función de k_r [ciclos/u]
            # En el pie del gráfico cartesiano se muestra en negro la frecuencia y debajo en rojo la longitud de onda λ
            raps_scale = self.combo_raps_scale.currentData() if hasattr(self, 'combo_raps_scale') else 'ln'
            kr_flat = K.ravel()
            p_orig_flat = (np.abs(self.F_shift_2d)**2).ravel()
            p_filt_flat = (np.abs(self.F_shift_filt)**2).ravel()

            max_kr = min(np.max(np.abs(self.u_vec)), np.max(np.abs(self.v_vec)))
            k_bins = np.linspace(0.0, max_kr, 50)
            k_cents = 0.5 * (k_bins[1:] + k_bins[:-1])

            p_orig_means = []
            p_filt_means = []
            for i in range(len(k_bins)-1):
                mask = (kr_flat >= k_bins[i]) & (kr_flat < k_bins[i+1])
                if np.any(mask):
                    p_orig_means.append(float(np.mean(p_orig_flat[mask])))
                    p_filt_means.append(float(np.mean(p_filt_flat[mask])))
                else:
                    p_orig_means.append(1e-12)
                    p_filt_means.append(1e-12)

            p_orig_arr = np.array(p_orig_means)
            p_filt_arr = np.array(p_filt_means)

            if raps_scale == 'linear':
                raps_y_lbl = f"Potencia Espectral P [{z}²·{u}²]"
                y_orig = p_orig_arr
                y_filt = p_filt_arr
                lbl_orig = "P(k) Original"
                lbl_filt = "P(k) Filtrado"
            elif raps_scale == 'ratio':
                raps_y_lbl = "Ganancia Espectral |H(k_r)|² [adimensional]"
                y_orig = np.ones_like(p_orig_arr)
                y_filt = p_filt_arr / np.maximum(1e-15, p_orig_arr)
                lbl_orig = "|H|² Base = 1.0"
                lbl_filt = "|H(k_r)|² = P_filt / P_orig"
            else: # 'ln'
                raps_y_lbl = f"ln(Potencia Espectral) [{z}²·{u}²]"
                y_orig = np.log(np.maximum(1e-12, p_orig_arr))
                y_filt = np.log(np.maximum(1e-12, p_filt_arr))
                lbl_orig = "ln(P) Original"
                lbl_filt = "ln(P) Filtrado"

            scale_tag = "Lineal" if raps_scale == 'linear' else ("Ganancia |H|²" if raps_scale == 'ratio' else "ln(P)")

            setup_ax_style(ax_spec, f"Espectro Radial de Potencia (RAPS) [{scale_tag}]", f"Número de Onda Radial k_r [ciclos/{u}]", raps_y_lbl)
            ax_spec.plot(k_cents, y_orig, color='#777777', linestyle='--', linewidth=1.5, marker='o', markersize=3, label=lbl_orig)
            ax_spec.plot(k_cents, y_filt, color='#cc0000', linestyle='-', linewidth=2.0, marker='s', markersize=3, label=lbl_filt)
            ax_spec.legend(facecolor='#ffffff', edgecolor='#808080', labelcolor='#000000', loc='best', fontsize=8)

            # Líneas guía de cortes de filtros (limpias, sin cajas flotantes)
            for idx, (kc, fname) in enumerate(cutoffs[:4]):
                if 1e-9 < kc <= max_kr:
                    ax_spec.axvline(kc, color='#cc0000', linestyle=':', linewidth=1.0, alpha=0.6)

            # Configuración limpia del pie cartesiano con f en negro y λ en rojo
            ax_spec.set_xlim(left=0.0, right=max_kr)
            if show_wavelength:
                setup_dual_frequency_wavelength_axis(ax_spec, u)
                ax_spec.set_xlabel(f"Número de Onda Radial k_r [ciclos/{u}]", color='#000000', fontsize=8, labelpad=24)

        try:
            fig.subplots_adjust(left=0.08, right=0.92, bottom=0.14, top=0.94, wspace=0.38, hspace=0.38)
        except Exception:
            fig.tight_layout()
        self.canvas_fourier.draw()

    # ==========================================================================
    # GESTIÓN DE PILA DE FILTROS 2D (SOLO 1 EJEMPLO POR TIPO)
    # ==========================================================================
    def refresh_filter_table(self):
        self.tbl_filters.blockSignals(True)
        self.tbl_filters.setRowCount(len(self.filtros_2d))

        for row, f in enumerate(self.filtros_2d):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Checked if f.is_enabled else Qt.Unchecked)
            self.tbl_filters.setItem(row, 0, chk_item)

            name_item = QTableWidgetItem(f.name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_filters.setItem(row, 1, name_item)

            summary_item = QTableWidgetItem(f.get_summary_text())
            summary_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_filters.setItem(row, 2, summary_item)

        self.tbl_filters.blockSignals(False)

    def on_filter_table_item_changed(self, item):
        if item.column() == 0:
            row = item.row()
            if 0 <= row < len(self.filtros_2d):
                self.filtros_2d[row].is_enabled = (item.checkState() == Qt.Checked)
                self.plot_fourier_dashboard()
                self.plot_master_viewer()

    def on_add_filter_clicked(self):
        max_kr = 0.2
        if self.u_vec is not None and self.v_vec is not None:
            max_kr = max(float(np.max(np.abs(self.u_vec))), float(np.max(np.abs(self.v_vec))))
        d_cell_approx = 0.5 / max(1e-7, max_kr)
        h_up = round(max(5.0, 2.0 * d_cell_approx), 1)
        nuevo = Filtro2D(name=f"Filtro {len(self.filtros_2d) + 1}", filter_type="math", formula_str=f"exp(-2 * pi * {h_up:g} * k)", is_enabled=True)
        dlg = Filtro2DDialog(nuevo, parent=self, max_kr_hint=max_kr, unit_xy=self.unit_xy, unit_z=self.unit_z)
        if dlg.exec_() == QDialog.Accepted:
            self.filtros_2d.append(nuevo)
            self.refresh_filter_table()
            self.plot_fourier_dashboard()
            self.plot_master_viewer()

    def show_presets_menu(self):
        menu = QMenu(self)
        u = self.unit_xy

        # Obtener espaciado y frecuencia de Nyquist real de la malla activa
        dx_eff = self.dx_val if (hasattr(self, 'dx_val') and self.dx_val is not None and self.dx_val > 0) else 100.0
        dy_eff = self.dy_val if (hasattr(self, 'dy_val') and self.dy_val is not None and self.dy_val > 0) else 100.0
        d_cell = max(dx_eff, dy_eff)
        k_nyq = 0.5 / max(1e-6, d_cell)

        # Frecuencias de corte adaptadas a la resolución de la malla actual
        kc_lp = round(k_nyq * 0.30, 6)
        kc_hp = round(k_nyq * 0.15, 6)
        kc_gauss = round(k_nyq * 0.35, 6)
        kc_taper = round(k_nyq * 0.75, 6)
        h_up = round(max(10.0, 2.0 * d_cell), 1)
        h_down = round(max(5.0, 0.5 * d_cell), 1)

        # EXACTAMENTE UN SOLO EJEMPLO POR CADA FAMILIA DE FILTRO (CALIBRADO A LA MALLA)
        presets = [
            (f"Upward Continuation (+{h_up:g} {u})", "math", f"exp(-2 * pi * {h_up:g} * k)"),
            (f"Downward Continuation (-{h_down:g} {u})", "math", f"exp(2 * pi * {h_down:g} * k)"),
            ("Primera Derivada Vertical (1VD: d/dz)", "math", "2 * pi * k"),
            ("Segunda Derivada Vertical (2VD: d²z/dz²)", "math", "(2 * pi * k)**2"),
            ("Derivada Horizontal X (d/dx)", "math", "2 * pi * 1j * u"),
            ("Derivada Horizontal Y (d/dy)", "math", "2 * pi * 1j * v"),
            (f"Paso Bajo Butterworth 2D (kc={kc_lp:g})", "math", f"1 / (1 + (k / {kc_lp:g})**8)"),
            (f"Paso Alto Butterworth 2D (kc={kc_hp:g})", "math", f"(k / {kc_hp:g})**8 / (1 + (k / {kc_hp:g})**8)"),
            (f"Filtro Gaussiano (kc={kc_gauss:g})", "math", f"exp(-(k / {kc_gauss:g})**2)"),
            ("Filtro Notch Direccional 45°", "math", "1 - exp(-((abs(theta - pi/4)) / (15 * pi / 180))**2)"),
            (f"Atenuación Lineal Taper (kc={kc_taper:g})", "math", f"max(0, 1 - k / {kc_taper:g})"),
            ("Ventana Radial con Colas", "window", ""),
        ]
        for title, ftype, form in presets:
            act = menu.addAction(title)
            act.triggered.connect(lambda checked, t=title, ft=ftype, f_str=form: self.add_preset_filter(t, ft, f_str))

        sender = self.sender()
        if sender:
            menu.exec_(sender.mapToGlobal(QPoint(0, sender.height())))

    def add_preset_filter(self, title, ftype, formula_str):
        clean_title = re.sub(r'^[^\w]+', '', title).strip()
        f = Filtro2D(name=clean_title, filter_type=ftype, formula_str=formula_str, is_enabled=True)
        if ftype == 'window':
            max_kr = 0.05
            if self.u_vec is not None:
                max_kr = float(min(np.max(np.abs(self.u_vec)), np.max(np.abs(self.v_vec))))
            f.k_low = 0.0
            f.k_high = max_kr * 0.4
            f.g_in = 1.0
            f.g_out = 0.0
            f.tail_low_type = 'smooth'
            f.tail_low_w = 0.0
            f.tail_high_type = 'smooth'
            f.tail_high_w = max_kr * 0.1
        self.filtros_2d.append(f)
        self.refresh_filter_table()
        self.plot_fourier_dashboard()
        self.plot_master_viewer()

    def edit_selected_filter(self):
        row = self.tbl_filters.currentRow()
        if 0 <= row < len(self.filtros_2d):
            max_kr = 0.2
            if self.u_vec is not None:
                max_kr = max(float(np.max(np.abs(self.u_vec))), float(np.max(np.abs(self.v_vec))))
            f = self.filtros_2d[row]
            dlg = Filtro2DDialog(f, parent=self, max_kr_hint=max_kr, unit_xy=self.unit_xy, unit_z=self.unit_z)
            if dlg.exec_() == QDialog.Accepted:
                self.refresh_filter_table()
                self.plot_fourier_dashboard()
                self.plot_master_viewer()

    def duplicate_selected_filter(self):
        row = self.tbl_filters.currentRow()
        if 0 <= row < len(self.filtros_2d):
            orig = self.filtros_2d[row]
            clone = Filtro2D(
                name=f"{orig.name} (Copia)", filter_type=orig.filter_type,
                formula_str=orig.formula_str, math_apply_band=orig.math_apply_band,
                k_low=orig.k_low, k_high=orig.k_high, g_in=orig.g_in, g_out=orig.g_out,
                tail_low_type=orig.tail_low_type, tail_low_w=orig.tail_low_w,
                tail_high_type=orig.tail_high_type, tail_high_w=orig.tail_high_w,
                is_enabled=orig.is_enabled, color=orig.color
            )
            self.filtros_2d.insert(row + 1, clone)
            self.refresh_filter_table()
            self.plot_fourier_dashboard()
            self.plot_master_viewer()

    def delete_selected_filter(self):
        row = self.tbl_filters.currentRow()
        if 0 <= row < len(self.filtros_2d):
            self.filtros_2d.pop(row)
            self.refresh_filter_table()
            self.plot_fourier_dashboard()
            self.plot_master_viewer()

    def clear_all_filters(self):
        if self.filtros_2d:
            self.filtros_2d.clear()
            self.refresh_filter_table()
            self.plot_fourier_dashboard()
            self.plot_master_viewer()

    def move_selected_filter_up(self):
        row = self.tbl_filters.currentRow()
        if row > 0:
            self.filtros_2d[row], self.filtros_2d[row - 1] = self.filtros_2d[row - 1], self.filtros_2d[row]
            self.refresh_filter_table()
            self.tbl_filters.selectRow(row - 1)
            self.plot_fourier_dashboard()
            self.plot_master_viewer()

    def move_selected_filter_down(self):
        row = self.tbl_filters.currentRow()
        if 0 <= row < len(self.filtros_2d) - 1:
            self.filtros_2d[row], self.filtros_2d[row + 1] = self.filtros_2d[row + 1], self.filtros_2d[row]
            self.refresh_filter_table()
            self.tbl_filters.selectRow(row + 1)
            self.plot_fourier_dashboard()
            self.plot_master_viewer()

    def on_filter_table_context_menu(self, pos):
        menu = QMenu(self)
        act_add = menu.addAction("Agregar Filtro...")
        act_add.triggered.connect(self.on_add_filter_clicked)

        row = self.tbl_filters.currentRow()
        if 0 <= row < len(self.filtros_2d):
            menu.addSeparator()
            act_edit = menu.addAction("Editar Filtro")
            act_edit.triggered.connect(self.edit_selected_filter)
            act_dup = menu.addAction("Duplicar Filtro")
            act_dup.triggered.connect(self.duplicate_selected_filter)
            act_del = menu.addAction("Eliminar Filtro")
            act_del.triggered.connect(self.delete_selected_filter)

        menu.addSeparator()
        act_clear = menu.addAction("Limpiar Todos los Filtros")
        act_clear.triggered.connect(self.clear_all_filters)
        menu.exec_(self.tbl_filters.viewport().mapToGlobal(pos))

    def on_filter_double_clicked(self, item):
        self.edit_selected_filter()

    def export_filtered_grid_quick(self):
        if self.grid_Z_filt is None:
            QMessageBox.warning(self, "Sin Malla Filtrada", "Aún no se ha calculado la malla filtrada.")
            return

        fpath, _ = QFileDialog.getSaveFileName(
            self, "Exportar Malla Filtrada", os.path.join(DESCARGAS_DIR, "malla_filtrada_fourier2D.tif"),
            "Raster TIFF (*.tif *.tiff);;CSV XYZ (*.csv);;Matriz CSV (*.csv);;Surfer Grid (*.grd)"
        )
        if not fpath:
            return

        ext = os.path.splitext(fpath)[1].lower()
        if ext in ('.tif', '.tiff'):
            self._write_tiff_grid(fpath, self.grid_Z_filt)
        elif ext == '.grd':
            self._write_surfer_grd(fpath, self.grid_Z_filt)
        else:
            xi, yi = np.meshgrid(self.grid_x_vec, self.grid_y_vec)
            df = pd.DataFrame({'X': xi.ravel(), 'Y': yi.ravel(), 'Z': self.grid_Z_filt.ravel()})
            df.to_csv(fpath, index=False)

        QMessageBox.information(self, "Exportación Exitosa", f"Malla filtrada exportada correctamente en:\n{fpath}")

    # ==========================================================================
    # PESTAÑA 5: TABLA & EXPORTACIÓN CON UNIDADES
    # ==========================================================================
    def update_mesh_table(self):
        if hasattr(self, 'rad_exp_taper') and self.rad_exp_taper.isChecked():
            Z = self.grid_Z_tapered if self.grid_Z_tapered is not None else self.grid_Z
            source_name = "Malla con Taper Bordes (Fourier)"
        elif self.rad_exp_filt.isChecked() and self.grid_Z_filt is not None:
            Z = self.grid_Z_filt
            source_name = "Malla Filtrada (Fourier 2D)"
        else:
            Z = self.grid_Z
            source_name = "Malla Original"

        if Z is None or self.grid_x_vec is None or self.grid_y_vec is None:
            return

        xi, yi = np.meshgrid(self.grid_x_vec, self.grid_y_vec)
        x_flat = xi.ravel()
        y_flat = yi.ravel()
        z_flat = Z.ravel()
        n_tot = len(x_flat)
        n_show = min(100, n_tot)

        self.lbl_table_info.setText(f"Mostrando primeros {n_show} de {n_tot:,} nodos [{source_name}]:")
        self.table_mesh_nodes.setRowCount(n_show)
        self.table_mesh_nodes.setHorizontalHeaderLabels([
            f"Coordenada X [{self.unit_xy}]", f"Coordenada Y [{self.unit_xy}]", f"Valor {self.z_col} [{self.unit_z}]"
        ])

        for i in range(n_show):
            self.table_mesh_nodes.setItem(i, 0, QTableWidgetItem(f"{x_flat[i]:.5f}"))
            self.table_mesh_nodes.setItem(i, 1, QTableWidgetItem(f"{y_flat[i]:.5f}"))
            self.table_mesh_nodes.setItem(i, 2, QTableWidgetItem(f"{z_flat[i]:.5f}"))

    def _get_active_export_grid(self):
        if hasattr(self, 'rad_exp_taper') and self.rad_exp_taper.isChecked():
            return (self.grid_Z_tapered if self.grid_Z_tapered is not None else self.grid_Z), "tapered"
        elif self.rad_exp_filt.isChecked() and self.grid_Z_filt is not None:
            return self.grid_Z_filt, "filtrada"
        return self.grid_Z, "original"

    def export_csv_xyz(self):
        Z, tag = self._get_active_export_grid()
        if Z is None or self.grid_x_vec is None or self.grid_y_vec is None:
            QMessageBox.warning(self, "Sin Malla", "No hay malla regular generada para exportar.")
            return

        fpath, _ = QFileDialog.getSaveFileName(self, "Guardar Malla CSV (XYZ)", os.path.join(DESCARGAS_DIR, f"malla_2d_{tag}_xyz.csv"), "CSV (*.csv)")
        if fpath:
            xi, yi = np.meshgrid(self.grid_x_vec, self.grid_y_vec)
            col_x = f"X_{self.unit_xy}"
            col_y = f"Y_{self.unit_xy}"
            col_z = f"{self.z_col}_{self.unit_z}"
            df = pd.DataFrame({col_x: xi.ravel(), col_y: yi.ravel(), col_z: Z.ravel()})
            df.to_csv(fpath, index=False)
            QMessageBox.information(self, "Exportación Exitosa", f"Malla XYZ guardada en:\n{fpath}")

    def export_csv_matrix(self):
        Z, tag = self._get_active_export_grid()
        if Z is None:
            QMessageBox.warning(self, "Sin Malla", "No hay malla regular generada para exportar.")
            return

        fpath, _ = QFileDialog.getSaveFileName(self, "Guardar Matriz CSV", os.path.join(DESCARGAS_DIR, f"malla_2d_{tag}_matrix.csv"), "CSV (*.csv)")
        if fpath:
            df = pd.DataFrame(Z, index=self.grid_y_vec, columns=self.grid_x_vec)
            df.to_csv(fpath)
            QMessageBox.information(self, "Exportación Exitosa", f"Matriz guardada en:\n{fpath}")

    def export_csv_blakely(self):
        if self.F_shift_2d is None or self.u_vec is None or self.v_vec is None:
            QMessageBox.warning(self, "Sin Fourier", "Primero debes calcular la FFT 2D en la Pestaña 4.")
            return

        fpath, _ = QFileDialog.getSaveFileName(self, "Guardar CSV Fourier Blakely", os.path.join(DESCARGAS_DIR, "fourier_reconstruccion.csv"), "CSV (*.csv)")
        if fpath:
            F = self.F_shift_filt if (self.rad_exp_filt.isChecked() and self.F_shift_filt is not None) else self.F_shift_2d
            ny, nx = F.shape
            u_flat = self.U_grid.ravel()
            v_flat = self.V_grid.ravel()
            f_flat = F.ravel()

            df = pd.DataFrame({
                f'U_ciclos_por_{self.unit_xy}': u_flat,
                f'V_ciclos_por_{self.unit_xy}': v_flat,
                'Real': np.real(f_flat),
                'Imag': np.imag(f_flat),
                f'dx_{self.unit_xy}': np.full_like(u_flat, self.dx_val),
                f'dy_{self.unit_xy}': np.full_like(u_flat, self.dy_val),
                'x0': np.full_like(u_flat, self.grid_x_vec[0]),
                'y0': np.full_like(u_flat, self.grid_y_vec[0]),
                'rows': np.full_like(u_flat, ny),
                'cols': np.full_like(u_flat, nx),
            })
            df.to_csv(fpath, index=False)
            QMessageBox.information(self, "Exportación Exitosa", f"Datos Fourier guardados en:\n{fpath}")

    def _write_surfer_grd(self, filepath, Z):
        ny, nx = Z.shape
        x_min, x_max = float(self.grid_x_vec[0]), float(self.grid_x_vec[-1])
        y_min, y_max = float(self.grid_y_vec[0]), float(self.grid_y_vec[-1])
        z_min, z_max = float(np.min(Z)), float(np.max(Z))

        with open(filepath, 'w', encoding='ascii') as f:
            f.write("DSAA\n")
            f.write(f"{nx} {ny}\n")
            f.write(f"{x_min:.6f} {x_max:.6f}\n")
            f.write(f"{y_min:.6f} {y_max:.6f}\n")
            f.write(f"{z_min:.6f} {z_max:.6f}\n")
            for r in range(ny):
                row_vals = " ".join([f"{v:.6f}" for v in Z[r, :]])
                f.write(f"{row_vals}\n")

    def export_surfer_grid(self):
        Z, tag = self._get_active_export_grid()
        if Z is None:
            QMessageBox.warning(self, "Sin Malla", "No hay malla regular generada para exportar.")
            return

        fpath, _ = QFileDialog.getSaveFileName(self, "Guardar Surfer ASCII Grid (.grd)", os.path.join(DESCARGAS_DIR, f"malla_{tag}.grd"), "Surfer Grid (*.grd)")
        if fpath:
            self._write_surfer_grd(fpath, Z)
            QMessageBox.information(self, "Exportación Exitosa", f"Archivo Surfer Grid guardado en:\n{fpath}")

    def export_json_grid(self):
        Z, tag = self._get_active_export_grid()
        if Z is None:
            QMessageBox.warning(self, "Sin Malla", "No hay malla regular generada para exportar.")
            return

        fpath, _ = QFileDialog.getSaveFileName(self, "Guardar JSON Estructurado", os.path.join(DESCARGAS_DIR, f"malla_{tag}.json"), "JSON (*.json)")
        if fpath:
            ny, nx = Z.shape
            data = {
                "tag": tag,
                "nx": nx, "ny": ny,
                "dx": float(self.dx_val), "dy": float(self.dy_val),
                "unit_xy": self.unit_xy,
                "unit_z": self.unit_z,
                "x_min": float(self.grid_x_vec[0]), "x_max": float(self.grid_x_vec[-1]),
                "y_min": float(self.grid_y_vec[0]), "y_max": float(self.grid_y_vec[-1]),
                "x_coords": self.grid_x_vec.tolist(),
                "y_coords": self.grid_y_vec.tolist(),
                "Z_matrix": Z.tolist(),
                "attribute_x": self.x_col,
                "attribute_y": self.y_col,
                "attribute_z": self.z_col
            }
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Exportación Exitosa", f"Archivo JSON guardado en:\n{fpath}")

    def _write_tiff_grid(self, filepath, Z):
        """
        Exporta la matriz 2D como raster TIFF Float32 incorporando las coordenadas
        locales en los tags estándar ModelPixelScaleTag y ModelTiepointTag (sin CRS forzado).
        """
        # La fila 0 en TIFF corresponde al borde superior (Y_max / Norte), invertimos verticalmente
        Z_flip = np.flipud(Z).astype(np.float32)
        img = Image.fromarray(Z_flip, mode='F')

        info = TiffImagePlugin.ImageFileDirectory_v2()
        dx = float(self.dx_val)
        dy = float(self.dy_val)
        x_min = float(self.grid_x_vec[0])
        y_max = float(self.grid_y_vec[-1])

        # ModelPixelScaleTag (33550): dx, dy, dz
        info[33550] = (dx, dy, 0.0)
        # ModelTiepointTag (33922): I=0, J=0, K=0, X=x_min, Y=y_max, Z=0
        info[33922] = (0.0, 0.0, 0.0, x_min, y_max, 0.0)

        img.save(filepath, tiffinfo=info)

    def export_tiff_grid(self):
        Z, tag = self._get_active_export_grid()
        if Z is None or self.grid_x_vec is None or self.grid_y_vec is None:
            QMessageBox.warning(self, "Sin Malla", "No hay malla regular generada para exportar.")
            return

        fpath, _ = QFileDialog.getSaveFileName(
            self, "Guardar Raster TIFF (.tif)",
            os.path.join(DESCARGAS_DIR, f"malla_2d_{tag}.tif"),
            "Raster TIFF (*.tif *.tiff)"
        )
        if fpath:
            self._write_tiff_grid(fpath, Z)
            QMessageBox.information(
                self, "Exportación Exitosa",
                f"Raster TIFF guardado con éxito (Float32, resolución dx={self.dx_val:g}, dy={self.dy_val:g}) en:\n{fpath}"
            )

    def open_synthetic_modeler(self):
        """
        Abre el modelador sintético espectral 2D (Blakely 1995) conmutando a la Pestaña 2 dedicada.
        """
        if hasattr(self, 'tabs') and self.tabs is not None and self.tabs.count() > 1:
            self.tabs.setCurrentIndex(1)
        else:
            dlg = VentanaModeladorSintetico(self, parent=self)
            dlg.exec_()


    def open_model_comparison(self):
        """
        Abre la ventana de comparación y álgebra de mapas píxel a píxel (Obs vs. Mod).
        Muestra Observaciones y Modelo arriba, y el resultado de la operación f(Obs, Mod) abajo.
        """
        if self.grid_Z_model is None:
            resp = QMessageBox.question(
                self, "Sin Modelo Sintético",
                "Aún no has creado ni fijado un modelo sintético en memoria.\n\n"
                "¿Deseas abrir el Modelador de Anomalías Sintéticas ahora para crearlo?",
                QMessageBox.Yes | QMessageBox.No
            )
            if resp == QMessageBox.Yes:
                self.open_synthetic_modeler()
            return
        dlg = VentanaComparacionModelos(self, parent=self)
        dlg.exec_()

    def open_fuente_equivalente_window(self):
        """
        Abre la ventana dedicada al Problema Inverso de Fuente Equivalente
        para la continuación de campos gravimétricos sobre topografía irregular a un plano Z0.
        """
        self.dlg_problema_inverso = VentanaProblemaInverso(self, parent=self)
        if getattr(self, 'grid_mdt', None) is not None:
            self.dlg_problema_inverso.raw_mdt_Z = self.grid_mdt.copy()
            self.dlg_problema_inverso.raw_mdt_x = self.mdt_x_vec.copy() if self.mdt_x_vec is not None else None
            self.dlg_problema_inverso.raw_mdt_y = self.mdt_y_vec.copy() if self.mdt_y_vec is not None else None
            self.dlg_problema_inverso.match_mdt_to_gravity()
        self.dlg_problema_inverso.showMaximized()
        self.dlg_problema_inverso.exec_()
        self.update_equivalent_source_filtered()
        if hasattr(self, 'widget_synth') and hasattr(self.widget_synth, 'refresh_available_obs_grids'):
            self.widget_synth.refresh_available_obs_grids()

    def update_equivalent_source_filtered(self):
        """Calcula o actualiza la malla de Fuente Equivalente Filtrada por Fourier 2D."""
        if getattr(self, 'grid_eq_source_z0', None) is None:
            self.grid_eq_source_z0_filt = None
            return None

        # Si la malla activa principal es la de fuente equivalente, grid_Z_filt ya es la filtrada
        if getattr(self, 'grid_Z', None) is not None:
            if self.grid_Z.shape == self.grid_eq_source_z0.shape and np.allclose(self.grid_Z, self.grid_eq_source_z0, atol=1e-4):
                self.grid_eq_source_z0_filt = self.grid_Z_filt.copy() if self.grid_Z_filt is not None else self.grid_eq_source_z0.copy()
                return self.grid_eq_source_z0_filt

        # Si no, filtrar grid_eq_source_z0 usando la pila de filtros Fourier actual
        try:
            G_z0 = self.grid_eq_source_z0
            ny, nx = G_z0.shape
            dx = float(self.dx_val) if hasattr(self, 'dx_val') and self.dx_val > 0 else 100.0
            dy = float(self.dy_val) if hasattr(self, 'dy_val') and self.dy_val > 0 else 100.0

            active_filters = [f for f in getattr(self, 'filtros_2d', []) if getattr(f, 'is_enabled', True)]
            if not active_filters:
                self.grid_eq_source_z0_filt = G_z0.copy()
                return self.grid_eq_source_z0_filt

            u_vec = (np.arange(nx) - (nx // 2)) * (1.0 / (nx * dx))
            v_vec = (np.arange(ny) - (ny // 2)) * (1.0 / (ny * dy))
            U_grid, V_grid = np.meshgrid(u_vec, v_vec)
            K_grid = np.sqrt(U_grid**2 + V_grid**2)
            Theta_grid = np.arctan2(V_grid, U_grid)

            if getattr(self, 'chk_cascade_mult', None) and self.chk_cascade_mult.isChecked():
                H_comp = np.ones((ny, nx), dtype=complex)
                for f in active_filters:
                    H_comp *= f.evaluate_2d(U_grid, V_grid, K_grid, Theta_grid)
            else:
                H_comp = np.zeros((ny, nx), dtype=complex)
                for f in active_filters:
                    H_comp += f.evaluate_2d(U_grid, V_grid, K_grid, Theta_grid)
                H_comp /= len(active_filters)

            F_2d = np.fft.fft2(G_z0)
            F_shift = np.fft.fftshift(F_2d) * H_comp
            Z_filt = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift)))
            self.grid_eq_source_z0_filt = Z_filt
            return self.grid_eq_source_z0_filt
        except Exception as e:
            self.grid_eq_source_z0_filt = self.grid_eq_source_z0.copy()
            return self.grid_eq_source_z0_filt

    def get_available_modeling_grids(self):
        """
        Retorna diccionario unificado con las 4 mallas para Modelado y Método Inverso Puntual:
          1. 'sin_filtrar': Malla Original / Interpolada
          2. 'filtrada': Malla Filtrada Fourier 2D
          3. 'fuente_equivalente': Malla Fuente Equivalente Plano Z0
          4. 'fuente_equivalente_filtrada': Malla Fuente Equivalente Filtrada
        """
        grids = {}
        # 1. Sin filtrar
        z_orig = getattr(self, 'grid_Z_orig', getattr(self, 'grid_Z', None))
        if z_orig is not None and getattr(self, 'grid_x_vec', None) is not None and getattr(self, 'grid_y_vec', None) is not None:
            ny, nx = z_orig.shape
            grids['sin_filtrar'] = {
                'key': 'sin_filtrar',
                'label': f"1. Malla Sin Filtrar (Original / Interpolada) [{nx}×{ny}]",
                'Z': z_orig.copy(),
                'x': self.grid_x_vec.copy(),
                'y': self.grid_y_vec.copy(),
                'z_elev': 0.0,
                'name': 'Original / Sin Filtrar'
            }

        # 2. Filtrada Fourier
        z_filt = getattr(self, 'grid_Z_filt', None)
        if z_filt is not None and getattr(self, 'grid_x_vec', None) is not None and getattr(self, 'grid_y_vec', None) is not None:
            ny, nx = z_filt.shape
            grids['filtrada'] = {
                'key': 'filtrada',
                'label': f"2. Malla Filtrada Fourier 2D [{nx}×{ny}]",
                'Z': z_filt.copy(),
                'x': self.grid_x_vec.copy(),
                'y': self.grid_y_vec.copy(),
                'z_elev': 0.0,
                'name': 'Filtrada Fourier'
            }

        # 3. Fuente Equivalente Plano Z0
        z_fe = getattr(self, 'grid_eq_source_z0', None)
        fe_win = getattr(self, 'dlg_problema_inverso', None)
        if z_fe is None and fe_win is not None and getattr(fe_win, 'last_results', None) is not None:
            z_fe = fe_win.last_results.get('G_plano', None)

        if z_fe is not None:
            ny, nx = z_fe.shape
            x_fe = getattr(self, 'grid_eq_source_x', None)
            if x_fe is None and fe_win is not None and getattr(fe_win, 'raster_x', None) is not None:
                x_fe = fe_win.raster_x
            if x_fe is None:
                x_fe = self.grid_x_vec

            y_fe = getattr(self, 'grid_eq_source_y', None)
            if y_fe is None and fe_win is not None and getattr(fe_win, 'raster_y', None) is not None:
                y_fe = fe_win.raster_y
            if y_fe is None:
                y_fe = self.grid_y_vec

            z0_val = float(getattr(self, 'grid_eq_source_z0_elev', getattr(fe_win, 'last_results', {}).get('Z0', 0.0)))

            grids['fuente_equivalente'] = {
                'key': 'fuente_equivalente',
                'label': f"3. Malla con Fuente Equivalente (Plano Z0={z0_val:.0f}m) [{nx}×{ny}]",
                'Z': z_fe.copy(),
                'x': x_fe.copy(),
                'y': y_fe.copy(),
                'z_elev': z0_val,
                'name': f"Fuente Equivalente (Z0={z0_val:.0f}m)"
            }

            # 4. Fuente Equivalente Filtrada
            z_fe_filt = getattr(self, 'grid_eq_source_z0_filt', None)
            if z_fe_filt is None:
                z_fe_filt = self.update_equivalent_source_filtered()
            if z_fe_filt is not None:
                grids['fuente_equivalente_filtrada'] = {
                    'key': 'fuente_equivalente_filtrada',
                    'label': f"4. Malla Fuente Equivalente Filtrada [{nx}×{ny}]",
                    'Z': z_fe_filt.copy(),
                    'x': x_fe.copy(),
                    'y': y_fe.copy(),
                    'z_elev': z0_val,
                    'name': 'Fuente Equivalente Filtrada'
                }

        return grids

    def browse_tiff_gravity_mesh(self):
        """
        Carga directamente un raster TIFF / GeoTIFF con mediciones gravimétricas,
        configurando sus parámetros de georreferenciación, unidades físicas y valores NoData
        mediante el Diálogo de Parámetros Raster TIFF.
        """
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Cargar Raster TIFF de Mediciones Gravimétricas", os.getcwd(),
            "Archivos TIFF (*.tif *.tiff);;Todos (*.*)"
        )
        if not fpath:
            return

        try:
            dlg = DialogoParametrosRasterTIFF(fpath, is_gravity=True, parent=self)
            if dlg.exec_() != QDialog.Accepted:
                return

            Z_grid, x_vec, y_vec, meta = dlg.get_result()
            self.grid_Z = Z_grid
            self.grid_Z_orig = Z_grid.copy()
            self.grid_x_vec = x_vec
            self.grid_y_vec = y_vec
            self.grid_dx = meta['dx']
            self.grid_dy = meta['dy']
            self.dx_val = float(meta['dx'])
            self.dy_val = float(meta['dy'])
            if hasattr(self, 'spin_dx'):
                self.spin_dx.blockSignals(True)
                self.spin_dx.setValue(float(meta['dx']))
                self.spin_dx.blockSignals(False)
            if hasattr(self, 'spin_dy'):
                self.spin_dy.blockSignals(True)
                self.spin_dy.setValue(float(meta['dy']))
                self.spin_dy.blockSignals(False)
            self.unit_xy = meta.get('unit_xy', 'm')
            self.unit_z = "mGal"

            # Sincronizar selectores de unidades si existen
            idx = 1 if 'km' in self.unit_xy.lower() else 0
            self.combo_unit_xy.setCurrentIndex(idx)
            self.combo_unit_z.setCurrentIndex(0)

            ny, nx = Z_grid.shape
            fname = os.path.basename(fpath)
            self.lbl_file_status.setText(f"TIFF: {fname} [{nx}×{ny} nodos]")
            self.status.showMessage(f"Malla TIFF de gravedad cargada: {fname} [{nx}×{ny} nodos] | Cotas: [{np.min(Z_grid):.2f}, {np.max(Z_grid):.2f}] mGal")

            self.recompute_fourier_2d()
            self.plot_master_viewer()
            self.update_mesh_table()
            self.tabs.setCurrentIndex(2)

            QMessageBox.information(
                self, "Raster TIFF de Gravedad Cargado",
                f"Se ha cargado con éxito la malla regular de gravedad desde TIFF:\n\n"
                f"Archivo: {fname}\n"
                f"Nodos: {nx} columnas × {ny} filas ({nx*ny:,} celdas)\n"
                f"Paso de Malla: ΔX = {self.grid_dx:.3f} {self.unit_xy}, ΔY = {self.grid_dy:.3f} {self.unit_xy}\n"
                f"Extensión X: [{x_vec[0]:,.2f}, {x_vec[-1]:,.2f}] {self.unit_xy}\n"
                f"Extensión Y: [{y_vec[0]:,.2f}, {y_vec[-1]:,.2f}] {self.unit_xy}\n"
                f"Rango Gravimétrico: [{np.min(Z_grid):.2f}, {np.max(Z_grid):.2f}] mGal\n\n"
                f"La malla ya está disponible en el Visualizador 2D, Análisis Fourier 2D y en Problema Inverso."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al Cargar TIFF", f"No se pudo cargar el archivo TIFF:\n{e}")

    def browse_tiff_mdt(self):
        """
        Carga un raster MDT (Topografía/Elevación) en formato TIFF/GeoTIFF,
        permitiendo que conviva con la gravedad y esté disponible para
        Fuente Equivalente, el Visualizador Maestro y el Método Inverso Puntual.
        """
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Cargar Raster Topografía MDT (TIFF / GeoTIFF)", os.getcwd(),
            "Archivos TIFF (*.tif *.tiff);;Todos (*.*)"
        )
        if not fpath:
            return

        try:
            dlg = DialogoParametrosRasterTIFF(fpath, is_gravity=False, parent=self)
            if dlg.exec_() != QDialog.Accepted:
                return

            Z_grid, x_vec, y_vec, meta = dlg.get_result()
            self.grid_mdt = Z_grid
            self.mdt_x_vec = x_vec
            self.mdt_y_vec = y_vec
            self.mdt_filepath = fpath

            ny, nx = Z_grid.shape
            fname = os.path.basename(fpath)
            self.status.showMessage(
                f"MDT Topografía cargado: {fname} [{nx}×{ny} nodos] | "
                f"Cotas: [{np.min(Z_grid):.1f}, {np.max(Z_grid):.1f}] m"
            )

            # Si ya está abierto el diálogo de Problema Inverso, actualizarlo
            if getattr(self, 'dlg_problema_inverso', None) is not None:
                self.dlg_problema_inverso.raw_mdt_Z = Z_grid.copy()
                self.dlg_problema_inverso.raw_mdt_x = x_vec.copy()
                self.dlg_problema_inverso.raw_mdt_y = y_vec.copy()
                self.dlg_problema_inverso.match_mdt_to_gravity()

            # Refrescar visualizador si está seleccionada la capa MDT
            if hasattr(self, 'combo_master_layer') and self.combo_master_layer.currentData() == 'mdt':
                self.plot_master_viewer()

            QMessageBox.information(
                self, "Raster MDT Cargado",
                f"Se ha cargado con éxito la topografía MDT:\n\n"
                f"Archivo: {fname}\n"
                f"Dimensiones: {nx} columnas × {ny} filas ({nx*ny:,} nodos)\n"
                f"Extensión X: [{x_vec[0]:,.2f}, {x_vec[-1]:,.2f}] m\n"
                f"Extensión Y: [{y_vec[0]:,.2f}, {y_vec[-1]:,.2f}] m\n"
                f"Rango de Cotas: [{np.min(Z_grid):.1f}, {np.max(Z_grid):.1f}] m\n\n"
                f"El MDT ahora convive en el entorno y está disponible para Fuente Equivalente, "
                f"Visualizador Maestro y Método Inverso Puntual."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al Cargar MDT", f"No se pudo cargar el archivo MDT:\n{e}")


def main():
    import ctypes
    try:
        myappid = 'geophysics.gravtt.suite.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    icon_path = os.path.join(BASE_DIR, 'assets', 'icon.ico')
    if not os.path.exists(icon_path):
        icon_path = os.path.join(BASE_DIR, 'assets', 'logo.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    splash = None
    logo_path = os.path.join(BASE_DIR, 'assets', 'logo.png')
    if os.path.exists(logo_path):
        pix = QPixmap(logo_path).scaled(340, 340, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.show()
        splash.showMessage("Iniciando GravTT...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        app.processEvents()

    win = GeneradorMallas2DWindow()
    win.showMaximized()
    if splash is not None:
        splash.finish(win)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
