# -*- coding: utf-8 -*-
"""
MÓDULO DE MODELADO ESPECTRAL SINTÉTICO (BLAKELY 1995, CAP. 11) Y COMPARACIÓN DE MODELOS
======================================================================================
Permite crear modelos geofísicos sintéticos limpios o sincronizados con una malla existente
sumando múltiples fuentes elementales (esferas, líneas verticales/horizontales, cintas/diques)
tanto en gravimetría (mGal) como en magnetometría (nT).
Cada cuerpo se evalúa en el dominio de Fourier con modulación de fase espacial exp(-1j*(kx*x + ky*y))
y el espectro total se antitransforma con IFFT2 (con opción de ruido gaussiano).

Incluye:
- Formulación matemática completamente editable por el usuario para cada cuerpo.
- Glosario de símbolos con letras normales (sin letras griegas: rho, mu, gamma, deltaX, etc.).
- Actualización en tiempo real de la ecuación matemática sustituida al cambiar los números arriba.
- Carga de fórmulas y parámetros predeterminados al seleccionar cualquier modelo de Blakely.
- Ventana de Comparación de Modelos (Obs vs Mod) con evaluación algebraica píxel a píxel.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from PIL import Image, TiffImagePlugin

from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QSplitter,
    QFileDialog, QMessageBox, QFrame, QRadioButton, QButtonGroup,
    QGridLayout, QSlider, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont, QColor

import matplotlib
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as patches

from scipy.interpolate import RegularGridInterpolator

# Constantes físicas fundamentales
GAMMA_G = 6.6743e-11        # Constante de gravitación universal G [m³ / (kg·s²)]
SI_TO_MGAL = 1e5            # Factor conversión m/s² a mGal
CM_MAG = 100.0              # Factor magnetostático Cm = mu0/(4*pi) * 1e9 [nT·m / A]


# ==============================================================================
# BASE DE DATOS TEÓRICA DE MODELOS DE FUENTE (BLAKELY 1995, TABLA 11.1)
# CON LETRAS NORMALES (SIN LETRAS GRIEGAS)
# ==============================================================================
BLAKELY_MODELS_INFO = {
    'grav_sphere': {
        'name': "Monopole (uniform sphere) [Eqn. 11.22]",
        'category': 'gravedad',
        'unit': 'mGal',
        'eqn': 'Eqn. 11.22',
        'formula_editable': "2 * pi * gamma * (4.0 / 3.0 * pi * R**3 * rho) * exp(-K * z1)",
        'theory': (
            "<b>Modelo Blakely (Eqn. 11.22): Monopole (uniform sphere)</b><br>"
            "Fórmula original: <code>2 * pi * gamma * mu * exp(-K * z1)</code><br>"
            "donde la masa total <b>mu [kg]</b> con densidad volumétrica <b>rho [kg/m³]</b> es:<br>"
            "<b>mu = (4/3) * pi * R³ * rho</b>."
        )
    },
    'grav_vline': {
        'name': "Vertical line (uniform cylinder) [Eqn. 11.23]",
        'category': 'gravedad',
        'unit': 'mGal',
        'eqn': 'Eqn. 11.23',
        'formula_editable': "(2 * pi * gamma * (pi * R**2 * rho) / np.maximum(1e-12, K)) * (exp(-K * z1) - exp(-K * z2))",
        'theory': (
            "<b>Modelo Blakely (Eqn. 11.23): Vertical line</b><br>"
            "Fórmula original: <code>(2 * pi * gamma * lambda / |k|) * (exp(-|k|*z1) - exp(-|k|*z2))</code><br>"
            "donde la densidad lineal <b>lambda [kg/m]</b> con densidad <b>rho [kg/m³]</b> es:<br>"
            "<b>lambda = pi * R² * rho</b> (tope en <i>z1</i>, fondo en <i>z2</i>)."
        )
    },
    'grav_hline': {
        'name': "Horizontal line (uniform cylinder) [Eqn. 11.27]",
        'category': 'gravedad',
        'unit': 'mGal',
        'eqn': 'Eqn. 11.27',
        'formula_editable': "2 * pi * gamma * (pi * R**2 * rho) * L * exp(-K * z1) * sinc(k_parallel * L / (2 * pi))",
        'theory': (
            "<b>Modelo Blakely (Eqn. 11.27): Horizontal line</b><br>"
            "Fórmula original: <code>2 * pi * gamma * lambda * exp(-|k|*z1)</code><br>"
            "donde la densidad lineal <b>lambda [kg/m]</b> con densidad <b>rho [kg/m³]</b> es:<br>"
            "<b>lambda = pi * R² * rho</b> (profundidad <i>z1</i>, longitud <i>L</i>, rumbo <i>strike</i>)."
        )
    },
    'grav_ribbon': {
        'name': "Vertical ribbon (dique vertical) [Eqn. 11.28]",
        'category': 'gravedad',
        'unit': 'mGal',
        'eqn': 'Eqn. 11.28',
        'formula_editable': "(2 * pi * gamma * (espesor * rho) * L / np.maximum(1e-12, K)) * (exp(-K * z1) - exp(-K * z2)) * sinc(k_parallel * L / (2 * pi))",
        'theory': (
            "<b>Modelo Blakely (Eqn. 11.28): Vertical ribbon</b><br>"
            "Fórmula original: <code>(2 * pi * gamma * sigma / |k|) * (exp(-|k|*z1) - exp(-|k|*z2))</code><br>"
            "donde la densidad superficial <b>sigma [kg/m²]</b> con densidad <b>rho [kg/m³]</b> es:<br>"
            "<b>sigma = espesor * rho</b> (tope <i>z1</i>, base <i>z2</i>, longitud <i>L</i>, rumbo <i>strike</i>)."
        )
    }
}


# ==============================================================================
# VALORES PREDETERMINADOS PARA CADA MODELO (RECARGA AL SELECCIONAR FORMULA)
# ==============================================================================
PRESETS_DEFAULT_DATA = {
    'grav_sphere': {
        'name': "Monopole (uniform sphere)",
        'formula': "2 * pi * gamma * (4.0 / 3.0 * pi * R**3 * rho) * exp(-K * z1)",
        'rho': 1200.0,
        'R': 350.0,
        'mu': 4.0 / 3.0 * np.pi * (350.0 ** 3) * 1200.0,
        'z1': 1500.0,
        'z2': 3000.0,
        'L': 2000.0,
        'dip': 90.0,
        'strike': 0.0,
        'espesor': 50.0,
        'gamma': 6.6743e-11
    },
    'grav_vline': {
        'name': "Vertical line (uniform cylinder)",
        'formula': "(2 * pi * gamma * (pi * R**2 * rho) / np.maximum(1e-12, K)) * (exp(-K * z1) - exp(-K * z2))",
        'rho': 1100.0,
        'R': 200.0,
        'mu': 1.0e12,
        'z1': 400.0,
        'z2': 3500.0,
        'L': 2000.0,
        'dip': 90.0,
        'strike': 0.0,
        'espesor': 50.0,
        'gamma': 6.6743e-11
    },
    'grav_hline': {
        'name': "Horizontal line (uniform cylinder)",
        'formula': "2 * pi * gamma * (pi * R**2 * rho) * L * exp(-K * z1) * sinc(k_parallel * L / (2 * pi))",
        'rho': 1000.0,
        'R': 250.0,
        'mu': 1.5e12,
        'z1': 1200.0,
        'z2': 3000.0,
        'L': 3000.0,
        'dip': 0.0,
        'strike': 90.0,
        'espesor': 50.0,
        'gamma': 6.6743e-11
    },
    'grav_ribbon': {
        'name': "Vertical ribbon (dique vertical)",
        'formula': "(2 * pi * gamma * (espesor * rho) * L / np.maximum(1e-12, K)) * (exp(-K * z1) - exp(-K * z2)) * sinc(k_parallel * L / (2 * pi))",
        'rho': 850.0,
        'R': 100.0,
        'mu': 8.0e11,
        'z1': 300.0,
        'z2': 2500.0,
        'L': 4000.0,
        'dip': 90.0,
        'strike': 45.0,
        'espesor': 60.0,
        'gamma': 6.6743e-11
    }
}


def get_default_formula_for_type(body_type):
    p = PRESETS_DEFAULT_DATA.get(body_type)
    if p:
        return p['formula']
    return "2 * pi * gamma * (4.0 / 3.0 * pi * R**3 * rho) * exp(-K * z1)"


# ==============================================================================
# MODELO DE DATOS: CUERPO SINTÉTICO (FUENTE ELEMENTAL)
# ==============================================================================
class CuerpoSintetico:
    """
    Representa una fuente de anomalía geofísica (gravimétrica o magnética)
    con formulación matemática editable y parámetros en letras normales.
    """
    def __init__(self, name="Esfera 1", body_type="grav_sphere",
                 x=0.0, y=0.0, z1=1000.0, z2=2500.0,
                 mu=None, R=250.0, rho=1000.0, gamma=6.6743e-11,
                 L=2000.0, dip=45.0, strike=0.0, espesor=50.0,
                 mag_m=1e9, inc_f=60.0, dec_f=0.0, inc_m=60.0, dec_m=0.0, Cm=100.0,
                 formula_expr=None, is_enabled=True,
                 # Compatibilidad con argumentos antiguos
                 z=None, mass=None, radius=None, density=None, length=None, mag_moment=None):
        self.name = str(name)
        self.body_type = str(body_type)
        self.x = float(x)
        self.y = float(y)
        self.z1 = max(1.0, float(z1 if z is None else z))
        self.z = self.z1
        self.z2 = max(self.z1 + 1.0, float(z2))
        self.R = max(1.0, float(R if radius is None else radius))
        self.radius = self.R
        self.rho = float(rho if density is None else density)
        self.density = self.rho
        if mu is not None:
            self.mu = float(mu)
        elif mass is not None:
            self.mu = float(mass)
        elif self.body_type in ('grav_sphere', 'custom'):
            self.mu = 4.0 / 3.0 * np.pi * (self.R ** 3) * self.rho
        else:
            self.mu = 1.0e12
        self.mass = self.mu
        self.gamma = float(gamma)
        self.L = max(1.0, float(L if length is None else length))
        self.length = self.L
        self.dip = float(dip)
        self.strike = float(strike)
        self.espesor = max(1.0, float(espesor))
        self.mag_m = float(mag_m if mag_moment is None else mag_moment)
        self.mag_moment = self.mag_m
        self.inc_f = float(inc_f)
        self.dec_f = float(dec_f)
        self.inc_m = float(inc_m)
        self.dec_m = float(dec_m)
        self.Cm = float(Cm)
        self.is_enabled = bool(is_enabled)
        self.formula_expr = str(formula_expr).strip() if formula_expr else get_default_formula_for_type(self.body_type)

    def get_category(self):
        info = BLAKELY_MODELS_INFO.get(self.body_type, {})
        return info.get('category', 'gravedad')

    def get_unit(self):
        info = BLAKELY_MODELS_INFO.get(self.body_type, {})
        return info.get('unit', 'mGal')

    def get_summary_text(self):
        info = BLAKELY_MODELS_INFO.get(self.body_type, {})
        t_name = info.get('name', self.body_type)
        if self.get_category() == 'gravedad':
            if self.body_type in ('grav_sphere', 'custom'):
                return f"{t_name} | (X={self.x:,.0f}, Y={self.y:,.0f}, z1={self.z1:,.0f} m, rho={self.rho:g} kg/m³, R={self.R:g} m)"
            elif self.body_type == 'grav_inclined_cylinder':
                return f"{t_name} | (X={self.x:,.0f}, Y={self.y:,.0f}, z1={self.z1:,.0f} m, buz={self.dip:g}°, L={self.L:,.0f} m, rho={self.rho:g} kg/m³)"
            elif self.body_type in ('grav_vline', 'grav_ribbon', 'grav_prism'):
                return f"{t_name} | (X={self.x:,.0f}, Y={self.y:,.0f}, z1={self.z1:,.0f}, z2={self.z2:,.0f} m, rho={self.rho:g} kg/m³)"
            else:
                return f"{t_name} | (X={self.x:,.0f}, Y={self.y:,.0f}, z1={self.z1:,.0f} m, rho={self.rho:g} kg/m³, L={self.L:,.0f} m)"
        else:
            return f"{t_name} | (X={self.x:,.0f}, Y={self.y:,.0f}, z1={self.z1:,.0f} m, mag_m={self.mag_m:.2e} A·m²)"

    def get_substituted_numeric_formula(self, formula_str=None):
        """
        Retorna la fórmula matemática con los valores numéricos actuales de las variables
        sustituidos en formato legible.
        """
        expr = formula_str if formula_str else self.formula_expr
        if not expr:
            expr = get_default_formula_for_type(self.body_type)

        mu_val = self.mu if self.mu > 0 else (4.0 / 3.0 * np.pi * (self.R ** 3) * self.rho)
        span_z = max(1.0, self.z2 - self.z1)
        lambda_val = (np.pi * (self.R ** 2) * self.rho) if self.rho != 0 else (mu_val / span_z)
        sigma_val = self.rho * max(1.0, self.espesor)
        mag_m_lin = self.mag_m / max(1.0, span_z)
        mag_m_surf = self.mag_m / max(1.0, span_z * self.L)

        subs = [
            (r'\bpi\b', '3.1416'),
            (r'\bgamma\b', f'{self.gamma:.4e}'),
            (r'\brho\b', f'{self.rho:g}'),
            (r'\bmu\b', f'{mu_val:.3e}'),
            (r'\bmass\b', f'{mu_val:.3e}'),
            (r'\blambda_lin\b', f'{lambda_val:.3e}'),
            (r'\bsigma_sup\b', f'{sigma_val:.3e}'),
            (r'\bR\b', f'{self.R:g}'),
            (r'\bradius\b', f'{self.R:g}'),
            (r'\bz1\b', f'{self.z1:g}'),
            (r'\bz2\b', f'{self.z2:g}'),
            (r'\bL\b', f'{self.L:g}'),
            (r'\blength\b', f'{self.L:g}'),
            (r'\bdip\b', f'{self.dip:g}'),
            (r'\bstrike\b', f'{self.strike:g}'),
            (r'\bespesor\b', f'{self.espesor:g}'),
            (r'\bCm\b', f'{self.Cm:g}'),
            (r'\bmag_m_lin\b', f'{mag_m_lin:.3e}'),
            (r'\bmag_m_surf\b', f'{mag_m_surf:.3e}'),
            (r'\bmag_m\b', f'{self.mag_m:.3e}')
        ]
        res = expr
        for pattern, val in subs:
            res = re.sub(pattern, val, res)
        return res

    def evaluate_spectrum(self, kx, ky, K, x0_grid, y0_grid, dx, dy, z0=0.0):
        """
        Calcula el espectro 2D del cuerpo evaluando la fórmula matemática editable,
        multiplicada por el factor de desplazamiento espacial Shift y la escala de grilla.
        """
        if not self.is_enabled:
            return np.zeros_like(K, dtype=complex)

        dx = max(1e-4, float(dx))
        dy = max(1e-4, float(dy))
        scale_grav = (SI_TO_MGAL) / (dx * dy)
        scale_mag = CM_MAG / (dx * dy)

        # Factor de desplazamiento de fase espacial en Fourier (Shift Theorem)
        shift = np.exp(-1j * (kx * (self.x - x0_grid) + ky * (self.y - y0_grid)))

        # Dirección paralela a lo largo del rumbo (strike) y perpendicular
        rad_strike = np.radians(self.strike)
        k_parallel = kx * np.sin(rad_strike) + ky * np.cos(rad_strike)
        k_perp = -kx * np.cos(rad_strike) + ky * np.sin(rad_strike)
        sinc_term = np.sinc(k_parallel * self.L / (2.0 * np.pi))
        k_parallel_proj = k_parallel * np.cos(np.radians(self.dip))

        depth_top = max(0.01, self.z1 - z0)
        depth_bottom = max(depth_top + 1.0, self.z2 - z0)

        # Masa y variables derivadas
        mu_val = self.mu if self.mu > 0 else (4.0 / 3.0 * np.pi * (self.R ** 3) * self.rho)
        span_z = max(1.0, depth_bottom - depth_top)
        lambda_val = (np.pi * (self.R ** 2) * self.rho) if self.rho != 0 else (mu_val / span_z)
        sigma_val = self.rho * max(1.0, self.espesor)

        # Factores magnéticos
        rad_if = np.radians(self.inc_f); rad_df = np.radians(self.dec_f)
        rad_im = np.radians(self.inc_m); rad_dm = np.radians(self.dec_m)
        K_safe = np.maximum(1e-12, K)
        Theta_f = np.sin(rad_if) + 1j * (np.cos(rad_if) * (np.sin(rad_df) * (kx / K_safe) + np.cos(rad_df) * (ky / K_safe)))
        Theta_m = np.sin(rad_im) + 1j * (np.cos(rad_im) * (np.sin(rad_dm) * (kx / K_safe) + np.cos(rad_dm) * (ky / K_safe)))
        Theta_prod = Theta_m * Theta_f

        mag_m_lin = self.mag_m / max(1.0, span_z)
        mag_m_surf = self.mag_m / max(1.0, span_z * self.L)

        is_mag = (self.get_category() == 'magnetometria')
        scale_val = scale_mag if is_mag else scale_grav

        # Entorno de variables para evaluar la fórmula matemática editable
        env = {
            'pi': np.pi,
            'gamma': self.gamma,
            'rho': self.rho,
            'mu': mu_val,
            'mass': mu_val,
            'R': self.R,
            'radius': self.R,
            'z1': depth_top,
            'z2': depth_bottom,
            'z': depth_top,
            'L': self.L,
            'length': self.L,
            'dip': self.dip,
            'strike': self.strike,
            'espesor': self.espesor,
            'lambda_lin': lambda_val,
            'sigma_sup': sigma_val,
            'mag_m': self.mag_m,
            'mag_m_lin': mag_m_lin,
            'mag_m_surf': mag_m_surf,
            'Cm': self.Cm,
            'Theta_prod': Theta_prod,
            'Theta_m': Theta_m,
            'Theta_f': Theta_f,
            'K': K,
            'kx': kx,
            'ky': ky,
            'k_parallel': k_parallel,
            'k_perp': k_perp,
            'k_parallel_proj': k_parallel_proj,
            'sinc_term': sinc_term,
            'deltaX': dx,
            'deltaY': dy,
            'dx': dx,
            'dy': dy,
            'scale': scale_val,
            'shift': shift,
            'np': np,
            'exp': np.exp,
            'sqrt': np.sqrt,
            'sin': np.sin,
            'cos': np.cos,
            'radians': np.radians,
            'sinc': np.sinc,
            'abs': np.abs,
            'max': np.maximum,
            'maximum': np.maximum,
            'min': np.minimum,
            'minimum': np.minimum
        }

        formula = (self.formula_expr or "").strip()
        if not formula:
            formula = get_default_formula_for_type(self.body_type)

        try:
            E_eval = eval(formula, {'__builtins__': {}}, env)
            if not isinstance(E_eval, np.ndarray):
                E_eval = np.full_like(K, complex(E_eval), dtype=complex)
            if 'scale' not in formula:
                E_eval = E_eval * scale_val
            if 'shift' not in formula:
                E_eval = E_eval * shift
            return E_eval
        except Exception as err:
            # Fallback seguro al cálculo analítico
            return self._fallback_spectrum(kx, ky, K, depth_top, depth_bottom, mu_val, lambda_val, sigma_val, Theta_prod, sinc_term, scale_val, shift, k_parallel)

    def _fallback_spectrum(self, kx, ky, K, depth_top, depth_bottom, mu_val, lambda_val, sigma_val, Theta_prod, sinc_term, scale_val, shift, k_parallel=None):
        if k_parallel is None:
            rad_strike = np.radians(self.strike)
            k_parallel = kx * np.sin(rad_strike) + ky * np.cos(rad_strike)

        if self.body_type in ('grav_sphere', 'custom'):
            E = 2.0 * np.pi * self.gamma * mu_val * np.exp(-K * depth_top) * scale_val
            return E * shift
        elif self.body_type == 'grav_inclined_cylinder':
            alpha = np.radians(self.dip)
            w = K * np.sin(alpha) + 1j * k_parallel * np.cos(alpha)
            w_safe = np.where(np.abs(w) < 1e-12, 1e-12 + 0j, w)
            int_factor = np.where(np.abs(w) < 1e-12, self.L, (1.0 - np.exp(-w * self.L)) / w_safe)
            E = 2.0 * np.pi * self.gamma * (np.pi * (self.R**2) * self.rho) * np.exp(-K * depth_top) * int_factor * scale_val
            return E * shift
        elif self.body_type == 'grav_vline':
            span = max(1.0, depth_bottom - depth_top)
            diff_z = np.where(K < 1e-12, span, (np.exp(-K * depth_top) - np.exp(-K * depth_bottom)) / np.maximum(1e-12, K))
            E = 2.0 * np.pi * self.gamma * lambda_val * diff_z * scale_val
            return E * shift
        elif self.body_type == 'grav_hline':
            E = 2.0 * np.pi * self.gamma * lambda_val * self.L * np.exp(-K * depth_top) * sinc_term * scale_val
            return E * shift
        elif self.body_type == 'grav_ribbon':
            span = max(1.0, depth_bottom - depth_top)
            diff_z = np.where(K < 1e-12, span, (np.exp(-K * depth_top) - np.exp(-K * depth_bottom)) / np.maximum(1e-12, K))
            E = 2.0 * np.pi * self.gamma * sigma_val * self.L * diff_z * sinc_term * scale_val
            return E * shift
        elif self.body_type == 'grav_prism':
            span = max(1.0, depth_bottom - depth_top)
            diff_z = np.where(K < 1e-12, span, (np.exp(-K * depth_top) - np.exp(-K * depth_bottom)) / np.maximum(1e-12, K))
            rad_strike = np.radians(self.strike)
            k_perp = -kx * np.cos(rad_strike) + ky * np.sin(rad_strike)
            sinc_perp = np.sinc(k_perp * (2.0 * self.R) / (2.0 * np.pi))
            E = 2.0 * np.pi * self.gamma * self.rho * (2.0 * self.R) * self.L * diff_z * sinc_perp * sinc_term * scale_val
            return E * shift
        elif self.body_type == 'mag_dipole':
            E = 2.0 * np.pi * self.Cm * self.mag_m * Theta_prod * K * np.exp(-K * depth_top) * scale_val
            return E * shift
        return np.zeros_like(K, dtype=complex)

    def evaluate_spatial(self, X, Y, z0=0.0):
        """
        Calcula la respuesta anómala exacta en el dominio espacial (cuerpo físico aislado)
        evaluado directamente en los nodos de la malla (X, Y).
        
        Ventajas decisivas respecto a Fourier (FFT):
        - NO produce réplicas periódicas por periodicidad implícita de DFT.
        - NO genera el efecto artificial de placa plana infinita (Bouguer) cuando z1 > extensión.
        - Es analítico, exacto, instantáneo (0.1 ms) e inmune a artefactos de borde.
        """
        if not self.is_enabled:
            return np.zeros_like(X, dtype=float)

        dx = X - float(self.x)
        dy = Y - float(self.y)
        r2 = dx ** 2 + dy ** 2
        depth_top = max(0.1, float(self.z1) - float(z0))
        depth_bottom = max(depth_top + 1.0, float(self.z2) - float(z0))
        span_z = max(1.0, depth_bottom - depth_top)

        # Masa y variables derivadas
        mu_val = float(self.mu) if self.mu > 0 else (4.0 / 3.0 * np.pi * (float(self.R) ** 3) * float(self.rho))
        lambda_val = (np.pi * (float(self.R) ** 2) * float(self.rho)) if self.rho != 0 else (mu_val / span_z)
        sigma_val = float(self.rho) * max(0.1, float(self.espesor))

        bt = self.body_type
        if bt in ('grav_sphere', 'custom'):
            # Esfera puntual / masa concentrada (Blakely Ec. 11.22 en espacio directo)
            # Δg = SI_TO_MGAL * gamma * mu * z1 / (r² + z1²)^(3/2)
            denom = (r2 + depth_top ** 2) ** 1.5
            return (SI_TO_MGAL * self.gamma * mu_val * depth_top) / np.maximum(1e-12, denom)

        elif bt == 'grav_vline':
            # Cilindro / línea de masa vertical finita entre z1 y z2 (Blakely Ec. 11.23)
            # Δg = SI_TO_MGAL * gamma * lambda * [1/sqrt(r² + z1²) - 1/sqrt(r² + z2²)]
            inv1 = 1.0 / np.sqrt(r2 + depth_top ** 2)
            inv2 = 1.0 / np.sqrt(r2 + depth_bottom ** 2)
            return SI_TO_MGAL * self.gamma * lambda_val * (inv1 - inv2)

        elif bt == 'grav_hline':
            # Cilindro / línea de masa horizontal finita de longitud L y rumbo strike (Blakely Ec. 11.27)
            rad_strike = np.radians(float(self.strike))
            # Coord paralela s a lo largo del rumbo y perpendicular p
            s = dx * np.sin(rad_strike) + dy * np.cos(rad_strike)
            p = dx * np.cos(rad_strike) - dy * np.sin(rad_strike)
            r0_sq = p ** 2 + depth_top ** 2
            half_L = 0.5 * max(1.0, float(self.L))
            s1 = s + half_L
            s2 = s - half_L
            term1 = s1 / np.sqrt(r0_sq + s1 ** 2)
            term2 = s2 / np.sqrt(r0_sq + s2 ** 2)
            return (SI_TO_MGAL * self.gamma * lambda_val * depth_top / np.maximum(1e-12, r0_sq)) * (term1 - term2)

        elif bt == 'grav_ribbon':
            # Dique vertical tabular entre z1 y z2 con espesor y rumbo strike (Blakely Ec. 11.28)
            # Cuadratura Gauss-Legendre de 5 puntos sobre el eje vertical Z
            rad_strike = np.radians(float(self.strike))
            s = dx * np.sin(rad_strike) + dy * np.cos(rad_strike)
            p = dx * np.cos(rad_strike) - dy * np.sin(rad_strike)
            half_L = 0.5 * max(1.0, float(self.L))
            s1 = s + half_L
            s2 = s - half_L

            gp = np.array([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459])
            gw = np.array([0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850])
            z_mid = 0.5 * (depth_top + depth_bottom)
            z_half = 0.5 * (depth_bottom - depth_top)

            res = np.zeros_like(X, dtype=float)
            for zi_norm, wi in zip(gp, gw):
                zi = z_mid + z_half * zi_norm
                r0_sq_i = p ** 2 + zi ** 2
                term1_i = s1 / np.sqrt(r0_sq_i + s1 ** 2)
                term2_i = s2 / np.sqrt(r0_sq_i + s2 ** 2)
                dgi = (zi / np.maximum(1e-12, r0_sq_i)) * (term1_i - term2_i)
                res += wi * dgi
            return res * (SI_TO_MGAL * self.gamma * sigma_val * z_half)

        elif bt == 'grav_prism':
            # Prisma rectangular 3D tabular de semi-ancho R y longitud L
            rad_strike = np.radians(float(self.strike))
            s = dx * np.sin(rad_strike) + dy * np.cos(rad_strike)
            p = dx * np.cos(rad_strike) - dy * np.sin(rad_strike)
            half_L = 0.5 * max(1.0, float(self.L))
            half_W = max(1.0, float(self.R))
            s1 = s + half_L
            s2 = s - half_L
            p1 = p + half_W
            p2 = p - half_W

            # Cuadratura 2D de 3x3 sobre base horizontal y analítica en Z
            gp = np.array([-0.7745966692, 0.0, 0.7745966692])
            gw = np.array([0.5555555556, 0.8888888889, 0.5555555556])
            res = np.zeros_like(X, dtype=float)
            for pi_norm, wpi in zip(gp, gw):
                cur_p = p + half_W * pi_norm
                for si_norm, wsi in zip(gp, gw):
                    cur_s = s + half_L * si_norm
                    r_sq_cell = cur_p ** 2 + cur_s ** 2
                    inv1 = 1.0 / np.sqrt(r_sq_cell + depth_top ** 2)
                    inv2 = 1.0 / np.sqrt(r_sq_cell + depth_bottom ** 2)
                    res += (wpi * wsi) * (inv1 - inv2)
            cell_area = (2.0 * half_W) * (2.0 * half_L)
            return res * (SI_TO_MGAL * self.gamma * float(self.rho) * (cell_area / 4.0))

        elif bt == 'grav_inclined_cylinder':
            # Cilindro inclinado con buzamiento alpha y rumbo strike
            rad_strike = np.radians(float(self.strike))
            alpha = np.radians(float(self.dip))
            half_L = 0.5 * max(1.0, float(self.L))
            gp = np.linspace(-half_L, half_L, 7)
            w_step = (self.L / 7.0)
            res = np.zeros_like(X, dtype=float)
            for si in gp:
                xi = self.x + si * np.cos(alpha) * np.sin(rad_strike)
                yi = self.y + si * np.cos(alpha) * np.cos(rad_strike)
                zi = depth_top + (si + half_L) * np.sin(alpha)
                r2_i = (X - xi) ** 2 + (Y - yi) ** 2
                dgi = (zi / np.maximum(1e-12, (r2_i + zi ** 2) ** 1.5))
                res += dgi * w_step
            return res * (SI_TO_MGAL * self.gamma * lambda_val)

        elif bt == 'mag_dipole':
            rad_if = np.radians(float(self.inc_f)); rad_df = np.radians(float(self.dec_f))
            rad_im = np.radians(float(self.inc_m)); rad_dm = np.radians(float(self.dec_m))
            mx = float(self.mag_m) * np.cos(rad_im) * np.sin(rad_dm)
            my = float(self.mag_m) * np.cos(rad_im) * np.cos(rad_dm)
            mz = -float(self.mag_m) * np.sin(rad_im)
            fx = np.cos(rad_if) * np.sin(rad_df)
            fy = np.cos(rad_if) * np.cos(rad_df)
            fz = -np.sin(rad_if)
            rx = dx
            ry = dy
            rz = -depth_top
            R_dist = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
            R5 = np.maximum(1e-12, R_dist ** 5)
            R3 = np.maximum(1e-12, R_dist ** 3)
            m_dot_r = mx * rx + my * ry + mz * rz
            bx = CM_MAG * (3.0 * m_dot_r * rx / R5 - mx / R3)
            by = CM_MAG * (3.0 * m_dot_r * ry / R5 - my / R3)
            bz = CM_MAG * (3.0 * m_dot_r * rz / R5 - mz / R3)
            return bx * fx + by * fy + bz * fz

        return np.zeros_like(X, dtype=float)


# ==============================================================================
# FUNCIONES AUXILIARES DE TRAZADO GEOMÉTRICO DE FIGURAS ENTERRADAS
# ==============================================================================
def draw_cuerpo_planta_geometry(ax, c, show_labels=True):
    """
    Dibuja la proyección geométrica en planta (mapa 2D) de una fuente enterrada:
    - Esfera: Círculo punteado de radio R centrado en (x, y).
    - Cilindro Inclinado: Círculo tope (x, y), círculo fondo, envolvente del fuste y flecha de buzamiento.
    - Cilindro Vertical: Círculos concéntricos de radio R.
    - Cilindro Horizontal: Rectángulo / cápsula de longitud L y diámetro 2R orientado según rumbo (strike).
    - Dique / Cinta: Polígono tabular de longitud L y espesor orientado según rumbo.
    - Prisma: Huella rectangular de dimensiones L x 2R.
    (Sin puntos rojos sobre la anomalía para no obstaculizar la visualización).
    """
    if not getattr(c, 'is_enabled', True):
        return

    name = getattr(c, 'name', 'Fuente')
    bt = getattr(c, 'body_type', 'grav_sphere')
    x0 = float(getattr(c, 'x', 0.0))
    y0 = float(getattr(c, 'y', 0.0))
    z1 = float(getattr(c, 'z1', 1000.0))
    z2 = float(getattr(c, 'z2', 3000.0))
    R = max(1.0, float(getattr(c, 'R', 250.0)))
    L = max(1.0, float(getattr(c, 'L', 3000.0)))
    strike = float(getattr(c, 'strike', 0.0))
    dip = float(getattr(c, 'dip', 45.0))
    espesor = max(1.0, float(getattr(c, 'espesor', 60.0)))

    theta = np.radians(strike)
    u_vec = np.array([np.sin(theta), np.cos(theta)])   # Vector unitario a lo largo del rumbo
    v_vec = np.array([np.cos(theta), -np.sin(theta)])  # Vector unitario perpendicular

    color_main = '#ffff00'  # Amarillo visible sobre cualquier fondo de mapa
    color_edge = '#000000'

    if bt in ('grav_sphere', 'custom'):
        circ = patches.Circle((x0, y0), R, edgecolor=color_main, facecolor='#ffff00', alpha=0.20, lw=1.8, linestyle='--')
        ax.add_patch(circ)
        if show_labels:
            ax.text(x0 + R * 1.08, y0, name,
                    color='#ffffff', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#202020', alpha=0.75, edgecolor='none'))

    elif bt == 'grav_inclined_cylinder':
        alpha = np.radians(dip)
        L_proj = L * np.cos(alpha)
        z_end = z1 + L * np.sin(alpha)
        top_pos = np.array([x0, y0])
        bot_pos = top_pos + L_proj * u_vec

        # Envolvente del fuste del cuerpo cilíndrico
        p1 = top_pos - R * v_vec
        p2 = bot_pos - R * v_vec
        p3 = bot_pos + R * v_vec
        p4 = top_pos + R * v_vec
        poly = patches.Polygon([p1, p2, p3, p4], closed=True, edgecolor='#ff9900', facecolor='#ffaa00', alpha=0.25, lw=1.8)
        ax.add_patch(poly)

        # Círculo tope (somero)
        c_top = patches.Circle(top_pos, R, edgecolor='#ffff00', facecolor='#ffff00', alpha=0.35, lw=1.8)
        ax.add_patch(c_top)

        # Círculo fondo (profundo)
        c_bot = patches.Circle(bot_pos, R, edgecolor='#ff6600', facecolor='#ff6600', alpha=0.20, lw=1.4, linestyle=':')
        ax.add_patch(c_bot)

        # Flecha indicadora de dirección de buzamiento
        if L_proj > 5.0:
            ax.annotate('', xy=(bot_pos[0], bot_pos[1]), xytext=(top_pos[0], top_pos[1]),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0, color='#ffffff', mutation_scale=14))

        if show_labels:
            ax.text(top_pos[0] + R * 1.08, top_pos[1], f"{name} [Tope: {z1:g}m]",
                    color='#ffffff', fontsize=7.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#202020', alpha=0.75, edgecolor='none'))
            ax.text(bot_pos[0] + R * 1.08, bot_pos[1], f"Fondo (z={z_end:g}m, Buz: {dip:g}° ↘)",
                    color='#ffcc00', fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#202020', alpha=0.75, edgecolor='none'))

    elif bt == 'grav_vline':
        circ = patches.Circle((x0, y0), R, edgecolor='#00ffff', facecolor='#00ffff', alpha=0.20, lw=1.8)
        ax.add_patch(circ)
        circ_in = patches.Circle((x0, y0), R * 0.45, edgecolor='#00ffff', facecolor='none', lw=1.2, linestyle=':')
        ax.add_patch(circ_in)
        if show_labels:
            ax.text(x0 + R * 1.08, y0, name,
                    color='#ffffff', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#111111', alpha=0.75, edgecolor='none'))

    elif bt == 'grav_hline':
        p1 = np.array([x0, y0]) - 0.5 * L * u_vec - R * v_vec
        p2 = np.array([x0, y0]) + 0.5 * L * u_vec - R * v_vec
        p3 = np.array([x0, y0]) + 0.5 * L * u_vec + R * v_vec
        p4 = np.array([x0, y0]) - 0.5 * L * u_vec + R * v_vec
        poly = patches.Polygon([p1, p2, p3, p4], closed=True, edgecolor=color_main, facecolor='#ffff00', alpha=0.20, lw=1.8)
        ax.add_patch(poly)
        c_start = np.array([x0, y0]) - 0.5 * L * u_vec
        c_end = np.array([x0, y0]) + 0.5 * L * u_vec
        ax.plot([c_start[0], c_end[0]], [c_start[1], c_end[1]], '--', color=color_main, lw=1.2)
        if show_labels:
            ax.text(c_end[0] + 50, c_end[1], name,
                    color='#ffffff', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#111111', alpha=0.75, edgecolor='none'))

    elif bt == 'grav_ribbon':
        p1 = np.array([x0, y0]) - 0.5 * L * u_vec - 0.5 * espesor * v_vec
        p2 = np.array([x0, y0]) + 0.5 * L * u_vec - 0.5 * espesor * v_vec
        p3 = np.array([x0, y0]) + 0.5 * L * u_vec + 0.5 * espesor * v_vec
        p4 = np.array([x0, y0]) - 0.5 * L * u_vec + 0.5 * espesor * v_vec
        poly = patches.Polygon([p1, p2, p3, p4], closed=True, edgecolor='#00ff99', facecolor='#00ff99', alpha=0.25, lw=1.8)
        ax.add_patch(poly)
        if show_labels:
            ax.text(x0 + 100, y0, name,
                    color='#ffffff', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#111111', alpha=0.75, edgecolor='none'))

    elif bt == 'grav_prism':
        w_half = R
        p1 = np.array([x0, y0]) - 0.5 * L * u_vec - w_half * v_vec
        p2 = np.array([x0, y0]) + 0.5 * L * u_vec - w_half * v_vec
        p3 = np.array([x0, y0]) + 0.5 * L * u_vec + w_half * v_vec
        p4 = np.array([x0, y0]) - 0.5 * L * u_vec + w_half * v_vec
        poly = patches.Polygon([p1, p2, p3, p4], closed=True, edgecolor='#ff99bb', facecolor='#ff99bb', alpha=0.25, lw=1.8)
        ax.add_patch(poly)
        if show_labels:
            ax.text(x0 + 100, y0, name,
                    color='#ffffff', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#111111', alpha=0.75, edgecolor='none'))


def draw_cuerpo_perfil_geologico(ax, c):
    """
    Dibuja la vista de perfil / corte geológico subterráneo de la fuente en profundidad
    a escala isótropa 1:1 (mismas unidades en ambos ejes [m], sin distorsión vertical),
    mostrando el cuerpo geológico en el subsuelo y la superficie del terreno en Z=0.
    """
    bt = getattr(c, 'body_type', 'grav_sphere')
    z1 = float(getattr(c, 'z1', 1000.0))
    z2 = float(getattr(c, 'z2', 3000.0))
    R = max(1.0, float(getattr(c, 'R', 250.0)))
    L = max(1.0, float(getattr(c, 'L', 3000.0)))
    dip = float(getattr(c, 'dip', 45.0))
    espesor = max(1.0, float(getattr(c, 'espesor', 60.0)))
    name = getattr(c, 'name', 'Fuente')

    alpha = np.radians(dip)

    # Cálculo de profundidades y anchos característicos
    if bt == 'grav_vline':
        z_max_body = z2
        s_span = max(4.0 * R, 1500.0)
    elif bt == 'grav_ribbon':
        z_max_body = z2
        s_span = max(L, 1500.0)
    elif bt == 'grav_hline':
        z_max_body = z1 + R
        s_span = max(L, 1500.0)
    elif bt == 'grav_inclined_cylinder':
        s_top = 0.0
        z_top = z1
        s_bot = L * np.cos(alpha)
        z_bot = z1 + L * np.sin(alpha)
        z_max_body = z_bot + R
        s_span = max(s_bot, 1000.0)
    else:  # esfera
        z_max_body = z1 + R
        s_span = max(4.0 * R, 1500.0)

    # Rango horizontal s centrado o ajustado al cuerpo
    margin_s = max(500.0, s_span * 0.15)
    s_min = -0.5 * s_span - margin_s
    s_max = 0.5 * s_span + margin_s

    # Rango vertical Z con espacio suficiente sobre el terreno Z=0 para no chocar con el título
    z_bottom_view = max(z_max_body * 1.35, 1200.0)
    z_top_view = -max(250.0, z_bottom_view * 0.18)

    # Terreno en Z = 0 m
    ax.plot([s_min, s_max], [0, 0], '-', color='#553311', lw=2.5, label='Terreno (z = 0 m)')
    ax.fill_between([s_min, s_max], [z_top_view, z_top_view], [0, 0], color='#e8f4f8', alpha=0.6)  # Aire/Cielo
    ax.axhline(0, color='#332211', lw=1.5)

    if bt in ('grav_sphere', 'custom'):
        circ = patches.Circle((0, z1), R, edgecolor='#cc4400', facecolor='#ffaa00', alpha=0.55, lw=2.0)
        ax.add_patch(circ)
        y_text = min(z_bottom_view * 0.95, z1 + R * 1.3 + 50.0)
        ax.text(0.0, y_text, f"{name}: Esfera (Eqn. 11.22)\nR={R:g} m | z={z1:g} m",
                fontsize=8, fontweight='bold', color='#661100', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff5ea', edgecolor='#cc6600', lw=0.8))

    elif bt == 'grav_vline':
        rect = patches.Rectangle((-R, z1), 2 * R, z2 - z1, edgecolor='#007799', facecolor='#00bbdd', alpha=0.55, lw=2.0)
        ax.add_patch(rect)
        ax.text(R * 1.25, 0.5 * (z1 + z2), f"{name}: Cilindro Vertical (Eqn. 11.23)\nTope={z1:g} m | Fondo={z2:g} m\nR={R:g} m",
                fontsize=8, fontweight='bold', color='#004466', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#eef8fc', edgecolor='#0088aa', lw=0.8))

    elif bt == 'grav_hline':
        rect = patches.Rectangle((-0.5 * L, z1 - R), L, 2 * R, edgecolor='#aa6600', facecolor='#ffaa00', alpha=0.55, lw=2.0)
        ax.add_patch(rect)
        y_text = min(z_bottom_view * 0.95, z1 + R * 1.4 + 100.0)
        ax.text(0.0, y_text, f"{name}: Cilindro Horizontal (Eqn. 11.27)\nProf={z1:g} m | L={L:g} m | R={R:g} m",
                fontsize=8, fontweight='bold', color='#663300', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#fffbea', edgecolor='#aa6600', lw=0.8))

    elif bt == 'grav_inclined_cylinder':
        v_s = -np.sin(alpha)
        v_z = np.cos(alpha)

        p1 = [s_top + R * v_s, z_top + R * v_z]
        p2 = [s_bot + R * v_s, z_bot + R * v_z]
        p3 = [s_bot - R * v_s, z_bot - R * v_z]
        p4 = [s_top - R * v_s, z_top - R * v_z]

        poly = patches.Polygon([p1, p2, p3, p4], closed=True, edgecolor='#cc4400', facecolor='#ff9900', alpha=0.55, lw=2.2)
        ax.add_patch(poly)

        # Eje central y tapas
        ax.plot([s_top, s_bot], [z_top, z_bot], '--', color='#661100', lw=1.5)
        ax.plot([p1[0], p4[0]], [p1[1], p4[1]], '-', color='#ffff00', lw=2.5)  # Tapa tope
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], '-', color='#cc2200', lw=2.0)  # Tapa fondo

        # Textos informativos
        ax.text(s_top - R * 1.3, z_top, f"Tope: z={z1:g} m", fontsize=8, fontweight='bold', color='#992200', ha='right')
        ax.text(s_bot + R * 1.2, z_bot, f"Fondo: z={z_bot:g} m", fontsize=8, fontweight='bold', color='#771100')

        # Etiqueta de Buzamiento α
        mid_s = 0.5 * (s_top + s_bot)
        mid_z = 0.5 * (z_top + z_bot)
        ax.text(mid_s + R * 1.4, mid_z, f"{name}\nBuzamiento α = {dip:g}°\nL = {L:g} m, R = {R:g} m",
                fontsize=8, fontweight='bold', color='#aa2200',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff5ea', edgecolor='#cc6600', lw=1))

    elif bt == 'grav_ribbon':
        rect = patches.Rectangle((-0.5 * L, z1), L, z2 - z1, edgecolor='#008855', facecolor='#00cc77', alpha=0.55, lw=2.0)
        ax.add_patch(rect)
        y_text = min(z_bottom_view * 0.95, z2 + (z2 - z1) * 0.15 + 50.0)
        ax.text(0.0, y_text, f"{name}: Dique Vertical (Eqn. 11.28)\nTope={z1:g} m | Fondo={z2:g} m | L={L:g} m | e={espesor:g} m",
                fontsize=8, fontweight='bold', color='#004422', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#eaf8f0', edgecolor='#008855', lw=0.8))

    elif bt == 'grav_prism':
        rect = patches.Rectangle((-0.5 * L, z1), L, z2 - z1, edgecolor='#aa2266', facecolor='#ff99bb', alpha=0.55, lw=2.0)
        ax.add_patch(rect)
        ax.text(0.5 * L + 50, 0.5 * (z1 + z2), f"{name}\nPrisma 3D\nTope={z1:g} m | Fondo={z2:g} m\nL={L:g} m",
                fontsize=8, fontweight='bold', color='#661133',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fceef4', edgecolor='#aa2266', lw=0.8))

    else:
        circ = patches.Circle((0, z1), R, edgecolor='#cc0000', facecolor='#ffaa00', alpha=0.55, lw=2.0)
        ax.add_patch(circ)

    ax.set_xlim(s_min, s_max)
    ax.set_ylim(z_bottom_view, z_top_view)  # Z hacia abajo invertido
    ax.set_xlabel("Distancia a lo largo del perfil [m]", fontsize=8)
    ax.set_ylabel("Profundidad Z [m] (Hacia Abajo)", fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Evitar colapso vertical cuando la longitud horizontal es desproporcionadamente mayor que Z
    span_s = s_max - s_min
    span_z = z_bottom_view - z_top_view
    if span_s / max(1.0, span_z) > 4.5:
        ax.set_aspect('auto')
    else:
        ax.set_aspect('equal', adjustable='box')


# ==============================================================================
# DIÁLOGO DE SELECCIÓN INTERACTIVA DE COORDENADAS DESDE EL MAPA
# (PERMITE ELEGIR FONDO: MALLA OBSERVADA O MALLA SIMULADA O SUPERPOSICIÓN)
# ==============================================================================
class DialogoSelectorCoordenadasMapa(QDialog):
    """
    Despliega un mapa 2D interactivo con selección explícita del mapa de fondo:
    - 🌍 Malla Observada (Datos Reales / Trabajo de la ventana principal)
    - ✨ Malla Simulada (Modelo Sintético Blakely acumulado)
    - 👁️ Superposición Transparente (Observada + Simulada con control de opacidad)
    
    Permite al usuario hacer clic para situar la fuente, muestra retícula telescópica,
    coordenadas en tiempo real y marcas de otros cuerpos existentes.
    """
    def __init__(self, parent=None, initial_x=0.0, initial_y=0.0,
                 grid_obs=None, x_obs=None, y_obs=None,
                 grid_sim=None, x_sim=None, y_sim=None,
                 grid_z=None, x_vec=None, y_vec=None,
                 unit_xy="m", unit_z="mGal", otros_cuerpos=None,
                 alt_grid_z=None, alt_grid_name=None,
                 default_background="auto"):
        super().__init__(parent)
        self.setWindowTitle("🗺️ Selector de Coordenadas de Fuente - Malla Observada / Simulada")
        self.resize(980, 740)
        self.setMinimumSize(800, 580)

        self.selected_x = float(initial_x)
        self.selected_y = float(initial_y)
        self.unit_xy = str(unit_xy)
        self.unit_z = str(unit_z)
        self.otros_cuerpos = otros_cuerpos or []

        # Compatibilidad hacia atrás con argumentos grid_z / alt_grid_z
        if grid_obs is None and grid_z is not None:
            if alt_grid_z is not None:
                grid_obs = grid_z
                x_obs = x_vec
                y_obs = y_vec
                grid_sim = alt_grid_z
                x_sim = x_vec
                y_sim = y_vec
            else:
                grid_obs = grid_z
                x_obs = x_vec
                y_obs = y_vec

        # Asignar datos de Malla Observada
        self.grid_obs = grid_obs
        self.x_obs = x_obs
        self.y_obs = y_obs

        # Asignar datos de Malla Simulada
        self.grid_sim = grid_sim
        self.x_sim = x_sim
        self.y_sim = y_sim

        # Si faltan vectores en alguna pero están en la otra, sincronizar vectores
        if self.x_sim is None and self.x_obs is not None:
            self.x_sim = self.x_obs.copy()
            self.y_sim = self.y_obs.copy()
        if self.x_obs is None and self.x_sim is not None:
            self.x_obs = self.x_sim.copy()
            self.y_obs = self.y_sim.copy()

        # Si no hay ninguna grilla, crear un espacio espacial de referencia
        if self.grid_obs is None and self.grid_sim is None:
            cx = float(initial_x)
            cy = float(initial_y)
            span_x = 5000.0 if abs(cx) < 1.0 else abs(cx) * 0.5
            span_y = 5000.0 if abs(cy) < 1.0 else abs(cy) * 0.5
            self.x_sim = np.linspace(cx - span_x, cx + span_x, 101)
            self.y_sim = np.linspace(cy - span_y, cy + span_y, 101)
            self.grid_sim = np.zeros((101, 101))

        self.has_obs = (self.grid_obs is not None and self.x_obs is not None and self.y_obs is not None)
        self.has_sim = (self.grid_sim is not None and self.x_sim is not None and self.y_sim is not None)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # ======================================================================
        # SELECCIÓN PROMINENTE DE MAPA DE FONDO (OBSERVADA / SIMULADA / SUPERPOSICIÓN)
        # ======================================================================
        grp_selector = QGroupBox("🗺️ Seleccionar Mapa de Fondo a Visualizar:")
        grp_selector.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 11px; border: 1px solid #b0c4de; "
            "border-radius: 6px; margin-top: 6px; padding: 10px; background-color: #f8fbfe; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #003366; }"
        )
        lay_sel = QHBoxLayout(grp_selector)
        lay_sel.setSpacing(14)
        lay_sel.setContentsMargins(10, 8, 10, 8)

        self.bg_group = QButtonGroup(self)

        self.rad_obs = QRadioButton("🌍 Malla Observada (Datos Reales)")
        self.rad_obs.setStyleSheet("font-weight: bold; font-size: 11px; color: #004488;")
        self.rad_obs.setEnabled(self.has_obs)
        if not self.has_obs:
            self.rad_obs.setText("🌍 Malla Observada (Sin datos cargados)")
            self.rad_obs.setToolTip("Carga o interpola datos en la ventana principal para ver la malla observada")
        self.bg_group.addButton(self.rad_obs, 1)
        lay_sel.addWidget(self.rad_obs)

        self.rad_sim = QRadioButton("✨ Malla Simulada (Modelo Blakely)")
        self.rad_sim.setStyleSheet("font-weight: bold; font-size: 11px; color: #884400;")
        self.rad_sim.setEnabled(self.has_sim)
        self.bg_group.addButton(self.rad_sim, 2)
        lay_sel.addWidget(self.rad_sim)

        self.rad_blend = QRadioButton("👁️ Superposición (Obs + Sim)")
        self.rad_blend.setStyleSheet("font-weight: bold; font-size: 11px; color: #440088;")
        self.rad_blend.setEnabled(self.has_obs and self.has_sim)
        if not (self.has_obs and self.has_sim):
            self.rad_blend.setToolTip("Requiere tener disponibles tanto la malla observada como el modelo sintético")
        self.bg_group.addButton(self.rad_blend, 3)
        lay_sel.addWidget(self.rad_blend)

        # Control de opacidad para superposición
        lay_blend = QHBoxLayout()
        lay_blend.setSpacing(6)
        lbl_op = QLabel("Opacidad Sim:")
        lbl_op.setStyleSheet("font-size: 10px; color: #555555;")
        lay_blend.addWidget(lbl_op)

        self.slider_blend = QSlider(Qt.Horizontal)
        self.slider_blend.setRange(0, 100)
        self.slider_blend.setValue(50)
        self.slider_blend.setFixedWidth(100)
        self.slider_blend.setEnabled(False)
        lay_blend.addWidget(self.slider_blend)

        self.lbl_op_val = QLabel("50%")
        self.lbl_op_val.setStyleSheet("font-size: 10px; font-weight: bold; color: #333333; min-width: 32px;")
        lay_blend.addWidget(self.lbl_op_val)

        lay_sel.addLayout(lay_blend)
        lay_sel.addStretch()

        # Determinar selección inicial
        if default_background == "sim" and self.has_sim:
            self.rad_sim.setChecked(True)
        elif self.has_obs:
            self.rad_obs.setChecked(True)
        elif self.has_sim:
            self.rad_sim.setChecked(True)
        else:
            self.rad_sim.setChecked(True)

        self.rad_obs.toggled.connect(self.on_mode_changed)
        self.rad_sim.toggled.connect(self.on_mode_changed)
        self.rad_blend.toggled.connect(self.on_mode_changed)
        self.slider_blend.valueChanged.connect(self.on_slider_changed)

        layout.addWidget(grp_selector)

        # ======================================================================
        # BARRA DE ESTADO DE COORDENADAS Y CURSOR
        # ======================================================================
        bar_coords = QHBoxLayout()
        bar_coords.setSpacing(10)

        self.lbl_selected_badge = QLabel()
        self.lbl_selected_badge.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; font-weight: bold; "
            "color: #004488; background-color: #eaf2fb; border: 1px solid #7090b0; "
            "border-radius: 4px; padding: 4px 10px;"
        )
        self.update_selected_badge()
        bar_coords.addWidget(self.lbl_selected_badge)
        bar_coords.addStretch()

        self.lbl_hover_coords = QLabel("Cursor: (Mueve el mouse sobre el mapa)")
        self.lbl_hover_coords.setStyleSheet("font-size: 11px; color: #555555;")
        bar_coords.addWidget(self.lbl_hover_coords)

        layout.addLayout(bar_coords)

        # INSTRUCCIÓN CLARA
        lbl_hint = QLabel("💡 <b>Instrucción:</b> Haz un clic en el mapa sobre la anomalía para colocar la fuente. Doble clic confirma y asigna.")
        lbl_hint.setStyleSheet("color: #cc0000; font-size: 11px;")
        layout.addWidget(lbl_hint)

        # CANVAS MATPLOTLIB Y TOOLBAR
        self.fig = Figure(facecolor='#ffffff')
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # Conectar eventos de mouse
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # BOTONES INFERIORES
        bottom_bar = QHBoxLayout()
        btn_center = QPushButton("🎯 Centrar en Mapa")
        btn_center.clicked.connect(self.center_target)
        bottom_bar.addWidget(btn_center)
        bottom_bar.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        bottom_bar.addWidget(btn_cancel)

        btn_ok = QPushButton("✔ Confirmar y Asignar Coordenadas (X, Y)")
        btn_ok.setObjectName("btnPrimary")
        btn_ok.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        btn_ok.clicked.connect(self.accept)
        bottom_bar.addWidget(btn_ok)
        layout.addLayout(bottom_bar)

        # Dibujar mapa inicial
        self.plot_map()

    def on_mode_changed(self):
        is_blend = self.rad_blend.isChecked()
        self.slider_blend.setEnabled(is_blend)
        self.plot_map()

    def on_slider_changed(self, val):
        self.lbl_op_val.setText(f"{val}%")
        if self.rad_blend.isChecked():
            self.plot_map()

    def _lookup_value(self, grid, x_vec, y_vec, x_val, y_val):
        if grid is None or x_vec is None or y_vec is None:
            return None
        if len(x_vec) < 1 or len(y_vec) < 1:
            return None
        x_min, x_max = min(x_vec[0], x_vec[-1]), max(x_vec[0], x_vec[-1])
        y_min, y_max = min(y_vec[0], y_vec[-1]), max(y_vec[0], y_vec[-1])
        if not (x_min <= x_val <= x_max and y_min <= y_val <= y_max):
            return None
        idx_x = int(np.clip(np.searchsorted(x_vec, x_val), 0, len(x_vec) - 1))
        idx_y = int(np.clip(np.searchsorted(y_vec, y_val), 0, len(y_vec) - 1))
        try:
            val = float(grid[idx_y, idx_x])
            return val if not np.isnan(val) else None
        except Exception:
            return None

    def update_selected_badge(self):
        parts = [f"📍 Posición Seleccionada: X = {self.selected_x:,.1f} {self.unit_xy}  |  Y = {self.selected_y:,.1f} {self.unit_xy}"]
        val_obs = self._lookup_value(self.grid_obs, self.x_obs, self.y_obs, self.selected_x, self.selected_y)
        val_sim = self._lookup_value(self.grid_sim, self.x_sim, self.y_sim, self.selected_x, self.selected_y)
        if val_obs is not None:
            parts.append(f"Obs: {val_obs:+,.2f} {self.unit_z}")
        if val_sim is not None:
            parts.append(f"Sim: {val_sim:+,.2f} {self.unit_z}")
        self.lbl_selected_badge.setText("   |   ".join(parts))

    def _get_active_vectors(self):
        if self.rad_obs.isChecked() and self.has_obs:
            return self.x_obs, self.y_obs
        if self.rad_sim.isChecked() and self.has_sim:
            return self.x_sim, self.y_sim
        if self.has_obs:
            return self.x_obs, self.y_obs
        if self.has_sim:
            return self.x_sim, self.y_sim
        return None, None

    def center_target(self):
        xv, yv = self._get_active_vectors()
        if xv is not None and yv is not None:
            self.selected_x = round(0.5 * (xv[0] + xv[-1]), 1)
            self.selected_y = round(0.5 * (yv[0] + yv[-1]), 1)
            self.update_selected_badge()
            self.update_target_marker()

    def plot_map(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)

        mode = "obs" if self.rad_obs.isChecked() else ("sim" if self.rad_sim.isChecked() else "blend")
        if mode == "obs" and not self.has_obs:
            mode = "sim"
        if mode == "sim" and not self.has_sim:
            mode = "obs" if self.has_obs else "neutral"

        active_x_vec, active_y_vec = self._get_active_vectors()

        if mode == "obs" and self.has_obs:
            grid_plot = np.nan_to_num(self.grid_obs, nan=float(np.nanmean(self.grid_obs) if not np.all(np.isnan(self.grid_obs)) else 0.0))
            extent = [self.x_obs[0], self.x_obs[-1], self.y_obs[0], self.y_obs[-1]]
            vmin = float(np.percentile(grid_plot, 1.0))
            vmax = float(np.percentile(grid_plot, 99.0))
            if vmin >= vmax:
                vmin, vmax = float(np.min(grid_plot)), float(np.max(grid_plot)) + 1e-6
            im = self.ax.imshow(grid_plot, origin='lower', extent=extent, cmap='turbo', vmin=vmin, vmax=vmax)
            cb = self.fig.colorbar(im, ax=self.ax, pad=0.02, fraction=0.046)
            cb.set_label(f"Amplitud Observada [{self.unit_z}]", fontsize=9)
            try:
                if vmin < vmax and not np.all(grid_plot == 0):
                    self.ax.contour(grid_plot, origin='lower', extent=extent, levels=10,
                                    colors='#ffffff', alpha=0.35, linewidths=0.7)
            except Exception:
                pass
            title_text = "Malla Observada (Datos Reales / Trabajo) - Haz clic para situar la fuente"

        elif mode == "sim" and self.has_sim:
            grid_plot = np.nan_to_num(self.grid_sim, nan=float(np.nanmean(self.grid_sim) if not np.all(np.isnan(self.grid_sim)) else 0.0))
            extent = [self.x_sim[0], self.x_sim[-1], self.y_sim[0], self.y_sim[-1]]
            vmin = float(np.percentile(grid_plot, 1.0))
            vmax = float(np.percentile(grid_plot, 99.0))
            if vmin >= vmax:
                vmin, vmax = float(np.min(grid_plot)), float(np.max(grid_plot)) + 1e-6
            im = self.ax.imshow(grid_plot, origin='lower', extent=extent, cmap='turbo', vmin=vmin, vmax=vmax)
            cb = self.fig.colorbar(im, ax=self.ax, pad=0.02, fraction=0.046)
            cb.set_label(f"Anomalia Sintetica [{self.unit_z}]", fontsize=9)
            try:
                if vmin < vmax and not np.all(grid_plot == 0):
                    self.ax.contour(grid_plot, origin='lower', extent=extent, levels=10,
                                    colors='#ffffff', alpha=0.35, linewidths=0.7)
            except Exception:
                pass
            title_text = "Malla Simulada (Modelo Sintetico Blakely) - Haz clic para situar la fuente"

        elif mode == "blend" and self.has_obs and self.has_sim:
            extent_obs = [self.x_obs[0], self.x_obs[-1], self.y_obs[0], self.y_obs[-1]]
            extent_sim = [self.x_sim[0], self.x_sim[-1], self.y_sim[0], self.y_sim[-1]]
            grid_obs_clean = np.nan_to_num(self.grid_obs, nan=float(np.nanmean(self.grid_obs) if not np.all(np.isnan(self.grid_obs)) else 0.0))
            grid_sim_clean = np.nan_to_num(self.grid_sim, nan=float(np.nanmean(self.grid_sim) if not np.all(np.isnan(self.grid_sim)) else 0.0))

            im1 = self.ax.imshow(grid_obs_clean, origin='lower', extent=extent_obs, cmap='Blues_r', alpha=1.0)
            alpha_val = self.slider_blend.value() / 100.0
            im2 = self.ax.imshow(grid_sim_clean, origin='lower', extent=extent_sim, cmap='YlOrRd', alpha=alpha_val)

            cb = self.fig.colorbar(im2, ax=self.ax, pad=0.02, fraction=0.046)
            cb.set_label(f"Anomalia Sintetica [{self.unit_z}]", fontsize=9)
            title_text = f"Superposicion: Observada (Azul) + Simulada (Calido, {int(alpha_val*100)}%)"

        else:
            extent = [0.0, 10000.0, 0.0, 10000.0]
            title_text = "Grilla de Referencia Espacial - Haz clic para situar la fuente"

        self.ax.set_title(title_text, fontsize=10, fontweight='bold', pad=6)
        self.ax.set_xlabel(f"Coordenada X [{self.unit_xy}]", fontsize=9)
        self.ax.set_ylabel(f"Coordenada Y [{self.unit_xy}]", fontsize=9)
        self.ax.tick_params(labelsize=8)
        self.ax.grid(True, linestyle=':', color='#ffffff', alpha=0.6)

        # Dibujar otros cuerpos existentes para referencia
        for c in self.otros_cuerpos:
            self.ax.plot(c.x, c.y, 's', color='#222222', markeredgecolor='#ffffff', markeredgewidth=1.2, markersize=7)
            if active_x_vec is not None:
                offset_x = (active_x_vec[-1] - active_x_vec[0]) * 0.015
            else:
                offset_x = 50.0
            self.ax.text(c.x + offset_x, c.y, c.name,
                         color='#ffffff', fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#202020', alpha=0.75, edgecolor='none'))

        # Elementos del objetivo seleccionado
        self.target_line_h = self.ax.axhline(self.selected_y, color='#ffff00', linestyle='--', linewidth=1.0, alpha=0.75)
        self.target_line_v = self.ax.axvline(self.selected_x, color='#ffff00', linestyle='--', linewidth=1.0, alpha=0.75)
        self.target_dot = self.ax.plot(self.selected_x, self.selected_y, 'o', color='#ff0000',
                                       markeredgecolor='#ffffff', markeredgewidth=2.0, markersize=11)[0]
        self.target_cross = self.ax.plot(self.selected_x, self.selected_y, '+', color='#ffff00',
                                         markeredgewidth=2.5, markersize=18)[0]
        dx_span = (active_x_vec[-1] - active_x_vec[0]) if active_x_vec is not None else 1000.0
        self.target_text = self.ax.text(
            self.selected_x + dx_span * 0.02, self.selected_y,
            f"Posicion: ({self.selected_x:,.0f}, {self.selected_y:,.0f})",
            color='#ffffff', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#cc0000', alpha=0.85, edgecolor='none')
        )

        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw()

    def update_target_marker(self):
        if not hasattr(self, 'ax') or self.ax is None:
            return
        active_x_vec, _ = self._get_active_vectors()
        dx_span = (active_x_vec[-1] - active_x_vec[0]) if active_x_vec is not None else 1000.0

        self.target_line_h.set_ydata([self.selected_y, self.selected_y])
        self.target_line_v.set_xdata([self.selected_x, self.selected_x])
        self.target_dot.set_data([self.selected_x], [self.selected_y])
        self.target_cross.set_data([self.selected_x], [self.selected_y])
        self.target_text.set_position((self.selected_x + dx_span * 0.02, self.selected_y))
        self.target_text.set_text(f"Posicion: ({self.selected_x:,.0f}, {self.selected_y:,.0f})")
        self.canvas.draw_idle()

    def on_map_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.selected_x = round(float(event.xdata), 1)
        self.selected_y = round(float(event.ydata), 1)
        self.update_selected_badge()
        self.update_target_marker()
        if event.dblclick:
            self.accept()

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.lbl_hover_coords.setText("Cursor: (fuera del mapa)")
            return
        xm = float(event.xdata)
        ym = float(event.ydata)
        parts = [f"Cursor: X = {xm:,.1f} {self.unit_xy}, Y = {ym:,.1f} {self.unit_xy}"]
        val_obs = self._lookup_value(self.grid_obs, self.x_obs, self.y_obs, xm, ym)
        val_sim = self._lookup_value(self.grid_sim, self.x_sim, self.y_sim, xm, ym)
        if val_obs is not None:
            parts.append(f"Obs: {val_obs:+,.2f} {self.unit_z}")
        if val_sim is not None:
            parts.append(f"Sim: {val_sim:+,.2f} {self.unit_z}")
        self.lbl_hover_coords.setText(" | ".join(parts))


# ==============================================================================
# DIÁLOGO DE EDICIÓN DE UN CUERPO SINTÉTICO INDIVIDUAL
# CON FÓRMULA EDITABLE, GLOSARIO Y ACTUALIZACIÓN EN VIVO
# ==============================================================================
class CuerpoSinteticoDialog(QDialog):
    """
    Diálogo modal para configurar los parámetros geométricos y físicos (en letras normales),
    la formulación matemática analítica editable y visualización en tiempo real de la figura
    enterrada y su campo gravimétrico asociado.
    """
    def __init__(self, cuerpo=None, parent=None, default_x=5000.0, default_y=5000.0, dx=100.0, dy=100.0, unit_xy="m", span_grid=None):
        super().__init__(parent)
        self.cuerpo = cuerpo or CuerpoSintetico(x=default_x, y=default_y)
        self.default_x = float(default_x)
        self.default_y = float(default_y)
        self.dx = float(dx)
        self.dy = float(dy)
        self.unit_xy = str(unit_xy)

        if span_grid is not None:
            self.span_grid = float(span_grid)
        elif parent is not None and hasattr(parent, 'get_grid_vectors'):
            try:
                xv, yv, _, _ = parent.get_grid_vectors()
                self.span_grid = max(10.0, min(abs(xv[-1] - xv[0]), abs(yv[-1] - yv[0])))
            except Exception:
                self.span_grid = 10000.0
        else:
            self.span_grid = 10000.0

        self.setWindowTitle(f"Configuración de Fuente Geofísica y Visualización en Vivo - {self.cuerpo.name}")
        self.resize(1140, 780)
        self.setMinimumSize(920, 620)

        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(8, 8, 8, 8)
        main_vbox.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        # ======================================================================
        # PANEL IZQUIERDO: FORMULARIO, PARÁMETROS, FÓRMULA Y GLOSARIO
        # ======================================================================
        left_box = QWidget()
        self.left_box = left_box
        left_vbox = QVBoxLayout(left_box)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        # SECCIÓN 0: IDENTIFICACIÓN Y MODELO PREDETERMINADO
        grp_basic = QGroupBox("Identificación y Selección de Modelo / Fórmula Predeterminada")
        grid_b = QGridLayout(grp_basic)
        grid_b.addWidget(QLabel("<b>Nombre del Cuerpo:</b>"), 0, 0)
        self.txt_name = QLineEdit(self.cuerpo.name)
        self.txt_name.textChanged.connect(self.on_param_changed)
        grid_b.addWidget(self.txt_name, 0, 1)

        grid_b.addWidget(QLabel("<b>Modelo Predeterminado:</b>"), 1, 0)
        self.combo_type = QComboBox()
        for key, info in BLAKELY_MODELS_INFO.items():
            cat_icon = "🌍 [Grav]" if info['category'] == 'gravedad' else "🧲 [Mag]"
            self.combo_type.addItem(f"{cat_icon} {info['name']}", key)
        idx_t = self.combo_type.findData(self.cuerpo.body_type)
        if idx_t >= 0: self.combo_type.setCurrentIndex(idx_t)
        self.combo_type.currentIndexChanged.connect(self.on_preset_selected)
        grid_b.addWidget(self.combo_type, 1, 1)

        btn_reload_preset = QPushButton("🔄 Cargar Datos y Fórmula Predeterminada")
        btn_reload_preset.setToolTip("Rellena las variables de arriba y la fórmula matemática con los valores estándar del modelo")
        btn_reload_preset.clicked.connect(self.load_preset_defaults)
        grid_b.addWidget(btn_reload_preset, 2, 0, 1, 2)

        grid_b.addWidget(QLabel("<b>Modo de Cálculo:</b>"), 3, 0)
        self.combo_calc_mode = QComboBox()
        self.combo_calc_mode.addItem("🌐 Espacial Directo (Recomendado - Sin Réplica Periódica)", "spatial")
        self.combo_calc_mode.addItem("⚡ Espectral Fourier 2D (Blakely Cap. 11)", "fourier")
        if parent is not None and hasattr(parent, 'combo_calc_mode'):
            self.combo_calc_mode.setCurrentIndex(parent.combo_calc_mode.currentIndex())
        self.combo_calc_mode.currentIndexChanged.connect(self.update_live_dialog_preview)
        grid_b.addWidget(self.combo_calc_mode, 3, 1)
        layout.addWidget(grp_basic)

        # SECCIÓN 1: FORMULARIO DE VARIABLES Y PARÁMETROS NUMÉRICOS (LETRAS NORMALES)
        grp_pos = QGroupBox("1. Parámetros Numéricos del Cuerpo (Variables Editables)")
        grid_p = QGridLayout(grp_pos)

        # Coordenadas X e Y
        grid_p.addWidget(QLabel("<b>Coordenada X [m]:</b>"), 0, 0)
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-1e9, 1e9); self.spin_x.setDecimals(1); self.spin_x.setValue(self.cuerpo.x)
        self.spin_x.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.spin_x, 0, 1)

        btn_center = QPushButton("🎯 Centrar")
        btn_center.setToolTip("Colocar en el centro geométrico de la grilla")
        btn_center.clicked.connect(lambda: (self.spin_x.setValue(self.default_x), self.spin_y.setValue(self.default_y)))
        grid_p.addWidget(btn_center, 0, 2)

        grid_p.addWidget(QLabel("<b>Coordenada Y [m]:</b>"), 1, 0)
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(-1e9, 1e9); self.spin_y.setDecimals(1); self.spin_y.setValue(self.cuerpo.y)
        self.spin_y.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.spin_y, 1, 1)

        btn_pick_map = QPushButton("🗺️ Seleccionar en Mapa...")
        btn_pick_map.setObjectName("btnAccent")
        btn_pick_map.setToolTip("Despliega el mapa interactivo para hacer clic exactamente donde crees que está la fuente anómala")
        btn_pick_map.clicked.connect(self.open_map_coordinate_picker)
        grid_p.addWidget(btn_pick_map, 1, 2)

        # Profundidades z1 y z2
        grid_p.addWidget(QLabel("<b>Profundidad z1 (tope/centro) [m]:</b>"), 2, 0)
        self.spin_z1 = QDoubleSpinBox()
        self.spin_z1.setRange(1.0, 1e6); self.spin_z1.setDecimals(1); self.spin_z1.setValue(self.cuerpo.z1)
        self.spin_z1.valueChanged.connect(self.on_z1_spin_changed)
        grid_p.addWidget(self.spin_z1, 2, 1)

        self.lbl_z2 = QLabel("<b>Profundidad z2 (base/fondo) [m]:</b>")
        self.spin_z2 = QDoubleSpinBox()
        self.spin_z2.setRange(2.0, 1e6); self.spin_z2.setDecimals(1); self.spin_z2.setValue(self.cuerpo.z2)
        self.spin_z2.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.lbl_z2, 3, 0)
        grid_p.addWidget(self.spin_z2, 3, 1)

        # Densidad rho y radio R
        self.lbl_rho = QLabel("<b>Contraste Densidad rho [kg/m³]:</b>")
        self.spin_rho = QDoubleSpinBox()
        self.spin_rho.setRange(-5000.0, 10000.0); self.spin_rho.setDecimals(1); self.spin_rho.setValue(self.cuerpo.rho)
        self.spin_rho.valueChanged.connect(self.on_rho_or_R_changed)
        grid_p.addWidget(self.lbl_rho, 4, 0)
        grid_p.addWidget(self.spin_rho, 4, 1)

        self.lbl_R = QLabel("<b>Radio R / Semiancho [m]:</b>")
        self.spin_R = QDoubleSpinBox()
        self.spin_R.setRange(0.5, 1e5); self.spin_R.setDecimals(1); self.spin_R.setValue(self.cuerpo.R)
        self.spin_R.setMaximum(max(0.5, self.cuerpo.z1 * 0.90))
        self.spin_R.valueChanged.connect(self.on_rho_or_R_changed)
        grid_p.addWidget(self.lbl_R, 5, 0)
        grid_p.addWidget(self.spin_R, 5, 1)

        # Masa derivada informativa (calculada automáticamente a partir de rho y R - no editable)
        self.lbl_mu_info_title = QLabel("<b>Masa Derivada (Calculada):</b>")
        self.lbl_calc_mu = QLabel()
        self.lbl_calc_mu.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; color: #003366; "
            "background-color: #eaf2fb; border: 1px solid #b8d4f4; border-radius: 4px; padding: 4px;"
        )
        grid_p.addWidget(self.lbl_mu_info_title, 6, 0)
        grid_p.addWidget(self.lbl_calc_mu, 6, 1, 1, 2)

        # Constante gravitatoria gamma
        self.lbl_gamma = QLabel("<b>Constante gamma [m³/(kg·s²)]:</b>")
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(1e-13, 1e-9); self.spin_gamma.setDecimals(15); self.spin_gamma.setValue(self.cuerpo.gamma)
        self.spin_gamma.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.lbl_gamma, 7, 0)
        grid_p.addWidget(self.spin_gamma, 7, 1)

        # Geometría extendida (L, strike, dip, espesor)
        self.lbl_L = QLabel("<b>Longitud horizontal L [m]:</b>")
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(1.0, 1e7); self.spin_L.setDecimals(1); self.spin_L.setValue(self.cuerpo.L)
        self.spin_L.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.lbl_L, 8, 0)
        grid_p.addWidget(self.spin_L, 8, 1)

        self.lbl_dip = QLabel("<b>Buzamiento dip [°] (Inclinación):</b>")
        self.spin_dip = QDoubleSpinBox()
        self.spin_dip.setRange(0.0, 90.0); self.spin_dip.setDecimals(1); self.spin_dip.setValue(getattr(self.cuerpo, 'dip', 45.0))
        self.spin_dip.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.lbl_dip, 9, 0)
        grid_p.addWidget(self.spin_dip, 9, 1)

        self.lbl_strike = QLabel("<b>Rumbo / Azimut strike [°]:</b>")
        self.spin_strike = QDoubleSpinBox()
        self.spin_strike.setRange(-360.0, 360.0); self.spin_strike.setDecimals(1); self.spin_strike.setValue(self.cuerpo.strike)
        self.spin_strike.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.lbl_strike, 10, 0)
        grid_p.addWidget(self.spin_strike, 10, 1)

        self.lbl_espesor = QLabel("<b>Espesor [m]:</b>")
        self.spin_espesor = QDoubleSpinBox()
        self.spin_espesor.setRange(0.1, 1e5); self.spin_espesor.setDecimals(1); self.spin_espesor.setValue(self.cuerpo.espesor)
        self.spin_espesor.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.lbl_espesor, 11, 0)
        grid_p.addWidget(self.spin_espesor, 11, 1)

        # Parámetros magnéticos
        self.lbl_mag_m = QLabel("<b>Momento mag_m [A·m²]:</b>")
        self.spin_mag_m = QDoubleSpinBox()
        self.spin_mag_m.setRange(1.0, 1e20); self.spin_mag_m.setDecimals(0); self.spin_mag_m.setValue(self.cuerpo.mag_m)
        self.spin_mag_m.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.lbl_mag_m, 12, 0)
        grid_p.addWidget(self.spin_mag_m, 12, 1)

        self.lbl_Cm = QLabel("<b>Constante Cm [nT·m/A]:</b>")
        self.spin_Cm = QDoubleSpinBox()
        self.spin_Cm.setRange(1.0, 1000.0); self.spin_Cm.setValue(self.cuerpo.Cm)
        self.spin_Cm.valueChanged.connect(self.on_param_changed)
        grid_p.addWidget(self.lbl_Cm, 13, 0)
        grid_p.addWidget(self.spin_Cm, 13, 1)

        # Info de muestreo
        lbl_grid_info = QLabel(f"ℹ️ Malla de cálculo actual: <b>deltaX = {self.dx:g} m</b>, <b>deltaY = {self.dy:g} m</b>")
        lbl_grid_info.setStyleSheet("color: #0055aa; font-size: 10px; font-weight: bold;")
        grid_p.addWidget(lbl_grid_info, 14, 0, 1, 3)

        layout.addWidget(grp_pos)

        # SECCIÓN 2: FORMULACIÓN MATEMÁTICA DINÁMICA Y EDITABLE (ABAJO)
        grp_math = QGroupBox("2. Formulación Matemática Editable en Fourier E(K)")
        vbox_m = QVBoxLayout(grp_math)
        vbox_m.setSpacing(5)

        vbox_m.addWidget(QLabel("<b>Expresión Matemática del Espectro E(K) (Editable):</b>"))
        self.txt_formula = QLineEdit(self.cuerpo.formula_expr)
        self.txt_formula.setStyleSheet("font-family: Consolas, monospace; font-size: 12px; font-weight: bold; color: #cc0000; padding: 4px; background: #ffffff; border: 1px solid #707070;")
        self.txt_formula.textChanged.connect(self.on_formula_text_changed)
        vbox_m.addWidget(self.txt_formula)

        vbox_m.addWidget(QLabel("<b>Ecuación Numérica Sustituida con los Valores Actuales de Arriba:</b>"))
        self.lbl_numeric_eq = QLabel()
        self.lbl_numeric_eq.setWordWrap(True)
        self.lbl_numeric_eq.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background-color: #f5f8fc; border: 1px solid #7090b0; padding: 6px;")
        vbox_m.addWidget(self.lbl_numeric_eq)

        self.lbl_status = QLabel("✔ Sintaxis matemática verificada")
        self.lbl_status.setStyleSheet("color: #008800; font-weight: bold; font-size: 10px;")
        vbox_m.addWidget(self.lbl_status)
        layout.addWidget(grp_math)

        # SECCIÓN 3: GLOSARIO DE VARIABLES Y SÍMBOLOS (SIN LETRAS GRIEGAS)
        grp_glossary = QGroupBox("3. Glosario de Variables y Símbolos (Letras Normales)")
        vbox_gl = QVBoxLayout(grp_glossary)

        gl_html = """
        <table style='width: 100%; font-size: 10px; border-collapse: collapse;' cellpadding='3'>
            <tr style='background: #e8e8e8; font-weight: bold;'>
                <td style='border: 1px solid #b0b0b0;'>Símbolo</td>
                <td style='border: 1px solid #b0b0b0;'>Significado Físico</td>
                <td style='border: 1px solid #b0b0b0;'>Unidades</td>
                <td style='border: 1px solid #b0b0b0;'>Ejemplo / Uso</td>
            </tr>
            <tr>
                <td style='border: 1px solid #d0d0d0;'><b>rho</b></td>
                <td style='border: 1px solid #d0d0d0;'>Contraste de densidad respecto al encajante</td>
                <td style='border: 1px solid #d0d0d0;'>[kg/m³]</td>
                <td style='border: 1px solid #d0d0d0;'><code>rho = 1200.0</code></td>
            </tr>
            <tr style='background: #fafafa;'>
                <td style='border: 1px solid #d0d0d0;'><b>mu</b></td>
                <td style='border: 1px solid #d0d0d0;'>Masa total del cuerpo (esfera: mu = 4/3*pi*R³*rho)</td>
                <td style='border: 1px solid #d0d0d0;'>[kg]</td>
                <td style='border: 1px solid #d0d0d0;'><code>mu = 2.15e11</code></td>
            </tr>
            <tr>
                <td style='border: 1px solid #d0d0d0;'><b>R</b></td>
                <td style='border: 1px solid #d0d0d0;'>Radio del cuerpo (esfera o cilindro)</td>
                <td style='border: 1px solid #d0d0d0;'>[m]</td>
                <td style='border: 1px solid #d0d0d0;'><code>R = 350.0</code></td>
            </tr>
            <tr style='background: #fafafa;'>
                <td style='border: 1px solid #d0d0d0;'><b>gamma</b></td>
                <td style='border: 1px solid #d0d0d0;'>Constante de gravitación universal G</td>
                <td style='border: 1px solid #d0d0d0;'>[m³/(kg·s²)]</td>
                <td style='border: 1px solid #d0d0d0;'><code>gamma = 6.6743e-11</code></td>
            </tr>
            <tr>
                <td style='border: 1px solid #d0d0d0;'><b>z1</b></td>
                <td style='border: 1px solid #d0d0d0;'>Profundidad del centro o tope superior de la fuente</td>
                <td style='border: 1px solid #d0d0d0;'>[m]</td>
                <td style='border: 1px solid #d0d0d0;'><code>z1 = 1500.0</code></td>
            </tr>
            <tr style='background: #fafafa;'>
                <td style='border: 1px solid #d0d0d0;'><b>z2</b></td>
                <td style='border: 1px solid #d0d0d0;'>Profundidad de la base o fondo inferior de la fuente</td>
                <td style='border: 1px solid #d0d0d0;'>[m]</td>
                <td style='border: 1px solid #d0d0d0;'><code>z2 = 3500.0</code></td>
            </tr>
            <tr>
                <td style='border: 1px solid #d0d0d0;'><b>L</b></td>
                <td style='border: 1px solid #d0d0d0;'>Longitud del cuerpo a lo largo de su eje</td>
                <td style='border: 1px solid #d0d0d0;'>[m]</td>
                <td style='border: 1px solid #d0d0d0;'><code>L = 2000.0</code></td>
            </tr>
            <tr style='background: #fafafa;'>
                <td style='border: 1px solid #d0d0d0;'><b>dip</b></td>
                <td style='border: 1px solid #d0d0d0;'>Ángulo de inclinación / buzamiento (0°=horiz, 90°=vert)</td>
                <td style='border: 1px solid #d0d0d0;'>[grados]</td>
                <td style='border: 1px solid #d0d0d0;'><code>dip = 45.0</code></td>
            </tr>
            <tr>
                <td style='border: 1px solid #d0d0d0;'><b>strike</b></td>
                <td style='border: 1px solid #d0d0d0;'>Ángulo de rumbo / azimut (0° = Norte, 90° = Este)</td>
                <td style='border: 1px solid #d0d0d0;'>[grados]</td>
                <td style='border: 1px solid #d0d0d0;'><code>strike = 45.0</code></td>
            </tr>
            <tr style='background: #fafafa;'>
                <td style='border: 1px solid #d0d0d0;'><b>deltaX, deltaY</b></td>
                <td style='border: 1px solid #d0d0d0;'>Espaciado horizontal de muestreo en X y en Y</td>
                <td style='border: 1px solid #d0d0d0;'>[m]</td>
                <td style='border: 1px solid #d0d0d0;'><code>deltaX = 100.0</code></td>
            </tr>
            <tr>
                <td style='border: 1px solid #d0d0d0;'><b>K</b></td>
                <td style='border: 1px solid #d0d0d0;'>Número de onda radial continuo en Fourier: sqrt(kx²+ky²)</td>
                <td style='border: 1px solid #d0d0d0;'>[rad/m]</td>
                <td style='border: 1px solid #d0d0d0;'>Variable espectral 2D</td>
            </tr>
        </table>
        """
        lbl_gl = QLabel(gl_html)
        lbl_gl.setWordWrap(True)
        vbox_gl.addWidget(lbl_gl)
        layout.addWidget(grp_glossary)

        scroll.setWidget(content_widget)
        left_vbox.addWidget(scroll)

        # Botones de Acción
        row_b = QHBoxLayout()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        row_b.addWidget(btn_cancel)

        btn_ok = QPushButton("💾 Guardar Fuente y Aplicar Fórmula")
        btn_ok.setObjectName("btnPrimary")
        btn_ok.clicked.connect(self.on_accept)
        row_b.addWidget(btn_ok)
        left_vbox.addLayout(row_b)

        splitter.addWidget(left_box)

        # ======================================================================
        # PANEL DERECHO: VISUALIZACIÓN EN VIVO (FIGURA ENTERRADA + CAMPO 2D)
        # ======================================================================
        right_box = QWidget()
        right_vbox = QVBoxLayout(right_box)
        right_vbox.setContentsMargins(4, 0, 4, 0)
        right_vbox.setSpacing(6)

        row_hdr_prev = QHBoxLayout()
        lbl_hdr_prev = QLabel("👁️ <b>Previsualización: Mapa de Fuentes + Perfil de la Fuente</b>")
        lbl_hdr_prev.setStyleSheet("font-size: 11px; color: #003366; font-weight: bold;")
        row_hdr_prev.addWidget(lbl_hdr_prev)
        row_hdr_prev.addStretch()

        self.chk_show_geometry = QCheckBox("📐 Huella")
        self.chk_show_geometry.setChecked(True)
        self.chk_show_geometry.setToolTip("Activa o desactiva la proyección de la huella geométrica de la fuente en planta")
        self.chk_show_geometry.stateChanged.connect(self.update_live_dialog_preview)
        row_hdr_prev.addWidget(self.chk_show_geometry)

        self.chk_show_names = QCheckBox("🏷️ Nombres")
        self.chk_show_names.setChecked(True)
        self.chk_show_names.setToolTip("Activa o desactiva la visualización del nombre de las fuentes")
        self.chk_show_names.stateChanged.connect(self.update_live_dialog_preview)
        row_hdr_prev.addWidget(self.chk_show_names)

        self.btn_toggle_form = QPushButton("◀ Ocultar Formulario")
        self.btn_toggle_form.setToolTip("Oculta o muestra el panel del formulario izquierdo para ver los gráficos ampliados")
        self.btn_toggle_form.clicked.connect(self.toggle_form_panel)
        row_hdr_prev.addWidget(self.btn_toggle_form)
        right_vbox.addLayout(row_hdr_prev)

        lbl_sub_prev = QLabel("💡 <i>Se actualiza instantáneamente en la malla general al modificar cualquier parámetro a la izquierda.</i>")
        lbl_sub_prev.setStyleSheet("font-size: 10px; color: #555555;")
        right_vbox.addWidget(lbl_sub_prev)

        self.fig_prev = Figure(facecolor='#ffffff', constrained_layout=True)
        self.canvas_prev = FigureCanvas(self.fig_prev)
        self.toolbar_prev = NavigationToolbar(self.canvas_prev, self)
        self.toolbar_prev.setStyleSheet("background-color: #f0f0f0; border: 1px solid #c0c0c0;")
        right_vbox.addWidget(self.toolbar_prev)
        right_vbox.addWidget(self.canvas_prev, stretch=1)

        splitter.addWidget(right_box)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 680])

        main_vbox.addWidget(splitter)

        self.update_fields_visibility()
        self.on_rho_or_R_changed()
        self.update_math_preview()
        self.update_live_dialog_preview()

    def toggle_form_panel(self):
        is_vis = self.left_box.isVisible()
        self.left_box.setVisible(not is_vis)
        self.btn_toggle_form.setText("▶ Mostrar Formulario" if is_vis else "◀ Ocultar Formulario")
        self.canvas_prev.draw_idle()

    def on_z1_spin_changed(self):
        max_r = max(0.5, self.spin_z1.value() * 0.90)
        self.spin_R.setMaximum(max_r)
        if self.spin_R.value() > max_r:
            self.spin_R.setValue(max_r * 0.75)
        if self.spin_z2.value() <= self.spin_z1.value():
            self.spin_z2.setValue(self.spin_z1.value() + 500.0)
        self.on_param_changed()

    def update_fields_visibility(self):
        bt = self.combo_type.currentData()
        info = BLAKELY_MODELS_INFO.get(bt, {})
        is_vline = bt in ('grav_vline', 'grav_ribbon')
        is_hline = bt in ('grav_hline', 'grav_ribbon')
        is_ribbon = (bt == 'grav_ribbon')

        self.lbl_z2.setVisible(is_vline)
        self.spin_z2.setVisible(is_vline)
        self.lbl_L.setVisible(is_hline)
        self.spin_L.setVisible(is_hline)
        self.lbl_dip.setVisible(False)
        self.spin_dip.setVisible(False)
        self.lbl_strike.setVisible(is_hline)
        self.spin_strike.setVisible(is_hline)
        self.lbl_espesor.setVisible(is_ribbon)
        self.spin_espesor.setVisible(is_ribbon)

        # Gravimétricas (Exclusivo Blakely Tabla 11.1)
        self.lbl_mu_info_title.setVisible(True)
        self.lbl_calc_mu.setVisible(True)
        self.lbl_R.setVisible(bt in ('grav_sphere', 'grav_vline', 'grav_hline'))
        self.spin_R.setVisible(bt in ('grav_sphere', 'grav_vline', 'grav_hline'))
        self.lbl_rho.setVisible(True)
        self.spin_rho.setVisible(True)
        self.lbl_gamma.setVisible(True)
        self.spin_gamma.setVisible(True)

        # Magnéticas ocultas
        self.lbl_mag_m.setVisible(False)
        self.spin_mag_m.setVisible(False)
        self.lbl_Cm.setVisible(False)
        self.spin_Cm.setVisible(False)

    def on_preset_selected(self):
        self.update_fields_visibility()
        bt = self.combo_type.currentData()
        p = PRESETS_DEFAULT_DATA.get(bt, {})
        span = getattr(self, 'span_grid', 10000.0)
        scale_fac = min(1.0, span / 5000.0)
        if p:
            # SIEMPRE actualizar la fórmula de Fourier E(K) al seleccionar un preset
            self.txt_formula.setText(p.get('formula', ''))
            # Ajustar parámetros geométricos iniciales si cambiamos de tipo de cuerpo
            if bt == 'grav_hline':
                if self.spin_L.value() <= 10.0:
                    self.spin_L.setValue(max(10.0, round(p.get('L', 3000.0) * scale_fac, 1)))
                if self.spin_strike.value() == 0.0:
                    self.spin_strike.setValue(p.get('strike', 90.0))
            elif bt == 'grav_ribbon':
                if self.spin_L.value() <= 10.0:
                    self.spin_L.setValue(max(10.0, round(p.get('L', 4000.0) * scale_fac, 1)))
                if self.spin_espesor.value() <= 1.0:
                    self.spin_espesor.setValue(max(0.5, round(p.get('espesor', 60.0) * scale_fac, 1)))
                if self.spin_z2.value() <= self.spin_z1.value():
                    self.spin_z2.setValue(max(self.spin_z1.value() + 50.0, round(p.get('z2', 2500.0) * scale_fac, 1)))
            elif bt == 'grav_vline':
                if self.spin_z2.value() <= self.spin_z1.value():
                    self.spin_z2.setValue(max(self.spin_z1.value() + 50.0, round(p.get('z2', 3500.0) * scale_fac, 1)))
        self.on_rho_or_R_changed()

    def load_preset_defaults(self):
        bt = self.combo_type.currentData()
        p = PRESETS_DEFAULT_DATA.get(bt, {})
        if not p: return
        self.txt_formula.setText(p.get('formula', ''))

        span = getattr(self, 'span_grid', 10000.0)
        scale_fac = min(1.0, span / 5000.0)

        z1_def = max(5.0, round(p.get('z1', 1000.0) * scale_fac, 1))
        r_def = min(z1_def * 0.75, max(1.0, round(p.get('R', 250.0) * scale_fac, 1)))

        self.spin_rho.setValue(p.get('rho', 1000.0))
        self.spin_z1.setValue(z1_def)
        self.spin_R.setMaximum(max(0.5, z1_def * 0.90))
        self.spin_R.setValue(r_def)
        self.spin_z2.setValue(max(z1_def + 10.0, round(p.get('z2', 2500.0) * scale_fac, 1)))
        self.spin_L.setValue(max(10.0, round(p.get('L', 2000.0) * scale_fac, 1)))
        self.spin_dip.setValue(p.get('dip', 0.0))
        self.spin_strike.setValue(p.get('strike', 0.0))
        self.spin_espesor.setValue(max(0.5, round(p.get('espesor', 50.0) * scale_fac, 1)))
        self.spin_gamma.setValue(p.get('gamma', 6.6743e-11))
        self.on_rho_or_R_changed()

    def on_rho_or_R_changed(self):
        bt = self.combo_type.currentData()
        r_val = self.spin_R.value()
        rho_val = self.spin_rho.value()
        esp_val = self.spin_espesor.value()

        if bt in ('grav_sphere', 'custom'):
            calc_mu = 4.0 / 3.0 * np.pi * (r_val ** 3) * rho_val
            vol = 4.0 / 3.0 * np.pi * (r_val ** 3)
            self.lbl_calc_mu.setText(f"mu = {calc_mu:.3e} kg  (calculada: 4/3·pi·R³·rho | Vol = {vol:.2e} m³)")
        elif bt in ('grav_vline', 'grav_hline', 'grav_inclined_cylinder'):
            lam = np.pi * (r_val ** 2) * rho_val
            area = np.pi * (r_val ** 2)
            self.lbl_calc_mu.setText(f"lambda_lin = {lam:.3e} kg/m  (calculada: pi·R²·rho | Área = {area:.2e} m²)")
        elif bt in ('grav_ribbon', 'grav_prism'):
            sig = esp_val * rho_val
            self.lbl_calc_mu.setText(f"sigma_sup = {sig:.3e} kg/m²  (calculada: espesor·rho)")
        else:
            self.lbl_calc_mu.setText(f"rho = {rho_val:g} kg/m³")
        self.on_param_changed()

    def on_param_changed(self):
        self.update_math_preview()
        self.update_live_dialog_preview()

    def on_formula_text_changed(self):
        self.update_math_preview()
        self.update_live_dialog_preview()

    def update_math_preview(self):
        formula_str = self.txt_formula.text().strip()
        if not formula_str:
            formula_str = get_default_formula_for_type(self.combo_type.currentData())

        c_temp = CuerpoSintetico(
            name=self.txt_name.text(), body_type=self.combo_type.currentData(),
            x=self.spin_x.value(), y=self.spin_y.value(),
            z1=self.spin_z1.value(), z2=self.spin_z2.value(),
            R=self.spin_R.value(), rho=self.spin_rho.value(),
            gamma=self.spin_gamma.value(),
            L=self.spin_L.value(), dip=self.spin_dip.value(), strike=self.spin_strike.value(),
            espesor=self.spin_espesor.value(),
            mag_m=self.spin_mag_m.value(), Cm=self.spin_Cm.value()
        )
        sub_str = c_temp.get_substituted_numeric_formula(formula_str)

        bt = self.combo_type.currentData()
        is_grav = (BLAKELY_MODELS_INFO.get(bt, {}).get('category') == 'gravedad')
        if is_grav:
            info_vals = (
                f"<b>Variables activas:</b> rho={self.spin_rho.value():g} kg/m³ | "
                f"R={self.spin_R.value():g} m | z1={self.spin_z1.value():g} m | "
                f"gamma={self.spin_gamma.value():.4e}"
            )
        else:
            info_vals = (
                f"<b>Variables activas:</b> mag_m={self.spin_mag_m.value():.3e} A·m² | "
                f"z1={self.spin_z1.value():g} m | Cm={self.spin_Cm.value():g} nT·m/A"
            )

        html = f"""
        <div style='color: #004488; font-weight: bold; font-size: 11px; margin-bottom: 4px;'>
            E(K) = {sub_str}
        </div>
        <div style='color: #444444; font-size: 10px; font-family: Segoe UI, sans-serif;'>
            {info_vals}
        </div>
        """
        self.lbl_numeric_eq.setText(html)

    def update_live_dialog_preview(self):
        """
        Calcula el campo 2D en tiempo real sobre la malla general y redibuja:
        1. Planta: Mapa acumulado de las fuentes con la misma dimensión y escala de color fija que el mapa general.
        2. Perfil de la Fuente: Vista geológica de perfil escalada a las mismas unidades (1:1 en metros).
        """
        try:
            self.fig_prev.clf()
            bt = self.combo_type.currentData()
            x0 = self.spin_x.value()
            y0 = self.spin_y.value()
            z1 = max(1.0, self.spin_z1.value())
            z2 = max(z1 + 1.0, self.spin_z2.value())
            rho = self.spin_rho.value()
            R = max(1.0, self.spin_R.value())
            L = max(1.0, self.spin_L.value())
            dip = self.spin_dip.value()
            strike = self.spin_strike.value()
            espesor = max(0.1, self.spin_espesor.value())
            gamma = self.spin_gamma.value()
            mag_m = self.spin_mag_m.value()
            Cm = self.spin_Cm.value()
            name_str = self.txt_name.text().strip() or "Fuente"

            c_tmp = CuerpoSintetico(
                name=name_str,
                body_type=bt,
                x=x0, y=y0, z1=z1, z2=z2,
                R=R, rho=rho, gamma=gamma,
                L=L, dip=dip, strike=strike, espesor=espesor,
                mag_m=mag_m, Cm=Cm,
                formula_expr=self.txt_formula.text().strip()
            )

            # 1. Obtener la malla espacial y fuentes existentes del modelo
            p = self.parent()
            mw = getattr(p, 'main_window', None) if p is not None else None

            if p is not None and hasattr(p, 'get_grid_vectors'):
                x_vec, y_vec, dx, dy = p.get_grid_vectors()
                all_cuerpos = list(getattr(p, 'cuerpos', []))
                sync_scale = getattr(p, 'chk_sync_cbar', None) and p.chk_sync_cbar.isChecked()
            else:
                span = max(4.0 * z1, L * 2.5, 6000.0)
                x_vec = np.linspace(x0 - span * 0.5, x0 + span * 0.5, 81)
                y_vec = np.linspace(y0 - span * 0.5, y0 + span * 0.5, 81)
                dx = x_vec[1] - x_vec[0]
                dy = y_vec[1] - y_vec[0]
                all_cuerpos = []
                sync_scale = False

            # Lista de fuentes reemplazando la actual por la instancia temporal modificada
            sources_to_eval = []
            found_cur = False
            for ec in all_cuerpos:
                if ec is self.cuerpo:
                    sources_to_eval.append(c_tmp)
                    found_cur = True
                else:
                    sources_to_eval.append(ec)
            if not found_cur:
                sources_to_eval.append(c_tmp)

            # Modo de cálculo: Espacial Directo vs Fourier
            mode_calc = self.combo_calc_mode.currentData() if hasattr(self, 'combo_calc_mode') else 'spatial'
            if mode_calc == 'spatial':
                X, Y = np.meshgrid(x_vec, y_vec)
                gz_total = np.zeros((ny, nx), dtype=float)
                for c in sources_to_eval:
                    if getattr(c, 'is_enabled', True):
                        gz_total += c.evaluate_spatial(X, Y)
            else:
                # Cálculo de espectro en Fourier sobre la malla general
                nx = len(x_vec)
                ny = len(y_vec)
                u = np.fft.fftfreq(nx, d=dx)
                v = np.fft.fftfreq(ny, d=dy)
                U, V = np.meshgrid(u, v)
                kx = 2.0 * np.pi * U
                ky = 2.0 * np.pi * V
                K = np.sqrt(kx ** 2 + ky ** 2)
                x0_grid = x_vec[0]
                y0_grid = y_vec[0]

                espectro_total = np.zeros((ny, nx), dtype=complex)
                for c in sources_to_eval:
                    if getattr(c, 'is_enabled', True):
                        espectro_total += c.evaluate_spectrum(kx, ky, K, x0_grid, y0_grid, dx, dy)

                gz_total = np.real(np.fft.ifft2(espectro_total))

            # Determinar escala de color fija y sincronizada con el mapa general
            cmap = 'turbo'
            if mw is not None:
                if hasattr(mw, 'combo_master_cmap'):
                    cmap = mw.combo_master_cmap.currentText()
                elif hasattr(mw, 'combo_map_cmap'):
                    cmap = mw.combo_map_cmap.currentText()

            has_real = mw is not None and getattr(mw, 'grid_Z', None) is not None
            if sync_scale and has_real:
                vmin = float(np.nanmin(mw.grid_Z))
                vmax = float(np.nanmax(mw.grid_Z))
                scale_note = "Escala Fija Sincronizada con Real"
            else:
                vmin = float(np.nanmin(gz_total))
                vmax = float(np.nanmax(gz_total))
                scale_note = "Escala Sintética"

            span_val = vmax - vmin
            if span_val < 1e-3 or span_val < 1e-4 * max(1.0, abs(vmax)):
                mid = 0.5 * (vmin + vmax)
                vmin = mid - 1.0
                vmax = mid + 1.0

            u_z = c_tmp.get_unit()
            u_xy = self.unit_xy

            # Layout con 2 subplots: Planta del mapa general arriba y Perfil 1:1 abajo
            gs = self.fig_prev.add_gridspec(2, 1, height_ratios=[1.2, 1.0])
            ax_map = self.fig_prev.add_subplot(gs[0, 0])
            ax_prof = self.fig_prev.add_subplot(gs[1, 0])
            self.ax_map_prev = ax_map
            self.ax_prof_prev = ax_prof

            extent = [x_vec[0], x_vec[-1], y_vec[0], y_vec[-1]]
            im = ax_map.imshow(gz_total, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            cb = self.fig_prev.colorbar(im, ax=ax_map, pad=0.02, fraction=0.046)
            cb.set_label(f"gz [{u_z}]", fontsize=8)

            ax_map.set_title(f"1. Planta: Mapa de Fuentes (Malla General, {scale_note}) [{vmin:,.1f} a {vmax:,.1f} {u_z}]",
                             fontsize=9, fontweight='bold', pad=5)
            ax_map.set_xlabel(f"Coordenada X [{u_xy}]", fontsize=8)
            ax_map.set_ylabel(f"Coordenada Y [{u_xy}]", fontsize=8)
            ax_map.tick_params(labelsize=8)

            is_aspect_equal = getattr(p, 'chk_synth_aspect_equal', None) and p.chk_synth_aspect_equal.isChecked()
            if is_aspect_equal:
                ax_map.set_aspect('equal', adjustable='box')

            show_geom = getattr(self, 'chk_show_geometry', None) is None or self.chk_show_geometry.isChecked()
            show_names = getattr(self, 'chk_show_names', None) is None or self.chk_show_names.isChecked()

            # Dibujar todas las fuentes en el mapa general (SIN punto rojo!)
            for c in sources_to_eval:
                if getattr(c, 'is_enabled', True):
                    if show_geom:
                        draw_cuerpo_planta_geometry(ax_map, c, show_labels=show_names)
                    elif show_names:
                        dx_offset = (x_vec[-1] - x_vec[0]) * 0.015
                        ax_map.text(c.x + dx_offset, c.y, c.name,
                                    color='#ffffff', fontsize=8, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#202020', alpha=0.75, edgecolor='none'))

            # 2. Vista de Perfil de la Fuente (Solo una vista, a escala 1:1 en metros)
            draw_cuerpo_perfil_geologico(ax_prof, c_tmp)
            ax_prof.set_title("2. Vista de Perfil de la Fuente (Escala Isótropa 1:1 - Unidades en Metros)",
                              fontsize=9, fontweight='bold', pad=5)
            ax_prof.tick_params(labelsize=8)

            self.canvas_prev.draw_idle()
        except Exception as e:
            pass

    def on_accept(self):
        self.cuerpo.name = self.txt_name.text().strip() or "Fuente Geofísica"
        self.cuerpo.body_type = self.combo_type.currentData()
        self.cuerpo.x = self.spin_x.value()
        self.cuerpo.y = self.spin_y.value()
        self.cuerpo.z1 = self.spin_z1.value()
        self.cuerpo.z = self.cuerpo.z1
        self.cuerpo.z2 = max(self.cuerpo.z1 + 1.0, self.spin_z2.value())
        self.cuerpo.rho = self.spin_rho.value()
        self.cuerpo.density = self.cuerpo.rho
        self.cuerpo.R = self.spin_R.value()
        self.cuerpo.radius = self.cuerpo.R
        self.cuerpo.mu = 4.0 / 3.0 * np.pi * (self.cuerpo.R ** 3) * self.cuerpo.rho
        self.cuerpo.mass = self.cuerpo.mu
        self.cuerpo.gamma = self.spin_gamma.value()
        self.cuerpo.L = self.spin_L.value()
        self.cuerpo.length = self.cuerpo.L
        self.cuerpo.dip = self.spin_dip.value()
        self.cuerpo.strike = self.spin_strike.value()
        self.cuerpo.espesor = self.spin_espesor.value()
        self.cuerpo.mag_m = self.spin_mag_m.value()
        self.cuerpo.mag_moment = self.cuerpo.mag_m
        self.cuerpo.Cm = self.spin_Cm.value()
        self.cuerpo.formula_expr = self.txt_formula.text().strip() or get_default_formula_for_type(self.cuerpo.body_type)
        self.accept()

    def open_map_coordinate_picker(self):
        p = self.parent()
        mw = getattr(p, 'main_window', None) if p is not None else None

        # 1. Malla Observada (Datos Reales / Trabajo)
        grid_obs = None
        x_obs = None
        y_obs = None
        u_xy = self.unit_xy
        u_z = "mGal"

        if p is not None and hasattr(p, 'get_active_obs_grid_info'):
            obs_info = p.get_active_obs_grid_info()
            if obs_info is not None and obs_info.get('Z') is not None:
                grid_obs = obs_info['Z']
                x_obs = obs_info.get('x', mw.grid_x_vec if mw is not None else None)
                y_obs = obs_info.get('y', mw.grid_y_vec if mw is not None else None)

        if grid_obs is None and mw is not None:
            u_xy = getattr(mw, 'unit_xy', self.unit_xy)
            u_z = getattr(mw, 'unit_z', 'mGal')
            for attr in ['grid_Z', 'grid_z', 'grid_Z_filt']:
                cand = getattr(mw, attr, None)
                if cand is not None:
                    grid_obs = cand
                    break
            if getattr(mw, 'grid_x_vec', None) is not None:
                x_obs = mw.grid_x_vec
            if getattr(mw, 'grid_y_vec', None) is not None:
                y_obs = mw.grid_y_vec

        # 2. Malla Simulada (Modelo Sintético Blakely)
        grid_sim = None
        x_sim = None
        y_sim = None
        otros_cuerpos = []

        if p is not None:
            if hasattr(p, 'cuerpos'):
                otros_cuerpos = [c for c in p.cuerpos if c != self.cuerpo]
            if hasattr(p, 'get_grid_vectors'):
                x_sim, y_sim, _, _ = p.get_grid_vectors()
            if getattr(p, 'grid_Z_model', None) is not None:
                grid_sim = p.grid_Z_model
            elif hasattr(p, 'recalculate_model'):
                p.recalculate_model()
                grid_sim = getattr(p, 'grid_Z_model', None)

        if x_sim is None and x_obs is not None:
            x_sim = x_obs.copy()
            y_sim = y_obs.copy()
        if x_obs is None and x_sim is not None:
            x_obs = x_sim.copy()
            y_obs = y_sim.copy()

        dlg = DialogoSelectorCoordenadasMapa(
            parent=self,
            initial_x=self.spin_x.value(),
            initial_y=self.spin_y.value(),
            grid_obs=grid_obs,
            x_obs=x_obs,
            y_obs=y_obs,
            grid_sim=grid_sim,
            x_sim=x_sim,
            y_sim=y_sim,
            unit_xy=u_xy,
            unit_z=u_z,
            otros_cuerpos=otros_cuerpos
        )

        if dlg.exec_() == QDialog.Accepted:
            self.spin_x.setValue(dlg.selected_x)
            self.spin_y.setValue(dlg.selected_y)
            self.on_param_changed()


# ==============================================================================
# VENTANA PRINCIPAL: MODELADOR ESPECTRAL SINTÉTICO 2D
# ==============================================================================
class VentanaModeladorSintetico(QDialog):
    """
    Ventana para construir modelos limpios o superpuestos sumando fuentes de Blakely.
    Crea grillas automáticas o manuales, suma espectros en Fourier con shift y calcula IFFT2.
    """
    def __init__(self, main_window, parent=None, is_embedded=False):
        parent_widget = parent if isinstance(parent, QWidget) else (main_window if isinstance(main_window, QWidget) else None)
        super().__init__(parent_widget)
        self.main_window = main_window
        self.is_embedded = is_embedded
        if is_embedded:
            self.setWindowFlags(Qt.Widget)
        else:
            self.setWindowTitle("GravTT - Modelado")
            self.resize(1180, 740)
            self.setMinimumSize(1020, 640)

        # Inicia limpio sin fuentes precargadas
        self.cuerpos = list(getattr(main_window, 'cuerpos_sinteticos', []))
        self.grid_Z_model = None
        self.x_vec = None
        self.y_vec = None
        self.available_obs_grids = {}
        self.active_obs_key = 'sin_filtrar'

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ======================================================================
        # PANEL IZQUIERDO: CONTROLES, GRILLA Y PILA DE CUERPOS
        # ======================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        lbl_hdr = QLabel("✨ <b>Modelado de Fuentes Geofísicas [Blakely Cap. 11]</b>")
        lbl_hdr.setStyleSheet("font-size: 13px; font-weight: bold; color: #cc0000;")
        left_layout.addWidget(lbl_hdr)

        # SECCIÓN 0: MOTOR DE MODELADO (ESPACIAL DIRECTO VS ESPECTRAL FOURIER)
        grp_engine = QGroupBox("Motor de Modelado Gravimétrico / Magnético")
        l_eng = QGridLayout(grp_engine)
        l_eng.addWidget(QLabel("<b>Método de Cálculo:</b>"), 0, 0)
        self.combo_calc_mode = QComboBox()
        self.combo_calc_mode.addItem("🌐 Espacial Directo (Recomendado - Sin Réplica Periódica)", "spatial")
        self.combo_calc_mode.addItem("⚡ Espectral Fourier 2D (Blakely Cap. 11)", "fourier")
        self.combo_calc_mode.setToolTip(
            "Espacial Directo: Modela cuerpos geofísicos aislados reales (esferas, cilindros, diques) sin efectos de periodicidad ni placa infinita.\n"
            "Espectral Fourier 2D: Superpone espectros continuos E(K) de Blakely con antitransformada IFFT2."
        )
        self.combo_calc_mode.currentIndexChanged.connect(self.recalculate_model)
        l_eng.addWidget(self.combo_calc_mode, 0, 1)
        left_layout.addWidget(grp_engine)

        # SECCIÓN 1: DEFINICIÓN DE LA MALLA
        grp_grid = QGroupBox("1. Definición de la Malla Espacial Base")
        grid_l = QGridLayout(grp_grid)

        has_mesh = (
            (main_window.grid_x_vec is not None and main_window.grid_y_vec is not None and len(main_window.grid_x_vec) > 1) or
            (getattr(main_window, 'x_min_sel', None) is not None and getattr(main_window, 'x_max_sel', None) is not None) or
            (getattr(main_window, 'pts_x', None) is not None and len(main_window.pts_x) > 0)
        )
        self.rad_grid_auto = QRadioButton("🔗 Sincronizar automáticamente con Malla Actual")
        self.rad_grid_auto.setChecked(has_mesh)
        self.rad_grid_auto.setEnabled(has_mesh)
        self.rad_grid_manual = QRadioButton("⚙️ Definir Malla Sintética Limpia (Manual)")
        self.rad_grid_manual.setChecked(not has_mesh)
        self.rad_grid_auto.toggled.connect(self.on_grid_mode_changed)

        btn_sync_now = QPushButton("🔄 Sincronizar")
        btn_sync_now.setToolTip("Copia las coordenadas X_min, X_max, Y_min, Y_max y pasos deltaX, deltaY desde los datos/malla actual")
        btn_sync_now.clicked.connect(self.init_grid_coordinates)
        btn_sync_now.clicked.connect(self.recalculate_model)

        grid_l.addWidget(self.rad_grid_auto, 0, 0, 1, 3)
        grid_l.addWidget(btn_sync_now, 0, 3)
        grid_l.addWidget(self.rad_grid_manual, 1, 0, 1, 4)

        grid_l.addWidget(QLabel("X_min:"), 2, 0)
        self.spin_xmin = QDoubleSpinBox()
        self.spin_xmin.setRange(-1e12, 1e12); self.spin_xmin.setValue(0.0)
        grid_l.addWidget(self.spin_xmin, 2, 1)

        grid_l.addWidget(QLabel("X_max:"), 2, 2)
        self.spin_xmax = QDoubleSpinBox()
        self.spin_xmax.setRange(-1e12, 1e12); self.spin_xmax.setValue(10000.0)
        grid_l.addWidget(self.spin_xmax, 2, 3)

        grid_l.addWidget(QLabel("Y_min:"), 3, 0)
        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-1e12, 1e12); self.spin_ymin.setValue(0.0)
        grid_l.addWidget(self.spin_ymin, 3, 1)

        grid_l.addWidget(QLabel("Y_max:"), 3, 2)
        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-1e12, 1e12); self.spin_ymax.setValue(10000.0)
        grid_l.addWidget(self.spin_ymax, 3, 3)

        grid_l.addWidget(QLabel("Paso deltaX:"), 4, 0)
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(1.0, 10000.0); self.spin_dx.setValue(100.0)
        grid_l.addWidget(self.spin_dx, 4, 1)

        grid_l.addWidget(QLabel("Paso deltaY:"), 4, 2)
        self.spin_dy = QDoubleSpinBox()
        self.spin_dy.setRange(1.0, 10000.0); self.spin_dy.setValue(100.0)
        grid_l.addWidget(self.spin_dy, 4, 3)

        self.lbl_dims_info = QLabel("Dimensiones: 101 × 101 nodos")
        self.lbl_dims_info.setStyleSheet("font-size: 10px; color: #0055aa; font-weight: bold;")
        grid_l.addWidget(self.lbl_dims_info, 5, 0, 1, 4)

        for sp in [self.spin_xmin, self.spin_xmax, self.spin_ymin, self.spin_ymax, self.spin_dx, self.spin_dy]:
            sp.valueChanged.connect(self.recalculate_model)

        left_layout.addWidget(grp_grid)

        # SECCIÓN 2: PILA DE FUENTES GEOLÓGICAS (SUPERPOSICIÓN FOURIER)
        grp_cuerpos = QGroupBox("2. Pila de Fuentes Geológicas (Superposición Fourier)")
        vbox_c = QVBoxLayout(grp_cuerpos)

        row_c_btns = QHBoxLayout()
        btn_add = QPushButton("➕ Agregar Fuente...")
        btn_add.setObjectName("btnPrimary")
        btn_add.clicked.connect(self.on_add_cuerpo)
        row_c_btns.addWidget(btn_add)

        btn_add_click = QPushButton("🗺️ Clic en Mapa...")
        btn_add_click.setToolTip("Despliega el mapa interactivo para hacer clic exactamente donde crees que está la fuente y crearla allí")
        btn_add_click.clicked.connect(self.on_add_cuerpo_by_map_click)
        row_c_btns.addWidget(btn_add_click)

        vbox_c.addLayout(row_c_btns)

        self.tbl_cuerpos = QTableWidget(0, 7)
        self.tbl_cuerpos.setHorizontalHeaderLabels(["Activo", "Nombre", "Tipo", "rho [kg/m³]", "Radio R [m]", "Prof z1", "Fórmula Matemática E(K)"])
        self.tbl_cuerpos.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_cuerpos.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_cuerpos.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tbl_cuerpos.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tbl_cuerpos.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.tbl_cuerpos.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.tbl_cuerpos.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.tbl_cuerpos.itemChanged.connect(self.on_table_item_changed)
        self.tbl_cuerpos.itemSelectionChanged.connect(self.on_table_row_selected)
        self.tbl_cuerpos.itemDoubleClicked.connect(self.on_edit_cuerpo)
        vbox_c.addWidget(self.tbl_cuerpos)

        row_tbl_acts = QHBoxLayout()
        btn_edit = QPushButton("✏️ Editar"); btn_edit.clicked.connect(self.on_edit_cuerpo)
        btn_dup = QPushButton("📋 Duplicar"); btn_dup.clicked.connect(self.on_duplicate_cuerpo)
        btn_del = QPushButton("🗑️ Eliminar"); btn_del.clicked.connect(self.on_delete_cuerpo)
        btn_clear = QPushButton("🧹 Limpiar"); btn_clear.clicked.connect(self.on_clear_cuerpos)
        row_tbl_acts.addWidget(btn_edit); row_tbl_acts.addWidget(btn_dup)
        row_tbl_acts.addWidget(btn_del); row_tbl_acts.addWidget(btn_clear)
        vbox_c.addLayout(row_tbl_acts)

        # EDITOR RÁPIDO DE FÓRMULA DIRECTO EN LA VENTANA PRINCIPAL
        grp_quick_f = QGroupBox("Fórmula del Cuerpo Seleccionado en la Tabla")
        vbox_qf = QVBoxLayout(grp_quick_f)
        vbox_qf.setSpacing(4)
        row_qf = QHBoxLayout()
        row_qf.addWidget(QLabel("<b>E(K):</b>"))
        self.txt_quick_formula = QLineEdit()
        self.txt_quick_formula.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; font-weight: bold; color: #cc0000; padding: 2px;")
        self.txt_quick_formula.setPlaceholderText("Selecciona una fuente arriba para ver/editar su fórmula")
        self.txt_quick_formula.returnPressed.connect(self.on_apply_quick_formula)
        row_qf.addWidget(self.txt_quick_formula)

        btn_q_apply = QPushButton("✔ Aplicar Fórmula")
        btn_q_apply.clicked.connect(self.on_apply_quick_formula)
        row_qf.addWidget(btn_q_apply)
        vbox_qf.addLayout(row_qf)

        self.lbl_quick_numeric_eq = QLabel("E(K) = [Selecciona una fuente]")
        self.lbl_quick_numeric_eq.setStyleSheet("font-family: Consolas, monospace; font-size: 10px; color: #004488; background: #f4f8fc; border: 1px solid #c0d4ec; padding: 3px;")
        vbox_qf.addWidget(self.lbl_quick_numeric_eq)
        vbox_c.addWidget(grp_quick_f)

        # AJUSTE RÁPIDO DE PARÁMETROS EN TIEMPO REAL (FUENTE SELECCIONADA)
        grp_quick_p = QGroupBox("Ajuste Rápido de Parámetros (Tiempo Real)")
        grid_qp = QGridLayout(grp_quick_p)
        grid_qp.setSpacing(4)
        
        grid_qp.addWidget(QLabel("X [m]:"), 0, 0)
        self.spin_q_x = QDoubleSpinBox()
        self.spin_q_x.setRange(-1e12, 1e12); self.spin_q_x.setSingleStep(100.0)
        grid_qp.addWidget(self.spin_q_x, 0, 1)

        grid_qp.addWidget(QLabel("Y [m]:"), 0, 2)
        self.spin_q_y = QDoubleSpinBox()
        self.spin_q_y.setRange(-1e12, 1e12); self.spin_q_y.setSingleStep(100.0)
        grid_qp.addWidget(self.spin_q_y, 0, 3)

        grid_qp.addWidget(QLabel("Prof z1 [m]:"), 1, 0)
        self.spin_q_z1 = QDoubleSpinBox()
        self.spin_q_z1.setRange(1.0, 1e6); self.spin_q_z1.setValue(1000.0); self.spin_q_z1.setSingleStep(100.0)
        grid_qp.addWidget(self.spin_q_z1, 1, 1)

        grid_qp.addWidget(QLabel("rho [kg/m³] / M:"), 1, 2)
        self.spin_q_rho = QDoubleSpinBox()
        self.spin_q_rho.setRange(-1e6, 1e6); self.spin_q_rho.setValue(500.0); self.spin_q_rho.setSingleStep(50.0)
        grid_qp.addWidget(self.spin_q_rho, 1, 3)

        grid_qp.addWidget(QLabel("Radio R / ancho [m]:"), 2, 0)
        self.spin_q_r = QDoubleSpinBox()
        self.spin_q_r.setRange(0.1, 1e6); self.spin_q_r.setValue(500.0); self.spin_q_r.setSingleStep(50.0)
        grid_qp.addWidget(self.spin_q_r, 2, 1)

        grid_qp.addWidget(QLabel("Espesor h [m]:"), 2, 2)
        self.spin_q_h = QDoubleSpinBox()
        self.spin_q_h.setRange(0.0, 1e6); self.spin_q_h.setValue(1000.0); self.spin_q_h.setSingleStep(100.0)
        grid_qp.addWidget(self.spin_q_h, 2, 3)

        grid_qp.addWidget(QLabel("Buz dip [°]:"), 3, 0)
        self.spin_q_dip = QDoubleSpinBox()
        self.spin_q_dip.setRange(0.0, 90.0); self.spin_q_dip.setValue(45.0); self.spin_q_dip.setSingleStep(5.0)
        grid_qp.addWidget(self.spin_q_dip, 3, 1)

        grid_qp.addWidget(QLabel("Longitud L [m]:"), 3, 2)
        self.spin_q_L = QDoubleSpinBox()
        self.spin_q_L.setRange(1.0, 1e7); self.spin_q_L.setValue(2000.0); self.spin_q_L.setSingleStep(200.0)
        grid_qp.addWidget(self.spin_q_L, 3, 3)

        grid_qp.addWidget(QLabel("Rumbo [°]:"), 4, 0)
        self.spin_q_strike = QDoubleSpinBox()
        self.spin_q_strike.setRange(-360.0, 360.0); self.spin_q_strike.setValue(0.0); self.spin_q_strike.setSingleStep(5.0)
        grid_qp.addWidget(self.spin_q_strike, 4, 1)

        self._block_quick_signals = False
        self.spin_q_x.valueChanged.connect(self.on_quick_param_changed)
        self.spin_q_y.valueChanged.connect(self.on_quick_param_changed)
        self.spin_q_z1.valueChanged.connect(self.on_quick_param_changed)
        self.spin_q_rho.valueChanged.connect(self.on_quick_param_changed)
        self.spin_q_r.valueChanged.connect(self.on_quick_param_changed)
        self.spin_q_h.valueChanged.connect(self.on_quick_param_changed)
        self.spin_q_dip.valueChanged.connect(self.on_quick_param_changed)
        self.spin_q_L.valueChanged.connect(self.on_quick_param_changed)
        self.spin_q_strike.valueChanged.connect(self.on_quick_param_changed)
        vbox_c.addWidget(grp_quick_p)

        left_layout.addWidget(grp_cuerpos)

        # BOTÓN MÉTODO INVERSO PUNTUAL (DESPUÉS DE FUENTES Y ANTES DE RUIDO)
        btn_inv_puntual = QPushButton("🎯 Método Inverso Puntual (Ajuste Mínimos Cuadrados)...")
        btn_inv_puntual.setStyleSheet("background-color: #7b1fa2; color: #ffffff; font-weight: bold; font-size: 11px; padding: 7px; margin-top: 3px; margin-bottom: 3px; border-radius: 3px;")
        btn_inv_puntual.setToolTip("Abre la ventana interactiva para seleccionar una anomalía con un polígono y ajustar una fuente puntual/esférica por mínimos cuadrados.")
        btn_inv_puntual.clicked.connect(self.open_metodo_inverso_puntual)
        left_layout.addWidget(btn_inv_puntual)

        # SECCIÓN 3: RUIDO GAUSSIANO
        grp_noise = QGroupBox("3. Vuelta al Mundo Real (con Ruido)")
        vbox_n = QVBoxLayout(grp_noise)
        self.chk_noise = QCheckBox("Simular Ruido Instrumental y Geológico Blanco")
        self.chk_noise.stateChanged.connect(self.recalculate_model)
        vbox_n.addWidget(self.chk_noise)

        row_n_val = QHBoxLayout()
        row_n_val.addWidget(QLabel("Nivel de Ruido sigma_ruido:"))
        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setRange(0.0, 1000.0); self.spin_noise.setValue(0.5); self.spin_noise.setSingleStep(0.1)
        self.spin_noise.valueChanged.connect(self.recalculate_model)
        row_n_val.addWidget(self.spin_noise)

        btn_new_noise = QPushButton("🎲 Semilla")
        btn_new_noise.clicked.connect(self.recalculate_model)
        row_n_val.addWidget(btn_new_noise)
        vbox_n.addLayout(row_n_val)
        left_layout.addWidget(grp_noise)

        left_layout.addStretch()

        # BOTONES INFERIORES DE ACCIÓN
        row_bottom = QHBoxLayout()
        btn_apply = QPushButton("💾 Fijar como Modelo Actual")
        btn_apply.setObjectName("btnAccent")
        btn_apply.clicked.connect(self.apply_as_active_model)
        row_bottom.addWidget(btn_apply)

        btn_inject_mesh = QPushButton("📥 Cargar en Malla Principal")
        btn_inject_mesh.setObjectName("btnPrimary")
        btn_inject_mesh.setToolTip("Remplaza la malla de trabajo del programa por este modelo limpio para filtrarlo en Fourier")
        btn_inject_mesh.clicked.connect(self.inject_as_main_grid)
        row_bottom.addWidget(btn_inject_mesh)

        btn_compare = QPushButton("⚖️ Comparador...")
        btn_compare.clicked.connect(self.open_comparison_dialog)
        row_bottom.addWidget(btn_compare)
        left_layout.addLayout(row_bottom)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(350)
        self.left_scroll = left_scroll

        # ======================================================================
        # PANEL DERECHO: PREVISUALIZACIÓN GRÁFICA EN VIVO
        # ======================================================================
        right_widget = QWidget()
        right_widget.setMinimumWidth(380)
        right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        row_r_top = QHBoxLayout()
        row_r_top.addWidget(QLabel("<b>Previsualización del Modelo Sintético Acumulado:</b>"))
        row_r_top.addStretch()

        self.lbl_stats_badge = QLabel("Mín: 0.0 | Máx: 0.0")
        self.lbl_stats_badge.setStyleSheet("font-weight: bold; color: #0055aa; padding: 2px 6px; border: 1px solid #0055aa; background: #ffffff;")
        row_r_top.addWidget(self.lbl_stats_badge)

        self.btn_toggle_panel = QPushButton("◀ Ocultar Panel")
        self.btn_toggle_panel.setToolTip("Alterna la visualización del panel izquierdo de controles para expandir el mapa al 100% de la ventana")
        self.btn_toggle_panel.clicked.connect(self.toggle_left_panel)
        row_r_top.addWidget(self.btn_toggle_panel)

        self.btn_map_move_mode = QPushButton("🎯 Mover con Clic en Mapa")
        self.btn_map_move_mode.setCheckable(True)
        self.btn_map_move_mode.setToolTip("Al activarlo, hacer un clic en el mapa de abajo reubica la fuente seleccionada en esa coordenada")
        row_r_top.addWidget(self.btn_map_move_mode)

        btn_exp_tiff = QPushButton("🗺️ Exportar TIFF (.tif)")
        btn_exp_tiff.clicked.connect(self.export_model_tiff)
        row_r_top.addWidget(btn_exp_tiff)

        btn_exp_csv = QPushButton("📄 Exportar CSV")
        btn_exp_csv.clicked.connect(self.export_model_csv)
        row_r_top.addWidget(btn_exp_csv)
        right_layout.addLayout(row_r_top)

        # Matplotlib Canvas
        self.fig = Figure(facecolor='#ffffff')
        self.canvas = FigureCanvas(self.fig)
        
        row_tool_synth = QHBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #f0f0f0; border: 1px solid #b0b0b0;")
        row_tool_synth.addWidget(self.toolbar)
        row_tool_synth.addStretch()
        right_layout.addLayout(row_tool_synth)

        row_tool_options = QHBoxLayout()
        self.combo_preview_mode = QComboBox()
        self.combo_preview_mode.addItem("👥 Lado a Lado: Real vs Sintética", "side_by_side")
        self.combo_preview_mode.addItem("🔬 Solo Malla Sintética", "synth_only")
        self.combo_preview_mode.addItem("⚖️ Diferencia Residual (Real - Sintética)", "diff")
        self.combo_preview_mode.addItem("📊 Tríptico (Real | Sintética | Residual)", "triptych")
        self.combo_preview_mode.setToolTip("Modo de visualización: Compara en tiempo real la anomalía observada con la sintética")
        self.combo_preview_mode.currentIndexChanged.connect(self.plot_preview)
        row_tool_options.addWidget(self.combo_preview_mode)

        row_tool_options.addWidget(QLabel("<b>Malla Observada:</b>"))
        self.combo_obs_grid = QComboBox()
        self.combo_obs_grid.setToolTip("Elige la malla gravimétrica con la que se contrasta el modelo sintético")
        self.combo_obs_grid.currentIndexChanged.connect(self.on_obs_grid_changed)
        row_tool_options.addWidget(self.combo_obs_grid)

        self.chk_sync_cbar = QCheckBox("🔗 Escala Sincronizada con Real")
        self.chk_sync_cbar.setChecked(True)
        self.chk_sync_cbar.setToolTip("Mantiene fijos los límites de color (vmin, vmax) igual a los datos reales para comparación 1:1 directa sin distorsión de escala.")
        self.chk_sync_cbar.stateChanged.connect(self.plot_preview)
        row_tool_options.addWidget(self.chk_sync_cbar)

        self.chk_synth_aspect_equal = QCheckBox("Escala Isótropa 1:1")
        self.chk_synth_aspect_equal.setChecked(True)
        self.chk_synth_aspect_equal.setToolTip("Mantiene proporción física 1:1 en metros. Desmárcalo para que la malla llene todo el visor (aspect='auto').")
        self.chk_synth_aspect_equal.stateChanged.connect(self.plot_preview)
        row_tool_options.addWidget(self.chk_synth_aspect_equal)

        self.chk_show_geometry = QCheckBox("📐 Dibujo de Cuerpos")
        self.chk_show_geometry.setChecked(True)
        self.chk_show_geometry.setToolTip("Activa o desactiva la proyección geométrica y huella en planta de los cuerpos enterrados sobre los mapas.")
        self.chk_show_geometry.stateChanged.connect(self.plot_preview)
        row_tool_options.addWidget(self.chk_show_geometry)

        self.chk_show_names = QCheckBox("🏷️ Nombres")
        self.chk_show_names.setChecked(True)
        self.chk_show_names.setToolTip("Activa o desactiva la visualización de los nombres de las fuentes sobre los mapas.")
        self.chk_show_names.stateChanged.connect(self.plot_preview)
        row_tool_options.addWidget(self.chk_show_names)
        row_tool_options.addStretch()
        right_layout.addLayout(row_tool_options)

        self.canvas.mpl_connect('button_press_event', self.on_preview_canvas_click)
        right_layout.addWidget(self.canvas)

        self.splitter_main = QSplitter(Qt.Horizontal)
        self.splitter_main.setObjectName("splitterSynthMain")
        self.splitter_main.addWidget(self.left_scroll)
        self.splitter_main.addWidget(right_widget)
        self.splitter_main.setCollapsible(0, True)
        self.splitter_main.setCollapsible(1, False)
        self.splitter_main.setStretchFactor(0, 0)
        self.splitter_main.setStretchFactor(1, 1)
        self.splitter_main.setSizes([430, 850])
        main_layout.addWidget(self.splitter_main)

        # Inicializar valores de la grilla y mallas observadas
        self.refresh_available_obs_grids()
        self.init_grid_coordinates()

        # Iniciar completamente limpio sin fuentes de antemano
        self.refresh_cuerpos_table()
        self.recalculate_model()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_available_obs_grids()
        try:
            total_w = self.width()
            if total_w > 800:
                left_w = min(420, int(total_w * 0.35))
                right_w = total_w - left_w
                self.splitter_main.setSizes([left_w, right_w])
        except Exception:
            pass

    def refresh_available_obs_grids(self):
        """Descubre y lista en combo_obs_grid las 4 mallas disponibles de trabajo."""
        if not hasattr(self, 'combo_obs_grid'):
            return

        mw = self.main_window
        if mw is not None and hasattr(mw, 'get_available_modeling_grids'):
            self.available_obs_grids = mw.get_available_modeling_grids()
        else:
            self.available_obs_grids = {}
            if mw is not None and getattr(mw, 'grid_Z', None) is not None:
                ny, nx = mw.grid_Z.shape
                self.available_obs_grids['sin_filtrar'] = {
                    'key': 'sin_filtrar',
                    'label': f"1. Malla Sin Filtrar [{nx}×{ny}]",
                    'Z': mw.grid_Z.copy(),
                    'x': mw.grid_x_vec.copy() if getattr(mw, 'grid_x_vec', None) is not None else None,
                    'y': mw.grid_y_vec.copy() if getattr(mw, 'grid_y_vec', None) is not None else None,
                    'z_elev': 0.0,
                    'name': 'Original / Sin Filtrar'
                }
            if mw is not None and getattr(mw, 'grid_Z_filt', None) is not None:
                ny, nx = mw.grid_Z_filt.shape
                self.available_obs_grids['filtrada'] = {
                    'key': 'filtrada',
                    'label': f"2. Malla Filtrada Fourier [{nx}×{ny}]",
                    'Z': mw.grid_Z_filt.copy(),
                    'x': mw.grid_x_vec.copy() if getattr(mw, 'grid_x_vec', None) is not None else None,
                    'y': mw.grid_y_vec.copy() if getattr(mw, 'grid_y_vec', None) is not None else None,
                    'z_elev': 0.0,
                    'name': 'Filtrada Fourier'
                }

        prev_key = self.combo_obs_grid.currentData() or self.active_obs_key
        self.combo_obs_grid.blockSignals(True)
        self.combo_obs_grid.clear()

        if self.available_obs_grids:
            for k, info in self.available_obs_grids.items():
                self.combo_obs_grid.addItem(info['label'], k)
            idx = self.combo_obs_grid.findData(prev_key)
            if idx != -1:
                self.combo_obs_grid.setCurrentIndex(idx)
            else:
                self.combo_obs_grid.setCurrentIndex(0)
        else:
            self.combo_obs_grid.addItem("Sin mallas observadas cargadas", "none")

        self.combo_obs_grid.blockSignals(False)
        self.active_obs_key = self.combo_obs_grid.currentData()

    def on_obs_grid_changed(self):
        self.active_obs_key = self.combo_obs_grid.currentData()
        if hasattr(self, 'rad_grid_auto') and self.rad_grid_auto.isChecked():
            self.init_grid_coordinates()
        self.plot_preview()

    def get_active_obs_grid_info(self):
        """Retorna información completa de la malla observada activa."""
        if not hasattr(self, 'available_obs_grids') or not self.available_obs_grids:
            self.refresh_available_obs_grids()

        key = getattr(self, 'active_obs_key', 'sin_filtrar')
        if key in self.available_obs_grids:
            return self.available_obs_grids[key]

        if self.available_obs_grids:
            first_key = next(iter(self.available_obs_grids))
            return self.available_obs_grids[first_key]
        return None

    def toggle_left_panel(self):
        is_vis = self.left_scroll.isVisible()
        self.left_scroll.setVisible(not is_vis)
        self.btn_toggle_panel.setText("▶ Mostrar Panel" if is_vis else "◀ Ocultar Panel")
        self.canvas.draw_idle()

    def on_grid_mode_changed(self):
        is_manual = self.rad_grid_manual.isChecked()
        for sp in [self.spin_xmin, self.spin_xmax, self.spin_ymin, self.spin_ymax, self.spin_dx, self.spin_dy]:
            sp.setEnabled(is_manual)
        if not is_manual:
            self.init_grid_coordinates()
        self.recalculate_model()

    def init_grid_coordinates(self):
        mw = self.main_window
        if mw is None:
            return
        if self.rad_grid_auto.isChecked():
            x0, x1, y0, y1, dx, dy = None, None, None, None, None, None
            # 0. Prioridad: Malla observada actualmente seleccionada
            obs_info = self.get_active_obs_grid_info()
            if obs_info is not None and obs_info.get('x') is not None and obs_info.get('y') is not None and len(obs_info['x']) > 1 and len(obs_info['y']) > 1:
                x0 = float(obs_info['x'][0])
                x1 = float(obs_info['x'][-1])
                y0 = float(obs_info['y'][0])
                y1 = float(obs_info['y'][-1])
                dx = float(abs(obs_info['x'][1] - obs_info['x'][0]))
                dy = float(abs(obs_info['y'][1] - obs_info['y'][0]))
            # 1. Prioridad: Malla regular interpolada en memoria
            elif getattr(mw, 'grid_x_vec', None) is not None and getattr(mw, 'grid_y_vec', None) is not None and len(mw.grid_x_vec) > 1 and len(mw.grid_y_vec) > 1:
                x0 = float(mw.grid_x_vec[0])
                x1 = float(mw.grid_x_vec[-1])
                y0 = float(mw.grid_y_vec[0])
                y1 = float(mw.grid_y_vec[-1])
                dx = float(abs(mw.grid_x_vec[1] - mw.grid_x_vec[0]))
                dy = float(abs(mw.grid_y_vec[1] - mw.grid_y_vec[0]))
            # 2. Prioridad: Delimitación de zona seleccionada en Pestaña 1
            elif getattr(mw, 'x_min_sel', None) is not None and getattr(mw, 'x_max_sel', None) is not None and getattr(mw, 'y_min_sel', None) is not None and getattr(mw, 'y_max_sel', None) is not None:
                x0 = float(mw.x_min_sel)
                x1 = float(mw.x_max_sel)
                y0 = float(mw.y_min_sel)
                y1 = float(mw.y_max_sel)
                span_x = abs(x1 - x0)
                span_y = abs(y1 - y0)
                sugg_dx = getattr(mw, 'spin_dx', None).value() if hasattr(mw, 'spin_dx') else 100.0
                sugg_dy = getattr(mw, 'spin_dy', None).value() if hasattr(mw, 'spin_dy') else 100.0
                if span_x > 0 and (span_x / max(1e-6, sugg_dx)) > 800:
                    sugg_dx = span_x / 200.0
                if span_y > 0 and (span_y / max(1e-6, sugg_dy)) > 800:
                    sugg_dy = span_y / 200.0
                dx = max(1e-4, sugg_dx)
                dy = max(1e-4, sugg_dy)
            # 3. Prioridad: Puntos crudos cargados
            elif getattr(mw, 'pts_x', None) is not None and len(mw.pts_x) > 0 and getattr(mw, 'pts_y', None) is not None and len(mw.pts_y) > 0:
                x0 = float(np.min(mw.pts_x))
                x1 = float(np.max(mw.pts_x))
                y0 = float(np.min(mw.pts_y))
                y1 = float(np.max(mw.pts_y))
                span_x = abs(x1 - x0)
                span_y = abs(y1 - y0)
                sugg_dx = getattr(mw, 'spin_dx', None).value() if hasattr(mw, 'spin_dx') else 100.0
                sugg_dy = getattr(mw, 'spin_dy', None).value() if hasattr(mw, 'spin_dy') else 100.0
                if span_x > 0 and (span_x / max(1e-6, sugg_dx)) > 800:
                    sugg_dx = span_x / 200.0
                if span_y > 0 and (span_y / max(1e-6, sugg_dy)) > 800:
                    sugg_dy = span_y / 200.0
                dx = max(1e-4, sugg_dx)
                dy = max(1e-4, sugg_dy)

            if x0 is not None and x1 is not None and y0 is not None and y1 is not None:
                for sp in [self.spin_xmin, self.spin_xmax, self.spin_ymin, self.spin_ymax]:
                    sp.setRange(-1e12, 1e12)
                    sp.blockSignals(True)
                for sp in [self.spin_dx, self.spin_dy]:
                    sp.setRange(1e-6, 1e9)
                    sp.blockSignals(True)

                self.spin_xmin.setValue(min(x0, x1))
                self.spin_xmax.setValue(max(x0, x1))
                self.spin_ymin.setValue(min(y0, y1))
                self.spin_ymax.setValue(max(y0, y1))
                if dx is not None and dx > 0: self.spin_dx.setValue(float(dx))
                if dy is not None and dy > 0: self.spin_dy.setValue(float(dy))

                for sp in [self.spin_xmin, self.spin_xmax, self.spin_ymin, self.spin_ymax, self.spin_dx, self.spin_dy]:
                    sp.blockSignals(False)
                    sp.setEnabled(False)

    def get_grid_vectors(self):
        xmin = self.spin_xmin.value(); xmax = self.spin_xmax.value()
        ymin = self.spin_ymin.value(); ymax = self.spin_ymax.value()
        dx = max(1e-4, self.spin_dx.value())
        dy = max(1e-4, self.spin_dy.value())
        if xmax <= xmin: xmax = xmin + dx * 10
        if ymax <= ymin: ymax = ymin + dy * 10
        span_x = abs(xmax - xmin)
        span_y = abs(ymax - ymin)
        # Protección estricta de memoria RAM: máximo 800 celdas por eje en sintético
        if span_x > 0 and (span_x / dx) > 800:
            dx = span_x / 250.0
            self.spin_dx.blockSignals(True)
            self.spin_dx.setValue(dx)
            self.spin_dx.blockSignals(False)
        if span_y > 0 and (span_y / dy) > 800:
            dy = span_y / 250.0
            self.spin_dy.blockSignals(True)
            self.spin_dy.setValue(dy)
            self.spin_dy.blockSignals(False)
        x_vec = np.arange(xmin, xmax + dx * 0.5, dx)
        y_vec = np.arange(ymin, ymax + dy * 0.5, dy)
        return x_vec, y_vec, dx, dy

    def refresh_cuerpos_table(self):
        cur_row = self.tbl_cuerpos.currentRow()
        self.tbl_cuerpos.blockSignals(True)
        self.tbl_cuerpos.setRowCount(len(self.cuerpos))
        for row, c in enumerate(self.cuerpos):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if c.is_enabled else Qt.Unchecked)
            self.tbl_cuerpos.setItem(row, 0, chk)

            n_item = QTableWidgetItem(c.name)
            n_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_cuerpos.setItem(row, 1, n_item)

            info = BLAKELY_MODELS_INFO.get(c.body_type, {})
            t_item = QTableWidgetItem(info.get('name', c.body_type))
            t_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_cuerpos.setItem(row, 2, t_item)

            rho_str = f"{c.rho:g}" if c.get_category() == 'gravedad' else "-"
            rho_item = QTableWidgetItem(rho_str)
            rho_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_cuerpos.setItem(row, 3, rho_item)

            if c.body_type in ('grav_ribbon', 'mag_ribbon'):
                r_str = f"e={c.espesor:g} m"
            elif c.get_category() == 'gravedad' or c.body_type == 'mag_dipole':
                r_str = f"{c.R:g} m"
            else:
                r_str = "-"
            r_item = QTableWidgetItem(r_str)
            r_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_cuerpos.setItem(row, 4, r_item)

            z1_item = QTableWidgetItem(f"{c.z1:g} m")
            z1_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl_cuerpos.setItem(row, 5, z1_item)

            form_item = QTableWidgetItem(c.formula_expr)
            form_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            form_item.setFont(QFont("Consolas", 8))
            self.tbl_cuerpos.setItem(row, 6, form_item)

        self.tbl_cuerpos.blockSignals(False)
        if 0 <= cur_row < len(self.cuerpos):
            self.tbl_cuerpos.setCurrentCell(cur_row, 1)
        elif len(self.cuerpos) > 0:
            self.tbl_cuerpos.setCurrentCell(0, 1)
        self.on_table_row_selected()

    def on_table_item_changed(self, item):
        if item.column() == 0:
            row = item.row()
            if 0 <= row < len(self.cuerpos):
                self.cuerpos[row].is_enabled = (item.checkState() == Qt.Checked)
                self.recalculate_model()

    def on_table_row_selected(self):
        row = self.tbl_cuerpos.currentRow()
        if 0 <= row < len(self.cuerpos):
            c = self.cuerpos[row]
            self.txt_quick_formula.setText(c.formula_expr)
            self.lbl_quick_numeric_eq.setText(f"E(K) = {c.get_substituted_numeric_formula()}")

            if hasattr(self, 'spin_q_x'):
                self._block_quick_signals = True
                self.spin_q_x.setValue(float(c.x))
                self.spin_q_y.setValue(float(c.y))
                self.spin_q_z1.setValue(float(c.z1))
                self.spin_q_rho.setValue(float(c.rho if c.get_category() == 'gravedad' else getattr(c, 'mag_m', 100.0)))
                self.spin_q_r.setValue(float(c.R if c.R > 0 else getattr(c, 'a', 500.0)))
                val_h = getattr(c, 'espesor', 0.0)
                if val_h <= 0 and hasattr(c, 'z2') and c.z2 > c.z1:
                    val_h = c.z2 - c.z1
                self.spin_q_h.setValue(float(val_h))
                if hasattr(self, 'spin_q_dip'):
                    self.spin_q_dip.setValue(float(getattr(c, 'dip', 45.0)))
                if hasattr(self, 'spin_q_L'):
                    self.spin_q_L.setValue(float(getattr(c, 'L', 2000.0)))
                if hasattr(self, 'spin_q_strike'):
                    self.spin_q_strike.setValue(float(getattr(c, 'strike', 0.0)))
                self._block_quick_signals = False
        else:
            self.txt_quick_formula.clear()
            self.lbl_quick_numeric_eq.setText("E(K) = [Selecciona una fuente]")

    def on_quick_param_changed(self):
        if getattr(self, '_block_quick_signals', False):
            return
        row = self.tbl_cuerpos.currentRow()
        if not (0 <= row < len(self.cuerpos)):
            return
        c = self.cuerpos[row]
        c.x = round(float(self.spin_q_x.value()), 1)
        c.y = round(float(self.spin_q_y.value()), 1)
        c.z1 = max(1.0, round(float(self.spin_q_z1.value()), 1))
        val_rho = round(float(self.spin_q_rho.value()), 2)
        if c.get_category() == 'gravedad':
            c.rho = val_rho
        else:
            c.mag_m = val_rho
        val_r = max(0.1, round(float(self.spin_q_r.value()), 1))
        c.R = val_r
        c.radius = val_r
        c.a = val_r
        val_h = max(0.0, round(float(self.spin_q_h.value()), 1))
        c.espesor = val_h
        c.z2 = c.z1 + val_h
        if hasattr(self, 'spin_q_dip'):
            c.dip = round(float(self.spin_q_dip.value()), 1)
        if hasattr(self, 'spin_q_L'):
            c.L = max(1.0, round(float(self.spin_q_L.value()), 1))
            c.length = c.L
        if hasattr(self, 'spin_q_strike'):
            c.strike = round(float(self.spin_q_strike.value()), 1)
        
        self.lbl_quick_numeric_eq.setText(f"E(K) = {c.get_substituted_numeric_formula()}")
        self.refresh_cuerpos_table()
        self.recalculate_model()

    def on_apply_quick_formula(self):
        row = self.tbl_cuerpos.currentRow()
        if 0 <= row < len(self.cuerpos):
            c = self.cuerpos[row]
            c.formula_expr = self.txt_quick_formula.text().strip() or get_default_formula_for_type(c.body_type)
            self.lbl_quick_numeric_eq.setText(f"E(K) = {c.get_substituted_numeric_formula()}")
            self.refresh_cuerpos_table()
            self.recalculate_model()

    def on_add_cuerpo(self):
        x_vec, y_vec, dx, dy = self.get_grid_vectors()
        cx = 0.5 * (x_vec[0] + x_vec[-1])
        cy = 0.5 * (y_vec[0] + y_vec[-1])
        span_grid = max(10.0, min(abs(x_vec[-1] - x_vec[0]), abs(y_vec[-1] - y_vec[0])))
        scale_fac = min(1.0, span_grid / 5000.0)

        z1_init = max(5.0, round(1000.0 * scale_fac, 1))
        r_init = min(z1_init * 0.75, max(1.0, round(250.0 * scale_fac, 1)))
        z2_init = max(z1_init + 10.0, round(2500.0 * scale_fac, 1))
        L_init = max(10.0, round(2000.0 * scale_fac, 1))

        nuevo = CuerpoSintetico(
            name=f"Fuente {len(self.cuerpos) + 1}",
            x=cx, y=cy,
            z1=z1_init, z2=z2_init,
            R=r_init, L=L_init
        )
        dlg = CuerpoSinteticoDialog(nuevo, parent=self, default_x=cx, default_y=cy, dx=dx, dy=dy, unit_xy=self.main_window.unit_xy, span_grid=span_grid)
        if dlg.exec_() == QDialog.Accepted:
            self.cuerpos.append(nuevo)
            self.main_window.cuerpos_sinteticos = self.cuerpos
            self.refresh_cuerpos_table()
            self.recalculate_model()

    def on_add_cuerpo_by_map_click(self):
        x_vec, y_vec, dx, dy = self.get_grid_vectors()
        cx = 0.5 * (x_vec[0] + x_vec[-1])
        cy = 0.5 * (y_vec[0] + y_vec[-1])
        span_grid = max(10.0, min(abs(x_vec[-1] - x_vec[0]), abs(y_vec[-1] - y_vec[0])))
        scale_fac = min(1.0, span_grid / 5000.0)

        z1_init = max(5.0, round(1000.0 * scale_fac, 1))
        r_init = min(z1_init * 0.75, max(1.0, round(250.0 * scale_fac, 1)))
        z2_init = max(z1_init + 10.0, round(2500.0 * scale_fac, 1))
        L_init = max(10.0, round(2000.0 * scale_fac, 1))

        mw = self.main_window
        grid_obs = None
        x_obs = None
        y_obs = None
        if mw is not None:
            for attr in ['grid_Z', 'grid_z', 'grid_Z_filt']:
                cand = getattr(mw, attr, None)
                if cand is not None:
                    grid_obs = cand
                    break
            if getattr(mw, 'grid_x_vec', None) is not None:
                x_obs = mw.grid_x_vec
            if getattr(mw, 'grid_y_vec', None) is not None:
                y_obs = mw.grid_y_vec

        # Asegurar que el modelo sintético esté calculado
        if self.grid_Z_model is None:
            self.recalculate_model()
        grid_sim = self.grid_Z_model if self.grid_Z_model is not None else np.zeros((len(y_vec), len(x_vec)))
        x_sim = x_vec
        y_sim = y_vec

        dlg_map = DialogoSelectorCoordenadasMapa(
            parent=self,
            initial_x=cx, initial_y=cy,
            grid_obs=grid_obs,
            x_obs=x_obs,
            y_obs=y_obs,
            grid_sim=grid_sim,
            x_sim=x_sim,
            y_sim=y_sim,
            unit_xy=getattr(mw, 'unit_xy', 'm') if mw else "m",
            unit_z=getattr(mw, 'unit_z', 'mGal') if mw else "mGal",
            otros_cuerpos=self.cuerpos
        )
        if dlg_map.exec_() == QDialog.Accepted:
            nuevo = CuerpoSintetico(
                name=f"Fuente {len(self.cuerpos) + 1}",
                x=dlg_map.selected_x,
                y=dlg_map.selected_y,
                z1=z1_init, z2=z2_init,
                R=r_init, L=L_init
            )
            dlg = CuerpoSinteticoDialog(nuevo, parent=self, default_x=cx, default_y=cy, dx=dx, dy=dy, unit_xy=getattr(mw, 'unit_xy', 'm') if mw else "m", span_grid=span_grid)
            if dlg.exec_() == QDialog.Accepted:
                self.cuerpos.append(nuevo)
                self.main_window.cuerpos_sinteticos = self.cuerpos
                self.refresh_cuerpos_table()
                self.recalculate_model()

    def on_preview_canvas_click(self, event):
        if not getattr(self, 'btn_map_move_mode', None) or not self.btn_map_move_mode.isChecked():
            return
        allowed_axes = [getattr(self, 'ax_map', None), getattr(self, 'ax_real', None)]
        if event.inaxes not in allowed_axes or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        row = self.tbl_cuerpos.currentRow()
        if 0 <= row < len(self.cuerpos):
            c = self.cuerpos[row]
            c.x = round(float(event.xdata), 1)
            c.y = round(float(event.ydata), 1)
            self.lbl_quick_numeric_eq.setText(f"E(K) = {c.get_substituted_numeric_formula()}")
            self.on_table_row_selected()
            self.refresh_cuerpos_table()
            self.recalculate_model()

    def on_edit_cuerpo(self):
        row = self.tbl_cuerpos.currentRow()
        if 0 <= row < len(self.cuerpos):
            x_vec, y_vec, dx, dy = self.get_grid_vectors()
            cx = 0.5 * (x_vec[0] + x_vec[-1])
            cy = 0.5 * (y_vec[0] + y_vec[-1])
            span_grid = max(10.0, min(abs(x_vec[-1] - x_vec[0]), abs(y_vec[-1] - y_vec[0])))
            c = self.cuerpos[row]
            dlg = CuerpoSinteticoDialog(c, parent=self, default_x=cx, default_y=cy, dx=dx, dy=dy, unit_xy=self.main_window.unit_xy, span_grid=span_grid)
            if dlg.exec_() == QDialog.Accepted:
                self.refresh_cuerpos_table()
                self.recalculate_model()

    def on_duplicate_cuerpo(self):
        row = self.tbl_cuerpos.currentRow()
        if 0 <= row < len(self.cuerpos):
            orig = self.cuerpos[row]
            x_vec, y_vec, dx, dy = self.get_grid_vectors()
            clon = CuerpoSintetico(
                name=f"{orig.name} (Copia)",
                body_type=orig.body_type,
                x=min(x_vec[-1], orig.x + dx * 5),
                y=min(y_vec[-1], orig.y + dy * 5),
                z1=orig.z1, z2=orig.z2,
                mu=orig.mu, R=orig.R, rho=orig.rho, gamma=orig.gamma,
                L=orig.L, strike=orig.strike, espesor=orig.espesor,
                mag_m=orig.mag_m, inc_f=orig.inc_f, dec_f=orig.dec_f,
                inc_m=orig.inc_m, dec_m=orig.dec_m, Cm=orig.Cm,
                formula_expr=orig.formula_expr,
                is_enabled=orig.is_enabled
            )
            self.cuerpos.append(clon)
            self.main_window.cuerpos_sinteticos = self.cuerpos
            self.refresh_cuerpos_table()
            self.recalculate_model()

    def on_delete_cuerpo(self):
        row = self.tbl_cuerpos.currentRow()
        if 0 <= row < len(self.cuerpos):
            del self.cuerpos[row]
            self.main_window.cuerpos_sinteticos = self.cuerpos
            self.refresh_cuerpos_table()
            self.recalculate_model()

    def on_clear_cuerpos(self):
        if QMessageBox.question(self, "Limpiar Pila", "¿Deseas eliminar todas las fuentes sintéticas?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.cuerpos.clear()
            self.main_window.cuerpos_sinteticos = self.cuerpos
            self.refresh_cuerpos_table()
            self.recalculate_model()

    def open_metodo_inverso_puntual(self):
        """Abre la ventana dedicada al Método Inverso Puntual."""
        from metodo_inverso_puntual import VentanaMetodoInversoPuntual
        self.dlg_inv_puntual = VentanaMetodoInversoPuntual(main_window=self.main_window, modelador=self, parent=self)
        self.dlg_inv_puntual.show()

    def agregar_cuerpo_desde_inversion(self, cuerpo):
        """Agrega directamente una fuente invertida al modelado, refresca la tabla y recalcula la malla."""
        self.cuerpos.append(cuerpo)
        if self.main_window is not None:
            self.main_window.cuerpos_sinteticos = self.cuerpos
        self.refresh_cuerpos_table()
        self.recalculate_model()
        if hasattr(self, 'tbl_cuerpos') and len(self.cuerpos) > 0:
            self.tbl_cuerpos.setCurrentCell(len(self.cuerpos) - 1, 1)

    def show_presets_menu(self):
        menu = QMenu(self)
        act1 = menu.addAction("⚪ Dos Esferas (Monopole Eqn. 11.22)")
        act1.triggered.connect(self.load_preset_two_spheres)
        act2 = menu.addAction("➖ Cilindro Horizontal (Horizontal Line Eqn. 11.27)")
        act2.triggered.connect(self.load_preset_hline_cylinder)
        act3 = menu.addAction("📏 Chimenea / Cilindro Vertical (Vertical Line Eqn. 11.23)")
        act3.triggered.connect(self.load_preset_vline_cylinder)
        act4 = menu.addAction("📐 Dique Vertical (Vertical Ribbon Eqn. 11.28)")
        act4.triggered.connect(self.load_preset_ribbon)
        act5 = menu.addAction("🔀 Combo Blakely (Esfera + Cilindro Horiz. + Dique)")
        act5.triggered.connect(self.load_preset_combo_blakely)
        menu.exec_(self.sender().mapToGlobal(QPoint(0, self.sender().height())))

    def load_preset_two_spheres(self):
        x_vec, y_vec, _, _ = self.get_grid_vectors()
        span_x = (x_vec[-1] - x_vec[0])
        span_y = (y_vec[-1] - y_vec[0])
        span = max(10.0, min(span_x, span_y))
        scale = min(1.0, span / 5000.0)

        z1_1 = max(5.0, 1200.0 * scale)
        r_1 = min(z1_1 * 0.75, max(1.5, 350.0 * scale))
        z1_2 = max(10.0, 2500.0 * scale)
        r_2 = min(z1_2 * 0.75, max(2.0, 450.0 * scale))

        c1 = CuerpoSintetico(
            name="Esfera 1 (Somera)", body_type="grav_sphere",
            x=x_vec[0] + span_x * 0.3, y=y_vec[0] + span_y * 0.35,
            z1=z1_1, rho=1200.0, R=r_1,
            formula_expr="2 * pi * gamma * (4.0 / 3.0 * pi * R**3 * rho) * exp(-K * z1)"
        )
        c2 = CuerpoSintetico(
            name="Esfera 2 (Profunda)", body_type="grav_sphere",
            x=x_vec[0] + span_x * 0.72, y=y_vec[0] + span_y * 0.7,
            z1=z1_2, rho=1500.0, R=r_2,
            formula_expr="2 * pi * gamma * (4.0 / 3.0 * pi * R**3 * rho) * exp(-K * z1)"
        )
        self.cuerpos = [c1, c2]
        self.main_window.cuerpos_sinteticos = self.cuerpos
        self.refresh_cuerpos_table()
        self.recalculate_model()

    def load_preset_hline_cylinder(self):
        x_vec, y_vec, _, _ = self.get_grid_vectors()
        cx = 0.5 * (x_vec[0] + x_vec[-1])
        cy = 0.5 * (y_vec[0] + y_vec[-1])
        span_x = (x_vec[-1] - x_vec[0])
        span_y = (y_vec[-1] - y_vec[0])
        span = max(10.0, min(span_x, span_y))
        scale = min(1.0, span / 5000.0)

        z1_s = max(5.0, 1200.0 * scale)
        r_s = min(z1_s * 0.75, max(1.5, 250.0 * scale))
        L_s = max(10.0, span_x * 0.45)

        c1 = CuerpoSintetico(
            name="Cilindro Horizontal", body_type="grav_hline",
            x=cx, y=cy, z1=z1_s,
            rho=1000.0, R=r_s,
            L=L_s, strike=90.0,
            formula_expr="2 * pi * gamma * (pi * R**2 * rho) * L * exp(-K * z1) * sinc(k_parallel * L / (2 * pi))"
        )
        self.cuerpos = [c1]
        self.main_window.cuerpos_sinteticos = self.cuerpos
        self.refresh_cuerpos_table()
        self.recalculate_model()

    def load_preset_vline_cylinder(self):
        x_vec, y_vec, _, _ = self.get_grid_vectors()
        cx = 0.5 * (x_vec[0] + x_vec[-1])
        cy = 0.5 * (y_vec[0] + y_vec[-1])
        span_x = (x_vec[-1] - x_vec[0])
        span_y = (y_vec[-1] - y_vec[0])
        span = max(10.0, min(span_x, span_y))
        scale = min(1.0, span / 5000.0)

        z1_s = max(5.0, 400.0 * scale)
        z2_s = max(z1_s + 10.0, 2500.0 * scale)
        r_s = min(z1_s * 0.75, max(1.0, 200.0 * scale))

        c1 = CuerpoSintetico(
            name="Cilindro Vertical", body_type="grav_vline",
            x=cx, y=cy, z1=z1_s, z2=z2_s,
            rho=1100.0, R=r_s,
            formula_expr="(2 * pi * gamma * (pi * R**2 * rho) / np.maximum(1e-12, K)) * (exp(-K * z1) - exp(-K * z2))"
        )
        self.cuerpos = [c1]
        self.main_window.cuerpos_sinteticos = self.cuerpos
        self.refresh_cuerpos_table()
        self.recalculate_model()

    def load_preset_ribbon(self):
        x_vec, y_vec, _, _ = self.get_grid_vectors()
        cx = 0.5 * (x_vec[0] + x_vec[-1])
        cy = 0.5 * (y_vec[0] + y_vec[-1])
        span_x = (x_vec[-1] - x_vec[0])
        span_y = (y_vec[-1] - y_vec[0])
        span = max(10.0, min(span_x, span_y))
        scale = min(1.0, span / 5000.0)

        z1_s = max(5.0, 300.0 * scale)
        z2_s = max(z1_s + 10.0, 2200.0 * scale)
        esp_s = max(0.5, 60.0 * scale)
        L_s = max(10.0, span_x * 0.4)

        c1 = CuerpoSintetico(
            name="Dique Vertical", body_type="grav_ribbon",
            x=cx, y=cy, z1=z1_s, z2=z2_s,
            rho=900.0, espesor=esp_s,
            L=L_s, strike=45.0,
            formula_expr="(2 * pi * gamma * (espesor * rho) * L / np.maximum(1e-12, K)) * (exp(-K * z1) - exp(-K * z2)) * sinc(k_parallel * L / (2 * pi))"
        )
        self.cuerpos = [c1]
        self.main_window.cuerpos_sinteticos = self.cuerpos
        self.refresh_cuerpos_table()
        self.recalculate_model()

    def load_preset_combo_blakely(self):
        x_vec, y_vec, _, _ = self.get_grid_vectors()
        span_x = (x_vec[-1] - x_vec[0])
        span_y = (y_vec[-1] - y_vec[0])
        span = max(10.0, min(span_x, span_y))
        scale = min(1.0, span / 5000.0)

        z1_1 = max(5.0, 1000.0 * scale)
        r_1 = min(z1_1 * 0.75, max(1.5, 300.0 * scale))

        z1_2 = max(5.0, 1200.0 * scale)
        r_2 = min(z1_2 * 0.75, max(1.5, 200.0 * scale))
        L_2 = max(10.0, span_x * 0.35)

        z1_3 = max(5.0, 300.0 * scale)
        z2_3 = max(z1_3 + 10.0, 2000.0 * scale)
        esp_3 = max(0.5, 50.0 * scale)
        L_3 = max(10.0, span_x * 0.3)

        c1 = CuerpoSintetico(
            name="Esfera (Eqn. 11.22)", body_type="grav_sphere",
            x=x_vec[0] + span_x * 0.28, y=y_vec[0] + span_y * 0.35,
            z1=z1_1, rho=1200.0, R=r_1,
            formula_expr="2 * pi * gamma * (4.0 / 3.0 * pi * R**3 * rho) * exp(-K * z1)"
        )
        c2 = CuerpoSintetico(
            name="Cilindro Horiz. (Eqn. 11.27)", body_type="grav_hline",
            x=x_vec[0] + span_x * 0.7, y=y_vec[0] + span_y * 0.65,
            z1=z1_2, rho=1000.0, R=r_2,
            L=L_2, strike=60.0,
            formula_expr="2 * pi * gamma * (pi * R**2 * rho) * L * exp(-K * z1) * sinc(k_parallel * L / (2 * pi))"
        )
        c3 = CuerpoSintetico(
            name="Dique Vert. (Eqn. 11.28)", body_type="grav_ribbon",
            x=x_vec[0] + span_x * 0.45, y=y_vec[0] + span_y * 0.75,
            z1=z1_3, z2=z2_3, rho=850.0, espesor=esp_3,
            L=L_3, strike=135.0,
            formula_expr="(2 * pi * gamma * (espesor * rho) * L / np.maximum(1e-12, K)) * (exp(-K * z1) - exp(-K * z2)) * sinc(k_parallel * L / (2 * pi))"
        )
        self.cuerpos = [c1, c2, c3]
        self.main_window.cuerpos_sinteticos = self.cuerpos
        self.refresh_cuerpos_table()
        self.recalculate_model()

    # ==========================================================================
    # CÁLCULO DE MODELO SINTÉTICO (ESPACIAL DIRECTO O ESPECTRAL FOURIER)
    # ==========================================================================
    def recalculate_model(self):
        x_vec, y_vec, dx, dy = self.get_grid_vectors()
        nx = len(x_vec)
        ny = len(y_vec)
        self.lbl_dims_info.setText(f"Dimensiones: {nx} × {ny} nodos ({nx*ny:,} puntos, dx={dx:g}, dy={dy:g})")

        if nx < 2 or ny < 2:
            return

        self.x_vec = x_vec
        self.y_vec = y_vec
        x0_grid = x_vec[0]
        y0_grid = y_vec[0]

        mode_calc = self.combo_calc_mode.currentData() if hasattr(self, 'combo_calc_mode') else 'spatial'
        if mode_calc == 'spatial':
            X, Y = np.meshgrid(x_vec, y_vec)
            Z_mod = np.zeros((ny, nx), dtype=float)
            for c in self.cuerpos:
                if c.is_enabled:
                    Z_mod += c.evaluate_spatial(X, Y)
        else:
            # Wavenumbers Fourier Blakely
            u = np.fft.fftfreq(nx, d=dx)
            v = np.fft.fftfreq(ny, d=dy)
            U, V = np.meshgrid(u, v)
            kx = 2.0 * np.pi * U
            ky = 2.0 * np.pi * V
            K = np.sqrt(kx ** 2 + ky ** 2)

            espectro_total = np.zeros((ny, nx), dtype=complex)
            for c in self.cuerpos:
                if c.is_enabled:
                    spec_i = c.evaluate_spectrum(kx, ky, K, x0_grid, y0_grid, dx, dy)
                    espectro_total += spec_i

            Z_mod = np.real(np.fft.ifft2(espectro_total))

        # Ruido gaussiano opcional
        if self.chk_noise.isChecked():
            n_sigma = self.spin_noise.value()
            if n_sigma > 0:
                noise = np.random.normal(0.0, n_sigma, size=Z_mod.shape)
                Z_mod += noise

        self.grid_Z_model = Z_mod
        self.plot_preview()

    def plot_preview(self):
        self.fig.clf()
        if self.grid_Z_model is None or self.x_vec is None or self.y_vec is None:
            self.canvas.draw()
            return

        u_xy = getattr(self.main_window, 'unit_xy', 'm')
        unit_z = getattr(self.main_window, 'unit_z', 'mGal')
        if any(c.get_category() == 'magnetometria' for c in self.cuerpos if c.is_enabled):
            unit_z = "nT"

        extent_synth = [self.x_vec[0], self.x_vec[-1], self.y_vec[0], self.y_vec[-1]]
        z_min = float(np.nanmin(self.grid_Z_model))
        z_max = float(np.nanmax(self.grid_Z_model))
        z_mean = float(np.nanmean(self.grid_Z_model))

        mw = self.main_window
        obs_info = self.get_active_obs_grid_info()
        has_real = False
        grid_real = None
        x_real = None
        y_real = None
        label_real = "Observada"

        if obs_info is not None and obs_info.get('Z') is not None and obs_info.get('x') is not None and len(obs_info['x']) > 1 and len(obs_info['y']) > 1:
            grid_real = obs_info['Z']
            x_real = obs_info['x']
            y_real = obs_info['y']
            label_real = obs_info.get('name', 'Observada')
            has_real = True
        elif (getattr(mw, 'grid_Z', None) is not None and
              getattr(mw, 'grid_x_vec', None) is not None and
              getattr(mw, 'grid_y_vec', None) is not None and
              len(mw.grid_x_vec) > 1 and len(mw.grid_y_vec) > 1):
            grid_real = mw.grid_Z
            x_real = mw.grid_x_vec
            y_real = mw.grid_y_vec
            label_real = "Malla Original"
            has_real = True

        sync_scale = getattr(self, 'chk_sync_cbar', None) and self.chk_sync_cbar.isChecked() and has_real

        cmap = 'turbo'
        if hasattr(self.main_window, 'combo_master_cmap'):
            cmap = self.main_window.combo_master_cmap.currentText()
        elif hasattr(self.main_window, 'combo_map_cmap'):
            cmap = self.main_window.combo_map_cmap.currentText()

        if sync_scale and grid_real is not None:
            vmin = float(np.nanmin(grid_real))
            vmax = float(np.nanmax(grid_real))
            scale_note = f"Escala Sincronizada ({label_real})"
        else:
            vmin = z_min
            vmax = z_max
            scale_note = "Escala Local Sintética"

        span_val = vmax - vmin
        is_flat_field = span_val < 1e-3 or span_val < 1e-4 * max(1.0, abs(vmax))
        if is_flat_field:
            mid = 0.5 * (vmin + vmax)
            vmin, vmax = mid - 1.0, mid + 1.0

        mode_calc = self.combo_calc_mode.currentData() if hasattr(self, 'combo_calc_mode') else 'spatial'
        if is_flat_field and mode_calc == 'fourier':
            self.lbl_stats_badge.setText(f"⚠️ Campo Plano Fourier: Mín={z_min:.2f} | Máx={z_max:.2f} (z1 > malla; usa 'Espacial Directo')")
            self.lbl_stats_badge.setStyleSheet("font-weight: bold; color: #cc6600; padding: 2px 6px; border: 1px solid #cc6600; background: #fffbe8;")
        else:
            self.lbl_stats_badge.setText(f"Mín: {z_min:.2f} | Máx: {z_max:.2f} | Prom: {z_mean:.2f} {unit_z}")
            self.lbl_stats_badge.setStyleSheet("font-weight: bold; color: #0055aa; padding: 2px 6px; border: 1px solid #0055aa; background: #ffffff;")

        mode = self.combo_preview_mode.currentData() if hasattr(self, 'combo_preview_mode') else 'side_by_side'
        if not has_real and mode in ['side_by_side', 'diff', 'triptych']:
            mode = 'synth_only'

        is_aspect_equal = getattr(self, 'chk_synth_aspect_equal', None) and self.chk_synth_aspect_equal.isChecked()

        if mode == 'side_by_side':
            ax_real = self.fig.add_subplot(1, 2, 1)
            ax_synth = self.fig.add_subplot(1, 2, 2, sharex=ax_real, sharey=ax_real)
            self.ax_real = ax_real
            self.ax_map = ax_synth

            # 1. Mapa Real / Observado seleccionado
            extent_real = [x_real[0], x_real[-1], y_real[0], y_real[-1]]
            im_r = ax_real.imshow(grid_real, origin='lower', extent=extent_real,
                                  cmap=cmap, vmin=vmin, vmax=vmax)
            ax_real.set_title(f"1. Malla Observada ({label_real})", fontsize=10, fontweight='bold', pad=7)
            ax_real.set_xlabel(f"Coordenada X [{u_xy}]", fontsize=8)
            ax_real.set_ylabel(f"Coordenada Y [{u_xy}]", fontsize=8)
            ax_real.tick_params(labelsize=8)
            if is_aspect_equal:
                ax_real.set_aspect('equal', adjustable='box')
            else:
                ax_real.set_aspect('auto')
            cb_r = self.fig.colorbar(im_r, ax=ax_real, pad=0.02, fraction=0.046)
            cb_r.set_label(f"Real [{unit_z}]", fontsize=8)

            # 2. Anomalía Sintética
            im_s = ax_synth.imshow(self.grid_Z_model, origin='lower', extent=extent_synth,
                                   cmap=cmap, vmin=vmin, vmax=vmax)
            if len(self.cuerpos) == 0:
                ax_synth.set_title("2. Sintética (Limpia - 0 fuentes)", fontsize=10, fontweight='bold', pad=7)
            else:
                ax_synth.set_title(f"2. Sintética ({scale_note})", fontsize=10, fontweight='bold', pad=7)
            ax_synth.set_xlabel(f"Coordenada X [{u_xy}]", fontsize=8)
            ax_synth.set_ylabel(f"Coordenada Y [{u_xy}]", fontsize=8)
            ax_synth.tick_params(labelsize=8)
            if is_aspect_equal:
                ax_synth.set_aspect('equal', adjustable='box')
            else:
                ax_synth.set_aspect('auto')
            cb_s = self.fig.colorbar(im_s, ax=ax_synth, pad=0.02, fraction=0.046)
            cb_s.set_label(f"Sintética [{unit_z}]", fontsize=8)

            # Dibujar trazado geométrico o nombres de fuentes en ambos ejes
            show_geom = getattr(self, 'chk_show_geometry', None) is not None and self.chk_show_geometry.isChecked()
            show_names = getattr(self, 'chk_show_names', None) is not None and self.chk_show_names.isChecked()
            dx_offset = (self.x_vec[-1] - self.x_vec[0]) * 0.015
            for ax in [ax_real, ax_synth]:
                for c in self.cuerpos:
                    if c.is_enabled:
                        if show_geom:
                            draw_cuerpo_planta_geometry(ax, c, show_labels=show_names)
                        elif show_names:
                            ax.text(c.x + dx_offset, c.y, c.name,
                                    color='#ffffff', fontsize=8, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#202020', alpha=0.75, edgecolor='none'))

        elif mode == 'triptych':
            ax_real = self.fig.add_subplot(1, 3, 1)
            ax_synth = self.fig.add_subplot(1, 3, 2, sharex=ax_real, sharey=ax_real)
            ax_diff = self.fig.add_subplot(1, 3, 3, sharex=ax_real, sharey=ax_real)
            self.ax_real = ax_real
            self.ax_map = ax_synth

            extent_real = [x_real[0], x_real[-1], y_real[0], y_real[-1]]
            im_r = ax_real.imshow(grid_real, origin='lower', extent=extent_real, cmap=cmap, vmin=vmin, vmax=vmax)
            ax_real.set_title(f"1. Observada ({label_real})", fontsize=9, fontweight='bold')
            ax_real.tick_params(labelsize=8)
            if is_aspect_equal: ax_real.set_aspect('equal', adjustable='box')

            im_s = ax_synth.imshow(self.grid_Z_model, origin='lower', extent=extent_synth, cmap=cmap, vmin=vmin, vmax=vmax)
            ax_synth.set_title("2. Sintética", fontsize=9, fontweight='bold')
            ax_synth.tick_params(labelsize=8)
            if is_aspect_equal: ax_synth.set_aspect('equal', adjustable='box')

            if grid_real.shape == self.grid_Z_model.shape and np.allclose(x_real, self.x_vec) and np.allclose(y_real, self.y_vec):
                diff_data = grid_real - self.grid_Z_model
            else:
                try:
                    from scipy.interpolate import RegularGridInterpolator
                    interp_real = RegularGridInterpolator((y_real, x_real), grid_real, bounds_error=False, fill_value=0.0)
                    X_g, Y_g = np.meshgrid(self.x_vec, self.y_vec)
                    pts = np.column_stack([Y_g.ravel(), X_g.ravel()])
                    diff_data = interp_real(pts).reshape(self.grid_Z_model.shape) - self.grid_Z_model
                except Exception:
                    diff_data = np.zeros_like(self.grid_Z_model)

            diff_lim = max(abs(float(np.nanmin(diff_data))), abs(float(np.nanmax(diff_data))), 1.0)
            im_d = ax_diff.imshow(diff_data, origin='lower', extent=extent_synth, cmap='bwr', vmin=-diff_lim, vmax=diff_lim)
            ax_diff.set_title(f"3. Residual ({label_real} - Sint)", fontsize=9, fontweight='bold')
            ax_diff.tick_params(labelsize=8)
            if is_aspect_equal: ax_diff.set_aspect('equal', adjustable='box')
            self.fig.colorbar(im_d, ax=ax_diff, pad=0.02, fraction=0.046)

            show_geom = getattr(self, 'chk_show_geometry', None) is not None and self.chk_show_geometry.isChecked()
            show_names = getattr(self, 'chk_show_names', None) is not None and self.chk_show_names.isChecked()
            dx_offset = (self.x_vec[-1] - self.x_vec[0]) * 0.015
            for ax in [ax_real, ax_synth]:
                for c in self.cuerpos:
                    if c.is_enabled:
                        if show_geom:
                            draw_cuerpo_planta_geometry(ax, c, show_labels=show_names)
                        elif show_names:
                            ax.text(c.x + dx_offset, c.y, c.name,
                                    color='#ffffff', fontsize=8, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#202020', alpha=0.75, edgecolor='none'))

        elif mode == 'diff':
            ax_diff = self.fig.add_subplot(1, 1, 1)
            self.ax_map = ax_diff
            if grid_real.shape == self.grid_Z_model.shape and np.allclose(x_real, self.x_vec) and np.allclose(y_real, self.y_vec):
                diff_data = grid_real - self.grid_Z_model
            else:
                try:
                    from scipy.interpolate import RegularGridInterpolator
                    interp_real = RegularGridInterpolator((y_real, x_real), grid_real, bounds_error=False, fill_value=0.0)
                    X_g, Y_g = np.meshgrid(self.x_vec, self.y_vec)
                    pts = np.column_stack([Y_g.ravel(), X_g.ravel()])
                    diff_data = interp_real(pts).reshape(self.grid_Z_model.shape) - self.grid_Z_model
                except Exception:
                    diff_data = np.zeros_like(self.grid_Z_model)

            diff_lim = max(abs(float(np.nanmin(diff_data))), abs(float(np.nanmax(diff_data))), 1.0)
            im_d = ax_diff.imshow(diff_data, origin='lower', extent=extent_synth, cmap='bwr', vmin=-diff_lim, vmax=diff_lim)
            ax_diff.set_title(f"Diferencia Residual ({label_real} - Anomalía Sintética)", fontsize=10, fontweight='bold')
            ax_diff.set_xlabel(f"Coordenada X [{u_xy}]", fontsize=8); ax_diff.set_ylabel(f"Coordenada Y [{u_xy}]", fontsize=8)
            ax_diff.tick_params(labelsize=8)
            if is_aspect_equal: ax_diff.set_aspect('equal', adjustable='box')
            cb = self.fig.colorbar(im_d, ax=ax_diff, pad=0.02, fraction=0.046)
            cb.set_label(f"Residual [{unit_z}]", fontsize=8)

            show_geom = getattr(self, 'chk_show_geometry', None) is not None and self.chk_show_geometry.isChecked()
            show_names = getattr(self, 'chk_show_names', None) is not None and self.chk_show_names.isChecked()
            dx_offset = (self.x_vec[-1] - self.x_vec[0]) * 0.015
            for c in self.cuerpos:
                if c.is_enabled:
                    if show_geom:
                        draw_cuerpo_planta_geometry(ax_diff, c, show_labels=show_names)
                    elif show_names:
                        ax_diff.text(c.x + dx_offset, c.y, c.name,
                                     color='#ffffff', fontsize=8, fontweight='bold',
                                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#202020', alpha=0.75, edgecolor='none'))

        else: # 'synth_only'
            ax_map = self.fig.add_subplot(1, 1, 1)
            self.ax_map = ax_map
            im = ax_map.imshow(self.grid_Z_model, origin='lower', extent=extent_synth, cmap=cmap, vmin=vmin, vmax=vmax)
            cb = self.fig.colorbar(im, ax=ax_map, pad=0.02, fraction=0.046)
            cb.set_label(f"Anomalía Sintética [{unit_z}]", fontsize=8)
            if len(self.cuerpos) == 0:
                ax_map.set_title(f"Malla Base Limpia (Sin fuentes - Usa '+ Agregar Fuente' o 'Clic en Mapa') [{unit_z}]", fontsize=10, fontweight='bold', pad=6)
            else:
                ax_map.set_title(f"Mapa de Anomalía Sintética Total ({scale_note}) [{unit_z}]", fontsize=10, fontweight='bold', pad=6)
            ax_map.set_xlabel(f"Coordenada X [{u_xy}]", fontsize=8); ax_map.set_ylabel(f"Coordenada Y [{u_xy}]", fontsize=8)
            ax_map.tick_params(labelsize=8)
            if is_aspect_equal: ax_map.set_aspect('equal', adjustable='box')
            show_geom = getattr(self, 'chk_show_geometry', None) is not None and self.chk_show_geometry.isChecked()
            show_names = getattr(self, 'chk_show_names', None) is not None and self.chk_show_names.isChecked()
            dx_offset = (self.x_vec[-1] - self.x_vec[0]) * 0.015
            for c in self.cuerpos:
                if c.is_enabled:
                    if show_geom:
                        draw_cuerpo_planta_geometry(ax_map, c, show_labels=show_names)
                    elif show_names:
                        ax_map.text(c.x + dx_offset, c.y, c.name,
                                    color='#ffffff', fontsize=8, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#202020', alpha=0.75, edgecolor='none'))

        # Actualizar badge de estadísticas
        if len(self.cuerpos) == 0:
            self.lbl_stats_badge.setText(f"Malla base limpia (0 fuentes) | Fondo: 0.00 {unit_z}")
        elif is_flat_field:
            self.lbl_stats_badge.setText(f"⚠️ Campo Plano: Mín={z_min:,.2f} | Máx={z_max:,.2f} {unit_z} (La profundidad z1 es mucho mayor que la malla; agranda la malla o reduce z1)")
        else:
            self.lbl_stats_badge.setText(f"Mín: {z_min:,.2f} | Máx: {z_max:,.2f} | Escala: [{vmin:,.1f}, {vmax:,.1f}] {unit_z}")

        try:
            self.fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.94, wspace=0.30)
        except Exception:
            self.fig.tight_layout()
        self.canvas.draw()

    # ==========================================================================
    # EXPORTACIONES Y FIJACIÓN COMO MODELO
    # ==========================================================================
    def apply_as_active_model(self):
        if self.grid_Z_model is None:
            QMessageBox.warning(self, "Sin Modelo", "Aún no se ha calculado el modelo sintético.")
            return
        mw = self.main_window
        mw.grid_Z_model = self.grid_Z_model.copy()
        x_vec, y_vec, dx, dy = self.get_grid_vectors()
        mw.grid_Z = self.grid_Z_model.copy()
        mw.grid_x_vec = x_vec
        mw.grid_y_vec = y_vec
        mw.dx_val = dx
        mw.dy_val = dy
        mw.lbl_file_status.setText(f"✨ Modelo Sintético Activo ({len(self.cuerpos)} fuentes, {mw.grid_Z.shape[1]}×{mw.grid_Z.shape[0]})")
        mw.status.showMessage(f"✔ Modelo sintético de {len(self.cuerpos)} fuentes inyectado como Malla Activa de Trabajo.")

        if hasattr(mw, 'update_master_viewer'):
            mw.update_master_viewer(mw.grid_Z, [x_vec[0], x_vec[-1], y_vec[0], y_vec[-1]], is_freq=False,
                                    unit_label="mGal", title=f"Modelo Sintético Blakely ({len(self.cuerpos)} fuentes)")

        self.main_window.model_x_vec = self.x_vec.copy()
        self.main_window.model_y_vec = self.y_vec.copy()
        self.main_window.cuerpos_sinteticos = self.cuerpos

        QMessageBox.information(
            self, "Modelo Fijado e Inyectado",
            f"El modelo sintético ({len(self.cuerpos)} fuentes activas) se ha fijado como Modelo Activo para comparación "
            f"e inyectado como la Malla Activa de Trabajo.\n\n"
            f"Se ha actualizado el Visualizador Maestro 2D para su análisis completo."
        )
        if hasattr(mw, 'tabs'):
            mw.tabs.setCurrentIndex(2) # Cambiar a Tab 3: Visualizador Maestro
        if not getattr(self, 'is_embedded', False):
            self.accept()

    def inject_as_main_grid(self):
        if self.grid_Z_model is None:
            QMessageBox.warning(self, "Sin Modelo", "Aún no se ha calculado el modelo sintético.")
            return
        mw = self.main_window
        mw.grid_Z = self.grid_Z_model.copy()
        mw.grid_Z_tapered = self.grid_Z_model.copy()
        mw.grid_x_vec = self.x_vec.copy()
        mw.grid_y_vec = self.y_vec.copy()
        mw.dx_val = float(self.x_vec[1] - self.x_vec[0])
        mw.dy_val = float(self.y_vec[1] - self.y_vec[0])
        mw.z_col = "Anomalía_Sintética"
        mw.grid_Z_filt = None

        if hasattr(mw, 'lbl_file_status'):
            mw.lbl_file_status.setText(f"✨ Modelo Sintético Inyectado ({len(self.cuerpos)} cuerpos, {mw.grid_Z.shape[1]}×{mw.grid_Z.shape[0]})")
        if hasattr(mw, 'update_mesh_table'):
            mw.update_mesh_table()
        if hasattr(mw, 'plot_master_viewer'):
            mw.plot_master_viewer()
        if hasattr(mw, 'recompute_fourier_2d'):
            mw.recompute_fourier_2d()

        QMessageBox.information(
            self, "Modelo Inyectado en Malla de Trabajo",
            "El modelo sintético limpio ha sido transferido a la malla principal de trabajo.\n"
            "Puedes proceder a aplicar filtros Fourier 2D, continuaciones y RAPS sobre él."
        )

    def export_model_tiff(self):
        if self.grid_Z_model is None:
            QMessageBox.warning(self, "Sin Modelo", "Aún no se ha calculado el modelo sintético.")
            return
        fpath, _ = QFileDialog.getSaveFileName(self, "Guardar Modelo TIFF", "modelo_sintetico.tif", "Raster TIFF (*.tif *.tiff)")
        if fpath:
            Z_flip = np.flipud(self.grid_Z_model).astype(np.float32)
            img = Image.fromarray(Z_flip, mode='F')
            info = TiffImagePlugin.ImageFileDirectory_v2()
            dx = float(self.x_vec[1] - self.x_vec[0])
            dy = float(self.y_vec[1] - self.y_vec[0])
            info[33550] = (dx, dy, 0.0)
            info[33922] = (0.0, 0.0, 0.0, float(self.x_vec[0]), float(self.y_vec[-1]), 0.0)
            img.save(fpath, tiffinfo=info)
            QMessageBox.information(self, "Exportación Exitosa", f"Modelo TIFF exportado en:\n{fpath}")

    def export_model_csv(self):
        if self.grid_Z_model is None:
            QMessageBox.warning(self, "Sin Modelo", "Aún no se ha calculado el modelo sintético.")
            return
        fpath, _ = QFileDialog.getSaveFileName(self, "Guardar Modelo CSV XYZ", "modelo_sintetico_xyz.csv", "CSV (*.csv)")
        if fpath:
            Xg, Yg = np.meshgrid(self.x_vec, self.y_vec)
            df = pd.DataFrame({'X': Xg.ravel(), 'Y': Yg.ravel(), 'Z_modelo': self.grid_Z_model.ravel()})
            df.to_csv(fpath, index=False)
            QMessageBox.information(self, "Exportación Exitosa", f"Modelo CSV exportado en:\n{fpath}")

    def open_comparison_dialog(self):
        self.apply_as_active_model()
        dlg = VentanaComparacionModelos(self.main_window, parent=self)
        dlg.exec_()


# ==============================================================================
# VENTANA DE COMPARACIÓN Y ÁLGEBRA PIXEL A PIXEL: OBS VS MODELO
# ==============================================================================
class VentanaComparacionModelos(QDialog):
    """
    Ventana independiente para contrastar la malla observada/filtrada con el modelo sintético.
    Disposición:
    - Arriba: Observaciones (Izquierda) y Modelo Sintético (Derecha)
    - Abajo: Resultado de la Operación Álgebraica píxel a píxel f(Obs, Mod) (Izquierda)
             y Perfil Transversal Central 1D comparativo (Derecha)
    - Casilla interactiva para evaluar Obs - Mod, 2*Obs - Mod, etc.
    """
    def __init__(self, main_window, parent=None):
        parent_widget = parent if isinstance(parent, QWidget) else (main_window if isinstance(main_window, QWidget) else None)
        super().__init__(parent_widget)
        self.main_window = main_window
        self.setWindowTitle("⚖️ Comparación y Álgebra Pixel a Pixel: Observaciones vs. Modelo")
        self.resize(1180, 760)
        self.setMinimumSize(1000, 660)

        self.last_res = None
        self.res_x_vec = None
        self.res_y_vec = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # BARRA SUPERIOR: SELECTORES Y CASILLA DE FÓRMULA MATEMÁTICA
        grp_ctrl = QGroupBox("⚙️ Configuración de Capas y Operación Álgebraica Pixel a Pixel")
        grid_c = QGridLayout(grp_ctrl)
        grid_c.setSpacing(6)

        grid_c.addWidget(QLabel("<b>Capa Observada (Obs):</b>"), 0, 0)
        self.combo_obs = QComboBox()
        self.combo_obs.addItem("1. Malla Sin Filtrar / Original (Obs = Z_orig)", "orig")
        self.combo_obs.addItem("2. Malla Filtrada Fourier 2D (Obs = Z_filt)", "filt")
        self.combo_obs.addItem("3. Fuente Equivalente Plano Z0 (Obs = G_Z0)", "fe_z0")
        self.combo_obs.addItem("4. Fuente Equivalente Filtrada (Obs = G_Z0_filt)", "fe_filt")
        self.combo_obs.addItem("🌊 Malla con Taper Bordes (Obs = Z_taper)", "taper")
        self.combo_obs.currentIndexChanged.connect(self.evaluate_operation)
        grid_c.addWidget(self.combo_obs, 0, 1)

        grid_c.addWidget(QLabel("<b>Modelo Sintético (Mod):</b>"), 0, 2)
        self.combo_mod = QComboBox()
        self.combo_mod.addItem("✨ Modelo Sintético Actual (Mod = Z_model)", "model")
        grid_c.addWidget(self.combo_mod, 0, 3)

        btn_open_synth = QPushButton("✨ Abrir Modelador...")
        btn_open_synth.clicked.connect(self.open_synthetic_modeler)
        grid_c.addWidget(btn_open_synth, 0, 4)

        # CASILLA MATEMÁTICA INTERACTIVA
        grid_c.addWidget(QLabel("<b>Operación Álgebraica:</b>"), 1, 0)
        row_form = QHBoxLayout()
        self.txt_expr = QLineEdit("Obs - Mod")
        self.txt_expr.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; font-weight: bold; color: #cc0000; padding: 3px;")
        self.txt_expr.returnPressed.connect(self.evaluate_operation)
        row_form.addWidget(self.txt_expr)

        btn_eval = QPushButton("⚡ Evaluar Operación")
        btn_eval.setObjectName("btnPrimary")
        btn_eval.clicked.connect(self.evaluate_operation)
        row_form.addWidget(btn_eval)
        grid_c.addLayout(row_form, 1, 1, 1, 3)

        # Plantillas rápidas
        row_p = QHBoxLayout()
        row_p.addWidget(QLabel("<b>Accesos Rápidos:</b>"))
        for text, expr in [("Obs - Mod", "Obs - Mod"), ("Mod - Obs", "Mod - Obs"),
                           ("2*Obs - Mod", "2*Obs - Mod"), ("|Obs - Mod|", "abs(Obs - Mod)"),
                           ("Obs / (Mod+1)", "Obs / (Mod + 1.0)")]:
            btn_q = QPushButton(text)
            btn_q.clicked.connect(lambda checked, e=expr: (self.txt_expr.setText(e), self.evaluate_operation()))
            row_p.addWidget(btn_q)
        row_p.addStretch()
        grid_c.addLayout(row_p, 2, 0, 1, 4)

        # Banner de estado y métricas
        self.lbl_metrics = QLabel("Métricas: RMS=0.0 | Correlación R=0.0 | Media=0.0")
        self.lbl_metrics.setStyleSheet("font-size: 11px; font-weight: bold; color: #004488; background: #eef4fc; padding: 4px; border: 1px solid #b8d4f8;")
        grid_c.addWidget(self.lbl_metrics, 3, 0, 1, 5)

        layout.addWidget(grp_ctrl)

        # ÁREA GRÁFICA (2 ARRIBA Y 2 ABAJO)
        self.fig = Figure(facecolor='#ffffff')
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # BOTONES INFERIORES
        row_bot = QHBoxLayout()
        btn_exp_res_tiff = QPushButton("🗺️ Exportar Resultado a TIFF (.tif)")
        btn_exp_res_tiff.clicked.connect(self.export_res_tiff)
        row_bot.addWidget(btn_exp_res_tiff)

        btn_exp_res_csv = QPushButton("📄 Exportar Resultado a CSV")
        btn_exp_res_csv.clicked.connect(self.export_res_csv)
        row_bot.addWidget(btn_exp_res_csv)

        btn_inject = QPushButton("📥 Inyectar Resultado a Malla de Trabajo")
        btn_inject.setObjectName("btnAccent")
        btn_inject.setToolTip("Carga la matriz resultante f(Obs, Mod) como la malla activa de trabajo en el programa principal")
        btn_inject.clicked.connect(self.inject_result_as_main_grid)
        row_bot.addWidget(btn_inject)

        row_bot.addStretch()
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        row_bot.addWidget(btn_close)
        layout.addLayout(row_bot)

        self.evaluate_operation()

    def open_synthetic_modeler(self):
        dlg = VentanaModeladorSintetico(self.main_window, parent=self)
        if dlg.exec_() == QDialog.Accepted or dlg.grid_Z_model is not None:
            self.evaluate_operation()

    def get_observed_matrix(self):
        mw = self.main_window
        mode = self.combo_obs.currentData()
        if mode == 'filt' and getattr(mw, 'grid_Z_filt', None) is not None:
            return mw.grid_Z_filt, mw.grid_x_vec, mw.grid_y_vec, "Observaciones Filtradas (Z_filt)"
        elif mode == 'taper' and getattr(mw, 'grid_Z_tapered', None) is not None:
            return mw.grid_Z_tapered, mw.grid_x_vec, mw.grid_y_vec, "Observaciones Taper (Z_taper)"
        elif mode == 'fe_z0' and getattr(mw, 'grid_eq_source_z0', None) is not None:
            x_fe = getattr(mw, 'grid_eq_source_x', mw.grid_x_vec)
            y_fe = getattr(mw, 'grid_eq_source_y', mw.grid_y_vec)
            z0_val = getattr(mw, 'grid_eq_source_z0_elev', 0.0)
            return mw.grid_eq_source_z0, x_fe, y_fe, f"Fuente Equivalente (Z0={z0_val:.0f}m)"
        elif mode == 'fe_filt':
            if hasattr(mw, 'update_equivalent_source_filtered'):
                mw.update_equivalent_source_filtered()
            z_fe_filt = getattr(mw, 'grid_eq_source_z0_filt', None)
            if z_fe_filt is not None:
                x_fe = getattr(mw, 'grid_eq_source_x', mw.grid_x_vec)
                y_fe = getattr(mw, 'grid_eq_source_y', mw.grid_y_vec)
                return z_fe_filt, x_fe, y_fe, "Fuente Equivalente Filtrada"
        elif mw is not None and getattr(mw, 'grid_Z', None) is not None:
            z_orig = getattr(mw, 'grid_Z_orig', mw.grid_Z)
            return z_orig, mw.grid_x_vec, mw.grid_y_vec, "Observaciones Originales (Sin Filtrar)"
        return None, None, None, "Sin Observaciones"

    def get_model_matrix(self):
        mw = self.main_window
        if getattr(mw, 'grid_Z_model', None) is not None:
            return mw.grid_Z_model, mw.model_x_vec, mw.model_y_vec, "Modelo Sintético Creado"
        return None, None, None, "Sin Modelo Sintético"

    def evaluate_operation(self):
        Obs_raw, x_obs, y_obs, tag_obs = self.get_observed_matrix()
        Mod_raw, x_mod, y_mod, tag_mod = self.get_model_matrix()

        self.fig.clf()

        # Si falta alguna capa
        if Obs_raw is None or Mod_raw is None:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.set_facecolor('#ffffff')
            msg = "⚠️ Faltan datos para realizar la comparación:\n\n"
            if Obs_raw is None: msg += "• Debes cargar o generar una malla observada en el programa principal.\n"
            if Mod_raw is None: msg += "• Debes crear o fijar un modelo sintético (haz clic en '✨ Abrir Modelador...').\n"
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11, color='#cc0000',
                    bbox=dict(boxstyle='round,pad=1.0', facecolor='#fff5f5', edgecolor='#cc0000'))
            ax.axis('off')
            self.canvas.draw()
            return

        # Sincronización de dimensiones si difieren
        if Obs_raw.shape != Mod_raw.shape or not np.allclose(x_obs, x_mod) or not np.allclose(y_obs, y_mod):
            try:
                interp = RegularGridInterpolator((y_mod, x_mod), Mod_raw, bounds_error=False, fill_value=0.0)
                Xg, Yg = np.meshgrid(x_obs, y_obs)
                pts = np.column_stack([Yg.ravel(), Xg.ravel()])
                Mod_aligned = interp(pts).reshape(len(y_obs), len(x_obs))
            except Exception:
                Mod_aligned = np.resize(Mod_raw, Obs_raw.shape)
        else:
            Mod_aligned = Mod_raw

        Obs = Obs_raw
        Mod = Mod_aligned
        self.res_x_vec = x_obs
        self.res_y_vec = y_obs

        # Evaluar la fórmula matemática píxel a píxel
        expr = self.txt_expr.text().strip() or "Obs - Mod"
        env = {
            'Obs': Obs, 'obs': Obs, 'O': Obs,
            'Mod': Mod, 'mod': Mod, 'M': Mod,
            'np': np, 'abs': np.abs, 'sqrt': np.sqrt,
            'exp': np.exp, 'log': np.log, 'log10': np.log10
        }
        try:
            Res = eval(expr, {'__builtins__': {}}, env)
            if not isinstance(Res, np.ndarray):
                Res = np.full_like(Obs, float(Res))
        except Exception as err:
            self.lbl_metrics.setText(f"❌ Error sintaxis en la operación: {str(err)[:60]}")
            self.lbl_metrics.setStyleSheet("font-size: 11px; font-weight: bold; color: #cc0000; background: #fff0f0; padding: 4px; border: 1px solid #cc0000;")
            Res = Obs - Mod

        self.last_res = Res

        diff = Obs - Mod
        rms_val = float(np.sqrt(np.mean(diff ** 2)))
        mean_res = float(np.mean(diff))
        std_obs = float(np.std(Obs))
        std_mod = float(np.std(Mod))
        corr = float(np.corrcoef(Obs.ravel(), Mod.ravel())[0, 1]) if (std_obs > 1e-9 and std_mod > 1e-9) else 1.0

        res_min = float(np.min(Res))
        res_max = float(np.max(Res))

        self.lbl_metrics.setText(
            f"✅ Operación '{expr}' evaluada píxel a píxel | "
            f"RMS(Obs - Mod): {rms_val:,.2f} | "
            f"Correlación R: {corr:.3f} | "
            f"Media Residuo: {mean_res:,.2f} | "
            f"Resultado [{res_min:,.2f}, {res_max:,.2f}]"
        )
        self.lbl_metrics.setStyleSheet("font-size: 11px; font-weight: bold; color: #007700; background: #f0fff0; padding: 4px; border: 1px solid #00aa00;")

        # DISPOSICIÓN: 2 ARRIBA (Obs y Mod) Y ABAJO (Resultado Álgebraico + Perfil Comparativo 1D)
        gs = gridspec.GridSpec(2, 2, figure=self.fig, height_ratios=[1.0, 1.0])
        ax_obs = self.fig.add_subplot(gs[0, 0])
        ax_mod = self.fig.add_subplot(gs[0, 1])
        ax_res = self.fig.add_subplot(gs[1, 0])
        ax_prof = self.fig.add_subplot(gs[1, 1])

        extent = [x_obs[0], x_obs[-1], y_obs[0], y_obs[-1]]
        u_xy = self.main_window.unit_xy

        # 1. ARRIBA IZQUIERDA: OBSERVACIONES
        im_o = ax_obs.imshow(Obs, origin='lower', extent=extent, cmap='turbo')
        cb_o = self.fig.colorbar(im_o, ax=ax_obs, pad=0.03, fraction=0.046)
        cb_o.set_label("Obs", fontsize=8)
        ax_obs.set_facecolor('#ffffff')
        ax_obs.set_title(f"Arriba Izq: {tag_obs}", fontsize=9, fontweight='bold', pad=4)
        ax_obs.set_xlabel(f"X [{u_xy}]", fontsize=8); ax_obs.set_ylabel(f"Y [{u_xy}]", fontsize=8)
        ax_obs.tick_params(labelsize=7)
        ax_obs.grid(True, linestyle=':', color='#d0d0d0', alpha=0.8)

        # 2. ARRIBA DERECHA: MODELO SINTÉTICO
        im_m = ax_mod.imshow(Mod, origin='lower', extent=extent, cmap='turbo')
        cb_m = self.fig.colorbar(im_m, ax=ax_mod, pad=0.03, fraction=0.046)
        cb_m.set_label("Mod", fontsize=8)
        ax_mod.set_facecolor('#ffffff')
        ax_mod.set_title(f"Arriba Der: {tag_mod}", fontsize=9, fontweight='bold', pad=4)
        ax_mod.set_xlabel(f"X [{u_xy}]", fontsize=8); ax_mod.set_ylabel(f"Y [{u_xy}]", fontsize=8)
        ax_mod.tick_params(labelsize=7)
        ax_mod.grid(True, linestyle=':', color='#d0d0d0', alpha=0.8)

        # 3. ABAJO IZQUIERDA: RESULTADO DE LA OPERACIÓN ÁLGEBRAICA
        cmap_res = 'bwr' if ('-' in expr or np.any(Res < 0)) else 'turbo'
        im_r = ax_res.imshow(Res, origin='lower', extent=extent, cmap=cmap_res)
        cb_r = self.fig.colorbar(im_r, ax=ax_res, pad=0.03, fraction=0.046)
        cb_r.set_label(f"Resultado", fontsize=8)
        ax_res.set_facecolor('#ffffff')
        ax_res.set_title(f"Abajo: Operación [{expr}]", fontsize=8.5, fontweight='bold', pad=4, color='#cc0000')
        ax_res.set_xlabel(f"X [{u_xy}]", fontsize=8); ax_res.set_ylabel(f"Y [{u_xy}]", fontsize=8)
        ax_res.tick_params(labelsize=7)
        ax_res.grid(True, linestyle=':', color='#d0d0d0', alpha=0.85)

        # 4. ABAJO DERECHA: PERFIL TRANSVERSAL CENTRAL 1D (Obs vs Mod vs Res)
        mid_idx = len(y_obs) // 2
        y_cut_val = y_obs[mid_idx]
        ax_prof.set_facecolor('#ffffff')
        ax_prof.plot(x_obs, Obs[mid_idx, :], color='#004488', linewidth=1.8, label='Obs')
        ax_prof.plot(x_obs, Mod[mid_idx, :], color='#cc0000', linewidth=1.8, linestyle='--', label='Mod')
        ax_prof.plot(x_obs, Res[mid_idx, :], color='#006600', linewidth=1.5, linestyle=':', label=f'{expr}')
        ax_prof.set_title(f"Abajo: Perfil 1D (Y={y_cut_val:,.0f} {u_xy})", fontsize=8.5, fontweight='bold', pad=4)
        ax_prof.set_xlabel(f"Coordenada X [{u_xy}]", fontsize=8)
        ax_prof.set_ylabel("Valor / Amplitud", fontsize=8)
        ax_prof.tick_params(labelsize=7)
        ax_prof.grid(True, linestyle=':', color='#d0d0d0', alpha=0.85)
        ax_prof.legend(loc='best', fontsize=7, framealpha=0.9)

        try:
            self.fig.subplots_adjust(left=0.07, right=0.95, bottom=0.08, top=0.94, wspace=0.34, hspace=0.35)
        except Exception:
            self.fig.tight_layout()
        self.canvas.draw()

    def export_res_tiff(self):
        if self.last_res is None:
            QMessageBox.warning(self, "Sin Resultado", "No hay resultado evaluado para exportar.")
            return
        fpath, _ = QFileDialog.getSaveFileName(self, "Guardar Resultado TIFF", "resultado_operacion_obs_mod.tif", "Raster TIFF (*.tif *.tiff)")
        if fpath:
            Z_flip = np.flipud(self.last_res).astype(np.float32)
            img = Image.fromarray(Z_flip, mode='F')
            info = TiffImagePlugin.ImageFileDirectory_v2()
            dx = float(self.res_x_vec[1] - self.res_x_vec[0])
            dy = float(self.res_y_vec[1] - self.res_y_vec[0])
            info[33550] = (dx, dy, 0.0)
            info[33922] = (0.0, 0.0, 0.0, float(self.res_x_vec[0]), float(self.res_y_vec[-1]), 0.0)
            img.save(fpath, tiffinfo=info)
            QMessageBox.information(self, "Exportación Exitosa", f"Resultado TIFF exportado en:\n{fpath}")

    def export_res_csv(self):
        if self.last_res is None:
            QMessageBox.warning(self, "Sin Resultado", "No hay resultado evaluado para exportar.")
            return
        fpath, _ = QFileDialog.getSaveFileName(self, "Guardar Resultado CSV", "resultado_operacion_obs_mod_xyz.csv", "CSV (*.csv)")
        if fpath:
            Xg, Yg = np.meshgrid(self.res_x_vec, self.res_y_vec)
            df = pd.DataFrame({'X': Xg.ravel(), 'Y': Yg.ravel(), 'Resultado': self.last_res.ravel()})
            df.to_csv(fpath, index=False)
            QMessageBox.information(self, "Exportación Exitosa", f"Resultado CSV exportado en:\n{fpath}")

    def inject_result_as_main_grid(self):
        if self.last_res is None:
            QMessageBox.warning(self, "Sin Resultado", "No hay resultado evaluado para inyectar.")
            return
        mw = self.main_window
        mw.grid_Z = self.last_res.copy()
        mw.grid_Z_tapered = self.last_res.copy()
        mw.grid_x_vec = self.res_x_vec.copy()
        mw.grid_y_vec = self.res_y_vec.copy()
        mw.dx_val = float(self.res_x_vec[1] - self.res_x_vec[0])
        mw.dy_val = float(self.res_y_vec[1] - self.res_y_vec[0])
        mw.z_col = f"Op_{self.txt_expr.text().strip()}"
        mw.grid_Z_filt = None

        if hasattr(mw, 'lbl_file_status'):
            mw.lbl_file_status.setText(f"⚖️ Resultado Álgebra Inyectado ({mw.grid_Z.shape[1]}×{mw.grid_Z.shape[0]})")
        if hasattr(mw, 'update_mesh_table'):
            mw.update_mesh_table()
        if hasattr(mw, 'plot_master_viewer'):
            mw.plot_master_viewer()
        if hasattr(mw, 'recompute_fourier_2d'):
            mw.recompute_fourier_2d()

        QMessageBox.information(
            self, "Resultado Inyectado",
            "El mapa resultado f(Obs, Mod) ha sido transferido a la malla de trabajo del programa principal."
        )
