import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import math
import base64
import tempfile
import os
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import requests
from PIL import Image
import traceback
import kaleido
import CoolProp.CoolProp as CP


# ========================
# CONSTANTS & UNIT CONVERSION
# ========================
BAR_TO_KPA = 100
KPA_TO_BAR = 0.01
C_TO_K = 273.15
F_TO_R = 459.67
PSI_TO_BAR = 0.0689476
G_CONST = 9.80665
MMHG_TO_BAR = 0.00133322

CONSTANTS = {
    "N1": {"gpm, psia": 1.00, "m³/h, bar": 0.865, "m³/h, kPa": 0.0865},
    "N2": {"mm": 0.00214, "inch": 890},
    "N4": {"mm": 76000, "inch": 17300},
    "N5": {"mm": 0.00241, "inch": 1000},
    "N6": {"kg/h, kPa, kg/m³": 2.73, "kg/h, bar, kg/m³": 27.3, "lb/h, psia, lb/ft³": 63.3},
    "N7": {"m³/h, kPa, K (standard)": 4.17, "m³/h, bar, K (standard)": 417, "scfh, psia, R": 1360},
    "N8": {"kg/h, kPa, K": 0.948, "kg/h, bar, K": 94.8, "lb/h, psia, R": 19.3},
    "N9": {"m³/h, kPa, K (standard)": 22.4, "m³/h, bar, K (standard)": 2240, "scfh, psia, R": 7320}
}

WATER_DENSITY_4C = 999.97
AIR_DENSITY_0C = 1.293

# ========================
# FLUID LIBRARY
# ========================
FLUID_LIBRARY = {
    "Water": {
        "type": "liquid",
        "coolprop_name": "Water",
        "visc_func": lambda t, p: calculate_kinematic_viscosity("Water", t, p),
        "k_func": None,
        "pv_func": lambda t, p: calculate_vapor_pressure("Water", t, p),
        "pc_func": lambda: CP.PropsSI('Pcrit', 'Water') / 1e5,
        "rho_func": lambda t, p: calculate_density("Water", t, p)
    },
    "Light Oil": {
        "type": "liquid",
        "coolprop_name": None,
        "sg": 0.85,
        "visc_func": lambda t, p: calculate_kinematic_viscosity("light_oil", t, p),
        "k_func": None,
        "pc": 25.0,
        "pv_func": lambda t, p: 0.0,
        "rho_func": lambda t, p: 0.85 * WATER_DENSITY_4C
    },
    "Heavy Oil": {
        "type": "liquid",
        "coolprop_name": None,
        "sg": 0.92,
        "visc_func": lambda t, p: calculate_kinematic_viscosity("heavy_oil", t, p),
        "k_func": None,
        "pc": 15.0,
        "pv_func": lambda t, p: 0.0,
        "rho_func": lambda t, p: 0.92 * WATER_DENSITY_4C
    },
    "Propane": {
        "type": "liquid",
        "coolprop_name": "Propane",
        "visc_func": lambda t, p: calculate_kinematic_viscosity("Propane", t, p),
        "k_func": None,
        "pv_func": lambda t, p: calculate_vapor_pressure("Propane", t, p),
        "pc_func": lambda: CP.PropsSI('Pcrit', 'Propane') / 1e5,
        "rho_func": lambda t, p: calculate_density("Propane", t, p)
    },
    "Air": {
        "type": "gas",
        "coolprop_name": "Air",
        "sg": 1.0,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Air", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Air", t, p)
    },
    "Natural Gas": {
        "type": "gas",
        "coolprop_name": "Methane",
        "sg": 0.6,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Methane", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Methane", t, p)
    },
    "Steam": {
        "type": "steam",
        "coolprop_name": "Water",
        "sg": None,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Water", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Water", t, p)
    },
    "CO2": {
        "type": "gas",
        "coolprop_name": "CarbonDioxide",
        "sg": 1.52,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("CarbonDioxide", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("CarbonDioxide", t, p)
    },
    "Ammonia": {
        "type": "gas",
        "coolprop_name": "Ammonia",
        "sg": 0.59,
        "visc_func": None,
        "k_func": lambda t, p: calculate_specific_heat_ratio("Ammonia", t, p),
        "pv_func": None,
        "rho_func": lambda t, p: calculate_density("Ammonia", t, p)
    }
}

# ========================
# FLUID PROPERTY FUNCTIONS
# ========================
def calculate_vapor_pressure(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            pv = CP.PropsSI('P', 'T', T, 'Q', 0, fluid_name) / 1e5
            return pv
    except:
        pass
    return 0.0

def calculate_density(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            rho = CP.PropsSI('D', 'T', T, 'P', P, fluid_name)
            return rho
    except:
        pass
    if fluid == "water":
        return WATER_DENSITY_4C
    elif fluid == "air":
        return (press_bar * 1e5) * 28.97 / (8.314462 * (temp_c + C_TO_K))
    return 1000

def calculate_kinematic_viscosity(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            mu = CP.PropsSI('V', 'T', T, 'P', P, fluid_name)
            rho = CP.PropsSI('D', 'T', T, 'P', P, fluid_name)
            nu = mu / rho * 1e6
            return nu
    except:
        pass
    if fluid == "water":
        return 1.79 / (1 + 0.0337 * temp_c + 0.00022 * temp_c**2)
    elif fluid == "light_oil":
        return 32 * math.exp(-0.03 * (temp_c - 40))
    elif fluid == "heavy_oil":
        return 100 * math.exp(-0.03 * (temp_c - 40))
    elif fluid == "propane":
        return 0.2 * math.exp(-0.02 * (temp_c - 20))
    return 1.0

def calculate_specific_heat_ratio(fluid: str, temp_c: float, press_bar: float) -> float:
    try:
        if fluid in FLUID_LIBRARY and FLUID_LIBRARY[fluid].get("coolprop_name"):
            fluid_name = FLUID_LIBRARY[fluid]["coolprop_name"]
            T = temp_c + C_TO_K
            P = press_bar * 1e5
            Cp = CP.PropsSI('C', 'T', T, 'P', P, fluid_name)
            Cv = CP.PropsSI('O', 'T', T, 'P', P, fluid_name)
            return Cp / Cv
    except:
        pass
    if fluid == "air":
        return 1.4 - 0.0001 * temp_c
    elif fluid == "natural_gas":
        return 1.31 - 0.00008 * temp_c
    elif fluid == "steam":
        return 1.33 - 0.0001 * temp_c
    elif fluid == "co2":
        return 1.28 - 0.00005 * temp_c
    elif fluid == "ammonia":
        return 1.32 - 0.00007 * temp_c
    return 1.4

def calculate_ff(pv: float, pc: float) -> float:
    if pc <= 0:
        return 0.96
    return 0.96 - 0.28 * math.sqrt(pv / pc)

# ========================
# VALVE DATABASE
# ========================
class Valve:
    def __init__(self, size_inch: int, rating_class: int, cv_table: dict, 
                 fl: float, xt_table: dict, fd: float, d_inch: float = None,
                 valve_type: int = 3):
        self.size = size_inch
        self.rating_class = rating_class
        self.cv_table = cv_table
        self.fl = fl
        self.xt_table = xt_table
        self.xt = xt_table[100]
        self.fd = fd
        self.diameter = d_inch if d_inch else size_inch * 0.95
        self.valve_type = valve_type
        
    def get_cv_at_opening(self, open_percent: float) -> float:
        open_percent = max(10, min(100, open_percent))
        keys = sorted(self.cv_table.keys())
        for i in range(len(keys)-1):
            if keys[i] <= open_percent <= keys[i+1]:
                x0, x1 = keys[i], keys[i+1]
                y0, y1 = self.cv_table[x0], self.cv_table[x1]
                return y0 + (y1 - y0) * (open_percent - x0) / (x1 - x0)
        if open_percent <= keys[0]:
            return self.cv_table[keys[0]]
        return self.cv_table[keys[-1]]
    
    def get_xt_at_opening(self, open_percent: float) -> float:
        open_percent = max(10, min(100, open_percent))
        keys = sorted(self.xt_table.keys())
        for i in range(len(keys)-1):
            if keys[i] <= open_percent <= keys[i+1]:
                x0, x1 = keys[i], keys[i+1]
                y0, y1 = self.xt_table[x0], self.xt_table[x1]
                return y0 + (y1 - y0) * (open_percent - x0) / (x1 - x0)
        if open_percent <= keys[0]:
            return self.xt_table[keys[0]]
        return self.xt_table[keys[-1]]

VALVE_DATABASE = [
    Valve(0.5, 150, 
          {10:1.2, 20:2.4, 30:4.0, 40:6.0, 50:8.5, 60:12, 70:16, 80:21, 90:26, 100:32}, 
          0.90, 
          {10:0.80, 20:0.78, 30:0.76, 40:0.74, 50:0.72, 60:0.70, 70:0.68, 80:0.66, 90:0.64, 100:0.62},
          0.8, valve_type=3),
    Valve(1, 150, 
          {10:3.0, 20:6.0, 30:10, 40:16, 50:24, 60:32, 70:42, 80:52, 90:62, 100:72}, 
          0.85, 
          {10:0.82, 20:0.80, 30:0.78, 40:0.76, 50:0.74, 60:0.72, 70:0.70, 80:0.68, 90:0.66, 100:0.64},
          0.8, valve_type=3),
    Valve(1.5, 150, 
          {10:6, 20:12, 30:20, 40:32, 50:48, 60:65, 70:85, 80:105, 90:125, 100:145}, 
          0.88, 
          {10:0.80, 20:0.78, 30:0.76, 40:0.74, 50:0.72, 60:0.70, 70:0.68, 80:0.66, 90:0.64, 100:0.62},
          0.8, valve_type=3),
    Valve(2, 150, 
          {10:10, 20:22, 30:36, 40:55, 50:80, 60:110, 70:140, 80:170, 90:200, 100:230}, 
          0.90, 
          {10:0.78, 20:0.76, 30:0.74, 40:0.72, 50:0.70, 60:0.68, 70:0.66, 80:0.64, 90:0.62, 100:0.60},
          0.8, valve_type=3),
    Valve(3, 300, 
          {10:25, 20:50, 30:80, 40:120, 50:170, 60:220, 70:280, 80:340, 90:400, 100:460}, 
          0.92, 
          {10:0.75, 20:0.73, 30:0.71, 40:0.69, 50:0.67, 60:0.65, 70:0.63, 80:0.61, 90:0.59, 100:0.57},
          0.8, valve_type=3),
    Valve(4, 300, 
          {10:45, 20:90, 30:140, 40:210, 50:300, 60:390, 70:490, 80:590, 90:690, 100:800}, 
          0.93, 
          {10:0.72, 20:0.70, 30:0.68, 40:0.66, 50:0.64, 60:0.62, 70:0.60, 80:0.58, 90:0.56, 100:0.54},
          0.8, valve_type=3),
    Valve(6, 300, 
          {10:100, 20:200, 30:320, 40:480, 50:680, 60:900, 70:1150, 80:1400, 90:1650, 100:1900}, 
          0.94, 
          {10:0.70, 20:0.68, 30:0.66, 40:0.64, 50:0.62, 60:0.60, 70:0.58, 80:0.56, 90:0.54, 100:0.52},
          0.8, valve_type=3),
    Valve(8, 300, 
          {10:180, 20:360, 30:580, 40:880, 50:1250, 60:1650, 70:2100, 80:2550, 90:3000, 100:3500}, 
          0.95, 
          {10:0.68, 20:0.66, 30:0.64, 40:0.62, 50:0.60, 60:0.58, 70:0.56, 80:0.54, 90:0.52, 100:0.50},
          0.8, valve_type=3),
]

VALVE_MODELS = {
    "0.5\" E31": "https://example.com/models/0_5E31.glb",
    "1\" E31": "https://example.com/models/1E31.glb",
    "1.5\" E31": "https://example.com/models/1_5E31.glb",
    "2\" E31": "https://example.com/models/2E31.glb",
    "3\" E32": "https://example.com/models/3E32.glb",
    "4\" E32": "https://example.com/models/4E32.glb",
    "6\" E32": "https://example.com/models/6E32.glb",
    "8\" E32": "https://example.com/models/8E32.glb",
}

# ========================
# CV CALCULATION MODULE
# ========================
def reynolds_number(flow_m3h: float, d_m: float, visc_cst: float) -> float:
    if visc_cst < 0.1:
        return 1e6
    N4 = CONSTANTS["N4"]["mm"]
    d_mm = d_m * 1000
    return N4 * flow_m3h / (visc_cst * d_mm)

def viscosity_correction(rev: float) -> float:
    if rev >= 40000:
        return 1.0
    elif rev <= 56:
        return 0.019 * (rev ** 0.67)
    else:
        return (rev / 40000) ** 0.15

def calculate_piping_factor_fp(valve_d_inch: float, pipe_d_inch: float, cv_100: float) -> float:
    if pipe_d_inch <= valve_d_inch or abs(pipe_d_inch - valve_d_inch) < 0.01:
        return 1.0
    d_ratio = valve_d_inch / pipe_d_inch
    sumK = 1.5 * (1 - d_ratio**2)**2
    term = 1 + (sumK / CONSTANTS["N2"]["inch"]) * (cv_100 / valve_d_inch**2)**2
    Fp = 1 / math.sqrt(term)
    return Fp

def calculate_flp(valve, valve_d_inch: float, pipe_d_inch: float, cv_100: float) -> float:
    if abs(pipe_d_inch - valve_d_inch) < 0.01:
        return valve.fl
    d_ratio = valve_d_inch / pipe_d_inch
    K1 = 0.5 * (1 - d_ratio**2)**2
    KB1 = 1 - d_ratio**4
    Ki = K1 + KB1
    term = (Ki / CONSTANTS["N2"]["inch"]) * (cv_100 / valve_d_inch**2)**2 + 1/valve.fl**2
    FLP = 1 / math.sqrt(term)
    return FLP

def calculate_x_tp(valve, valve_d_inch: float, pipe_d_inch: float, Fp: float) -> float:
    if abs(pipe_d_inch - valve_d_inch) < 0.01:
        return valve.xt
    xT = valve.get_xt_at_opening(100)
    d_ratio = valve_d_inch / pipe_d_inch
    K1 = 0.5 * (1 - d_ratio**2)**2
    KB1 = 1 - d_ratio**4
    Ki = K1 + KB1
    cv_100 = valve.get_cv_at_opening(100)
    term = 1 + (xT * Ki / CONSTANTS["N5"]["inch"]) * (cv_100 / valve_d_inch**2)**2
    xTP = xT / Fp**2 * (1 / term)
    return xTP

def cv_liquid(flow: float, p1: float, p2: float, sg: float, fl: float, 
              pv: float, pc: float, visc_cst: float, d_m: float, fp: float = 1.0) -> tuple:
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        return 0, {'error': 'Invalid pressures', 'theoretical_cv': 0, 'fp': fp, 'fr': 1.0, 'reynolds': 0, 'is_choked': False, 'ff': 0, 'dp_max': 0}
    
    N1 = CONSTANTS["N1"]["m³/h, bar"]
    dp = p1 - p2
    if dp <= 0:
        return 0, {'theoretical_cv': 0, 'fp': fp, 'fr': 1.0, 'reynolds': 0, 'is_choked': False, 'ff': 0, 'dp_max': 0}
    
    ff = calculate_ff(pv, pc)
    dp_max = fl**2 * (p1 - ff * pv)
    
    if dp < dp_max:
        cv_pseudo = (flow / N1) * math.sqrt(sg / dp)
    else:
        cv_pseudo = (flow / N1) * math.sqrt(sg) / (fl * math.sqrt(p1 - ff * pv))
    
    rev = reynolds_number(flow, d_m, visc_cst)
    fr = viscosity_correction(rev)
    
    if dp < dp_max:
        theoretical_cv = (flow / N1) * math.sqrt(sg / dp)
    else:
        theoretical_cv = (flow / N1) * math.sqrt(sg) / (fl * math.sqrt(p1 - ff * pv))
    
    cv_after_fp = theoretical_cv / fp
    corrected_cv = cv_after_fp / fr
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'fr': fr,
        'reynolds': rev,
        'is_choked': (dp >= dp_max),
        'ff': ff,
        'dp_max': dp_max,
        'fl': fl
    }
    
    return corrected_cv, details

def cv_gas(flow: float, p1: float, p2: float, sg: float, t: float, k: float, 
           xt: float, z: float, fp: float = 1.0) -> tuple:
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        return 0, {'error': 'Invalid pressures', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': 0, 'is_choked': False, 'x_crit': 0}
    
    x = (p1 - p2) / p1
    if x <= 0:
        return 0, {'error': 'Negative pressure drop', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': 0, 'is_choked': False, 'x_crit': 0}
    
    fk = k / 1.4
    x_crit = fk * xt
    
    if x >= x_crit:
        y = 0.667
        x = x_crit
        is_choked = True
    else:
        y = 1 - x / (3 * fk * xt)
        is_choked = False
    
    N7 = CONSTANTS["N7"]["m³/h, bar, K (standard)"]
    term = (sg * (t + C_TO_K) * z) / x
    if term < 0:
        return 0, {'error': 'Negative value in sqrt', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': y, 'is_choked': is_choked, 'x_crit': x_crit}
    
    theoretical_cv = (flow / (N7 * fp * p1 * y)) * math.sqrt(term)
    corrected_cv = theoretical_cv
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'expansion_factor': y,
        'is_choked': is_choked,
        'x_crit': x_crit,
        'xt': xt
    }
    
    return corrected_cv, details

def cv_steam(flow: float, p1: float, p2: float, rho: float, k: float, 
             xt: float, fp: float = 1.0) -> tuple:
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        return 0, {'error': 'Invalid pressures', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': 0, 'is_choked': False, 'x_crit': 0}
    
    x = (p1 - p2) / p1
    if x <= 0:
        return 0, {'error': 'Negative pressure drop', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': 0, 'is_choked': False, 'x_crit': 0}
    
    fk = k / 1.4
    x_crit = fk * xt
    
    if x >= x_crit:
        y = 0.667
        x = x_crit
        is_choked = True
    else:
        y = 1 - x / (3 * fk * xt)
        is_choked = False
    
    N6 = CONSTANTS["N6"]["kg/h, bar, kg/m³"]
    term = x * p1 * rho
    if term <= 0:
        return 0, {'error': 'Invalid term in sqrt', 'theoretical_cv': 0, 'fp': fp, 'expansion_factor': y, 'is_choked': is_choked, 'x_crit': x_crit}
    
    theoretical_cv = flow / (N6 * y * math.sqrt(term))
    corrected_cv = theoretical_cv / fp
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'expansion_factor': y,
        'is_choked': is_choked,
        'x_crit': x_crit,
        'xt': xt
    }
    
    return corrected_cv, details

def check_cavitation(p1: float, p2: float, pv: float, fl: float, pc: float) -> tuple:
    if pc <= 0:
        return False, 0, 0, "Critical pressure not available"
    if p1 <= 0 or p2 < 0 or p1 <= p2:
        return False, 0, 0, "Invalid pressures"
    ff = calculate_ff(pv, pc)
    dp = p1 - p2
    if dp <= 0:
        return False, 0, 0, "No pressure drop"
    dp_max = fl**2 * (p1 - ff * pv)
    km = fl**2
    sigma = (p1 - pv) / dp
    if dp >= dp_max:
        return True, sigma, km, "Choked flow - cavitation likely"
    elif sigma < 1.5 * km:
        return False, sigma, km, "Severe cavitation risk"
    elif sigma < 2 * km:
        return False, sigma, km, "Moderate cavitation risk"
    elif sigma < 4 * km:
        return False, sigma, km, "Mild cavitation risk"
    return False, sigma, km, "Minimal cavitation risk"

# ========================
# PDF REPORT GENERATION
# ========================
class PDFReport(FPDF):
    def __init__(self, logo_bytes=None, logo_type=None):
        super().__init__()
        self.logo_bytes = logo_bytes
        self.logo_type = logo_type
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        if self.logo_bytes and self.logo_type:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{self.logo_type.lower()}") as tmpfile:
                    tmpfile.write(self.logo_bytes)
                    tmpfile_path = tmpfile.name
                self.image(tmpfile_path, x=10, y=8, w=30)
                os.unlink(tmpfile_path)
            except Exception as e:
                self.cell(0, 10, f"Logo error: {str(e)}", 0, 1)
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Control Valve Sizing Report', 0, 1, 'C')
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'B', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
        
    def add_table(self, headers, data):
        col_widths = [40] * len(headers)
        self.set_font('Arial', 'B', 10)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 10)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 6, str(item), 1)
            self.ln()

def generate_pdf_report(scenarios, valve, op_points, req_cvs, warnings, cavitation_info, plot_bytes=None, logo_bytes=None, logo_type=None):
    pdf = PDFReport(logo_bytes=logo_bytes, logo_type=logo_type)
    pdf.add_page()
    pdf.chapter_title('Project Information')
    pdf.cell(0, 10, f'Project: Valve Sizing Analysis', 0, 1)
    pdf.cell(0, 10, f'Generated by: Valve Sizing Software', 0, 1)
    pdf.ln(5)
    
    pdf.chapter_title('Selected Valve Details')
    valve_text = (
        f"Size: {valve.size}\" E{valve.valve_type}{valve.rating_class}\n"
        f"Type: {'Globe' if valve.valve_type == 3 else 'Axial'}\n"
        f"Rating Class: {valve.rating_class}\n"
        f"Fl (Liquid Recovery): {valve.fl:.3f}\n"
        f"Xt (Pressure Drop Ratio): {valve.xt:.3f}\n"
        f"Fd (Valve Style Modifier): {valve.fd:.2f}\n"
        f"Internal Diameter: {valve.diameter:.2f} in"
    )
    pdf.chapter_body(valve_text)
    
    pdf.chapter_title('Valve Cv Characteristics')
    cv_table_data = []
    for open_percent, cv in valve.cv_table.items():
        cv_table_data.append([f"{open_percent}%", f"{cv:.1f}"])
    pdf.add_table(['Opening %', 'Cv Value'], cv_table_data)
    
    pdf.chapter_title('Sizing Results')
    results_data = []
    for i, scenario in enumerate(scenarios):
        actual_cv = valve.get_cv_at_opening(op_points[i])
        margin = (actual_cv / req_cvs[i] - 1) * 100 if req_cvs[i] > 0 else 0
        results_data.append([
            scenario["name"],
            f"{req_cvs[i]:.1f}",
            f"{valve.size}\"",
            f"{op_points[i]:.1f}%",
            f"{actual_cv:.1f}",
            f"{margin:.1f}%",
            warnings[i] + (" " + cavitation_info[i] if cavitation_info[i] else "")
        ])
    pdf.add_table(
        ['Scenario', 'Req Cv', 'Valve Size', 'Opening %', 'Actual Cv', 'Margin %', 'Warnings'],
        results_data
    )
    
    pdf.chapter_title('Detailed Calculations')
    for i, scenario in enumerate(scenarios):
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 10, f'Scenario {i+1}: {scenario["name"]}', 0, 1)
        pdf.set_font('Arial', '', 10)
        calc_text = (
            f"Fluid Type: {scenario['fluid_type'].title()}\n"
            f"Flow Rate: {scenario['flow']} "
            f"{'m³/h' if scenario['fluid_type']=='liquid' else 'kg/h' if scenario['fluid_type']=='steam' else 'std m³/h'}\n"
            f"Inlet Pressure (P1): {scenario['p1']:.2f} bar a\n"
            f"Outlet Pressure (P2): {scenario['p2']:.2f} bar a\n"
            f"Pressure Drop (dP): {scenario['p1'] - scenario['p2']:.2f} bar\n"
            f"Temperature: {scenario['temp']}°C\n"
        )
        if scenario["fluid_type"] == "liquid":
            calc_text += (
                f"Specific Gravity: {scenario['sg']:.3f}\n"
                f"Viscosity: {scenario['visc']} cSt\n"
                f"Vapor Pressure: {scenario['pv']:.4f} bar a\n"
                f"Critical Pressure: {scenario['pc']:.2f} bar a\n"
                f"Cavitation Status: {cavitation_info[i]}\n"
            )
        elif scenario["fluid_type"] == "gas":
            calc_text += (
                f"Specific Gravity (air=1): {scenario['sg']:.3f}\n"
                f"Specific Heat Ratio (k): {scenario['k']:.3f}\n"
                f"Compressibility Factor (Z): {scenario['z']:.3f}\n"
            )
        else:
            calc_text += (
                f"Density: {scenario['rho']:.3f} kg/m³\n"
                f"Specific Heat Ratio (k): {scenario['k']:.3f}\n"
            )
        calc_text += (
            f"Pipe Diameter: {scenario['pipe_d']} in\n"
            f"Required Cv: {req_cvs[i]:.1f}\n"
            f"Operating Point: {op_points[i]:.1f}% open\n"
            f"Actual Cv at Operating Point: {actual_cv:.1f}\n"
            f"Margin: {margin:.1f}%\n"
            f"Warnings: {warnings[i]}{', ' + cavitation_info[i] if cavitation_info[i] else ''}\n\n"
        )
        pdf.multi_cell(0, 5, calc_text)
    
    if plot_bytes:
        pdf.chapter_title('Valve Cv Characteristic Curve')
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_plot:
                tmp_plot.write(plot_bytes)
                tmp_plot_path = tmp_plot.name
            pdf.image(tmp_plot_path, x=10, w=180)
            os.unlink(tmp_plot_path)
        except Exception as e:
            pdf.cell(0, 10, f"Failed to insert plot: {str(e)}", 0, 1)
    
    pdf_string = pdf.output(dest='S').encode('latin1')
    pdf_bytes = BytesIO(pdf_string)
    pdf_bytes.seek(0)
    return pdf_bytes

# ========================
# SIMULATION RESULTS
# ========================
def get_simulation_image(valve_name):
    simulation_images = {
        "0.5\" E31": "https://via.placeholder.com/1200x900.png?text=High-Res+Simulation+0.5E31",
        "1\" E31": "https://via.placeholder.com/1200x900.png?text=High-Res+Simulation+1E31",
        "1.5\" E31": "https://via.placeholder.com/1200x900.png?text=High-Res+Simulation+1.5E31",
        "2\" E31": "https://via.placeholder.com/1200x900.png?text=High-Res+Simulation+2E31",
        "3\" E32": "https://via.placeholder.com/1200x900.png?text=High-Res+Simulation+3E32",
        "4\" E32": "https://via.placeholder.com/1200x900.png?text=High-Res+Simulation+4E32",
        "6\" E32": "https://via.placeholder.com/1200x900.png?text=High-Res+Simulation+6E32",
        "8\" E32": "https://via.placeholder.com/1200x900.png?text=High-Res+Simulation+8E32",
    }
    return simulation_images.get(valve_name, "https://via.placeholder.com/1200x900.png?text=Simulation+Not+Available")

# ========================
# FLOW RATE VS PRESSURE DROP GRAPH
# ========================
def generate_flow_vs_dp_graph(scenario, valve, op_point, details, req_cv):
    # Get actual Cv at operating point
    actual_cv = valve.get_cv_at_opening(op_point)
    valve_cv_effective = actual_cv * details['fp']
    
    # Determine max pressure drop
    if scenario['fluid_type'] == "liquid":
        max_dp = details['dp_max']
    elif scenario['fluid_type'] in ["gas", "steam"]:
        max_dp = details['x_crit'] * scenario['p1']
    else:
        max_dp = scenario['p1'] - scenario['p2']
    
    # Create pressure drop range (from 1/10 max to max)
    min_dp = max_dp / 10
    dp_range = np.linspace(min_dp, max_dp, 50)
    flow_rates = []
    
    # Calculate flow rates for each dp
    for dp in dp_range:
        if scenario['fluid_type'] == "liquid":
            if dp <= details['dp_max']:
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * math.sqrt(dp / scenario['sg'])
            else:
                flow = valve_cv_effective * CONSTANTS["N1"]["m³/h, bar"] * details['fl'] * math.sqrt(
                    (scenario['p1'] - details['ff'] * scenario['pv']) / scenario['sg'])
            flow_rates.append(flow)
            
        elif scenario['fluid_type'] == "gas":
            x = dp / scenario['p1']
            x_crit = details['x_crit']
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details['xt'])
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N7"]["m³/h, bar, K (standard)"] * scenario['p1'] * Y * math.sqrt(
                x / (scenario['sg'] * (scenario['temp'] + C_TO_K) * scenario['z']))
            flow_rates.append(flow)
            
        elif scenario['fluid_type'] == "steam":
            x = dp / scenario['p1']
            x_crit = details['x_crit']
            fk = scenario['k'] / 1.4
            if x < x_crit:
                Y = 1 - x / (3 * fk * details['xt'])
            else:
                Y = 0.667
                x = x_crit
            flow = valve_cv_effective * CONSTANTS["N6"]["kg/h, bar, kg/m³"] * Y * math.sqrt(
                x * scenario['p1'] * scenario['rho'])
            flow_rates.append(flow)
    
    # Current operating point
    current_dp = scenario['p1'] - scenario['p2']
    current_flow = scenario['flow']
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dp_range, 
        y=flow_rates, 
        mode='lines',
        name='Flow Rate',
        line=dict(color='blue', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[current_dp], 
        y=[current_flow], 
        mode='markers',
        name='Operating Point',
        marker=dict(size=12, color='red')
    ))
    
    # Add max flow annotation
    if max_dp > 0:
        max_flow = flow_rates[-1]
        fig.add_annotation(
            x=max_dp,
            y=max_flow,
            text=f'Max Flow: {max_flow:.1f}',
            showarrow=True,
            arrowhead=1,
            ax=-50,
            ay=-30
        )
    
    fig.update_layout(
        title=f'Flow Rate vs Pressure Drop - {scenario["name"]}',
        xaxis_title='Pressure Drop (bar)',
        yaxis_title=f'Flow Rate ({"m³/h" if scenario["fluid_type"]=="liquid" else "std m³/h" if scenario["fluid_type"]=="gas" else "kg/h"})',
        legend_title='Legend',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    fig.update_xaxes(range=[0, max_dp * 1.1])
    fig.update_yaxes(range=[0, max(flow_rates) * 1.1])
    
    return fig

# ========================
# RECOMMENDED VALVE LOGIC
# ========================
def evaluate_valve_for_scenario(valve, scenario):
    pipe_d = scenario["pipe_d"]
    valve_d = valve.diameter
    cv_100 = valve.get_cv_at_opening(100)
    fp = calculate_piping_factor_fp(valve_d, pipe_d, cv_100)
    
    if scenario["fluid_type"] in ["gas", "steam"]:
        if abs(pipe_d - valve_d) > 0.01:
            xt = calculate_x_tp(valve, valve_d, pipe_d, fp)
        else:
            xt = valve.get_xt_at_opening(100)
    else:
        xt = valve.get_xt_at_opening(100)
    
    # Calculate velocity
    valve_diameter_m = valve.diameter * 0.0254
    valve_area = math.pi * (valve_diameter_m/2)**2
    
    if scenario['fluid_type'] == "liquid":
        flow_m3s = scenario['flow'] / 3600
        velocity = flow_m3s / valve_area
    elif scenario['fluid_type'] == "gas":
        p_std = 1.01325
        T_std = 288.15
        z_std = 1.0
        T1 = scenario['temp'] + C_TO_K
        actual_flow_m3h = scenario['flow'] * (p_std / scenario['p1']) * (T1 / T_std) * (scenario['z'] / z_std)
        actual_flow_m3s = actual_flow_m3h / 3600
        velocity = actual_flow_m3s / valve_area
    else:
        volume_flow_m3h = scenario['flow'] / scenario['rho']
        volume_flow_m3s = volume_flow_m3h / 3600
        velocity = volume_flow_m3s / valve_area
    
    # Calculate required Cv
    if scenario["fluid_type"] == "liquid":
        if scenario.get('fluid_library') in FLUID_LIBRARY:
            fluid_data = FLUID_LIBRARY[scenario['fluid_library']]
            scenario["visc"] = fluid_data["visc_func"](scenario["temp"], scenario["p1"])
            scenario["pv"] = fluid_data["pv_func"](scenario["temp"], scenario["p1"])
            if "pc_func" in fluid_data:
                scenario["pc"] = fluid_data["pc_func"]()
        
        cv_req, details = cv_liquid(
            flow=scenario["flow"],
            p1=scenario["p1"],
            p2=scenario["p2"],
            sg=scenario["sg"],
            fl=valve.fl,
            pv=scenario["pv"],
            pc=scenario["pc"],
            visc_cst=scenario["visc"],
            d_m=valve.diameter * 0.0254,
            fp=fp
        )
        
        if scenario["pc"] > 0:
            choked, sigma, km, cav_msg = check_cavitation(
                scenario["p1"], scenario["p2"], scenario["pv"], valve.fl, scenario["pc"]
            )
            details['is_choked'] = choked
            details['cavitation_severity'] = cav_msg
        else:
            details['cavitation_severity'] = "Critical pressure not available"
        
    elif scenario["fluid_type"] == "gas":
        if scenario.get('fluid_library') in FLUID_LIBRARY:
            fluid_data = FLUID_LIBRARY[scenario['fluid_library']]
            scenario["k"] = fluid_data["k_func"](scenario["temp"], scenario["p1"])
        
        cv_req, details = cv_gas(
            flow=scenario["flow"],
            p1=scenario["p1"],
            p2=scenario["p2"],
            sg=scenario["sg"],
            t=scenario["temp"],
            k=scenario["k"],
            xt=xt,
            z=scenario["z"],
            fp=fp
        )
        details['cavitation_severity'] = "N/A"
        
    else:
        if scenario.get('fluid_library') in FLUID_LIBRARY:
            fluid_data = FLUID_LIBRARY[scenario['fluid_library']]
            scenario["rho"] = fluid_data["rho_func"](scenario["temp"], scenario["p1"])
            scenario["k"] = fluid_data["k_func"](scenario["temp"], scenario["p1"])
        
        cv_req, details = cv_steam(
            flow=scenario["flow"],
            p1=scenario["p1"],
            p2=scenario["p2"],
            rho=scenario["rho"],
            k=scenario["k"],
            xt=xt,
            fp=fp
        )
        details['cavitation_severity'] = "N/A"
    
    if 'error' in details:
        return {
            "op_point": 0,
            "req_cv": 0,
            "theoretical_cv": 0,
            "warning": details['error'],
            "cavitation_info": "N/A",
            "status": "red",
            "margin": 0,
            "details": details,
            "velocity": velocity
        }
    
    open_percent = 10
    while open_percent <= 100:
        cv_valve = valve.get_cv_at_opening(open_percent)
        if cv_valve >= cv_req:
            break
        open_percent += 1
    
    warn = ""
    if open_percent < 20:
        warn = "Low opening (<20%)"
    elif open_percent > 80:
        warn = "High opening (>80%)"
    
    status = "green"
    if details.get('is_choked', False):
        status = "red"
    elif "Severe" in details.get('cavitation_severity', ""):
        status = "orange"
    elif "Moderate" in details.get('cavitation_severity', ""):
        status = "yellow"
    elif open_percent < 20 or open_percent > 80:
        status = "yellow"
    
    return {
        "op_point": open_percent,
        "req_cv": cv_req,
        "theoretical_cv": details['theoretical_cv'],
        "warning": warn,
        "cavitation_info": details.get('cavitation_severity', "N/A"),
        "status": status,
        "margin": (cv_valve / cv_req - 1) * 100 if cv_req > 0 else 0,
        "details": details,
        "velocity": velocity
    }

def find_recommended_valve(scenarios):
    best_valve = None
    best_score = float('-inf')
    all_valve_results = []
    
    for valve in VALVE_DATABASE:
        valve_results = []
        valve_score = 0
        is_suitable = True
        
        for scenario in scenarios:
            result = evaluate_valve_for_scenario(valve, scenario)
            valve_results.append(result)
            
            if result["status"] == "red":
                valve_score -= 100
                is_suitable = False
            elif result["status"] == "orange":
                valve_score -= 50
            elif result["status"] == "yellow":
                valve_score -= 10
            else:
                valve_score += 20
                
            valve_score += max(0, 10 - abs(result["op_point"] - 50)/5)
        
        valve_display_name = get_valve_display_name(valve)
        all_valve_results.append({
            "valve": valve,
            "results": valve_results,
            "score": valve_score,
            "display_name": valve_display_name
        })
        
        if is_suitable and valve_score > best_score:
            best_valve = {
                "valve": valve,
                "results": valve_results,
                "score": valve_score,
                "display_name": valve_display_name
            }
            best_score = valve_score
    
    if best_valve is None and all_valve_results:
        all_valve_results.sort(key=lambda x: x["score"], reverse=True)
        best_valve = all_valve_results[0]
    
    return best_valve, all_valve_results

# ========================
# STREAMLIT APPLICATION
# ========================
def get_valve_display_name(valve):
    rating_code_map = {
        150: 1,
        300: 2,
        600: 3,
        900: 4,
        1500: 5,
        2500: 6
    }
    rating_code = rating_code_map.get(valve.rating_class, valve.rating_class)
    return f"{valve.size}\" E{valve.valve_type}{rating_code}"

def create_valve_dropdown():
    valves = sorted(VALVE_DATABASE, key=lambda v: (v.size, v.rating_class, v.valve_type))
    valve_options = {get_valve_display_name(v): v for v in valves}
    return valve_options

def create_fluid_dropdown():
    return ["Select Fluid Library..."] + list(FLUID_LIBRARY.keys())

def scenario_input_form(scenario_num, scenario_data=None):
    default_values = {
        "sg": 1.0,
        "visc": 1.0,
        "pv": 0.023,
        "pc": 220.55,
        "k": 1.4,
        "z": 1.0,
        "rho": 1.0,
        "fluid_type": "liquid"
    }
    
    if scenario_data is None:
        scenario_data = {
            "name": f"Scenario {scenario_num}",
            "fluid_type": "liquid",
            "flow": 10.0 if scenario_num == 1 else 50.0,
            "p1": 10.0,
            "p2": 6.0,
            "temp": 20.0,
            "pipe_d": 2.0
        }
        scenario_data = {**default_values, **scenario_data}
    else:
        for key, default in default_values.items():
            if key not in scenario_data:
                scenario_data[key] = default
    
    st.subheader(f"Scenario {scenario_num}")
    scenario_name = st.text_input("Scenario Name", value=scenario_data["name"], key=f"name_{scenario_num}")
    
    col1, col2 = st.columns(2)
    with col1:
        fluid_library = st.selectbox(
            "Fluid Library", 
            create_fluid_dropdown(), 
            key=f"fluid_library_{scenario_num}"
        )
    
    with col2:
        if fluid_library != "Select Fluid Library...":
            fluid_data = FLUID_LIBRARY[fluid_library]
            fluid_type = fluid_data["type"]
            # FIX: Added unique key to text_input
            st.text_input("Fluid Type", value=fluid_type.capitalize(), disabled=True, key=f"fluid_type_text_{scenario_num}")
        else:
            try:
                index_val = ["Liquid", "Gas", "Steam"].index(scenario_data["fluid_type"].capitalize())
            except (ValueError, AttributeError):
                index_val = 0
            fluid_type = st.selectbox(
                "Fluid Type", 
                ["Liquid", "Gas", "Steam"], 
                index=index_val,
                key=f"fluid_type_{scenario_num}"
            ).lower()
    
    col1, col2 = st.columns(2)
    with col1:
        flow_label = "Flow Rate (m³/h)" if fluid_type == "liquid" else "Flow Rate (std m³/h)" if fluid_type == "gas" else "Flow Rate (kg/h)"
        flow_value = st.number_input(
            flow_label, 
            min_value=0.0, 
            max_value=100000.0, 
            value=scenario_data["flow"], 
            step=0.1,
            key=f"flow_{scenario_num}"
        )
        p1 = st.number_input(
            "Inlet Pressure (bar a)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=scenario_data["p1"], 
            step=0.1,
            key=f"p1_{scenario_num}"
        )
        p2 = st.number_input(
            "Outlet Pressure (bar a)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=scenario_data["p2"], 
            step=0.1,
            key=f"p2_{scenario_num}"
        )
        temp = st.number_input(
            "Temperature (°C)", 
            min_value=-200.0, 
            max_value=1000.0, 
            value=scenario_data["temp"], 
            step=1.0,
            key=f"temp_{scenario_num}"
        )
    
    with col2:
        if fluid_library != "Select Fluid Library...":
            fluid_data = FLUID_LIBRARY[fluid_library]
            scenario_data["fluid_type"] = fluid_data["type"]
            if fluid_data.get("visc_func") and fluid_data["type"] == "liquid":
                scenario_data["visc"] = fluid_data["visc_func"](temp, p1)
            if fluid_data.get("k_func") and fluid_data["type"] in ["gas", "steam"]:
                scenario_data["k"] = fluid_data["k_func"](temp, p1)
            if fluid_data.get("pv_func") and fluid_data["type"] == "liquid":
                scenario_data["pv"] = fluid_data["pv_func"](temp, p1)
            if fluid_data.get("pc_func") and fluid_data["type"] == "liquid":
                scenario_data["pc"] = fluid_data["pc_func"]()
            if fluid_data.get("rho_func") and fluid_data["type"] == "steam":
                scenario_data["rho"] = fluid_data["rho_func"](temp, p1)
            if fluid_data.get("sg") is not None:
                scenario_data["sg"] = fluid_data["sg"]
        
        if fluid_type in ["liquid", "gas"]:
            sg = st.number_input(
                "Specific Gravity (water=1)" if fluid_type == "liquid" else "Specific Gravity (air=1)",
                min_value=0.01, 
                max_value=10.0, 
                value=scenario_data["sg"], 
                step=0.01,
                key=f"sg_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type == "liquid":
            visc = st.number_input(
                "Viscosity (cSt)", 
                min_value=0.01, 
                max_value=10000.0, 
                value=scenario_data["visc"], 
                step=0.1,
                key=f"visc_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
            pv = st.number_input(
                "Vapor Pressure (bar a)", 
                min_value=0.0, 
                max_value=100.0, 
                value=scenario_data["pv"], 
                step=0.0001,
                format="%.4f",
                key=f"pv_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
            pc = st.number_input(
                "Critical Pressure (bar a)", 
                min_value=0.0, 
                max_value=1000.0, 
                value=scenario_data["pc"], 
                step=0.1,
                key=f"pc_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type in ["gas", "steam"]:
            k = st.number_input(
                "Specific Heat Ratio (k=Cp/Cv)", 
                min_value=1.0, 
                max_value=2.0, 
                value=scenario_data["k"], 
                step=0.01,
                key=f"k_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type == "gas":
            z = st.number_input(
                "Compressibility Factor (Z)", 
                min_value=0.1, 
                max_value=2.0, 
                value=scenario_data["z"], 
                step=0.01,
                key=f"z_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
        
        if fluid_type == "steam":
            rho = st.number_input(
                "Density (kg/m³)", 
                min_value=0.01, 
                max_value=2000.0, 
                value=scenario_data["rho"], 
                step=0.1,
                key=f"rho_{scenario_num}",
                disabled=(fluid_library != "Select Fluid Library...")
            )
        
        pipe_d = st.number_input(
            "Pipe Diameter (inch)", 
            min_value=0.1, 
            max_value=100.0, 
            value=scenario_data["pipe_d"], 
            step=0.1,
            key=f"pipe_d_{scenario_num}"
        )
    
    return {
        "name": scenario_name,
        "fluid_type": fluid_type,
        "flow": flow_value,
        "p1": p1,
        "p2": p2,
        "temp": temp,
        "sg": sg if fluid_type in ["liquid", "gas"] else scenario_data["sg"],
        "visc": visc if fluid_type == "liquid" else scenario_data["visc"],
        "pv": pv if fluid_type == "liquid" else scenario_data["pv"],
        "pc": pc if fluid_type == "liquid" else scenario_data["pc"],
        "k": k if fluid_type in ["gas", "steam"] else scenario_data["k"],
        "z": z if fluid_type == "gas" else scenario_data["z"],
        "rho": rho if fluid_type == "steam" else scenario_data["rho"],
        "pipe_d": pipe_d,
        "fluid_library": fluid_library
    }

def plot_cv_curve(valve, op_points, req_cvs, theoretical_cvs, scenario_names):
    openings = list(range(0, 101, 5))
    cv_values = [valve.get_cv_at_opening(op) for op in openings]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=openings, 
        y=cv_values, 
        mode='lines',
        name='Valve Cv',
        line=dict(color='blue', width=3)))
    
    for i, (op, req_cv, theoretical_cv) in enumerate(zip(op_points, req_cvs, theoretical_cvs)):
        actual_cv = valve.get_cv_at_opening(op)
        fig.add_trace(go.Scatter(
            x=[op], 
            y=[actual_cv], 
            mode='markers+text',
            name=f'Scenario {i+1} Operating Point',
            marker=dict(size=12, color='red'),
            text=[f'S{i+1}'],
            textposition="top center"
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[req_cv, req_cv],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            name=f'Corrected Cv S{i+1}',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[theoretical_cv, theoretical_cv],
            mode='lines',
            line=dict(color='green', dash='dot', width=1),
            name=f'Theoretical Cv S{i+1}',
            showlegend=False
        ))
    
    for i, (req_cv, theoretical_cv) in enumerate(zip(req_cvs, theoretical_cvs)):
        fig.add_annotation(
            x=100,
            y=req_cv,
            text=f'Corrected S{i+1}: {req_cv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=10,
            align='right',
            font=dict(color='red')
        )
        fig.add_annotation(
            x=100,
            y=theoretical_cv,
            text=f'Theoretical S{i+1}: {theoretical_cv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=-10,
            align='right',
            font=dict(color='green')
        )
    
    fig.update_layout(
        title=f'{valve.size}" Valve Cv Characteristic',
        xaxis_title='Opening Percentage (%)',
        yaxis_title='Cv Value',
        legend_title='Legend',
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, max(cv_values) * 1.1])
    return fig

def valve_3d_viewer(valve_name, model_url):
    html_code = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <model-viewer src="{model_url}"
                  alt="{valve_name}"
                  auto-rotate
                  camera-controls
                  style="width: 100%; height: 500px;">
    </model-viewer>
    """
    components.html(html_code, height=520)

def main():
    st.set_page_config(
        page_title="Control Valve Sizing",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        html {
            font-size: 18px;
        }
        .stApp {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 15px 25px;
            border-radius: 10px 10px 0 0;
            font-size: 18px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        .stButton button {
            width: 100%;
            font-weight: bold;
            font-size: 18px;
        }
        .result-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 18px;
        }
        .warning-card {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
        }
        .success-card {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
        }
        .danger-card {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
        }
        .cavitation-card {
            background-color: #ffe8cc;
            border-left: 5px solid #fd7e14;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            padding: 10px 0;
        }
        .simulation-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            width: 80%;
            max-width: 900px;
            max-height: 80vh;
            overflow: auto;
        }
        .modal-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 999;
        }
        .stMetric {
            font-size: 20px !important;
        }
        .stNumberInput, .stTextInput, .stSelectbox {
            font-size: 18px;
        }
        .stMarkdown {
            font-size: 18px;
        }
        .valve-table {
            width: 100%;
            border-collapse: collapse;
        }
        .valve-table th {
            background-color: #2c3e50;
            color: white;
            padding: 10px;
            text-align: center;
        }
        .valve-table td {
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .status-green {
            background-color: #d4edda;
        }
        .status-yellow {
            background-color: #fff3cd;
        }
        .status-orange {
            background-color: #ffe8cc;
        }
        .status-red {
            background-color: #f8d7da;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'valve' not in st.session_state:
        st.session_state.valve = None
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = None
    if 'logo_bytes' not in st.session_state:
        st.session_state.logo_bytes = None
    if 'logo_type' not in st.session_state:
        st.session_state.logo_type = None
    if 'show_simulation' not in st.session_state:
        st.session_state.show_simulation = False
    if 'show_3d_viewer' not in st.session_state:
        st.session_state.show_3d_viewer = False
    if 'recommended_valve' not in st.session_state:
        st.session_state.recommended_valve = None
    if 'all_valve_results' not in st.session_state:
        st.session_state.all_valve_results = None
    
    col1, col2 = st.columns([1, 4])
    with col1:
        default_logo = "logo.png"
        if os.path.exists(default_logo):
            st.image(default_logo, width=100)
        else:
            st.image("https://via.placeholder.com/100x100?text=LOGO", width=100)
    with col2:
        st.title("Control Valve Sizing Program")
        st.markdown("**ISA/IEC Standards Compliant Valve Sizing with Enhanced Visualization**")
    
    with st.sidebar:
        st.header("VASTAŞ Logo")
        logo_upload = st.file_uploader("Upload VASTAŞ logo", type=["png", "jpg", "jpeg"], key="logo_uploader")
        if logo_upload is not None:
            st.session_state.logo_bytes = logo_upload.getvalue()
            st.session_state.logo_type = "PNG"
            st.success("Logo uploaded successfully!")
        if st.session_state.logo_bytes:
            st.image(Image.open(BytesIO(st.session_state.logo_bytes)), use_container_width=True)
        elif os.path.exists("logo.png"):
            st.image(Image.open("logo.png"), use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x100?text=VASTAŞ+Logo", use_container_width=True)
        
        st.header("Valve Selection")
        valve_options = create_valve_dropdown()
        selected_valve_name = st.selectbox("Select Valve", list(valve_options.keys()))
        selected_valve = valve_options[selected_valve_name]
        
        st.header("Actions")
        calculate_btn = st.button("Calculate Opening", type="primary", use_container_width=True)
        export_btn = st.button("Export PDF Report", use_container_width=True)
        view_3d_btn = st.button("View 3D Model", use_container_width=True)
        show_simulation_btn = st.button("Show Simulation Results", use_container_width=True)
        
        st.header("Valve Details")
        st.markdown(f"**Size:** {selected_valve.size}\"")
        st.markdown(f"**Type:** {'Globe' if selected_valve.valve_type == 3 else 'Axial'}")
        st.markdown(f"**Rating Class:** {selected_valve.rating_class}")
        st.markdown(f"**Fl (Liquid Recovery):** {selected_valve.fl:.3f}")
        st.markdown(f"**Xt (Pressure Drop Ratio):** {selected_valve.xt:.3f}")
        st.markdown(f"**Fd (Style Modifier):** {selected_valve.fd:.2f}")
        st.markdown(f"**Internal Diameter:** {selected_valve.diameter:.2f} in")
        
        st.subheader("Cv Characteristics")
        cv_data = {"Opening %": list(selected_valve.cv_table.keys()), "Cv": list(selected_valve.cv_table.values())}
        cv_df = pd.DataFrame(cv_data)
        st.dataframe(cv_df, hide_index=True, height=300)
    
    if view_3d_btn:
        st.session_state.show_3d_viewer = True
        st.session_state.show_simulation = False
    if show_simulation_btn:
        st.session_state.show_simulation = True
        st.session_state.show_3d_viewer = False
    
    tab1, tab2, tab3, tab_results = st.tabs(["Scenario 1", "Scenario 2", "Scenario 3", "Results"])
    
    with tab1:
        scenario1 = scenario_input_form(1)
    with tab2:
        scenario2 = scenario_input_form(2)
    with tab3:
        scenario3 = scenario_input_form(3)
    
    scenarios = []
    if scenario1["flow"] > 0:
        scenarios.append(scenario1)
    if scenario2["flow"] > 0:
        scenarios.append(scenario2)
    if scenario3["flow"] > 0:
        scenarios.append(scenario3)
    st.session_state.scenarios = scenarios
    
    if calculate_btn:
        if not scenarios:
            st.error("Please define at least one scenario with flow > 0.")
            st.stop()
        selected_valve_results = []
        for scenario in scenarios:
            result = evaluate_valve_for_scenario(selected_valve, scenario)
            selected_valve_results.append(result)
        recommended_valve, all_valve_results = find_recommended_valve(scenarios)
        st.session_state.results = {
            "selected_valve": selected_valve,
            "selected_valve_results": selected_valve_results,
            "recommended_valve": recommended_valve,
            "all_valve_results": all_valve_results
        }
    
    with tab_results:
        if st.session_state.results:
            results = st.session_state.results
            selected_valve = results["selected_valve"]
            selected_valve_results = results["selected_valve_results"]
            recommended_valve = results["recommended_valve"]
            
            if recommended_valve:
                st.subheader("Recommended Valve")
                st.markdown(f"**{recommended_valve['display_name']}** - Score: {recommended_valve['score']:.1f}")
                
                # Show each scenario result for recommended valve in card layout
                for i, scenario in enumerate(scenarios):
                    result = recommended_valve["results"][i]
                    actual_cv = recommended_valve["valve"].get_cv_at_opening(result["op_point"])
                    
                    # Determine status class
                    status_class = ""
                    if result["status"] == "green":
                        status_class = "success-card"
                    elif result["status"] == "yellow":
                        status_class = "warning-card"
                    elif result["status"] == "orange":
                        status_class = "cavitation-card"
                    elif result["status"] == "red":
                        status_class = "danger-card"
                    
                    # Combine warnings
                    warn_msgs = []
                    if result["warning"]:
                        warn_msgs.append(result["warning"])
                    if result["cavitation_info"]:
                        warn_msgs.append(result["cavitation_info"])
                    warn_text = ", ".join(warn_msgs)
                    
                    with st.container():
                        st.markdown(f"<div class='result-card {status_class}'>", unsafe_allow_html=True)
                        cols = st.columns([1.8, 1, 1, 1, 1, 1, 1, 1.5])
                        cols[0].markdown(f"**{scenario['name']}**")
                        cols[1].metric("Req Cv", f"{result['req_cv']:.1f}")
                        cols[2].metric("Theo Cv", f"{result['theoretical_cv']:.2f}")
                        cols[3].metric("Valve Cv", f"{actual_cv:.1f}")
                        cols[4].metric("Valve Size", f"{recommended_valve['valve'].size}\"")
                        cols[5].metric("Opening", f"{result['op_point']:.1f}%")
                        cols[6].metric("Margin", f"{result['margin']:.1f}%", 
                                      delta_color="inverse" if result['margin'] < 0 else "normal")
                        cols[7].markdown(f"**{warn_text}**")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with st.expander("Why this valve is recommended"):
                    st.markdown("""
                    - **Green status**: Optimal performance (good operating range, no cavitation)
                    - **Yellow status**: Acceptable but suboptimal (moderate cavitation or bad opening)
                    - **Orange status**: Severe cavitation risk
                    - **Red status**: Choked flow (unacceptable)
                    """)
                    st.markdown(f"""
                    **Selection criteria**:
                    - Highest overall score considering operating point, cavitation risk, and flow conditions
                    - Valve size: {recommended_valve['valve'].size}\"
                    - Valve type: {'Globe' if recommended_valve['valve'].valve_type == 3 else 'Axial'}
                    """)
            else:
                st.warning("No suitable valve found for all scenarios. Consider modifying your scenarios.")
            
            st.subheader(f"Selected Valve: {get_valve_display_name(selected_valve)} Cv Characteristic")
            fig = plot_cv_curve(
                selected_valve, 
                [r["op_point"] for r in selected_valve_results],
                [r["req_cv"] for r in selected_valve_results],
                [r["theoretical_cv"] for r in selected_valve_results],
                [s["name"] for s in scenarios]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Selected Valve Performance")
            for i, scenario in enumerate(scenarios):
                result = selected_valve_results[i]
                actual_cv = selected_valve.get_cv_at_opening(result["op_point"])
                status = "success-card" if result["status"] == "green" else "warning-card"
                if result["status"] == "yellow":
                    status = "warning-card"
                elif result["status"] == "orange":
                    status = "cavitation-card"
                elif result["status"] == "red":
                    status = "danger-card"
                warn_msgs = []
                if result["warning"]:
                    warn_msgs.append(result["warning"])
                if result["cavitation_info"]:
                    warn_msgs.append(result["cavitation_info"])
                warn_text = ", ".join(warn_msgs)
                
                with st.container():
                    st.markdown(f"<div class='result-card {status}'>", unsafe_allow_html=True)
                    cols = st.columns([1.8, 1, 1, 1, 1, 1, 1, 1.5])
                    cols[0].markdown(f"**{scenario['name']}**")
                    cols[1].metric("Req Cv", f"{result['req_cv']:.1f}")
                    cols[2].metric("Theo Cv", f"{result['theoretical_cv']:.2f}")
                    cols[3].metric("Valve Cv", f"{actual_cv:.1f}")
                    cols[4].metric("Valve Size", f"{selected_valve.size}\"")
                    cols[5].metric("Opening", f"{result['op_point']:.1f}%")
                    cols[6].metric("Margin", f"{result['margin']:.1f}%", 
                                  delta_color="inverse" if result['margin'] < 0 else "normal")
                    cols[7].markdown(f"**{warn_text}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    with st.expander(f"Detailed Calculations for {scenario['name']}"):
                        st.subheader("Calculation Parameters")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Fl (Liquid Recovery):** {selected_valve.fl:.3f}")
                            st.markdown(f"**Fd (Valve Style Modifier):** {selected_valve.fd:.2f}")
                            st.markdown(f"**Fp (Piping Factor):** {result['details'].get('fp', 1.0):.4f}")
                            
                        with col2:
                            if scenario["fluid_type"] == "liquid":
                                st.markdown(f"**FF (Critical Pressure Ratio):** {result['details'].get('ff', 0.96):.4f}")
                                st.markdown(f"**Fr (Viscosity Correction):** {result['details'].get('fr', 1.0):.4f}")
                                st.markdown(f"**Reynolds Number:** {result['details'].get('reynolds', 0):.0f}")
                        
                        st.markdown(f"**Max Pressure Drop (ΔPmax):** {result['details'].get('dp_max', 0):.2f} bar")
                        st.markdown(f"**Average Velocity in Valve:** {result.get('velocity', 0):.2f} m/s")
                        
                        st.subheader("Flow Rate vs Pressure Drop")
                        flow_fig = generate_flow_vs_dp_graph(
                            scenario,
                            selected_valve,
                            result["op_point"],
                            result["details"],
                            result["req_cv"]
                        )
                        st.plotly_chart(flow_fig, use_container_width=True)
            
            st.subheader("All Valves Evaluation")
            st.markdown("""
            **Status colors**:
            - <span style="background-color:#d4edda; padding:2px 5px;">Green</span>: Optimal
            - <span style="background-color:#fff3cd; padding:2px 5px;">Yellow</span>: Warning (moderate issue)
            - <span style="background-color:#ffe8cc; padding:2px 5px;">Orange</span>: Severe cavitation
            - <span style="background-color:#f8d7da; padding:2px 5px;">Red</span>: Choked flow (unacceptable)
            """, unsafe_allow_html=True)
            all_valves_table_html = """
            <table class="valve-table">
                <thead>
                    <tr>
                        <th>Valve</th>
            """
            for i, scenario in enumerate(scenarios):
                all_valves_table_html += f'<th>{scenario["name"]} Status</th>'
            all_valves_table_html += """
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
            """
            all_valve_results = sorted(
                st.session_state.results["all_valve_results"], 
                key=lambda x: x["score"], 
                reverse=True
            )
            for valve_result in all_valve_results:
                all_valves_table_html += f'<tr><td>{valve_result["display_name"]}</td>'
                for result in valve_result["results"]:
                    status_class = ""
                    if result["status"] == "green":
                        status_class = "status-green"
                    elif result["status"] == "yellow":
                        status_class = "status-yellow"
                    elif result["status"] == "orange":
                        status_class = "status-orange"
                    elif result["status"] == "red":
                        status_class = "status-red"
                    all_valves_table_html += f'<td class="{status_class}">{result["status"]}</td>'
                all_valves_table_html += f'<td>{valve_result["score"]:.1f}</td></tr>'
            all_valves_table_html += "</tbody></table>"
            st.markdown(all_valves_table_html, unsafe_allow_html=True)
            
            st.subheader("Detailed Results")
            for i, scenario in enumerate(scenarios):
                with st.expander(f"Scenario {i+1}: {scenario['name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Process Conditions**")
                        st.markdown(f"- Fluid Type: {scenario['fluid_type'].title()}")
                        st.markdown(f"- Flow Rate: {scenario['flow']} "
                                    f"{'m³/h' if scenario['fluid_type']=='liquid' else 'kg/h' if scenario['fluid_type']=='steam' else 'std m³/h'}")
                        st.markdown(f"- Inlet Pressure (P1): {scenario['p1']:.2f} bar a")
                        st.markdown(f"- Outlet Pressure (P2): {scenario['p2']:.2f} bar a")
                        st.markdown(f"- Pressure Drop (dP): {scenario['p1'] - scenario['p2']:.2f} bar")
                        st.markdown(f"- Temperature: {scenario['temp']}°C")
                    with col2:
                        st.markdown("**Fluid Properties**")
                        if scenario["fluid_type"] == "liquid":
                            st.markdown(f"- Specific Gravity: {scenario['sg']:.3f}")
                            st.markdown(f"- Viscosity: {scenario['visc']} cSt")
                            st.markdown(f"- Vapor Pressure: {scenario['pv']:.4f} bar a")
                            st.markdown(f"- Critical Pressure: {scenario['pc']:.2f} bar a")
                        elif scenario["fluid_type"] == "gas":
                            st.markdown(f"- Specific Gravity (air=1): {scenario['sg']:.3f}")
                            st.markdown(f"- Specific Heat Ratio (k): {scenario['k']:.3f}")
                            st.markdown(f"- Compressibility Factor (Z): {scenario['z']:.3f}")
                        else:
                            st.markdown(f"- Density: {scenario['rho']:.3f} kg/m³")
                            st.markdown(f"- Specific Heat Ratio (k): {scenario['k']:.3f}")
                        st.markdown(f"- Pipe Diameter: {scenario['pipe_d']} in")
                    st.markdown("**Sizing Results**")
                    st.markdown(f"- Theoretical Cv: {selected_valve_results[i]['theoretical_cv']:.1f}")
                    st.markdown(f"- Corrected Cv: {selected_valve_results[i]['req_cv']:.1f}")
                    st.markdown(f"- Operating Point: {selected_valve_results[i]['op_point']:.1f}% open")
                    st.markdown(f"- Actual Cv at Operating Point: {selected_valve.get_cv_at_opening(selected_valve_results[i]['op_point']):.1f}")
                    st.markdown(f"- Margin: {selected_valve_results[i]['margin']:.1f}%")
                    status_msg = "Optimal" if selected_valve_results[i]["status"] == "green" else "Warning" if selected_valve_results[i]["status"] == "yellow" else "Severe cavitation" if selected_valve_results[i]["status"] == "orange" else "Choked flow"
                    st.markdown(f"- Status: {status_msg}")
                    if selected_valve_results[i]["warning"] or selected_valve_results[i]["cavitation_info"] != "N/A":
                        st.warning(f"⚠️ {selected_valve_results[i]['warning']}, {selected_valve_results[i]['cavitation_info']}")
        else:
            st.info("Click 'Calculate Opening' in the sidebar to see results")
        
        if st.session_state.show_3d_viewer:
            st.subheader("3D Valve Model")
            model_url = VALVE_MODELS.get(selected_valve_name, "https://example.com/models/default.glb")
            valve_3d_viewer(selected_valve_name, model_url)
        if st.session_state.show_simulation:
            st.subheader("Simulation Results")
            image_url = get_simulation_image(selected_valve_name)
            st.image(image_url, caption=f"Simulation Results for {selected_valve_name}", use_column_width=True)
    
    if export_btn:
        if not st.session_state.results:
            st.error("Please calculate results before exporting.")
            st.stop()
        try:
            fig = plot_cv_curve(
                st.session_state.results["selected_valve"], 
                [r["op_point"] for r in st.session_state.results["selected_valve_results"]],
                [r["req_cv"] for r in st.session_state.results["selected_valve_results"]],
                [r["theoretical_cv"] for r in st.session_state.results["selected_valve_results"]],
                [s["name"] for s in st.session_state.scenarios]
            )
            plot_bytes = fig.to_image(format="png")
            pdf_bytes = generate_pdf_report(
                st.session_state.scenarios,
                st.session_state.results["selected_valve"],
                [r["op_point"] for r in st.session_state.results["selected_valve_results"]],
                [r["req_cv"] for r in st.session_state.results["selected_valve_results"]],
                [r["warning"] for r in st.session_state.results["selected_valve_results"]],
                [r["cavitation_info"] for r in st.session_state.results["selected_valve_results"]],
                plot_bytes,
                st.session_state.logo_bytes,
                st.session_state.logo_type
            )
            st.sidebar.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"valve_sizing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            st.sidebar.success("PDF report generated successfully!")
        except Exception as e:
            st.sidebar.error(f"PDF generation failed: {str(e)}")
            st.sidebar.text(traceback.format_exc())

if __name__ == "__main__":
    main()
