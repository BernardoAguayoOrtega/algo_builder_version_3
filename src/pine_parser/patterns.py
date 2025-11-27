"""
Pattern definitions for Pine Script parsing.

This module defines the mapping between Pine Script input declarations
and our StrategyConfig fields.

Each pattern specifies:
- pine_names: Variable names to look for in Pine Script
- pine_labels: Input labels/titles to look for
- input_type: Expected input type (bool, int, float, string)
- config_field: Corresponding field in StrategyConfig
- default: Default value if not found
"""

from typing import Dict, List, Any

# =============================================================================
# INPUT PATTERNS
# Maps Pine Script inputs to StrategyConfig fields
# =============================================================================

INPUT_PATTERNS: Dict[str, Dict[str, Any]] = {
    # ===== Configuracion Basica =====
    "entry_pips": {
        "pine_names": ["entry_pip", "pips_entrada", "pipsEntrada"],
        "pine_labels": ["Pips para entrada", "Pips entrada"],
        "input_type": "int",
        "config_field": "entry_pips",
        "default": 1,
    },
    "sl_pips": {
        "pine_names": ["sl_pip", "pips_sl", "pipsSL"],
        "pine_labels": ["Pips para Stop Loss", "Pips SL"],
        "input_type": "int",
        "config_field": "sl_pips",
        "default": 1,
    },

    # ===== Sentido (Direction) =====
    "trade_longs": {
        "pine_names": ["operar_largos", "usarLargos", "tradeLongs"],
        "pine_labels": ["Largos", "Operar Largos", "Trade Longs"],
        "input_type": "bool",
        "config_field": "trade_longs",
        "default": True,
    },
    "trade_shorts": {
        "pine_names": ["operar_cortos", "usarCortos", "tradeShorts"],
        "pine_labels": ["Cortos", "Operar Cortos", "Trade Shorts"],
        "input_type": "bool",
        "config_field": "trade_shorts",
        "default": True,
    },

    # ===== Entradas (Entry Patterns) =====
    "use_sacudida": {
        "pine_names": ["usar_patron_sacudida", "usarSacudida", "sacudida"],
        "pine_labels": ["Sacudida", "Usar Sacudida"],
        "input_type": "bool",
        "config_field": "use_sacudida",
        "default": True,
    },
    "use_engulfing": {
        "pine_names": ["usar_patron_envolvente", "usarEnvolvente", "envolvente"],
        "pine_labels": ["Envolvente", "Usar Envolvente", "Engulfing"],
        "input_type": "bool",
        "config_field": "use_engulfing",
        "default": True,
    },
    "use_climatic_volume": {
        "pine_names": ["usar_patron_vol_climatico", "usarVolClimatico", "volClimatico"],
        "pine_labels": ["Volumen climático", "Volumen Climatico", "Climatic Volume"],
        "input_type": "bool",
        "config_field": "use_climatic_volume",
        "default": False,
    },

    # ===== Salidas (Exits) =====
    "use_sl": {
        "pine_names": ["usar_sl_original", "usarSL", "useSL"],
        "pine_labels": ["SL Original", "Stop Loss", "Usar SL"],
        "input_type": "bool",
        "config_field": "use_sl",
        "default": True,
    },
    "use_tp_ratio": {
        "pine_names": ["usar_salida_tp_ratio", "usarTP", "useTP"],
        "pine_labels": ["TP por Ratio", "Take Profit", "Usar TP"],
        "input_type": "bool",
        "config_field": "use_tp_ratio",
        "default": True,
    },
    "tp_ratio": {
        "pine_names": ["target_ratio", "ratioTP", "tpRatio"],
        "pine_labels": ["Ratio Take Profit", "TP Ratio", "Ratio TP"],
        "input_type": "float",
        "config_field": "tp_ratio",
        "default": 1.0,
    },
    "use_n_bars_exit": {
        "pine_names": ["usar_salida_n_velas", "usarNVelas", "useNBars"],
        "pine_labels": ["Salida por N velas", "N Bars Exit", "Salida N velas"],
        "input_type": "bool",
        "config_field": "use_n_bars_exit",
        "default": False,
    },
    "n_bars_exit": {
        "pine_names": ["n_velas_salida", "nVelas", "nBars"],
        "pine_labels": ["Número de velas", "N velas", "N Bars"],
        "input_type": "int",
        "config_field": "n_bars_exit",
        "default": 5,
    },

    # ===== Filtros (Filters) =====
    "ma_filter": {
        "pine_names": ["filtro_mm50200", "filtroMA", "maFilter"],
        "pine_labels": ["Cruce MM 50/200", "MA Filter", "Filtro MA"],
        "input_type": "string",
        "config_field": "ma_filter",
        "default": "Sin filtro",
        "options": ["Sin filtro", "Alcista (MM50>200)", "Bajista (MM50<200)"],
    },

    # ===== Sesiones (Sessions) =====
    "use_london": {
        "pine_names": ["usarLondon", "useLondon", "london"],
        "pine_labels": ["London", "Londres"],
        "input_type": "bool",
        "config_field": "use_london",
        "default": True,
    },
    "use_newyork": {
        "pine_names": ["usarNewYork", "useNewYork", "newYork"],
        "pine_labels": ["New York", "Nueva York", "NY"],
        "input_type": "bool",
        "config_field": "use_newyork",
        "default": True,
    },
    "use_tokyo": {
        "pine_names": ["usarTokio", "useTokyo", "tokyo"],
        "pine_labels": ["Tokio", "Tokyo"],
        "input_type": "bool",
        "config_field": "use_tokyo",
        "default": True,
    },

    # ===== Dias de la semana (Days) =====
    "trade_monday": {
        "pine_names": ["operarLunes", "tradeMon", "monday"],
        "pine_labels": ["Lunes", "Monday"],
        "input_type": "bool",
        "config_field": "trade_monday",
        "default": True,
    },
    "trade_tuesday": {
        "pine_names": ["operarMartes", "tradeTue", "tuesday"],
        "pine_labels": ["Martes", "Tuesday"],
        "input_type": "bool",
        "config_field": "trade_tuesday",
        "default": True,
    },
    "trade_wednesday": {
        "pine_names": ["operarMiercoles", "tradeWed", "wednesday"],
        "pine_labels": ["Miércoles", "Miercoles", "Wednesday"],
        "input_type": "bool",
        "config_field": "trade_wednesday",
        "default": True,
    },
    "trade_thursday": {
        "pine_names": ["operarJueves", "tradeThu", "thursday"],
        "pine_labels": ["Jueves", "Thursday"],
        "input_type": "bool",
        "config_field": "trade_thursday",
        "default": True,
    },
    "trade_friday": {
        "pine_names": ["operarViernes", "tradeFri", "friday"],
        "pine_labels": ["Viernes", "Friday"],
        "input_type": "bool",
        "config_field": "trade_friday",
        "default": True,
    },
    "trade_saturday": {
        "pine_names": ["operarSabado", "tradeSat", "saturday"],
        "pine_labels": ["Sábado", "Sabado", "Saturday"],
        "input_type": "bool",
        "config_field": "trade_saturday",
        "default": True,
    },
    "trade_sunday": {
        "pine_names": ["operarDomingo", "tradeSun", "sunday"],
        "pine_labels": ["Domingo", "Sunday"],
        "input_type": "bool",
        "config_field": "trade_sunday",
        "default": True,
    },

    # ===== Gestion de Riesgo (Risk Management) =====
    "risk_type": {
        "pine_names": ["tipo_gestion", "riskType", "tipoRiesgo"],
        "pine_labels": ["Tipo de gestión de riesgo", "Risk Type", "Tipo gestion"],
        "input_type": "string",
        "config_field": "risk_type",
        "default": "fixed_size",
        "options": ["Tamaño fijo", "Riesgo monetario fijo", "Riesgo % equity"],
    },
    "fixed_size": {
        "pine_names": ["tamano_fijo_qty", "tamanoFijo", "fixedSize"],
        "pine_labels": ["Tamaño fijo", "Fixed Size"],
        "input_type": "float",
        "config_field": "fixed_size",
        "default": 1.0,
    },
    "fixed_risk_money": {
        "pine_names": ["riesgo_monetario", "riesgoMonetario", "fixedRisk"],
        "pine_labels": ["Riesgo monetario", "Fixed Risk"],
        "input_type": "float",
        "config_field": "fixed_risk_money",
        "default": 100.0,
    },
    "risk_percent": {
        "pine_names": ["porc_riesgo_equity", "porcRiesgo", "riskPercent"],
        "pine_labels": ["Porcentaje de equity", "Risk Percent", "% Equity"],
        "input_type": "float",
        "config_field": "risk_percent",
        "default": 1.0,
    },
    "qty_step": {
        "pine_names": ["qty_step", "qtyStep", "pasoQty"],
        "pine_labels": ["Paso de cantidad", "Qty Step"],
        "input_type": "float",
        "config_field": "qty_step",
        "default": 1.0,
    },
    "qty_min": {
        "pine_names": ["qty_min", "qtyMin", "minQty"],
        "pine_labels": ["Cantidad mínima", "Min Qty"],
        "input_type": "float",
        "config_field": "qty_min",
        "default": 1.0,
    },
}


# =============================================================================
# VALUE MAPPINGS
# Maps Pine Script string values to StrategyConfig values
# =============================================================================

RISK_TYPE_MAP = {
    "Tamaño fijo": "fixed_size",
    "Riesgo monetario fijo": "fixed_risk_money",
    "Riesgo % equity": "risk_percent",
}

# MA filter uses same values in both
MA_FILTER_MAP = {
    "Sin filtro": "Sin filtro",
    "Alcista (MM50>200)": "Alcista (MM50>200)",
    "Bajista (MM50<200)": "Bajista (MM50<200)",
}


def get_all_pine_names() -> List[str]:
    """Get all Pine Script variable names we're looking for."""
    names = []
    for pattern in INPUT_PATTERNS.values():
        names.extend(pattern["pine_names"])
    return names


def get_all_pine_labels() -> List[str]:
    """Get all Pine Script input labels we're looking for."""
    labels = []
    for pattern in INPUT_PATTERNS.values():
        labels.extend(pattern["pine_labels"])
    return labels


def get_pattern_by_pine_name(name: str) -> dict:
    """Find pattern definition by Pine Script variable name."""
    for key, pattern in INPUT_PATTERNS.items():
        if name in pattern["pine_names"]:
            return pattern
    return None


def get_pattern_by_label(label: str) -> dict:
    """Find pattern definition by Pine Script input label."""
    label_lower = label.lower()
    for key, pattern in INPUT_PATTERNS.items():
        for pine_label in pattern["pine_labels"]:
            if pine_label.lower() == label_lower:
                return pattern
    return None
