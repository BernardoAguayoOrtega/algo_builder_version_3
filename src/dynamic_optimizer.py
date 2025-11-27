"""
Dynamic Optimizer - Generates optimization parameters from extracted Pine Script inputs.

Instead of hardcoded optimization options (Sacudida, Engulfing, etc.), this module
dynamically generates optimizable parameters based on what the Pine Script defines.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from itertools import product
import numpy as np

from .pine_parser import ExtractedInput, InputType, UniversalParsedStrategy


@dataclass
class OptimizableParam:
    """A parameter that can be optimized."""
    var_name: str
    label: str
    input_type: InputType
    default_value: Any
    # For optimization
    enabled: bool = False
    values_to_test: List[Any] = field(default_factory=list)

    def generate_default_values(self, inp: ExtractedInput) -> List[Any]:
        """Generate default optimization values based on input type."""
        if inp.input_type == InputType.BOOL:
            return [True, False]

        elif inp.input_type == InputType.STRING and inp.options:
            return inp.options

        elif inp.input_type == InputType.INT:
            default = inp.default_value or 1
            min_val = int(inp.min_val) if inp.min_val else max(1, default - 5)
            max_val = int(inp.max_val) if inp.max_val else default + 5
            step = int(inp.step) if inp.step else 1
            # Generate 5-7 values around the default
            values = list(range(min_val, max_val + 1, step))
            if len(values) > 7:
                # Sample evenly
                indices = np.linspace(0, len(values) - 1, 7, dtype=int)
                values = [values[i] for i in indices]
            return values

        elif inp.input_type == InputType.FLOAT:
            default = inp.default_value or 1.0
            min_val = float(inp.min_val) if inp.min_val else max(0.1, default * 0.5)
            max_val = float(inp.max_val) if inp.max_val else default * 2.0
            step = float(inp.step) if inp.step else 0.1
            # Generate 5-7 values around the default
            values = np.arange(min_val, max_val + step, step).tolist()
            if len(values) > 7:
                # Sample evenly
                indices = np.linspace(0, len(values) - 1, 7, dtype=int)
                values = [round(values[i], 4) for i in indices]
            else:
                values = [round(v, 4) for v in values]
            return values

        return [inp.default_value]


@dataclass
class DynamicOptimizationSettings:
    """Dynamic optimization settings based on Pine Script inputs."""
    params: Dict[str, OptimizableParam] = field(default_factory=dict)

    def get_enabled_params(self) -> Dict[str, OptimizableParam]:
        """Get only the parameters that are enabled for optimization."""
        return {k: v for k, v in self.params.items() if v.enabled}

    def estimate_combinations(self) -> int:
        """Estimate total number of combinations to test."""
        count = 1
        for param in self.params.values():
            if param.enabled and param.values_to_test:
                count *= len(param.values_to_test)
        return count


def create_optimizable_params(parsed: UniversalParsedStrategy) -> Dict[str, OptimizableParam]:
    """
    Create optimizable parameters from parsed Pine Script.

    Args:
        parsed: Parsed Pine Script strategy

    Returns:
        Dictionary of var_name -> OptimizableParam
    """
    params = {}

    for var_name, inp in parsed.inputs.items():
        param = OptimizableParam(
            var_name=var_name,
            label=inp.label,
            input_type=inp.input_type,
            default_value=inp.default_value,
            enabled=False,
            values_to_test=[],
        )
        # Generate default values to test
        param.values_to_test = param.generate_default_values(inp)
        params[var_name] = param

    return params


def generate_dynamic_combinations(
    base_values: Dict[str, Any],
    settings: DynamicOptimizationSettings,
    max_combinations: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations for optimization.

    Args:
        base_values: Current input values (defaults)
        settings: Optimization settings with enabled params
        max_combinations: Maximum combinations to generate

    Returns:
        List of parameter value dictionaries to test
    """
    enabled = settings.get_enabled_params()

    if not enabled:
        # No optimization, return base values only
        return [base_values.copy()]

    # Build lists for product
    var_names = list(enabled.keys())
    value_lists = [enabled[name].values_to_test for name in var_names]

    # Generate all combinations
    combinations = []
    for values in product(*value_lists):
        combo = base_values.copy()
        for var_name, value in zip(var_names, values):
            combo[var_name] = value
        combinations.append(combo)

    # Apply max_combinations limit
    if max_combinations and len(combinations) > max_combinations:
        # Sample evenly
        step = len(combinations) // max_combinations
        combinations = combinations[::step][:max_combinations]

    return combinations


def get_param_summary(values: Dict[str, Any], enabled_params: Dict[str, OptimizableParam]) -> str:
    """Generate a summary string of parameter values."""
    parts = []
    for var_name, param in enabled_params.items():
        value = values.get(var_name, param.default_value)
        # Shorten label
        short_label = param.label[:10] + "..." if len(param.label) > 12 else param.label
        if param.input_type == InputType.BOOL:
            parts.append(f"{short_label}:{'Y' if value else 'N'}")
        elif param.input_type == InputType.FLOAT:
            parts.append(f"{short_label}:{value:.2f}")
        else:
            parts.append(f"{short_label}:{value}")
    return " | ".join(parts) if parts else "Default"
