"""Preset bias parameters for titratable amino acids.

Multi-configuration bias data from cubic box simulations.
All presets use FNEX 5.5 implicit constraint, RMLA bond theta impr,
cutoffs cutnb=14.0, ctofnb=12.0, ctonnb=10.0, and T=298.15 K.
"""

from typing import TypedDict


class BiasParams(TypedDict, total=False):
    """Bias parameters for a single residue."""

    lam: list[float]  # Linear bias terms [s1, s2, s3?]
    c: list[float]  # Quadratic coupling [s1s2, s1s3?, s2s3?]
    x: list[float]  # Cross-terms [s1s2, s1s3?, s2s1, s2s3?, s3s1?, s3s2?]
    s: list[float]  # Shift terms [s1s2, s1s3?, s2s1, s2s3?, s3s1?, s3s2?]


class ConfigParams(TypedDict):
    """Full configuration with all residue biases."""

    ASP: BiasParams
    GLU: BiasParams
    HSP: BiasParams
    LYS: BiasParams
    TYR: BiasParams


# Default configuration
DEFAULT_CONFIG = "pme_nn_vswitch"

# All available configurations
PRESET_CONFIGS: dict[str, ConfigParams] = {
    # =========================================================================
    # PME EX (Ewald eXact) configurations
    # =========================================================================
    "pme_ex_vswitch": {
        "ASP": {
            "lam": [0.000, 51.889, 51.776],
            "c": [-143.538, -142.432, -119.077],
            "x": [-7.429, -7.858, -9.785, 0.155, -10.270, 0.261],
            "s": [0.613, 0.750, 0.200, -1.975, 0.327, -2.052],
        },
        "GLU": {
            "lam": [0.000, 66.354, 66.186],
            "c": [-138.938, -139.203, -120.222],
            "x": [-8.328, -8.337, -10.835, 0.619, -10.856, 0.806],
            "s": [0.638, 0.676, 0.194, -2.183, 0.324, -2.264],
        },
        "HSP": {
            "lam": [0.000, -0.482, -16.558],
            "c": [-59.040, -70.016, -78.316],
            "x": [-2.959, -2.777, -2.161, 2.813, -1.548, 2.948],
            "s": [0.340, 0.102, -0.263, -1.106, -0.311, -0.942],
        },
        "LYS": {
            "lam": [0.000, 42.936],
            "c": [-48.064],
            "x": [-4.509, -3.880],
            "s": [1.249, 0.254],
        },
        "TYR": {
            "lam": [0.000, -132.627],
            "c": [-139.345],
            "x": [-10.544, -11.433],
            "s": [2.092, 1.918],
        },
        # Neutral Red N3-protonated (NREDO/NRD2, pKa=6.8)
        # Converged from 03_nr_n3 cubic box simulation (192 runs)
        "NRED": {
            "lam": [0.000, 67.588],
            "c": [-30.536],
            "x": [-13.579, -11.579],
            "s": [2.580, 2.228],
        },
    },
    "pme_ex_vfswitch": {
        "ASP": {
            "lam": [0.000, 51.934, 51.734],
            "c": [-143.031, -142.392, -119.153],
            "x": [-7.354, -7.598, -10.281, 0.342, -9.876, 0.352],
            "s": [0.515, 0.579, 0.335, -2.119, 0.153, -2.113],
        },
        "GLU": {
            "lam": [0.000, 66.374, 66.174],
            "c": [-141.287, -138.194, -117.623],
            "x": [-7.334, -8.636, -9.993, -0.192, -10.789, -0.108],
            "s": [0.387, 0.703, 0.030, -1.993, 0.210, -1.998],
        },
        "HSP": {
            "lam": [0.000, -0.050, -16.064],
            "c": [-58.307, -67.646, -79.953],
            "x": [-2.957, -3.075, -2.541, 3.200, -2.453, 3.625],
            "s": [0.346, 0.164, -0.084, -1.192, 0.026, -1.073],
        },
        "LYS": {
            "lam": [0.000, 42.840],
            "c": [-48.726],
            "x": [-4.306, -3.776],
            "s": [1.142, 0.330],
        },
        "TYR": {
            "lam": [0.000, -133.199],
            "c": [-159.509],
            "x": [-3.715, -4.933],
            "s": [0.119, 0.170],
        },
    },
    # =========================================================================
    # PME NN (Nearest Neighbor) configurations
    # =========================================================================
    "pme_nn_vswitch": {
        "ASP": {
            "lam": [0.000, 52.096, 52.002],
            "c": [-139.100, -138.202, -118.043],
            "x": [-8.083, -8.750, -9.732, -0.098, -10.447, 0.028],
            "s": [0.546, 0.863, 0.212, -2.064, 0.352, -2.096],
        },
        "GLU": {
            "lam": [0.000, 66.505, 66.332],
            "c": [-136.226, -135.872, -115.040],
            "x": [-8.655, -9.097, -10.401, -0.897, -10.621, -0.930],
            "s": [0.669, 0.890, 0.100, -1.895, 0.203, -1.986],
        },
        "HSP": {
            "lam": [0.000, 0.141, -16.402],
            "c": [-70.204, -69.968, -75.827],
            "x": [-0.397, -3.668, 0.396, 1.199, -3.122, 1.334],
            "s": [-0.265, 0.345, -0.956, -0.662, 0.077, -0.363],
        },
        "LYS": {
            "lam": [0.000, 43.011],
            "c": [-53.484],
            "x": [-4.201, -3.899],
            "s": [1.101, 0.333],
        },
        "TYR": {
            "lam": [0.000, -133.514],
            "c": [-155.242],
            "x": [-3.753, -5.329],
            "s": [-0.195, 0.741],
        },
    },
    "pme_nn_vfswitch": {
        "ASP": {
            "lam": [0.000, 52.092, 51.904],
            "c": [-149.220, -130.445, -114.910],
            "x": [-4.904, -11.121, -6.752, -1.005, -12.944, -1.039],
            "s": [-0.130, 1.367, -0.495, -1.884, 0.920, -1.894],
        },
        "GLU": {
            "lam": [0.000, 66.259, 66.105],
            "c": [-133.204, -133.022, -120.380],
            "x": [-9.469, -9.619, -11.621, 0.876, -11.446, 1.004],
            "s": [0.690, 0.701, 0.507, -2.358, 0.448, -2.428],
        },
        "HSP": {
            "lam": [0.000, 0.367, -16.038],
            "c": [-63.688, -74.274, -79.480],
            "x": [-2.658, -2.563, -1.828, 2.343, -1.686, 2.611],
            "s": [0.239, 0.228, -0.370, -0.856, -0.419, -0.912],
        },
        "LYS": {
            "lam": [0.000, 43.096],
            "c": [-52.277],
            "x": [-4.906, -4.042],
            "s": [1.406, 0.360],
        },
        "TYR": {
            "lam": [0.000, -132.388],
            "c": [-154.782],
            "x": [-4.125, -5.133],
            "s": [0.255, 0.231],
        },
    },
    # =========================================================================
    # PME ON configurations
    # =========================================================================
    "pme_on_vswitch": {
        "ASP": {
            "lam": [0.000, 52.074, 51.908],
            "c": [-204.249, -204.649, -204.449],
            "x": [-7.372, -7.275, -10.171, -0.120, -9.772, -0.027],
            "s": [0.560, 0.498, 0.307, -1.940, 0.172, -2.041],
        },
        "GLU": {
            "lam": [0.000, 66.317, 66.080],
            "c": [-201.018, -202.839, -203.864],
            "x": [-8.084, -7.559, -10.697, -0.375, -10.088, -0.079],
            "s": [0.599, 0.560, 0.290, -1.915, 0.126, -2.019],
        },
        "HSP": {
            "lam": [0.000, 0.042, -16.334],
            "c": [-153.634, -150.322, -156.182],
            "x": [-2.511, -3.346, -1.832, 3.172, -2.239, 3.476],
            "s": [0.262, 0.220, -0.356, -1.231, -0.151, -0.994],
        },
        "LYS": {
            "lam": [0.000, 42.944],
            "c": [-112.865],
            "x": [-4.338, -3.805],
            "s": [1.311, 0.250],
        },
        "TYR": {
            "lam": [0.000, -132.736],
            "c": [-156.951],
            "x": [-2.763, -3.812],
            "s": [-0.193, -0.003],
        },
    },
    "pme_on_vfswitch": {
        "ASP": {
            "lam": [0.000, 51.829, 51.719],
            "c": [-203.808, -205.002, -204.052],
            "x": [-7.780, -7.231, -10.471, -0.234, -9.872, -0.067],
            "s": [0.703, 0.559, 0.415, -1.933, 0.243, -1.986],
        },
        "GLU": {
            "lam": [0.000, 66.130, 66.043],
            "c": [-208.802, -196.127, -200.510],
            "x": [-5.783, -9.894, -7.859, -1.416, -12.122, -1.096],
            "s": [0.101, 1.052, -0.367, -1.696, 0.667, -1.878],
        },
        "HSP": {
            "lam": [0.000, -0.580, -16.812],
            "c": [-152.905, -149.596, -155.808],
            "x": [-2.678, -3.514, -1.830, 3.118, -2.573, 3.444],
            "s": [0.255, 0.367, -0.436, -1.117, -0.136, -1.067],
        },
        "LYS": {
            "lam": [0.000, 42.879],
            "c": [-113.004],
            "x": [-4.377, -3.687],
            "s": [1.264, 0.287],
        },
        "TYR": {
            "lam": [0.000, -132.437],
            "c": [-132.999],
            "x": [-10.987, -11.824],
            "s": [2.125, 2.020],
        },
    },
    # =========================================================================
    # FSHIFT configurations
    # =========================================================================
    "fshift_vswitch": {
        "ASP": {
            "lam": [0.000, 37.156, 37.016],
            "c": [-43.744, -41.114, -26.026],
            "x": [-8.097, -8.821, -10.465, -0.027, -11.195, -0.092],
            "s": [0.557, 0.682, 0.247, -2.272, 0.418, -2.208],
        },
        "GLU": {
            "lam": [0.000, 49.653, 49.498],
            "c": [-37.965, -38.295, -26.785],
            "x": [-9.297, -9.086, -11.758, 0.135, -11.438, 0.188],
            "s": [0.686, 0.599, 0.322, -2.315, 0.128, -2.358],
        },
        "HSP": {
            "lam": [0.000, -5.374, -15.084],
            "c": [-57.245, -54.783, -56.870],
            "x": [-2.463, -2.963, -2.169, 3.255, -2.592, 3.573],
            "s": [0.213, 0.174, -0.215, -1.188, -0.050, -1.113],
        },
        "LYS": {
            "lam": [0.000, 24.337],
            "c": [-46.912],
            "x": [-4.321, -3.903],
            "s": [1.338, 0.228],
        },
        "TYR": {
            "lam": [0.000, -85.508],
            "c": [-51.096],
            "x": [-3.830, -5.149],
            "s": [0.175, 0.018],
        },
    },
    "fshift_vfswitch": {
        "ASP": {
            "lam": [0.000, 36.991, 36.855],
            "c": [-42.075, -42.088, -27.509],
            "x": [-8.569, -8.534, -11.101, 0.187, -10.876, 0.388],
            "s": [0.702, 0.733, 0.310, -2.144, 0.211, -2.275],
        },
        "GLU": {
            "lam": [0.000, 49.496, 49.270],
            "c": [-37.765, -40.394, -27.061],
            "x": [-9.505, -8.626, -11.880, 0.344, -10.976, 0.310],
            "s": [0.730, 0.628, 0.347, -2.439, 0.229, -2.436],
        },
        "HSP": {
            "lam": [0.000, -5.428, -15.059],
            "c": [-54.182, -55.395, -58.056],
            "x": [-3.705, -3.193, -2.997, 3.426, -2.270, 3.895],
            "s": [0.681, 0.388, -0.213, -1.185, -0.177, -1.015],
        },
        "LYS": {
            "lam": [0.000, 24.239],
            "c": [-47.108],
            "x": [-4.170, -3.913],
            "s": [1.214, 0.353],
        },
        "TYR": {
            "lam": [0.000, -85.339],
            "c": [-50.302],
            "x": [-4.156, -5.528],
            "s": [0.223, 0.156],
        },
    },
    # =========================================================================
    # FSWITCH configurations
    # =========================================================================
    "fswitch_vswitch": {
        "ASP": {
            "lam": [0.000, 44.260, 44.141],
            "c": [-52.335, -55.728, -26.857],
            "x": [-8.664, -7.487, -11.363, 0.021, -10.171, 0.046],
            "s": [0.674, 0.417, 0.406, -2.153, 0.098, -2.224],
        },
        "GLU": {
            "lam": [0.000, 57.395, 57.383],
            "c": [-57.132, -36.204, -17.753],
            "x": [-6.264, -13.372, -9.041, -2.653, -15.598, -2.570],
            "s": [-0.163, 1.498, -0.233, -1.690, 1.241, -1.753],
        },
        "HSP": {
            "lam": [0.000, -3.069, -15.609],
            "c": [-64.472, -67.191, -60.137],
            "x": [-3.155, -2.347, -3.128, 3.333, -1.777, 3.954],
            "s": [0.339, 0.115, -0.020, -0.930, -0.447, -1.253],
        },
        "LYS": {
            "lam": [0.000, 31.513],
            "c": [-58.518],
            "x": [-4.166, -3.932],
            "s": [1.230, 0.294],
        },
        "TYR": {
            "lam": [0.000, -104.979],
            "c": [-59.217],
            "x": [-4.952, -6.439],
            "s": [0.419, 0.423],
        },
    },
    "fswitch_vfswitch": {
        "ASP": {
            "lam": [0.000, 44.002, 43.818],
            "c": [-53.025, -53.793, -27.908],
            "x": [-8.314, -7.959, -10.689, 0.245, -10.554, 0.409],
            "s": [0.531, 0.549, 0.166, -2.203, 0.165, -2.349],
        },
        "GLU": {
            "lam": [0.000, 57.265, 57.122],
            "c": [-47.827, -49.704, -27.015],
            "x": [-9.809, -8.916, -12.161, 0.135, -11.407, 0.126],
            "s": [0.748, 0.536, 0.415, -2.425, 0.321, -2.344],
        },
        "HSP": {
            "lam": [0.000, -3.771, -16.120],
            "c": [-67.239, -65.532, -58.995],
            "x": [-2.551, -2.666, -1.948, 3.076, -2.172, 3.332],
            "s": [0.300, 0.186, -0.437, -0.931, -0.387, -0.902],
        },
        "LYS": {
            "lam": [0.000, 31.570],
            "c": [-58.914],
            "x": [-4.248, -3.802],
            "s": [1.346, 0.268],
        },
        "TYR": {
            "lam": [0.000, -105.279],
            "c": [-61.071],
            "x": [-4.176, -5.675],
            "s": [0.183, 0.167],
        },
    },
}

# Legacy alias for default config
PRESET_CONFIG = PRESET_CONFIGS[DEFAULT_CONFIG]
PRESET_BIASES = PRESET_CONFIG  # Another legacy alias


def list_configs() -> list[str]:
    """List all available configuration names."""
    return sorted(PRESET_CONFIGS.keys())


def get_config(name: str | None = None) -> ConfigParams:
    """Get a full configuration by name.

    Args:
        name: Configuration name. If None, returns default.

    Returns:
        Dictionary of residue -> bias parameters.

    Raises:
        KeyError: If configuration name not found.
    """
    if name is None:
        name = DEFAULT_CONFIG
    if name not in PRESET_CONFIGS:
        raise KeyError(
            f"Unknown config '{name}'. Available: {list_configs()}"
        )
    return PRESET_CONFIGS[name]


def list_presets(config: str | None = None) -> list[str]:
    """List available residue presets for a configuration.

    Args:
        config: Configuration name. If None, uses default.

    Returns:
        List of residue names (ASP, GLU, HSP, LYS, TYR).
    """
    cfg = get_config(config)
    return sorted(cfg.keys())


def get_preset_biases(
    residue: str, config: str | None = None
) -> BiasParams:
    """Get bias parameters for a specific residue.

    Args:
        residue: Residue name (ASP, GLU, HSP, LYS, TYR).
        config: Configuration name. If None, uses default.

    Returns:
        Dictionary with lam, c, x, s bias arrays.

    Raises:
        KeyError: If residue or config not found.
    """
    cfg = get_config(config)
    residue = residue.upper()
    if residue not in cfg:
        raise KeyError(
            f"Unknown residue '{residue}'. Available: {list_presets(config)}"
        )
    return cfg[residue]


def get_bias_params_only(
    residue: str, config: str | None = None
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Get bias parameters as separate arrays.

    Args:
        residue: Residue name.
        config: Configuration name.

    Returns:
        Tuple of (lam, c, x, s) arrays.
    """
    params = get_preset_biases(residue, config)
    return params["lam"], params["c"], params["x"], params["s"]


def write_preset_variables(
    residue: str,
    output_path: str,
    config: str | None = None,
    nreps: int = 3,
    temp: float = 298.15,
) -> None:
    """Write bias parameters to a CHARMM variable file.

    Args:
        residue: Residue name.
        output_path: Path to write the .inp file.
        config: Configuration name.
        nreps: Number of replicas.
        temp: Temperature in K.
    """
    params = get_preset_biases(residue, config)
    lam = params["lam"]
    c = params["c"]
    x = params["x"]
    s = params["s"]

    nsubs = len(lam)
    is_3state = nsubs == 3

    lines = [
        f"* Preset variables for {residue} ({config or DEFAULT_CONFIG})",
        "*",
        "",
    ]

    # Lambda terms
    for i, val in enumerate(lam, 1):
        lines.append(f"set lams1s{i} = {val:10.3f}")

    # C terms (quadratic coupling)
    if is_3state:
        lines.append(f"set cs1s1s1s2 = {c[0]:10.3f}")
        lines.append(f"set cs1s1s1s3 = {c[1]:10.3f}")
        lines.append(f"set cs1s2s1s3 = {c[2]:10.3f}")
    else:
        lines.append(f"set cs1s1s1s2 = {c[0]:10.3f}")

    # X terms (cross)
    if is_3state:
        lines.append(f"set xs1s1s1s2 = {x[0]:10.3f}")
        lines.append(f"set xs1s1s1s3 = {x[1]:10.3f}")
        lines.append(f"set xs1s2s1s1 = {x[2]:10.3f}")
        lines.append(f"set xs1s2s1s3 = {x[3]:10.3f}")
        lines.append(f"set xs1s3s1s1 = {x[4]:10.3f}")
        lines.append(f"set xs1s3s1s2 = {x[5]:10.3f}")
    else:
        lines.append(f"set xs1s1s1s2 = {x[0]:10.3f}")
        lines.append(f"set xs1s2s1s1 = {x[1]:10.3f}")

    # S terms (shift)
    if is_3state:
        lines.append(f"set ss1s1s1s2 = {s[0]:10.3f}")
        lines.append(f"set ss1s1s1s3 = {s[1]:10.3f}")
        lines.append(f"set ss1s2s1s1 = {s[2]:10.3f}")
        lines.append(f"set ss1s2s1s3 = {s[3]:10.3f}")
        lines.append(f"set ss1s3s1s1 = {s[4]:10.3f}")
        lines.append(f"set ss1s3s1s2 = {s[5]:10.3f}")
    else:
        lines.append(f"set ss1s1s1s2 = {s[0]:10.3f}")
        lines.append(f"set ss1s2s1s1 = {s[1]:10.3f}")

    # Metadata
    lines.extend([
        f'set sysname = "{residue.lower()}"',
        "trim sysname from 2",
        "set nnodes = 1",
        f"set nreps = {nreps}",
        "set ncentral = 0",
        "set nblocks = 3",
        "set nsites = 1",
        f"set nsubs1 = {nsubs}",
        f"set temp = {temp}",
        "set minimize = 0",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
