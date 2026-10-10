"""Versioned labels from the author's Derm7PtDatasetGroupInfrequent.

Source: https://github.com/jeremykawahara/derm7pt/blob/master/derm7pt/dataset.py
Verified 2026-10-04. Unknown labels fail rather than silently becoming MISC.
"""

LABEL_SCHEMA_VERSION = 1
LABEL_SOURCE = "https://github.com/jeremykawahara/derm7pt/blob/master/derm7pt/dataset.py"
DIAGNOSIS_CLASSES = ("BCC", "NEV", "MEL", "MISC", "SK")
CLASS_ORDER = DIAGNOSIS_CLASSES
_DIAGNOSIS_GROUPS = (
    ("basal cell carcinoma",),
    ("nevus", "blue nevus", "clark nevus", "combined nevus", "congenital nevus", "dermal nevus", "recurrent nevus", "reed or spitz nevus"),
    ("melanoma", "melanoma (in situ)", "melanoma (less than 0.76 mm)", "melanoma (0.76 to 1.5 mm)", "melanoma (more than 1.5 mm)", "melanoma metastasis"),
    ("df/lt/mls/misc", "dermatofibroma", "lentigo", "melanosis", "miscellaneous", "vascular lesion"),
    ("seborrheic keratosis",),
)
DIAGNOSIS_MAP = {label: index for index, labels in enumerate(_DIAGNOSIS_GROUPS) for label in labels}
ATTRIBUTE_CLASSES = {
    "pigment_network": ("absent", "typical", "atypical"),
    "blue_whitish_veil": ("absent", "present"),
    "vascular_structures": ("absent", "regular", "irregular"),
    "pigmentation": ("absent", "regular", "irregular"),
    "streaks": ("absent", "regular", "irregular"),
    "dots_and_globules": ("absent", "regular", "irregular"),
    "regression_structures": ("absent", "present"),
}
_ATTRIBUTE_GROUPS = {
    "pigment_network": (("absent",), ("typical",), ("atypical",)),
    "blue_whitish_veil": (("absent",), ("present",)),
    "vascular_structures": (("absent",), ("regular", "arborizing", "comma", "hairpin", "within regression", "wreath"), ("dotted/irregular", "dotted", "linear irregular")),
    "pigmentation": (("absent",), ("regular", "diffuse regular", "localized regular"), ("irregular", "diffuse irregular", "localized irregular")),
    "streaks": (("absent",), ("regular",), ("irregular",)),
    "dots_and_globules": (("absent",), ("regular",), ("irregular",)),
    "regression_structures": (("absent",), ("present", "blue areas", "white areas", "combinations")),
}
ATTRIBUTE_MAPS = {name: {label: index for index, group in enumerate(groups) for label in group} for name, groups in _ATTRIBUTE_GROUPS.items()}
_MISSING = {"", "nan", "na", "n/a", "null", "none", "-1"}


def _normal(value):
    return "" if value is None else str(value).strip().lower()


def encode_diagnosis(value):
    label = _normal(value)
    if label not in DIAGNOSIS_MAP:
        raise ValueError(f"Unknown Derm7pt diagnosis: {value!r}")
    return DIAGNOSIS_MAP[label]


def encode_attribute(name, value):
    """Return (class index, valid mask); missing is -1, not absent."""
    if name not in ATTRIBUTE_MAPS:
        raise ValueError(f"Unknown attribute name: {name!r}")
    label = _normal(value)
    if label in _MISSING:
        return -1, False
    if label not in ATTRIBUTE_MAPS[name]:
        raise ValueError(f"Unknown Derm7pt {name} label: {value!r}")
    return ATTRIBUTE_MAPS[name][label], True


diagnosis_index = encode_diagnosis


def label_contract():
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "source_url": LABEL_SOURCE,
        "verified_date": "2026-10-04",
        "diagnosis_classes": list(DIAGNOSIS_CLASSES),
        "class_order": list(CLASS_ORDER),
        "diagnosis_map": DIAGNOSIS_MAP,
        "attribute_classes": {name: list(classes) for name, classes in ATTRIBUTE_CLASSES.items()},
        "attribute_maps": ATTRIBUTE_MAPS,
        "missing_attribute_index": -1,
    }
