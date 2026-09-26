"""Shared, script-blind normalization for business entity records."""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import polars as pl

_WHITESPACE_RE = re.compile(r"\s+")

# Applied to complete tokens only, after punctuation has been separated.
LEGAL_SUFFIXES = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "llc": "limited liability company",
    "co": "company",
    "plc": "public limited company",
    "llp": "limited liability partnership",
    "sarl": "societe a responsabilite limitee",
    "sas": "societe par actions simplifiee",
    "sasu": "societe par actions simplifiee unipersonnelle",
    "pty": "proprietary",
}


def _as_text(value: Any) -> str:
    """Convert missing or non-text values into safe text."""
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value)


def _replace_punctuation(text: str) -> str:
    """Replace punctuation with spaces while preserving Unicode letters/digits."""
    characters: list[str] = []

    for character in text:
        if character == "&":
            characters.extend((" ", "and", " "))
        elif unicodedata.category(character).startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)

    return "".join(characters)


def _canonicalize_legal_suffixes(tokens: list[str]) -> list[str]:
    """Expand known legal suffix tokens without changing arbitrary substrings."""
    normalized_tokens: list[str] = []

    for token in tokens:
        replacement = LEGAL_SUFFIXES.get(token)
        if replacement is None:
            normalized_tokens.append(token)
        else:
            normalized_tokens.extend(replacement.split())

    return normalized_tokens


def normalize_text(value: Any, *, canonicalize_suffixes: bool = True) -> str:
    """Normalize a business name or address without transliteration.

    The same function is used for every country and writing system.
    Unicode letters and digits are preserved; punctuation becomes whitespace.
    """
    text = _as_text(value)
    text = unicodedata.normalize("NFKC", text).lower()
    text = _replace_punctuation(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    tokens = text.split()

    if canonicalize_suffixes:
        tokens = _canonicalize_legal_suffixes(tokens)

    return " ".join(tokens)


def normalize_name(value: Any) -> str:
    """Normalize a business name."""
    return normalize_text(value, canonicalize_suffixes=True)


def normalize_address(value: Any) -> str:
    """Normalize a business address.

    Legal suffix expansion is also harmless for addresses and keeps the
    normalization interface consistent, while digits are intentionally kept.
    """
    return normalize_text(value, canonicalize_suffixes=True)


def normalize_dataframe(frame: Any) -> Any:
    """Add normalized name and address columns to a Polars DataFrame.

    The input frame is not mutated. All original columns are retained.
    """
    import polars as pl

    required_columns = {"business_name", "business_address"}
    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    return frame.with_columns(
        [
            pl.col("business_name")
            .map_elements(normalize_name, return_dtype=pl.String)
            .alias("normalized_name"),
            pl.col("business_address")
            .map_elements(normalize_address, return_dtype=pl.String)
            .alias("normalized_address"),
        ]
    )

if __name__ == "__main__":
    examples = [
    "राम  मार्केटिंग,  प्राइवेट लिमिटेड.",       # Devanagari + extra spaces + comma/period
    "ਪੰਜਾਬ   ਟ੍ਰੇਡਿੰਗ   ਕੰਪਨੀ,",                  # Punjabi (Gurmukhi) + extra spaces + trailing comma
    "தமிழ்நாடு   எலெக்ட்ரானிக்ஸ்.",              # Tamil + extra spaces + period
    "Café-Français & Fils S.A.R.L.",             # French accents + & + hyphen + suffix
    "  RAM & SONS PVT. LTD.  ",                  # English, mixed punctuation + suffix, leading/trailing space
    "O'Brien's Auto-Repair Corp.",               # apostrophe + hyphen + suffix
    "!!!Discount Mart!!!",                       # symbol-heavy junk
]

    for value in examples:
        print(f"{value!r} -> {normalize_name(value)!r}\n\n")

    address_examples = [
        "1712 Montebello Avenue, Phoenix, AZ",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "",
        None,
    ]

    for value in address_examples:
        print(f"{value!r} -> {normalize_address(value)!r}\n\n")
    data = Path("../../../data/student_resource/dataset")
    source3_path = data / "train" / "train_source3.tsv"

    df = pl.read_csv(
        source3_path,
        separator="\t",
        infer_schema=False,
        n_rows=10
    )
    normalized_df = normalize_dataframe(df)
    print(
        normalized_df.select(
            [
                "entity_id",
                "business_name",
                "normalized_name",
                "business_address",
                "normalized_address",
                "country",
            ]
        )
    )