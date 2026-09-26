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
            .fill_null("")
            .map_elements(
                normalize_name,
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .fill_null("")
            .alias("normalized_name"),
            pl.col("business_address")
            .fill_null("")
            .map_elements(
                normalize_address,
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .fill_null("")
            .alias("normalized_address"),
        ]
    )

def normalize_tsv_to_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Normalize a TSV file and write the result to a Parquet cache.

    The input is scanned lazily, so the complete TSV is not first collected
    into a large in-memory DataFrame.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input TSV not found: {input_path}")

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Pass force=True to overwrite it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pl.scan_csv(
        input_path,
        separator="\t",
        has_header=True,
        infer_schema=False,
    )

    columns = set(frame.collect_schema().names())
    required_columns = {"business_name", "business_address"}
    missing_columns = required_columns - columns

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    normalized = frame.with_columns(
        [
            pl.col("business_name")
            .fill_null("")
            .map_elements(
                normalize_name,
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .fill_null("")
            .alias("normalized_name"),
            pl.col("business_address")
            .fill_null("")
            .map_elements(
                normalize_address,
                return_dtype=pl.String,
                skip_nulls=False,
            )
            .fill_null("")
            .alias("normalized_address"),
        ]
    )

    normalized.sink_parquet(
        output_path,
        engine="streaming",
    )

    return output_path


if __name__ == "__main__":
    dataset_dir = Path("../../../data/student_resource/dataset")

    # output1 = normalize_tsv_to_parquet(
    #     input_path=dataset_dir / "test" / "test_source1.tsv",
    #     output_path=Path("cache/normalized/test_source1.parquet"),
    #     force=True,
    # )
    # print(f"Wrote normalized Parquet: {output1}")
    # output2 = normalize_tsv_to_parquet(
    #         input_path=dataset_dir / "test" / "test_source2.tsv",
    #         output_path=Path("cache/normalized/test_source2.parquet"),
    #         force=True,
    #     )
    # print(f"Wrote normalized Parquet: {output2}")
    output3 = normalize_tsv_to_parquet(
            input_path=dataset_dir / "test" / "test_source3.tsv",
            output_path=Path("cache/normalized/test_source3.parquet"),
            force=True,
        )
    print(f"Wrote normalized Parquet: {output3}")

   

# if __name__ == "__main__":
#     examples = [
#     "राम  मार्केटिंग,  प्राइवेट लिमिटेड.",       # Devanagari + extra spaces + comma/period
#     "ਪੰਜਾਬ   ਟ੍ਰੇਡਿੰਗ   ਕੰਪਨੀ,",                  # Punjabi (Gurmukhi) + extra spaces + trailing comma
#     "தமிழ்நாடு   எலெக்ட்ரானிக்ஸ்.",              # Tamil + extra spaces + period
#     "Café-Français & Fils S.A.R.L.",             # French accents + & + hyphen + suffix
#     "  RAM & SONS PVT. LTD.  ",                  # English, mixed punctuation + suffix, leading/trailing space
#     "O'Brien's Auto-Repair Corp.",               # apostrophe + hyphen + suffix
#     "!!!Discount Mart!!!",                       # symbol-heavy junk
# ]

#     for value in examples:
#         print(f"{value!r} -> {normalize_name(value)!r}\n\n")

#     address_examples = [
#         "1712 Montebello Avenue, Phoenix, AZ",
#         "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
#         "",
#         None,
#     ]

#     for value in address_examples:
#         print(f"{value!r} -> {normalize_address(value)!r}\n\n")
#     data = Path("../../../data/student_resource/dataset")
#     source3_path = data / "train" / "train_source3.tsv"

#     df = pl.read_csv(
#         source3_path,
#         separator="\t",
#         infer_schema=False,
#         n_rows=10
#     )
#     normalized_df = normalize_dataframe(df)
#     print(
#         normalized_df.select(
#             [
#                 "entity_id",
#                 "business_name",
#                 "normalized_name",
#                 "business_address",
#                 "normalized_address",
#                 "country",
#             ]
#         )
#     )
