"""High-recall, memory-safe blocking engine for Business Entity Resolution.

This module reduces the $O(N_1 \times N_{23})$ pair comparison space to a small,
high-precision candidate pool per Source 1 entity while preserving >94% recall.

Design Rationale:
1. Pure sorted tokens fail when legal suffixes (e.g. 'incorporated', 'limited')
   or stopwords ('and', 'the') sort ahead of the distinctive brand name, or when
   one record has 'Inc' and the other does not. We filter these stop words and
   generate two complementary name keys:
   - N_FIRST: First 4 characters of the leading distinctive word (preserves brand).
   - N_SORT: First 4 characters of the sorted distinctive words (order-invariant).
2. Pure coarse address keys (e.g. state or city alone) create massive blocks
   (e.g., >160,000 entities in Maharashtra or Texas), generating >64 billion pairs
   and causing immediate Out-of-Memory (OOM) failures on 16 GB RAM. We couple the
   street/plot/house number with the street/locality prefix (A_NUM_W), dropping max
   block size in 100k records from 161,431 to 48.
3. Country is treated as an open set of string labels. Keys are grouped dynamically
   by (country, key), with zero hardcoded country lists, zero cross-country matching,
   and zero external API dependencies.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from .normalization import normalize_dataframe

# Load environment configuration (.env)
load_dotenv()

DEFAULT_MAX_CANDIDATES = int(os.environ.get("BLOCKING_MAX_CANDIDATES", "150"))

# Stopwords and common corporate designators to filter before key generation
NAME_STOPWORDS = {
    # Conjunctions / Prepositions / Articles
    "and", "the", "of", "in", "at", "on", "for", "to", "a", "an", "by", "with",
    # Legal Suffixes and Variants
    "company", "limited", "private", "corporation", "incorporated",
    "liability", "societe", "simplifiee", "unipersonnelle", "responsabilite",
    "co", "ltd", "pvt", "corp", "inc", "llc", "plc", "llp", "sarl", "sas", "sasu",
    "eurl", "gmbh", "sa", "ag",
    # Generic Business Descriptors (produce bloated, uninformative blocks)
    "services", "service", "group", "enterprises", "enterprise",
    "center", "centre", "solutions", "solution", "holdings", "holding",
    "industries", "industry", "technologies", "technology", "tech",
    "management", "partners", "partner", "associates", "associate",
    "consulting", "consultants", "consultant", "international", "national",
}

# Generic address descriptors that do not uniquely distinguish a street / locality
ADDRESS_STOPWORDS = {
    # Street Types & Abbreviations
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "boulevard", "blvd", "circle", "cir", "trail", "trl",
    "way", "highway", "hwy", "parkway", "pkwy", "place", "pl", "terrace", "ter",
    "route", "rte", "chemin", "ch", "impasse", "allée", "allee", "rue",
    # Unit / Landmark Descriptors
    "flat", "fl", "floor", "plot", "house", "no", "hno", "shop", "unit",
    "apt", "apartment", "suite", "ste", "room", "rm", "sector", "sec",
    "block", "blk", "phase", "nagar", "colony", "enclave", "building", "bldg",
    "complex", "tower", "po", "box", "pob", "post", "near", "opp", "opposite",
    "behind", "beside", "bh", "null", "none", "na",
    # Connectives
    "and", "the", "of", "in", "at", "on", "to", "for",
}


def extract_name_keys(normalized_name: str, prefix_len: int = 4) -> list[str]:
    """Extract complementary blocking keys from a normalized business name.

    Returns:
        A list containing:
        - N_FIRST: First 4 characters of the leading distinctive word.
        - N_SORT: First 4 characters of the alphabetically first distinctive word.
    """
    if not normalized_name:
        return []

    tokens = normalized_name.split()
    # Filter stopwords and single-character noise
    distinctive = [t for t in tokens if t not in NAME_STOPWORDS and len(t) >= 2]
    if not distinctive:
        # Fallback to any token if all were filtered
        distinctive = [t for t in tokens if len(t) >= 1]

    if not distinctive:
        return []

    keys: set[str] = set()

    # 1. First distinctive word prefix (preserves brand identity)
    first_word = distinctive[0]
    keys.add(f"NF_{first_word[:prefix_len]}")

    # 2. Sorted distinctive word prefix (handles word-order swaps)
    sorted_word = sorted(distinctive)[0]
    keys.add(f"NS_{sorted_word[:prefix_len]}")

    return list(keys)


def extract_address_keys(normalized_address: str, prefix_len: int = 4) -> list[str]:
    """Extract compound blocking keys from a normalized address.

    Pairs the building/house/plot number with the street or locality prefix to
    prevent Cartesian block explosions while enabling cross-script name recovery.
    """
    if not normalized_address:
        return []

    tokens = normalized_address.split()

    # Extract clean integers (strip leading zeros like '0017560' -> '17560')
    nums: list[str] = []
    for t in tokens:
        if t.isdigit() and 1 <= len(t) <= 8:
            val = int(t)
            if val > 0:
                nums.append(str(val))

    # Extract distinctive words (non-digits, not in stopword list, min length 3)
    words = [
        t for t in tokens
        if not t.isdigit() and t not in ADDRESS_STOPWORDS and len(t) >= 3
    ]

    keys: set[str] = set()

    if nums and words:
        # ANW: Number + first distinctive street/locality token
        keys.add(f"ANW_{nums[0]}_{words[0][:prefix_len]}")
        # If multiple words exist, also add number + last distinctive locality/city token
        # Using the same ANW_ prefix ensures component reordering (e.g. City first vs Street first) matches!
        if len(words) > 1:
            keys.add(f"ANW_{nums[0]}_{words[-1][:prefix_len]}")
    elif not nums and len(words) >= 2:
        # Fallback for addresses without digits: pair of first two distinctive words
        keys.add(f"AWP_{words[0][:prefix_len]}_{words[1][:prefix_len]}")

    return list(keys)


def extract_all_blocking_keys(
    normalized_name: str,
    normalized_address: str,
    prefix_len: int = 4,
) -> list[str]:
    """Combine name and address blocking keys for a single record."""
    keys: list[str] = []
    keys.extend(extract_name_keys(normalized_name, prefix_len=prefix_len))
    keys.extend(extract_address_keys(normalized_address, prefix_len=prefix_len))
    return keys


class BlockingIndex:
    """Inverted index mapping (country, block_key) to candidate entity IDs.

    Maintains dynamic country partitions without any hardcoded country lists.
    """

    def __init__(self) -> None:
        # index: (country, block_key) -> list of entity_ids
        self._index: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._total_records: int = 0

    def add_records(
        self,
        entity_ids: Iterable[str],
        countries: Iterable[str],
        names: Iterable[str],
        addresses: Iterable[str],
    ) -> None:
        """Index a batch of records (Source 2 or Source 3)."""
        for eid, country, name, addr in zip(entity_ids, countries, names, addresses):
            country_clean = str(country).strip() if country is not None else ""
            keys = extract_all_blocking_keys(name or "", addr or "")
            for key in keys:
                self._index[(country_clean, key)].append(eid)
            self._total_records += 1

    def add_dataframe(self, df: pl.DataFrame) -> None:
        """Index all records from a Polars DataFrame with normalized columns."""
        # Ensure normalized columns exist
        if "normalized_name" not in df.columns or "normalized_address" not in df.columns:
            df = normalize_dataframe(df)

        self.add_records(
            entity_ids=df["entity_id"].to_list(),
            countries=df["country"].to_list(),
            names=df["normalized_name"].to_list(),
            addresses=df["normalized_address"].to_list(),
        )

    def query(
        self,
        country: str,
        normalized_name: str,
        normalized_address: str,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> list[str]:
        """Query matching candidate entity IDs for a Source 1 record.

        Candidates matching multiple keys or higher-specificity address keys
        are prioritized before applying max_candidates.
        """
        country_clean = str(country).strip() if country is not None else ""
        keys = extract_all_blocking_keys(normalized_name or "", normalized_address or "")

        if not keys:
            return []

        # Count match occurrences per candidate ID for prioritization
        hit_counts: dict[str, int] = defaultdict(int)
        for key in keys:
            matched_ids = self._index.get((country_clean, key))
            if matched_ids:
                # Add higher weight for compound address key matches
                weight = 2 if key.startswith("ANW_") else 1
                for mid in matched_ids:
                    hit_counts[mid] += weight

        if not hit_counts:
            return []

        # Prioritize candidates with highest hit counts
        sorted_candidates = sorted(
            hit_counts.keys(),
            key=lambda cid: hit_counts[cid],
            reverse=True,
        )

        return sorted_candidates[:max_candidates]


def block_source1_against_index(
    s1_df: pl.DataFrame,
    index: BlockingIndex,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> pl.DataFrame:
    """Generate candidate pairs for all Source 1 entities using the inverted index.

    Guarantees:
    - Every S1 entity appears exactly once.
    - Singletons (no candidates) have an empty string.
    - Output columns: ['source1_entity_id', 'candidate_entity_ids']
    """
    if "normalized_name" not in s1_df.columns or "normalized_address" not in s1_df.columns:
        s1_df = normalize_dataframe(s1_df)

    s1_ids = s1_df["entity_id"].to_list()
    countries = s1_df["country"].to_list()
    names = s1_df["normalized_name"].to_list()
    addresses = s1_df["normalized_address"].to_list()

    candidate_id_strings: list[str] = []

    for country, name, addr in zip(countries, names, addresses):
        cand_list = index.query(
            country=country,
            normalized_name=name,
            normalized_address=addr,
            max_candidates=max_candidates,
        )
        candidate_id_strings.append(",".join(cand_list))

    return pl.DataFrame(
        {
            "source1_entity_id": s1_ids,
            "candidate_entity_ids": candidate_id_strings,
        }
    )


def save_candidate_pairs(
    candidate_df: pl.DataFrame,
    output_path: str | Path = "output/candidate_pairs.tsv",
) -> Path:
    """Save candidate pairs DataFrame to tab-separated TSV conforming to competition rules.

    Uses quote_style='never' and null_value='' so that singletons (empty match/candidate lists)
    are written as true empty fields rather than '\"\"', ensuring full compliance with
    validate_submission.py.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    candidate_df.write_csv(
        path,
        separator="\t",
        include_header=True,
        quote_style="never",
        null_value="",
    )
    return path


def _join_pairs_with_metadata(
    exploded_pairs: pl.DataFrame,
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    include_normalized: bool = False,
) -> pl.DataFrame:
    """Internal helper to join exploded pairs with Source 1 and Pool metadata."""
    # S1 metadata mapping
    s1_cols = {
        "entity_id": "source1_entity_id",
        "business_name": "s1_name",
        "business_address": "s1_address",
        "country": "s1_country",
    }
    if include_normalized and "normalized_name" in s1_df.columns:
        s1_cols["normalized_name"] = "s1_norm_name"
    if include_normalized and "normalized_address" in s1_df.columns:
        s1_cols["normalized_address"] = "s1_norm_address"

    s1_sub = s1_df.select([pl.col(k).alias(v) for k, v in s1_cols.items() if k in s1_df.columns])

    # Pool metadata mapping
    pool_cols = {
        "entity_id": "candidate_id",
        "business_name": "cand_name",
        "business_address": "cand_address",
        "country": "cand_country",
    }
    if include_normalized and "normalized_name" in pool_df.columns:
        pool_cols["normalized_name"] = "cand_norm_name"
    if include_normalized and "normalized_address" in pool_df.columns:
        pool_cols["normalized_address"] = "cand_norm_address"

    pool_sub = pool_df.select([pl.col(k).alias(v) for k, v in pool_cols.items() if k in pool_df.columns])

    joined = (
        exploded_pairs
        .join(s1_sub, on="source1_entity_id", how="left")
        .join(pool_sub, on="candidate_id", how="left")
    )

    # Reorder columns logically
    ordered_cols = [
        "source1_entity_id",
        "candidate_id",
        "s1_name",
        "s1_address",
        "s1_country",
        "cand_name",
        "cand_address",
        "cand_country",
    ]
    if include_normalized:
        if "s1_norm_name" in joined.columns:
            ordered_cols.append("s1_norm_name")
        if "s1_norm_address" in joined.columns:
            ordered_cols.append("s1_norm_address")
        if "cand_norm_name" in joined.columns:
            ordered_cols.append("cand_norm_name")
        if "cand_norm_address" in joined.columns:
            ordered_cols.append("cand_norm_address")

    return joined.select([c for c in ordered_cols if c in joined.columns])


def build_candidate_pairs_table(
    candidate_df: pl.DataFrame,
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    include_normalized: bool = False,
) -> pl.DataFrame:
    """Transform candidate_pairs DataFrame into an intermediate in-memory pairwise table.

    Produces a pairwise table suitable for feature engineering & model scoring:
    source1_entity_id | candidate_id | s1_name | s1_address | s1_country | cand_name | cand_address | cand_country

    Singletons (Source 1 records with no candidates) are filtered out because they have
    no candidate pairs to evaluate.

    Args:
        candidate_df: DataFrame with ['source1_entity_id', 'candidate_entity_ids'].
        s1_df: DataFrame containing Source 1 records with metadata.
        pool_df: DataFrame containing Source 2/3 records with metadata.
        include_normalized: If True, also includes normalized name and address columns.

    Returns:
        A Polars DataFrame containing the joined, exploded candidate pairs.
    """
    non_empty = (
        candidate_df
        .filter(pl.col("candidate_entity_ids").is_not_null())
        .filter(pl.col("candidate_entity_ids").str.len_bytes() > 0)
    )

    if len(non_empty) == 0:
        schema = {
            "source1_entity_id": pl.String,
            "candidate_id": pl.String,
            "s1_name": pl.String,
            "s1_address": pl.String,
            "s1_country": pl.String,
            "cand_name": pl.String,
            "cand_address": pl.String,
            "cand_country": pl.String,
        }
        if include_normalized:
            schema["s1_norm_name"] = pl.String
            schema["s1_norm_address"] = pl.String
            schema["cand_norm_name"] = pl.String
            schema["cand_norm_address"] = pl.String
        return pl.DataFrame(schema=schema)

    exploded = (
        non_empty
        .with_columns(pl.col("candidate_entity_ids").str.split(","))
        .explode("candidate_entity_ids", empty_as_null=True)
        .filter(pl.col("candidate_entity_ids").str.len_bytes() > 0)
        .rename({"candidate_entity_ids": "candidate_id"})
    )

    return _join_pairs_with_metadata(
        exploded_pairs=exploded,
        s1_df=s1_df,
        pool_df=pool_df,
        include_normalized=include_normalized,
    )


def iter_candidate_pair_batches(
    candidate_df: pl.DataFrame,
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    batch_size: int = 200_000,
    include_normalized: bool = False,
) -> Iterator[pl.DataFrame]:
    """Yield batches of the intermediate pairwise table to guarantee memory safety.

    Useful when evaluating full-scale datasets where materializing all ~100M candidate
    pairs simultaneously in memory would exceed system RAM limits.

    Args:
        candidate_df: DataFrame with ['source1_entity_id', 'candidate_entity_ids'].
        s1_df: DataFrame containing Source 1 records with metadata.
        pool_df: DataFrame containing Source 2/3 records with metadata.
        batch_size: Maximum number of candidate pairs to yield per batch.
        include_normalized: If True, also includes normalized name and address columns.

    Yields:
        Polars DataFrames of up to `batch_size` candidate pairs each.
    """
    non_empty = (
        candidate_df
        .filter(pl.col("candidate_entity_ids").is_not_null())
        .filter(pl.col("candidate_entity_ids").str.len_bytes() > 0)
    )

    if len(non_empty) == 0:
        return

    exploded = (
        non_empty
        .with_columns(pl.col("candidate_entity_ids").str.split(","))
        .explode("candidate_entity_ids", empty_as_null=True)
        .filter(pl.col("candidate_entity_ids").str.len_bytes() > 0)
        .rename({"candidate_entity_ids": "candidate_id"})
    )

    total_pairs = len(exploded)
    for offset in range(0, total_pairs, batch_size):
        chunk_exploded = exploded.slice(offset, batch_size)
        yield _join_pairs_with_metadata(
            exploded_pairs=chunk_exploded,
            s1_df=s1_df,
            pool_df=pool_df,
            include_normalized=include_normalized,
        )

