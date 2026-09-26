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

import itertools
import os
import re
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
    "eurl", "gmbh", "sa", "ag", "pty", "proprietary",
    # Generic Business Descriptors (produce bloated, uninformative blocks)
    "services", "service", "group", "enterprises", "enterprise",
    "center", "centre", "solutions", "solution", "holdings", "holding",
    "industries", "industry", "technologies", "technology", "tech",
    "management", "partners", "partner", "associates", "associate",
    "consulting", "consultants", "consultant", "international", "national",
    "global", "commercial", "trading", "operating",
    # DBA / Alias connectives & noise
    "dba", "doing", "business", "as",
    # Web & Domain noise
    "com", "org", "net", "www", "io",
    # Country / Geographic Noise in Business Names (frequent non-distinctive descriptors)
    "india", "indian", "usa", "america", "american", "france", "french",
    # Common Honorifics in Indian Business Names
    "shri", "sri", "smt", "dr", "mr", "mrs", "ms",
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
    "behind", "beside", "bh", "null", "none", "na", "door",
    # Directional Descriptors
    "north", "south", "east", "west", "central", "new", "old",
    # Connectives
    "and", "the", "of", "in", "at", "on", "to", "for",
}


def extract_name_keys(normalized_name: str, prefix_len: int = 4) -> list[str]:
    """Extract complementary blocking keys from a normalized business name.

    Handles trade-name / DBA splits and secondary brand tokens.
    Returns:
        - NF_: First prefix_len characters of leading distinctive word(s).
        - NS_: First prefix_len characters of primary sorted distinctive word.
        - NS2_: First prefix_len characters of secondary sorted distinctive word (if present).
    """
    if not normalized_name:
        return []

    # Split on DBA / trade-name indicators
    parts = re.split(r"\b(?:dba|doing business as|d/b/a|t/a|trading as)\b", normalized_name)

    keys: list[str] = []
    all_distinctive: list[str] = []

    for part in parts:
        tokens = [
            t for t in part.split()
            if t not in NAME_STOPWORDS and len(t) >= 2 and re.search(r"\w", t)
        ]
        if tokens:
            keys.append(f"NF_{tokens[0][:prefix_len]}")
            all_distinctive.extend(tokens)

    if not all_distinctive:
        # Fallback to any token with alphanumeric characters if all were stopwords
        tokens = [t for t in normalized_name.split() if len(t) >= 1 and re.search(r"\w", t)]
        if tokens:
            keys.append(f"NF_{tokens[0][:prefix_len]}")
            all_distinctive.extend(tokens)

    if all_distinctive:
        sorted_distinct = sorted(list(set(all_distinctive)))
        keys.append(f"NS_{sorted_distinct[0][:prefix_len]}")
        if len(sorted_distinct) > 1:
            keys.append(f"NS2_{sorted_distinct[1][:prefix_len]}")

    return list(dict.fromkeys(keys))


def extract_address_keys(normalized_address: str, prefix_len: int = 4) -> list[str]:
    """Extract compound blocking keys from a normalized address.

    Pairs up to 2 building/plot numbers with top distinctive locality tokens (ANW_)
    and generates universal sorted token-pair keys (AWP_) to bridge missing numbers.
    """
    if not normalized_address:
        return []

    tokens = normalized_address.split()

    # Extract unique positive integers (strip leading zeros like '0017560' -> '17560')
    nums: list[str] = []
    for t in tokens:
        if t.isdigit() and 1 <= len(t) <= 8:
            val = int(t)
            if val > 0 and str(val) not in nums:
                nums.append(str(val))

    # Extract distinctive words (non-digits, not in stopword list, min length 3, has word chars)
    words = [
        t for t in tokens
        if not t.isdigit() and t not in ADDRESS_STOPWORDS and len(t) >= 3 and re.search(r"\w", t)
    ]

    keys: list[str] = []

    # Pass 1: ANW_ compound keys (cross up to 2 numbers with top 3 distinctive words + last word)
    if nums and words:
        for num in nums[:2]:
            for w in words[:3]:
                keys.append(f"ANW_{num}_{w[:prefix_len]}")
            if len(words) > 3:
                keys.append(f"ANW_{num}_{words[-1][:prefix_len]}")

    # Pass 2: Universal AWP_ (sorted word pair combinations across top 3 words)
    if len(words) >= 2:
        sorted_w = sorted(list(set(words)))[:3]
        for a, b in itertools.combinations(sorted_w, 2):
            keys.append(f"AWP_{a[:prefix_len]}_{b[:prefix_len]}")

    return list(dict.fromkeys(keys))


def extract_all_blocking_keys(
    normalized_name: str,
    normalized_address: str,
    prefix_len: int = 4,
) -> list[str]:
    """Combine name and address blocking keys for a single record."""
    keys: list[str] = []
    keys.extend(extract_address_keys(normalized_address, prefix_len=prefix_len))
    keys.extend(extract_name_keys(normalized_name, prefix_len=prefix_len))
    return list(dict.fromkeys(keys))


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
        max_block_size: int = 1000,
    ) -> list[str]:
        """Query matching candidate entity IDs for a Source 1 record.

        Retrieval is prioritized across specificity tiers:
        Tier 1: ANW_ (Address + Number compound)
        Tier 2: NF_ (First brand token prefix)
        Tier 3: AWP_ (Address Word Pair combinations)
        Tier 4: NS_ (Primary sorted brand token)
        Tier 5: NS2_ (Secondary sorted brand token)
        Other keys as residual.

        Blocks with more than max_block_size entries are skipped as uninformative.
        """
        country_clean = str(country).strip() if country is not None else ""
        keys = extract_all_blocking_keys(normalized_name or "", normalized_address or "")

        if not keys:
            return []

        # Partition keys into prioritized tiers
        tier1 = [k for k in keys if k.startswith("ANW_")]
        tier2 = [k for k in keys if k.startswith("NF_")]
        tier3 = [k for k in keys if k.startswith("AWP_")]
        tier4 = [k for k in keys if k.startswith("NS_")]
        tier5 = [k for k in keys if k.startswith("NS2_")]
        tier_other = [
            k for k in keys
            if not (k.startswith("ANW_") or k.startswith("NF_") or k.startswith("AWP_") or k.startswith("NS_") or k.startswith("NS2_"))
        ]

        candidates: list[str] = []
        seen: set[str] = set()

        for tier in (tier1, tier2, tier3, tier4, tier5, tier_other):
            for key in tier:
                matched_ids = self._index.get((country_clean, key))
                if matched_ids and len(matched_ids) <= max_block_size:
                    for cid in matched_ids:
                        if cid not in seen:
                            seen.add(cid)
                            candidates.append(cid)
                            if len(candidates) >= max_candidates:
                                return candidates
                elif matched_ids and len(matched_ids) > max_block_size:
                    # Guard: If no candidates found yet, sample up to 20 from large blocks
                    if len(candidates) == 0:
                        for cid in matched_ids[:20]:
                            if cid not in seen:
                                seen.add(cid)
                                candidates.append(cid)

            if len(candidates) >= max_candidates:
                break

        return candidates[:max_candidates]


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


def build_candidate_id_pairs(candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Explode compact candidates without joining the full source pool.

    Blocking only needs to persist the pair IDs. Feature engineering can join
    these IDs to normalized source data in its own bounded batches. Avoiding a
    full Source 2/3 metadata join here is critical at multi-million-row scale.
    """
    non_empty = candidate_df.filter(
        pl.col("candidate_entity_ids").fill_null("").str.len_bytes() > 0
    )
    if len(non_empty) == 0:
        return pl.DataFrame(
            schema={
                "source1_entity_id": pl.String,
                "candidate_id": pl.String,
            }
        )

    return (
        non_empty
        .with_columns(pl.col("candidate_entity_ids").str.split(","))
        .explode("candidate_entity_ids", empty_as_null=True)
        .filter(pl.col("candidate_entity_ids").str.len_bytes() > 0)
        .rename({"candidate_entity_ids": "candidate_id"})
        .select("source1_entity_id", "candidate_id")
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
    # Accumulate only enough compact rows to form one pair batch. Never
    # explode the complete candidate table before yielding.
    pending_rows: list[tuple[str, str]] = []
    pending_pairs = 0

    def flush_pending() -> pl.DataFrame | None:
        nonlocal pending_rows, pending_pairs
        if not pending_rows:
            return None

        pending = pl.DataFrame(
            {
                "source1_entity_id": [row[0] for row in pending_rows],
                "candidate_entity_ids": [row[1] for row in pending_rows],
            }
        )
        exploded = (
            pending
            .with_columns(pl.col("candidate_entity_ids").str.split(","))
            .explode("candidate_entity_ids", empty_as_null=True)
            .filter(pl.col("candidate_entity_ids").str.len_bytes() > 0)
            .rename({"candidate_entity_ids": "candidate_id"})
        )
        result = _join_pairs_with_metadata(
            exploded_pairs=exploded,
            s1_df=s1_df,
            pool_df=pool_df,
            include_normalized=include_normalized,
        )
        pending_rows = []
        pending_pairs = 0
        return result

    for source1_id, candidate_ids in candidate_df.iter_rows():
        ids = [candidate_id for candidate_id in (candidate_ids or "").split(",") if candidate_id]
        if not ids:
            continue

        for start in range(0, len(ids), batch_size):
            id_chunk = ids[start : start + batch_size]
            if pending_rows and pending_pairs + len(id_chunk) > batch_size:
                result = flush_pending()
                if result is not None:
                    yield result

            pending_rows.append((source1_id, ",".join(id_chunk)))
            pending_pairs += len(id_chunk)

    result = flush_pending()
    if result is not None:
        yield result


def write_candidate_outputs(
    candidate_df: pl.DataFrame,
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    candidate_tsv_path: str | Path = "output/candidate_pairs.tsv",
    pair_cache_dir: str | Path = "cache/candidates/test_pairs",
    batch_size: int = 50_000,
    force: bool = False,
) -> tuple[Path, Path, int]:
    """Write compact candidates and long-form candidate parts together.

    The compact TSV is the required submission artifact. The Parquet parts are
    an internal cache consumed by feature engineering. Both are produced from
    the same bounded candidate chunks so feature experiments do not rerun
    blocking.
    """
    candidate_tsv_path = Path(candidate_tsv_path)
    pair_cache_dir = Path(pair_cache_dir)
    candidate_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    pair_cache_dir.mkdir(parents=True, exist_ok=True)

    if candidate_tsv_path.exists() and not force:
        raise FileExistsError(
            f"Candidate TSV already exists: {candidate_tsv_path}. "
            "Pass force=True to replace it."
        )

    existing_parts = sorted(pair_cache_dir.glob("part-*.parquet"))
    if existing_parts and not force:
        raise FileExistsError(
            f"Candidate cache already contains Parquet parts: {pair_cache_dir}. "
            "Pass force=True to replace them."
        )
    if force:
        for part in existing_parts:
            part.unlink()

    return write_candidate_outputs_from_batches(
        candidate_batches=[candidate_df],
        s1_df=s1_df,
        pool_df=pool_df,
        candidate_tsv_path=candidate_tsv_path,
        pair_cache_dir=pair_cache_dir,
        force=force,
        batch_size=batch_size,
    )


def iter_blocking_candidate_batches(
    s1_df: pl.DataFrame,
    index: BlockingIndex,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    batch_size: int = 50_000,
) -> Iterator[pl.DataFrame]:
    """Yield compact candidate batches without materializing all S1 results."""
    for offset in range(0, len(s1_df), batch_size):
        yield block_source1_against_index(
            s1_df.slice(offset, batch_size),
            index=index,
            max_candidates=max_candidates,
        )


def write_candidate_outputs_from_batches(
    candidate_batches: Iterable[pl.DataFrame],
    s1_df: pl.DataFrame,
    pool_df: pl.DataFrame,
    candidate_tsv_path: str | Path = "output/candidate_pairs.tsv",
    pair_cache_dir: str | Path = "cache/candidates/test_pairs",
    batch_size: int = 50_000,
    force: bool = False,
) -> tuple[Path, Path, int]:
    """Write compact TSV and Parquet parts as candidate batches arrive."""
    candidate_tsv_path = Path(candidate_tsv_path)
    pair_cache_dir = Path(pair_cache_dir)
    candidate_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    pair_cache_dir.mkdir(parents=True, exist_ok=True)

    if candidate_tsv_path.exists() and not force:
        raise FileExistsError(
            f"Candidate TSV already exists: {candidate_tsv_path}. "
            "Pass force=True to replace it."
        )

    existing_parts = sorted(pair_cache_dir.glob("part-*.parquet"))
    if existing_parts and not force:
        raise FileExistsError(
            f"Candidate cache already contains Parquet parts: {pair_cache_dir}. "
            "Pass force=True to replace them."
        )
    if force:
        for part in existing_parts:
            part.unlink()

    pair_count = 0
    part_count = 0
    with candidate_tsv_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("source1_entity_id\tcandidate_entity_ids\n")

        for batch_number, candidate_chunk in enumerate(candidate_batches, start=1):
            for source1_id, candidate_ids in candidate_chunk.iter_rows():
                output.write(f"{source1_id}\t{candidate_ids or ''}\n")
            pair_chunk = build_candidate_id_pairs(candidate_chunk)
            if len(pair_chunk) == 0:
                continue

            part_path = pair_cache_dir / f"part-{part_count:05d}.parquet"
            pair_chunk.write_parquet(part_path)
            pair_count += len(pair_chunk)
            part_count += 1
            print(
                f"   Wrote batch {batch_number}: {len(candidate_chunk)} S1 rows, "
                f"{len(pair_chunk)} candidate pairs"
            )

    return candidate_tsv_path, pair_cache_dir, pair_count
