"""Unit tests for the blocking module."""

import unittest
from pathlib import Path
import tempfile
import polars as pl

from business_entity_resolution.src.blocking import (
    BlockingIndex,
    block_source1_against_index,
    build_candidate_pairs_table,
    extract_address_keys,
    extract_all_blocking_keys,
    extract_name_keys,
    iter_candidate_pair_batches,
    save_candidate_pairs,
)
from business_entity_resolution.src.normalization import normalize_address, normalize_name


class TestBlocking(unittest.TestCase):
    def test_name_keys_basic_and_suffix(self):
        # Legal suffix 'Inc.' should be filtered out so both share NF_prim and NS_mone
        n1 = normalize_name("Prime Money")
        n2 = normalize_name("Prime Money Inc.")
        k1 = set(extract_name_keys(n1))
        k2 = set(extract_name_keys(n2))
        
        self.assertTrue(k1.intersection(k2), f"Expected intersection between {k1} and {k2}")
        self.assertIn("NF_prim", k1)
        self.assertIn("NF_prim", k2)

    def test_name_keys_word_transposition(self):
        # 'Orelee's Barbershop' vs 'Barbershop Orelee's'
        n1 = normalize_name("Orelee's Barbershop")
        n2 = normalize_name("Barbershop Orelee's")
        k1 = set(extract_name_keys(n1))
        k2 = set(extract_name_keys(n2))
        
        # Word order differs, but sorted key NS_barb matches both!
        self.assertTrue(k1.intersection(k2), f"Expected transposition match: {k1} vs {k2}")
        self.assertIn("NS_barb", k1)
        self.assertIn("NS_barb", k2)

    def test_name_keys_french_accents(self):
        n1 = normalize_name("Café-Français & Fils S.A.R.L.")
        keys = extract_name_keys(n1)
        self.assertTrue(len(keys) > 0)
        self.assertIn("NF_café", keys)

    def test_name_keys_non_latin(self):
        n1 = normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
        keys = extract_name_keys(n1)
        self.assertTrue(len(keys) > 0)
        self.assertTrue(any(k.startswith("NF_") for k in keys))

    def test_address_keys_leading_zeros(self):
        # '0017560 ELLIS ROAD' vs '17560 Ellis Rd'
        a1 = normalize_address("TAHLEQUAH, OK, 0017560 ELLIS ROAD")
        a2 = normalize_address("17560 Ellis Rd, Tahlequah, Oklahoma")
        k1 = set(extract_address_keys(a1))
        k2 = set(extract_address_keys(a2))
        
        common = k1.intersection(k2)
        self.assertTrue(len(common) > 0, f"Expected common address keys: {k1} vs {k2}")
        self.assertIn("ANW_17560_elli", common)

    def test_address_keys_empty_or_no_number(self):
        # Empty address
        self.assertEqual(extract_address_keys(""), [])
        self.assertEqual(extract_address_keys(None), [])
        
        # Address without number: fallback to word pair
        a = normalize_address("Lake Town Block A, Kolkata, West Bengal")
        keys = extract_address_keys(a)
        self.assertTrue(any(k.startswith("AWP_") for k in keys))

    def test_country_isolation_in_blocking(self):
        index = BlockingIndex()
        # Add S2 records for US and India with identical business names
        s2_records = pl.DataFrame({
            "entity_id": ["S2-US-1", "S2-IN-1"],
            "country": ["US", "India"],
            "business_name": ["Apex Enterprises", "Apex Enterprises"],
            "business_address": ["100 Main St, Austin, TX", "100 Main St, Mumbai, MH"],
        })
        index.add_dataframe(s2_records)

        # Query with S1 record from US
        cands_us = index.query("US", normalize_name("Apex Enterprises"), "")
        self.assertIn("S2-US-1", cands_us)
        self.assertNotIn("S2-IN-1", cands_us, "US query must never retrieve India records!")

        # Query with S1 record from India
        cands_in = index.query("India", normalize_name("Apex Enterprises"), "")
        self.assertIn("S2-IN-1", cands_in)
        self.assertNotIn("S2-US-1", cands_in, "India query must never retrieve US records!")

    def test_candidate_capping(self):
        index = BlockingIndex()
        # Create 10 dummy S2 records that all share the same key
        s2_records = pl.DataFrame({
            "entity_id": [f"S2-{i}" for i in range(10)],
            "country": ["US"] * 10,
            "business_name": ["Super Global Store"] * 10,
            "business_address": ["100 Oak St"] * 10,
        })
        index.add_dataframe(s2_records)

        # Capped query with max_candidates=3
        cands = index.query("US", normalize_name("Super Global Store"), "", max_candidates=3)
        self.assertEqual(len(cands), 3)

    def test_end_to_end_candidate_df_and_save(self):
        index = BlockingIndex()
        s2_records = pl.DataFrame({
            "entity_id": ["S2-101", "S2-102"],
            "country": ["US", "US"],
            "business_name": ["Alpha Retail Inc", "Beta Technologies"],
            "business_address": ["500 Elm Street, Dallas, TX", "600 Pine Street, Austin, TX"],
        })
        index.add_dataframe(s2_records)

        s1_records = pl.DataFrame({
            "entity_id": ["S1-001", "S1-002", "S1-SINGLETON"],
            "country": ["US", "US", "US"],
            "business_name": ["Alpha Retail", "Beta Tech", "Unique Unmatched Corp"],
            "business_address": ["500 Elm St, Dallas", "600 Pine St, Austin", "999 Nowhere Rd"],
        })

        cand_df = block_source1_against_index(s1_records, index, max_candidates=5)
        self.assertEqual(cand_df.columns, ["source1_entity_id", "candidate_entity_ids"])
        self.assertEqual(len(cand_df), 3)

        rows = dict(cand_df.iter_rows())
        self.assertIn("S2-101", rows["S1-001"])
        self.assertIn("S2-102", rows["S1-002"])
        self.assertEqual(rows["S1-SINGLETON"], "")  # singleton is empty string

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "candidate_pairs.tsv"
            save_candidate_pairs(cand_df, out_file)
            self.assertTrue(out_file.exists())
            
            # Read back as raw text to verify tab separation
            with open(out_file, encoding="utf-8") as f:
                lines = [line.strip().split("\t") for line in f if line.strip()]
            self.assertEqual(lines[0], ["source1_entity_id", "candidate_entity_ids"])
            self.assertEqual(len(lines), 4)

    def test_build_candidate_pairs_table(self):
        s1 = pl.DataFrame({
            "entity_id": ["S1-001", "S1-002", "S1-003"],
            "business_name": ["alpha cafe", "beta store", "gamma single"],
            "business_address": ["10 main street", "20 oak ave", "30 pine road"],
            "country": ["US", "US", "US"],
        })
        pool = pl.DataFrame({
            "entity_id": ["S2-101", "S3-202", "S3-004"],
            "business_name": ["alpha café", "alpha coffee", "beta shop"],
            "business_address": ["10 main st", "12 oak road", "20 oak avenue"],
            "country": ["US", "US", "US"],
        })
        cand_df = pl.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002", "S1-003"],
            "candidate_entity_ids": ["S2-101,S3-202", "S3-004", ""],
        })

        table = build_candidate_pairs_table(cand_df, s1_df=s1, pool_df=pool)

        expected_columns = [
            "source1_entity_id",
            "candidate_id",
            "s1_name",
            "s1_address",
            "s1_country",
            "cand_name",
            "cand_address",
            "cand_country",
        ]
        self.assertEqual(table.columns, expected_columns)
        self.assertEqual(len(table), 3)  # 2 candidates for S1-001 + 1 for S1-002 = 3 pairs (singleton S1-003 excluded)

        # Verify correct row pairing
        row1 = table.filter((pl.col("source1_entity_id") == "S1-001") & (pl.col("candidate_id") == "S2-101")).to_dicts()[0]
        self.assertEqual(row1["s1_name"], "alpha cafe")
        self.assertEqual(row1["cand_name"], "alpha café")
        self.assertEqual(row1["s1_address"], "10 main street")
        self.assertEqual(row1["cand_address"], "10 main st")

    def test_iter_candidate_pair_batches(self):
        s1 = pl.DataFrame({
            "entity_id": ["S1-001", "S1-002"],
            "business_name": ["alpha cafe", "beta store"],
            "business_address": ["10 main street", "20 oak ave"],
            "country": ["US", "US"],
        })
        pool = pl.DataFrame({
            "entity_id": ["S2-101", "S3-202", "S3-004"],
            "business_name": ["alpha café", "alpha coffee", "beta shop"],
            "business_address": ["10 main st", "12 oak road", "20 oak avenue"],
            "country": ["US", "US", "US"],
        })
        cand_df = pl.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "candidate_entity_ids": ["S2-101,S3-202", "S3-004"],
        })

        # Total 3 pairs, batch_size=2 should yield 2 batches: one of 2 rows, one of 1 row
        batches = list(iter_candidate_pair_batches(cand_df, s1, pool, batch_size=2))
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 1)
        concatenated = pl.concat(batches)
        self.assertEqual(len(concatenated), 3)

    def test_dba_trade_name_splitting(self):
        """Test that DBA and 'doing business as' entities share keys with their trade name."""
        keys_alias = extract_name_keys("lyravera dba jeniece glass incorporated")
        keys_brand = extract_name_keys("jeniece glass incorporated")
        # Both must contain 'NF_jeni' (from jeniece)
        self.assertTrue(any("jeni" in k for k in keys_alias))
        self.assertTrue(any("jeni" in k for k in keys_brand))
        self.assertTrue(bool(set(keys_alias) & set(keys_brand)))

    def test_multi_number_address_matching(self):
        """Test that addresses with multiple numbers match regardless of number ordering."""
        keys_s1 = extract_address_keys("unit 229 columbus 2687 livingston avenue oh")
        keys_s3 = extract_address_keys("2687 livingston avenue unit 229 columbus oh")
        # Both have 229 and 2687, and should share ANW_ keys
        common_keys = set(keys_s1) & set(keys_s3)
        self.assertTrue(len(common_keys) > 0)

    def test_number_vs_no_number_universal_awp(self):
        """Test that an address with a number shares AWP keys with one without a number."""
        keys_num = extract_address_keys("306 wrights lane bldg brian griffith dentistry prestonsburg ky")
        keys_no_num = extract_address_keys("prestonsburg bldg brian griffith dentistry kentucky wrights lane")
        # Universal AWP ensures both generate word pair keys from distinctive words
        common_keys = set(keys_num) & set(keys_no_num)
        self.assertTrue(len(common_keys) > 0)
        self.assertTrue(any(k.startswith("AWP_") for k in common_keys))

    def test_honorific_and_country_stopwords(self):
        """Test that Indian honorifics (Smt, Shri) and generic countries do not become primary keys."""
        keys = extract_name_keys("smt radhika traders india private limited")
        # 'smt', 'india', 'private', 'limited' are filtered, leaving 'radhika' and 'traders'
        self.assertFalse(any(k.endswith("_smt") or k.endswith("_indi") for k in keys))
        self.assertTrue(any(k.endswith("_radh") for k in keys))


if __name__ == "__main__":
    unittest.main()

