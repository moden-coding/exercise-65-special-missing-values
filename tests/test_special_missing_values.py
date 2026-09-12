#!/usr/bin/env python3
"""Tests for the Special Missing Values pandas assignment."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.special_missing_values import main, special_missing_values


class TestSpecialMissingValues(unittest.TestCase):
    """special_missing_values() -> UK top 40 chart data with real NaNs."""

    def test_shape(self):
        df = special_missing_values()
        self.assertEqual(
            df.shape,
            (17, 7),
            msg="special_missing_values() should return a DataFrame with "
            "shape (17, 7). Got shape %r." % (df.shape,),
        )

    def test_columns(self):
        df = special_missing_values()
        np.testing.assert_array_equal(
            df.columns,
            ["Pos", "LW", "Title", "Artist", "Publisher", "Peak Pos", "WoC"],
            err_msg="The DataFrame's columns should be ['Pos', 'LW', "
            "'Title', 'Artist', 'Publisher', 'Peak Pos', 'WoC'], in this "
            "order.",
        )

    def test_main_calls_special_missing_values_and_reads_a_csv(self):
        with patch(
            "src.special_missing_values.special_missing_values",
            wraps=special_missing_values,
        ) as psmv, patch(
            "src.special_missing_values.pd.read_csv", wraps=pd.read_csv
        ) as prc:
            main()
            psmv.assert_called()
            prc.assert_called()

    def test_content(self):
        df = special_missing_values()
        np.testing.assert_array_equal(
            df["Pos"],
            [3, 4, 6, 9, 10, 12, 15, 16, 21, 22, 24, 30, 31, 34, 35, 38, 39],
            err_msg="The values in the 'Pos' column were incorrect: the rows "
            "with special/placeholder missing-value markers should have "
            "been treated as real missing data, not dropped or kept as "
            "text.",
        )


if __name__ == "__main__":
    unittest.main()
