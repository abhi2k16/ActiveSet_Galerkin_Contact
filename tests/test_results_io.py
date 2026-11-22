"""Result files round-trip arrays and metadata; LaTeX macros are generated, not typed."""

import numpy as np
import pytest

from certcorrect.results_io import load_result, save_result, write_tex_macros


def test_round_trip(tmp_path):
    path = save_result(tmp_path / "x" / "r.npz", {"err": np.arange(3.0)}, {"problem": "membrane", "seed": 0})
    arrays, record = load_result(path)
    np.testing.assert_array_equal(arrays["err"], np.arange(3.0))
    assert record["meta"]["problem"] == "membrane"
    assert "fea_engine" in record["environment"]


def test_missing_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_result(tmp_path / "missing.npz")


def test_tex_macros(tmp_path):
    path = write_tex_macros(tmp_path / "numbers.tex", {"effMembrane": "1.03"})
    assert "\\newcommand{\\effMembrane}{1.03}" in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        write_tex_macros(tmp_path / "bad.tex", {"eff_1": "1"})
