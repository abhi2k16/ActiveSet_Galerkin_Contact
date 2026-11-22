"""Package skeleton: every module imports without side effects, and the wrappers reach the engines."""

import importlib

import pytest

MODULES = ["certcorrect", "certcorrect.problems", "certcorrect.fom", "certcorrect.rom", "certcorrect.bases",
           "certcorrect.environment", "certcorrect.config", "certcorrect.certificate", "certcorrect.corrections",
           "certcorrect.active_set", "certcorrect.multipliers", "certcorrect.offline_online", "certcorrect.verified",
           "certcorrect.results_io", "certcorrect.plotting", "certcorrect.cli", "certcorrect.baselines",
           "certcorrect.baselines.zhang_slack", "certcorrect.baselines.enriched_galerkin",
           "certcorrect.baselines.conformal"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_environment_records_engine_versions():
    from certcorrect.environment import environment_info

    info = environment_info()
    for key in ("fea_engine", "rom_engine", "certcorrect", "numpy", "scipy"):
        assert info.get(key), key


def test_make_problem_rejects_unknown_name():
    from certcorrect.config import make_problem

    with pytest.raises(ValueError):
        make_problem("no-such-problem")
