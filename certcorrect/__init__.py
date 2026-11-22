"""Certified post-solve correction of Galerkin reduced-order contact models.

Layering (Guidlines/SCRIPT_GUIDELINES.md, Sec. 6): the engine packages ``fea_engine`` and ``rom_engine`` are imported
only in the wrapper modules ``problems``, ``fom``, ``rom``, ``bases`` and ``environment``. No package of another
research project is imported. Method modules (``corrections``, ``active_set``, ``multipliers``, ``certificate``,
``offline_online``, ``verified``, ``baselines``) work on NumPy/SciPy arrays. Experiment scripts import from
``certcorrect`` only.
"""

__version__ = "0.1.0"
