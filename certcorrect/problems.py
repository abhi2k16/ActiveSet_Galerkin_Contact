"""Parametrised frictionless contact benchmarks (L1 level of FINAL_RESEARCH_PLAN.md, Sec. 5), assembled with
``fea_engine``. This is the only module that builds meshes and stiffness matrices.

Convention: ``K(mu) u - B^T lam = f(mu)``, ``g = B u - c(mu) >= 0``, ``lam >= 0``, ``lam . g = 0``; ``K`` SPD on the free
unknowns, ``B`` a node selection (one ``+1`` per row, distinct columns), so it has full row rank.

Every stiffness is stored in affine form ``K(mu) = sum_q theta_q(mu) K_q`` with ``theta_q >= 0`` (needed by the
min-theta coercivity bound of ``certificate.min_theta_alpha``):
  * ``Membrane``                   : Poisson membrane (``Quad4Laplace``), moving Gaussian load, curved rigid floor;
                                     one term, ``theta = (1,)``. Parameters (peak load, centre x, centre y).
  * ``ElasticBlock``               : 3D linear-elastic block (``Hex8Solid3D``), lateral faces clamped, Gaussian pressure
                                     on the top, curved foundation with moving centre; one term.
                                     Parameters (pressure, foundation x, foundation y).
  * ``ElasticBlock(layered=True)`` : two layers, ``K = K_lower + rho K_upper``, ``theta = (1, rho)``; first parameter rho.
Geometry, loads and parameter ranges are the M2 definitions (re-implemented here, not imported), so the results of
the two projects are comparable.
"""

import warnings

import numpy as np
import scipy.sparse as sp

from fea_engine import FESystem, Hex8Solid3D, Quad4Laplace
from fea_engine.material import D_solid3d, Material
from fea_engine.mesh import MultiBlockMesh, box_mesh, rectangle_mesh

GEOMETRY_TOL = 1e-9                   # coordinate tolerance for picking boundary nodes on the unit grids

# Out-of-range queries (X5): load scaled by a factor in OUT_LOAD_FACTOR times the training maximum ("load
# extrapolation"), or load / foundation centre moved up to OUT_CENTRE_PAD outside its range ("patch shift"); for the
# layered block the stiffness ratio is in addition drawn from OUT_RATIO_FACTOR times its training maximum.
OUT_LOAD_FACTOR = (1.3, 2.0)
OUT_CENTRE_PAD = 0.15
OUT_RATIO_FACTOR = (1.2, 2.0)
OUT_MODES = ("load", "shift")


class ContactProblem:
    """Interface shared by the benchmarks."""

    name: str
    n_dof: int
    n_contact: int
    B: sp.csr_matrix
    lo: np.ndarray                    # training box of the parameters
    hi: np.ndarray
    param_names: tuple
    stiffness_terms: tuple            # K_q (sparse, n_dof x n_dof)
    load_axis: int                    # parameter scaled in load extrapolation
    centre_axes: tuple                # parameters moved in a patch shift

    # ---- affine stiffness
    def theta(self, mu: np.ndarray) -> np.ndarray:
        """Non-negative coefficients ``theta_q(mu)`` of ``K(mu) = sum_q theta_q K_q``."""
        raise NotImplementedError

    @property
    def k_varies(self) -> bool:
        return len(self.stiffness_terms) > 1

    def stiffness(self, mu: np.ndarray) -> sp.csr_matrix:
        th = self.theta(mu)
        K = th[0] * self.stiffness_terms[0]
        for coeff, term in zip(th[1:], self.stiffness_terms[1:]):
            K = K + coeff * term
        return K.tocsr()

    # ---- data
    def load(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def offset(self, mu: np.ndarray) -> np.ndarray:
        """Constraint offset ``c(mu)`` (obstacle height seen by the constrained unknowns)."""
        raise NotImplementedError

    @property
    def char_gap(self) -> float:
        """Characteristic gap scale: ``||c||`` at the centre of the training box."""
        return float(np.linalg.norm(self.offset(0.5 * (self.lo + self.hi))))

    # ---- sampling
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` parameters, uniform in the training box."""
        return self.lo + (self.hi - self.lo) * rng.random((n, len(self.lo)))

    test_lo: np.ndarray | None = None  # test box, if different from the training box (lift-off benchmark)
    test_hi: np.ndarray | None = None

    def sample_test(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` test parameters: uniform in the test box if one is set, else in the training box."""
        if self.test_lo is None:
            return self.sample(rng, n)
        return self.test_lo + (self.test_hi - self.test_lo) * rng.random((n, len(self.test_lo)))

    def in_box(self, mus: np.ndarray) -> np.ndarray:
        return np.all((mus >= self.lo) & (mus <= self.hi), axis=1)

    def sample_outside(self, rng: np.random.Generator, n: int, mode: str) -> np.ndarray:
        """``n`` parameters outside the training box: ``mode="load"`` (load extrapolation) or ``"shift"`` (centre
        moved out of its range, by rejection)."""
        if mode not in OUT_MODES:
            raise ValueError(f"mode must be one of {OUT_MODES}, got {mode!r}")
        out = self.sample(rng, n)
        if mode == "load":
            out[:, self.load_axis] = self.hi[self.load_axis] * rng.uniform(*OUT_LOAD_FACTOR, size=n)
        else:
            axes = list(self.centre_axes)
            for i in range(n):
                while True:
                    cand = rng.uniform(self.lo[axes] - OUT_CENTRE_PAD, self.hi[axes] + OUT_CENTRE_PAD)
                    if np.any((cand < self.lo[axes]) | (cand > self.hi[axes])):
                        out[i, axes] = cand
                        break
        return self._extend_outside(rng, out)

    def _extend_outside(self, rng: np.random.Generator, mus: np.ndarray) -> np.ndarray:
        return mus


def selection_matrix(positions: np.ndarray, n_dof: int) -> sp.csr_matrix:
    """Rows pick the unknowns at ``positions`` (one ``+1`` per row)."""
    m = len(positions)
    if len(np.unique(positions)) != m:
        raise ValueError("selection positions must be distinct (B must have full row rank)")
    return sp.csr_matrix((np.ones(m), (np.arange(m), positions)), shape=(m, n_dof))


def _restrict(K, free: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(K)[free][:, free]


def _on_square_edge(xy: np.ndarray) -> np.ndarray:
    """Nodes on the boundary of the unit square (first two coordinates)."""
    x, y = xy[:, 0], xy[:, 1]
    return (x < GEOMETRY_TOL) | (x > 1 - GEOMETRY_TOL) | (y < GEOMETRY_TOL) | (y > 1 - GEOMETRY_TOL)


# ---------------------------------------------------------------------------------------------------------- membrane

class Membrane(ContactProblem):
    """Unit-square membrane on an ``(n + 1)^2`` node grid, clamped edge, ``-Laplace(u) = F``, downward Gaussian load
    ``F = mu_0 exp(-|x - (mu_1, mu_2)|^2 / (2 sigma^2))``. Floor ``-(d0 + kappa r^2)``, ``r`` from the plate centre;
    every ``stride``-th interior node (in both directions) may touch it."""

    def __init__(self, n: int = 25, stride: int = 2, d0: float = 0.3, kappa: float = 1.5, sigma: float = 0.12,
                 load_range=(30.0, 200.0), centre_range=(0.35, 0.65), train_x_range=None, test_x_range=None,
                 name: str = "membrane"):
        """``train_x_range`` / ``test_x_range``: if given, the load centre's x-coordinate of training / test queries is
        drawn from these intervals instead of ``centre_range`` (lift-off benchmark: the true contact patch of a test
        query lies where the training snapshots had none)."""
        if n < 2 or stride < 1:
            raise ValueError("need n >= 2 and stride >= 1")
        self.name = name
        self.n, self.stride, self.d0, self.kappa, self.sigma = n, stride, d0, kappa, sigma
        mesh = rectangle_mesh(1.0, 1.0, n, n)
        fes = FESystem(mesh, Quad4Laplace(), sparse=True)
        fes.assemble_stiffness(1.0)
        fes.assemble_mass(1.0)
        fes.fix_dofs(np.flatnonzero(_on_square_edge(mesh.nodes)), ["u"])
        free = np.asarray(fes.free_dofs)
        self.mesh, self.free = mesh, free
        self.stiffness_terms = (_restrict(fes.K, free),)
        self._M = _restrict(fes.M, free)              # consistent mass: load vector of the Gaussian field
        self.xy = mesh.nodes[free]
        self.n_dof = len(free)
        grid = np.rint(self.xy * n).astype(int)
        chosen = np.flatnonzero((grid[:, 0] % stride == 0) & (grid[:, 1] % stride == 0))
        self.B = selection_matrix(chosen, self.n_dof)
        self.n_contact = len(chosen)
        self._offset = -(d0 + kappa * np.sum((self.xy[chosen] - 0.5) ** 2, axis=1))
        self.lo = np.array([load_range[0], centre_range[0], centre_range[0]])
        self.hi = np.array([load_range[1], centre_range[1], centre_range[1]])
        self.param_names = ("peak load", "centre x", "centre y")
        self.load_axis, self.centre_axes = 0, (1, 2)
        if train_x_range is not None:
            self.lo[1], self.hi[1] = train_x_range
        if test_x_range is not None:
            self.test_lo, self.test_hi = self.lo.copy(), self.hi.copy()
            self.test_lo[1], self.test_hi[1] = test_x_range

    def theta(self, mu):
        return np.ones(1)

    def load(self, mu):
        r2 = np.sum((self.xy - np.asarray(mu[1:3])) ** 2, axis=1)
        return -(self._M @ (mu[0] * np.exp(-r2 / (2 * self.sigma**2))))

    def offset(self, mu):
        return self._offset


# ----------------------------------------------------------------------------------------------------- elastic block

class ElasticBlock(ContactProblem):
    """Block ``[0,1]^2 x [0, depth]``, lateral faces clamped, Gaussian pressure ``P`` downward on the top face; bottom
    interior nodes may touch the foundation ``z >= -(d0 + kappa ((x - xc)^2 + (y - yc)^2))``.

    Parameters ``(P, xc, yc)``; with ``layered``, ``(rho, P, xc, yc)``, ``rho = E_upper / E_lower``, the layers split at
    ``depth / 2``."""

    def __init__(self, n_xy: int = 16, nz: int = 4, depth: float = 0.5, nu: float = 0.3, layered: bool = False,
                 d0: float = 0.02, kappa: float = 1.0, sigma: float = 0.3, load_range=(0.10, 0.60),
                 centre_range=(0.35, 0.65), ratio_range=(0.5, 2.0)):
        if layered and nz < 2:
            raise ValueError("a layered block needs nz >= 2")
        self.name = "layered" if layered else "block"
        self.layered, self.n_xy, self.nz, self.depth, self.nu = layered, n_xy, nz, depth, nu
        self.d0, self.kappa, self.sigma = d0, kappa, sigma
        mesh = box_mesh(1.0, 1.0, depth, n_xy, n_xy, nz)
        x = mesh.nodes
        D = D_solid3d(Material(E=1.0, nu=nu, rho=1.0))
        side = np.flatnonzero(_on_square_edge(x))
        self.stiffness_terms, free = self._assemble(mesh, D, side)
        self.mesh, self.free, self.n_dof = mesh, free, len(free)
        position = -np.ones(3 * len(x), dtype=int)                # global dof -> free index
        position[free] = np.arange(len(free))
        interior = np.setdiff1d(np.arange(len(x)), side)
        bottom = interior[x[interior, 2] < GEOMETRY_TOL]
        top = interior[x[interior, 2] > depth - GEOMETRY_TOL]
        self.xy_contact = x[bottom, :2]
        self.n_contact = len(bottom)
        self.B = selection_matrix(position[3 * bottom + 2], self.n_dof)
        self._top_dof = position[3 * top + 2]
        h = 1.0 / n_xy                                             # lumped nodal area of the top face
        self._top_weight = np.exp(-np.sum((x[top, :2] - 0.5) ** 2, axis=1) / (2 * sigma**2)) * h * h
        lo, hi = [load_range[0], centre_range[0], centre_range[0]], [load_range[1], centre_range[1], centre_range[1]]
        names = ("pressure", "foundation x", "foundation y")
        if layered:
            lo, hi, names = [ratio_range[0]] + lo, [ratio_range[1]] + hi, ("stiffness ratio",) + names
        self.lo, self.hi, self.param_names = np.array(lo), np.array(hi), names
        p0 = 1 if layered else 0
        self.load_axis, self.centre_axes = p0, (p0 + 1, p0 + 2)

    def _assemble(self, mesh, D, side):
        """Stiffness terms on the free unknowns: ``(K,)`` or ``(K_lower, K_upper)``."""
        if not self.layered:
            fes = FESystem(mesh, Hex8Solid3D(), sparse=True)
            fes.fix_dofs(side, [0, 1, 2])
            free = np.asarray(fes.free_dofs)
            fes.assemble_stiffness(D)
            return (_restrict(fes.K, free),), free
        zc = mesh.nodes[mesh.elements, 2].mean(axis=1)
        upper = zc > self.depth / 2
        layered_mesh = MultiBlockMesh(nodes=mesh.nodes, blocks={"lower": mesh.elements[~upper],
                                                                 "upper": mesh.elements[upper]}, dim=3)
        fes = FESystem(layered_mesh, {"lower": Hex8Solid3D(), "upper": Hex8Solid3D()}, sparse=True)
        fes.fix_dofs(side, [0, 1, 2])
        free = np.asarray(fes.free_dofs)
        terms = []
        with warnings.catch_warnings():            # re-assembly replaces K on purpose (one layer at a time)
            warnings.simplefilter("ignore", UserWarning)
            for lower, upper_scale in ((1.0, 0.0), (0.0, 1.0)):
                fes.assemble_stiffness({"lower": lower * D, "upper": upper_scale * D})
                terms.append(_restrict(fes.K, free))
        return tuple(terms), free

    def theta(self, mu):
        return np.array([1.0, mu[0]]) if self.layered else np.ones(1)

    def load(self, mu):
        f = np.zeros(self.n_dof)
        f[self._top_dof] = -mu[self.load_axis] * self._top_weight
        return f

    def offset(self, mu):
        centre = np.asarray(mu[self.centre_axes[0]:self.centre_axes[1] + 1])
        return -(self.d0 + self.kappa * np.sum((self.xy_contact - centre) ** 2, axis=1))

    def _extend_outside(self, rng, mus):
        if self.layered:
            mus[:, 0] = self.hi[0] * rng.uniform(*OUT_RATIO_FACTOR, size=len(mus))
        return mus
