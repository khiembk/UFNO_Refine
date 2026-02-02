import os
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver


# -----------------------------
# Rock physics map R[c] -> vp
# -----------------------------
def generate_velocity_model_zx(sg_zx: np.ndarray) -> np.ndarray:
    """
    sg_zx: (nz, nx)  (z, x)
    returns vp_zx: (nz, nx)
    """
    vp_brine = 2200.0
    vp_co2 = 1500.0
    vp = vp_brine * (1.0 - sg_zx) + vp_co2 * sg_zx

    nz, nx = vp.shape
    vp_background = np.ones((nz, nx), dtype=np.float32) * 2000.0
    vp_background[int(0.6 * nz):, :] = 2600.0

    vp = 0.7 * vp + 0.3 * vp_background
    vp = gaussian_filter(vp, sigma=1.0)
    return vp.astype(np.float32)


# -----------------------------
# Build solver with consistent dt/tn/nt
# -----------------------------
def build_solver_xz(nx: int, nz: int, nbl: int, nt_target: int, vmax: float,
                    spacing=(10.0, 10.0), f0=0.025):
    """
    Build Devito model/geometry/solver in (x, z) ordering.

    Key: model is initialized with vp=vmax to get a safe dt,
    then tn is computed from THAT dt so geometry.nt == nt_target.
    """
    origin = (0.0, 0.0)

    vp_init = np.ones((nx, nz), dtype=np.float32) * float(vmax)
    model = Model(
        vp=vp_init,
        origin=origin,
        spacing=spacing,
        shape=(nx, nz),
        nbl=nbl,
        space_order=8,
        bcs="damp",
        dtype=np.float32
    )

    dt = float(model.critical_dt)
    tn = (nt_target - 1) * dt

    # coords are (x, z)
    src_positions = np.array([[model.domain_size[0] / 2.0, 20.0]], dtype=np.float32)

    n_receivers = 101
    rec_x = np.linspace(20.0, model.domain_size[0] - 20.0, n_receivers).astype(np.float32)
    rec_positions = np.zeros((n_receivers, 2), dtype=np.float32)
    rec_positions[:, 0] = rec_x
    rec_positions[:, 1] = 20.0

    geometry = AcquisitionGeometry(
        model=model,
        rec_positions=rec_positions,
        src_positions=src_positions,
        t0=0.0,
        tn=tn,
        f0=f0,
        src_type="Ricker"
    )

    solver = AcousticWaveSolver(
        model=model,
        geometry=geometry,
        space_order=8,
        kernel="OT2"
    )

    # sanity
    if geometry.nt != nt_target:
        print(f"[WARN] geometry.nt={geometry.nt} != nt_target={nt_target} (dt={dt}, tn={tn})")

    return model, solver, dt, geometry.nt, n_receivers


# -----------------------------
# Update vp on model (pad to include nbl)
# -----------------------------
def set_vp_on_model(model, vp_xz: np.ndarray):
    """
    vp_xz: (nx, nz) in physical domain.
    model.vp.data is padded (nx+2*nbl, nz+2*nbl).
    """
    nbl = model.nbl
    vp_pad = np.pad(vp_xz, pad_width=((nbl, nbl), (nbl, nbl)), mode="edge")
    model.vp.data[:] = vp_pad


def simulate_shot(model, solver, vp_xz: np.ndarray, nt: int) -> np.ndarray:
    set_vp_on_model(model, vp_xz)

    # IMPORTANT: do NOT override dt here; use geometry's dt/nt
    rec, _, _ = solver.forward()

    shot = rec.data[:nt, :].astype(np.float32)  # (nt, nrec)
    shot /= (np.max(np.abs(shot)) + 1e-8)
    return shot


def time_derivative_central(u: np.ndarray, dt: float) -> np.ndarray:
    """
    u: (..., nt, nrec)  -> du/dt with same shape
    """
    du = np.zeros_like(u, dtype=np.float32)
    du[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * dt)
    du[..., 0, :] = (u[..., 1, :] - u[..., 0, :]) / dt
    du[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / dt
    return du


def main():
    sg_u = torch.load("../datasets/sg_test_u.pt", map_location="cpu")
    print(f"Loaded sg_test_u.pt → shape: {tuple(sg_u.shape)}")

    # sg_u: (N, nz, nx, Tflow)
    N, nz, nx, Tflow = sg_u.shape

    # config
    nt_target = 151
    nbl = 50
    vmax = 2600.0   # worst-case vp (match your background max)
    spacing = (10.0, 10.0)
    f0 = 0.025

    # Build solver in (x,z) with consistent dt/tn/nt
    model, solver, dt, nt, nrec = build_solver_xz(
        nx=nx, nz=nz, nbl=nbl, nt_target=nt_target, vmax=vmax,
        spacing=spacing, f0=f0
    )
    print(f"Using dt={dt:.6e}, nt={nt}, nrec={nrec}")

    # Allocate arrays using the REAL nt
    u = np.zeros((N, Tflow, nt, nrec), dtype=np.float32)
    vp_all = np.zeros((N, Tflow, nx, nz), dtype=np.float32)  # store vp in (x,z)

    pbar = tqdm(total=N * Tflow, desc="Generating (u, vp)")

    for i in range(N):
        for t in range(Tflow):
            sg_zx = sg_u[i, :, :, t].numpy().astype(np.float32)     # (nz, nx)
            vp_zx = generate_velocity_model_zx(sg_zx)               # (nz, nx)

            # Devito model is (x,z) -> transpose
            vp_xz = vp_zx.T                                         # (nx, nz)
            vp_all[i, t] = vp_xz

            u[i, t] = simulate_shot(model, solver, vp_xz, nt=nt)
            pbar.update(1)

    pbar.close()

    # du/dt along wave time axis
    u_t = time_derivative_central(u, dt=dt)

    torch.save(torch.from_numpy(u), "../datasets/wave_test_u.pt")
    torch.save(torch.from_numpy(u_t), "../datasets/wave_test_ut.pt")
    torch.save(torch.from_numpy(vp_all), "../datasets/wave_test_vp.pt")

    print("Saved:")
    print("  seismic_test_u.pt  :", u.shape)
    print("  seismic_test_ut.pt :", u_t.shape)
    print("  seismic_test_vp.pt :", vp_all.shape)


if __name__ == "__main__":
    os.environ["TMPDIR"] = "/home/aiotlab/mnt/khiemtt/devito_tmp"
    os.environ["DEVITO_JIT_CACHE"] = "/home/aiotlab/mnt/khiemtt/devito_tmp"
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    os.makedirs(os.environ["DEVITO_JIT_CACHE"], exist_ok=True)
    os.environ["DEVITO_LOGGING"] = "ERROR"
    os.environ["OMP_NUM_THREADS"] = "8"
    main()
