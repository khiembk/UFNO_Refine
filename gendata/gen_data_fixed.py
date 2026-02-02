import os
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver


def generate_velocity_model_zx(sg_zx: np.ndarray) -> np.ndarray:
    """
    Input:  sg_zx shape (nz, nx)  (z, x)
    Output: vp_zx shape (nz, nx)  (z, x)
    """
    vp_brine = 2200.0
    vp_co2   = 1500.0
    vp = vp_brine * (1.0 - sg_zx) + vp_co2 * sg_zx

    nz, nx = vp.shape
    vp_background = np.ones((nz, nx), dtype=np.float32) * 2000.0
    vp_background[int(0.6 * nz):, :] = 2600.0  # deeper faster layer (along z)

    vp = 0.7 * vp + 0.3 * vp_background
    vp = gaussian_filter(vp, sigma=1.0)
    return vp.astype(np.float32)


def build_solver_xz(nx: int, nz: int, nbl: int, nt: int, dt: float, f0: float):
    """
    Build Devito solver in (x, z) order.
    """
    spacing = (10.0, 10.0)
    origin = (0.0, 0.0)

    # vp_init should match physical domain shape (nx, nz)
    vp_init = np.ones((nx, nz), dtype=np.float32) * 2000.0

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

    tn = (nt - 1) * dt

    # coordinates are (x, z)
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

    return model, solver


def set_vp_on_model(model, vp_xz: np.ndarray):
    """
    vp_xz is (nx, nz) for physical domain.
    We pad it to match model.vp.data shape (nx+2*nbl, nz+2*nbl).
    """
    nbl = model.nbl
    vp_pad = np.pad(vp_xz, pad_width=((nbl, nbl), (nbl, nbl)), mode="edge")
    model.vp.data[:] = vp_pad


def simulate_shot(model, solver, vp_xz: np.ndarray, nt: int, dt: float) -> np.ndarray:
    set_vp_on_model(model, vp_xz)
    rec, _, _ = solver.forward(dt=dt)

    shot = rec.data[:nt, :].astype(np.float32)
    shot /= (np.max(np.abs(shot)) + 1e-8)
    return shot


def time_derivative_central(u: np.ndarray, dt: float) -> np.ndarray:
    """
    u shape (..., nt, nrec)
    returns du/dt with same shape using central differences.
    """
    du = np.zeros_like(u, dtype=np.float32)
    du[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * dt)
    du[..., 0, :]    = (u[..., 1, :] - u[..., 0, :]) / dt
    du[..., -1, :]   = (u[..., -1, :] - u[..., -2, :]) / dt
    return du


def main():
    sg_u = torch.load("../datasets/sg_test_u.pt", map_location="cpu")
    print(f"Loaded sg_test_u.pt → shape: {tuple(sg_u.shape)}")

    # sg_u: (N, nz, nx, Tflow)
    N, nz, nx, Tflow = sg_u.shape

    # Wave sim settings
    nbl = 50
    nt = 151
    f0 = 0.025

    # --- choose a dt safe for maximum expected vp ---
    # If you know your vp max ~2600, use that worst-case for dt.
    # Build a temporary model with vmax to get a safe critical_dt.
    vmax = 2600.0
    tmp_model = Model(
        vp=np.ones((nx, nz), dtype=np.float32) * vmax,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        shape=(nx, nz),
        nbl=nbl,
        space_order=8,
        bcs="damp",
        dtype=np.float32
    )
    dt = float(tmp_model.critical_dt)  # stable for vmax
    print(f"Using fixed dt={dt:.6e} (stable for vmax={vmax})")

    model, solver = build_solver_xz(nx=nx, nz=nz, nbl=nbl, nt=nt, dt=dt, f0=f0)

    # Outputs
    u = np.zeros((N, Tflow, nt, 101), dtype=np.float32)
    vp_all = np.zeros((N, Tflow, nx, nz), dtype=np.float32)  # store R[c] in x,z order

    pbar = tqdm(total=N * Tflow, desc="Generating (u, vp)")

    for i in range(N):
        for t in range(Tflow):
            sg_zx = sg_u[i, :, :, t].numpy().astype(np.float32)  # (nz, nx)
            vp_zx = generate_velocity_model_zx(sg_zx)            # (nz, nx)

            # IMPORTANT: transpose to (nx, nz) for Devito (x, z)
            vp_xz = vp_zx.T

            vp_all[i, t] = vp_xz
            u[i, t] = simulate_shot(model, solver, vp_xz, nt=nt, dt=dt)
            pbar.update(1)

    pbar.close()

    # Compute du/dt along wave time axis
    u_t = time_derivative_central(u, dt=dt)

    torch.save(torch.from_numpy(u), "../datasets/wave_test_u.pt")
    torch.save(torch.from_numpy(u_t), "../datasets/wave_test_ut.pt")
    torch.save(torch.from_numpy(vp_all), "../datasets/wave_test_vp.pt")

    print("Saved:")
    print("  wave_test_u.pt   :", u.shape)
    print("  wave_test_ut.pt  :", u_t.shape)
    print("  wave_test_vp.pt  :", vp_all.shape)


if __name__ == "__main__":
    os.environ["TMPDIR"] = "/home/aiotlab/mnt/khiemtt/devito_tmp"
    os.environ["DEVITO_JIT_CACHE"] = "/home/aiotlab/mnt/khiemtt/devito_tmp"
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    os.makedirs(os.environ["DEVITO_JIT_CACHE"], exist_ok=True)
    os.environ["DEVITO_LOGGING"] = "ERROR"
    os.environ["OMP_NUM_THREADS"] = "8"
    main()
