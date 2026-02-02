import os
import argparse
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver


# -----------------------------
# Rock physics: R[c] -> vp
# -----------------------------
def generate_velocity_model_zx(
    sg_zx: np.ndarray,
    vp_brine: float = 2200.0,
    vp_co2: float = 1500.0,
    bg_top: float = 2000.0,
    bg_bot: float = 2600.0,
    bg_interface_frac: float = 0.6,
    mix_sat: float = 0.7,
    smooth_sigma: float = 1.0
) -> np.ndarray:
    """
    sg_zx: (nz, nx) (z, x) saturation in [0,1]
    returns vp_zx: (nz, nx)
    """
    vp = vp_brine * (1.0 - sg_zx) + vp_co2 * sg_zx

    nz, nx = vp.shape
    vp_background = np.ones((nz, nx), dtype=np.float32) * float(bg_top)
    vp_background[int(bg_interface_frac * nz):, :] = float(bg_bot)

    vp = float(mix_sat) * vp + (1.0 - float(mix_sat)) * vp_background
    vp = gaussian_filter(vp, sigma=float(smooth_sigma))
    return vp.astype(np.float32)


# -----------------------------
# Build solver for a target tn
# -----------------------------
def build_solver_xz_with_tn(
    nx: int,
    nz: int,
    nbl: int,
    tn: float,
    vmax: float,
    spacing=(10.0, 10.0),
    f0: float = 0.025,
    space_order: int = 8,
    kernel: str = "OT2",
):
    """
    Build model/geometry/solver in (x,z) ordering.
    geometry duration is tn (seconds), so geometry.nt is determined internally.
    """
    origin = (0.0, 0.0)
    spacing = (float(spacing[0]), float(spacing[1]))

    # Use vmax so critical_dt is safe for all vp (assuming vp <= vmax)
    vp_init = np.ones((nx, nz), dtype=np.float32) * float(vmax)
    model = Model(
        vp=vp_init,
        origin=origin,
        spacing=spacing,
        shape=(nx, nz),
        nbl=nbl,
        space_order=space_order,
        bcs="damp",
        dtype=np.float32
    )

    # geometry (x,z)
    src_positions = np.array([[model.domain_size[0] / 2.0, 20.0]], dtype=np.float32)

    nrec = 101
    rec_x = np.linspace(20.0, model.domain_size[0] - 20.0, nrec).astype(np.float32)
    rec_positions = np.zeros((nrec, 2), dtype=np.float32)
    rec_positions[:, 0] = rec_x
    rec_positions[:, 1] = 20.0

    geometry = AcquisitionGeometry(
        model=model,
        rec_positions=rec_positions,
        src_positions=src_positions,
        t0=0.0,
        tn=float(tn),
        f0=float(f0),
        src_type="Ricker"
    )

    solver = AcousticWaveSolver(
        model=model,
        geometry=geometry,
        space_order=space_order,
        kernel=kernel
    )

    return model, solver, nrec


# -----------------------------
# Update vp (pad to include nbl)
# -----------------------------
def set_vp_on_model(model, vp_xz: np.ndarray):
    """
    vp_xz: (nx, nz) physical domain. model.vp.data is padded.
    """
    nbl = model.nbl
    vp_pad = np.pad(vp_xz, ((nbl, nbl), (nbl, nbl)), mode="edge")
    model.vp.data[:] = vp_pad


def pad_or_trim_time(shot: np.ndarray, nt_out: int) -> np.ndarray:
    """
    shot: (nt, nrec) -> enforce (nt_out, nrec)
    """
    nt, nrec = shot.shape
    if nt == nt_out:
        return shot
    if nt > nt_out:
        return shot[:nt_out, :]
    out = np.zeros((nt_out, nrec), dtype=shot.dtype)
    out[:nt, :] = shot
    return out


# -----------------------------
# Fine sim -> coarse sampling
# -----------------------------
def simulate_shot_coarse(
    model,
    solver,
    vp_xz: np.ndarray,
    dt_sim: float,
    k: int,
    nt_out: int,
    normalize: bool = True
) -> np.ndarray:
    """
    Simulate with fine dt_sim, then sample every k steps.
    Output timestep is dt_out = k*dt_sim and output length nt_out.

    Returns: (nt_out, nrec)
    """
    set_vp_on_model(model, vp_xz)

    # Fine simulation (stable dt_sim)
    rec, _, _ = solver.forward(dt=float(dt_sim))
    shot_fine = rec.data.astype(np.float32)  # (nt_fine, nrec)

    if normalize:
        shot_fine /= (np.max(np.abs(shot_fine)) + 1e-8)

    # Coarse sampling: take every k-th time sample
    shot_coarse = shot_fine[::k, :]

    # Enforce exact nt_out length
    shot_coarse = pad_or_trim_time(shot_coarse, nt_out)
    return shot_coarse


def coarse_time_derivative(u: np.ndarray, dt_out: float) -> np.ndarray:
    """
    u: (..., nt_out, nrec)
    returns du/dt using forward difference on coarse grid:
      du[n] = (u[n+1] - u[n]) / dt_out
    Last timestep du[-1] = 0
    """
    du = np.zeros_like(u, dtype=np.float32)
    du[..., :-1, :] = (u[..., 1:, :] - u[..., :-1, :]) / float(dt_out)
    du[..., -1, :] = 0.0
    return du


# -----------------------------
# Main generation
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--sg_path", type=str, default="../datasets/sg_test_u.pt")
    parser.add_argument("--out_u", type=str, default="../datasets/seismic_coarse_u.pt")
    parser.add_argument("--out_ut", type=str, default="../datasets/seismic_coarse_ut.pt")
    parser.add_argument("--out_vp", type=str, default="../datasets/seismic_coarse_vp.pt")
    parser.add_argument("--out_meta", type=str, default="../datasets/seismic_coarse_meta.pt")

    # Flow time subsampling
    parser.add_argument("--flow_stride", type=int, default=1,
                        help="Use every k-th flow snapshot (outer time).")

    # Wave time hyperparams (THIS is your 'longer timestep')
    parser.add_argument("--nt_out", type=int, default=151,
                        help="Number of OUTPUT (coarse) time samples saved per shot.")
    parser.add_argument("--k", type=int, default=4,
                        help="Downsample factor in wave time: dt_out = k * dt_sim.")
    parser.add_argument("--dt_scale", type=float, default=0.9,
                        help="Fine timestep: dt_sim = dt_scale * critical_dt(vmax). Must be <= 1.0 typically.")
    parser.add_argument("--vmax", type=float, default=2600.0,
                        help="Upper bound of vp used for stable critical_dt calculation.")

    # Solver/grid
    parser.add_argument("--spacing", type=float, nargs=2, default=(10.0, 10.0))
    parser.add_argument("--nbl", type=int, default=50)
    parser.add_argument("--space_order", type=int, default=8)
    parser.add_argument("--kernel", type=str, default="OT2")
    parser.add_argument("--f0", type=float, default=0.025)

    # Normalize
    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    # Load saturation snapshots
    sg_u = torch.load(args.sg_path, map_location="cpu")
    print(f"Loaded {args.sg_path} -> shape {tuple(sg_u.shape)}")
    # sg_u: (N, nz, nx, Tflow)
    N, nz, nx, Tflow = sg_u.shape

    # Choose flow frames (outer time)
    flow_ids = list(range(0, Tflow, max(1, args.flow_stride)))
    Tflow_eff = len(flow_ids)
    print(f"flow_stride={args.flow_stride} -> using {Tflow_eff}/{Tflow} flow frames")

    # Build a temp model to get dt_base from vmax
    tmp_model = Model(
        vp=np.ones((nx, nz), dtype=np.float32) * float(args.vmax),
        origin=(0.0, 0.0),
        spacing=(float(args.spacing[0]), float(args.spacing[1])),
        shape=(nx, nz),
        nbl=int(args.nbl),
        space_order=int(args.space_order),
        bcs="damp",
        dtype=np.float32
    )
    dt_base = float(tmp_model.critical_dt)

    # Fine and coarse timesteps
    dt_sim = float(args.dt_scale) * dt_base
    k = int(args.k)
    dt_out = k * dt_sim

    # Record length (seconds) to cover nt_out samples at dt_out
    tn = (int(args.nt_out) - 1) * dt_out

    print(f"dt_base(critical)={dt_base:.6e}")
    print(f"dt_sim={dt_sim:.6e}  (fine, stable)")
    print(f"k={k} -> dt_out={dt_out:.6e} (coarse effective timestep)")
    print(f"nt_out={args.nt_out}, tn={tn:.6e}")

    # Build solver/geometry for duration tn
    model, solver, nrec = build_solver_xz_with_tn(
        nx=nx, nz=nz, nbl=int(args.nbl),
        tn=tn, vmax=float(args.vmax),
        spacing=tuple(args.spacing),
        f0=float(args.f0),
        space_order=int(args.space_order),
        kernel=str(args.kernel)
    )

    # Allocate outputs (coarse-time)
    u = np.zeros((N, Tflow_eff, int(args.nt_out), nrec), dtype=np.float32)
    vp_all = np.zeros((N, Tflow_eff, nx, nz), dtype=np.float32)

    pbar = tqdm(total=N * Tflow_eff, desc="Generating coarse seismic (u, vp)")

    for i in range(N):
        for j, tf in enumerate(flow_ids):
            sg_zx = sg_u[i, :, :, tf].numpy().astype(np.float32)   # (nz,nx)
            vp_zx = generate_velocity_model_zx(sg_zx)             # (nz,nx)
            vp_xz = vp_zx.T                                       # (nx,nz) for Devito

            vp_all[i, j] = vp_xz
            u[i, j] = simulate_shot_coarse(
                model, solver, vp_xz,
                dt_sim=dt_sim, k=k, nt_out=int(args.nt_out),
                normalize=bool(args.normalize)
            )
            pbar.update(1)

    pbar.close()

    # Coarse derivative with respect to coarse timestep dt_out
    u_t_coarse = coarse_time_derivative(u, dt_out=dt_out)

    # Save time axes/meta (important for training)
    t_wave = (np.arange(int(args.nt_out), dtype=np.float32) * dt_out)

    meta = {
        "dt_base": torch.tensor(dt_base),
        "dt_sim": torch.tensor(dt_sim),
        "k": torch.tensor(k),
        "dt_out": torch.tensor(dt_out),
        "nt_out": torch.tensor(int(args.nt_out)),
        "tn": torch.tensor(tn),
        "t_wave": torch.from_numpy(t_wave),
        "flow_ids": torch.tensor(flow_ids, dtype=torch.int64),
        "flow_stride": torch.tensor(int(args.flow_stride)),
        "vmax": torch.tensor(float(args.vmax)),
        "spacing": torch.tensor([float(args.spacing[0]), float(args.spacing[1])]),
        "nbl": torch.tensor(int(args.nbl)),
        "space_order": torch.tensor(int(args.space_order)),
        "f0": torch.tensor(float(args.f0)),
        "kernel": args.kernel
    }

    torch.save(torch.from_numpy(u), args.out_u)
    torch.save(torch.from_numpy(u_t_coarse), args.out_ut)
    torch.save(torch.from_numpy(vp_all), args.out_vp)
    torch.save(meta, args.out_meta)

    print("Saved:")
    print(" ", args.out_u,  u.shape)
    print(" ", args.out_ut, u_t_coarse.shape)
    print(" ", args.out_vp, vp_all.shape)
    print(" ", args.out_meta, "(meta dict)")


if __name__ == "__main__":
    os.environ["TMPDIR"] = "/home/aiotlab/mnt/khiemtt/devito_tmp"
    os.environ["DEVITO_JIT_CACHE"] = "/home/aiotlab/mnt/khiemtt/devito_tmp"
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    os.makedirs(os.environ["DEVITO_JIT_CACHE"], exist_ok=True)
    os.environ["DEVITO_LOGGING"] = "ERROR"
    os.environ["OMP_NUM_THREADS"] = "8"
    main()
