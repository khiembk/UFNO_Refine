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
# Flow time-warp: c_lambda(t) = c_1(lambda * t)
# -----------------------------
def sample_flow_with_lambda(sg_u_i: torch.Tensor, t_out_idx: int, Tflow: int, lambda_flow: float) -> torch.Tensor:
    """
    sg_u_i: [nz, nx, Tflow]  (torch)
    t_out_idx: integer index in [0, Tflow-1] of the *output timeline*
    lambda_flow: scaling in dc/dt = lambda*S

    Returns sg_zx (torch) at resampled time, shape [nz, nx]
    """
    if Tflow <= 1:
        return sg_u_i[:, :, 0]

    # normalized output time
    t_out = float(t_out_idx) / float(Tflow - 1)

    # time-warp: t_src = lambda * t_out, clamp to [0,1]
    t_src = lambda_flow * t_out
    if t_src >= 1.0:
        return sg_u_i[:, :, -1]
    if t_src <= 0.0:
        return sg_u_i[:, :, 0]

    # fractional index in original timeline
    idx_f = t_src * float(Tflow - 1)
    i0 = int(np.floor(idx_f))
    i1 = min(i0 + 1, Tflow - 1)
    w = float(idx_f - i0)

    # linear interpolation
    return (1.0 - w) * sg_u_i[:, :, i0] + w * sg_u_i[:, :, i1]


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
    nbl = model.nbl
    vp_pad = np.pad(vp_xz, ((nbl, nbl), (nbl, nbl)), mode="edge")
    model.vp.data[:] = vp_pad


def pad_or_trim_time(shot: np.ndarray, nt_out: int) -> np.ndarray:
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
    set_vp_on_model(model, vp_xz)

    rec, _, _ = solver.forward(dt=float(dt_sim))
    shot_fine = rec.data.astype(np.float32)

    if normalize:
        shot_fine /= (np.max(np.abs(shot_fine)) + 1e-8)

    shot_coarse = shot_fine[::k, :]
    shot_coarse = pad_or_trim_time(shot_coarse, nt_out)
    return shot_coarse


def coarse_time_derivative(u: np.ndarray, dt_out: float) -> np.ndarray:
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
    parser.add_argument("--out_u", type=str, default="../datasets/seismic_coarse_u_1.pt")
    parser.add_argument("--out_ut", type=str, default="../datasets/seismic_coarse_ut_1.pt")
    parser.add_argument("--out_vp", type=str, default="../datasets/seismic_coarse_vp_1.pt")
    parser.add_argument("--out_meta", type=str, default="../datasets/seismic_coarse_meta_1.pt")

    # Flow scaling hyperparam: dc/dt = lambda * S
    parser.add_argument("--lambda_flow", type=float, default=1.0,
                        help="Scale flow speed: c_lambda(t)=c_original(lambda*t). Examples: 1, 5, 10.")

    # Optional: choose which output flow frames to generate u on
    parser.add_argument("--flow_stride", type=int, default=1,
                        help="Use every k-th output flow frame index (still applies after lambda warp).")

    # Wave time hyperparams (coarse sampling in wave simulation)
    parser.add_argument("--nt_out", type=int, default=151)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--dt_scale", type=float, default=0.9)
    parser.add_argument("--vmax", type=float, default=2600.0)

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

    # Output flow indices we will generate u at (output timeline indices)
    out_flow_ids = list(range(0, Tflow, max(1, args.flow_stride)))
    Tflow_eff = len(out_flow_ids)

    print(f"lambda_flow={args.lambda_flow}")
    print(f"flow_stride={args.flow_stride} -> generating u for {Tflow_eff}/{Tflow} output flow frames")

    # Build temp model for dt_base
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

    dt_sim = float(args.dt_scale) * dt_base
    k = int(args.k)
    dt_out = k * dt_sim
    tn = (int(args.nt_out) - 1) * dt_out

    print(f"dt_base(critical)={dt_base:.6e}")
    print(f"dt_sim={dt_sim:.6e}")
    print(f"k={k} -> dt_out={dt_out:.6e}")
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

    # Allocate outputs (coarse wave-time)
    u = np.zeros((N, Tflow_eff, int(args.nt_out), nrec), dtype=np.float32)
    vp_all = np.zeros((N, Tflow_eff, nx, nz), dtype=np.float32)

    pbar = tqdm(total=N * Tflow_eff, desc="Generating u with lambda-flow warp")

    for i in range(N):
        # Pre-slice once to [nz,nx,Tflow] for speed
        sg_i = sg_u[i]  # torch, [nz,nx,Tflow]
        for j, tf_out in enumerate(out_flow_ids):
            # --- THIS is the key change: time-warped flow state ---
            sg_zx_torch = sample_flow_with_lambda(
                sg_u_i=sg_i,
                t_out_idx=tf_out,
                Tflow=Tflow,
                lambda_flow=float(args.lambda_flow)
            )
            sg_zx = sg_zx_torch.numpy().astype(np.float32)  # (nz,nx)

            vp_zx = generate_velocity_model_zx(sg_zx)
            vp_xz = vp_zx.T  # (nx,nz) for Devito

            vp_all[i, j] = vp_xz
            u[i, j] = simulate_shot_coarse(
                model, solver, vp_xz,
                dt_sim=dt_sim, k=k, nt_out=int(args.nt_out),
                normalize=bool(args.normalize)
            )
            pbar.update(1)

    pbar.close()

    u_t_coarse = coarse_time_derivative(u, dt_out=dt_out)
    t_wave = (np.arange(int(args.nt_out), dtype=np.float32) * dt_out)

    meta = {
        "lambda_flow": torch.tensor(float(args.lambda_flow)),
        "dt_base": torch.tensor(dt_base),
        "dt_sim": torch.tensor(dt_sim),
        "k": torch.tensor(k),
        "dt_out": torch.tensor(dt_out),
        "nt_out": torch.tensor(int(args.nt_out)),
        "tn": torch.tensor(tn),
        "t_wave": torch.from_numpy(t_wave),
        "out_flow_ids": torch.tensor(out_flow_ids, dtype=torch.int64),
        "flow_stride": torch.tensor(int(args.flow_stride)),
        "Tflow_original": torch.tensor(int(Tflow)),
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
