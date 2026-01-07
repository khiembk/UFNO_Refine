import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver


import os


def generate_velocity_model(sg):
    vp_brine = 2200.0
    vp_co2 = 1500.0

    vp = vp_brine * (1.0 - sg) + vp_co2 * sg

    nz, nx = vp.shape
    vp_background = np.ones((nz, nx), dtype=np.float32) * 2000.0
    vp_background[int(0.6 * nz):, :] = 2600.0

    vp = 0.7 * vp + 0.3 * vp_background
    vp = gaussian_filter(vp, sigma=1.0)

    return vp.astype(np.float32)


def simulate_shot_record(vp):
    """
    Returns: shot (151, 101)
    """

    # -----------------------
    # Model
    # -----------------------
    nz, nx = vp.shape
    spacing = (10.0, 10.0)
    origin = (0.0, 0.0)
    nbl = 50

    model = Model(
        vp=vp,
        origin=origin,
        spacing=spacing,
        shape=(nz, nx),
        nbl=nbl,
        space_order=8,
        bcs="damp",
        dtype=np.float32
    )

    # -----------------------
    # Time axis
    # -----------------------
    nt = 151
    dt = model.critical_dt
    tn = (nt - 1) * dt
    f0 = 0.025  # 25 Hz

    # -----------------------
    # Source position (x, z)
    # -----------------------
    src_positions = np.array(
        [[model.domain_size[0] / 2.0, 20.0]],
        dtype=np.float32
    )

    # -----------------------
    # Receivers
    # -----------------------
    n_receivers = 101
    rec_x = np.linspace(
        20.0,
        model.domain_size[0] - 20.0,
        n_receivers
    )

    rec_positions = np.zeros((n_receivers, 2), dtype=np.float32)
    rec_positions[:, 0] = rec_x
    rec_positions[:, 1] = 20.0

    # -----------------------
    # Geometry (SOURCE IS CREATED HERE)
    # -----------------------
    geometry = AcquisitionGeometry(
        model=model,
        rec_positions=rec_positions,
        src_positions=src_positions,
        t0=0.0,
        tn=tn,
        f0=f0,
        src_type="Ricker"   # 🚨 CRITICAL
    )

    # -----------------------
    # Solver
    # -----------------------
    solver = AcousticWaveSolver(
        model=model,
        geometry=geometry,
        space_order=8,
        kernel="OT2"
    )

    # -----------------------
    # Forward modeling
    # -----------------------
    rec, _, _ = solver.forward(dt=dt)

    # -----------------------
    # Output
    # -----------------------
    shot = rec.data[:nt, :].astype(np.float32)
    shot /= (np.max(np.abs(shot)) + 1e-8)

    return shot

def build_solver(nz, nx):
    spacing = (10.0, 10.0)
    origin = (0.0, 0.0)
    nbl = 50

    vp_init = np.ones((nz, nx), dtype=np.float32) * 2000.0

    model = Model(
        vp=vp_init,
        origin=origin,
        spacing=spacing,
        shape=(nz, nx),
        nbl=nbl,
        space_order=8,
        bcs="damp",
        dtype=np.float32
    )

    nt = 151
    dt = model.critical_dt
    tn = (nt - 1) * dt
    f0 = 0.025

    src_positions = np.array(
        [[model.domain_size[0] / 2.0, 20.0]],
        dtype=np.float32
    )

    n_receivers = 101
    rec_x = np.linspace(
        20.0,
        model.domain_size[0] - 20.0,
        n_receivers
    )

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

    return model, solver, nt

def simulate_shot_fast(model, solver, vp, nt):
    nbl = model.nbl
    model.vp.data[nbl:-nbl, nbl:-nbl] = vp
    # model.vp.data[:] = vp
    rec, _, _ = solver.forward(dt=model.critical_dt)

    shot = rec.data[:nt].astype(np.float32)
    shot /= (np.max(np.abs(shot)) + 1e-8)

    return shot


def main():
    sg_u = torch.load("../datasets/sg_test_u.pt", map_location="cpu")
    print(f"Loaded sg_test_u.pt → shape: {sg_u.shape}")

    num_samples, nz, nx, num_timesteps = sg_u.shape

    model, solver, nt = build_solver(nz, nx)

    seismic_u = np.zeros(
        (num_samples, num_timesteps, nt, 101),
        dtype=np.float32
    )

    pbar = tqdm(
        total=num_samples * num_timesteps,
        desc="Generating seismic (FAST)"
    )

    for i in range(num_samples):
        for t in range(num_timesteps):
            sg = sg_u[i, :, :, t].numpy()
            vp = generate_velocity_model(sg)

            seismic_u[i, t] = simulate_shot_fast(
                model, solver, vp, nt
            )

            pbar.update(1)

    pbar.close()

    torch.save(
        torch.from_numpy(seismic_u),
        "../datasets/seismic_test_u.pt"
    )

    print("Saved seismic_test_u.pt")
    print("Final shape:", seismic_u.shape)



def main_old():
    sg_u = torch.load("../datasets/sg_test_u.pt", map_location="cpu")
    print(f"Loaded sg_test_u.pt → shape: {sg_u.shape}")

    num_samples, nz, nx, num_timesteps = sg_u.shape

    seismic_u = np.zeros(
        (num_samples, num_timesteps, 151, 101),
        dtype=np.float32
    )

    pbar = tqdm(
        total=num_samples * num_timesteps,
        desc="Generating seismic"
    )

    for i in range(num_samples):
        for t in range(num_timesteps):
            sg = sg_u[i, :, :, t].numpy()
            vp = generate_velocity_model(sg)

            seismic_u[i, t] = simulate_shot_record(vp)
            pbar.update(1)

    pbar.close()

    torch.save(
        torch.from_numpy(seismic_u),
        "../datasets/seismic_test_u.pt"
    )

    print("Saved seismic_test_u.pt")
    print("Final shape:", seismic_u.shape)




if __name__=="__main__":
    os.environ["TMPDIR"] = "/home/aiotlab/mnt/khiemtt/devito_tmp"
    os.environ["DEVITO_JIT_CACHE"] = "/home/aiotlab/mnt/khiemtt/devito_tmp"
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    os.environ["DEVITO_LOGGING"] = "ERROR"
    os.environ["OMP_NUM_THREADS"] = "8"   # adjust to your CPU
    os.environ["DEVITO_JIT_CACHE"] = "/home/aiotlab/mnt/khiemtt/devito_tmp"
    os.makedirs(os.environ["DEVITO_JIT_CACHE"], exist_ok=True)
    main()