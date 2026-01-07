import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from devito import Grid
from examples.seismic import Model, Receiver, RickerSource, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver


# ============================================================
# Load saturation data
# ============================================================

def load_saturation_data(path):
    data = torch.load(path, map_location="cpu")
    print(f"Loaded {path} → shape: {data.shape}")
    return data


# ============================================================
# Velocity model from gas saturation
# ============================================================

def generate_velocity_model(sg):
    """
    sg: (nz, nx)
    """
    vp_brine = 2200.0
    vp_co2 = 1500.0

    vp = vp_brine * (1.0 - sg) + vp_co2 * sg

    nz, nx = vp.shape

    vp_background = np.ones((nz, nx), dtype=np.float32) * 2000.0
    vp_background[int(0.6 * nz):, :] = 2600.0

    vp = 0.7 * vp + 0.3 * vp_background
    vp = gaussian_filter(vp, sigma=1.0)

    return vp.astype(np.float32)


# ============================================================
# Single-shot seismic simulation
# ============================================================

def simulate_shot_record(vp):
    """
    Returns:
        shot: (nt=151, n_receivers=101)
    """

    # -----------------------
    # Grid / model
    # -----------------------
    nz, nx = vp.shape
    spacing = (10.0, 10.0)
    origin = (0.0, 0.0)
    nbl = 50

    model = Model(
        vp=vp,
        origin=origin,
        shape=(nz, nx),
        spacing=spacing,
        nbl=nbl,
        space_order=8,
        bcs="damp",
        dtype=np.float32
    )

    # -----------------------
    # Time axis (FIXED)
    # -----------------------
    nt = 151
    dt = model.critical_dt
    tn = (nt - 1) * dt

    f0 = 0.025  # 25 Hz

    # -----------------------
    # Source position
    # -----------------------
    src_positions = np.zeros((1, 2), dtype=np.float32)
    src_positions[0, 0] = model.domain_size[0] / 2.0
    src_positions[0, 1] = 20.0

    # -----------------------
    # Receiver array
    # -----------------------
    n_receivers = 101
    rec_x = np.linspace(100.0, (nx - 10) * spacing[0], n_receivers)

    rec_positions = np.zeros((n_receivers, 2), dtype=np.float32)
    rec_positions[:, 0] = rec_x
    rec_positions[:, 1] = 20.0

    # -----------------------
    # Geometry
    # -----------------------
    geometry = AcquisitionGeometry(
        model,
        rec_positions,
        src_positions,
        t0=0.0,
        tn=tn,
        f0=f0
    )

    # -----------------------
    # Explicit Ricker source (IMPORTANT)
    # -----------------------
    geometry.src = RickerSource(
        name="src",
        grid=model.grid,
        f0=f0,
        time_range=geometry.time_axis,
        npoint=1
    )
    geometry.src.coordinates.data[:] = src_positions

    # -----------------------
    # Receivers
    # -----------------------
    rec = Receiver(
        name="rec",
        grid=model.grid,
        time_range=geometry.time_axis,
        coordinates=rec_positions
    )

    # -----------------------
    # Solver
    # -----------------------
    solver = AcousticWaveSolver(model, geometry, space_order=8)

    rec_data, _, _ = solver.forward(src=geometry.src, rec=rec, dt=dt)

    # -----------------------
    # Output
    # -----------------------
    shot = rec_data.data[:nt].astype(np.float32)
    shot /= (np.max(np.abs(shot)) + 1e-8)

    return shot


# ============================================================
# Main loop
# ============================================================

def main():

    sg_u = load_saturation_data("../datasets/sg_test_u.pt")

    num_samples, nz, nx, num_timesteps = sg_u.shape

    nt = 151
    n_receivers = 101

    print(
        f"Generating seismic data for "
        f"{num_samples} samples × {num_timesteps} timesteps "
        f"= {num_samples * num_timesteps} shots"
    )

    seismic_u = np.zeros(
        (num_samples, num_timesteps, nt, n_receivers),
        dtype=np.float32
    )

    pbar = tqdm(total=num_samples * num_timesteps, desc="Generating seismic")

    for i in range(num_samples):
        for t in range(num_timesteps):
            sg = sg_u[i, :, :, t].numpy()
            vp = generate_velocity_model(sg)

            shot = simulate_shot_record(vp)
            seismic_u[i, t] = shot

            pbar.update(1)

    pbar.close()

    save_path = "../datasets/seismic_test_u.pt"
    torch.save(torch.from_numpy(seismic_u), save_path)

    print(f"\nSaved seismic dataset to {save_path}")
    print(f"Final shape: {seismic_u.shape}")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
