import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# Schwarzschild GR ray tracing + quantitative plots
# Outputs:
#   1) alpha_vs_b.png
#   2) ray_paths.png
#   3) lensed_with_true_disk.png   (TRUE finite disk hit test)
# Units: geometric units G=c=1.
# ============================================================

# -------------------------
# Physical parameters
# -------------------------
M = 1.0
rs = 2.0 * M
r_ph = 3.0 * M
bc = 3.0 * np.sqrt(3.0) * M

# -------------------------
# Scattering experiment setup
# -------------------------
r_obs = 200.0 * M
u_obs = 1.0 / r_obs

# -------------------------
# Rendering camera setup
# (Zoom out => increase r_cam and/or reduce fov_deg to get more star background)
# -------------------------
W, H = 1000, 1000
fov_deg = 55.0                 # was 70.0 (narrower => more background / smaller BH+disk)
fov = np.deg2rad(fov_deg)
r_cam = 80.0 * M               # was 30.0 (farther => smaller BH+disk, more sky)

# -------------------------
# TRUE horizontal disk (plane z=0), finite annulus
# -------------------------
rin = 10.0 * M
rout = 14.0 * M
DISK_RGB = np.array([1.0, 0.45, 0.12], dtype=np.float64)
disk_power = 1.6
disk_strength = 0.95

# -------------------------
# Background (2:1 equirectangular JPG)
# -------------------------
BG_PATH = "/Users/haoyandeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Y1 Summer Project/starmap_2020_4k_print.jpg"
bg_img = Image.open(BG_PATH).convert("RGB")
bg = np.asarray(bg_img).astype(np.float64) / 255.0
H_bg, W_bg = bg.shape[:2]
print("Background loaded:", W_bg, "x", H_bg, "aspect =", W_bg / H_bg)


def sample_bg_direction(n):
    x, y, z = n
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    u = (0.5 + lon/(2*np.pi)) % 1.0
    v = 0.5 - lat/np.pi
    px = int(np.clip(u*(W_bg-1), 0, W_bg-1))
    py = int(np.clip(v*(H_bg-1), 0, H_bg-1))
    return bg[py, px]


# ============================================================
# GR null geodesic in Schwarzschild (equatorial):
# u'' + u = 3 M u^2, u = 1/r, independent variable phi
# ============================================================

def trace_ray_deflection(b, dphi=1e-3, max_steps=350000):
    u = u_obs
    inside = 1.0/(b*b) - u*u + 2.0*M*u*u*u
    if inside <= 0:
        return "capture", None
    w = +np.sqrt(inside)
    phi = 0.0

    for _ in range(max_steps):
        r = 1.0 / u
        if r <= rs * 1.0001:
            return "capture", None
        if r >= r_obs and w < 0:
            return "escape", (phi - np.pi)

        def fu(u_, w_): return w_
        def fw(u_, w_): return -u_ + 3.0*M*u_*u_

        k1u = dphi * fu(u, w);                 k1w = dphi * fw(u, w)
        k2u = dphi * fu(u+0.5*k1u, w+0.5*k1w); k2w = dphi * fw(u+0.5*k1u, w+0.5*k1w)
        k3u = dphi * fu(u+0.5*k2u, w+0.5*k2w); k3w = dphi * fw(u+0.5*k2u, w+0.5*k2w)
        k4u = dphi * fu(u+k3u,   w+k3w);       k4w = dphi * fw(u+k3u,   w+k3w)

        u += (k1u + 2*k2u + 2*k3u + k4u)/6.0
        w += (k1w + 2*k2w + 2*k3w + k4w)/6.0
        phi += dphi

        if u <= 0:
            return "escape", (phi - np.pi)

    return "capture", None


def trace_ray_full_equatorial(b, dphi=1e-3, max_steps=500000, store_every=35):
    u = u_obs
    inside = 1.0/(b*b) - u*u + 2.0*M*u*u*u
    if inside <= 0:
        return {"status": "capture", "traj_xy": ([], [])}

    w = +np.sqrt(inside)
    phi = 0.0
    xs, ys = [], []

    def record(u_val, phi_val):
        r = 1.0 / u_val
        xs.append(r*np.cos(phi_val))
        ys.append(r*np.sin(phi_val))

    record(u, phi)

    for step in range(max_steps):
        r = 1.0/u
        if r <= rs * 1.0001:
            return {"status": "capture", "traj_xy": (xs, ys)}
        if r >= r_obs and w < 0:
            return {"status": "escape", "traj_xy": (xs, ys)}

        def fu(u_, w_): return w_
        def fw(u_, w_): return -u_ + 3.0*M*u_*u_

        k1u = dphi * fu(u, w);                 k1w = dphi * fw(u, w)
        k2u = dphi * fu(u+0.5*k1u, w+0.5*k1w); k2w = dphi * fw(u+0.5*k1u, w+0.5*k1w)
        k3u = dphi * fu(u+0.5*k2u, w+0.5*k2w); k3w = dphi * fw(u+0.5*k2u, w+0.5*k2w)
        k4u = dphi * fu(u+k3u,   w+k3w);       k4w = dphi * fw(u+k3u,   w+k3w)

        u += (k1u + 2*k2u + 2*k3u + k4u)/6.0
        w += (k1w + 2*k2w + 2*k3w + k4w)/6.0
        phi += dphi

        if step % store_every == 0:
            record(u, phi)

        if u <= 0:
            return {"status": "escape", "traj_xy": (xs, ys)}

    return {"status": "capture", "traj_xy": (xs, ys)}


# ============================================================
# TRUE 3D disk intersection (disk plane is z=0)
# ============================================================

def make_plane_basis(r0, n0):
    L = np.cross(r0, n0)
    Ln = np.linalg.norm(L)
    e3 = (L / Ln) if Ln > 1e-12 else np.array([0.0, 1.0, 0.0], dtype=np.float64)

    r0p = r0 - np.dot(r0, e3) * e3
    r0pn = np.linalg.norm(r0p)
    if r0pn < 1e-12:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        r0p = tmp - np.dot(tmp, e3) * e3
        r0pn = np.linalg.norm(r0p)

    e1 = r0p / r0pn
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def trace_ray_hit_disk(r0, n0, dphi=1e-3, max_steps=120000):
    e1, e2, _ = make_plane_basis(r0, n0)
    b = np.linalg.norm(np.cross(r0, n0))

    u = 1.0 / r_obs
    inside = 1.0/(b*b) - u*u + 2.0*M*u*u*u
    if inside <= 0:
        return "capture", None

    w = +np.sqrt(inside)
    phi = 0.0

    r = 1.0/u
    pos = r*(np.cos(phi)*e1 + np.sin(phi)*e2)
    z_prev = pos[2]
    r_prev = r

    for _ in range(max_steps):
        r = 1.0/u
        if r <= rs * 1.0001:
            return "capture", None

        pos = r*(np.cos(phi)*e1 + np.sin(phi)*e2)
        z_now = pos[2]

        if (z_prev == 0.0) or (z_prev * z_now < 0.0):
            t = 0.0 if (z_prev == z_now) else (z_prev / (z_prev - z_now))
            r_cross = r_prev + t*(r - r_prev)
            if rin <= r_cross <= rout:
                return "disk", r_cross

        if r >= r_obs and w < 0:
            n_out = pos / np.linalg.norm(pos)
            return "escape", n_out

        def fu(u_, w_): return w_
        def fw(u_, w_): return -u_ + 3.0*M*u_*u_

        k1u = dphi * fu(u, w);                 k1w = dphi * fw(u, w)
        k2u = dphi * fu(u+0.5*k1u, w+0.5*k1w); k2w = dphi * fw(u+0.5*k1u, w+0.5*k1w)
        k3u = dphi * fu(u+0.5*k2u, w+0.5*k2w); k3w = dphi * fw(u+0.5*k2u, w+0.5*k2w)
        k4u = dphi * fu(u+k3u,   w+k3w);       k4w = dphi * fw(u+k3u,   w+k3w)

        u_new = u + (k1u + 2*k2u + 2*k3u + k4u)/6.0
        w_new = w + (k1w + 2*k2w + 2*k3w + k4w)/6.0
        phi_new = phi + dphi

        r_prev = r
        z_prev = z_now
        u, w, phi = u_new, w_new, phi_new

        if u <= 0:
            n_out = pos / np.linalg.norm(pos)
            return "escape", n_out

    return "capture", None


# ============================================================
# Outputs
# ============================================================

def save_alpha_vs_b():
    b_min = bc * 1.001
    b_max = 12.0 * M
    Nb = 300

    b_grid = np.linspace(b_min, b_max, Nb)
    alpha_grid = np.full_like(b_grid, np.nan, dtype=np.float64)

    for i, b in enumerate(b_grid):
        status, alpha = trace_ray_deflection(b, dphi=1e-3, max_steps=350000)
        if status == "escape":
            alpha_grid[i] = alpha
        if (i+1) % 30 == 0:
            print(f"alpha(b): {i+1}/{Nb} done")

    plt.figure()
    plt.plot(b_grid, alpha_grid)
    plt.axvline(bc, linestyle="--", label="b_c = 3√3 M")
    plt.xlabel("impact parameter b (units of M)")
    plt.ylabel("deflection angle α (rad)")
    plt.title("GR deflection angle vs b (numerical integration)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("alpha_vs_b.png", dpi=220)
    plt.close()
    print("Saved: alpha_vs_b.png")


def save_ray_paths():
    b_list = [0.98*bc, 1.01*bc, 5.8*M, 7.0*M, 9.0*M, 12.0*M]

    plt.figure(figsize=(8, 8))
    th = np.linspace(0, 2*np.pi, 800)
    plt.plot(rs*np.cos(th), rs*np.sin(th), linewidth=2, label="Event horizon r=2M")
    plt.plot(r_ph*np.cos(th), r_ph*np.sin(th), linestyle="--", linewidth=2, label="Photon sphere r=3M")
    plt.plot(rin*np.cos(th), rin*np.sin(th), linestyle=":", linewidth=2, label="Disk inner edge")
    plt.plot(rout*np.cos(th), rout*np.sin(th), linestyle=":", linewidth=2, label="Disk outer edge")

    for b in b_list:
        res = trace_ray_full_equatorial(b, dphi=1e-3, max_steps=500000, store_every=35)
        xs, ys = res["traj_xy"]
        plt.plot(xs, ys, linewidth=1.8, label=f"b={b:.2f} ({res['status']})")
        print(f"Path: b={b:.3f} -> {res['status']}")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-60*M, 60*M)
    plt.ylim(-60*M, 60*M)
    plt.xlabel("x (units of M)")
    plt.ylabel("y (units of M)")
    plt.title("GR null geodesics around a Schwarzschild black hole (equatorial plane)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("ray_paths.png", dpi=220)
    plt.close()
    print("Saved: ray_paths.png")


def render_true_disk():
    img = np.zeros((H, W, 3), dtype=np.float64)

    aspect = W / H
    tan_half = np.tan(fov / 2.0)

    inc_deg = 80.0
    inc = np.deg2rad(inc_deg)

    cam_pos = np.array([r_cam*np.sin(inc), 0.0, r_cam*np.cos(inc)], dtype=np.float64)
    forward = -cam_pos / np.linalg.norm(cam_pos)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-12:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    for j in range(H):
        for i in range(W):
            px = (2.0*(i + 0.5)/W - 1.0) * aspect * tan_half
            py = (1.0 - 2.0*(j + 0.5)/H) * tan_half

            n0 = (forward + px*right + py*up)
            n0 /= np.linalg.norm(n0)

            status, info = trace_ray_hit_disk(cam_pos, n0, dphi=1e-3, max_steps=120000)

            if status == "capture":
                img[j, i] = 0.0

            elif status == "disk":
                r_hit = info
                I = disk_strength * (rin / max(r_hit, 1e-9))**disk_power
                I = np.clip(I, 0.0, 1.0)
                base = 0.05 * np.ones(3)
                img[j, i] = (1.0 - I) * base + I * DISK_RGB

            else:
                n_out = info
                img[j, i] = sample_bg_direction(n_out)

        if (j+1) % 20 == 0:
            print(f"Render: row {j+1}/{H} done")

    plt.imsave("lensed_with_true_disk.png", np.clip(img, 0.0, 1.0))
    print("Saved: lensed_with_true_disk.png")


if __name__ == "__main__":
    save_alpha_vs_b()
    save_ray_paths()
    render_true_disk()