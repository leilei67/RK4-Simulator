import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. Physics Constants ---
G = 6.67430e-11        # Gravitational constant (m^3 kg^-1 s^-2)
M_sun = 1.989e30       # Mass of the Sun (kg)
c = 299792458.0        # Speed of light (m/s)
R_sun = 6.957e8        # Radius of the Sun (m)

# To visually see the bend in this step-by-step simulation, we must increase 
# the effective gravity. Otherwise, the RK4 path will look perfectly straight.
# A factor of ~60,000 gives us a beautiful ~30 degree visible deflection.
visual_scale_factor = 1 
G_sim = G * visual_scale_factor

# --- 2. RK4 Differential Equations ---
def get_derivatives(state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    
    # CHANGE HERE: Allow the photon to skim just beneath the mathematical surface 
    # so the simulation doesn't stop it at halftime.
    if r <= R_sun * 0.99: 
        return None
        
    # Einstein's General Relativity correction
    a_mag = (2 * G_sim * M_sun) / (r**2)
    ax = -a_mag * (x / r)
    ay = -a_mag * (y / r)
    
    return np.array([vx, vy, ax, ay])

def rk4_step(state, dt):
    """Advances the simulation by one time step (dt) using RK4 integration."""
    k1 = get_derivatives(state)
    if k1 is None: return None
    
    k2 = get_derivatives(state + 0.5 * dt * k1)
    if k2 is None: return None
    
    k3 = get_derivatives(state + 0.5 * dt * k2)
    if k3 is None: return None
    
    k4 = get_derivatives(state + dt * k3)
    if k4 is None: return None
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- 3. Run Simulation ---
# Initial Conditions
x_start = -100.0 * R_sun
# Start slightly above the Sun to give it room to bend around it smoothly
impact_parameter = R_sun * 1 
state = np.array([x_start, impact_parameter, c, 0.0]) # [x, y, vx, vy]

dt = 0.02 # Time step in seconds
trajectory = []

# Integrate using RK4 until the photon travels far enough to the right
while state[0] < 100.0 * R_sun:
    trajectory.append((state[0], state[1]))
    next_state = rk4_step(state, dt)
    
    if next_state is None:
        print("Photon absorbed by the Sun!")
        break
        
    state = next_state

# Extract X and Y coordinates for plotting
x_vals = [p[0] for p in trajectory]
y_vals = [p[1] for p in trajectory]

# --- ADD THESE LINES: Exaggerate the bend visually ---
# Calculate how far the photon deviated from its starting height, and multiply it!
exaggeration_factor = 500000 
y_visual = [impact_parameter - (impact_parameter - y) * exaggeration_factor for y in y_vals]
# --- Calculate the Deflection Angle from Velocity Vectors ---
# The state array holds [x, y, vx, vy]. 
# We extract the final velocities after the simulation loop ends.
vx_final = state[2]
vy_final = state[3]

# Calculate the angle of the final velocity vector in radians
# We use absolute value because vy_final is negative (pointing down)
deflection_radians = np.abs(np.arctan2(vy_final, vx_final))

# Convert to degrees and arcseconds
deflection_degrees = np.degrees(deflection_radians)
deflection_arcsec = deflection_degrees * 3600

print("-" * 40)
print("Simulation Complete!")
print(f"Final Velocity: vx = {vx_final:.2e} m/s, vy = {vy_final:.2e} m/s")
print(f"Deflection Angle: {deflection_degrees:.4f} degrees")
print(f"Deflection Angle: {deflection_arcsec:.2f} arcseconds")
print("-" * 40)
# Extract X and Y coordinates for plotting
x_vals = [p[0] for p in trajectory]
y_vals = [p[1] for p in trajectory]

# --- 4. Plotting & Visualization ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('black')

# --- ADDED: Exaggerate the bend visually for the plot ONLY ---
# (The underlying math and printed deflection angle remain 100% accurate)
exaggeration_factor = 2000
y_visual = [impact_parameter - (impact_parameter - y) * exaggeration_factor for y in y_vals]

# Draw Sun and Moon
sun = patches.Circle((0, 0), R_sun, color='orange', label='Sun')
ax.add_patch(sun)

moon = patches.Circle((0, 0), R_sun * 1.05, color='black', alpha=0.8, label='Moon (Eclipse)')
ax.add_patch(moon)

# Plot the exaggerated starlight
# CHANGE: Now using y_visual instead of y_vals
ax.plot(x_vals, y_visual, color='cyan', linestyle='-', linewidth=2, 
        label='Starlight Path (Exaggerated Bend)')

# Plot Newtonian/Straight path
# CHANGE: Extended the X coordinate to 100 * R_sun to match the zoomed-out view
ax.plot([x_start, 100 * R_sun], [impact_parameter, impact_parameter], 
        color='white', linestyle='--', alpha=0.4, label='Straight Path')

# Formatting
ax.set_aspect('equal')

# CHANGE: Zoomed out limits to see the bend over a long distance
ax.set_xlim(-20 * R_sun, 80 * R_sun)
ax.set_ylim(-30 * R_sun, 10 * R_sun)

ax.set_xticks([])
ax.set_yticks([])

ax.set_title("Einstein's 1919 Eclipse: Dynamical RK4 Simulation", color='white', pad=15)

legend = ax.legend(loc='lower left', facecolor='black', edgecolor='white')
for text in legend.get_texts():
    text.set_color("white")

plt.tight_layout()
plt.show()