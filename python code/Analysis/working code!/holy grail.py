from raytracer import calculate_deflection
import numpy as np
import matplotlib.pyplot as plt

M_sun = 1.989e30
Mpc = 3.086e22

class LensSystem:
    def __init__(self, d_L_Mpc, d_S_Mpc, d_LS_Mpc, r_ein_kpc, log_Mstar):
        self.d_L_Mpc = d_L_Mpc
        self.d_S_Mpc = d_S_Mpc
        self.d_LS_Mpc = d_LS_Mpc
        self.r_ein_kpc = r_ein_kpc
        
        self.m_L = (10**log_Mstar) * M_sun
        self.d_L = d_L_Mpc * Mpc
        self.d_LS = d_LS_Mpc * Mpc
        self.d_S = d_S_Mpc * Mpc
        self.r_ein = r_ein_kpc * Mpc / 1000
        self.initial_theta_guess = 2 * self.r_ein / self.d_L

    def _beta(self, theta, alpha):
        return theta - (self.d_LS / self.d_S) * alpha

    def _calculate_beta(self, theta):
        x_values, y_values, deflection = calculate_deflection(theta, self.d_L, self.m_L)
        return self._beta(theta, deflection)

    def binarysearch(self, depth=10):
        theta_values = []
        beta_values = []

        current_theta = self.initial_theta_guess
        step_size = self.initial_theta_guess / 2

        for i in range(depth):
            b = self._calculate_beta(current_theta)
            theta_values.append(current_theta)
            beta_values.append(b)

            if b > 0:
                current_theta -= step_size
            else:
                current_theta += step_size

            step_size /= 2

        return theta_values, beta_values

    def predicted_r_ein_kpc(self, theta_values):
        return self.d_L_Mpc * 1000 * theta_values[-1]
    


    def plot(self, theta_values, beta_values):
        plt.plot(theta_values, beta_values, marker='x')
        plt.xlabel("theta")
        plt.ylabel("beta")
        plt.title(f"d_L={self.d_L_Mpc} Mpc")
        plt.show()


# --- Run for each dataset ---
datasets = []
with open('data.csv', 'r') as f:
    f.readline()
    for line in f:
        datasets.append(line.strip().split(","))
for i in range(len(datasets)):
    for j in range(5):
        datasets[i][j] = float(datasets[i][j])

f = open('final_values.txt', 'w')
for data in datasets:
    lens = LensSystem(*data)
    theta_values, beta_values = lens.binarysearch(depth = 20)
    print(f"Predicted R_ein: {lens.predicted_r_ein_kpc(theta_values):.4f} kpc")
    f.write(f"{lens.predicted_r_ein_kpc(theta_values):.4f}\n")
f.close()
    

