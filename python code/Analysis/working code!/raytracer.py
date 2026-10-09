import numpy as np
import matplotlib.pyplot as plt

G = 6.6743e-11
c = 2.99792e8
mPc = 3.086e22
pi = np.pi

delta_phi = 1e-5

def function_1(phi, U, V, eps):
    return V

def function_2(phi, U, V, eps):
    return eps * U**2 - U

def initialise_V0(theta, U0):
    return U0 / np.tan(theta)

def raytrace(theta, d_L, m_L):
    eps =  3 * G * m_L / c**2


    x_values = []
    y_values = []
    U_values = []
    phi_values = []

    phi = -pi / 2
    U = 1 / d_L   # equivalent to old U0 = 1/133.56 in scaled variables
    V = initialise_V0(theta, U)

    while 1/U < 1.01 * d_L and phi < np.pi/2 + 0.1:
        K1 = function_1(phi, U, V, eps)
        H1 = function_2(phi, U, V, eps)

        K2 = function_1(phi + delta_phi/2, U + delta_phi*K1/2, V + delta_phi*H1/2, eps)
        H2 = function_2(phi + delta_phi/2, U + delta_phi*K1/2, V + delta_phi*H1/2, eps)

        K3 = function_1(phi + delta_phi/2, U + delta_phi*K2/2, V + delta_phi*H2/2, eps)
        H3 = function_2(phi + delta_phi/2, U + delta_phi*K2/2, V + delta_phi*H2/2, eps)

        K4 = function_1(phi + delta_phi, U + delta_phi*K3, V + delta_phi*H3, eps)
        H4 = function_2(phi + delta_phi, U + delta_phi*K3, V + delta_phi*H3, eps)

        r = 1 / U
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        x_values.append(x)
        y_values.append(y)
        U_values.append(U)
        phi_values.append(phi)

        U += (delta_phi / 6) * (K1 + 2*K2 + 2*K3 + K4)
        V += (delta_phi / 6) * (H1 + 2*H2 + 2*H3 + H4)
        phi += delta_phi

    return np.array(x_values), np.array(y_values), np.array(U_values), np.array(phi_values)

def calculate_angle2(x, y):
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]

    if np.arctan(slope) < 0: 
        return pi - abs(np.arctan(slope)) #this is if the deflection is large enough to catapult the light ray into quadrant 2
    else:
        return np.arctan(slope) #this is if the deflection keeps the light ray in quadrant 1. Potential room for screw-ups if light ray ends up in quadrant 3 or even 4. Need to do a x, y parity check


def calculate_deflection(theta, d_L, m_L):
    x_values, y_values, U_values, phi_values = raytrace(theta, d_L, m_L)
    n = 1000
    angle_out = calculate_angle2(x_values[-n:], y_values[-n:]) 
    angle_in = pi/2-theta
    deflection = angle_out - angle_in
    #print("total angle of deflection =", deflection * 206265, "arcsec")
    return x_values, y_values, deflection

#x_values, y_values, deflection = calculate_deflection(0.004650433736, 149597870700 , 1.989e30) #nice, theta resulting in impact parameter approx arctan(radius of sun/distance to sun)
#print(deflection* 206265)
#plt.plot(x_values, y_values)
#lt.show()

