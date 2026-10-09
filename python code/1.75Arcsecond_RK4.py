import numpy as np 
import matplotlib.pyplot as plt 

b = 6.96*(10**8)    # impact paramater approximetely radius of sun. 
eps = 6.35*(10**-6) # coefficent representing the version of the null geodesic differntial equation scaled by factor 1/R
pi = np.pi 

x_values = []       
y_values = [] 
U_values =[]
phi_values = []

#functions defining set of coupled differntial equation describing the path light takes around a star 
def function_1(phi,U,V):          # defines firt diffenrtial equation 
    return V

def function_2(phi,U,V):          # defines second differntial equation 
    return (eps*U**2)-U

# function performes RK4 on coupled differntial equation to produce a set of x and y vallues defining the trajectory of the light as it grazes the sun 
# note: only one asymtotic direction is calculated hence final deflection angle will be double that determined
# note: U is restircted from going to 0 as near 0 the value of U blows up 
def RK4_coupled(delta_phi, phi_0, U_0, V_0):
    U = U_0 
    V = V_0 
    phi = phi_0
    # loop terminates if U get too small and range of phi is 0 to 2 radians 
    while U > 10**-4 and phi < 2:  

        K1 = function_1(phi,U,V)
        H1 = function_2(phi,U,V) 

        K2 = function_1(phi + delta_phi/2, U + delta_phi*K1/2, V + delta_phi*H1/2)
        H2 = function_2(phi + delta_phi/2, U + delta_phi*K1/2, V + delta_phi*H1/2)

        K3 = function_1(phi + delta_phi/2, U + delta_phi*K2/2, V + delta_phi*H2/2)
        H3 = function_2(phi + delta_phi/2, U + delta_phi*K2/2, V + delta_phi*H2/2)

        K4 = function_1(phi + delta_phi, U + delta_phi*K3, V + delta_phi*H3)
        H4 = function_2(phi + delta_phi, U + delta_phi*K3, V + delta_phi*H3)

        r = b/U 
        x = r*np.cos(phi)
        y = r*np.sin(phi)

        x_values.append(x)
        y_values.append(y)
        U_values.append(U)
        phi_values.append(phi)

        U = U + (delta_phi/6)*(K1 + 2*K2 + 2*K3 + K4)
        V = V + (delta_phi/6)*(H1 + 2*H2 + 2*H3 + H4) 

        phi = phi + delta_phi 

RK4_coupled(0.000001, 0, 1, 0)

plt.plot(x_values,y_values)
# function determines outgoing and icoming angles the light makes to the horizontal
def calculate_angles(x_0,x_1): 
    x = x_values[x_0:x_1]
    y = y_values[x_0:x_1]

    gradients = np.gradient(y,x)
    sum = 0 
    for i in range (1,len(gradients)):
        sum = sum + gradients[i] 
    average = sum/len(gradients)
    angle = np.arctan(average)

    return angle 

deflection = calculate_angles(1570000,len(x_values)) - calculate_angles(100,10000)
print("total angle of deflection = ", 2*deflection*206265)

plt.show()