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
    while 1/U < 14000 and phi < np.pi/2 + 0.1:  

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

def initialise_V0(theta, U0):
    #theta = the outgoing ray angle with respect to the y axis
    #U0 must be given in its scaled value
    return (U0)/np.tan(theta)


#angular radius of the sun is approx 4.6e-3 radians
#distance from earth to sun is roughly 133.6 * b
# 0.00748 radians is the outgoing value of theta that results in the impact parameter (distance of closest approach) to be the radius of the sun
RK4_coupled(0.00001, (-1*np.pi)/2, 1/(133.56), initialise_V0(0.00748, 1/(133.56)))


plt.plot(x_values,y_values)
# function determines outgoing and icoming angles the light makes to the horizontal
def calculate_angles(x_0,x_1): 

    # !!! better to use a linear fit using statistical analysis but this works for now
    x = x_values[x_0:x_1]
    y = y_values[x_0:x_1]

    gradients = []
    for i in range(1, len(x)):
        gradients.append((y[i]-y[i-1])/(x[i]-x[i-1]))
    sum = 0 
    for i in range (1,len(gradients)):
        sum = sum + gradients[i] 
    average = sum/len(gradients)
    angle = np.arctan(average)

    return angle 

deflection = calculate_angles(len(x_values)-1000,len(x_values)-1) - calculate_angles(0, 1000)
print("total angle of deflection = ", deflection*206265)

plt.show()
# note: final line looks like a straight line only because the actual deflection is tiny
# but you can see how the final line starts at the coordinate x=0 y= -1 * distance from earth to sun