import numpy as np
import matplotlib.pyplot as plt
import h5py
from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

import logging
logger = logging.getLogger(__name__)

import time

#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

from dedalus.extras import plot_tools

Zgr = 1e-2

# Basis and domain

resolution = 256 #500
# normalization factors
r_0 = 1.2e8
T_0 = 1000
P_0 = 1 #1e5 #1
r_outer = 1.2e12/r_0
#r_inner = 1.2e9/r_0
r_inner = 1.2e9/r_0

log_inner = np.log(r_inner) 
log_outer = np.log(r_outer)

log_basis = de.Chebyshev('r', resolution, interval=(log_inner,log_outer)) 
domain = de.Domain([log_basis], np.float64)

# Problem

ncc_cutoff = 1e-2 #1e-10
tolerance = 1e-2 #1e-6
problem = de.NLBVP(domain,variables = ['P', 'T'], ncc_cutoff = ncc_cutoff) # P and T are actually log(P) and log(T)

# Parameters

problem.parameters['T_0'] = T_0 
problem.parameters['r_0'] = r_0
problem.parameters['P_0'] = P_0 # unused

problem.parameters['Mc'] = 5*5.972e27 # 5* mass of earth (in g)
problem.parameters['mu'] = 2.34 * 1.6735575e-24 #mH multiplied by hydrogen atom mass 
problem.parameters['kb'] = 1.38064852e-16 # g*cm**2/(K*s**2)
problem.parameters['G'] = 6.67408e-8 # cm**3/(g*s**2)
problem.parameters['dMtot'] = 1e-5*5.972e27/3.154e7 # 10e-5* mass of earth, g/s #3.154e7 is conversion from yr to s
problem.parameters['sig'] = 5.670367e-5 # (cgs units) e-5 in g/(K^4s^3), Stefan Boltzmann Const (normally it's e-8 in [W⋅m−2⋅K−4])
problem.parameters['s0'] = 1e-4 # cm
problem.parameters['rho_o'] = 3 #g cm^-3 grain internal density
problem.parameters['sigma_b'] = 5.6704e-5 # erg*cm^-2*s^-1*K^-4 Stefan Boltzmann 
rcore = 1.2e9
problem.parameters['grad_rad_cst'] = - 3*problem.parameters['dMtot']/(64*np.pi*rcore*problem.parameters['sig'])
problem.parameters['Zgr'] = Zgr # 1e-2

problem.parameters['rhodisk'] = 1e-11 # 1e-11 #g/cm**3
problem.parameters['Tdisk'] = 150 # 150 # kelvin

# Normalized equations
problem.parameters['grad'] = 0.28 # adiabatic gradient

problem.parameters['eq1cst'] = -1*problem.parameters['G']*problem.parameters['Mc']*problem.parameters['mu']/(T_0*r_0*problem.parameters['kb'])
print(problem.parameters['eq1cst'])
#problem.add_equation('r2*dr(P) = eq1cst*P/T')
problem.add_equation('exp(r) * dr(P) = eq1cst/exp(T)')
#problem.add_equation('dr(T) = -1*dr(P)*T/P*grad')
problem.add_equation('dr(T) = dr(P)*grad')

# Boundary Equations

#problem.add_bc("right(T) = Tdisk/T_0") # disk temp in kelvins
problem.add_bc("right(T) = log(Tdisk/T_0)") # disk temp in kelvins
problem.add_bc("right(P) = log(rhodisk*kb*Tdisk/mu/P_0)") # gas law 
#problem.add_bc("right(P) = rhodisk*kb*Tdisk/mu/T_0") # gas law 
#problem.add_bc("right(P) = rhodisk*kb*mu * right(T)") # gas law 


#problem.add_bc("left(P) = rhodisk*kb*Tdisk/mu/T_0 * 100") 
#problem.add_bc("left(P) =  2500") 

# solver = problem.build_solver(de.timesteppers.RK443)
solver = problem.build_solver()

# initial conditions and referencing local grid state fields

r = domain.grid(0)
R = np.log(np.exp(r)*r_0)
T = solver.state['T']
P = solver.state['P']


Tcore = 1000 #?
Pcore = 100 #???

R_gas = 8.314e7 # cgs 
gravity = problem.parameters["G"] * problem.parameters["Mc"] / (r_inner*r_0)**2 # cm/s^2 
molar_mass = 2.24 # g/mol
z_0 = R_gas*Tcore/(molar_mass*gravity) 
P['g'] = np.log(np.log(Pcore*np.exp(-(R)/(z_0))))

#T['g'] = 10000/T_0
T['g'] = np.log(30. * 150/T_0)

#output = solver.evaluator.add_file_handler('output', sim_dt =0.1 , max_writes = 100 )
#output.add_task('integ(P, 'r')', layout='g', name='IntegP')
#output.add_task('integ(T, 'r')', layout='g', name='IntegT')
#output.add_system(solver.state, layout='g')

T_list = [np.copy(T['g'])]
P_list = [np.copy(P['g'])]

# Iterations

pert = solver.perturbations.data
pert.fill(1+tolerance)
start_time = time.time()

while np.sum(np.abs(pert)) > tolerance and time.time() - start_time < 120:
    solver.newton_iteration()
    T_list.append(np.copy(T['g'])) # save
    
    P_list.append(np.copy(P['g'])) # save 
    logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
    logger.info('T iterate: {}'.format(T['g'][0]))
    logger.info('P iterate: {}'.format(P['g'][0]))
        
end_time = time.time()

np.savetxt("Pressure.txt",P_list[-1])
np.savetxt("Temperature.txt",T_list[-1])

P = P_list[-1]
T = P_list[-1]


print(" ")
print(" ") 
print("Done") 








