# Copyright 2018 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import os
import pickle

import numpy as np
from scipy.integrate import odeint

PLOT = True
SAVEROOT = '/Users/danielhernandez/work/supervind/data/'
SAVEDIR = 'pendulum002/'
SAVEFILE = 'datadict'
ODE = 'pendulum'
NSAMPS = 200
NTBINS = 100
PARAMS = [(0.25, 5.0), (0.5, 3.0)]
Y0_RANGES = np.array([[0, np.pi], [-0.05, 0.05]])
IS_1D = True

def pendulum(y, _, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt
    
def noisy_integrate(ode, y0, Tbins, ode_params, noise_scale=0.0):
    """
    """
    t = np.linspace(0, 10, Tbins)
#     print(ode_params)
    sol = odeint(ode, y0, t, args=ode_params)
    
    noise = np.random.normal(scale=noise_scale, size=sol.shape)
    return sol + noise
    
def generate_data(ode, y0ranges, params, Nsamps=200, Tbins=100, noise_scale=0.1,
                  is_1D_output=IS_1D, data_scale=1.0):
    """
    """
    num_ids = len(params)

    yDim = len(y0ranges)
    Y0data = []
    data = np.zeros((Nsamps, Tbins, yDim))
    Ids = np.zeros(Nsamps)
    
    for d in range(yDim):
        Y0data.append(np.random.uniform(low=y0ranges[d][0], high=y0ranges[d][1], size=Nsamps))
    Y0 = np.stack(Y0data).T
    
    for samp in range(Nsamps):
        j = np.random.choice(list(range(num_ids)))
        data[samp] = data_scale*noisy_integrate(ode, Y0[samp], Tbins, params[j],
                                                 noise_scale=noise_scale)
        Ids[samp] = j
    if is_1D_output:
        Ydata = data[:,:,0:1]
    Ytrain, Yvalid, Ytest = Ydata[:-Nsamps//5], Ydata[-Nsamps//5:-Nsamps//10], Ydata[-Nsamps//10:]
    Idtrain, Idvalid, Idtest = Ids[:-Nsamps//5], Ids[-Nsamps//5:-Nsamps//10], Ids[-Nsamps//10:]
    
    datadict = {'Ytrain' : Ytrain, 'Yvalid' : Yvalid, 'Ytest' : Ytest,
                'Idtrain' : Idtrain, 'Idvalid' : Idvalid, 'Idtest' : Idtest,
                'Data' : data}
    
    return datadict
        
    
    
if __name__ == '__main__':
    """
    """
    odedict = {'pendulum' : pendulum}
    ddict = generate_data(odedict[ODE], Y0_RANGES, params=PARAMS, Nsamps=NSAMPS, Tbins=NTBINS,
                          noise_scale=0.05)
    data_path = SAVEROOT + SAVEDIR
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(data_path + SAVEFILE, 'wb+') as data_file:
        pickle.dump(ddict, data_file)

    data = ddict['Data']
    Idtrain = ddict['Idtrain']
    
    if PLOT:
        import matplotlib.pyplot as plt
        for i in range(10):
            c = 'b' if Idtrain[i] == 0 else 'r'
            plt.plot(data[i,:,0], color=c)
#             plt.plot(data[i,:,1], color='r')
        plt.show()
    
    
    
    