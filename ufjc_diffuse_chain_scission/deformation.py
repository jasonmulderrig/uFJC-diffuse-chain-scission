# import necessary libraries
import numpy as np
import os
import pathlib
import sys
import matplotlib.pyplot as plt
from types import SimpleNamespace

class AppliedDeformation(object):

    def __init__(self, parameters, F_func):

        if hasattr(parameters, "material") == False or hasattr(parameters, "deformation") == False:
            sys.exit("Need to specify either material parameters, deformation parameters, or both in order to define the applied deformation history")

        self.mp = parameters.material
        self.dp = parameters.deformation

        self.F_func = F_func # argument: t # sec
        
        self.define_deformation()
    
    def define_deformation(self):
        """
        Define the applied deformation history
        """

        # Deformation stepping calculations
        t_temp       = np.linspace(self.dp.t_min, self.dp.t_max, int(1e5)) # sec
        max_F_dot    = np.max(np.abs(np.diff(self.F_func(t_temp))/np.diff(t_temp))) # 1/sec
        t_scale      = 1./max_F_dot # sec
        t_step       = np.around(self.dp.t_step_modify_factor*t_scale, 2) # sec
        t_step_chunk = self.dp.t_step_chunk_modify_factor*t_step # sec

        self.dp.t_scale      = t_scale
        self.dp.t_step       = t_step
        self.dp.t_step_chunk = t_step_chunk

        if self.dp.t_step > self.dp.t_max:
            sys.exit('Error: The time step is larger than the total deformation time! Adjust the value of t_step_modify_factor to correct for this.')

        # determine the time step chunk: the number of time steps required to pass between data-saving instances in the applied deformation history
        t_step_chunk_num = int(np.around(self.dp.t_step_chunk/self.dp.t_step))
        if t_step_chunk_num < 1:
            t_step_chunk_num = 1

        # initialize the chunk counter and associated constants/lists
        chunk_counter  = 0
        chunk_indx_val = 0
        chunk_indx     = []

        # Initialization step: allocate time and stretch results, dependent upon the type of deformation being accounted for 
        t_val    = self.dp.t_min # initialize the time value at zero
        t        = [] # sec
        t_chunks = [] # sec
        if self.dp.deformation_type == 'uniaxial':
            lmbda_1_val    = 1 # assuming no pre-swelling of the network
            lmbda_1        = [] # unitless
            lmbda_1_chunks = [] # unitless
        elif self.dp.deformation_type == 'equibiaxial': pass
        elif self.dp.deformation_type == 'simple_shear': pass

        # Append to appropriate lists
        t.append(t_val)
        t_chunks.append(t_val)
        chunk_indx.append(chunk_indx_val)
        if self.dp.deformation_type == 'uniaxial':
            lmbda_1.append(lmbda_1_val)
            lmbda_1_chunks.append(lmbda_1_val)
        elif self.dp.deformation_type == 'equibiaxial': pass
        elif self.dp.deformation_type == 'simple_shear': pass

        # update the chunk iteration counter
        chunk_counter  += 1
        chunk_indx_val += 1

        # advance to the first time step
        t_val += self.dp.t_step

        while t_val <= self.dp.t_max:
            # Calculate displacement at a particular time step
            if self.dp.deformation_type == 'uniaxial':
                lmbda_1_val = self.F_func(t_val)
            elif self.dp.deformation_type == 'equibiaxial': pass
            elif self.dp.deformation_type == 'simple_shear': pass

            # Append to appropriate lists
            t.append(t_val)
            if self.dp.deformation_type == 'uniaxial':
                lmbda_1.append(lmbda_1_val)
            elif self.dp.deformation_type == 'equibiaxial': pass
            elif self.dp.deformation_type == 'simple_shear': pass

            if chunk_counter == t_step_chunk_num:
                # Append to appropriate lists
                t_chunks.append(t_val)
                chunk_indx.append(chunk_indx_val)
                if self.dp.deformation_type == 'uniaxial':
                    lmbda_1_chunks.append(lmbda_1_val)
                elif self.dp.deformation_type == 'equibiaxial': pass
                elif self.dp.deformation_type == 'simple_shear': pass

                # update the time step chunk iteration counter
                chunk_counter = 0

            # advance to the next time step
            t_val          += self.dp.t_step
            chunk_counter  += 1
            chunk_indx_val += 1
        
        # Calculate displacement
        if self.dp.deformation_type == 'uniaxial':
            u        = [x-1 for x in lmbda_1]
            u_chunks = [x-1 for x in lmbda_1_chunks]
        elif self.dp.deformation_type == 'equibiaxial': pass
        elif self.dp.deformation_type == 'simple_shear': pass

        # If the endpoint of the chunked applied deformation is not equal to the true endpoint of the applied deformation, then give the user the option to kill the simulation, or proceed on
        if chunk_indx[-1] != len(t)-1:
            terminal_statement = input('The endpoint of the chunked applied deformation is not equal to the endpoint of the actual applied deformation. Do you wish to kill the simulation here, or proceed on? (yes or no) ')
            if terminal_statement.lower() == 'yes':
                sys.exit()
            else: pass

        if self.dp.deformation_type == 'uniaxial':
            self.t              = t
            self.t_chunks       = t_chunks
            self.lmbda_1        = lmbda_1
            self.lmbda_1_chunks = lmbda_1_chunks
            self.u              = u
            self.u_chunks       = u_chunks
            self.chunk_indx     = chunk_indx
        elif self.dp.deformation_type == 'equibiaxial': pass
        elif self.dp.deformation_type == 'simple_shear': pass
    
    def finalization(self):
        """
        Plot the applied deformation history
        """
        pass
