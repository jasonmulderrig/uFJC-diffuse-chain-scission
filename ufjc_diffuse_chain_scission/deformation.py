# import necessary libraries
import numpy as np
import os
import pathlib
import sys
from types import SimpleNamespace

class AppliedDeformation(object):

    def __init__(self, parameters, F_func, initialize_lmbda, store_initialized_lmbda_, calculate_lmbda_func, store_calculated_lmbda, store_calculated_lmbda_chunk_post_processing, calculate_u_func, save2deformation):

        if hasattr(parameters, "deformation") == False:
            sys.exit("Need to specify deformation parameters in order to define the applied deformation history")

        self.dp = parameters.deformation

        self.F_func                                       = F_func # argument: t # sec
        self.initialize_lmbda                             = initialize_lmbda
        self.store_initialized_lmbda                      = store_initialized_lmbda_
        self.calculate_lmbda_func                         = calculate_lmbda_func
        self.store_calculated_lmbda                       = store_calculated_lmbda
        self.store_calculated_lmbda_chunk_post_processing = store_calculated_lmbda_chunk_post_processing
        self.calculate_u_func                             = calculate_u_func
        self.save2deformation                             = save2deformation
        
        self.define_deformation()
    
    def define_deformation(self):
        """
        Define the applied deformation history
        """
        deformation = SimpleNamespace()

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
        lmbda    = self.initialize_lmbda()

        # Append to appropriate lists
        t.append(t_val)
        t_chunks.append(t_val)
        chunk_indx.append(chunk_indx_val)
        lmbda = self.store_initialized_lmbda(lmbda)

        # update the chunk iteration counter
        chunk_counter  += 1
        chunk_indx_val += 1

        # advance to the first time step
        t_val += self.dp.t_step

        while t_val <= self.dp.t_max:
            # Calculate displacement at a particular time step
            lmbda_val = self.calculate_lmbda_func(t_val)

            # Append to appropriate lists
            t.append(t_val)
            lmbda = self.store_calculated_lmbda(lmbda, lmbda_val)

            if chunk_counter == t_step_chunk_num:
                # Append to appropriate lists
                t_chunks.append(t_val)
                chunk_indx.append(chunk_indx_val)
                lmbda = self.store_calculated_lmbda_chunk_post_processing(lmbda, lmbda_val)

                # update the time step chunk iteration counter
                chunk_counter = 0

            # advance to the next time step
            t_val          += self.dp.t_step
            chunk_counter  += 1
            chunk_indx_val += 1
        
        u = self.calculate_u_func(lmbda)

        # If the endpoint of the chunked applied deformation is not equal to the true endpoint of the applied deformation, then give the user the option to kill the simulation, or proceed on
        if chunk_indx[-1] != len(t)-1:
            terminal_statement = input('The endpoint of the chunked applied deformation is not equal to the endpoint of the actual applied deformation. Do you wish to kill the simulation here, or proceed on? (yes or no) ')
            if terminal_statement.lower() == 'yes':
                sys.exit()
            else: pass
        
        deformation.t          = t
        deformation.t_chunks   = t_chunks
        deformation.chunk_indx = chunk_indx
        deformation            = self.save2deformation(deformation, lmbda, u)

        for key, value in vars(deformation).items(): setattr(self, key, value)
    
    def finalization(self):
        """
        Plot the applied deformation history
        """
        pass
