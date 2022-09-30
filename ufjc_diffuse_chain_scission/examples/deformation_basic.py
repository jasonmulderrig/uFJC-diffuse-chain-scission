# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import generate_savedir, default_parameters, AppliedDeformation, latex_formatting_figure, save_current_figure
import numpy as np
import matplotlib.pyplot as plt

# Problem
class AppliedDeformationHistory(AppliedDeformation):

    def __init__(self):

        # Parameters
        self.parameters = default_parameters()
        self.set_user_parameters()
        
        # Setup filesystem
        self.savedir = generate_savedir(self.prefix())

        self.deformation = AppliedDeformation(self.parameters, self.F_func, self.initialize_lmbda, self.store_initialized_lmbda, self.calculate_lmbda_func, self.store_calculated_lmbda, self.store_calculated_lmbda_chunk_post_processing, self.calculate_u_func, self.save2deformation)
    
    def prefix(self):
        return "deformation"
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """
        dp = self.parameters.deformation

        dp.deformation_type = "uniaxial"

        # Parameters used in F_func
        strain_rate = 1 # 1/sec
        max_strain = dp.t_max*strain_rate/2

        dp.strain_rate = strain_rate
        dp.max_strain  = max_strain

        # Deformation stepping calculations
        max_F_dot = strain_rate
        t_scale = 1./max_F_dot # sec
        t_step_modify_factor = 1e-2
        t_step = np.around(t_step_modify_factor*t_scale, 2) # sec
        t_step_chunk_modify_factor = 3
        t_step_chunk = t_step_chunk_modify_factor*t_step # sec
        
        dp.max_F_dot                  = max_F_dot
        dp.t_step_modify_factor       = t_step_modify_factor
        dp.t_step                     = t_step
        dp.t_step_chunk_modify_factor = t_step_chunk_modify_factor
        dp.t_step_chunk               = t_step_chunk
    
    def F_func(self, t):
        """
        Function defining the deformation
        """
        dp = self.parameters.deformation

        return 1 + dp.strain_rate*(t-dp.t_min)*np.heaviside(dp.max_strain-dp.strain_rate*(t-dp.t_min), 0.5) \
            + (2*dp.max_strain - dp.strain_rate*(t-dp.t_min))*np.heaviside(dp.strain_rate*(t-dp.t_min)-dp.max_strain, 0.5)
    
    def initialize_lmbda(self):
        lmbda_1        = [] # unitless
        lmbda_1_chunks = [] # unitless

        return lmbda_1, lmbda_1_chunks
    
    def store_initialized_lmbda(self, lmbda):
        lmbda_1_val = 1 # assuming no pre-stretching
        
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]
        
        lmbda_1.append(lmbda_1_val)
        lmbda_1_chunks.append(lmbda_1_val)
        
        return lmbda_1, lmbda_1_chunks
    
    def calculate_lmbda_func(self, t_val):
        lmbda_1_val = self.F_func(t_val)

        return lmbda_1_val
    
    def store_calculated_lmbda(self, lmbda, lmbda_val):
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]
        lmbda_1_val    = lmbda_val
        
        lmbda_1.append(lmbda_1_val)
        
        return lmbda_1, lmbda_1_chunks
    
    def store_calculated_lmbda_chunk_post_processing(self, lmbda, lmbda_val):
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]
        lmbda_1_val    = lmbda_val
        
        lmbda_1_chunks.append(lmbda_1_val)
        
        return lmbda_1, lmbda_1_chunks
    
    def calculate_u_func(self, lmbda):
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]

        u_1        = [lmbda_1_val-1 for lmbda_1_val in lmbda_1]
        u_1_chunks = [lmbda_1_chunks_val-1 for lmbda_1_chunks_val in lmbda_1_chunks]

        return u_1, u_1_chunks
    
    def save2deformation(self, deformation, lmbda, u):
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]

        u_1        = u[0]
        u_1_chunks = u[1]

        deformation.lmbda_1        = lmbda_1
        deformation.lmbda_1_chunks = lmbda_1_chunks
        deformation.u_1            = u_1
        deformation.u_1_chunks     = u_1_chunks

        return deformation
    
    def finalization(self):
        """
        Plot the applied deformation history
        """

        deformation = self.deformation
        ppp = self.parameters.post_processing
        
        latex_formatting_figure(ppp)
        
        fig = plt.figure()
        plt.plot(deformation.t, deformation.lmbda_1, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_1$', 30, "uniaxial-rate-independent-t-vs-lmbda_1")

        fig = plt.figure()
        plt.plot(deformation.t, deformation.u_1, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$u_1$', 30, "uniaxial-rate-independent-t-vs-u_1")

if __name__ == '__main__':

    AppliedDeformationHistory().finalization()