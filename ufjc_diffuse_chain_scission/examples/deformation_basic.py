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

        AppliedDeformation.__init__(self, self.parameters, self.F_func)
    
    def prefix(self):
        return "deformation"
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """
        p = self.parameters

        # Parameters used in F_func
        strain_rate = 1 # 1/sec
        max_strain = p.deformation.t_max*strain_rate/2

        p.deformation.strain_rate = strain_rate
        p.deformation.max_strain  = max_strain

        # Deformation stepping calculations
        max_F_dot = strain_rate
        t_scale = 1./max_F_dot # sec
        t_step_modify_factor = 1e-2
        t_step = np.around(t_step_modify_factor*t_scale, 2) # sec
        t_step_chunk_modify_factor = 3
        t_step_chunk = t_step_chunk_modify_factor*t_step # sec
        
        p.deformation.max_F_dot                  = max_F_dot
        p.deformation.t_step_modify_factor       = t_step_modify_factor
        p.deformation.t_step                     = t_step
        p.deformation.t_step_chunk_modify_factor = t_step_chunk_modify_factor
        p.deformation.t_step_chunk               = t_step_chunk
    
    def F_func(self, t):
        """
        Function defining the deformation
        """
        dp = self.parameters.deformation

        return 1 + dp.strain_rate*(t-dp.t_min)*np.heaviside(dp.max_strain-dp.strain_rate*(t-dp.t_min), 0.5) \
            + (2*dp.max_strain - dp.strain_rate*(t-dp.t_min))*np.heaviside(dp.strain_rate*(t-dp.t_min)-dp.max_strain, 0.5)
    
    def finalization(self):
        """
        Plot the applied deformation history
        """
        
        latex_formatting_figure(self.parameters.post_processing)
        
        fig = plt.figure()
        plt.plot(self.t, self.lmbda_1, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_1$', 30, "uniaxial-rate-independent-t-vs-lmbda_1")

        fig = plt.figure()
        plt.plot(self.t, self.u, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$u$', 30, "uniaxial-rate-independent-t-vs-u")

if __name__ == '__main__':

    AppliedDeformationHistory().finalization()