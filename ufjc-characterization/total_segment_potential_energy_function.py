# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import uFJCDiffuseChainScissionCharacterizer, GeneralizeduFJC, latex_formatting_figure, save_current_figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class TotalSegmentPotentialEnergyFunctionCharacterizer(uFJCDiffuseChainScissionCharacterizer):

    def __init__(self):

        uFJCDiffuseChainScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """

        p = self.parameters

        p.characterizer.lmbda_nu_inc      = 0.0001
        p.characterizer.lmbda_nu_min      = 0.50
        p.characterizer.lmbda_nu_max      = 2.50
        p.characterizer.lmbda_nu_hat_num  = 4

        p.characterizer.color_list = ['blue', 'green', 'red', 'purple']
    
    def prefix(self):
        return "total_segment_potential_energy_function"
    
    def characterization(self):

        cp = self.parameters.characterizer

        single_chain = GeneralizeduFJC(rate_dependence = 'rate_independent', nu = cp.nu_single_chain_list[1], zeta_nu_char = cp.zeta_nu_char_single_chain_list[2], kappa_nu = cp.kappa_nu_single_chain_list[2]) # nu=125, zeta_nu_char=100, kappa_nu=1000

        # Define the applied segment stretch values to the chain
        lmbda_nu_hat_steps = np.linspace(single_chain.lmbda_nu_ref, single_chain.lmbda_nu_crit, cp.lmbda_nu_hat_num)
        # Define the segment stretch values to calculate over
        lmbda_nu_num_steps = int(np.around((cp.lmbda_nu_max-cp.lmbda_nu_min)/cp.lmbda_nu_inc)) + 1
        lmbda_nu_steps     = np.linspace(cp.lmbda_nu_min, cp.lmbda_nu_max, lmbda_nu_num_steps)
        
        lmbda_nu_hat___hat_chunk                 = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        lmbda_nu_locmin_hat___hat_chunk          = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        u_nu_tot_locmin_hat___hat_chunk          = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        overline_u_nu_tot_locmin_hat___hat_chunk = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        lmbda_nu_locmax_hat___hat_chunk          = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        u_nu_tot_locmax_hat___hat_chunk          = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        overline_u_nu_tot_locmax_hat___hat_chunk = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        lmbda_nu___hat_chunk                     = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        u_nu_tot_hat___hat_chunk                 = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        overline_u_nu_tot_hat___hat_chunk        = [0. for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        
        for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num):
            lmbda_nu_hat        = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_locmin_hat = single_chain.lmbda_nu_locmin_hat_func(lmbda_nu_hat)
            u_nu_tot_locmin_hat = single_chain.u_nu_tot_hat_func(lmbda_nu_locmin_hat, lmbda_nu_hat)
            lmbda_nu_locmax_hat = single_chain.lmbda_nu_locmax_hat_func(lmbda_nu_hat)
            if lmbda_nu_locmax_hat == np.inf:
                u_nu_tot_locmax_hat = np.nan
            else:
                u_nu_tot_locmax_hat = single_chain.u_nu_tot_hat_func(lmbda_nu_locmax_hat, lmbda_nu_hat)
            
            # Make arrays to allocate results
            lmbda_nu     = []
            u_nu_tot_hat = []
            
            # Calculate results through specified segment stretch values
            for lmbda_nu_indx in range(lmbda_nu_num_steps):
                lmbda_nu_val     = lmbda_nu_steps[lmbda_nu_indx]
                u_nu_tot_hat_val = single_chain.u_nu_tot_hat_func(lmbda_nu_val, lmbda_nu_hat)
                
                lmbda_nu.append(lmbda_nu_val)
                u_nu_tot_hat.append(u_nu_tot_hat_val)
            
            overline_u_nu_tot_locmin_hat = u_nu_tot_locmin_hat/single_chain.zeta_nu_char
            overline_u_nu_tot_locmax_hat = u_nu_tot_locmax_hat/single_chain.zeta_nu_char
            overline_u_nu_tot_hat        = [x/single_chain.zeta_nu_char for x in u_nu_tot_hat]

            lmbda_nu_hat___hat_chunk[lmbda_nu_hat_indx]                 = lmbda_nu_hat
            lmbda_nu_locmin_hat___hat_chunk[lmbda_nu_hat_indx]          = lmbda_nu_locmin_hat
            u_nu_tot_locmin_hat___hat_chunk[lmbda_nu_hat_indx]          = u_nu_tot_locmin_hat
            overline_u_nu_tot_locmin_hat___hat_chunk[lmbda_nu_hat_indx] = overline_u_nu_tot_locmin_hat
            lmbda_nu_locmax_hat___hat_chunk[lmbda_nu_hat_indx]          = lmbda_nu_locmax_hat
            u_nu_tot_locmax_hat___hat_chunk[lmbda_nu_hat_indx]          = u_nu_tot_locmax_hat
            overline_u_nu_tot_locmax_hat___hat_chunk[lmbda_nu_hat_indx] = overline_u_nu_tot_locmax_hat
            lmbda_nu___hat_chunk[lmbda_nu_hat_indx]                     = lmbda_nu
            u_nu_tot_hat___hat_chunk[lmbda_nu_hat_indx]                 = u_nu_tot_hat
            overline_u_nu_tot_hat___hat_chunk[lmbda_nu_hat_indx]        = overline_u_nu_tot_hat
        
        self.single_chain = single_chain
        
        self.lmbda_nu_hat___hat_chunk                 = lmbda_nu_hat___hat_chunk
        self.lmbda_nu_locmin_hat___hat_chunk          = lmbda_nu_locmin_hat___hat_chunk
        self.u_nu_tot_locmin_hat___hat_chunk          = u_nu_tot_locmin_hat___hat_chunk
        self.overline_u_nu_tot_locmin_hat___hat_chunk = overline_u_nu_tot_locmin_hat___hat_chunk
        self.lmbda_nu_locmax_hat___hat_chunk          = lmbda_nu_locmax_hat___hat_chunk
        self.u_nu_tot_locmax_hat___hat_chunk          = u_nu_tot_locmax_hat___hat_chunk
        self.overline_u_nu_tot_locmax_hat___hat_chunk = overline_u_nu_tot_locmax_hat___hat_chunk
        self.lmbda_nu___hat_chunk                     = lmbda_nu___hat_chunk
        self.u_nu_tot_hat___hat_chunk                 = u_nu_tot_hat___hat_chunk
        self.overline_u_nu_tot_hat___hat_chunk        = overline_u_nu_tot_hat___hat_chunk

    def finalization(self):
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        # retrieve plot formatting specifications
        chain_parameter_label_list     = [r'$\hat{\lambda}_{\nu}='+str(round(self.lmbda_nu_hat___hat_chunk[lmbda_nu_hat_indx], 4))+'$' for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num)]
        chain_parameter_label_list[-1] = r'$\hat{\lambda}_{\nu}=\lambda_{\nu}^{crit}='+str(round(self.lmbda_nu_hat___hat_chunk[-1], 4))+'$'
        
        fig = plt.figure()
        for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num):
            lmbda_nu_locmin_hat = self.lmbda_nu_locmin_hat___hat_chunk[lmbda_nu_hat_indx]
            u_nu_tot_locmin_hat = self.u_nu_tot_locmin_hat___hat_chunk[lmbda_nu_hat_indx]
            lmbda_nu_locmax_hat = self.lmbda_nu_locmax_hat___hat_chunk[lmbda_nu_hat_indx]
            u_nu_tot_locmax_hat = self.u_nu_tot_locmax_hat___hat_chunk[lmbda_nu_hat_indx]
            lmbda_nu            = self.lmbda_nu___hat_chunk[lmbda_nu_hat_indx]
            u_nu_tot_hat        = self.u_nu_tot_hat___hat_chunk[lmbda_nu_hat_indx]
            plt.plot(lmbda_nu, u_nu_tot_hat, linestyle='-', color=cp.color_list[lmbda_nu_hat_indx], alpha=1, linewidth=2.5, label=chain_parameter_label_list[lmbda_nu_hat_indx])
            plt.plot(lmbda_nu_locmin_hat, u_nu_tot_locmin_hat, marker='o', color=cp.color_list[lmbda_nu_hat_indx], alpha=1, linewidth=2.5)
            if lmbda_nu_locmax_hat == np.inf:
                continue
            else:
                plt.plot(lmbda_nu_locmax_hat, u_nu_tot_locmax_hat, marker='s', color=cp.color_list[lmbda_nu_hat_indx], alpha=1, linewidth=2.5)
        plt.legend(loc='best')
        plt.xlim([lmbda_nu[0], lmbda_nu[-1]])
        plt.ylim([-500, 4])
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_{\nu}$', 30, r'$\hat{u}_{\nu}^{tot}$', 30, "u_nu_tot_hat-vs-lmbda_nu")

        fig = plt.figure()
        for lmbda_nu_hat_indx in range(cp.lmbda_nu_hat_num):
            lmbda_nu_locmin_hat          = self.lmbda_nu_locmin_hat___hat_chunk[lmbda_nu_hat_indx]
            overline_u_nu_tot_locmin_hat = self.overline_u_nu_tot_locmin_hat___hat_chunk[lmbda_nu_hat_indx]
            lmbda_nu_locmax_hat          = self.lmbda_nu_locmax_hat___hat_chunk[lmbda_nu_hat_indx]
            overline_u_nu_tot_locmax_hat = self.overline_u_nu_tot_locmax_hat___hat_chunk[lmbda_nu_hat_indx]
            lmbda_nu                     = self.lmbda_nu___hat_chunk[lmbda_nu_hat_indx]
            overline_u_nu_tot_hat        = self.overline_u_nu_tot_hat___hat_chunk[lmbda_nu_hat_indx]
            plt.plot(lmbda_nu, overline_u_nu_tot_hat, linestyle='-', color=cp.color_list[lmbda_nu_hat_indx], alpha=1, linewidth=2.5, label=chain_parameter_label_list[lmbda_nu_hat_indx])
            plt.plot(lmbda_nu_locmin_hat, overline_u_nu_tot_locmin_hat, marker='o', color=cp.color_list[lmbda_nu_hat_indx], alpha=1, linewidth=2.5)
            if lmbda_nu_locmax_hat == np.inf:
                continue
            else:
                plt.plot(lmbda_nu_locmax_hat, overline_u_nu_tot_locmax_hat, marker='s', color=cp.color_list[lmbda_nu_hat_indx], alpha=1, linewidth=2.5)
        plt.legend(loc='best')
        plt.xlim([lmbda_nu[0], lmbda_nu[-1]])
        plt.ylim([-5, 0.05])
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_{\nu}$', 30, r'$\overline{\hat{u}}_{\nu}^{tot}$', 30, "overline_u_nu_tot_hat-vs-lmbda_nu")

if __name__ == '__main__':

    characterizer = TotalSegmentPotentialEnergyFunctionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()