# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import uFJCDiffuseChainScissionCharacterizer, GeneralizeduFJC, latex_formatting_figure, save_current_figure, save_current_figure_no_labels
import numpy as np
from pynverse import inversefunc
import matplotlib.pyplot as plt

class ChainMechanicalResponseCharacterizer(uFJCDiffuseChainScissionCharacterizer):

    def __init__(self):

        uFJCDiffuseChainScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """

        p = self.parameters

        p.characterizer.lmbda_c_eq_min       = 0.001
        p.characterizer.lmbda_c_eq_num_steps = 2501

        p.characterizer.color_list = ['orange', 'blue', 'green', 'red', 'purple']
    
    def prefix(self):
        return "chain_mechanical_response"
    
    def characterization(self):

        # Define the inverse Langevin function needed for force calculations
        langevin    = lambda x: 1./np.tanh(x) - 1./x # Langevin function
        invlangevin = inversefunc(langevin, domain=[1.e-16, 1e6])

        cp = self.parameters.characterizer

        # Evaluate zeta_nu_char
        chain_mech_resp_zeta_nu_char_single_chain_list = [GeneralizeduFJC(rate_dependence = 'rate_independent', nu = cp.nu_single_chain_list[1], zeta_nu_char = cp.zeta_nu_char_single_chain_list[single_chain_num_indx], kappa_nu = cp.kappa_nu_single_chain_list[2]) for single_chain_num_indx in range(len(cp.zeta_nu_char_single_chain_list))] # nu=125, kappa_nu=1000

        chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk = [0. for single_chain_num_indx in range(len(chain_mech_resp_zeta_nu_char_single_chain_list))]
        chain_mech_resp_zeta_nu_char_lmbda_nu___single_chain_chunk   = [0. for single_chain_num_indx in range(len(chain_mech_resp_zeta_nu_char_single_chain_list))]
        chain_mech_resp_zeta_nu_char_xi_c_exact___single_chain_chunk = [0. for single_chain_num_indx in range(len(chain_mech_resp_zeta_nu_char_single_chain_list))]

        for single_chain_num_indx in range(len(chain_mech_resp_zeta_nu_char_single_chain_list)):
            single_chain = chain_mech_resp_zeta_nu_char_single_chain_list[single_chain_num_indx]

            # Define the values of the equilibrium chain stretch to calculate over
            lmbda_c_eq_steps = np.linspace(cp.lmbda_c_eq_min, single_chain.lmbda_c_eq_crit, cp.lmbda_c_eq_num_steps)

            # Make arrays to allocate results
            lmbda_c_eq = []
            lmbda_nu   = []
            xi_c_exact = []

            # Calculate results through specified equilibrium chain stretch values
            for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_steps):
                lmbda_c_eq_val    = lmbda_c_eq_steps[lmbda_c_eq_indx]
                lmbda_nu_val      = single_chain.lmbda_nu_func(lmbda_c_eq_val)
                lmbda_comp_nu_val = lmbda_c_eq_val - lmbda_nu_val + 1.
                xi_c_exact_val    = invlangevin(lmbda_comp_nu_val)
                
                lmbda_c_eq.append(lmbda_c_eq_val)
                lmbda_nu.append(lmbda_nu_val)
                xi_c_exact.append(xi_c_exact_val)
            
            chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk[single_chain_num_indx] = lmbda_c_eq
            chain_mech_resp_zeta_nu_char_lmbda_nu___single_chain_chunk[single_chain_num_indx]   = lmbda_nu
            chain_mech_resp_zeta_nu_char_xi_c_exact___single_chain_chunk[single_chain_num_indx] = xi_c_exact
        
        self.chain_mech_resp_zeta_nu_char_single_chain_list = chain_mech_resp_zeta_nu_char_single_chain_list
        
        self.chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk = chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk
        self.chain_mech_resp_zeta_nu_char_lmbda_nu___single_chain_chunk   = chain_mech_resp_zeta_nu_char_lmbda_nu___single_chain_chunk
        self.chain_mech_resp_zeta_nu_char_xi_c_exact___single_chain_chunk = chain_mech_resp_zeta_nu_char_xi_c_exact___single_chain_chunk

        # Evaluate kappa_nu
        chain_mech_resp_kappa_nu_single_chain_list = [GeneralizeduFJC(rate_dependence = 'rate_independent', nu = cp.nu_single_chain_list[1], zeta_nu_char = cp.zeta_nu_char_single_chain_list[2], kappa_nu = cp.kappa_nu_single_chain_list[single_chain_num_indx]) for single_chain_num_indx in range(len(cp.kappa_nu_single_chain_list))] # nu=125, kappa_nu=1000

        chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk = [0. for single_chain_num_indx in range(len(chain_mech_resp_kappa_nu_single_chain_list))]
        chain_mech_resp_kappa_nu_lmbda_nu___single_chain_chunk   = [0. for single_chain_num_indx in range(len(chain_mech_resp_kappa_nu_single_chain_list))]
        chain_mech_resp_kappa_nu_xi_c_exact___single_chain_chunk = [0. for single_chain_num_indx in range(len(chain_mech_resp_kappa_nu_single_chain_list))]

        for single_chain_num_indx in range(len(chain_mech_resp_kappa_nu_single_chain_list)):
            single_chain = chain_mech_resp_kappa_nu_single_chain_list[single_chain_num_indx]

            # Define the values of the equilibrium chain stretch to calculate over
            lmbda_c_eq_steps = np.linspace(cp.lmbda_c_eq_min, single_chain.lmbda_c_eq_crit, cp.lmbda_c_eq_num_steps)

            # Make arrays to allocate results
            lmbda_c_eq = []
            lmbda_nu   = []
            xi_c_exact = []

            # Calculate results through specified equilibrium chain stretch values
            for lmbda_c_eq_indx in range(cp.lmbda_c_eq_num_steps):
                lmbda_c_eq_val    = lmbda_c_eq_steps[lmbda_c_eq_indx]
                lmbda_nu_val      = single_chain.lmbda_nu_func(lmbda_c_eq_val)
                lmbda_comp_nu_val = lmbda_c_eq_val - lmbda_nu_val + 1.
                xi_c_exact_val    = invlangevin(lmbda_comp_nu_val)
                
                lmbda_c_eq.append(lmbda_c_eq_val)
                lmbda_nu.append(lmbda_nu_val)
                xi_c_exact.append(xi_c_exact_val)
            
            chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk[single_chain_num_indx] = lmbda_c_eq
            chain_mech_resp_kappa_nu_lmbda_nu___single_chain_chunk[single_chain_num_indx]   = lmbda_nu
            chain_mech_resp_kappa_nu_xi_c_exact___single_chain_chunk[single_chain_num_indx] = xi_c_exact
        
        self.chain_mech_resp_kappa_nu_single_chain_list = chain_mech_resp_kappa_nu_single_chain_list
        
        self.chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk = chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk
        self.chain_mech_resp_kappa_nu_lmbda_nu___single_chain_chunk   = chain_mech_resp_kappa_nu_lmbda_nu___single_chain_chunk
        self.chain_mech_resp_kappa_nu_xi_c_exact___single_chain_chunk = chain_mech_resp_kappa_nu_xi_c_exact___single_chain_chunk

    def finalization(self):
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        # Evaluate zeta_nu_char
        lmbda_c_eq_max = 0
        
        # retrieve and plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        for single_chain_num_indx in range(len(self.chain_mech_resp_zeta_nu_char_single_chain_list)):
            lmbda_c_eq     = self.chain_mech_resp_zeta_nu_char_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]
            lmbda_nu       = self.chain_mech_resp_zeta_nu_char_lmbda_nu___single_chain_chunk[single_chain_num_indx]
            xi_c_exact     = self.chain_mech_resp_zeta_nu_char_xi_c_exact___single_chain_chunk[single_chain_num_indx]
            lmbda_c_eq_max = max([lmbda_c_eq_max, lmbda_c_eq[-1]])
            ax1.semilogy(lmbda_c_eq, xi_c_exact, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5, label=cp.zeta_nu_char_label_single_chain_list[single_chain_num_indx])
            ax1.plot(lmbda_c_eq[-1], xi_c_exact[-1], marker='x', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)
            ax2.plot(lmbda_c_eq, lmbda_nu, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)
            ax2.plot(lmbda_c_eq[-1], lmbda_nu[-1], marker='x', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)
        
        ax1.legend(loc='best')
        ax1.set_ylim([1e-2, 1e4])
        ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        plt.xlim([-0.05, lmbda_c_eq_max + 0.05])
        plt.xlabel(r'$\lambda_c^{eq}$', fontsize=30)
        save_current_figure_no_labels(self.savedir, "zeta_nu_char-xi_c-lmbda_nu-vs-lmbda_c_eq")

        # Evaluate kappa_nu
        lmbda_c_eq_max = 0
        
        # retrieve and plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        for single_chain_num_indx in range(len(self.chain_mech_resp_kappa_nu_single_chain_list)):
            lmbda_c_eq     = self.chain_mech_resp_kappa_nu_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]
            lmbda_nu       = self.chain_mech_resp_kappa_nu_lmbda_nu___single_chain_chunk[single_chain_num_indx]
            xi_c_exact     = self.chain_mech_resp_kappa_nu_xi_c_exact___single_chain_chunk[single_chain_num_indx]
            lmbda_c_eq_max = max([lmbda_c_eq_max, lmbda_c_eq[-1]])
            ax1.semilogy(lmbda_c_eq, xi_c_exact, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5, label=cp.kappa_nu_label_single_chain_list[single_chain_num_indx])
            ax1.plot(lmbda_c_eq[-1], xi_c_exact[-1], marker='x', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)
            ax2.plot(lmbda_c_eq, lmbda_nu, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)
            ax2.plot(lmbda_c_eq[-1], lmbda_nu[-1], marker='x', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)
        
        ax1.legend(loc='best')
        ax1.set_ylim([1e-2, 1e4])
        ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        plt.xlim([-0.05, lmbda_c_eq_max + 0.05])
        plt.xlabel(r'$\lambda_c^{eq}$', fontsize=30)
        save_current_figure_no_labels(self.savedir, "kappa_nu-xi_c-lmbda_nu-vs-lmbda_c_eq")

if __name__ == '__main__':

    characterizer = ChainMechanicalResponseCharacterizer()
    characterizer.characterization()
    characterizer.finalization()