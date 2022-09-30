# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import uFJCDiffuseChainScissionCharacterizer, GeneralizeduFJC, latex_formatting_figure, save_current_figure_no_labels
import numpy as np
import matplotlib.pyplot as plt

class RateIndependentSegmentScissionCharacterizer(uFJCDiffuseChainScissionCharacterizer):

    def __init__(self):

        uFJCDiffuseChainScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """
        pass
    
    def prefix(self):
        return "rate_independent_segment_scission"
    
    def characterization(self):

        cp = self.parameters.characterizer

        single_chain = GeneralizeduFJC(rate_dependence = 'rate_independent', nu = cp.nu_single_chain_list[1], zeta_nu_char = cp.zeta_nu_char_single_chain_list[2], kappa_nu = cp.kappa_nu_single_chain_list[2]) # nu=125, zeta_nu_char=100, kappa_nu=1000

        # Define the applied segment stretch values to calculate over
        lmbda_nu_hat_num_steps = int(np.around((single_chain.lmbda_nu_crit-single_chain.lmbda_nu_ref)/single_chain.lmbda_nu_hat_inc)) + 1
        lmbda_nu_hat_steps     = np.linspace(single_chain.lmbda_nu_ref, single_chain.lmbda_nu_crit, lmbda_nu_hat_num_steps)
        
        # Make arrays to allocate results
        lmbda_nu_hat        = []
        e_nu_sci_hat        = []
        epsilon_nu_sci_hat  = []
        p_nu_sci_hat        = []
        p_nu_sur_hat        = []
        epsilon_nu_diss_hat = []

        # initialization
        lmbda_nu_hat_max = 0
        
        # Calculate results through specified applied segment stretch values
        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val       = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max       = max([lmbda_nu_hat_max, lmbda_nu_hat_val])
            e_nu_sci_hat_val       = single_chain.e_nu_sci_hat_func(lmbda_nu_hat_val)
            epsilon_nu_sci_hat_val = single_chain.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
            p_nu_sci_hat_val       = single_chain.p_nu_sci_hat_func(lmbda_nu_hat_val)
            p_nu_sur_hat_val       = single_chain.p_nu_sur_hat_func(lmbda_nu_hat_val)
            
            if lmbda_nu_hat_indx == 0: # initialization
                epsilon_nu_diss_hat_val = 0
            else:
                epsilon_nu_diss_hat_val = single_chain.epsilon_nu_diss_hat_func(lmbda_nu_hat_max, lmbda_nu_hat_val, lmbda_nu_hat[lmbda_nu_hat_indx-1], epsilon_nu_diss_hat[lmbda_nu_hat_indx-1])
            
            lmbda_nu_hat.append(lmbda_nu_hat_val)
            e_nu_sci_hat.append(e_nu_sci_hat_val)
            epsilon_nu_sci_hat.append(epsilon_nu_sci_hat_val)
            p_nu_sci_hat.append(p_nu_sci_hat_val)
            p_nu_sur_hat.append(p_nu_sur_hat_val)
            epsilon_nu_diss_hat.append(epsilon_nu_diss_hat_val)
        
        overline_e_nu_sci_hat        = [e_nu_sci_hat_val/single_chain.zeta_nu_char for e_nu_sci_hat_val in e_nu_sci_hat]
        overline_epsilon_nu_sci_hat  = [epsilon_nu_sci_hat_val/single_chain.zeta_nu_char for epsilon_nu_sci_hat_val in epsilon_nu_sci_hat]
        overline_epsilon_nu_diss_hat = [epsilon_nu_diss_hat_val/single_chain.zeta_nu_char for epsilon_nu_diss_hat_val in epsilon_nu_diss_hat]
        
        self.single_chain = single_chain
        
        self.lmbda_nu_hat                 = lmbda_nu_hat
        self.overline_e_nu_sci_hat        = overline_e_nu_sci_hat
        self.overline_epsilon_nu_sci_hat  = overline_epsilon_nu_sci_hat
        self.p_nu_sci_hat                 = p_nu_sci_hat
        self.p_nu_sur_hat                 = p_nu_sur_hat
        self.overline_epsilon_nu_diss_hat = overline_epsilon_nu_diss_hat

    def finalization(self):
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        ax1.plot(self.lmbda_nu_hat, self.overline_e_nu_sci_hat, linestyle=':', color='blue', alpha=1, linewidth=2.5, label=r'$\overline{\hat{e}_{\nu}^{sci}}$')
        ax1.plot(self.lmbda_nu_hat, self.overline_epsilon_nu_sci_hat, linestyle='-', color='blue', alpha=1, linewidth=2.5, label=r'$\overline{\hat{\varepsilon}_{\nu}^{sci}}$')
        ax1.plot(self.lmbda_nu_hat, self.overline_epsilon_nu_diss_hat, linestyle=(0, (3, 1, 1, 1)), color='blue', alpha=1, linewidth=2.5, label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        ax2.plot(self.lmbda_nu_hat, self.p_nu_sci_hat, linestyle='-', color='blue', alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sci}$')
        ax2.plot(self.lmbda_nu_hat, self.p_nu_sur_hat, linestyle=':', color='blue', alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sur}$')

        ax1.legend(loc='best', handlelength=3)
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_ylabel(r'$\overline{\hat{e}_{\nu}^{sci}},~\overline{\hat{\varepsilon}_{\nu}^{sci}},~\overline{\hat{\varepsilon}_{\nu}^{diss}}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best')
        ax2.set_ylim([-0.05, 1.05])
        ax2.set_ylabel(r'$\hat{p}_{\nu}^{sci},~\hat{p}_{\nu}^{sur}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=30)
        save_current_figure_no_labels(self.savedir, "rate-independent-segment-scission-indicators-vs-lmbda_nu_hat")

if __name__ == '__main__':

    characterizer = RateIndependentSegmentScissionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()