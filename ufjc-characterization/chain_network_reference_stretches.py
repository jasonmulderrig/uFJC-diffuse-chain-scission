# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import uFJCDiffuseChainScissionCharacterizer, GeneralizeduFJC, latex_formatting_figure, save_current_figure_no_labels, save_pickle_object, load_pickle_object
import numpy as np
import matplotlib.pyplot as plt

class ChainNetworkReferenceStretches(uFJCDiffuseChainScissionCharacterizer):

    def __init__(self):

        uFJCDiffuseChainScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """

        p = self.parameters

        p.characterizer.color_list     = ['orange', 'blue', 'green']
        p.characterizer.linestyle_list = ['-', '--', ':']
    
    def prefix(self):
        return "chain_network_reference_stretches"
    
    def characterization(self):

        cp = self.parameters.characterizer

        single_chain_chain_network_list = [[[GeneralizeduFJC(rate_dependence = 'rate_independent', nu = cp.nu_chain_network_list[nu_indx], zeta_nu_char = cp.zeta_nu_char_chain_network_list[zeta_nu_char_indx], kappa_nu = cp.kappa_nu_chain_network_list[kappa_nu_indx]) for nu_indx in range(len(cp.nu_chain_network_list))] for kappa_nu_indx in range(len(cp.kappa_nu_chain_network_list))] for zeta_nu_char_indx in range(len(cp.zeta_nu_char_chain_network_list))]
        A_nu_chain_network              = [[[single_chain.A_nu for single_chain in chain_network___kappa_nu_chunk] for chain_network___kappa_nu_chunk in chain_network___zeta_nu_char_chunk] for chain_network___zeta_nu_char_chunk in single_chain_chain_network_list]
        Lambda_nu_ref_chain_network     = [[[single_chain.Lambda_nu_ref for single_chain in chain_network___kappa_nu_chunk] for chain_network___kappa_nu_chunk in chain_network___zeta_nu_char_chunk] for chain_network___zeta_nu_char_chunk in single_chain_chain_network_list]

        save_pickle_object(self.savedir, A_nu_chain_network, "A_nu_chain_network")
        save_pickle_object(self.savedir, Lambda_nu_ref_chain_network, "Lambda_nu_ref_chain_network")

    def finalization(self):
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        A_nu_chain_network          = load_pickle_object(self.savedir, "A_nu_chain_network")
        Lambda_nu_ref_chain_network = load_pickle_object(self.savedir, "Lambda_nu_ref_chain_network")

        # plot results
        latex_formatting_figure(ppp)

        # comparison with the inextensible Gaussian chain assumption
        A_nu_list = A_nu_chain_network[1][1][:] # zeta_nu_char = 100, kappa_nu = 1000

        inext_gaussian_A_nu_list     = [1/np.sqrt(nu_val) for nu_val in cp.nu_chain_network_list]
        inext_gaussian_A_nu_err_list = [np.abs((inext_gaussian_A_nu_val-A_nu_val)/A_nu_val)*100 for inext_gaussian_A_nu_val, A_nu_val in zip(inext_gaussian_A_nu_list, A_nu_list)]

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        
        ax1.semilogx(cp.nu_chain_network_list, A_nu_list, linestyle='-', color='blue', alpha=1, linewidth=2.5, label=r'$u\textrm{FJC}$')
        ax1.semilogx(cp.nu_chain_network_list, inext_gaussian_A_nu_list, linestyle='--', color='red', alpha=1, linewidth=2.5, label=r'$\textrm{inextensible Gaussian chain}$')
        ax1.legend(loc='best')
        ax1.set_ylabel(r'$\mathcal{A}_{\nu}$', fontsize=30)
        ax1.grid(True, alpha=0.25)
        
        ax2.loglog(cp.nu_chain_network_list, inext_gaussian_A_nu_err_list, linestyle='-', color='blue', alpha=1, linewidth=2.5)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=30)
        ax2.grid(True, alpha=0.25)
        
        plt.xlabel(r'$\nu$', fontsize=30)
        save_current_figure_no_labels(self.savedir, "A_nu-gen-ufjc-model-framework-and-inextensible-Gaussian-chain-comparison")

        # retrieve and plot results for constant zeta_b_char in each subplot
        fig, axs = plt.subplots(len(cp.zeta_nu_char_chain_network_list), sharex=True)
        axs = axs.ravel()
        
        for zeta_nu_char_indx in range(len(cp.zeta_nu_char_chain_network_list)):
            for kappa_nu_indx in range(len(cp.kappa_nu_chain_network_list)):
                A_nu_list = A_nu_chain_network[zeta_nu_char_indx][kappa_nu_indx][:]
                axs[zeta_nu_char_indx].semilogx(cp.nu_chain_network_list, A_nu_list, linestyle=cp.linestyle_list[kappa_nu_indx], color=cp.color_list[kappa_nu_indx], alpha=1, linewidth=2.5, label=cp.kappa_nu_label_chain_network_list[kappa_nu_indx])
            axs[zeta_nu_char_indx].set_ylabel(r'$\mathcal{A}_{\nu}$', fontsize=20)
            axs[zeta_nu_char_indx].legend(loc='best')
            axs[zeta_nu_char_indx].grid(True, alpha=0.25)
            axs[zeta_nu_char_indx].set_title(cp.zeta_nu_char_label_chain_network_list[zeta_nu_char_indx], fontsize=15)
        
        plt.xlabel(r'$\nu$', fontsize=20)
        save_current_figure_no_labels(self.savedir, "zeta_nu_char-A_nu-vs-nu")

        fig, axs = plt.subplots(len(cp.zeta_nu_char_chain_network_list), sharex=True)
        axs = axs.ravel()
        
        for zeta_nu_char_indx in range(len(cp.zeta_nu_char_chain_network_list)):
            for kappa_nu_indx in range(len(cp.kappa_nu_chain_network_list)):
                Lambda_nu_ref_list = Lambda_nu_ref_chain_network[zeta_nu_char_indx][kappa_nu_indx][:]
                axs[zeta_nu_char_indx].semilogx(cp.nu_chain_network_list, Lambda_nu_ref_list, linestyle=cp.linestyle_list[kappa_nu_indx], color=cp.color_list[kappa_nu_indx], alpha=1, linewidth=2.5, label=cp.kappa_nu_label_chain_network_list[kappa_nu_indx])
            axs[zeta_nu_char_indx].set_ylabel(r'$\Lambda_{\nu}^{ref}$', fontsize=20)
            axs[zeta_nu_char_indx].legend(loc='best')
            axs[zeta_nu_char_indx].grid(True, alpha=0.25)
            axs[zeta_nu_char_indx].set_title(cp.zeta_nu_char_label_chain_network_list[zeta_nu_char_indx], fontsize=15)
        
        plt.xlabel(r'$\nu$', fontsize=20)
        save_current_figure_no_labels(self.savedir, "zeta_nu_char-Lambda_nu_ref-vs-nu")
        
        # retrieve and plot results for constant kappa_b in each subplot
        fig, axs = plt.subplots(len(cp.kappa_nu_chain_network_list), sharex=True)
        axs = axs.ravel()
        
        for kappa_nu_indx in range(len(cp.kappa_nu_chain_network_list)):
            for zeta_nu_char_indx in range(len(cp.zeta_nu_char_chain_network_list)):
                A_nu_list = A_nu_chain_network[zeta_nu_char_indx][kappa_nu_indx][:]
                axs[kappa_nu_indx].semilogx(cp.nu_chain_network_list, A_nu_list, linestyle=cp.linestyle_list[zeta_nu_char_indx], color=cp.color_list[zeta_nu_char_indx], alpha=1, linewidth=2.5, label=cp.zeta_nu_char_label_chain_network_list[zeta_nu_char_indx])
            axs[kappa_nu_indx].set_ylabel(r'$\mathcal{A}_{\nu}$', fontsize=20)
            axs[kappa_nu_indx].legend(loc='best')
            axs[kappa_nu_indx].grid(True, alpha=0.25)
            axs[kappa_nu_indx].set_title(cp.kappa_nu_label_chain_network_list[kappa_nu_indx], fontsize=15)
        
        plt.xlabel(r'$\nu$', fontsize=20)
        save_current_figure_no_labels(self.savedir, "kappa_nu-A_nu-vs-nu")

        fig, axs = plt.subplots(len(cp.kappa_nu_chain_network_list), sharex=True)
        axs = axs.ravel()
        
        for kappa_nu_indx in range(len(cp.kappa_nu_chain_network_list)):
            for zeta_nu_char_indx in range(len(cp.zeta_nu_char_chain_network_list)):
                Lambda_nu_ref_list = Lambda_nu_ref_chain_network[zeta_nu_char_indx][kappa_nu_indx][:]
                axs[kappa_nu_indx].semilogx(cp.nu_chain_network_list, Lambda_nu_ref_list, linestyle=cp.linestyle_list[zeta_nu_char_indx], color=cp.color_list[zeta_nu_char_indx], alpha=1, linewidth=2.5, label=cp.zeta_nu_char_label_chain_network_list[zeta_nu_char_indx])
            axs[kappa_nu_indx].set_ylabel(r'$\Lambda_{\nu}^{ref}$', fontsize=20)
            axs[kappa_nu_indx].legend(loc='best')
            axs[kappa_nu_indx].grid(True, alpha=0.25)
            axs[kappa_nu_indx].set_title(cp.kappa_nu_label_chain_network_list[kappa_nu_indx], fontsize=15)
        
        plt.xlabel(r'$\nu$', fontsize=20)
        save_current_figure_no_labels(self.savedir, "kappa_nu-Lambda_nu_ref-vs-nu")

if __name__ == '__main__':

    characterizer = ChainNetworkReferenceStretches()
    # characterizer.characterization()
    characterizer.finalization()