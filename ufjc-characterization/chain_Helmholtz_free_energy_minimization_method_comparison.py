# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import uFJCDiffuseChainScissionCharacterizer, GeneralizeduFJC, latex_formatting_figure, save_current_figure, save_current_figure_no_labels
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class ChainHelmholtzFreeEnergyMinimizationMethodComparisonCharacterizer(uFJCDiffuseChainScissionCharacterizer):

    def __init__(self):

        uFJCDiffuseChainScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """

        p = self.parameters

        p.characterizer.lmbda_c_eq_inc                      = 0.001
        p.characterizer.lmbda_c_eq_min                      = 0.01
        p.characterizer.zeta_nu_char_lmbda_c_eq_crit_factor = 1.3
        p.characterizer.kappa_nu_lmbda_c_eq_crit_factor     = 1.5

        p.characterizer.color_list = ['orange', 'blue', 'green', 'red', 'purple']
    
    def prefix(self):
        return "chain_Helmholtz_free_energy_minimization_method_comparison"
    
    def characterization(self):

        # Define the nondimensional Helmholtz free energy per Kuhn segment for segment stretches below the critical segment stretch
        def subcrit_psi_cnu_func(lmbda_nu, lmbda_c_eq, zeta_nu_char, kappa_nu):
            def s_cnu_func(lmbda_comp_nu):
                return 0.0602726941412868*lmbda_comp_nu**8 + 0.00103401966455583*lmbda_comp_nu**7 - 0.162726405850159*lmbda_comp_nu**6 - 0.00150537112388157*lmbda_comp_nu**5 \
                    - 0.00350216312906114*lmbda_comp_nu**4 - 0.00254138511870934*lmbda_comp_nu**3 + 0.488744117329956*lmbda_comp_nu**2 + 0.0071635921950366*lmbda_comp_nu \
                        - 0.999999503781195*np.log(1.00000000002049 - lmbda_comp_nu) - 0.992044340231098*np.log(lmbda_comp_nu + 0.98498877114821) - 0.0150047080499398
            
            def u_nu_func(lmbda_nu, zeta_nu_char, kappa_nu):
                return 0.5*kappa_nu*( lmbda_nu - 1. )**2 - zeta_nu_char

            lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.

            return u_nu_func(lmbda_nu, zeta_nu_char, kappa_nu) + s_cnu_func(lmbda_comp_nu)
        
        # Define the nondimensional Helmholtz free energy per Kuhn segment for segment stretches above the critical segment stretch
        def supercrit_psi_cnu_func(lmbda_nu, lmbda_c_eq, zeta_nu_char, kappa_nu):
            def s_cnu_func(lmbda_comp_nu):
                return 0.0602726941412868*lmbda_comp_nu**8 + 0.00103401966455583*lmbda_comp_nu**7 - 0.162726405850159*lmbda_comp_nu**6 - 0.00150537112388157*lmbda_comp_nu**5 \
                    - 0.00350216312906114*lmbda_comp_nu**4 - 0.00254138511870934*lmbda_comp_nu**3 + 0.488744117329956*lmbda_comp_nu**2 + 0.0071635921950366*lmbda_comp_nu \
                        - 0.999999503781195*np.log(1.00000000002049 - lmbda_comp_nu) - 0.992044340231098*np.log(lmbda_comp_nu + 0.98498877114821) - 0.0150047080499398
            
            def u_nu_func(lmbda_nu, zeta_nu_char, kappa_nu):
                return -zeta_nu_char**2 / ( 2.*kappa_nu*( lmbda_nu - 1. )**2 )

            lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.

            return u_nu_func(lmbda_nu, zeta_nu_char, kappa_nu) + s_cnu_func(lmbda_comp_nu)
        
        cp = self.parameters.characterizer

        # Evaluate zeta_nu_char
        psi_minimization_zeta_nu_char_single_chain_list = [GeneralizeduFJC(rate_dependence = 'rate_independent', nu = cp.nu_single_chain_list[1], zeta_nu_char = cp.psi_minimization_zeta_nu_char_single_chain_list[single_chain_num_indx], kappa_nu = cp.kappa_nu_single_chain_list[2]) for single_chain_num_indx in range(len(cp.psi_minimization_zeta_nu_char_single_chain_list))] # nu=125, kappa_nu=1000

        psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk       = [0. for single_chain_num_indx in range(len(psi_minimization_zeta_nu_char_single_chain_list))]
        psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk         = [0. for single_chain_num_indx in range(len(psi_minimization_zeta_nu_char_single_chain_list))]
        psi_minimization_zeta_nu_char_lmbda_nu_psimin___single_chain_chunk  = [0. for single_chain_num_indx in range(len(psi_minimization_zeta_nu_char_single_chain_list))]
        psi_minimization_zeta_nu_char_lmbda_nu_mthderr___single_chain_chunk = [0. for single_chain_num_indx in range(len(psi_minimization_zeta_nu_char_single_chain_list))]

        for single_chain_num_indx in range(len(psi_minimization_zeta_nu_char_single_chain_list)):
            single_chain = psi_minimization_zeta_nu_char_single_chain_list[single_chain_num_indx]

            lmbda_c_eq_max = cp.zeta_nu_char_lmbda_c_eq_crit_factor*single_chain.lmbda_c_eq_crit

            # Define the values of the equilibrium chain stretch to calculate over
            lmbda_c_eq_num_steps = int(np.around((lmbda_c_eq_max-cp.lmbda_c_eq_min)/cp.lmbda_c_eq_inc)) + 1
            lmbda_c_eq_steps     = np.linspace(cp.lmbda_c_eq_min, lmbda_c_eq_max, lmbda_c_eq_num_steps)

            # Make arrays to allocate results
            lmbda_c_eq       = []
            lmbda_nu         = []
            lmbda_nu_psimin  = []
            lmbda_nu_mthderr = []

            # Calculate results through specified equilibrium chain stretch values
            for lmbda_c_eq_indx in range(lmbda_c_eq_num_steps):
                lmbda_c_eq_val = lmbda_c_eq_steps[lmbda_c_eq_indx]
                lmbda_nu_val   = single_chain.lmbda_nu_func(lmbda_c_eq_val)
                
                if lmbda_c_eq_val < 1.0:
                    result = optimize.minimize(subcrit_psi_cnu_func, 1.0, args=(lmbda_c_eq_val, single_chain.zeta_nu_char, single_chain.kappa_nu))
                
                elif lmbda_c_eq_val < single_chain.lmbda_c_eq_crit:
                    result = optimize.minimize(subcrit_psi_cnu_func, lmbda_c_eq_val, args=(lmbda_c_eq_val, single_chain.zeta_nu_char, single_chain.kappa_nu))
                
                else:
                    result = optimize.minimize(supercrit_psi_cnu_func, lmbda_c_eq_val, args=(lmbda_c_eq_val, single_chain.zeta_nu_char, single_chain.kappa_nu))
                
                lmbda_nu_psimin_val  = result.x
                lmbda_nu_mthderr_val = np.abs((lmbda_nu_psimin_val-lmbda_nu_val)/lmbda_nu_val)*100
                
                lmbda_c_eq.append(lmbda_c_eq_val)
                lmbda_nu.append(lmbda_nu_val)
                lmbda_nu_psimin.append(lmbda_nu_psimin_val)
                lmbda_nu_mthderr.append(lmbda_nu_mthderr_val)
            
            psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]       = lmbda_c_eq
            psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk[single_chain_num_indx]         = lmbda_nu
            psi_minimization_zeta_nu_char_lmbda_nu_psimin___single_chain_chunk[single_chain_num_indx]  = lmbda_nu_psimin
            psi_minimization_zeta_nu_char_lmbda_nu_mthderr___single_chain_chunk[single_chain_num_indx] = lmbda_nu_mthderr
        
        self.psi_minimization_zeta_nu_char_single_chain_list = psi_minimization_zeta_nu_char_single_chain_list
        
        self.psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk       = psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk
        self.psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk         = psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk
        self.psi_minimization_zeta_nu_char_lmbda_nu_psimin___single_chain_chunk  = psi_minimization_zeta_nu_char_lmbda_nu_psimin___single_chain_chunk
        self.psi_minimization_zeta_nu_char_lmbda_nu_mthderr___single_chain_chunk = psi_minimization_zeta_nu_char_lmbda_nu_mthderr___single_chain_chunk


        # Evaluate kappa_nu
        psi_minimization_kappa_nu_single_chain_list = [GeneralizeduFJC(rate_dependence = 'rate_independent', nu = cp.nu_single_chain_list[1], zeta_nu_char = cp.zeta_nu_char_single_chain_list[2], kappa_nu = cp.psi_minimization_kappa_nu_single_chain_list[single_chain_num_indx]) for single_chain_num_indx in range(len(cp.psi_minimization_kappa_nu_single_chain_list))] # nu=125, zeta_nu_char=100

        psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk       = [0. for single_chain_num_indx in range(len(psi_minimization_kappa_nu_single_chain_list))]
        psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk         = [0. for single_chain_num_indx in range(len(psi_minimization_kappa_nu_single_chain_list))]
        psi_minimization_kappa_nu_lmbda_nu_psimin___single_chain_chunk  = [0. for single_chain_num_indx in range(len(psi_minimization_kappa_nu_single_chain_list))]
        psi_minimization_kappa_nu_lmbda_nu_mthderr___single_chain_chunk = [0. for single_chain_num_indx in range(len(psi_minimization_kappa_nu_single_chain_list))]

        for single_chain_num_indx in range(len(psi_minimization_kappa_nu_single_chain_list)):
            single_chain = psi_minimization_kappa_nu_single_chain_list[single_chain_num_indx]

            lmbda_c_eq_max = cp.kappa_nu_lmbda_c_eq_crit_factor*single_chain.lmbda_c_eq_crit

            # Define the values of the equilibrium chain stretch to calculate over
            lmbda_c_eq_num_steps = int(np.around((lmbda_c_eq_max-cp.lmbda_c_eq_min)/cp.lmbda_c_eq_inc)) + 1
            lmbda_c_eq_steps     = np.linspace(cp.lmbda_c_eq_min, lmbda_c_eq_max, lmbda_c_eq_num_steps)

            # Make arrays to allocate results
            lmbda_c_eq       = []
            lmbda_nu         = []
            lmbda_nu_psimin  = []
            lmbda_nu_mthderr = []

            # Calculate results through specified equilibrium chain stretch values
            for lmbda_c_eq_indx in range(lmbda_c_eq_num_steps):
                lmbda_c_eq_val = lmbda_c_eq_steps[lmbda_c_eq_indx]
                lmbda_nu_val   = single_chain.lmbda_nu_func(lmbda_c_eq_val)
                
                if lmbda_c_eq_val < 1.0:
                    result = optimize.minimize(subcrit_psi_cnu_func, 1.0, args=(lmbda_c_eq_val, single_chain.zeta_nu_char, single_chain.kappa_nu))
                
                elif lmbda_c_eq_val < single_chain.lmbda_c_eq_crit:
                    result = optimize.minimize(subcrit_psi_cnu_func, lmbda_c_eq_val, args=(lmbda_c_eq_val, single_chain.zeta_nu_char, single_chain.kappa_nu))
                
                else:
                    result = optimize.minimize(supercrit_psi_cnu_func, lmbda_c_eq_val, args=(lmbda_c_eq_val, single_chain.zeta_nu_char, single_chain.kappa_nu))
                
                lmbda_nu_psimin_val  = result.x
                lmbda_nu_mthderr_val = np.abs((lmbda_nu_psimin_val-lmbda_nu_val)/lmbda_nu_val)*100
                
                lmbda_c_eq.append(lmbda_c_eq_val)
                lmbda_nu.append(lmbda_nu_val)
                lmbda_nu_psimin.append(lmbda_nu_psimin_val)
                lmbda_nu_mthderr.append(lmbda_nu_mthderr_val)
            
            psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]       = lmbda_c_eq
            psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk[single_chain_num_indx]         = lmbda_nu
            psi_minimization_kappa_nu_lmbda_nu_psimin___single_chain_chunk[single_chain_num_indx]  = lmbda_nu_psimin
            psi_minimization_kappa_nu_lmbda_nu_mthderr___single_chain_chunk[single_chain_num_indx] = lmbda_nu_mthderr
        
        self.psi_minimization_kappa_nu_single_chain_list = psi_minimization_kappa_nu_single_chain_list
        
        self.psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk       = psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk
        self.psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk         = psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk
        self.psi_minimization_kappa_nu_lmbda_nu_psimin___single_chain_chunk  = psi_minimization_kappa_nu_lmbda_nu_psimin___single_chain_chunk
        self.psi_minimization_kappa_nu_lmbda_nu_mthderr___single_chain_chunk = psi_minimization_kappa_nu_lmbda_nu_mthderr___single_chain_chunk

    def finalization(self):
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        # Evaluate zeta_nu_char
        lmbda_c_eq_max = 0
        lmbda_nu_max   = 0

        fig = plt.figure()
        for single_chain_num_indx in range(len(self.psi_minimization_zeta_nu_char_single_chain_list)):
            lmbda_c_eq      = self.psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]
            lmbda_nu        = self.psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_crit   = self.psi_minimization_zeta_nu_char_single_chain_list[single_chain_num_indx].lmbda_nu_crit
            lmbda_c_eq_crit = self.psi_minimization_zeta_nu_char_single_chain_list[single_chain_num_indx].lmbda_c_eq_crit
            lmbda_c_eq_max  = max([lmbda_c_eq_max, lmbda_c_eq[-1]])
            lmbda_nu_max    = max([lmbda_nu_max, lmbda_nu[-1]])
            
            plt.vlines(x=lmbda_c_eq_crit, ymin=lmbda_nu[0]-0.05, ymax=lmbda_nu_crit, linestyle=':', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=1)
            plt.hlines(y=lmbda_nu_crit, xmin=lmbda_c_eq[0]-0.05, xmax=lmbda_c_eq_crit, linestyle=':', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=1)
        
        for single_chain_num_indx in range(len(self.psi_minimization_zeta_nu_char_single_chain_list)):
            lmbda_c_eq       = self.psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]
            lmbda_nu         = self.psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_psimin  = self.psi_minimization_zeta_nu_char_lmbda_nu_psimin___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_mthderr = self.psi_minimization_zeta_nu_char_lmbda_nu_mthderr___single_chain_chunk[single_chain_num_indx]
            plt.plot(lmbda_c_eq, lmbda_nu, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5, label=cp.psi_minimization_zeta_nu_char_label_single_chain_list[single_chain_num_indx])
            plt.plot(lmbda_c_eq, lmbda_nu_psimin, linestyle='-.', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)

        plt.legend(loc='best')
        plt.xlim([-0.05, lmbda_c_eq_max + 0.1])
        plt.ylim([0.95, lmbda_nu_max + 0.1])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_c^{eq}$', 30, r'$\lambda_{\nu}$', 30, "zeta_nu_char-lmbda_nu-vs-lmbda_c_eq-method-comparison")
        
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        for single_chain_num_indx in range(len(self.psi_minimization_zeta_nu_char_single_chain_list)):
            lmbda_c_eq       = self.psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]
            lmbda_nu         = self.psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_psimin  = self.psi_minimization_zeta_nu_char_lmbda_nu_psimin___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_mthderr = self.psi_minimization_zeta_nu_char_lmbda_nu_mthderr___single_chain_chunk[single_chain_num_indx]
            lmbda_c_eq_crit  = self.psi_minimization_zeta_nu_char_single_chain_list[single_chain_num_indx].lmbda_c_eq_crit

            lmbda_c_eq__lmbda_c_eq_crit = [x/lmbda_c_eq_crit for x in lmbda_c_eq]

            ax1.plot(lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5, label=cp.psi_minimization_zeta_nu_char_label_single_chain_list[single_chain_num_indx])
            ax1.plot(lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu_psimin, linestyle='-.', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)
            
            ax2.semilogy(lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu_mthderr, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5, label=cp.psi_minimization_zeta_nu_char_label_single_chain_list[single_chain_num_indx])
        
        ax1.legend(loc='best')
        ax1.set_ylim([0.95, lmbda_nu_max + 0.1])
        ax1.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        plt.xlim([-0.05, cp.zeta_nu_char_lmbda_c_eq_crit_factor + 0.05])
        plt.xlabel(r'$\lambda_c^{eq}/(\lambda_c^{eq})^{crit}$', fontsize=30)
        save_current_figure_no_labels(self.savedir, "zeta_nu_char-lmbda_nu-vs-lmbda_c_eq__lmbda_c_eq_crit-method-comparison")

        
        # Evaluate zeta_nu_char
        lmbda_c_eq_max = 0
        lmbda_nu_max   = 0

        fig = plt.figure()
        for single_chain_num_indx in range(len(self.psi_minimization_kappa_nu_single_chain_list)):
            lmbda_c_eq      = self.psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]
            lmbda_nu        = self.psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_crit   = self.psi_minimization_kappa_nu_single_chain_list[single_chain_num_indx].lmbda_nu_crit
            lmbda_c_eq_crit = self.psi_minimization_kappa_nu_single_chain_list[single_chain_num_indx].lmbda_c_eq_crit
            lmbda_c_eq_max  = max([lmbda_c_eq_max, lmbda_c_eq[-1]])
            lmbda_nu_max    = max([lmbda_nu_max, lmbda_nu[-1]])
            
            plt.vlines(x=lmbda_c_eq_crit, ymin=lmbda_nu[0]-0.05, ymax=lmbda_nu_crit, linestyle=':', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=1)
            plt.hlines(y=lmbda_nu_crit, xmin=lmbda_c_eq[0]-0.05, xmax=lmbda_c_eq_crit, linestyle=':', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=1)
        
        for single_chain_num_indx in range(len(self.psi_minimization_kappa_nu_single_chain_list)):
            lmbda_c_eq       = self.psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]
            lmbda_nu         = self.psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_psimin  = self.psi_minimization_kappa_nu_lmbda_nu_psimin___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_mthderr = self.psi_minimization_kappa_nu_lmbda_nu_mthderr___single_chain_chunk[single_chain_num_indx]
            plt.plot(lmbda_c_eq, lmbda_nu, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5, label=cp.psi_minimization_kappa_nu_label_single_chain_list[single_chain_num_indx])
            plt.plot(lmbda_c_eq, lmbda_nu_psimin, linestyle='-.', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)

        plt.legend(loc='best')
        plt.xlim([-0.05, lmbda_c_eq_max + 0.1])
        plt.ylim([0.95, lmbda_nu_max + 0.1])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_c^{eq}$', 30, r'$\lambda_{\nu}$', 30, "kappa_nu-lmbda_nu-vs-lmbda_c_eq-method-comparison")
        
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        for single_chain_num_indx in range(len(self.psi_minimization_kappa_nu_single_chain_list)):
            lmbda_c_eq       = self.psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk[single_chain_num_indx]
            lmbda_nu         = self.psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_psimin  = self.psi_minimization_kappa_nu_lmbda_nu_psimin___single_chain_chunk[single_chain_num_indx]
            lmbda_nu_mthderr = self.psi_minimization_kappa_nu_lmbda_nu_mthderr___single_chain_chunk[single_chain_num_indx]
            lmbda_c_eq_crit  = self.psi_minimization_kappa_nu_single_chain_list[single_chain_num_indx].lmbda_c_eq_crit

            lmbda_c_eq__lmbda_c_eq_crit = [x/lmbda_c_eq_crit for x in lmbda_c_eq]

            ax1.plot(lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5, label=cp.psi_minimization_kappa_nu_label_single_chain_list[single_chain_num_indx])
            ax1.plot(lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu_psimin, linestyle='-.', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5)
            
            ax2.semilogy(lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu_mthderr, linestyle='-', color=cp.color_list[single_chain_num_indx], alpha=1, linewidth=2.5, label=cp.psi_minimization_kappa_nu_label_single_chain_list[single_chain_num_indx])
        
        ax1.legend(loc='best')
        ax1.set_ylim([0.95, lmbda_nu_max + 0.1])
        ax1.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        plt.xlim([-0.05, cp.kappa_nu_lmbda_c_eq_crit_factor + 0.05])
        plt.xlabel(r'$\lambda_c^{eq}/(\lambda_c^{eq})^{crit}$', fontsize=30)
        save_current_figure_no_labels(self.savedir, "kappa_nu-lmbda_nu-vs-lmbda_c_eq__lmbda_c_eq_crit-method-comparison")

if __name__ == '__main__':

    characterizer = ChainHelmholtzFreeEnergyMinimizationMethodComparisonCharacterizer()
    characterizer.characterization()
    characterizer.finalization()