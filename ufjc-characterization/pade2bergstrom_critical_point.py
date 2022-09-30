# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import uFJCDiffuseChainScissionCharacterizer, GeneralizeduFJC, latex_formatting_figure, save_current_figure, save_pickle_object, load_pickle_object
from scipy import optimize
import matplotlib.pyplot as plt

class Pade2BergstromCriticalPointCharacterizer(uFJCDiffuseChainScissionCharacterizer):

    def __init__(self):

        uFJCDiffuseChainScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """

        p = self.parameters

        # List of physically relevant kappa_nu 
        # kappa_nu from 1 -> 10000 reduced the curve fit quality over the range of physically relevant kappa_nu
        p.characterizer.kappa_nu_pade2berg_crit_list = [i for i in range(100, 10001)]

    def prefix(self):
        return "pade2bergstrom_critical_point"
    
    def characterization(self):

        def lmbda_c_eq_pade2berg_func_kappa_nu_fit(kappa_nu, n, b):
            return 1./(kappa_nu**n) + b

        cp = self.parameters.characterizer

        single_chain_list         = [GeneralizeduFJC(rate_dependence = 'rate_independent', nu = 25, zeta_nu_char = 100, kappa_nu = cp.kappa_nu_pade2berg_crit_list[kappa_nu_indx]) for kappa_nu_indx in range(len(cp.kappa_nu_pade2berg_crit_list))]
        lmbda_c_eq_pade2berg_crit = [single_chain.pade2berg_crit_func()[1] for single_chain in single_chain_list]

        popt, pcov = optimize.curve_fit(lmbda_c_eq_pade2berg_func_kappa_nu_fit, cp.kappa_nu_pade2berg_crit_list, lmbda_c_eq_pade2berg_crit)
        n = popt[0]
        b = popt[1]
        print("n = {}".format(n))
        print("b = {}".format(b))

        lmbda_c_eq_pade2berg_crit_curve_fit = [lmbda_c_eq_pade2berg_func_kappa_nu_fit(cp.kappa_nu_pade2berg_crit_list[kappa_nu_indx], n, b) for kappa_nu_indx in range(len(cp.kappa_nu_pade2berg_crit_list))]

        save_pickle_object(self.savedir, cp.kappa_nu_pade2berg_crit_list, "kappa_nu_pade2berg_crit")
        save_pickle_object(self.savedir, lmbda_c_eq_pade2berg_crit, "lmbda_c_eq_pade2berg_crit")
        save_pickle_object(self.savedir, lmbda_c_eq_pade2berg_crit_curve_fit, "lmbda_c_eq_pade2berg_crit_curve_fit")

    def finalization(self):
        ppp = self.parameters.post_processing

        kappa_nu_pade2berg_crit             = load_pickle_object(self.savedir, "kappa_nu_pade2berg_crit")
        lmbda_c_eq_pade2berg_crit           = load_pickle_object(self.savedir, "lmbda_c_eq_pade2berg_crit")
        lmbda_c_eq_pade2berg_crit_curve_fit = load_pickle_object(self.savedir, "lmbda_c_eq_pade2berg_crit_curve_fit")

        # plot results
        latex_formatting_figure(ppp)

        fig = plt.figure()
        plt.plot(kappa_nu_pade2berg_crit, lmbda_c_eq_pade2berg_crit, linestyle='-', color='blue', alpha=1, linewidth=2.5, label=r'$\textrm{Exact calculation}$')
        plt.plot(kappa_nu_pade2berg_crit, lmbda_c_eq_pade2berg_crit_curve_fit, linestyle='--', color='red', alpha=1, linewidth=2.5, label=r'$\textrm{Curve fit}$')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\kappa_{\nu}$', 30, r'$(\lambda_c^{eq})^{pade2berg}$', 30, "pade2berg-critical-point-equilibrium-chain-stretch-vs-kappa_nu")

if __name__ == '__main__':

    characterizer = Pade2BergstromCriticalPointCharacterizer()
    # characterizer.characterization()
    characterizer.finalization()