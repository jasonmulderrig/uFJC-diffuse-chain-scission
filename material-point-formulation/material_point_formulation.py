# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import uFJCDiffuseChainScissionCharacterizer, EqualStrainGeneralizeduFJC, EqualForceGeneralizeduFJC, GeneralizeduFJC, latex_formatting_figure, save_current_figure, save_current_figure_no_labels
import numpy as np
from math import floor, log10
from scipy import optimize
from scipy import constants
import matplotlib.pyplot as plt

class MaterialPointFormulationCharacterizer(uFJCDiffuseChainScissionCharacterizer):

    def __init__(self):

        uFJCDiffuseChainScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """

        p = self.parameters

        # Fundamental material constants
        k_B          = constants.value(u"Boltzmann constant") # J/K
        N_A          = constants.value(u"Avogadro constant") # 1/mol
        h            = constants.value(u"Planck constant") # J/Hz
        hbar         = h/(2*np.pi) # J*sec
        T            = 298 # absolute room temperature, K
        beta         = 1./(k_B*T) # 1/J
        omega_0      = 1./(beta*hbar) # J/(J*sec) = 1/sec
        zeta_nu_char = 300
        kappa_nu     = 2300
        nu_b         = None
        zeta_b_char  = None
        kappa_b      = None

        p.material.k_B          = k_B
        p.material.N_A          = N_A
        p.material.h            = h
        p.material.hbar         = hbar
        p.material.T            = T
        p.material.beta         = beta
        p.material.omega_0      = omega_0
        p.material.zeta_nu_char = zeta_nu_char
        p.material.kappa_nu     = kappa_nu
        p.material.nu_b         = nu_b
        p.material.zeta_b_char  = zeta_b_char
        p.material.kappa_b      = kappa_b

        # Define the chain length statistics in the network
        nu_distribution = "itskov"
        nu_list         = [i for i in range(5, 16)] # nu = 5 -> nu = 15
        nu_min          = min(nu_list)
        nu_bar          = 8
        Delta_nu        = nu_bar-nu_min
        nu_list         = nu_list
        nu_min          = nu_min
        nu_bar          = nu_bar
        Delta_nu        = Delta_nu

        p.material.nu_distribution = nu_distribution
        p.material.nu_list         = nu_list
        p.material.nu_min          = nu_min
        p.material.nu_bar          = nu_bar
        p.material.Delta_nu        = Delta_nu
        p.material.nu_list         = nu_list
        p.material.nu_min          = nu_min
        p.material.nu_bar          = nu_bar
        p.material.Delta_nu        = Delta_nu

        # Define various characteristics of the deformation for the network
        network_model                      = "statistical_mechanics_model"
        phenomenological_model             = "neo_hookean"
        physical_dimension                 = 2
        physical_dimensionality            = "two_dimensional"
        incompressibility_assumption       = "nearly_incompressible"
        macro2micro_deformation_assumption = "nonaffine"
        micro2macro_homogenization_scheme  = "eight_chain_model"
        chain_level_load_sharing           = "equal_strain"
        rate_dependence                    = "rate_independent"
        two_dimensional_formulation        = "plane_strain"
        microdisk_quadrature_order         = 1
        microsphere_quadrature_order       = 1

        p.material.network_model                      = network_model
        p.material.phenomenological_model             = phenomenological_model
        p.material.physical_dimension                 = physical_dimension
        p.material.physical_dimensionality            = physical_dimensionality
        p.material.incompressibility_assumption       = incompressibility_assumption
        p.material.macro2micro_deformation_assumption = macro2micro_deformation_assumption
        p.material.micro2macro_homogenization_scheme  = micro2macro_homogenization_scheme
        p.material.chain_level_load_sharing           = chain_level_load_sharing
        p.material.rate_dependence                    = rate_dependence
        p.material.two_dimensional_formulation        = two_dimensional_formulation
        p.material.microdisk_quadrature_order         = microdisk_quadrature_order
        p.material.microsphere_quadrature_order       = microsphere_quadrature_order

        p.characterizer.lmbda_c_min = 0.001
        p.characterizer.lmbda_c_inc = 0.001
        p.characterizer.d_c_max     = 0.999

    def prefix(self):
        return "material_point_formulation"
    
    def characterization(self):

        def equal_strain_generalized_ufjc_network(characterizer_parameters, material_parameters):
            
            cp = characterizer_parameters
            mp = material_parameters

            # List of single chains obeying the equal strain chain level load sharing assumption
            single_chain_list = [EqualStrainGeneralizeduFJC(chain_level_load_sharing = mp.chain_level_load_sharing, rate_dependence = mp.rate_dependence, omega_0 = mp.omega_0, nu = mp.nu_list[nu_indx], nu_b = mp.nu_b, zeta_b_char = mp.zeta_b_char, kappa_b = mp.kappa_b, zeta_nu_char = mp.zeta_nu_char, kappa_nu = mp.kappa_nu) for nu_indx in range(len(mp.nu_list))]
            
            # Separate out specified parameters
            nu_list                   = [single_chain_list[nu_indx].nu for nu_indx in range(len(single_chain_list))]
            nu_min                    = min(nu_list)
            nu_max                    = max(nu_list)
            nu_num                    = len(nu_list)
            P_nu_list                 = [P_nu(mp, nu_list[nu_indx]) for nu_indx in range(len(nu_list))]
            P_nu_sum                  = np.sum(P_nu_list)
            nu_b                      = single_chain_list[0].nu_b
            zeta_b_char               = single_chain_list[0].zeta_b_char
            kappa_b                   = single_chain_list[0].kappa_b
            zeta_nu_char              = single_chain_list[0].zeta_nu_char
            kappa_nu                  = single_chain_list[0].kappa_nu
            lmbda_nu_ref              = single_chain_list[0].lmbda_nu_ref
            lmbda_c_eq_ref            = single_chain_list[0].lmbda_c_eq_ref
            lmbda_nu_crit             = single_chain_list[0].lmbda_nu_crit
            lmbda_c_eq_crit           = single_chain_list[0].lmbda_c_eq_crit
            xi_c_crit                 = single_chain_list[0].xi_c_crit
            lmbda_nu_pade2berg_crit   = single_chain_list[0].lmbda_nu_pade2berg_crit
            lmbda_c_eq_pade2berg_crit = single_chain_list[0].lmbda_c_eq_pade2berg_crit
            A_nu_list                 = [single_chain_list[nu_indx].A_nu for nu_indx in range(len(single_chain_list))]
            Lambda_nu_ref_list        = [single_chain_list[nu_indx].Lambda_nu_ref for nu_indx in range(len(single_chain_list))]
            
            # Retain specified parameters
            cp.single_chain_list         = single_chain_list
            cp.nu_list                   = nu_list
            cp.nu_min                    = nu_min
            cp.nu_max                    = nu_max
            cp.nu_num                    = nu_num
            cp.P_nu_list                 = P_nu_list
            cp.P_nu_sum                  = P_nu_sum
            cp.nu_b                      = nu_b
            cp.zeta_b_char               = zeta_b_char
            cp.kappa_b                   = kappa_b
            cp.zeta_nu_char              = zeta_nu_char
            cp.kappa_nu                  = kappa_nu
            cp.lmbda_nu_ref              = lmbda_nu_ref
            cp.lmbda_c_eq_ref            = lmbda_c_eq_ref
            cp.lmbda_nu_crit             = lmbda_nu_crit
            cp.lmbda_c_eq_crit           = lmbda_c_eq_crit
            cp.xi_c_crit                 = xi_c_crit
            cp.lmbda_nu_pade2berg_crit   = lmbda_nu_pade2berg_crit
            cp.lmbda_c_eq_pade2berg_crit = lmbda_c_eq_pade2berg_crit
            cp.A_nu_list                 = A_nu_list
            cp.Lambda_nu_ref_list        = Lambda_nu_ref_list

            return cp
        
        # def equal_force_generalized_ufjc_network(characterizer_parameters, material_parameters):
            
        #     cp = characterizer_parameters
        #     mp = material_parameters

        #     # List of single chains
        #     single_chain_list = [GeneralizeduFJC(chain_level_load_sharing = mp.chain_level_load_sharing, rate_dependence = mp.rate_dependence, omega_0 = mp.omega_0, nu = nu_list[nu_indx], zeta_b_char = zeta_b_char, kappa_b = kappa_b) for nu_indx in range(len(nu_list))]
            
        #     # Separate out specified parameters
        #     nu_list   = [single_chain_list[nu_indx].nu for nu_indx in range(len(single_chain_list))]
        #     nu_min    = min(nu_list)
        #     nu_max    = max(nu_list)
        #     nu_num    = len(nu_list)
        #     P_nu_list = [P_nu(nu_list[nu_indx]) for nu_indx in range(len(nu_list))]
        #     A_nu_list = [single_chain_list[nu_indx].A_nu for nu_indx in range(len(single_chain_list))]
            
        #     # Calculate the statistical equal force chain parameters
        #     Nu = np.sum([P_nu(nu_list[nu_indx])*nu_list[nu_indx] for nu_indx in range(len(nu_list))])
        #     A  = np.sum([P_nu(nu_list[nu_indx])*A_nu_list[nu_indx]*nu_list[nu_indx] for nu_indx in range(len(nu_list))])/Nu
        #     single_chain = EqualForceGeneralizeduFJC(chain_level_load_sharing = mp.chain_level_load_sharing, rate_dependence = mp.rate_dependence, omega_0 = mp.omega_0, Nu = Nu, A = A, zeta_b_char = zeta_b_char, kappa_b = kappa_b)
            
        #     # Separate out specified parameters
        #     kappa_b                   = single_chain.kappa_b
        #     zeta_b_char               = single_chain.zeta_b_char
        #     lmbda_b_ref               = single_chain.lmbda_b_ref
        #     lmbda_cA_ref              = single_chain.lmbda_cA_ref
        #     lmbda_b_crit              = single_chain.lmbda_b_crit
        #     lmbda_cA_crit             = single_chain.lmbda_cA_crit
        #     lmbda_b_pade2berg_crit    = single_chain.lmbda_b_pade2berg_crit
        #     lmbda_cA_pade2berg_crit   = single_chain.lmbda_cA_pade2berg_crit
        #     lmbda_b_bsci_hat_rms      = single_chain.lmbda_b_bsci_hat_rms
        #     epsilon_b_sci_hat_rms     = single_chain.epsilon_b_sci_hat_rms
        #     lmbda_b_csci_hat_rms      = single_chain.lmbda_b_csci_hat_rms
        #     epsilon_c_sci_hat_rms     = single_chain.epsilon_c_sci_hat_rms
        #     epsilon_c_sci_hat_rms__Nu = epsilon_c_sci_hat_rms/Nu
        #     Lambda_b_ref              = single_chain.Lambda_b_ref
            
        #     # Retain specified parameters
        #     cp.nu_list                   = nu_list
        #     cp.nu_min                    = nu_min
        #     cp.nu_max                    = nu_max
        #     cp.nu_num                    = nu_num
        #     cp.P_nu_list                 = P_nu_list
        #     cp.Nu                        = Nu
        #     cp.A                         = A
        #     cp.single_chain              = single_chain
        #     cp.kappa_b                   = kappa_b
        #     cp.zeta_b_char               = zeta_b_char
        #     cp.lmbda_b_ref               = lmbda_b_ref
        #     cp.lmbda_cA_ref              = lmbda_cA_ref
        #     cp.lmbda_b_crit              = lmbda_b_crit
        #     cp.lmbda_cA_crit             = lmbda_cA_crit
        #     cp.lmbda_b_pade2berg_crit    = lmbda_b_pade2berg_crit
        #     cp.lmbda_cA_pade2berg_crit   = lmbda_cA_pade2berg_crit
        #     cp.lmbda_b_bsci_hat_rms      = lmbda_b_bsci_hat_rms
        #     cp.epsilon_b_sci_hat_rms     = epsilon_b_sci_hat_rms
        #     cp.lmbda_b_csci_hat_rms      = lmbda_b_csci_hat_rms
        #     cp.epsilon_c_sci_hat_rms     = epsilon_c_sci_hat_rms
        #     cp.epsilon_c_sci_hat_rms__Nu = epsilon_c_sci_hat_rms__Nu
        #     cp.Lambda_b_ref              = Lambda_b_ref

        #     return cp

        def P_nu(material_parameters, nu):
            
            mp = material_parameters

            if mp.nu_distribution == "itskov":
                return (1/(mp.Delta_nu+1))*(1+(1/mp.Delta_nu))**(mp.nu_min-nu)
        
        # def D_c_func(single_chain_list, lmbda_c):
        #     D_c_val = 0.0


        mp = self.parameters.material
        cp = self.parameters.characterizer

        if mp.chain_level_load_sharing == 'equal_strain':
            cp = equal_strain_generalized_ufjc_network(cp, mp)
        elif mp.chain_level_load_sharing == 'equal_force':
            sys.exit("Error: The equal force chain level load sharing framework is not properly derived at this point. Switch to the equal strain chain level load sharing framework.")
            # cp = equal_force_generalized_ufjc_network(cp, mp)
        


        def l_nu_eq_nu_kappa_nu_fit_func(r_nu, f_c):
            
            def f_c_func_l_nu_eq_nu_kappa_nu_fit(r_nu, l_nu_eq, nu, kappa_nu):
                
                # Function to calculate the critical equilibrium chain stretch value, as a function of nondimensional segment stiffness, below and above which the Pade and Bergstrom approximates are to be respectively used
                def lmbda_c_eq_pade2berg_crit_func(kappa_nu):
                    n = 0.818706900266885 # Calculated from scipy optimize curve_fit analysis
                    b = 0.61757545643322586 # Calculated from scipy optimize curve_fit analysis
                    return 1./(kappa_nu**n) + b
                
                # Function to calculate the critical equilibrium chain stretch value
                def lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu):
                    return 1. + np.sqrt(zeta_nu_char/kappa_nu) - np.sqrt(1./(kappa_nu*zeta_nu_char)) 
                
                # Segment stretch as a function of equilibrium chain stretch
                def lmbda_nu_func(lmbda_c_eq):
                    
                    lmbda_nu = []
                    
                    for lmbda_c_eq_indx in range(len(lmbda_c_eq)):
                        lmbda_c_eq_val = lmbda_c_eq[lmbda_c_eq_indx]
                    
                        if lmbda_c_eq_val == 0.: # analytical solution, achieved with the use of the Pade approximate
                            lmbda_nu_val = 1.
                        
                        elif lmbda_c_eq_val < lmbda_c_eq_pade2berg_crit: # Pade approximate
                            alpha_tilde = 1.
                            beta_tilde  = -(( 3.*( kappa_nu + 1. ) + lmbda_c_eq_val*( 2.*kappa_nu + 3. ))/( kappa_nu + 1. ))
                            gamma_tilde = (( 2.*kappa_nu + lmbda_c_eq_val*( 4.*kappa_nu + 6. + lmbda_c_eq_val*( kappa_nu + 3. )))/( kappa_nu + 1. ))
                            delta_tilde = (( 2. - lmbda_c_eq_val*( 2.*kappa_nu + lmbda_c_eq_val*( kappa_nu + 3. + lmbda_c_eq_val )))/( kappa_nu + 1. ))
                            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
                            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
                            
                            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
                            
                            if arccos_arg >= 1.-1.e-14:
                                arccos_arg =  1.-1.e-14
                            elif arccos_arg < -1.+1.e-14:
                                arccos_arg =  -1.+1.e-14
                            
                            lmbda_nu_val = 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)
                        
                        elif lmbda_c_eq_val < lmbda_c_eq_crit: # Bergstrom approximate
                            lmbda_nu_val = ( lmbda_c_eq_val + 1. + np.sqrt( lmbda_c_eq_val**2 - 2.*lmbda_c_eq_val + 1. + 4./kappa_nu ))/2.
                        
                        else: # Bergstrom approximate
                            alpha_tilde = 1.
                            beta_tilde  = -3.
                            gamma_tilde = 3. - zeta_nu_char**2/kappa_nu
                            delta_tilde = zeta_nu_char**2/kappa_nu*lmbda_c_eq_val - 1.
                            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
                            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
                            
                            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
                            
                            if arccos_arg >= 1.-1.e-14:
                                arccos_arg =  1.-1.e-14
                            elif arccos_arg < -1.+1.e-14:
                                arccos_arg =  -1.+1.e-14
                            
                            lmbda_nu_val = 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)
                        
                        lmbda_nu.append(lmbda_nu_val)
                    
                    return np.asarray(lmbda_nu)
                
                # Nondimensional chain force function
                def xi_c_func(lmbda_nu, lmbda_c_eq):
                    # Jedynak R[9,2] inverse Langevin approximate function
                    def inv_L_init_func(lmbda_comp_nu):
                        return lmbda_comp_nu*( 3. - 1.00651*lmbda_comp_nu**2 - 0.962251*lmbda_comp_nu**4 + 1.47353*lmbda_comp_nu**6 - 0.48953*lmbda_comp_nu**8 )/( ( 1. - lmbda_comp_nu )*( 1. + 1.01524*lmbda_comp_nu ) )
                    
                    lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
                    
                    return inv_L_init_func(lmbda_comp_nu)
                
                nu_b                      = l_nu_eq/l_b_eq
                zeta_nu_char              = nu_b*zeta_b_char
                lmbda_c_eq_pade2berg_crit = lmbda_c_eq_pade2berg_crit_func(kappa_nu)
                lmbda_c_eq_crit           = lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu)
                lmbda_c_eq                = r_nu/(nu*l_nu_eq) # nm/nm
                lmbda_nu                  = lmbda_nu_func(lmbda_c_eq)
                xi_c                      = xi_c_func(lmbda_nu, lmbda_c_eq)
                f_c                       = xi_c/(beta*l_nu_eq) # 1/(1/(nN*nm))/nm = nN
                
                return f_c
            
            def f_c_func_nu_kappa_nu_fit(r_nu, nu, kappa_nu):
                
                # Function to calculate the critical equilibrium chain stretch value, as a function of nondimensional segment stiffness, below and above which the Pade and Bergstrom approximates are to be respectively used
                def lmbda_c_eq_pade2berg_crit_func(kappa_nu):
                    n = 0.818706900266885 # Calculated from scipy optimize curve_fit analysis
                    b = 0.61757545643322586 # Calculated from scipy optimize curve_fit analysis
                    return 1./(kappa_nu**n) + b
                
                # Function to calculate the critical equilibrium chain stretch value
                def lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu):
                    return 1. + np.sqrt(zeta_nu_char/kappa_nu) - np.sqrt(1./(kappa_nu*zeta_nu_char)) 
                
                # Segment stretch as a function of equilibrium chain stretch
                def lmbda_nu_func(lmbda_c_eq):
                    
                    lmbda_nu = []
                    
                    for lmbda_c_eq_indx in range(len(lmbda_c_eq)):
                        lmbda_c_eq_val = lmbda_c_eq[lmbda_c_eq_indx]
                    
                        if lmbda_c_eq_val == 0.: # analytical solution, achieved with the use of the Pade approximate
                            lmbda_nu_val = 1.
                        
                        elif lmbda_c_eq_val < lmbda_c_eq_pade2berg_crit: # Pade approximate
                            alpha_tilde = 1.
                            beta_tilde  = -(( 3.*( kappa_nu + 1. ) + lmbda_c_eq_val*( 2.*kappa_nu + 3. ))/( kappa_nu + 1. ))
                            gamma_tilde = (( 2.*kappa_nu + lmbda_c_eq_val*( 4.*kappa_nu + 6. + lmbda_c_eq_val*( kappa_nu + 3. )))/( kappa_nu + 1. ))
                            delta_tilde = (( 2. - lmbda_c_eq_val*( 2.*kappa_nu + lmbda_c_eq_val*( kappa_nu + 3. + lmbda_c_eq_val )))/( kappa_nu + 1. ))
                            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
                            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
                            
                            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
                            
                            if arccos_arg >= 1.-1.e-14:
                                arccos_arg =  1.-1.e-14
                            elif arccos_arg < -1.+1.e-14:
                                arccos_arg =  -1.+1.e-14
                            
                            lmbda_nu_val = 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)
                        
                        elif lmbda_c_eq_val < lmbda_c_eq_crit: # Bergstrom approximate
                            lmbda_nu_val = ( lmbda_c_eq_val + 1. + np.sqrt( lmbda_c_eq_val**2 - 2.*lmbda_c_eq_val + 1. + 4./kappa_nu ))/2.
                        
                        else: # Bergstrom approximate
                            alpha_tilde = 1.
                            beta_tilde  = -3.
                            gamma_tilde = 3. - zeta_nu_char**2/kappa_nu
                            delta_tilde = zeta_nu_char**2/kappa_nu*lmbda_c_eq_val - 1.
                            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
                            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
                            
                            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
                            
                            if arccos_arg >= 1.-1.e-14:
                                arccos_arg =  1.-1.e-14
                            elif arccos_arg < -1.+1.e-14:
                                arccos_arg =  -1.+1.e-14
                            
                            lmbda_nu_val = 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)
                        
                        lmbda_nu.append(lmbda_nu_val)
                    
                    return np.asarray(lmbda_nu)
                
                # Nondimensional chain force function
                def xi_c_func(lmbda_nu, lmbda_c_eq):
                    # Jedynak R[9,2] inverse Langevin approximate function
                    def inv_L_init_func(lmbda_comp_nu):
                        return lmbda_comp_nu*( 3. - 1.00651*lmbda_comp_nu**2 - 0.962251*lmbda_comp_nu**4 + 1.47353*lmbda_comp_nu**6 - 0.48953*lmbda_comp_nu**8 )/( ( 1. - lmbda_comp_nu )*( 1. + 1.01524*lmbda_comp_nu ) )
                    
                    lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
                    
                    return inv_L_init_func(lmbda_comp_nu)
                
                nu_b                      = l_nu_eq/l_b_eq
                zeta_nu_char              = nu_b*zeta_b_char
                lmbda_c_eq_pade2berg_crit = lmbda_c_eq_pade2berg_crit_func(kappa_nu)
                lmbda_c_eq_crit           = lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu)
                lmbda_c_eq                = r_nu/(nu*l_nu_eq) # nm/nm
                lmbda_nu                  = lmbda_nu_func(lmbda_c_eq)
                xi_c                      = xi_c_func(lmbda_nu, lmbda_c_eq)
                f_c                       = xi_c/(beta*l_nu_eq) # 1/(1/(nN*nm))/nm = nN
                
                return f_c
            
            def f_c_func_kappa_nu_fit(r_nu, kappa_nu):
                
                # Function to calculate the critical equilibrium chain stretch value, as a function of nondimensional segment stiffness, below and above which the Pade and Bergstrom approximates are to be respectively used
                def lmbda_c_eq_pade2berg_crit_func(kappa_nu):
                    n = 0.818706900266885 # Calculated from scipy optimize curve_fit analysis
                    b = 0.61757545643322586 # Calculated from scipy optimize curve_fit analysis
                    return 1./(kappa_nu**n) + b
                
                # Function to calculate the critical equilibrium chain stretch value
                def lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu):
                    return 1. + np.sqrt(zeta_nu_char/kappa_nu) - np.sqrt(1./(kappa_nu*zeta_nu_char)) 
                
                # Segment stretch as a function of equilibrium chain stretch
                def lmbda_nu_func(lmbda_c_eq):
                    
                    lmbda_nu = []
                    
                    for lmbda_c_eq_indx in range(len(lmbda_c_eq)):
                        lmbda_c_eq_val = lmbda_c_eq[lmbda_c_eq_indx]
                    
                        if lmbda_c_eq_val == 0.: # analytical solution, achieved with the use of the Pade approximate
                            lmbda_nu_val = 1.
                        
                        elif lmbda_c_eq_val < lmbda_c_eq_pade2berg_crit: # Pade approximate
                            alpha_tilde = 1.
                            beta_tilde  = -(( 3.*( kappa_nu + 1. ) + lmbda_c_eq_val*( 2.*kappa_nu + 3. ))/( kappa_nu + 1. ))
                            gamma_tilde = (( 2.*kappa_nu + lmbda_c_eq_val*( 4.*kappa_nu + 6. + lmbda_c_eq_val*( kappa_nu + 3. )))/( kappa_nu + 1. ))
                            delta_tilde = (( 2. - lmbda_c_eq_val*( 2.*kappa_nu + lmbda_c_eq_val*( kappa_nu + 3. + lmbda_c_eq_val )))/( kappa_nu + 1. ))
                            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
                            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
                            
                            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
                            
                            if arccos_arg >= 1.-1.e-14:
                                arccos_arg =  1.-1.e-14
                            elif arccos_arg < -1.+1.e-14:
                                arccos_arg =  -1.+1.e-14
                            
                            lmbda_nu_val = 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)
                        
                        elif lmbda_c_eq_val < lmbda_c_eq_crit: # Bergstrom approximate
                            lmbda_nu_val = ( lmbda_c_eq_val + 1. + np.sqrt( lmbda_c_eq_val**2 - 2.*lmbda_c_eq_val + 1. + 4./kappa_nu ))/2.
                        
                        else: # Bergstrom approximate
                            alpha_tilde = 1.
                            beta_tilde  = -3.
                            gamma_tilde = 3. - zeta_nu_char**2/kappa_nu
                            delta_tilde = zeta_nu_char**2/kappa_nu*lmbda_c_eq_val - 1.
                            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
                            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
                            
                            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
                            
                            if arccos_arg >= 1.-1.e-14:
                                arccos_arg =  1.-1.e-14
                            elif arccos_arg < -1.+1.e-14:
                                arccos_arg =  -1.+1.e-14
                            
                            lmbda_nu_val = 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)
                        
                        lmbda_nu.append(lmbda_nu_val)
                    
                    return np.asarray(lmbda_nu)
                
                # Nondimensional chain force function
                def xi_c_func(lmbda_nu, lmbda_c_eq):
                    # Jedynak R[9,2] inverse Langevin approximate function
                    def inv_L_init_func(lmbda_comp_nu):
                        return lmbda_comp_nu*( 3. - 1.00651*lmbda_comp_nu**2 - 0.962251*lmbda_comp_nu**4 + 1.47353*lmbda_comp_nu**6 - 0.48953*lmbda_comp_nu**8 )/( ( 1. - lmbda_comp_nu )*( 1. + 1.01524*lmbda_comp_nu ) )
                    
                    lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
                    
                    return inv_L_init_func(lmbda_comp_nu)
                
                nu_b                      = l_nu_eq/l_b_eq
                zeta_nu_char              = nu_b*zeta_b_char
                lmbda_c_eq_pade2berg_crit = lmbda_c_eq_pade2berg_crit_func(kappa_nu)
                lmbda_c_eq_crit           = lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu)
                lmbda_c_eq                = r_nu/(nu*l_nu_eq) # nm/nm
                lmbda_nu                  = lmbda_nu_func(lmbda_c_eq)
                xi_c                      = xi_c_func(lmbda_nu, lmbda_c_eq)
                f_c                       = xi_c/(beta*l_nu_eq) # 1/(1/(nN*nm))/nm = nN
                
                return f_c
            
            r_nu_fit_max = r_nu[-1]
            
            r_nu_fit_num_steps = int(np.around((r_nu_fit_max-cp.r_nu_fit_min)/cp.r_nu_fit_inc)) + 1
            r_nu_fit           = np.linspace(cp.r_nu_fit_min, r_nu_fit_max, r_nu_fit_num_steps) # nm
            
            l_nu_eq_min  = l_b_eq # nm
            
            popt, pcov = optimize.curve_fit(f_c_func_l_nu_eq_nu_kappa_nu_fit, r_nu, f_c, bounds=((l_nu_eq_min, cp.nu_min, cp.kappa_nu_min), (cp.l_nu_eq_max, cp.nu_max, cp.kappa_nu_max)))
            
            l_nu_eq  = popt[0]
            nu       = popt[1]
            kappa_nu = popt[2]
            
            l_nu_eq_init  = l_nu_eq
            nu_init       = nu
            kappa_nu_init = kappa_nu
            
            nu_b = l_nu_eq/l_b_eq
            
            print('l_nu_eq = {}'.format(l_nu_eq))
            print('nu = {}'.format(nu))
            print('kappa_nu = {}'.format(kappa_nu))
            
            f_c_fit           = f_c_func_l_nu_eq_nu_kappa_nu_fit(r_nu_fit, l_nu_eq, nu, kappa_nu)
            r_nu_fit_intgr_nu = np.copy(r_nu_fit)
            
            nu_b_floor = np.floor(nu_b)
            nu_b_ceil  = np.ceil(nu_b)
            
            l_nu_eq_floor = nu_b_floor*l_b_eq
            l_nu_eq_ceil  = nu_b_ceil*l_b_eq
            
            l_nu_eq        = l_nu_eq_floor
            popt, pcov     = optimize.curve_fit(f_c_func_nu_kappa_nu_fit, r_nu, f_c, bounds=((cp.nu_min, cp.kappa_nu_min), (cp.nu_max, cp.kappa_nu_max)))
            nu_floor       = popt[0]
            kappa_nu_floor = popt[1]
            residual_floor = np.linalg.norm( f_c - f_c_func_l_nu_eq_nu_kappa_nu_fit(r_nu, l_nu_eq_floor, nu_floor, kappa_nu_floor))
            
            l_nu_eq       = l_nu_eq_ceil
            popt, pcov    = optimize.curve_fit(f_c_func_nu_kappa_nu_fit, r_nu, f_c, bounds=((cp.nu_min, cp.kappa_nu_min), (cp.nu_max, cp.kappa_nu_max)))
            nu_ceil       = popt[0]
            kappa_nu_ceil = popt[1]
            residual_ceil = np.linalg.norm( f_c - f_c_func_l_nu_eq_nu_kappa_nu_fit(r_nu, l_nu_eq_ceil, nu_ceil, kappa_nu_ceil))
            
            if residual_floor <= residual_ceil:
                l_nu_eq = l_nu_eq_floor
                nu      = nu_floor
            else:
                l_nu_eq = l_nu_eq_ceil
                nu      = nu_ceil
            
            print('\nl_nu_eq = {}'.format(l_nu_eq))
            print('nu = {}'.format(nu))
            
            nu_floor = np.floor(nu)
            nu_ceil  = np.ceil(nu)
            
            nu             = nu_floor
            popt, pcov     = optimize.curve_fit(f_c_func_kappa_nu_fit, r_nu, f_c, bounds=(cp.kappa_nu_min, cp.kappa_nu_max))
            kappa_nu_floor = popt[0]
            residual_floor = np.linalg.norm( f_c - f_c_func_l_nu_eq_nu_kappa_nu_fit(r_nu, l_nu_eq, nu_floor, kappa_nu_floor))
            
            nu            = nu_ceil
            popt, pcov    = optimize.curve_fit(f_c_func_kappa_nu_fit, r_nu, f_c, bounds=(cp.kappa_nu_min, cp.kappa_nu_max))
            kappa_nu_ceil = popt[0]
            residual_ceil = np.linalg.norm( f_c - f_c_func_l_nu_eq_nu_kappa_nu_fit(r_nu, l_nu_eq, nu_ceil, kappa_nu_ceil))
            
            if residual_floor <= residual_ceil:
                nu       = nu_floor
                kappa_nu = kappa_nu_floor
            else:
                nu       = nu_ceil
                kappa_nu = kappa_nu_ceil
            
            print('\nnu = {}'.format(nu))
            print('kappa_nu = {}'.format(kappa_nu))
            
            l_nu_eq_intgr_nu  = l_nu_eq
            intgr_nu          = nu
            kappa_nu_intgr_nu = kappa_nu
            
            f_c_fit_intgr_nu = f_c_func_l_nu_eq_nu_kappa_nu_fit(r_nu_fit_intgr_nu, l_nu_eq_intgr_nu, intgr_nu, kappa_nu_intgr_nu)
            
            l_nu_eq  = l_nu_eq_init
            nu       = nu_init
            kappa_nu = kappa_nu_init
            
            return r_nu_fit, r_nu_fit_intgr_nu, f_c_fit, f_c_fit_intgr_nu, l_nu_eq, l_nu_eq_intgr_nu, nu, intgr_nu, kappa_nu, kappa_nu_intgr_nu

        k_B     = constants.value(u'Boltzmann constant') # J/K
        h       = constants.value(u'Planck constant') # J/Hz
        hbar    = h/(2*np.pi) # J*sec
        beta    = 1./(k_B*self.T) # 1/J
        omega_0 = 1./(beta*hbar) # J/(J*sec) = 1/sec

        beta = beta/(1e9*1e9) # 1/J = 1/(N*m) -> 1/(nN*m) -> 1/(nN*nm)

        cp = self.parameters.characterizer

        polymer_type             = cp.paper_authors2polymer_type_dict[self.paper_authors]
        chain_backbone_bond_type = cp.polymer_type_label2chain_backbone_bond_type_dict[polymer_type]
        self.data_file_prefix    = self.paper_authors+'-'+polymer_type+'-'+chain_backbone_bond_type+'-'+self.chain

        f_c_max     = cp.chain_backbone_bond_type2f_c_max_dict[chain_backbone_bond_type]
        zeta_b_char = cp.chain_backbone_bond_type2zeta_b_char_dict[chain_backbone_bond_type]
        l_b_eq      = cp.chain_backbone_bond_type2l_b_eq_dict[chain_backbone_bond_type]

        # Curve fit single chain AFM tensile test results
        r_nu, f_c = chain_data_file_load_func(self.data_file_prefix, cp.chain_data_directory) # nm, nN respectively

        r_nu_fit, r_nu_fit_intgr_nu, f_c_fit, f_c_fit_intgr_nu, l_nu_eq, l_nu_eq_intgr_nu, nu, intgr_nu, kappa_nu, kappa_nu_intgr_nu = l_nu_eq_nu_kappa_nu_fit_func(r_nu, f_c) # nm, nN, nm, unitless, unitless, respectively

        lmbda_c_eq              = r_nu/(nu*l_nu_eq) # nm/nm
        lmbda_c_eq_fit          = r_nu_fit/(nu*l_nu_eq) # nm/nm
        lmbda_c_eq_intgr_nu     = r_nu/(intgr_nu*l_nu_eq_intgr_nu) # nm/nm
        lmbda_c_eq_fit_intgr_nu = r_nu_fit_intgr_nu/(intgr_nu*l_nu_eq_intgr_nu) # nm/nm

        xi_c              = f_c*beta*l_nu_eq # nN*nm/(nN*nm)
        xi_c_fit          = f_c_fit*beta*l_nu_eq # nN*nm/(nN*nm)
        xi_c_intgr_nu     = f_c*beta*l_nu_eq_intgr_nu # nN*nm/(nN*nm)
        xi_c_fit_intgr_nu = f_c_fit_intgr_nu*beta*l_nu_eq_intgr_nu # nN*nm/(nN*nm)

        nu_b          = l_nu_eq/l_b_eq
        nu_b_intgr_nu = l_nu_eq_intgr_nu/l_b_eq

        zeta_nu_char          = nu_b*zeta_b_char
        zeta_nu_char_intgr_nu = nu_b_intgr_nu*zeta_b_char

        kappa_b          = kappa_nu/nu_b
        kappa_b_intgr_nu = kappa_nu_intgr_nu/nu_b_intgr_nu

        xi_c_max = f_c_max*beta*l_nu_eq_intgr_nu

        self.r_nu              = r_nu
        self.r_nu_fit          = r_nu_fit
        self.r_nu_fit_intgr_nu = r_nu_fit_intgr_nu

        self.lmbda_c_eq              = lmbda_c_eq
        self.lmbda_c_eq_fit          = lmbda_c_eq_fit
        self.lmbda_c_eq_intgr_nu     = lmbda_c_eq_intgr_nu
        self.lmbda_c_eq_fit_intgr_nu = lmbda_c_eq_fit_intgr_nu

        self.f_c              = f_c
        self.f_c_fit          = f_c_fit
        self.f_c_fit_intgr_nu = f_c_fit_intgr_nu

        self.xi_c              = xi_c
        self.xi_c_fit          = xi_c_fit
        self.xi_c_intgr_nu     = xi_c_intgr_nu
        self.xi_c_fit_intgr_nu = xi_c_fit_intgr_nu
        
        self.l_b_eq           = l_b_eq
        self.l_nu_eq          = l_nu_eq
        self.l_nu_eq_intgr_nu = l_nu_eq_intgr_nu

        self.nu_b          = nu_b
        self.nu_b_intgr_nu = nu_b_intgr_nu

        self.nu       = nu
        self.intgr_nu = intgr_nu
        
        self.zeta_b_char           = zeta_b_char
        self.zeta_nu_char          = zeta_nu_char
        self.zeta_nu_char_intgr_nu = zeta_nu_char_intgr_nu

        self.kappa_b          = kappa_b
        self.kappa_b_intgr_nu = kappa_b_intgr_nu

        self.kappa_nu          = kappa_nu
        self.kappa_nu_intgr_nu = kappa_nu_intgr_nu

        self.xi_c_max = xi_c_max


        # Initialize a rate-dependent generalizeduFJC which matches the chain used in the AFM tensile test where the bond and segment numbers in the chain are strictly integer values
        rate_dependent_single_chain = GeneralizeduFJC(rate_dependence = 'rate_dependent', nu = intgr_nu, zeta_nu_char = zeta_nu_char_intgr_nu, kappa_nu = kappa_nu_intgr_nu, omega_0 = omega_0)

        f_c_crit = rate_dependent_single_chain.xi_c_crit/(beta*l_nu_eq_intgr_nu) # (nN*nm)/nm = nN

        # Define the applied chain force values to calculate over
        f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN

        t_steps___f_c_dot_chunk                       = [0. for f_c_dot_indx in range(len(cp.f_c_dot_list))]
        xi_c___f_c_dot_chunk                          = [0. for f_c_dot_indx in range(len(cp.f_c_dot_list))]
        lmbda_nu___f_c_dot_chunk                      = [0. for f_c_dot_indx in range(len(cp.f_c_dot_list))]
        lmbda_c_eq___f_c_dot_chunk                    = [0. for f_c_dot_indx in range(len(cp.f_c_dot_list))]
        gamma_c___f_c_dot_chunk                       = [0. for f_c_dot_indx in range(len(cp.f_c_dot_list))]
        overline_epsilon_cnu_diss_hat___f_c_dot_chunk = [0. for f_c_dot_indx in range(len(cp.f_c_dot_list))]

        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            # Define the timestamp values to calculate over
            f_c_dot = cp.f_c_dot_list[f_c_dot_indx] # nN/sec
            t_steps = f_c_steps/f_c_dot # nN/(nN/sec) = sec

            # Make arrays to allocate results
            xi_c                    = []
            lmbda_nu                = []
            lmbda_c_eq              = []
            p_nu_sci_hat            = []
            p_nu_sci_hat_cum_intgrl = []
            gamma_c                 = []
            epsilon_cnu_diss_hat    = []

            # Calculate results through applied chain force values
            for f_c_indx in range(cp.f_c_num_steps):
                t_val            = t_steps[f_c_indx]
                xi_c_val         = f_c_steps[f_c_indx]*beta*l_nu_eq_intgr_nu # nN*nm/(nN*nm)
                lmbda_nu_val     = rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
                lmbda_c_eq_val   = rate_dependent_single_chain.lmbda_c_eq_func(lmbda_nu_val)
                p_nu_sci_hat_val = rate_dependent_single_chain.p_nu_sci_hat_func(lmbda_nu_val)

                if f_c_indx == 0: # initialization
                    p_nu_sci_hat_cum_intgrl_val = 0
                    gamma_c_val                 = 0
                    epsilon_cnu_diss_hat_val    = 0
                else:
                    p_nu_sci_hat_cum_intgrl_val = rate_dependent_single_chain.p_nu_sci_hat_cum_intgrl_func(p_nu_sci_hat_val, t_val, p_nu_sci_hat[f_c_indx-1], t_steps[f_c_indx-1], p_nu_sci_hat_cum_intgrl[f_c_indx-1])
                    gamma_c_val                 = rate_dependent_single_chain.gamma_c_func(p_nu_sci_hat_cum_intgrl_val)
                    epsilon_cnu_diss_hat_val    = rate_dependent_single_chain.epsilon_cnu_diss_hat_func(p_nu_sci_hat_val = p_nu_sci_hat_val, p_nu_sci_hat_cum_intgrl_val = p_nu_sci_hat_cum_intgrl_val, t_val = t_val, lmbda_nu_hat_val = lmbda_nu_val, t_prior = t_steps[f_c_indx-1], epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat[f_c_indx-1])

                xi_c.append(xi_c_val)
                lmbda_nu.append(lmbda_nu_val)
                lmbda_c_eq.append(lmbda_c_eq_val)
                p_nu_sci_hat.append(p_nu_sci_hat_val)
                p_nu_sci_hat_cum_intgrl.append(p_nu_sci_hat_cum_intgrl_val)
                gamma_c.append(gamma_c_val)
                epsilon_cnu_diss_hat.append(epsilon_cnu_diss_hat_val)
            
            overline_epsilon_cnu_diss_hat = [epsilon_cnu_diss_hat_val/rate_dependent_single_chain.zeta_nu_char for epsilon_cnu_diss_hat_val in epsilon_cnu_diss_hat]

            t_steps___f_c_dot_chunk[f_c_dot_indx]                       = t_steps
            xi_c___f_c_dot_chunk[f_c_dot_indx]                          = xi_c
            lmbda_nu___f_c_dot_chunk[f_c_dot_indx]                      = lmbda_nu
            lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx]                    = lmbda_c_eq
            gamma_c___f_c_dot_chunk[f_c_dot_indx]                       = gamma_c
            overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx] = overline_epsilon_cnu_diss_hat
        
        self.rate_dependent_t_steps___f_c_dot_chunk                       = t_steps___f_c_dot_chunk
        self.rate_dependent_xi_c___f_c_dot_chunk                          = xi_c___f_c_dot_chunk
        self.rate_dependent_lmbda_nu___f_c_dot_chunk                      = lmbda_nu___f_c_dot_chunk
        self.rate_dependent_lmbda_c_eq___f_c_dot_chunk                    = lmbda_c_eq___f_c_dot_chunk
        self.rate_dependent_gamma_c___f_c_dot_chunk                       = gamma_c___f_c_dot_chunk
        self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk = overline_epsilon_cnu_diss_hat___f_c_dot_chunk


        # Initialize a rate-independent generalizeduFJC which matches the chain used in the AFM tensile test where the bond and segment numbers in the chain are strictly integer values
        rate_independent_single_chain = GeneralizeduFJC(rate_dependence = 'rate_independent', nu = intgr_nu, zeta_nu_char = zeta_nu_char_intgr_nu, kappa_nu = kappa_nu_intgr_nu)

        f_c_crit = rate_independent_single_chain.xi_c_crit/(beta*l_nu_eq_intgr_nu) # (nN*nm)/nm = nN

        # Define the applied chain force values to calculate over
        f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN

        xi_c                 = []
        lmbda_nu             = []
        lmbda_c_eq           = []
        p_c_sci_hat          = []
        epsilon_cnu_sci_hat  = []
        epsilon_cnu_diss_hat = []

        # initialization
        lmbda_nu_max = 0

        # Calculate results through applied chain force values
        for f_c_indx in range(cp.f_c_num_steps):
            xi_c_val                = f_c_steps[f_c_indx]*beta*l_nu_eq_intgr_nu # nN*nm/(nN*nm)
            lmbda_nu_val            = rate_independent_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
            lmbda_nu_max            = max([lmbda_nu_max, lmbda_nu_val])
            lmbda_c_eq_val          = rate_independent_single_chain.lmbda_c_eq_func(lmbda_nu_val)
            p_c_sci_hat_val         = rate_independent_single_chain.p_c_sci_hat_func(lmbda_nu_val)
            epsilon_cnu_sci_hat_val = rate_independent_single_chain.epsilon_cnu_sci_hat_func(lmbda_nu_val)

            if f_c_indx == 0: # initialization
                epsilon_cnu_diss_hat_val = 0
            else:
                epsilon_cnu_diss_hat_val = rate_independent_single_chain.epsilon_cnu_diss_hat_func(lmbda_nu_hat_max = lmbda_nu_max, lmbda_nu_hat_val = lmbda_nu_val, lmbda_nu_hat_val_prior = lmbda_nu[f_c_indx-1], epsilon_cnu_diss_hat_val_prior = epsilon_cnu_diss_hat[f_c_indx-1])
            
            xi_c.append(xi_c_val)
            lmbda_nu.append(lmbda_nu_val)
            lmbda_c_eq.append(lmbda_c_eq_val)
            p_c_sci_hat.append(p_c_sci_hat_val)
            epsilon_cnu_sci_hat.append(epsilon_cnu_sci_hat_val)
            epsilon_cnu_diss_hat.append(epsilon_cnu_diss_hat_val)

        overline_epsilon_cnu_sci_hat  = [epsilon_cnu_sci_hat_val/rate_independent_single_chain.zeta_nu_char for epsilon_cnu_sci_hat_val in epsilon_cnu_sci_hat]
        overline_epsilon_cnu_diss_hat = [epsilon_cnu_diss_hat_val/rate_independent_single_chain.zeta_nu_char for epsilon_cnu_diss_hat_val in epsilon_cnu_diss_hat]

        self.rate_independent_xi_c                          = xi_c
        self.rate_independent_lmbda_nu                      = lmbda_nu
        self.rate_independent_lmbda_c_eq                    = lmbda_c_eq
        self.rate_independent_p_c_sci_hat                   = p_c_sci_hat
        self.rate_independent_overline_epsilon_cnu_sci_hat  = overline_epsilon_cnu_sci_hat
        self.rate_independent_overline_epsilon_cnu_diss_hat = overline_epsilon_cnu_diss_hat

    def finalization(self):
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        # save numerical results in text files
        print("\nl_b_eq = {}".format(self.l_b_eq))
        print("l_nu_eq = {}".format(self.l_nu_eq))
        print("l_nu_eq_intgr_nu = {}".format(self.l_nu_eq_intgr_nu))

        print("nu_b = {}".format(self.nu_b))
        print("nu_b_intgr_nu = {}".format(self.nu_b_intgr_nu))

        print("nu = {}".format(self.nu))
        print("intgr_nu = {}".format(self.intgr_nu))

        print("zeta_b_char = {}".format(self.zeta_b_char))
        print("zeta_nu_char = {}".format(self.zeta_nu_char))
        print("zeta_nu_char_intgr_nu = {}".format(self.zeta_nu_char_intgr_nu))

        print("kappa_b = {}".format(self.kappa_b))
        print("kappa_b_intgr_nu = {}".format(self.kappa_b_intgr_nu))

        print("kappa_nu = {}".format(self.kappa_nu))
        print("kappa_nu_intgr_nu = {}".format(self. kappa_nu_intgr_nu))

        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-l_b_eq'+'.txt', [self.l_b_eq])
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-l_nu_eq'+'.txt', [self.l_nu_eq])
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-l_nu_eq_intgr_nu'+'.txt', [self.l_nu_eq_intgr_nu])

        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-nu_b'+'.txt', [self.nu_b])
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-nu_b_intgr_nu'+'.txt', [self.nu_b_intgr_nu])
        
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-nu'+'.txt', [self.nu])
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-intgr_nu'+'.txt', [self.intgr_nu])
        
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-zeta_b_char'+'.txt', [self.zeta_b_char])
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-zeta_nu_char'+'.txt', [self.zeta_nu_char])
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-zeta_nu_char_intgr_nu'+'.txt', [self.zeta_nu_char_intgr_nu])
        
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-kappa_b'+'.txt', [self.kappa_b])
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-kappa_b_intgr_nu'+'.txt', [self.kappa_b_intgr_nu])

        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-kappa_nu'+'.txt', [self.kappa_nu])
        np.savetxt(self.savedir+self.data_file_prefix+'-generalized-uFJC-curve-fit-kappa_nu_intgr_nu'+'.txt', [self.kappa_nu_intgr_nu])

        # plot results
        latex_formatting_figure(ppp)

        # plot curve fit results
        fig = plt.figure()
        plt.scatter(self.r_nu, self.f_c, color='blue', marker='o', alpha=1, linewidth=2.5, label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        plt.plot(self.r_nu_fit, self.f_c_fit, linestyle='--', color='red', alpha=1, linewidth=2.5, label=r'$u\textrm{FJC model fit}$')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        plt.xlim(cp.paper_authors_chain2xlim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        plt.ylim(cp.paper_authors_chain2ylim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        save_current_figure(self.savedir, r'$r_{\nu}~(nm)$', 30, r'$f_c~(nN)$', 30, self.data_file_prefix+"-f_c-vs-r_nu-generalized-uFJC-curve-fit")

        fig = plt.figure()
        plt.scatter(self.r_nu, self.f_c, color='blue', marker='o', alpha=1, linewidth=2.5, label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        plt.plot(self.r_nu_fit_intgr_nu, self.f_c_fit_intgr_nu, linestyle='--', color='red', alpha=1, linewidth=2.5, label=r'$u\textrm{FJC model fit}$')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        plt.xlim(cp.paper_authors_chain2xlim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        plt.ylim(cp.paper_authors_chain2ylim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        save_current_figure(self.savedir, r'$r_{\nu}~(nm)$', 30, r'$f_c~(nN)$', 30, self.data_file_prefix+"-intgr_nu-f_c-vs-r_nu-generalized-uFJC-curve-fit")
        
        fig = plt.figure()
        plt.scatter(self.lmbda_c_eq, self.xi_c, color='blue', marker='o', alpha=1, linewidth=2.5, label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        plt.plot(self.lmbda_c_eq_fit, self.xi_c_fit, linestyle='--', color='red', alpha=1, linewidth=2.5, label=r'$u\textrm{FJC model fit}$')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        plt.xlim([0, 1.15])
        save_current_figure(self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_{c}$', 30, self.data_file_prefix+"-xi_c-vs-lmbda_c_eq-generalized-uFJC-curve-fit")
        
        fig = plt.figure()
        plt.scatter(self.lmbda_c_eq_intgr_nu, self.xi_c_intgr_nu, color='blue', marker='o', alpha=1, linewidth=2.5, label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        plt.plot(self.lmbda_c_eq_fit_intgr_nu, self.xi_c_fit_intgr_nu, linestyle='--', color='red', alpha=1, linewidth=2.5, label=r'$u\textrm{FJC model fit}$')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        plt.xlim([0, 1.15])
        save_current_figure(self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_{c}$', 30, self.data_file_prefix+"-intgr_nu-xi_c-vs-lmbda_c_eq-gen-uFJC-curve-fit")

        # plot rate-dependent chain results
        t_max = 0
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [2, 1, 1, 1]}, sharex=True)
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            t_max = max([t_max, self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx][-1]])
            ax1.semilogx(self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_xi_c___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5, label=cp.f_c_dot_label_list[f_c_dot_indx])
            ax2.semilogx(self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_lmbda_nu___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5)
            ax3.semilogx(self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_gamma_c___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5)
            ax4.semilogx(self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5)
        ax1.hlines(y=self.xi_c_max, xmin=0, xmax=t_max, linestyle='--', color='black', alpha=1, linewidth=1)
        ax1.legend(loc='best')
        ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        ax3.set_ylabel(r'$\gamma_c$', fontsize=20)
        ax3.grid(True, alpha=0.25)
        ax4.set_yticks([0.0, 0.25, 0.5])
        ax4.set_ylabel(r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=20)
        ax4.grid(True, alpha=0.25)
        plt.xlabel(r'$t~(sec)$', fontsize=30)
        save_current_figure_no_labels(self.savedir, self.data_file_prefix+"-rate-dependent-xi_c-lmbda_nu-gamma_c-overline_epsilon_cnu_diss_hat-vs-time")
        
        # plot rate-independent and rate-dependent chain results together
        # plot rate-dependent chain results
        lmbda_c_eq_max = 0
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1.25, 1, 1.5, 1.5]}, sharex=True)
        ax1.scatter(self.lmbda_c_eq_intgr_nu, self.xi_c_intgr_nu, color='red', marker='o', alpha=1, linewidth=1, label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            lmbda_c_eq_max = max([lmbda_c_eq_max, self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx][-1]])
            ax1.plot(self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_xi_c___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5)
            ax2.plot(self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_lmbda_nu___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5)
            ax3.plot(self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_gamma_c___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5, label=cp.f_c_dot_label_list[f_c_dot_indx])
            ax4.plot(self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5)
        ax1.hlines(y=self.xi_c_max, xmin=-0.05, xmax=lmbda_c_eq_max + 0.05, linestyle='--', color='black', alpha=1, linewidth=1)
        # plot rate-independent chain results
        ax1.plot(self.rate_independent_lmbda_c_eq, self.rate_independent_xi_c, linestyle='-', color='blue', alpha=1, linewidth=2.5, label=r'$\textrm{rate-independent chain}$')
        ax2.plot(self.rate_independent_lmbda_c_eq, self.rate_independent_lmbda_nu, linestyle='-', color='blue', alpha=1, linewidth=2.5)
        ax3.plot(self.rate_independent_lmbda_c_eq, self.rate_independent_p_c_sci_hat, linestyle='-', color='blue', alpha=1, linewidth=2.5)
        ax4.plot(self.rate_independent_lmbda_c_eq, self.rate_independent_overline_epsilon_cnu_sci_hat, linestyle='-', color='black', alpha=1, linewidth=2.5)
        ax4.plot(self.rate_independent_lmbda_c_eq, self.rate_independent_overline_epsilon_cnu_diss_hat, linestyle='-', color='blue', alpha=1, linewidth=2.5)

        ax1.legend(loc='best')
        ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        ax3.legend(loc='best')
        ax3.set_ylabel(r'$\gamma_c,~\hat{p}_c^{sci}$', fontsize=20)
        ax3.grid(True, alpha=0.25)
        ax4.set_yticks([0.0, 0.25, 0.5])
        ax4.set_ylabel(r'$\overline{\hat{\varepsilon}_{c\nu}^{sci}},~\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=20)
        ax4.grid(True, alpha=0.25)
        plt.xlim([-0.05, lmbda_c_eq_max + 0.05])
        plt.xlabel(r'$\lambda_c^{eq}$', fontsize=30)
        save_current_figure_no_labels(self.savedir, self.data_file_prefix+"-rate-independent-and-rate-dependent-chains-vs-lmbda_c_eq")

        # plot rate-independent and rate-dependent chain results together while omitting chain scission energy
        # plot rate-dependent chain results
        lmbda_c_eq_max = 0
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1.25, 1, 1.5, 1.5]}, sharex=True)
        ax1.scatter(self.lmbda_c_eq_intgr_nu, self.xi_c_intgr_nu, color='red', marker='o', alpha=1, linewidth=1, label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            lmbda_c_eq_max = max([lmbda_c_eq_max, self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx][-1]])
            ax1.plot(self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_xi_c___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5)
            ax2.plot(self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_lmbda_nu___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5)
            ax3.plot(self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_gamma_c___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5, label=cp.f_c_dot_label_list[f_c_dot_indx])
            ax4.plot(self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx], self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx], linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx], alpha=1, linewidth=2.5)
        ax1.hlines(y=self.xi_c_max, xmin=-0.05, xmax=lmbda_c_eq_max + 0.05, linestyle='--', color='black', alpha=1, linewidth=1)
        # plot rate-independent chain results
        ax1.plot(self.rate_independent_lmbda_c_eq, self.rate_independent_xi_c, linestyle='-', color='blue', alpha=1, linewidth=2.5, label=r'$\textrm{rate-independent chain}$')
        ax2.plot(self.rate_independent_lmbda_c_eq, self.rate_independent_lmbda_nu, linestyle='-', color='blue', alpha=1, linewidth=2.5)
        ax3.plot(self.rate_independent_lmbda_c_eq, self.rate_independent_p_c_sci_hat, linestyle='-', color='blue', alpha=1, linewidth=2.5)
        ax4.plot(self.rate_independent_lmbda_c_eq, self.rate_independent_overline_epsilon_cnu_diss_hat, linestyle='-', color='blue', alpha=1, linewidth=2.5)

        ax1.legend(loc='best')
        ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        ax3.legend(loc='best')
        ax3.set_ylabel(r'$\gamma_c,~\hat{p}_c^{sci}$', fontsize=20)
        ax3.grid(True, alpha=0.25)
        ax4.set_yticks([0.0, 0.25, 0.5])
        ax4.set_ylabel(r'$\overline{\hat{\varepsilon}_{c\nu}^{sci}},~\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=20)
        ax4.grid(True, alpha=0.25)
        plt.xlim([-0.05, lmbda_c_eq_max + 0.05])
        plt.xlabel(r'$\lambda_c^{eq}$', fontsize=30)
        save_current_figure_no_labels(self.savedir, self.data_file_prefix+"-rate-independent-and-rate-dependent-chains-vs-lmbda_c_eq-no-epsilon_cnu_diss")


if __name__ == '__main__':

    characterizer = MaterialPointFormulationCharacterizer()
    characterizer.characterization()
    characterizer.finalization()