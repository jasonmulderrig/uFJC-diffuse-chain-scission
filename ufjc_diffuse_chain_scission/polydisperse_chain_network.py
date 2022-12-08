################################################################################################################################
# General setup
################################################################################################################################

# Import necessary libraries
from __future__ import division
from dolfin import *
from .single_chain import GeneralizeduFJC, EqualStrainGeneralizeduFJC, EqualForceGeneralizeduFJC
from .microsphere_quadrature import MicrosphereQuadratureScheme
from .microdisk_quadrature import MicrodiskQuadratureScheme
import sys
import numpy as np
from types import SimpleNamespace
from copy import deepcopy

# Numerical tolerance parameters
min_exponent = np.log(sys.float_info.min)/np.log(10)
max_exponent = np.log(sys.float_info.max)/np.log(10)

class TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass
    
    def homogeneous_strong_form_initialization(self):
        results = SimpleNamespace()
        chunks  = SimpleNamespace()

        # initialize lists to zeros - necessary for irreversibility
        chunks = self.strong_form_initialize_sigma_chunks(chunks)
        
        # lmbda_c
        chunks.lmbda_c_chunks = []
        results.lmbda_c_val   = 0.
        # lmbda_c_eq
        chunks.lmbda_c_eq_chunks     = []
        chunks.lmbda_c_eq_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.lmbda_c_eq_val       = [0. for nu_indx in range(self.nu_num)]
        # lmbda_nu
        chunks.lmbda_nu_chunks     = []
        chunks.lmbda_nu_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.lmbda_nu_val       = [0. for nu_indx in range(self.nu_num)]
        # lmbda_nu_max
        chunks.lmbda_nu_max_chunks     = []
        chunks.lmbda_nu_max_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.lmbda_nu_max_val       = [0. for nu_indx in range(self.nu_num)]
        # upsilon_c
        chunks.upsilon_c_chunks     = []
        chunks.upsilon_c_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.upsilon_c_val       = [0. for nu_indx in range(self.nu_num)]
        # Upsilon_c
        chunks.Upsilon_c_chunks     = []
        chunks.Upsilon_c_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.Upsilon_c_val       = [0. for nu_indx in range(self.nu_num)]
        # d_c
        chunks.d_c_chunks     = []
        chunks.d_c_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.d_c_val       = [0. for nu_indx in range(self.nu_num)]
        # D_c
        chunks.D_c_chunks = []
        results.D_c_val   = 0.
        # epsilon_cnu_diss_hat
        chunks.epsilon_cnu_diss_hat_chunks     = []
        chunks.epsilon_cnu_diss_hat_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.epsilon_cnu_diss_hat_val       = [0. for nu_indx in range(self.nu_num)]
        # Epsilon_cnu_diss_hat
        chunks.Epsilon_cnu_diss_hat_chunks = []
        results.Epsilon_cnu_diss_hat_val   = 0.
        # epsilon_c_diss_hat
        chunks.epsilon_c_diss_hat_chunks     = []
        chunks.epsilon_c_diss_hat_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.epsilon_c_diss_hat_val       = [0. for nu_indx in range(self.nu_num)]
        # Epsilon_c_diss_hat
        chunks.Epsilon_c_diss_hat_chunks = []
        results.Epsilon_c_diss_hat_val   = 0.
        # overline_epsilon_cnu_diss_hat
        chunks.overline_epsilon_cnu_diss_hat_chunks     = []
        chunks.overline_epsilon_cnu_diss_hat_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.overline_epsilon_cnu_diss_hat_val       = [0. for nu_indx in range(self.nu_num)]
        # overline_Epsilon_cnu_diss_hat
        chunks.overline_Epsilon_cnu_diss_hat_chunks = []
        results.overline_Epsilon_cnu_diss_hat_val   = 0.
        # overline_epsilon_c_diss_hat
        chunks.overline_epsilon_c_diss_hat_chunks     = []
        chunks.overline_epsilon_c_diss_hat_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.overline_epsilon_c_diss_hat_val       = [0. for nu_indx in range(self.nu_num)]
        # overline_Epsilon_c_diss_hat
        chunks.overline_Epsilon_c_diss_hat_chunks = []
        results.overline_Epsilon_c_diss_hat_val   = 0.
        # sigma_hyp_val
        results.sigma_hyp_val = 0.

        return results, chunks

    def homogeneous_strong_form_solve_step(self, deformation, results):
        F_val, C_val, b_val = self.lr_cg_deformation_gradient_func(deformation)

        Upsilon_c_val                     = 0
        D_c_val                           = 0
        Epsilon_cnu_diss_hat_val          = 0
        Epsilon_c_diss_hat_val            = 0
        overline_Epsilon_cnu_diss_hat_val = 0
        overline_Epsilon_c_diss_hat_val   = 0
        sigma_hyp_val                     = 0
        
        lmbda_c_val = np.sqrt(np.trace(C_val)/3.)
        
        for nu_indx in range(self.nu_num):
            A_nu___nu_val       = self.A_nu_list[nu_indx]
            lmbda_c_eq___nu_val = lmbda_c_val*A_nu___nu_val
            lmbda_nu___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_func(lmbda_c_eq___nu_val)
            # impose irreversibility
            lmbda_nu_max___nu_val                  = max([results.lmbda_nu_max_val[nu_indx], lmbda_nu___nu_val])
            upsilon_c___nu_val                     = self.single_chain_list[nu_indx].upsilon_c_func(self.k_cond_val, lmbda_nu_max___nu_val)
            d_c___nu_val                           = self.single_chain_list[nu_indx].d_c_func(self.k_cond_val, lmbda_nu_max___nu_val)
            epsilon_cnu_diss_hat___nu_val          = self.single_chain_list[nu_indx].epsilon_cnu_diss_hat_func(lmbda_nu_hat_max = lmbda_nu_max___nu_val, lmbda_nu_hat_val = lmbda_nu___nu_val, lmbda_nu_hat_val_prior = results.lmbda_nu_val[nu_indx], epsilon_cnu_diss_hat_val_prior = results.epsilon_cnu_diss_hat_val[nu_indx])
            epsilon_c_diss_hat___nu_val            = epsilon_cnu_diss_hat___nu_val*self.nu_list[nu_indx]
            overline_epsilon_cnu_diss_hat___nu_val = epsilon_cnu_diss_hat___nu_val/self.zeta_nu_char
            overline_epsilon_c_diss_hat___nu_val   = epsilon_c_diss_hat___nu_val/self.zeta_nu_char

            results.lmbda_c_eq_val[nu_indx]                    = lmbda_c_eq___nu_val
            results.lmbda_nu_val[nu_indx]                      = lmbda_nu___nu_val
            results.lmbda_nu_max_val[nu_indx]                  = lmbda_nu_max___nu_val
            results.upsilon_c_val[nu_indx]                     = upsilon_c___nu_val
            results.d_c_val[nu_indx]                           = d_c___nu_val
            results.epsilon_cnu_diss_hat_val[nu_indx]          = epsilon_cnu_diss_hat___nu_val
            results.epsilon_c_diss_hat_val[nu_indx]            = epsilon_c_diss_hat___nu_val
            results.overline_epsilon_cnu_diss_hat_val[nu_indx] = overline_epsilon_cnu_diss_hat___nu_val
            results.overline_epsilon_c_diss_hat_val[nu_indx]   = overline_epsilon_c_diss_hat___nu_val

            Upsilon_c_val                     += self.P_nu_list[nu_indx]*upsilon_c___nu_val
            D_c_val                           += self.P_nu_list[nu_indx]*d_c___nu_val
            Epsilon_cnu_diss_hat_val          += self.P_nu_list[nu_indx]*epsilon_cnu_diss_hat___nu_val
            Epsilon_c_diss_hat_val            += self.P_nu_list[nu_indx]*epsilon_c_diss_hat___nu_val
            overline_Epsilon_cnu_diss_hat_val += self.P_nu_list[nu_indx]*overline_epsilon_cnu_diss_hat___nu_val
            overline_Epsilon_c_diss_hat_val   += self.P_nu_list[nu_indx]*overline_epsilon_c_diss_hat___nu_val
            sigma_hyp_val                     += upsilon_c___nu_val*self.P_nu_list[nu_indx]*self.nu_list[nu_indx]*self.A_nu_list[nu_indx]*self.single_chain_list[nu_indx].xi_c_func(lmbda_nu___nu_val, lmbda_c_eq___nu_val)/(3.*lmbda_c_val)
        
        Upsilon_c_val                     = Upsilon_c_val/self.P_nu_sum
        D_c_val                           = D_c_val/self.P_nu_sum
        Epsilon_cnu_diss_hat_val          = Epsilon_cnu_diss_hat_val/self.P_nu_sum
        Epsilon_c_diss_hat_val            = Epsilon_c_diss_hat_val/self.P_nu_sum
        overline_Epsilon_cnu_diss_hat_val = overline_Epsilon_cnu_diss_hat_val/self.P_nu_sum
        overline_Epsilon_c_diss_hat_val   = overline_Epsilon_c_diss_hat_val/self.P_nu_sum
        
        results.lmbda_c_val                       = lmbda_c_val
        results.Upsilon_c_val                     = Upsilon_c_val
        results.D_c_val                           = D_c_val
        results.Epsilon_cnu_diss_hat_val          = Epsilon_cnu_diss_hat_val
        results.Epsilon_c_diss_hat_val            = Epsilon_c_diss_hat_val
        results.overline_Epsilon_cnu_diss_hat_val = overline_Epsilon_cnu_diss_hat_val
        results.overline_Epsilon_c_diss_hat_val   = overline_Epsilon_c_diss_hat_val
        results.sigma_hyp_val                     = sigma_hyp_val

        return results
    
    def homogeneous_strong_form_chunk_post_processing(self, deformation, results, chunks):
        for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
            # first dimension is nu_chunk_val: list[nu_chunk_val]
            chunks.lmbda_c_eq_chunks_val[nu_chunk_indx]                    = results.lmbda_c_eq_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.lmbda_nu_chunks_val[nu_chunk_indx]                      = results.lmbda_nu_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.lmbda_nu_max_chunks_val[nu_chunk_indx]                  = results.lmbda_nu_max_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.upsilon_c_chunks_val[nu_chunk_indx]                     = results.upsilon_c_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.d_c_chunks_val[nu_chunk_indx]                           = results.d_c_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.epsilon_cnu_diss_hat_chunks_val[nu_chunk_indx]          = results.epsilon_cnu_diss_hat_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.epsilon_c_diss_hat_chunks_val[nu_chunk_indx]            = results.epsilon_c_diss_hat_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.overline_epsilon_cnu_diss_hat_chunks_val[nu_chunk_indx] = results.overline_epsilon_cnu_diss_hat_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.overline_epsilon_c_diss_hat_chunks_val[nu_chunk_indx]   = results.overline_epsilon_c_diss_hat_val[self.nu_chunks_indx_list[nu_chunk_indx]]

        sigma_val = self.strong_form_calculate_sigma_func(results.sigma_hyp_val, deformation)
        chunks    = self.strong_form_store_calculated_sigma_chunks(sigma_val, chunks)

        # first dimension is t_chunk_val: list[t_chunk_val]
        chunks.lmbda_c_chunks.append(results.lmbda_c_val)
        chunks.Upsilon_c_chunks.append(results.Upsilon_c_val)
        chunks.D_c_chunks.append(results.D_c_val)
        chunks.Epsilon_cnu_diss_hat_chunks.append(results.Epsilon_cnu_diss_hat_val)
        chunks.Epsilon_c_diss_hat_chunks.append(results.Epsilon_c_diss_hat_val)
        chunks.overline_Epsilon_cnu_diss_hat_chunks.append(results.overline_Epsilon_cnu_diss_hat_val)
        chunks.overline_Epsilon_c_diss_hat_chunks.append(results.overline_Epsilon_c_diss_hat_val)
        
        # first dimension is t_chunk_val, second dimension is nu_chunk_val: list[t_chunk_val][nu_chunk_val]
        chunks.lmbda_c_eq_chunks.append(deepcopy(chunks.lmbda_c_eq_chunks_val))
        chunks.lmbda_nu_chunks.append(deepcopy(chunks.lmbda_nu_chunks_val))
        chunks.lmbda_nu_max_chunks.append(deepcopy(chunks.lmbda_nu_max_chunks_val))
        chunks.upsilon_c_chunks.append(deepcopy(chunks.upsilon_c_chunks_val))
        chunks.d_c_chunks.append(deepcopy(chunks.d_c_chunks_val))
        chunks.epsilon_cnu_diss_hat_chunks.append(deepcopy(chunks.epsilon_cnu_diss_hat_chunks_val))
        chunks.epsilon_c_diss_hat_chunks.append(deepcopy(chunks.epsilon_c_diss_hat_chunks_val))
        chunks.overline_epsilon_cnu_diss_hat_chunks.append(deepcopy(chunks.overline_epsilon_cnu_diss_hat_chunks_val))
        chunks.overline_epsilon_c_diss_hat_chunks.append(deepcopy(chunks.overline_epsilon_c_diss_hat_chunks_val))

        return chunks

    def fenics_variational_formulation(self, parameters, fem):
        femp = parameters.fem

        # Create function space
        fem.V_u =  VectorFunctionSpace(fem.mesh, "CG", femp.u_degree)

        # Define solution, trial, and test functions, respectively
        fem.u   = Function(fem.V_u)
        fem.du  = TrialFunction(fem.V_u)
        fem.v_u = TestFunction(fem.V_u)

        # Define objects needed for calculations
        fem.I        = Identity(len(fem.u))
        fem.V_scalar = FunctionSpace(fem.mesh, "DG", 0)
        fem.V_tensor = TensorFunctionSpace(fem.mesh, "DG", 0)

        # Define body force and traction force
        fem.b = Constant((0.0, 0.0)) # Body force per unit volume
        fem.t = Constant((0.0, 0.0)) # Traction force on the boundary

        # Initialization
        # fem.lmbda_c_max_prior = project(Constant(1.0), fem.V_scalar) # lambda_c = 1 at the initial reference configuration
        fem.lmbda_c_max = project(Constant(1.0), fem.V_scalar) # lambda_c = 1 at the initial reference configuration

        # Kinematics
        fem.F           = fem.I + grad(fem.u) # deformation gradient tensor
        fem.F_inv       = inv(fem.F) # inverse deformation gradient tensor
        fem.J           = det(fem.F) # volume ratio
        fem.C           = fem.F.T*fem.F # right Cauchy-Green tensor
        fem.I_C         = tr(fem.C)+1 # 2D plane strain form of the trace of right Cauchy-Green tensor, where F_33 = 1 always -- this is the case of plane strain
        fem.lmbda_c     = sqrt(fem.I_C/3.0)
        # fem.lmbda_c_max = project(conditional(gt(fem.lmbda_c, fem.lmbda_c_max_prior), fem.lmbda_c, fem.lmbda_c_max_prior), fem.V_scalar)

        # Calculate the work function using the weak form; specify the quadrature degree for efficiency pk2_stress_ufl_func
        fem.WF = (inner(self.pk2_stress_ufl_func(fem), grad(fem.v_u)))*dx(metadata=femp.metadata) - dot(fem.b, fem.v_u)*dx(metadata=femp.metadata) - dot(fem.t, fem.v_u)*ds

        # Calculate the Gateaux derivative
        fem.Jac = derivative(fem.WF, fem.u, fem.du)

        return fem
    
    def solve_u(self, fem):
        """
        Solve the displacement problem
        """
        problem_u = NonlinearVariationalProblem(fem.WF, fem.u, fem.bc_u, J=fem.Jac)
        solver_u = NonlinearVariationalSolver(problem_u)

        solver_u.parameters.update(fem.dict_solver_u_parameters)
        info(solver_u.parameters, True)
        (iter, converged) = solver_u.solve()

        return fem
    
    # def fenics_weak_form_solve_step(self, fem):
    #     """
    #     Solve the weak form, which will provide the solution to the displacements and diffuse chain damage, and account for network irreversibility
    #     """
    #     fem = self.solve_u(fem)

    #     # Account for network irreversibility
    #     # lmbda_c_max_check = project(conditional(gt(fem.lmbda_c, fem.lmbda_c_max), fem.lmbda_c, fem.lmbda_c_max), fem.V_scalar)
    #     # fem.lmbda_c_max.assign(lmbda_c_max_check)
    #     fem.lmbda_c_max = project(conditional(gt(fem.lmbda_c, fem.lmbda_c_max), fem.lmbda_c, fem.lmbda_c_max), fem.V_scalar)

    #     return fem

    # def fenics_weak_form_solve_step(self, fem):
    #     """
    #     Solve the weak form, which will provide the solution to the displacements and diffuse chain damage, and account for network irreversibility
    #     """
    #     fem = self.solve_u(fem)

    #     # Account for network irreversibility
    #     lmbda_c_max_prior_check = project(conditional(gt(fem.lmbda_c, fem.lmbda_c_max_prior), fem.lmbda_c, fem.lmbda_c_max_prior), fem.V_scalar)
    #     fem.lmbda_c_max_prior.assign(lmbda_c_max_prior_check)

    #     return fem

    def fenics_weak_form_solve_step(self, fem):
        """
        Solve the weak form, which will provide the solution to the displacements and diffuse chain damage, and account for network irreversibility
        Use a staggered solution scheme to solve for the displacements and diffuse chain damage
        """
        # Implement the staggered solution scheme to solve for the displacements and diffuse chain damage
        iter_sss = 1
        error_d_c = 1.

        while iter_sss < self.iter_max_d_c_val and error_d_c > self.tol_d_c_val:
            error_d_c = 0.
            # solve for the displacements while holding diffuse chain damage fixed at the prior calculated value
            # lmbda_c_max is the proxy for fixed prior calculated diffuse chain damage
            fem = self.solve_u(fem)

            # Account for network irreversibility
            # lmbda_c_max_solve proxy for calculated diffuse chain damage
            lmbda_c_max_prior = project(fem.lmbda_c_max, fem.V_scalar)
            lmbda_c_max_solve = project(conditional(gt(fem.lmbda_c, fem.lmbda_c_max), fem.lmbda_c, fem.lmbda_c_max), fem.V_scalar)

            # Calculate error in diffuse chain damage for each chain segment number
            for nu_indx in range(self.nu_num):
                # Error in diffuse chain damage from the prior calculated value
                fem.lmbda_c_max = lmbda_c_max_prior # fem.lmbda_c_max.assign(lmbda_c_max_prior)
                d_c_val_prior = project(self.d_c_ufl_func(nu_indx, fem), fem.V_scalar)
                
                # Error in diffuse chain damage from the currently solved calculated value
                # Accounts for network irreversibility
                fem.lmbda_c_max = lmbda_c_max_solve # fem.lmbda_c_max.assign(lmbda_c_max_solve)
                d_c_val_solve = project(self.d_c_ufl_func(nu_indx, fem), fem.V_scalar)
                
                # Overall error in diffuse chain damage
                error_d_c_val      = d_c_val_solve.vector() - d_c_val_prior.vector()
                error_linf_d_c_val = error_d_c_val.norm('linf')
                error_d_c          = np.maximum(error_d_c, error_linf_d_c_val)
            
            # Monitor the results
            print("Diffuse chain damage staggered solution scheme: Iteration # {0:3d}; error = {1:>14.8f}".format(iter_sss, error_d_c))

            # Update iteration
            iter_sss += 1
        
        return fem

    # def fenics_weak_form_solve_step(self, fem):
    #     """
    #     Solve the weak form, which will provide the solution to the displacements and diffuse chain damage, and account for network irreversibility
    #     Use a staggered solution scheme to solve for the displacements and diffuse chain damage
    #     """
    #     # Implement the staggered solution scheme to solve for the displacements and diffuse chain damage
    #     iter_sss = 1
    #     error_d_c = 1.

    #     while iter_sss < self.iter_max_d_c_val and error_d_c > self.tol_d_c_val:
    #         error_d_c = 0.
    #         # solve for the displacements while holding diffuse chain damage fixed at the prior calculated value
    #         # lmbda_c_max is the proxy for fixed prior calculated diffuse chain damage
    #         fem = self.solve_u(fem)

    #         # Account for network irreversibility
    #         # lmbda_c_max_solve proxy for calculated diffuse chain damage
    #         lmbda_c_max_prior = project(fem.lmbda_c_max_prior, fem.V_scalar)
    #         lmbda_c_max_solve = project(conditional(gt(fem.lmbda_c, fem.lmbda_c_max_prior), fem.lmbda_c, fem.lmbda_c_max_prior), fem.V_scalar)

    #         # Calculate error in diffuse chain damage for each chain segment number
    #         for nu_indx in range(self.nu_num):
    #             # Error in diffuse chain damage from the prior calculated value
    #             fem.lmbda_c_max.assign(lmbda_c_max_prior)
    #             d_c_val_prior = project(self.d_c_ufl_func(nu_indx, fem), fem.V_scalar)
                
    #             # Error in diffuse chain damage from the currently solved calculated value
    #             # Accounts for network irreversibility
    #             fem.lmbda_c_max.assign(lmbda_c_max_solve)
    #             d_c_val_solve = project(self.d_c_ufl_func(nu_indx, fem), fem.V_scalar)
                
    #             # Overall error in diffuse chain damage
    #             error_d_c_val      = d_c_val_solve.vector() - d_c_val_prior.vector()
    #             error_linf_d_c_val = error_d_c_val.norm('linf')
    #             error_d_c          = np.maximum(error_d_c, error_linf_d_c_val)
            
    #         # Monitor the results
    #         print("Diffuse chain damage staggered solution scheme: Iteration # {0:3d}; error = {1:>14.8f}".format(iter_sss, error_d_c))

    #         # Update iteration
    #         fem.lmbda_c_max_prior.assign(fem.lmbda_c_max)
    #         iter_sss += 1
        
    #     return fem

    
    def fenics_weak_form_initialization(self, fem, parameters):
        gp      = parameters.geometry
        chunks  = SimpleNamespace()

        # initialize lists to zeros - necessary for irreversibility
        chunks = self.weak_form_initialize_deformation_sigma_chunks(gp.meshpoints, chunks)
        
        # lmbda_c
        chunks.lmbda_c_chunks     = []
        chunks.lmbda_c_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_c_eq
        chunks.lmbda_c_eq_chunks     = []
        chunks.lmbda_c_eq_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_nu
        chunks.lmbda_nu_chunks     = []
        chunks.lmbda_nu_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_nu_max
        chunks.lmbda_nu_max_chunks     = []
        chunks.lmbda_nu_max_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # upsilon_c
        chunks.upsilon_c_chunks     = []
        chunks.upsilon_c_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # Upsilon
        chunks.Upsilon_c_chunks     = []
        chunks.Upsilon_c_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # d_c
        chunks.d_c_chunks     = []
        chunks.d_c_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # D
        chunks.D_c_chunks     = []
        chunks.D_c_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # epsilon_cnu_diss_hat
        chunks.epsilon_cnu_diss_hat_chunks     = []
        chunks.epsilon_cnu_diss_hat_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # Epsilon_cnu_diss_hat
        chunks.Epsilon_cnu_diss_hat_chunks     = []
        chunks.Epsilon_cnu_diss_hat_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # epsilon_c_diss_hat
        chunks.epsilon_c_diss_hat_chunks     = []
        chunks.epsilon_c_diss_hat_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # Epsilon_c_diss_hat
        chunks.Epsilon_c_diss_hat_chunks     = []
        chunks.Epsilon_c_diss_hat_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # overline_epsilon_cnu_diss_hat
        chunks.overline_epsilon_cnu_diss_hat_chunks     = []
        chunks.overline_epsilon_cnu_diss_hat_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # overline_Epsilon_cnu_diss_hat
        chunks.overline_Epsilon_cnu_diss_hat_chunks     = []
        chunks.overline_Epsilon_cnu_diss_hat_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # overline_epsilon_c_diss_hat
        chunks.overline_epsilon_c_diss_hat_chunks     = []
        chunks.overline_epsilon_c_diss_hat_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # overline_Epsilon_c_diss_hat
        chunks.overline_Epsilon_c_diss_hat_chunks     = []
        chunks.overline_Epsilon_c_diss_hat_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]

        return fem, chunks

    def fenics_weak_form_chunk_post_processing(self, deformation, chunks, fem, file_results, parameters):
        """
        Post-processing at the end of each time iteration chunk
        """
        gp   = parameters.geometry
        femp = parameters.fem
        ppp  = parameters.post_processing

        file_results.parameters["rewrite_function_mesh"] = ppp.rewrite_function_mesh
        file_results.parameters["flush_output"]          = ppp.flush_output
        file_results.parameters["functions_share_mesh"]  = ppp.functions_share_mesh

        if ppp.save_u:
            fem.u.rename("Displacement", "u")
            file_results.write(fem.u, deformation.t_val)
        
        # sigma
        if ppp.save_sigma_mesh:
            sigma_val = project(self.pk2_stress_ufl_func(fem)/fem.J*fem.F.T, fem.V_tensor)
            sigma_val.rename("Normalized Cauchy stress", "sigma_val")
            file_results.write(sigma_val, deformation.t_val)
        
        if ppp.save_sigma_chunks:
            sigma_val = project(self.pk2_stress_ufl_func(fem)/fem.J*fem.F.T, fem.V_tensor)
            chunks    = self.weak_form_store_calculated_sigma_chunks(sigma_val, femp.tensor2vector_indx_dict, gp.meshpoints, chunks)
        
        # F
        if ppp.save_F_mesh:
            F_val = project(fem.F, fem.V_tensor)
            F_val.rename("Deformation gradient", "F_val")
            file_results.write(F_val, deformation.t_val)
        
        if ppp.save_F_chunks:
            F_val  = project(fem.F, fem.V_tensor)
            chunks = self.weak_form_store_calculated_deformation_chunks(F_val, femp.tensor2vector_indx_dict, gp.meshpoints, chunks)
        
        # lmbda_c
        if ppp.save_lmbda_c_mesh:
            lmbda_c_val = project(fem.lmbda_c, fem.V_scalar)
            lmbda_c_val.rename("Chain stretch", "lmbda_c_val")
            file_results.write(lmbda_c_val, deformation.t_val)
        
        if ppp.save_lmbda_c_chunks:
            lmbda_c_val = project(fem.lmbda_c, fem.V_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.lmbda_c_chunks_val[meshpoint_indx] = lmbda_c_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_c_chunks.append(deepcopy(chunks.lmbda_c_chunks_val))
        
        # lmbda_c_eq
        if ppp.save_lmbda_c_eq_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx             = self.nu_chunks_indx_list[nu_chunk_indx]
                lmbda_c_eq___nu_val = project(self.lmbda_c_eq_ufl_func(nu_indx, fem), fem.V_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Equilibrium chain stretch nu = "+nu_str
                parameter_str = "lmbda_c_eq___nu_"+nu_str+"_val"

                lmbda_c_eq___nu_val.rename(name_str, parameter_str)
                file_results.write(lmbda_c_eq___nu_val, deformation.t_val)
        
        if ppp.save_lmbda_c_eq_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx             = self.nu_chunks_indx_list[nu_chunk_indx]
                    lmbda_c_eq___nu_val = project(self.lmbda_c_eq_ufl_func(nu_indx, fem), fem.V_scalar)
                    chunks.lmbda_c_eq_chunks_val[meshpoint_indx][nu_chunk_indx] = lmbda_c_eq___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_c_eq_chunks.append(deepcopy(chunks.lmbda_c_eq_chunks_val))
        
        # lmbda_nu
        if ppp.save_lmbda_nu_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx            = self.nu_chunks_indx_list[nu_chunk_indx]
                lmbda_nu___nu_val  = project(self.lmbda_nu_ufl_func(nu_indx, fem), fem.V_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Segment stretch nu = "+nu_str
                parameter_str = "lmbda_nu___nu_"+nu_str+"_val"

                lmbda_nu___nu_val.rename(name_str, parameter_str)
                file_results.write(lmbda_nu___nu_val, deformation.t_val)
        
        if ppp.save_lmbda_nu_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx           = self.nu_chunks_indx_list[nu_chunk_indx]
                    lmbda_nu___nu_val = project(self.lmbda_nu_ufl_func(nu_indx, fem), fem.V_scalar)
                    chunks.lmbda_nu_chunks_val[meshpoint_indx][nu_chunk_indx] = lmbda_nu___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_nu_chunks.append(deepcopy(chunks.lmbda_nu_chunks_val))
        
        # lmbda_nu_max
        if ppp.save_lmbda_nu_max_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx               = self.nu_chunks_indx_list[nu_chunk_indx]
                lmbda_nu_max___nu_val = project(self.lmbda_nu_max_ufl_func(nu_indx, fem), fem.V_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Maximum segment stretch nu = "+nu_str
                parameter_str = "lmbda_nu___nu_"+nu_str+"_val"

                lmbda_nu_max___nu_val.rename(name_str, parameter_str)
                file_results.write(lmbda_nu_max___nu_val, deformation.t_val)
        
        if ppp.save_lmbda_nu_max_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx               = self.nu_chunks_indx_list[nu_chunk_indx]
                    lmbda_nu_max___nu_val = project(self.lmbda_nu_max_ufl_func(nu_indx, fem), fem.V_scalar)
                    chunks.lmbda_nu_max_chunks_val[meshpoint_indx][nu_chunk_indx] = lmbda_nu_max___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_nu_max_chunks.append(deepcopy(chunks.lmbda_nu_max_chunks_val))
        
        # upsilon_c
        if ppp.save_upsilon_c_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx            = self.nu_chunks_indx_list[nu_chunk_indx]
                upsilon_c___nu_val = project(self.upsilon_c_ufl_func(nu_indx, fem), fem.V_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Chain survival nu = "+nu_str
                parameter_str = "upsilon_c___nu_"+nu_str+"_val"

                upsilon_c___nu_val.rename(name_str, parameter_str)
                file_results.write(upsilon_c___nu_val, deformation.t_val)
        
        if ppp.save_upsilon_c_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx            = self.nu_chunks_indx_list[nu_chunk_indx]
                    upsilon_c___nu_val = project(self.upsilon_c_ufl_func(nu_indx, fem), fem.V_scalar)
                    chunks.upsilon_c_chunks_val[meshpoint_indx][nu_chunk_indx] = upsilon_c___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.upsilon_c_chunks.append(deepcopy(chunks.upsilon_c_chunks_val))

        # Upsilon_c
        if ppp.save_Upsilon_c_mesh:
            Upsilon_c_val = project(self.Upsilon_c_ufl_func(fem), fem.V_scalar)
            Upsilon_c_val.rename("Average chain survival", "Upsilon_c_val")
            file_results.write(Upsilon_c_val, deformation.t_val)
        
        if ppp.save_Upsilon_c_chunks:
            Upsilon_c_val = project(self.Upsilon_c_ufl_func(fem), fem.V_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.Upsilon_c_chunks_val[meshpoint_indx] = Upsilon_c_val(gp.meshpoints[meshpoint_indx])
            chunks.Upsilon_c_chunks.append(deepcopy(chunks.Upsilon_c_chunks_val))
        
        # d_c
        if ppp.save_d_c_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx      = self.nu_chunks_indx_list[nu_chunk_indx]
                d_c___nu_val = project(self.d_c_ufl_func(nu_indx, fem), fem.V_scalar)
                
                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Chain damage nu = "+nu_str
                parameter_str = "d_c___nu_"+nu_str+"_val"

                d_c___nu_val.rename(name_str, parameter_str)
                file_results.write(d_c___nu_val, deformation.t_val)
        
        if ppp.save_d_c_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx      = self.nu_chunks_indx_list[nu_chunk_indx]
                    d_c___nu_val = project(self.d_c_ufl_func(nu_indx, fem), fem.V_scalar)
                    chunks.d_c_chunks_val[meshpoint_indx][nu_chunk_indx] = d_c___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.d_c_chunks.append(deepcopy(chunks.d_c_chunks_val))

        # D_c
        if ppp.save_D_c_mesh:
            D_c_val = project(self.D_c_ufl_func(fem), fem.V_scalar)
            D_c_val.rename("Average chain damage", "D_c_val")
            file_results.write(D_c_val, deformation.t_val)
        
        if ppp.save_D_c_chunks:
            D_c_val = project(self.D_c_ufl_func(fem), fem.V_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.D_c_chunks_val[meshpoint_indx] = D_c_val(gp.meshpoints[meshpoint_indx])
            chunks.D_c_chunks.append(deepcopy(chunks.D_c_chunks_val))
        
        # epsilon_cnu_diss_hat
        if ppp.save_epsilon_cnu_diss_mesh: pass
        
        if ppp.save_epsilon_cnu_diss_chunks: pass
        
        # Epsilon_cnu_diss_hat
        if ppp.save_Epsilon_cnu_diss_mesh: pass
        
        if ppp.save_Epsilon_cnu_diss_chunks: pass
        
        # epsilon_c_diss_hat
        if ppp.save_epsilon_c_diss_mesh: pass
        
        if ppp.save_epsilon_c_diss_chunks: pass
        
        # Epsilon_c_diss_hat
        if ppp.save_Epsilon_c_diss_mesh: pass
        
        if ppp.save_Epsilon_c_diss_chunks: pass
        
        # overline_epsilon_cnu_diss_hat
        if ppp.save_overline_epsilon_cnu_diss_mesh: pass
        
        if ppp.save_overline_epsilon_cnu_diss_chunks: pass
        
        # overline_Epsilon_cnu_diss_hat
        if ppp.save_overline_Epsilon_cnu_diss_mesh: pass
        
        if ppp.save_overline_Epsilon_cnu_diss_chunks: pass
        
        # overline_epsilon_c_diss_hat
        if ppp.save_overline_epsilon_c_diss_mesh: pass
        
        if ppp.save_overline_epsilon_c_diss_chunks: pass
        
        # overline_Epsilon_c_diss_hat
        if ppp.save_overline_Epsilon_c_diss_mesh: pass
        
        if ppp.save_overline_Epsilon_c_diss_chunks: pass
        
        return chunks

    def pk2_stress_ufl_func(self, fem):
        pk2_stress_val = Constant(0.0)*fem.I
        for nu_indx in range(self.nu_num):
            # determine equilibrium chain stretch and segement stretch
            A_nu___nu_val           = self.single_chain_list[nu_indx].A_nu
            lmbda_c_eq___nu_val     = fem.lmbda_c*A_nu___nu_val
            lmbda_c_eq_max___nu_val = fem.lmbda_c_max*A_nu___nu_val
            lmbda_nu___nu_val       = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq___nu_val)
            lmbda_nu_max___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
            # determine chain damage
            upsilon_c___nu_val = self.single_chain_list[nu_indx].upsilon_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
            # determine stress response
            pk2_stress_val += upsilon_c___nu_val*self.P_nu_list[nu_indx]*self.nu_list[nu_indx]*self.A_nu_list[nu_indx]*self.single_chain_list[nu_indx].xi_c_ufl_func(lmbda_nu___nu_val, lmbda_c_eq___nu_val)/(3.*fem.lmbda_c)*fem.F
        pk2_stress_val += self.K_G*(fem.J-1)*fem.J*fem.F_inv.T
        return pk2_stress_val
    
    def Upsilon_c_ufl_func(self, fem):
        Upsilon_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            A_nu___nu_val           = self.single_chain_list[nu_indx].A_nu
            lmbda_c_eq_max___nu_val = fem.lmbda_c_max*A_nu___nu_val
            lmbda_nu_max___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
            upsilon_c___nu_val      = self.single_chain_list[nu_indx].upsilon_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
            Upsilon_c_val           += self.P_nu_list[nu_indx]*upsilon_c___nu_val
        Upsilon_c_val = Upsilon_c_val/self.P_nu_sum
        return Upsilon_c_val

    def D_c_ufl_func(self, fem):
        D_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            A_nu___nu_val           = self.single_chain_list[nu_indx].A_nu
            lmbda_c_eq_max___nu_val = fem.lmbda_c_max*A_nu___nu_val
            lmbda_nu_max___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
            d_c___nu_val            = self.single_chain_list[nu_indx].d_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
            D_c_val                 += self.P_nu_list[nu_indx]*d_c___nu_val
        D_c_val = D_c_val/self.P_nu_sum
        return D_c_val
    
    def lmbda_c_eq_ufl_func(self, nu_indx, fem):
        A_nu___nu_val       = self.single_chain_list[nu_indx].A_nu
        lmbda_c_eq___nu_val = fem.lmbda_c*A_nu___nu_val
        return lmbda_c_eq___nu_val

    def lmbda_nu_ufl_func(self, nu_indx, fem):
        A_nu___nu_val       = self.single_chain_list[nu_indx].A_nu
        lmbda_c_eq___nu_val = fem.lmbda_c*A_nu___nu_val
        lmbda_nu___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq___nu_val)
        return lmbda_nu___nu_val
    
    def lmbda_nu_max_ufl_func(self, nu_indx, fem):
        A_nu___nu_val           = self.single_chain_list[nu_indx].A_nu
        lmbda_c_eq_max___nu_val = fem.lmbda_c_max*A_nu___nu_val
        lmbda_nu_max___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
        return lmbda_nu_max___nu_val

    def upsilon_c_ufl_func(self, nu_indx, fem):
        A_nu___nu_val           = self.single_chain_list[nu_indx].A_nu
        lmbda_c_eq_max___nu_val = fem.lmbda_c_max*A_nu___nu_val
        lmbda_nu_max___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
        upsilon_c___nu_val      = self.single_chain_list[nu_indx].upsilon_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
        return upsilon_c___nu_val

    def d_c_ufl_func(self, nu_indx, fem):
        A_nu___nu_val           = self.single_chain_list[nu_indx].A_nu
        lmbda_c_eq_max___nu_val = fem.lmbda_c_max*A_nu___nu_val
        lmbda_nu_max___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
        d_c___nu_val            = self.single_chain_list[nu_indx].d_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
        return d_c___nu_val










    # def pk2_stress_ufl_func(self, fem):
    #     pk2_stress_val = Constant(0.0)*fem.I
    #     for nu_indx in range(self.nu_num):
    #         # determine equilibrium chain stretch and segement stretch
    #         A_nu___nu_val           = self.single_chain_list[nu_indx].A_nu
    #         lmbda_c_eq___nu_val     = fem.lmbda_c*A_nu___nu_val
    #         lmbda_c_eq_max___nu_val = fem.lmbda_c_max_prior*A_nu___nu_val
    #         lmbda_nu___nu_val       = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq___nu_val)
    #         lmbda_nu_max___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
    #         # determine chain damage
    #         upsilon_c___nu_val = self.single_chain_list[nu_indx].upsilon_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
    #         # determine stress response
    #         pk2_stress_val += upsilon_c___nu_val*self.P_nu_list[nu_indx]*self.nu_list[nu_indx]*self.A_nu_list[nu_indx]*self.single_chain_list[nu_indx].xi_c_ufl_func(lmbda_nu___nu_val, lmbda_c_eq___nu_val)/(3.*fem.lmbda_c)*fem.F
    #     pk2_stress_val += self.K_G*(fem.J-1)*fem.J*fem.F_inv.T
    #     return pk2_stress_val
    
    # def Upsilon_c_ufl_func(self, fem):
    #     Upsilon_c_val = Constant(0.0)
    #     for nu_indx in range(self.nu_num):
    #         A_nu___nu_val           = self.single_chain_list[nu_indx].A_nu
    #         lmbda_c_eq_max___nu_val = fem.lmbda_c_max_prior*A_nu___nu_val
    #         lmbda_nu_max___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
    #         upsilon_c___nu_val      = self.single_chain_list[nu_indx].upsilon_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
    #         Upsilon_c_val           += self.P_nu_list[nu_indx]*upsilon_c___nu_val
    #     Upsilon_c_val = Upsilon_c_val/self.P_nu_sum
    #     return Upsilon_c_val

    # def D_c_ufl_func(self, fem):
    #     D_c_val = Constant(0.0)
    #     for nu_indx in range(self.nu_num):
    #         A_nu___nu_val           = self.single_chain_list[nu_indx].A_nu
    #         lmbda_c_eq_max___nu_val = fem.lmbda_c_max_prior*A_nu___nu_val
    #         lmbda_nu_max___nu_val   = self.single_chain_list[nu_indx].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
    #         d_c___nu_val            = self.single_chain_list[nu_indx].d_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
    #         D_c_val                 += self.P_nu_list[nu_indx]*d_c___nu_val
    #     D_c_val = D_c_val/self.P_nu_sum
    #     return D_c_val
    
    # def lmbda_c_eq_ufl_func(self, fem):
    #     A_nu___nu_val       = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].A_nu
    #     lmbda_c_eq___nu_val = fem.lmbda_c*A_nu___nu_val
    #     return lmbda_c_eq___nu_val

    # def lmbda_nu_ufl_func(self, fem):
    #     A_nu___nu_val       = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].A_nu
    #     lmbda_c_eq___nu_val = fem.lmbda_c*A_nu___nu_val
    #     lmbda_nu___nu_val   = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].lmbda_nu_ufl_func(lmbda_c_eq___nu_val)
    #     return lmbda_nu___nu_val
    
    # def lmbda_nu_max_ufl_func(self, fem):
    #     A_nu___nu_val           = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].A_nu
    #     lmbda_c_eq_max___nu_val = fem.lmbda_c_max_prior*A_nu___nu_val
    #     lmbda_nu_max___nu_val   = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
    #     return lmbda_nu_max___nu_val

    # def upsilon_c_ufl_func(self, fem):
    #     A_nu___nu_val           = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].A_nu
    #     lmbda_c_eq_max___nu_val = fem.lmbda_c_max_prior*A_nu___nu_val
    #     lmbda_nu_max___nu_val   = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
    #     upsilon_c___nu_val      = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].upsilon_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
    #     return upsilon_c___nu_val

    # def d_c_ufl_func(self, fem):
    #     A_nu___nu_val           = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].A_nu
    #     lmbda_c_eq_max___nu_val = fem.lmbda_c_max_prior*A_nu___nu_val
    #     lmbda_nu_max___nu_val   = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].lmbda_nu_ufl_func(lmbda_c_eq_max___nu_val)
    #     d_c___nu_val            = self.single_chain_list[self.nu_chunks_indx_list[self.nu_chunk_indx]].d_c_ufl_func(self.k_cond_val, lmbda_nu_max___nu_val, lmbda_c_eq_max___nu_val)
    #     return d_c___nu_val

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork:

    def __init__(self): pass


################################################################################################################################
# Generalized extensible freely jointed chain (gen-uFJC) characterization class
################################################################################################################################

class GeneralizeduFJCNetwork(TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStressIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStressIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStressNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStressNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStressCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStressCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork, TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork,
                                TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork, TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork,
                                ThreeDimensionalIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, ThreeDimensionalIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                ThreeDimensionalIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, ThreeDimensionalIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                ThreeDimensionalIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork, ThreeDimensionalIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork,
                                ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentNetwork, ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentNetwork,
                                ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork, ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork,
                                ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                ThreeDimensionalNearlyIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork, ThreeDimensionalNearlyIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork,
                                ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentNetwork, ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentNetwork,
                                ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork, ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork,
                                ThreeDimensionalCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork, ThreeDimensionalCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork,
                                ThreeDimensionalCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork, ThreeDimensionalCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork,
                                ThreeDimensionalCompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork, ThreeDimensionalCompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork,
                                ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentNetwork, ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentNetwork,
                                ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork, ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork):
    
    ############################################################################################################################
    # Initialization
    ############################################################################################################################
    
    def __init__(self, parameters, strong_form_initialize_sigma_chunks, lr_cg_deformation_gradient_func, strong_form_calculate_sigma_func, strong_form_store_calculated_sigma_chunks, weak_form_initialize_deformation_sigma_chunks, weak_form_store_calculated_sigma_chunks, weak_form_store_calculated_deformation_chunks):

        # Check the correctness of the specified parameters
        if hasattr(parameters, "material") == False or hasattr(parameters, "deformation") == False:
            sys.exit("Need to specify either material parameters, deformation parameters, or both in order to define the generalized uFJC network")
        
        mp = parameters.material
        dp = parameters.deformation

        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks           = strong_form_initialize_sigma_chunks
        self.lr_cg_deformation_gradient_func               = lr_cg_deformation_gradient_func
        self.strong_form_calculate_sigma_func              = strong_form_calculate_sigma_func
        self.strong_form_store_calculated_sigma_chunks     = strong_form_store_calculated_sigma_chunks
        self.weak_form_initialize_deformation_sigma_chunks = weak_form_initialize_deformation_sigma_chunks
        self.weak_form_store_calculated_sigma_chunks       = weak_form_store_calculated_sigma_chunks
        self.weak_form_store_calculated_deformation_chunks = weak_form_store_calculated_deformation_chunks

        # Retain specified parameters
        self.network_model                      = getattr(mp, "network_model")
        self.physical_dimension                 = getattr(mp, "physical_dimension")
        self.incompressibility_assumption       = getattr(mp, "incompressibility_assumption")
        self.macro2micro_deformation_assumption = getattr(mp, "macro2micro_deformation_assumption")
        self.micro2macro_homogenization_scheme  = getattr(mp, "micro2macro_homogenization_scheme")
        self.chain_level_load_sharing           = getattr(mp, "chain_level_load_sharing")
        self.rate_dependence                    = getattr(mp, "rate_dependence")
        self.two_dimensional_formulation        = getattr(mp, "two_dimensional_formulation")
        self.microdisk_quadrature_order         = getattr(mp, "microdisk_quadrature_order")
        self.microsphere_quadrature_order       = getattr(mp, "microsphere_quadrature_order")

        self.omega_0 = getattr(mp, "omega_0")

        self.nu_chunks_indx_list    = getattr(mp, "nu_chunks_indx_list")
        self.point_chunks_indx_list = getattr(mp, "point_chunks_indx_list")

        self.deformation_type        = getattr(dp, "deformation_type")
        self.K_G                     = getattr(dp, "K_G")
        self.lmbda_damping_init      = getattr(dp, "lmbda_damping_init")
        self.min_lmbda_damping_val   = getattr(dp, "min_lmbda_damping_val")
        self.iter_max_Gamma_val_NR   = getattr(dp, "iter_max_Gamma_val_NR")
        self.tol_Gamma_val_NR        = getattr(dp, "tol_Gamma_val_NR")
        self.iter_max_lmbda_c_val_NR = getattr(dp, "iter_max_lmbda_c_val_NR")
        self.tol_lmbda_c_val_NR      = getattr(dp, "tol_lmbda_c_val_NR")
        self.iter_max_stag_NR        = getattr(dp, "iter_max_stag_NR")
        self.tol_lmbda_c_val_stag_NR = getattr(dp, "tol_lmbda_c_val_stag_NR")
        self.tol_Gamma_val_stag_NR   = getattr(dp, "tol_Gamma_val_stag_NR")
        self.epsilon                 = getattr(dp, "epsilon")
        self.max_J_val_cond          = getattr(dp, "max_J_val_cond")
        self.iter_max_d_c_val        = getattr(dp, "iter_max_d_c_val")
        self.tol_d_c_val             = getattr(dp, "tol_d_c_val")
        self.k_cond_val              = getattr(dp, "k_cond_val")


        if self.network_model != "statistical_mechanics_model":
            sys.exit("Error: This GeneralizeduFJCNetwork material class corresponds to a statistical mechanics model.")
        
        if self.physical_dimension != 2 and self.physical_dimension != 3:
            sys.exit("Error: Need to specify either a 2D or a 3D problem.")
        
        if self.incompressibility_assumption != "incompressible" and self.incompressibility_assumption != "nearly_incompressible" and self.incompressibility_assumption != "compressible":
            sys.exit("Error: Need to specify a proper incompressibility assumption for the material. The material is either incompressible, nearly incompressible, or compressible>")

        if self.macro2micro_deformation_assumption != 'affine' and self.macro2micro_deformation_assumption != 'nonaffine':
            sys.exit('Error: Need to specify the macro-to-micro deformation assumption in the network. Either affine deformation or non-affine deformation can be used.')
        
        if self.micro2macro_homogenization_scheme != 'eight_chain_model' and self.micro2macro_homogenization_scheme != 'full_network_microdisk_model' and self.micro2macro_homogenization_scheme != 'full_network_microsphere_model':
            sys.exit('Error: Need to specify the micro-to-macro homogenization scheme in the network. Either the eight chain model, the full network microdisk micro-to-macro homogenization scheme, or the full network microdisk micro-to-macro homogenization scheme can be used.')
        
        if self.chain_level_load_sharing != 'equal_strain' and self.chain_level_load_sharing != 'equal_force':
            sys.exit('Error: Need to specify the load sharing assumption that the network/chains in the generalized uFJC network obey. Either the equal strain chain level load sharing assumption or the equal force chain level load sharing assumption can be used.')
        
        if self.rate_dependence != 'rate_dependent' and self.rate_dependence != 'rate_independent':
            sys.exit('Error: Need to specify the network/chain dependence on the rate of applied deformation. Either rate-dependent or rate-independent deformation can be used.')
        
        if self.rate_dependence == 'rate_dependent' and self.omega_0 is None:
            sys.exit('Error: Need to specify the microscopic frequency of segments in the network for rate-dependent network deformation.')
        
        if self.macro2micro_deformation_assumption == 'affine' and self.micro2macro_homogenization_scheme == 'eight_chain_model':
            sys.exit('Error: The eight chain micro-to-macro homogenization scheme technically exhibits the non-affine macro-to-micro deformation assumption.')
        
        if self.macro2micro_deformation_assumption == 'nonaffine' and (self.micro2macro_homogenization_scheme == 'full_network_microsphere_model' or self.micro2macro_homogenization_scheme == 'full_network_microdisk_model') and self.chain_level_load_sharing == 'equal_strain':
            sys.exit('Error: In the non-affine macro-to-micro deformation assumption utilizing either the full network microsphere micro-to-macro homogenization scheme or the full network microdisk micro-to-macro homogenization scheme, the generalized uFJCs are required to obey the equal force load sharing assumption.')
        
        if self.physical_dimension == 2:
            if self.two_dimensional_formulation != "plane_strain" and self.two_dimensional_formulation != "generalized_plane_strain" and self.two_dimensional_formulation != "plane_stress":
                sys.exit("Error: Need to specify a proper two-dimensional formulation. Either plane strain, generalized plane strain, or plane stress can be used.")
            
            if self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                sys.exit("Error: For a 2D problem, the full network microsphere micro-to-macro homogenization scheme cannot be used. Either the eight chain model or the full network microdisk micro-to-macro homogenization scheme can be used for 2D problems.")
            
            # Specify full network microdisk quadrature scheme, if necessary
            elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                if self.microdisk_quadrature_order is None:
                    sys.exit('Error: Need to specify microdisk quadrature order number in order to utilize the full network microdisk micro-to-macro homogenization scheme.')
                else:
                    self.microdisk = MicrodiskQuadratureScheme(self.microdisk_quadrature_order)
        
        elif self.physical_dimension == 3:
            if self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                sys.exit("Error: For a 2D problem, the full network microdisk micro-to-macro homogenization scheme cannot be used. Either the eight chain model or the full network microsphere micro-to-macro homogenization scheme can be used for 2D problems.")
            
            # Specify full network microsphere quadrature scheme, if necessary
            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                if self.microsphere_quadrature_order is None:
                    sys.exit('Error: Need to specify microsphere quadrature order number in order to utilize the full network microsphere micro-to-macro homogenization scheme.')
                else:
                    self.microsphere = MicrosphereQuadratureScheme(self.microsphere_quadrature_order)
        
        
        # Specify chain composition
        if self.chain_level_load_sharing == 'equal_strain': self.equal_strain_generalized_ufjc_network(mp)
        elif self.chain_level_load_sharing == 'equal_force': self.equal_force_generalized_ufjc_network(mp)

        # Specify network. DO NOT CHANGE THESE CONDITIONAL STATEMENTS WHATSOEVER!!
        if self.physical_dimension == 2:
            if self.two_dimensional_formulation == "plane_strain":
                if self.incompressibility_assumption == "incompressible":
                    if self.macro2micro_deformation_assumption == 'nonaffine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain': pass
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                    elif self.macro2micro_deformation_assumption == 'affine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent': 
                                    TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                elif self.incompressibility_assumption == "nearly_incompressible":
                    if self.macro2micro_deformation_assumption == 'nonaffine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain': pass
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                    elif self.macro2micro_deformation_assumption == 'affine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent': 
                                    TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                elif self.incompressibility_assumption == "compressible":
                    if self.macro2micro_deformation_assumption == 'nonaffine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain': pass
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                    elif self.macro2micro_deformation_assumption == 'affine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent': 
                                    TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
            elif self.two_dimensional_formulation == "generalized_plane_strain":
                if self.incompressibility_assumption == "incompressible":
                    if self.macro2micro_deformation_assumption == 'nonaffine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain': pass
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                    elif self.macro2micro_deformation_assumption == 'affine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent': 
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                elif self.incompressibility_assumption == "nearly_incompressible":
                    if self.macro2micro_deformation_assumption == 'nonaffine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain': pass
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                    elif self.macro2micro_deformation_assumption == 'affine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent': 
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                elif self.incompressibility_assumption == "compressible":
                    if self.macro2micro_deformation_assumption == 'nonaffine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain': pass
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                    elif self.macro2micro_deformation_assumption == 'affine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent': 
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
            elif self.two_dimensional_formulation == "plane_stress":
                if self.incompressibility_assumption == "incompressible":
                    if self.macro2micro_deformation_assumption == 'nonaffine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain': pass
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                    elif self.macro2micro_deformation_assumption == 'affine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent': 
                                    TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                elif self.incompressibility_assumption == "nearly_incompressible":
                    if self.macro2micro_deformation_assumption == 'nonaffine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain': pass
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressNearlyIncompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                    elif self.macro2micro_deformation_assumption == 'affine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent': 
                                    TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                elif self.incompressibility_assumption == "compressible":
                    if self.macro2micro_deformation_assumption == 'nonaffine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain': pass
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressCompressibleNonaffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                        elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
                    elif self.macro2micro_deformation_assumption == 'affine':
                        if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                        elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model':
                            if self.chain_level_load_sharing == 'equal_strain':
                                if self.rate_dependence == 'rate_independent': 
                                    TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualStrainRateDependentNetwork.__init__(self)
                            elif self.chain_level_load_sharing == 'equal_force':
                                if self.rate_dependence == 'rate_independent':
                                    TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualForceRateIndependentNetwork.__init__(self)
                                elif self.rate_dependence == 'rate_dependent':
                                    TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrodiskModelEqualForceRateDependentNetwork.__init__(self)
                            elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model': pass
        elif self.physical_dimension == 3:
            if self.incompressibility_assumption == "incompressible":
                if self.macro2micro_deformation_assumption == 'nonaffine':
                    if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                        if self.chain_level_load_sharing == 'equal_strain':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                        elif self.chain_level_load_sharing == 'equal_force':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                    elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                        if self.chain_level_load_sharing == 'equal_strain': pass
                        elif self.chain_level_load_sharing == 'equal_force':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork.__init__(self)
                    elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model': pass
                elif self.macro2micro_deformation_assumption == 'affine':
                    if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                    elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                        if self.chain_level_load_sharing == 'equal_strain':
                            if self.rate_dependence == 'rate_independent': 
                                ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentNetwork.__init__(self)
                        elif self.chain_level_load_sharing == 'equal_force':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork.__init__(self)
                    elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model': pass
            elif self.incompressibility_assumption == "nearly_incompressible":
                if self.macro2micro_deformation_assumption == 'nonaffine':
                    if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                        if self.chain_level_load_sharing == 'equal_strain':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                        elif self.chain_level_load_sharing == 'equal_force':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                    elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                        if self.chain_level_load_sharing == 'equal_strain': pass
                        elif self.chain_level_load_sharing == 'equal_force':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalNearlyIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalNearlyIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork.__init__(self)
                    elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model': pass
                elif self.macro2micro_deformation_assumption == 'affine':
                    if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                    elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                        if self.chain_level_load_sharing == 'equal_strain':
                            if self.rate_dependence == 'rate_independent': 
                                ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentNetwork.__init__(self)
                        elif self.chain_level_load_sharing == 'equal_force':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork.__init__(self)
                    elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model': pass
            elif self.incompressibility_assumption == "compressible":
                if self.macro2micro_deformation_assumption == 'nonaffine':
                    if self.micro2macro_homogenization_scheme == 'eight_chain_model':
                        if self.chain_level_load_sharing == 'equal_strain':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalCompressibleNonaffineEightChainModelEqualStrainRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalCompressibleNonaffineEightChainModelEqualStrainRateDependentNetwork.__init__(self)
                        elif self.chain_level_load_sharing == 'equal_force':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalCompressibleNonaffineEightChainModelEqualForceRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalCompressibleNonaffineEightChainModelEqualForceRateDependentNetwork.__init__(self)
                    elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                        if self.chain_level_load_sharing == 'equal_strain': pass
                        elif self.chain_level_load_sharing == 'equal_force':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalCompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalCompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork.__init__(self)
                    elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model': pass
                elif self.macro2micro_deformation_assumption == 'affine':
                    if self.micro2macro_homogenization_scheme == 'eight_chain_model': pass
                    elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                        if self.chain_level_load_sharing == 'equal_strain':
                            if self.rate_dependence == 'rate_independent': 
                                ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentNetwork.__init__(self)
                        elif self.chain_level_load_sharing == 'equal_force':
                            if self.rate_dependence == 'rate_independent':
                                ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentNetwork.__init__(self)
                            elif self.rate_dependence == 'rate_dependent':
                                ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentNetwork.__init__(self)
                    elif self.micro2macro_homogenization_scheme == 'full_network_microdisk_model': pass
    
    def equal_strain_generalized_ufjc_network(self, material_parameters):
        
        mp = material_parameters

        # List of single chains obeying the equal strain chain level load sharing assumption
        single_chain_list = [EqualStrainGeneralizeduFJC(chain_level_load_sharing = mp.chain_level_load_sharing, rate_dependence = mp.rate_dependence, omega_0 = mp.omega_0, nu = mp.nu_list[nu_indx], nu_b = mp.nu_b, zeta_b_char = mp.zeta_b_char, kappa_b = mp.kappa_b, zeta_nu_char = mp.zeta_nu_char, kappa_nu = mp.kappa_nu) for nu_indx in range(len(mp.nu_list))]
        
        # Separate out specified parameters
        nu_list                   = [single_chain_list[nu_indx].nu for nu_indx in range(len(single_chain_list))]
        nu_min                    = min(nu_list)
        nu_max                    = max(nu_list)
        nu_num                    = len(nu_list)
        P_nu_list                 = [self.P_nu(mp, nu_list[nu_indx]) for nu_indx in range(len(nu_list))]
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
        self.single_chain_list         = single_chain_list
        self.nu_list                   = nu_list
        self.nu_min                    = nu_min
        self.nu_max                    = nu_max
        self.nu_num                    = nu_num
        self.P_nu_list                 = P_nu_list
        self.P_nu_sum                  = P_nu_sum
        self.nu_b                      = nu_b
        self.zeta_b_char               = zeta_b_char
        self.kappa_b                   = kappa_b
        self.zeta_nu_char              = zeta_nu_char
        self.kappa_nu                  = kappa_nu
        self.lmbda_nu_ref              = lmbda_nu_ref
        self.lmbda_c_eq_ref            = lmbda_c_eq_ref
        self.lmbda_nu_crit             = lmbda_nu_crit
        self.lmbda_c_eq_crit           = lmbda_c_eq_crit
        self.xi_c_crit                 = xi_c_crit
        self.lmbda_nu_pade2berg_crit   = lmbda_nu_pade2berg_crit
        self.lmbda_c_eq_pade2berg_crit = lmbda_c_eq_pade2berg_crit
        self.A_nu_list                 = A_nu_list
        self.Lambda_nu_ref_list        = Lambda_nu_ref_list
    
    def equal_force_generalized_ufjc_network(self, material_parameters):
        
        mp = material_parameters

        # List of single chains
        single_chain_list = [GeneralizeduFJC(chain_level_load_sharing = mp.chain_level_load_sharing, rate_dependence = mp.rate_dependence, omega_0 = mp.omega_0, nu = nu_list[nu_indx], zeta_b_char = zeta_b_char, kappa_b = kappa_b) for nu_indx in range(len(nu_list))]
        
        # Separate out specified parameters
        nu_list   = [single_chain_list[nu_indx].nu for nu_indx in range(len(single_chain_list))]
        nu_min    = min(nu_list)
        nu_max    = max(nu_list)
        nu_num    = len(nu_list)
        P_nu_list = [P_nu(nu_list[nu_indx]) for nu_indx in range(len(nu_list))]
        A_nu_list = [single_chain_list[nu_indx].A_nu for nu_indx in range(len(single_chain_list))]
        
        # Calculate the statistical equal force chain parameters
        Nu = np.sum([P_nu(nu_list[nu_indx])*nu_list[nu_indx] for nu_indx in range(len(nu_list))])
        A  = np.sum([P_nu(nu_list[nu_indx])*A_nu_list[nu_indx]*nu_list[nu_indx] for nu_indx in range(len(nu_list))])/Nu
        single_chain = EqualForceGeneralizeduFJC(chain_level_load_sharing = mp.chain_level_load_sharing, rate_dependence = mp.rate_dependence, omega_0 = mp.omega_0, Nu = Nu, A = A, zeta_b_char = zeta_b_char, kappa_b = kappa_b)
        
        # Separate out specified parameters
        kappa_b                   = single_chain.kappa_b
        zeta_b_char               = single_chain.zeta_b_char
        lmbda_b_ref               = single_chain.lmbda_b_ref
        lmbda_cA_ref              = single_chain.lmbda_cA_ref
        lmbda_b_crit              = single_chain.lmbda_b_crit
        lmbda_cA_crit             = single_chain.lmbda_cA_crit
        lmbda_b_pade2berg_crit    = single_chain.lmbda_b_pade2berg_crit
        lmbda_cA_pade2berg_crit   = single_chain.lmbda_cA_pade2berg_crit
        lmbda_b_bsci_hat_rms      = single_chain.lmbda_b_bsci_hat_rms
        epsilon_b_sci_hat_rms     = single_chain.epsilon_b_sci_hat_rms
        lmbda_b_csci_hat_rms      = single_chain.lmbda_b_csci_hat_rms
        epsilon_c_sci_hat_rms     = single_chain.epsilon_c_sci_hat_rms
        epsilon_c_sci_hat_rms__Nu = epsilon_c_sci_hat_rms/Nu
        Lambda_b_ref              = single_chain.Lambda_b_ref
        
        # Retain specified parameters
        self.nu_list                   = nu_list
        self.nu_min                    = nu_min
        self.nu_max                    = nu_max
        self.nu_num                    = nu_num
        self.P_nu_list                 = P_nu_list
        self.Nu                        = Nu
        self.A                         = A
        self.single_chain              = single_chain
        self.kappa_b                   = kappa_b
        self.zeta_b_char               = zeta_b_char
        self.lmbda_b_ref               = lmbda_b_ref
        self.lmbda_cA_ref              = lmbda_cA_ref
        self.lmbda_b_crit              = lmbda_b_crit
        self.lmbda_cA_crit             = lmbda_cA_crit
        self.lmbda_b_pade2berg_crit    = lmbda_b_pade2berg_crit
        self.lmbda_cA_pade2berg_crit   = lmbda_cA_pade2berg_crit
        self.lmbda_b_bsci_hat_rms      = lmbda_b_bsci_hat_rms
        self.epsilon_b_sci_hat_rms     = epsilon_b_sci_hat_rms
        self.lmbda_b_csci_hat_rms      = lmbda_b_csci_hat_rms
        self.epsilon_c_sci_hat_rms     = epsilon_c_sci_hat_rms
        self.epsilon_c_sci_hat_rms__Nu = epsilon_c_sci_hat_rms__Nu
        self.Lambda_b_ref              = Lambda_b_ref


    def P_nu(self, material_parameters, nu):
        
        mp = material_parameters

        if mp.nu_distribution == "itskov":
            return (1/(mp.Delta_nu+1))*(1+(1/mp.Delta_nu))**(mp.nu_min-nu)