################################################################################################################################
# General setup
################################################################################################################################

# Import necessary libraries
from __future__ import division
from dolfin import *
from .microsphere_quadrature import MicrosphereQuadratureScheme
from .microdisk_quadrature import MicrodiskQuadratureScheme
import sys
import numpy as np
from types import SimpleNamespace
from copy import deepcopy

class TwoDimensionalPlaneStrainIncompressibleNeoHookeanRateIndependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalPlaneStrainIncompressibleNeoHookeanRateDependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalPlaneStrainNearlyIncompressibleNeoHookeanRateIndependentNetwork:

    def __init__(self):
        pass

    def homogeneous_strong_form_initialization(self):
        pass

    def homogeneous_strong_form_solve_step(self, deformation, results):
        pass

    def homogeneous_strong_form_post_processing(self, deformation, results, chunks):
        pass

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

        # Kinematics
        fem.F     = fem.I + grad(fem.u) # deformation gradient tensor
        fem.F_inv = inv(fem.F) # inverse deformation gradient tensor
        fem.J     = det(fem.F) # volume ratio
        fem.C     = fem.F.T*fem.F # right Cauchy-Green tensor
        fem.B     = fem.F*fem.F.T # left Cauchy-Green tensor
        fem.I_C = tr(fem.C)+1 # 2D form of the trace of right Cauchy-Green tensor, where F_33 = 1 always -- this is the case of plane strain
        
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
    
    def fenics_weak_form_solve_step(self, fem):
        """
        Solve the weak form, which will provide the solution to the displacements and diffuse chain damage, and account for network irreversibility
        """
        fem = self.solve_u(fem)

        return fem
    
    def fenics_weak_form_initialization(self, parameters):
        gp      = parameters.geometry
        chunks  = SimpleNamespace()

        # initialize lists to zeros - necessary for irreversibility
        chunks = self.weak_form_initialize_deformation_sigma_chunks(gp.meshpoints, chunks)
        
        return chunks
    
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
    
        return chunks
    
    def pk2_stress_ufl_func(self, fem):
        return (fem.B + self.K_G*(fem.J-1)*fem.I)*fem.J*fem.F_inv.T

class TwoDimensionalPlaneStrainNearlyIncompressibleNeoHookeanRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleNeoHookeanRateIndependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalPlaneStrainCompressibleNeoHookeanRateDependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNeoHookeanRateIndependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNeoHookeanRateDependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNeoHookeanRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNeoHookeanRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleNeoHookeanRateIndependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalGeneralizedPlaneStrainCompressibleNeoHookeanRateDependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleNeoHookeanRateIndependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalPlaneStressIncompressibleNeoHookeanRateDependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleNeoHookeanRateIndependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressNearlyIncompressibleNeoHookeanRateDependentNetwork:

    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleNeoHookeanRateIndependentNetwork:
    
    def __init__(self): pass

class TwoDimensionalPlaneStressCompressibleNeoHookeanRateDependentNetwork:
    
    def __init__(self): pass

class ThreeDimensionalIncompressibleNeoHookeanRateIndependentNetwork:
    
    def __init__(self): pass

class ThreeDimensionalIncompressibleNeoHookeanRateDependentNetwork:
    
    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleNeoHookeanRateIndependentNetwork:

    def __init__(self): pass

class ThreeDimensionalNearlyIncompressibleNeoHookeanRateDependentNetwork:

    def __init__(self): pass

class ThreeDimensionalCompressibleNeoHookeanRateIndependentNetwork:
    
    def __init__(self): pass

class ThreeDimensionalCompressibleNeoHookeanRateDependentNetwork:
    
    def __init__(self): pass


class PhenomenologicalNetwork(TwoDimensionalPlaneStrainIncompressibleNeoHookeanRateIndependentNetwork, TwoDimensionalPlaneStrainIncompressibleNeoHookeanRateDependentNetwork,
                                TwoDimensionalPlaneStrainNearlyIncompressibleNeoHookeanRateIndependentNetwork, TwoDimensionalPlaneStrainNearlyIncompressibleNeoHookeanRateDependentNetwork,
                                TwoDimensionalPlaneStrainCompressibleNeoHookeanRateIndependentNetwork, TwoDimensionalPlaneStrainCompressibleNeoHookeanRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainIncompressibleNeoHookeanRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainIncompressibleNeoHookeanRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNeoHookeanRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNeoHookeanRateDependentNetwork,
                                TwoDimensionalGeneralizedPlaneStrainCompressibleNeoHookeanRateIndependentNetwork, TwoDimensionalGeneralizedPlaneStrainCompressibleNeoHookeanRateDependentNetwork,
                                TwoDimensionalPlaneStressIncompressibleNeoHookeanRateIndependentNetwork, TwoDimensionalPlaneStressIncompressibleNeoHookeanRateDependentNetwork,
                                TwoDimensionalPlaneStressNearlyIncompressibleNeoHookeanRateIndependentNetwork, TwoDimensionalPlaneStressNearlyIncompressibleNeoHookeanRateDependentNetwork,
                                TwoDimensionalPlaneStressCompressibleNeoHookeanRateIndependentNetwork, TwoDimensionalPlaneStressCompressibleNeoHookeanRateDependentNetwork,
                                ThreeDimensionalIncompressibleNeoHookeanRateIndependentNetwork, ThreeDimensionalIncompressibleNeoHookeanRateDependentNetwork,
                                ThreeDimensionalNearlyIncompressibleNeoHookeanRateIndependentNetwork, ThreeDimensionalNearlyIncompressibleNeoHookeanRateDependentNetwork,
                                ThreeDimensionalCompressibleNeoHookeanRateIndependentNetwork, ThreeDimensionalCompressibleNeoHookeanRateDependentNetwork):

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
        self.rate_dependence                    = getattr(mp, "rate_dependence")
        self.two_dimensional_formulation        = getattr(mp, "two_dimensional_formulation")

        self.phenomenological_model = getattr(mp, "phenomenological_model")
        
        self.deformation_type        = getattr(dp, "deformation_type")
        self.K_G                     = getattr(dp, "K_G")

        if self.network_model != "phenomenological_model":
            sys.exit("Error: This PhenomenologicalNetwork material class corresponds to some phenomenological model.")
        
        if self.physical_dimension != 2 and self.physical_dimension != 3:
            sys.exit("Error: Need to specify either a 2D or a 3D problem.")
        
        if self.incompressibility_assumption != "incompressible" and self.incompressibility_assumption != "nearly_incompressible" and self.incompressibility_assumption != "compressible":
            sys.exit("Error: Need to specify a proper incompressibility assumption for the material. The material is either incompressible, nearly incompressible, or compressible>")
        
        if self.rate_dependence != 'rate_dependent' and self.rate_dependence != 'rate_independent':
            sys.exit('Error: Need to specify the network/chain dependence on the rate of applied deformation. Either rate-dependent or rate-independent deformation can be used.')
        
        if self.physical_dimension == 2:
            if self.two_dimensional_formulation != "plane_strain" and self.two_dimensional_formulation != "generalized_plane_strain" and self.two_dimensional_formulation != "plane_stress":
                sys.exit("Error: Need to specify a proper two-dimensional formulation. Either plane strain, generalized plane strain, or plane stress can be used.")
        
        # Specify phenomenological model. Do not change these conditional statements!!
        if self.physical_dimension == 2:
            if self.two_dimensional_formulation == "plane_strain":
                if self.incompressibility_assumption == "incompressible":
                    if self.phenomenological_model == 'neo_Hookean':
                        if self.rate_dependence == 'rate_independent':
                            TwoDimensionalPlaneStrainIncompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                        elif self.rate_dependence == 'rate_dependent':
                            TwoDimensionalPlaneStrainIncompressibleNeoHookeanRateDependentNetwork.__init__(self)
                    else: pass
                elif self.incompressibility_assumption == "nearly_incompressible":
                    if self.phenomenological_model == 'neo_Hookean':
                        if self.rate_dependence == 'rate_independent':
                            TwoDimensionalPlaneStrainNearlyIncompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                        elif self.rate_dependence == 'rate_dependent':
                            TwoDimensionalPlaneStrainNearlyIncompressibleNeoHookeanRateDependentNetwork.__init__(self)
                    else: pass
                elif self.incompressibility_assumption == "compressible":
                    if self.phenomenological_model == 'neo_Hookean':
                        if self.rate_dependence == 'rate_independent':
                            TwoDimensionalPlaneStrainCompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                        elif self.rate_dependence == 'rate_dependent':
                            TwoDimensionalPlaneStrainCompressibleNeoHookeanRateDependentNetwork.__init__(self)
                    else: pass
            elif self.two_dimensional_formulation == "generalized_plane_strain":
                if self.incompressibility_assumption == "incompressible":
                    if self.phenomenological_model == 'neo_Hookean':
                        if self.rate_dependence == 'rate_independent':
                            TwoDimensionalGeneralizedPlaneStrainIncompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                        elif self.rate_dependence == 'rate_dependent':
                            TwoDimensionalGeneralizedPlaneStrainIncompressibleNeoHookeanRateDependentNetwork.__init__(self)
                    else: pass
                elif self.incompressibility_assumption == "nearly_incompressible":
                    if self.phenomenological_model == 'neo_Hookean':
                        if self.rate_dependence == 'rate_independent':
                            TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                        elif self.rate_dependence == 'rate_dependent':
                            TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNeoHookeanRateDependentNetwork.__init__(self)
                    else: pass
                elif self.incompressibility_assumption == "compressible":
                    if self.phenomenological_model == 'neo_Hookean':
                        if self.rate_dependence == 'rate_independent':
                            TwoDimensionalGeneralizedPlaneStrainCompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                        elif self.rate_dependence == 'rate_dependent':
                            TwoDimensionalGeneralizedPlaneStrainCompressibleNeoHookeanRateDependentNetwork.__init__(self)
                    else: pass
            elif self.two_dimensional_formulation == "plane_stress":
                if self.incompressibility_assumption == "incompressible":
                    if self.phenomenological_model == 'neo_Hookean':
                        if self.rate_dependence == 'rate_independent':
                            TwoDimensionalPlaneStressIncompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                        elif self.rate_dependence == 'rate_dependent':
                            TwoDimensionalPlaneStressIncompressibleNeoHookeanRateDependentNetwork.__init__(self)
                    else: pass
                elif self.incompressibility_assumption == "nearly_incompressible":
                    if self.phenomenological_model == 'neo_Hookean':
                        if self.rate_dependence == 'rate_independent':
                            TwoDimensionalPlaneStressNearlyIncompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                        elif self.rate_dependence == 'rate_dependent':
                            TwoDimensionalPlaneStressNearlyIncompressibleNeoHookeanRateDependentNetwork.__init__(self)
                    else: pass
                elif self.incompressibility_assumption == "compressible":
                    if self.phenomenological_model == 'neo_Hookean':
                        if self.rate_dependence == 'rate_independent':
                            TwoDimensionalPlaneStressCompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                        elif self.rate_dependence == 'rate_dependent':
                            TwoDimensionalPlaneStressCompressibleNeoHookeanRateDependentNetwork.__init__(self)
                    else: pass
        elif self.physical_dimension == 3:
            if self.incompressibility_assumption == "incompressible":
                if self.phenomenological_model == 'neo_Hookean':
                    if self.rate_dependence == 'rate_independent':
                        ThreeDimensionalIncompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                    elif self.rate_dependence == 'rate_dependent':
                        ThreeDimensionalIncompressibleNeoHookeanRateDependentNetwork.__init__(self)
                else: pass
            elif self.incompressibility_assumption == "nearly_incompressible":
                if self.phenomenological_model == 'neo_Hookean':
                    if self.rate_dependence == 'rate_independent':
                        ThreeDimensionalNearlyIncompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                    elif self.rate_dependence == 'rate_dependent':
                        ThreeDimensionalNearlyIncompressibleNeoHookeanRateDependentNetwork.__init__(self)
                else: pass
            elif self.incompressibility_assumption == "compressible":
                if self.phenomenological_model == 'neo_Hookean':
                    if self.rate_dependence == 'rate_independent':
                        ThreeDimensionalCompressibleNeoHookeanRateIndependentNetwork.__init__(self)
                    elif self.rate_dependence == 'rate_dependent':
                        ThreeDimensionalCompressibleNeoHookeanRateDependentNetwork.__init__(self)
                else: pass
