################################################################################################################################
# General setup
################################################################################################################################

# Import necessary libraries
from __future__ import division
from dolfin import *
from .single_chain import GeneralizeduFJC, EqualStrainGeneralizeduFJC, EqualForceGeneralizeduFJC
from .microsphere_quadrature import MicrosphereQuadratureScheme
import sys
import numpy as np
from types import SimpleNamespace
from copy import deepcopy

class neoHookean:

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
        
        if fem.dimension == 2:
            fem.I_C = tr(fem.C)+1 # 2D form of the trace of right Cauchy-Green tensor, where F_33 = 1 always -- this is the case of plane strain
        elif fem.dimension == 3:
            fem.I_C = tr(fem.C) # 3D form of the trace of right Cauchy-Green tensor
        
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
        if self.deformation_type == 'uniaxial':
            chunks.F_11_chunks         = []
            chunks.F_11_chunks_val     = [0. for meshpoint_indx in range(len(gp.meshpoints))]
            chunks.sigma_11_chunks     = []
            chunks.sigma_11_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        elif self.deformation_type == 'equibiaxial': pass
        elif self.deformation_type == 'simple_shear': pass
        
        return chunks
    
    def fenics_weak_form_post_processing(self, deformation, chunks, fem, file_results, parameters):
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
            if self.deformation_type == 'uniaxial':
                for meshpoint_indx in range(len(gp.meshpoints)):
                    chunks.sigma_11_chunks_val[meshpoint_indx] = sigma_val(gp.meshpoints[meshpoint_indx])[femp.tensor2vector_indx_dict["11"]]
                chunks.sigma_11_chunks.append(deepcopy(chunks.sigma_11_chunks_val))
            
            elif self.deformation_type == 'equibiaxial': pass
            elif self.deformation_type == 'simple_shear': pass
        
        # F
        if ppp.save_F_mesh:
            F_val = project(fem.F, fem.V_tensor)
            F_val.rename("Deformation gradient", "F_val")
            file_results.write(F_val, deformation.t_val)
        
        if ppp.save_F_chunks:
            F_val = project(fem.F, fem.V_tensor)
            if self.deformation_type == 'uniaxial':
                for meshpoint_indx in range(len(gp.meshpoints)):
                    chunks.F_11_chunks_val[meshpoint_indx] = F_val(gp.meshpoints[meshpoint_indx])[femp.tensor2vector_indx_dict["11"]]
                chunks.F_11_chunks.append(deepcopy(chunks.F_11_chunks_val))
            
            elif self.deformation_type == 'equibiaxial': pass
            elif self.deformation_type == 'simple_shear': pass
    
        return chunks
    
    def pk2_stress_ufl_func(self, fem):
        return (fem.B + self.K_G*(fem.J-1)*fem.I)*fem.J*fem.F_inv.T

class PhenomenologicalNetwork(neoHookean):

    def __init__(self, parameters):

        # Check the correctness of the specified parameters
        if hasattr(parameters, "material") == False or hasattr(parameters, "deformation") == False:
            sys.exit("Need to specify either material parameters, deformation parameters, or both in order to define the generalized uFJC network")
        
        mp = parameters.material
        dp = parameters.deformation

        # Retain specified parameters
        self.phenomenological_model = getattr(mp, "phenomenological_model")
        
        self.deformation_type        = getattr(dp, "deformation_type")
        self.K_G                     = getattr(dp, "K_G")

        # Specify phenomenological model. Do not change these conditional statements!!
        if self.phenomenological_model == 'neo_Hookean':
            neoHookean.__init__(self)