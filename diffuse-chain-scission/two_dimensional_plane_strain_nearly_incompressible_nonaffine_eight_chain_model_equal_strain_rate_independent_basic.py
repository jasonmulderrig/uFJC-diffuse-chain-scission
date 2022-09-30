# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import uFJCDiffuseChainScissionProblem, GeneralizeduFJCNetwork, gmsh_mesher, mesh_topologier, latex_formatting_figure, save_current_figure
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Problem
class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentTest(uFJCDiffuseChainScissionProblem):

    def __init__(self, L, H, elem_size=0.01):

        self.L = L
        self.H = H
        self.elem_size = elem_size
        
        uFJCDiffuseChainScissionProblem.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """

        p = self.parameters

        x_centerpoint = float(self.L/2)
        y_centerpoint = float(self.H/2)

        centerpoint = (x_centerpoint, y_centerpoint)

        x_centerpoint_string = "{:.1f}".format(x_centerpoint)
        y_centerpoint_string = "{:.1f}".format(y_centerpoint)

        centerpoint_label = '('+x_centerpoint_string+', '+y_centerpoint_string+')'

        p.geometry.meshpoints            = [centerpoint]
        p.geometry.meshpoints_label_list = [r'$'+centerpoint_label+'$']
        p.geometry.meshpoints_color_list = ['black']
        p.geometry.meshpoints_name_list  = ['centerpoint']

        p.material.macro2micro_deformation_assumption = "nonaffine"

        # Parameters used in F_func
        strain_rate = 1 # 1/sec
        max_strain = p.deformation.t_max*strain_rate/2

        p.deformation.strain_rate = strain_rate
        p.deformation.max_strain  = max_strain

        # Deformation stepping calculations
        max_F_dot = strain_rate
        t_scale = 1./max_F_dot # sec
        t_step_modify_factor = 1e-2
        t_step = np.around(t_step_modify_factor*t_scale, 2) # sec
        t_step_chunk_modify_factor = 5
        t_step_chunk = t_step_chunk_modify_factor*t_step # sec
        
        p.deformation.max_F_dot                  = max_F_dot
        p.deformation.t_step_modify_factor       = t_step_modify_factor
        p.deformation.t_step                     = t_step
        p.deformation.t_step_chunk_modify_factor = t_step_chunk_modify_factor
        p.deformation.t_step_chunk               = t_step_chunk

        p.post_processing.save_lmbda_c_eq_chunks   = True
        p.post_processing.save_lmbda_nu_chunks     = True
        p.post_processing.save_lmbda_nu_max_chunks = True
        p.post_processing.save_upsilon_c_chunks    = True
        p.post_processing.save_d_c_chunks          = True
    
    def prefix(self):
        mp = self.parameters.material
        return mp.physical_dimensionality+"_"+mp.two_dimensional_formulation+"_"+mp.incompressibility_assumption+"_"+mp.macro2micro_deformation_assumption+"_"+mp.micro2macro_homogenization_scheme+"_"+mp.chain_level_load_sharing+"_"+mp.rate_dependence
    
    def define_mesh(self):
        """
        Define the mesh for the problem
        """
        
        # use obsolete version of string formatting here because brackets are essential for use in the gmsh script
        geofile = \
            """
            elem_size = DefineNumber[ %g, Name "Parameters/elem_size" ];
            L         = DefineNumber[ %g, Name "Parameters/L"];
            H         = DefineNumber[ %g, Name "Parameters/H"];
            Point(1) = {0, 0, 0, elem_size};
            Point(2) = {L, 0, 0, elem_size};
            Point(3) = {L, H, 0, elem_size};
            Point(4) = {0, H, 0, elem_size};
            Line(1) = {1, 2}; 
            Line(2) = {2, 3}; 
            Line(3) = {3, 4}; 
            Line(4) = {4, 1};
            Line Loop(1) = {1, 2, 3, 4};
            Plane Surface(1) = {1};
            Transfinite Surface{1} AlternateRight;
            Mesh.MshFileVersion = 2.0;
            """ % (self.elem_size, self.L, self.H)

        L_string         = "{:.1f}".format(self.L)
        H_string         = "{:.1f}".format(self.H)
        elem_size_string = "{:.3f}".format(self.elem_size)

        mesh_type = "gmsh_two_dimensional_rectangle"
        mesh_name = mesh_type+"_"+L_string+"_"+H_string+"_"+elem_size_string

        return gmsh_mesher(geofile, self.prefix(), mesh_name)
    
    def F_func(self, t):
        """
        Function defining the deformation
        """
        dp = self.parameters.deformation

        return 1 + dp.strain_rate*(t-dp.t_min)*np.heaviside(dp.max_strain-dp.strain_rate*(t-dp.t_min), 0.5) \
            + (2*dp.max_strain - dp.strain_rate*(t-dp.t_min))*np.heaviside(dp.strain_rate*(t-dp.t_min)-dp.max_strain, 0.5)
    
    def initialize_lmbda(self):
        lmbda_1        = [] # unitless
        lmbda_1_chunks = [] # unitless

        return lmbda_1, lmbda_1_chunks
    
    def store_initialized_lmbda(self, lmbda):
        lmbda_1_val = 1 # assuming no pre-stretching
        
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]
        
        lmbda_1.append(lmbda_1_val)
        lmbda_1_chunks.append(lmbda_1_val)
        
        return lmbda_1, lmbda_1_chunks
    
    def calculate_lmbda_func(self, t_val):
        lmbda_1_val = self.F_func(t_val)

        return lmbda_1_val
    
    def store_calculated_lmbda(self, lmbda, lmbda_val):
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]
        lmbda_1_val    = lmbda_val
        
        lmbda_1.append(lmbda_1_val)
        
        return lmbda_1, lmbda_1_chunks
    
    def store_calculated_lmbda_chunk_post_processing(self, lmbda, lmbda_val):
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]
        lmbda_1_val    = lmbda_val
        
        lmbda_1_chunks.append(lmbda_1_val)
        
        return lmbda_1, lmbda_1_chunks
    
    def calculate_u_func(self, lmbda):
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]

        u_1        = [lmbda_1_val-1 for lmbda_1_val in lmbda_1]
        u_1_chunks = [lmbda_1_chunks_val-1 for lmbda_1_chunks_val in lmbda_1_chunks]

        return u_1, u_1_chunks
    
    def save2deformation(self, deformation, lmbda, u):
        lmbda_1        = lmbda[0]
        lmbda_1_chunks = lmbda[1]

        u_1        = u[0]
        u_1_chunks = u[1]

        deformation.lmbda_1        = lmbda_1
        deformation.lmbda_1_chunks = lmbda_1_chunks
        deformation.u_1            = u_1
        deformation.u_1_chunks     = u_1_chunks

        return deformation
    
    def strong_form_initialize_sigma_chunks(self, chunks):
        sigma_11_chunks = [] # unitless
        chunks.sigma_11_chunks = sigma_11_chunks

        return chunks
    
    def lr_cg_deformation_gradient_func(self, deformation):
        lmbda_1_val = deformation.lmbda_1[deformation.t_indx]
        lmbda_2_val = 1./lmbda_1_val
        F_val = np.diagflat([lmbda_1_val, lmbda_2_val])
        C_val = np.einsum('jJ,jK->JK', F_val, F_val)
        b_val = np.einsum('jJ,kJ->jk', F_val, F_val)

        return F_val, C_val, b_val
    
    def strong_form_calculate_sigma_func(self, sigma_hyp_val, deformation):
        dp = self.parameters.deformation

        F_val, C_val, b_val = self.lr_cg_deformation_gradient_func(deformation)
        
        sigma_11_val = sigma_hyp_val*b_val[0][0] + dp.K_G*( np.linalg.det(F_val) - 1. )

        return sigma_11_val
    
    def strong_form_store_calculated_sigma_chunks(self, sigma_val, chunks):
        sigma_11_val = sigma_val
        chunks.sigma_11_chunks.append(sigma_11_val)

        return chunks
    
    def weak_form_initialize_deformation_sigma_chunks(self, meshpoints, chunks):
        chunks.F_11_chunks         = []
        chunks.F_11_chunks_val     = [0. for meshpoint_indx in range(len(meshpoints))]
        chunks.sigma_11_chunks     = []
        chunks.sigma_11_chunks_val = [0. for meshpoint_indx in range(len(meshpoints))]

        return chunks
    
    def weak_form_store_calculated_sigma_chunks(self, sigma_val, tensor2vector_indx_dict, meshpoints, chunks):
        for meshpoint_indx in range(len(meshpoints)):
            chunks.sigma_11_chunks_val[meshpoint_indx] = sigma_val(meshpoints[meshpoint_indx])[tensor2vector_indx_dict["11"]]
        chunks.sigma_11_chunks.append(deepcopy(chunks.sigma_11_chunks_val))

        return chunks
    
    def weak_form_store_calculated_deformation_chunks(self, F_val, tensor2vector_indx_dict, meshpoints, chunks):
        for meshpoint_indx in range(len(meshpoints)):
            chunks.F_11_chunks_val[meshpoint_indx] = F_val(meshpoints[meshpoint_indx])[tensor2vector_indx_dict["11"]]
        chunks.F_11_chunks.append(deepcopy(chunks.F_11_chunks_val))

        return chunks

    def define_material(self):
        """
        Return material that will be set in the model
        """
        material = GeneralizeduFJCNetwork(self.parameters, self.strong_form_initialize_sigma_chunks, self.lr_cg_deformation_gradient_func, self.strong_form_calculate_sigma_func, self.strong_form_store_calculated_sigma_chunks, self.weak_form_initialize_deformation_sigma_chunks, self.weak_form_store_calculated_sigma_chunks, self.weak_form_store_calculated_deformation_chunks)
        
        return material
    
    def define_bc_u(self):
        """
        Return a list of displacement-controlled (Dirichlet) boundary conditions
        """
        self.fem.lines = MeshFunction("size_t", self.fem.mesh, self.fem.mesh.topology().dim()-1)
        self.fem.lines.set_all(0)
        self.fem.points = MeshFunction("size_t", self.fem.mesh, self.fem.mesh.topology().dim()-2)
        self.fem.points.set_all(0)

        L = self.L

        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0.)

        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], L)
        
        class OriginPoint(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0.) and near(x[1], 0.)

        LeftBoundary().mark(self.fem.lines, 1)
        RightBoundary().mark(self.fem.lines, 2)
        OriginPoint().mark(self.fem.points, 1)

        mesh_topologier(self.fem.lines, self.prefix(), "lines")
        mesh_topologier(self.fem.points, self.prefix(), "points")

        self.fem.u_x_expression = Expression("u_x", u_x=0., degree=0)

        bc_I   = DirichletBC(self.fem.V_u.sub(0), Constant(0.), LeftBoundary())
        bc_II  = DirichletBC(self.fem.V_u.sub(0), self.fem.u_x_expression, RightBoundary())
        bc_III = DirichletBC(self.fem.V_u.sub(1), Constant(0.), OriginPoint(), method='pointwise')

        return [bc_I, bc_II, bc_III]
    
    def set_loading(self):
        """
        Update Dirichlet boundary conditions"
        """
        self.fem.u_x_expression.u_x = self.L*self.deformation.u_1[self.deformation.t_indx]

    def set_homogeneous_strong_form_deformation_finalization(self):
        """
        Plot the chunked results from the homogeneous strong form deformation
        """
        
        mp                 = self.parameters.material
        ppp                = self.parameters.post_processing
        deformation        = self.deformation
        strong_form_chunks = self.strong_form_chunks
        
        # plot results
        latex_formatting_figure(ppp)
        
        # lmbda_c
        fig = plt.figure()
        plt.plot(deformation.t_chunks, strong_form_chunks.lmbda_c_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-lmbda_c")

        fig = plt.figure()
        plt.plot(deformation.lmbda_1_chunks, strong_form_chunks.lmbda_c_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\lambda_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-lmbda_c")

        # lmbda_c_eq
        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            lmbda_c_eq___nu_chunk = [lmbda_c_eq_chunk[nu_chunk_indx] for lmbda_c_eq_chunk in strong_form_chunks.lmbda_c_eq_chunks]
            plt.plot(deformation.t_chunks, lmbda_c_eq___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_c^{eq}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-lmbda_c_eq")

        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            lmbda_c_eq___nu_chunk = [lmbda_c_eq_chunk[nu_chunk_indx] for lmbda_c_eq_chunk in strong_form_chunks.lmbda_c_eq_chunks]
            plt.plot(deformation.lmbda_1_chunks, lmbda_c_eq___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\lambda_c^{eq}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-lmbda_c_eq")

        # lmbda_nu
        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            lmbda_nu___nu_chunk = [lmbda_nu_chunk[nu_chunk_indx] for lmbda_nu_chunk in strong_form_chunks.lmbda_nu_chunks]
            plt.plot(deformation.t_chunks, lmbda_nu___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_{\nu}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-lmbda_nu")

        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            lmbda_nu___nu_chunk = [lmbda_nu_chunk[nu_chunk_indx] for lmbda_nu_chunk in strong_form_chunks.lmbda_nu_chunks]
            plt.plot(deformation.lmbda_1_chunks, lmbda_nu___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\lambda_{\nu}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-lmbda_nu")
        
        # lmbda_nu_max
        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            lmbda_nu_max___nu_chunk = [lmbda_nu_max_chunk[nu_chunk_indx] for lmbda_nu_max_chunk in strong_form_chunks.lmbda_nu_max_chunks]
            plt.plot(deformation.t_chunks, lmbda_nu_max___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_{\nu}^{max}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-lmbda_nu_max")

        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            lmbda_nu_max___nu_chunk = [lmbda_nu_max_chunk[nu_chunk_indx] for lmbda_nu_max_chunk in strong_form_chunks.lmbda_nu_max_chunks]
            plt.plot(deformation.lmbda_1_chunks, lmbda_nu_max___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\lambda_{\nu}^{max}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-lmbda_nu_max")

        # upsilon_c
        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            upsilon_c___nu_chunk = [upsilon_c_chunk[nu_chunk_indx] for upsilon_c_chunk in strong_form_chunks.upsilon_c_chunks]
            plt.plot(deformation.t_chunks, upsilon_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\upsilon_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-upsilon_c")

        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            upsilon_c___nu_chunk = [upsilon_c_chunk[nu_chunk_indx] for upsilon_c_chunk in strong_form_chunks.upsilon_c_chunks]
            plt.plot(deformation.lmbda_1_chunks, upsilon_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\upsilon_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-upsilon_c")

        # Upsilon_c
        fig = plt.figure()
        plt.plot(deformation.t_chunks, strong_form_chunks.Upsilon_c_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\Upsilon_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-Upsilon_c")

        fig = plt.figure()
        plt.plot(deformation.lmbda_1_chunks, strong_form_chunks.Upsilon_c_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\Upsilon_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-Upsilon_c")
        
        # d_c
        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            d_c___nu_chunk = [d_c_chunk[nu_chunk_indx] for d_c_chunk in strong_form_chunks.d_c_chunks]
            plt.plot(deformation.t_chunks, d_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$d_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-d_c")

        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            d_c___nu_chunk = [d_c_chunk[nu_chunk_indx] for d_c_chunk in strong_form_chunks.d_c_chunks]
            plt.plot(deformation.lmbda_1_chunks, d_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$d_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-d_c")
        
        # D_c
        fig = plt.figure()
        plt.plot(deformation.t_chunks, strong_form_chunks.D_c_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$D_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-D_c")

        fig = plt.figure()
        plt.plot(deformation.lmbda_1_chunks, strong_form_chunks.D_c_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$D_c$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-D_c")

        # epsilon_cnu_diss_hat
        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            epsilon_cnu_diss_hat___nu_chunk = [epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for epsilon_cnu_diss_hat_chunk in strong_form_chunks.epsilon_cnu_diss_hat_chunks]
            plt.plot(deformation.t_chunks, epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\hat{\varepsilon}_{c\nu}^{diss}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-epsilon_cnu_diss_hat")

        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            epsilon_cnu_diss_hat___nu_chunk = [epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for epsilon_cnu_diss_hat_chunk in strong_form_chunks.epsilon_cnu_diss_hat_chunks]
            plt.plot(deformation.lmbda_1_chunks, epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\hat{\varepsilon}_{c\nu}^{diss}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-epsilon_cnu_diss_hat")

        # Epsilon_cnu_diss_hat
        fig = plt.figure()
        plt.plot(deformation.t_chunks, strong_form_chunks.Epsilon_cnu_diss_hat_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\hat{E}_{c\nu}^{diss}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-Epsilon_cnu_diss_hat")

        fig = plt.figure()
        plt.plot(deformation.lmbda_1_chunks, strong_form_chunks.Epsilon_cnu_diss_hat_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\hat{E}_{c\nu}^{diss}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-Epsilon_cnu_diss_hat")

        # epsilon_c_diss_hat
        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            epsilon_c_diss_hat___nu_chunk = [epsilon_c_diss_hat_chunk[nu_chunk_indx] for epsilon_c_diss_hat_chunk in strong_form_chunks.epsilon_c_diss_hat_chunks]
            plt.plot(deformation.t_chunks, epsilon_c_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\hat{\varepsilon}_c^{diss}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-epsilon_c_diss_hat")

        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            epsilon_c_diss_hat___nu_chunk = [epsilon_c_diss_hat_chunk[nu_chunk_indx] for epsilon_c_diss_hat_chunk in strong_form_chunks.epsilon_c_diss_hat_chunks]
            plt.plot(deformation.lmbda_1_chunks, epsilon_c_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\hat{\varepsilon}_c^{diss}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-epsilon_c_diss_hat")
        
        # Epsilon_c_diss_hat
        fig = plt.figure()
        plt.plot(deformation.t_chunks, strong_form_chunks.Epsilon_c_diss_hat_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\hat{E}_c^{diss}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-Epsilon_c_diss_hat")

        fig = plt.figure()
        plt.plot(deformation.lmbda_1_chunks, strong_form_chunks.Epsilon_c_diss_hat_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\hat{E}_c^{diss}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-Epsilon_c_diss_hat")

        # overline_epsilon_cnu_diss_hat
        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            overline_epsilon_cnu_diss_hat___nu_chunk = [overline_epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_cnu_diss_hat_chunk in strong_form_chunks.overline_epsilon_cnu_diss_hat_chunks]
            plt.plot(deformation.t_chunks, overline_epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-overline_epsilon_cnu_diss_hat")

        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            overline_epsilon_cnu_diss_hat___nu_chunk = [overline_epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_cnu_diss_hat_chunk in strong_form_chunks.overline_epsilon_cnu_diss_hat_chunks]
            plt.plot(deformation.lmbda_1_chunks, overline_epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-overline_epsilon_cnu_diss_hat")

        # overline_Epsilon_cnu_diss_hat
        fig = plt.figure()
        plt.plot(deformation.t_chunks, strong_form_chunks.overline_Epsilon_cnu_diss_hat_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{E}_{c\nu}^{diss}}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-overline_Epsilon_cnu_diss_hat")

        fig = plt.figure()
        plt.plot(deformation.lmbda_1_chunks, strong_form_chunks.overline_Epsilon_cnu_diss_hat_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\overline{\hat{E}_{c\nu}^{diss}}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-overline_Epsilon_cnu_diss_hat")

        # overline_epsilon_c_diss_hat
        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            overline_epsilon_c_diss_hat___nu_chunk = [overline_epsilon_c_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_c_diss_hat_chunk in strong_form_chunks.overline_epsilon_c_diss_hat_chunks]
            plt.plot(deformation.t_chunks, overline_epsilon_c_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{\varepsilon}_c^{diss}}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-overline_epsilon_c_diss_hat")

        fig = plt.figure()
        for nu_chunk_indx in range(len(mp.nu_chunks_list)):
            overline_epsilon_c_diss_hat___nu_chunk = [overline_epsilon_c_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_c_diss_hat_chunk in strong_form_chunks.overline_epsilon_c_diss_hat_chunks]
            plt.plot(deformation.lmbda_1_chunks, overline_epsilon_c_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\overline{\hat{\varepsilon}_c^{diss}}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-overline_epsilon_c_diss_hat")
        
        # overline_Epsilon_c_diss_hat
        fig = plt.figure()
        plt.plot(deformation.t_chunks, strong_form_chunks.overline_Epsilon_c_diss_hat_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{E}_c^{diss}}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-overline_Epsilon_c_diss_hat")

        fig = plt.figure()
        plt.plot(deformation.lmbda_1_chunks, strong_form_chunks.overline_Epsilon_c_diss_hat_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\overline{\hat{E}_c^{diss}}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-overline_Epsilon_c_diss_hat")
        
        # sigma_11
        fig = plt.figure()
        plt.plot(deformation.t_chunks, strong_form_chunks.sigma_11_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$t$', 30, r'$\sigma_{11}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-t-vs-sigma_11")

        fig = plt.figure()
        plt.plot(deformation.lmbda_1_chunks, strong_form_chunks.sigma_11_chunks, linestyle='-', color='black', alpha=1, linewidth=2.5)
        plt.grid(True, alpha=0.25)
        save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\sigma_{11}$', 30, "homogeneous-strong-form-uniaxial-rate-independent-lmbda_1-vs-sigma_11")
    
    def set_fenics_weak_form_deformation_finalization(self):
        """
        Plot the chunked results from the weak form deformation in FEniCS
        """

        gp               = self.parameters.geometry
        mp               = self.parameters.material
        dp               = self.parameters.deformation
        ppp              = self.parameters.post_processing
        deformation      = self.deformation
        weak_form_chunks = self.weak_form_chunks

        # plot results
        latex_formatting_figure(ppp)

        # lmbda_c
        if ppp.save_lmbda_c_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                lmbda_c___meshpoint_chunk = [lmbda_c_chunk[meshpoint_indx] for lmbda_c_chunk in weak_form_chunks.lmbda_c_chunks]
                plt.plot(deformation.t_chunks, lmbda_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_c")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                lmbda_c___meshpoint_chunk = [lmbda_c_chunk[meshpoint_indx] for lmbda_c_chunk in weak_form_chunks.lmbda_c_chunks]
                plt.plot(deformation.lmbda_1_chunks, lmbda_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\lambda_c$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-lmbda_c")

        # lmbda_c_eq
        if ppp.save_lmbda_c_eq_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_c_eq___meshpoint_chunk = [lmbda_c_eq_chunk[meshpoint_indx] for lmbda_c_eq_chunk in weak_form_chunks.lmbda_c_eq_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_c_eq___nu_chunk = [lmbda_c_eq_chunk[nu_chunk_indx] for lmbda_c_eq_chunk in lmbda_c_eq___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, lmbda_c_eq___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_c^{eq}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_c_eq"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_c_eq___meshpoint_chunk = [lmbda_c_eq_chunk[meshpoint_indx] for lmbda_c_eq_chunk in weak_form_chunks.lmbda_c_eq_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_c_eq___nu_chunk = [lmbda_c_eq_chunk[nu_chunk_indx] for lmbda_c_eq_chunk in lmbda_c_eq___meshpoint_chunk]
                    plt.plot(deformation.lmbda_1_chunks, lmbda_c_eq___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\lambda_c^{eq}$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-lmbda_c_eq"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_nu
        if ppp.save_lmbda_nu_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu___meshpoint_chunk = [lmbda_nu_chunk[meshpoint_indx] for lmbda_nu_chunk in weak_form_chunks.lmbda_nu_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu___nu_chunk = [lmbda_nu_chunk[nu_chunk_indx] for lmbda_nu_chunk in lmbda_nu___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, lmbda_nu___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_{\nu}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_nu"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu___meshpoint_chunk = [lmbda_nu_chunk[meshpoint_indx] for lmbda_nu_chunk in weak_form_chunks.lmbda_nu_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu___nu_chunk = [lmbda_nu_chunk[nu_chunk_indx] for lmbda_nu_chunk in lmbda_nu___meshpoint_chunk]
                    plt.plot(deformation.lmbda_1_chunks, lmbda_nu___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\lambda_{\nu}$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-lmbda_nu"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_nu_max
        if ppp.save_lmbda_nu_max_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu_max___meshpoint_chunk = [lmbda_nu_max_chunk[meshpoint_indx] for lmbda_nu_max_chunk in weak_form_chunks.lmbda_nu_max_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu_max___nu_chunk = [lmbda_nu_max_chunk[nu_chunk_indx] for lmbda_nu_max_chunk in lmbda_nu_max___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, lmbda_nu_max___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_{\nu}^{max}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_nu_max"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu_max___meshpoint_chunk = [lmbda_nu_max_chunk[meshpoint_indx] for lmbda_nu_max_chunk in weak_form_chunks.lmbda_nu_max_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu_max___nu_chunk = [lmbda_nu_max_chunk[nu_chunk_indx] for lmbda_nu_max_chunk in lmbda_nu_max___meshpoint_chunk]
                    plt.plot(deformation.lmbda_1_chunks, lmbda_nu_max___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\lambda_{\nu}^{max}$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-lmbda_nu_max"+"_"+gp.meshpoints_name_list[meshpoint_indx])

        # upsilon_c
        if ppp.save_upsilon_c_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                upsilon_c___meshpoint_chunk = [upsilon_c_chunk[meshpoint_indx] for upsilon_c_chunk in weak_form_chunks.upsilon_c_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    upsilon_c___nu_chunk = [upsilon_c_chunk[nu_chunk_indx] for upsilon_c_chunk in upsilon_c___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, upsilon_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\upsilon_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-upsilon_c"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                upsilon_c___meshpoint_chunk = [upsilon_c_chunk[meshpoint_indx] for upsilon_c_chunk in weak_form_chunks.upsilon_c_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    upsilon_c___nu_chunk = [upsilon_c_chunk[nu_chunk_indx] for upsilon_c_chunk in upsilon_c___meshpoint_chunk]
                    plt.plot(deformation.lmbda_1_chunks, upsilon_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\upsilon_c$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-upsilon_c"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # Upsilon_c
        if ppp.save_Upsilon_c_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                Upsilon_c___meshpoint_chunk = [Upsilon_c_chunk[meshpoint_indx] for Upsilon_c_chunk in weak_form_chunks.Upsilon_c_chunks]
                plt.plot(deformation.t_chunks, Upsilon_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\Upsilon_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-Upsilon_c")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                Upsilon_c___meshpoint_chunk = [Upsilon_c_chunk[meshpoint_indx] for Upsilon_c_chunk in weak_form_chunks.Upsilon_c_chunks]
                plt.plot(deformation.lmbda_1_chunks, Upsilon_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\Upsilon_c$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-Upsilon_c")

        # d_c
        if ppp.save_d_c_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                d_c___meshpoint_chunk = [d_c_chunk[meshpoint_indx] for d_c_chunk in weak_form_chunks.d_c_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    d_c___nu_chunk = [d_c_chunk[nu_chunk_indx] for d_c_chunk in d_c___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, d_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$d_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-d_c"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                d_c___meshpoint_chunk = [d_c_chunk[meshpoint_indx] for d_c_chunk in weak_form_chunks.d_c_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    d_c___nu_chunk = [d_c_chunk[nu_chunk_indx] for d_c_chunk in d_c___meshpoint_chunk]
                    plt.plot(deformation.lmbda_1_chunks, d_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$d_c$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-d_c"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # D_c
        if ppp.save_D_c_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                D_c___meshpoint_chunk = [D_c_chunk[meshpoint_indx] for D_c_chunk in weak_form_chunks.D_c_chunks]
                plt.plot(deformation.t_chunks, D_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$D_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-D_c")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                D_c___meshpoint_chunk = [D_c_chunk[meshpoint_indx] for D_c_chunk in weak_form_chunks.D_c_chunks]
                plt.plot(deformation.lmbda_1_chunks, D_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$D_c$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-D_c")
        
        # sigma
        if ppp.save_sigma_chunks:
            if dp.deformation_type == 'uniaxial':
                fig = plt.figure()
                for meshpoint_indx in range(len(gp.meshpoints)):
                    sigma_11___meshpoint_chunk = [sigma_11_chunk[meshpoint_indx] for sigma_11_chunk in weak_form_chunks.sigma_11_chunks]
                    plt.plot(deformation.t_chunks, sigma_11___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\sigma_{11}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-sigma_11")

                fig = plt.figure()
                for meshpoint_indx in range(len(gp.meshpoints)):
                    sigma_11___meshpoint_chunk = [sigma_11_chunk[meshpoint_indx] for sigma_11_chunk in weak_form_chunks.sigma_11_chunks]
                    plt.plot(deformation.lmbda_1_chunks, sigma_11___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$\sigma_{11}$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-sigma_11")
            
            elif dp.deformation_type == 'equibiaxial': pass
            elif dp.deformation_type == 'simple_shear': pass

        # F
        if ppp.save_F_chunks:
            if dp.deformation_type == 'uniaxial':
                fig = plt.figure()
                for meshpoint_indx in range(len(gp.meshpoints)):
                    F_11___meshpoint_chunk = [F_11_chunk[meshpoint_indx] for F_11_chunk in weak_form_chunks.F_11_chunks]
                    plt.plot(deformation.t_chunks, F_11___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$F_{11}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-F_11")

                fig = plt.figure()
                for meshpoint_indx in range(len(gp.meshpoints)):
                    F_11___meshpoint_chunk = [F_11_chunk[meshpoint_indx] for F_11_chunk in weak_form_chunks.F_11_chunks]
                    plt.plot(deformation.lmbda_1_chunks, F_11___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$\lambda_1$', 30, r'$F_{11}$', 30, "fenics-weak-form-uniaxial-rate-independent-lmbda_1-vs-F_11")
            
            elif dp.deformation_type == 'equibiaxial': pass
            elif dp.deformation_type == 'simple_shear': pass

if __name__ == '__main__':

    L, H, N = 1.0, 1.0, 30
    elem_size = float(L/N)
    problem = TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentTest(L, H, elem_size)
    problem.solve_homogeneous_strong_form_deformation()
    problem.solve_fenics_weak_form_deformation()