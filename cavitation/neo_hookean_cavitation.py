# import necessary libraries
from __future__ import division
from dolfin import *
from ufjc_diffuse_chain_scission import uFJCDiffuseChainScissionProblem, PhenomenologicalNetwork, gmsh_mesher, mesh_topologier, latex_formatting_figure, save_current_figure
import numpy as np
import matplotlib.pyplot as plt

# Problem
class neoHookeanCavitation(uFJCDiffuseChainScissionProblem):

    def __init__(self, L, H, r_cavity, near_cavity_elem_size=0.001, far_cavity_elem_size=0.1):

        self.L = L
        self.H = H
        self.r_cavity = r_cavity
        self.near_cavity_elem_size = near_cavity_elem_size
        self.far_cavity_elem_size  = far_cavity_elem_size
        
        uFJCDiffuseChainScissionProblem.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """

        p = self.parameters

        x_cavity_surface_point = 0.75*self.r_cavity # np.cos(np.pi/4)*self.r_cavity
        y_cavity_surface_point = 0.75*self.r_cavity # np.sin(np.pi/4)*self.r_cavity

        cavity_surface_point = (x_cavity_surface_point,y_cavity_surface_point)

        x_cavity_surface_point_string = "{:.4f}".format(x_cavity_surface_point)
        y_cavity_surface_point_string = "{:.4f}".format(y_cavity_surface_point)

        cavity_surface_point_label = '('+x_cavity_surface_point_string+', '+y_cavity_surface_point_string+')'

        p.geometry.meshpoints            = [cavity_surface_point]
        p.geometry.meshpoints_label_list = [r'$'+cavity_surface_point_label+'$']
        p.geometry.meshpoints_color_list = ['black']
        p.geometry.meshpoints_name_list  = ['cavity_surface_point']

        p.material.network_model = "phenomenological_model"

        # Parameters used in F_func
        r_strain_rate = 0.0025 # 1/sec
        t_max = 100 # sec

        p.deformation.r_strain_rate = r_strain_rate
        p.deformation.t_max = t_max

        # Deformation stepping calculations
        max_F_dot = r_strain_rate
        t_scale = 1./max_F_dot # sec
        t_step_modify_factor = 1e-3
        t_step = np.around(t_step_modify_factor*t_scale, 2) # sec
        t_step_chunk_modify_factor = 1
        t_step_chunk = t_step_chunk_modify_factor*t_step # sec
        
        p.deformation.max_F_dot                  = max_F_dot
        p.deformation.t_step_modify_factor       = t_step_modify_factor
        p.deformation.t_step                     = t_step
        p.deformation.t_step_chunk_modify_factor = t_step_chunk_modify_factor
        p.deformation.t_step_chunk               = t_step_chunk

        p.post_processing.save_lmbda_c_mesh     = False
        p.post_processing.save_lmbda_c_chunks   = False
        p.post_processing.save_Upsilon_c_mesh   = False
        p.post_processing.save_Upsilon_c_chunks = False
        p.post_processing.save_D_c_mesh         = False
        p.post_processing.save_D_c_chunks       = False
    
    def prefix(self):
        mp = self.parameters.material
        return mp.phenomenological_model
    
    def F_func(self, t):
        """
        Function defining the deformation
        """
        dp = self.parameters.deformation

        return dp.r_strain_rate*t
    
    def define_mesh(self):
        """
        Define the mesh for the problem
        """
        # use obsolete version of string formatting here because brackets are essential for use in the gmsh script
        geofile = \
            """
            Mesh.Algorithm = 8;
            near_cavity_elem_size = DefineNumber[ %g, Name "Parameters/near_cavity_elem_size" ];
            far_cavity_elem_size  = DefineNumber[ %g, Name "Parameters/far_cavity_elem_size" ];
            r_cavity  = DefineNumber[ %g, Name "Parameters/r_cavity" ];
            L         = DefineNumber[ %g, Name "Parameters/L"];
            H         = DefineNumber[ %g, Name "Parameters/H"];
            Point(1) = {0, 0, 0, near_cavity_elem_size};
            Point(2) = {r_cavity, 0, 0, near_cavity_elem_size};
            Point(3) = {2*r_cavity, 0, 0, near_cavity_elem_size};
            Point(4) = {L, 0, 0, far_cavity_elem_size};
            Point(5) = {L, H, 0, far_cavity_elem_size};
            Point(6) = {0, H, 0, far_cavity_elem_size};
            Point(7) = {0, 2*r_cavity, 0, near_cavity_elem_size};
            Point(8) = {0, r_cavity, 0, near_cavity_elem_size};
            Line(1) = {2, 3}; 
            Line(2) = {3, 4}; 
            Line(3) = {4, 5};
            Line(4) = {5, 6};
            Line(5) = {6, 7};
            Line(6) = {7, 8};
            Circle(7) = {8, 1, 2};
            Circle(8) = {7, 1, 3};
            Curve Loop(21) = {1, -8, 6, 7};
            Curve Loop(22) = {2, 3, 4, 5, 8};
            Plane Surface(31) = {21};
            Plane Surface(32) = {-21, -22};
            Mesh.MshFileVersion = 2.0;
            """ % (self.near_cavity_elem_size, self.far_cavity_elem_size, self.r_cavity, self.L, self.H)

        L_string              = "{:.1f}".format(self.L)
        H_string              = "{:.1f}".format(self.H)
        r_cavity_string       = "{:.1f}".format(self.r_cavity)
        near_cavity_elem_size = "{:.3f}".format(self.near_cavity_elem_size)
        far_cavity_elem_size  = "{:.1f}".format(self.far_cavity_elem_size)

        mesh_type = "2D_plane_strain_cavity"
        mesh_name = mesh_type+"_"+L_string+"_"+H_string+"_"+r_cavity_string+"_"+near_cavity_elem_size+"_"+far_cavity_elem_size

        return gmsh_mesher(geofile, self.prefix(), mesh_name)
    
    def define_material(self):
        """
        Return material that will be set in the model
        """
        material = PhenomenologicalNetwork(self.parameters)
        
        return material
    
    def define_bc_u(self):
        """
        Return a list of displacement-controlled (Dirichlet) boundary conditions
        """
        self.fem.lines = MeshFunction("size_t", self.fem.mesh, self.fem.mesh.topology().dim()-1)
        self.fem.lines.set_all(0)
        self.fem.points = MeshFunction("size_t", self.fem.mesh, self.fem.mesh.topology().dim()-2)
        self.fem.points.set_all(0)

        L        = self.L
        H        = self.H
        r_cavity = self.r_cavity

        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0., DOLFIN_EPS)

        class BottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], 0., DOLFIN_EPS)

        class Cavity(SubDomain):
            def inside(self, x, on_boundary):
                r_cavity_sq = x[0]**2 + x[1]**2
                return r_cavity_sq <= (r_cavity + DOLFIN_EPS)**2

        class PinPoint(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], L, DOLFIN_EPS) and near(x[1], H, DOLFIN_EPS)

        LeftBoundary().mark(self.fem.lines, 1)
        BottomBoundary().mark(self.fem.lines, 2)
        Cavity().mark(self.fem.lines, 3)
        PinPoint().mark(self.fem.points, 1)

        mesh_topologier(self.fem.lines, self.prefix(), "lines")
        mesh_topologier(self.fem.points, self.prefix(), "points")

        self.fem.u_r_expression = Expression(["x[0]*u_r/r_0", "x[1]*u_r/r_0"], r_0=r_cavity, u_r=0., degree=1)

        bc_I   = DirichletBC(self.fem.V_u.sub(0), Constant(0.), LeftBoundary())
        bc_II  = DirichletBC(self.fem.V_u.sub(1), Constant(0.), BottomBoundary())
        bc_III = DirichletBC(self.fem.V_u, self.fem.u_r_expression, Cavity())
        bc_IV  = DirichletBC(self.fem.V_u, Constant((0., 0.)), PinPoint(), method='pointwise')

        return [bc_I, bc_II, bc_III, bc_IV]
    
    def set_loading(self):
        """
        Update Dirichlet boundary conditions"
        """
        self.fem.u_r_expression.u_r = self.F_func(self.deformation.t[self.deformation.t_indx])

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
        
        deformation.u_r_chunks = deformation.lmbda_1_chunks

        # plot results
        latex_formatting_figure(ppp)
        
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
                    plt.plot(deformation.u_r_chunks, sigma_11___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_r$', 30, r'$\sigma_{11}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_r-vs-sigma_11")
            
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
                    plt.plot(deformation.u_r_chunks, F_11___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_r$', 30, r'$F_{11}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_r-vs-F_11")
            
            elif dp.deformation_type == 'equibiaxial': pass
            elif dp.deformation_type == 'simple_shear': pass

if __name__ == '__main__':

    L, H = 1.0, 1.0
    r_cavity = 0.05
    near_cavity_elem_size = 0.001
    far_cavity_elem_size  = 0.2
    problem = neoHookeanCavitation(L, H, r_cavity, near_cavity_elem_size, far_cavity_elem_size)
    problem.solve_fenics_weak_form_deformation()