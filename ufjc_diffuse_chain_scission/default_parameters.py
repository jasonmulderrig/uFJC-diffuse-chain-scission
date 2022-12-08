# import necessary libraries
from types import SimpleNamespace
from dolfin import *
import numpy as np
from scipy import constants

def default_parameters():

    parameters = SimpleNamespace()
    subset_list = ["characterizer", "pre_processing", "problem", "geometry", "material", "fem", "deformation", "solver_u", "solver_u_settings", "post_processing"]
    for subparset in subset_list:
        subparset_is = eval("default_"+subparset+"_parameters()")
        setattr(parameters, subparset, subparset_is)
    return parameters

def default_characterizer_parameters():

    characterizer = SimpleNamespace()

    nu_single_chain_list       = [int(5**( 2*i - 1 )) for i in range(1, 4)] # 5, 125, 3125
    nu_label_single_chain_list = [r'$\nu='+str(nu_single_chain_list[i])+'$' for i in range(len(nu_single_chain_list))]

    characterizer.nu_single_chain_list       = nu_single_chain_list
    characterizer.nu_label_single_chain_list = nu_label_single_chain_list

    zeta_nu_char_single_chain_list       = [10, 50, 100, 500, 1000]
    zeta_nu_char_single_chain_list       = zeta_nu_char_single_chain_list[::-1] # Reverse the order of the zeta_nu_char list
    zeta_nu_char_label_single_chain_list = [r'$\zeta_{\nu}^{char}='+str(zeta_nu_char_single_chain_list[i])+'$' for i in range(len(zeta_nu_char_single_chain_list))]

    characterizer.zeta_nu_char_single_chain_list       = zeta_nu_char_single_chain_list
    characterizer.zeta_nu_char_label_single_chain_list = zeta_nu_char_label_single_chain_list

    kappa_nu_single_chain_list       = [100, 500, 1000, 5000, 10000]
    kappa_nu_label_single_chain_list = [r'$\kappa_{\nu}='+str(kappa_nu_single_chain_list[i])+'$' for i in range(len(kappa_nu_single_chain_list))]

    characterizer.kappa_nu_single_chain_list       = kappa_nu_single_chain_list
    characterizer.kappa_nu_label_single_chain_list = kappa_nu_label_single_chain_list

    psi_minimization_zeta_nu_char_single_chain_list       = zeta_nu_char_single_chain_list[0:4]
    psi_minimization_zeta_nu_char_label_single_chain_list = zeta_nu_char_label_single_chain_list[0:4]

    characterizer.psi_minimization_zeta_nu_char_single_chain_list       = psi_minimization_zeta_nu_char_single_chain_list
    characterizer.psi_minimization_zeta_nu_char_label_single_chain_list = psi_minimization_zeta_nu_char_label_single_chain_list

    psi_minimization_kappa_nu_single_chain_list       = [int(kappa_nu_single_chain_list[i]/10) for i in range(len(kappa_nu_single_chain_list))]
    psi_minimization_kappa_nu_label_single_chain_list = [r'$\kappa_{\nu}='+str(psi_minimization_kappa_nu_single_chain_list[i])+'$' for i in range(len(psi_minimization_kappa_nu_single_chain_list))]

    characterizer.psi_minimization_kappa_nu_single_chain_list       = psi_minimization_kappa_nu_single_chain_list
    characterizer.psi_minimization_kappa_nu_label_single_chain_list = psi_minimization_kappa_nu_label_single_chain_list

    bergapprx_lmbda_nu_cutoff = 0.84136

    characterizer.bergapprx_lmbda_nu_cutoff = bergapprx_lmbda_nu_cutoff

    nu_chain_network_list = [i for i in range(5, 5**5+1)] # nu = 5 -> nu = 3125

    characterizer.nu_chain_network_list = nu_chain_network_list

    zeta_nu_char_chain_network_list       = [50, 100, 500]
    zeta_nu_char_label_chain_network_list = [r'$\zeta_{\nu}^{char}='+str(zeta_nu_char_chain_network_list[i])+'$' for i in range(len(zeta_nu_char_chain_network_list))]

    characterizer.zeta_nu_char_chain_network_list       = zeta_nu_char_chain_network_list
    characterizer.zeta_nu_char_label_chain_network_list = zeta_nu_char_label_chain_network_list

    kappa_nu_chain_network_list       = [500, 1000, 5000]
    kappa_nu_label_chain_network_list = [r'$\kappa_{\nu}='+str(kappa_nu_chain_network_list[i])+'$' for i in range(len(kappa_nu_chain_network_list))]

    characterizer.kappa_nu_chain_network_list       = kappa_nu_chain_network_list
    characterizer.kappa_nu_label_chain_network_list = kappa_nu_label_chain_network_list

    return characterizer

def default_pre_processing_parameters():
    
    pre_processing = SimpleNamespace()

    form_compiler_optimize          = True
    form_compiler_cpp_optimize      = True
    form_compiler_representation    = "uflacs"
    form_compiler_quadrature_degree = 4

    pre_processing.form_compiler_optimize          = form_compiler_optimize
    pre_processing.form_compiler_cpp_optimize      = form_compiler_cpp_optimize
    pre_processing.form_compiler_representation    = form_compiler_representation
    pre_processing.form_compiler_quadrature_degree = form_compiler_quadrature_degree

    return pre_processing

def default_problem_parameters():
    
    problem = SimpleNamespace()

    hsize = 1e-3

    problem.hsize = hsize

    return problem

def default_geometry_parameters():
    
    geometry = SimpleNamespace()

    meshpoints            = [(0,0)]
    meshpoints_label_list = ['']
    meshpoints_color_list = ['']
    meshpoints_name_list  = ['']

    geometry.meshpoints            = meshpoints
    geometry.meshpoints_label_list = meshpoints_label_list
    geometry.meshpoints_color_list = meshpoints_color_list
    geometry.meshpoints_name_list  = meshpoints_name_list

    return geometry

def default_material_parameters():
    
    material = SimpleNamespace()

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

    material.k_B          = k_B
    material.N_A          = N_A
    material.h            = h
    material.hbar         = hbar
    material.T            = T
    material.beta         = beta
    material.omega_0      = omega_0
    material.zeta_nu_char = zeta_nu_char
    material.kappa_nu     = kappa_nu
    material.nu_b         = nu_b
    material.zeta_b_char  = zeta_b_char
    material.kappa_b      = kappa_b

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

    material.nu_distribution = nu_distribution
    material.nu_list         = nu_list
    material.nu_min          = nu_min
    material.nu_bar          = nu_bar
    material.Delta_nu        = Delta_nu
    material.nu_list         = nu_list
    material.nu_min          = nu_min
    material.nu_bar          = nu_bar
    material.Delta_nu        = Delta_nu

    # Define chain lengths to chunk during deformation
    nu_chunks_list = nu_list[::2] # nu = 5, nu = 7, ..., nu = 15
    nu_chunks_indx_list = nu_chunks_list.copy()
    for i in range(len(nu_chunks_list)):
        nu_chunks_indx_list[i] = nu_list.index(nu_chunks_list[i])
    nu_chunks_label_list = [r'$\nu='+str(nu_list[nu_chunks_indx_list[i]])+'$' for i in range(len(nu_chunks_list))]
    nu_chunks_color_list = ['orange', 'blue', 'green', 'red', 'purple', 'brown']
    
    material.nu_chunks_list       = nu_chunks_list
    material.nu_chunks_indx_list  = nu_chunks_indx_list
    material.nu_chunks_label_list = nu_chunks_label_list
    material.nu_chunks_color_list = nu_chunks_color_list

    # Define the points to chunk during deformation
    point_chunks_indx_list = [0, 1, 2]

    material.point_chunks_indx_list = point_chunks_indx_list

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

    material.network_model                      = network_model
    material.phenomenological_model             = phenomenological_model
    material.physical_dimension                 = physical_dimension
    material.physical_dimensionality            = physical_dimensionality
    material.incompressibility_assumption       = incompressibility_assumption
    material.macro2micro_deformation_assumption = macro2micro_deformation_assumption
    material.micro2macro_homogenization_scheme  = micro2macro_homogenization_scheme
    material.chain_level_load_sharing           = chain_level_load_sharing
    material.rate_dependence                    = rate_dependence
    material.two_dimensional_formulation        = two_dimensional_formulation
    material.microdisk_quadrature_order         = microdisk_quadrature_order
    material.microsphere_quadrature_order       = microsphere_quadrature_order

    return material

def default_fem_parameters():
    
    fem = SimpleNamespace()

    u_degree                = 1
    metadata                = {"quadrature_degree": 4}
    tensor2vector_indx_dict = {"11": 0, "12": 1, "13": 2, "21": 3, "22": 4, "23": 5, "31": 6, "32": 7, "33": 8}

    fem.u_degree                = u_degree
    fem.metadata                = metadata
    fem.tensor2vector_indx_dict = tensor2vector_indx_dict
    
    return fem

def default_deformation_parameters():

    deformation = SimpleNamespace()

    deformation_type             = "uniaxial"
    deformation.deformation_type = deformation_type

    # general network deformation parameters
    K_G                     = 100
    lmbda_damping_init      = 1e3
    min_lmbda_damping_val   = 1e-8
    iter_max_Gamma_val_NR   = 1000
    tol_Gamma_val_NR        = 1e-2
    iter_max_lmbda_c_val_NR = 1000
    tol_lmbda_c_val_NR      = 1e-4
    iter_max_stag_NR        = 2000
    tol_lmbda_c_val_stag_NR = 1e-4
    tol_Gamma_val_stag_NR   = 1e-2
    epsilon                 = 1e-12
    max_J_val_cond          = 1e12
    iter_max_d_c_val        = 1000
    tol_d_c_val             = 1e-4
    k_cond_val              = 1e-8

    deformation.K_G                     = K_G
    deformation.lmbda_damping_init      = lmbda_damping_init
    deformation.min_lmbda_damping_val   = min_lmbda_damping_val
    deformation.iter_max_Gamma_val_NR   = iter_max_Gamma_val_NR
    deformation.tol_Gamma_val_NR        = tol_Gamma_val_NR
    deformation.iter_max_lmbda_c_val_NR = iter_max_lmbda_c_val_NR
    deformation.tol_lmbda_c_val_NR      = tol_lmbda_c_val_NR
    deformation.iter_max_stag_NR        = iter_max_stag_NR
    deformation.tol_lmbda_c_val_stag_NR = tol_lmbda_c_val_stag_NR
    deformation.tol_Gamma_val_stag_NR   = tol_Gamma_val_stag_NR
    deformation.epsilon                 = epsilon
    deformation.max_J_val_cond          = max_J_val_cond
    deformation.iter_max_d_c_val        = iter_max_d_c_val
    deformation.tol_d_c_val             = tol_d_c_val
    deformation.k_cond_val              = k_cond_val

    # timing parameters
    t_min = 0.  # sec
    t_max = 15. # sec

    deformation.t_min = t_min
    deformation.t_max = t_max

    return deformation

def default_solver_u_parameters():
    
    solver_u = SimpleNamespace()

    nonlinear_solver = "snes"

    solver_u.nonlinear_solver = nonlinear_solver

    return solver_u

def default_solver_u_settings_parameters():
    
    solver_u_settings = SimpleNamespace()

    linear_solver           = "mumps"
    method                  = "newtontr"
    line_search             = "cp"
    preconditioner          = "hypre_amg"
    maximum_iterations      = 200
    absolute_tolerance      = 1e-8
    relative_tolerance      = 1e-7
    solution_tolerance      = 1e-7
    report                  = True
    error_on_nonconvergence = False

    solver_u_settings.linear_solver           = linear_solver
    solver_u_settings.method                  = method
    solver_u_settings.line_search             = line_search
    solver_u_settings.preconditioner          = preconditioner
    solver_u_settings.maximum_iterations      = maximum_iterations
    solver_u_settings.absolute_tolerance      = absolute_tolerance
    solver_u_settings.relative_tolerance      = relative_tolerance
    solver_u_settings.solution_tolerance      = solution_tolerance
    solver_u_settings.report                  = report
    solver_u_settings.error_on_nonconvergence = error_on_nonconvergence

    return solver_u_settings

def default_post_processing_parameters():
    
    post_processing = SimpleNamespace()

    ext = "xdmf"
    file_results = "results."+ext

    post_processing.ext = ext
    post_processing.file_results = file_results

    save_u                                = True
    save_lmbda_c_mesh                     = True
    save_lmbda_c_chunks                   = True
    save_lmbda_c_eq_mesh                  = False
    save_lmbda_c_eq_chunks                = False
    save_lmbda_nu_mesh                    = False
    save_lmbda_nu_chunks                  = False
    save_lmbda_nu_max_mesh                = False
    save_lmbda_nu_max_chunks              = False
    save_upsilon_c_mesh                   = False
    save_upsilon_c_chunks                 = False
    save_Upsilon_c_mesh                   = True
    save_Upsilon_c_chunks                 = True
    save_d_c_mesh                         = False
    save_d_c_chunks                       = False
    save_D_c_mesh                         = True
    save_D_c_chunks                       = True
    save_epsilon_cnu_diss_mesh            = False
    save_epsilon_cnu_diss_chunks          = False
    save_Epsilon_cnu_diss_mesh            = False
    save_Epsilon_cnu_diss_chunks          = False
    save_epsilon_c_diss_mesh              = False
    save_epsilon_c_diss_chunks            = False
    save_Epsilon_c_diss_mesh              = False
    save_Epsilon_c_diss_chunks            = False
    save_overline_epsilon_cnu_diss_mesh   = False
    save_overline_epsilon_cnu_diss_chunks = False
    save_overline_Epsilon_cnu_diss_mesh   = False
    save_overline_Epsilon_cnu_diss_chunks = False
    save_overline_epsilon_c_diss_mesh     = False
    save_overline_epsilon_c_diss_chunks   = False
    save_overline_Epsilon_c_diss_mesh     = False
    save_overline_Epsilon_c_diss_chunks   = False
    save_sigma_mesh                       = True
    save_sigma_chunks                     = True
    save_F_mesh                           = True
    save_F_chunks                         = True

    post_processing.save_u                                = save_u
    post_processing.save_lmbda_c_mesh                     = save_lmbda_c_mesh
    post_processing.save_lmbda_c_chunks                   = save_lmbda_c_chunks
    post_processing.save_lmbda_c_eq_mesh                  = save_lmbda_c_eq_mesh
    post_processing.save_lmbda_c_eq_chunks                = save_lmbda_c_eq_chunks
    post_processing.save_lmbda_nu_mesh                    = save_lmbda_nu_mesh
    post_processing.save_lmbda_nu_chunks                  = save_lmbda_nu_chunks
    post_processing.save_lmbda_nu_max_mesh                = save_lmbda_nu_max_mesh
    post_processing.save_lmbda_nu_max_chunks              = save_lmbda_nu_max_chunks
    post_processing.save_upsilon_c_mesh                   = save_upsilon_c_mesh
    post_processing.save_upsilon_c_chunks                 = save_upsilon_c_chunks
    post_processing.save_Upsilon_c_mesh                   = save_Upsilon_c_mesh
    post_processing.save_Upsilon_c_chunks                 = save_Upsilon_c_chunks
    post_processing.save_d_c_mesh                         = save_d_c_mesh
    post_processing.save_d_c_chunks                       = save_d_c_chunks
    post_processing.save_D_c_mesh                         = save_D_c_mesh
    post_processing.save_D_c_chunks                       = save_D_c_chunks
    post_processing.save_epsilon_cnu_diss_mesh            = save_epsilon_cnu_diss_mesh
    post_processing.save_epsilon_cnu_diss_chunks          = save_epsilon_cnu_diss_chunks
    post_processing.save_Epsilon_cnu_diss_mesh            = save_Epsilon_cnu_diss_mesh
    post_processing.save_Epsilon_cnu_diss_chunks          = save_Epsilon_cnu_diss_chunks
    post_processing.save_epsilon_c_diss_mesh              = save_epsilon_c_diss_mesh
    post_processing.save_epsilon_c_diss_chunks            = save_epsilon_c_diss_chunks
    post_processing.save_Epsilon_c_diss_mesh              = save_Epsilon_c_diss_mesh
    post_processing.save_Epsilon_c_diss_chunks            = save_Epsilon_c_diss_chunks
    post_processing.save_overline_epsilon_cnu_diss_mesh   = save_overline_epsilon_cnu_diss_mesh
    post_processing.save_overline_epsilon_cnu_diss_chunks = save_overline_epsilon_cnu_diss_chunks
    post_processing.save_overline_Epsilon_cnu_diss_mesh   = save_overline_Epsilon_cnu_diss_mesh
    post_processing.save_overline_Epsilon_cnu_diss_chunks = save_overline_Epsilon_cnu_diss_chunks
    post_processing.save_overline_epsilon_c_diss_mesh     = save_overline_epsilon_c_diss_mesh
    post_processing.save_overline_epsilon_c_diss_chunks   = save_overline_epsilon_c_diss_chunks
    post_processing.save_overline_Epsilon_c_diss_mesh     = save_overline_Epsilon_c_diss_mesh
    post_processing.save_overline_Epsilon_c_diss_chunks   = save_overline_Epsilon_c_diss_chunks
    post_processing.save_sigma_mesh                       = save_sigma_mesh
    post_processing.save_sigma_chunks                     = save_sigma_chunks
    post_processing.save_F_mesh                           = save_F_mesh
    post_processing.save_F_chunks                         = save_F_chunks

    rewrite_function_mesh = False
    flush_output          = True
    functions_share_mesh  = True

    post_processing.rewrite_function_mesh = rewrite_function_mesh
    post_processing.flush_output          = flush_output
    post_processing.functions_share_mesh  = functions_share_mesh

    axes_linewidth      = 1.0
    font_family         = "sans-serif"
    text_usetex         = True
    ytick_right         = True
    ytick_direction     = "in"
    xtick_top           = True
    xtick_direction     = "in"
    xtick_minor_visible = True

    post_processing.axes_linewidth      = axes_linewidth
    post_processing.font_family         = font_family
    post_processing.text_usetex         = text_usetex
    post_processing.ytick_right         = ytick_right
    post_processing.ytick_direction     = ytick_direction
    post_processing.xtick_top           = xtick_top
    post_processing.xtick_direction     = xtick_direction
    post_processing.xtick_minor_visible = xtick_minor_visible

    return post_processing