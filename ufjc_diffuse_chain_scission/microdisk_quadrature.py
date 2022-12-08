################################################################################################################################
# General setup
################################################################################################################################

# Import necessary libraries
import numpy as np
import quadpy as qp

################################################################################################################################
# Full network microdisk quadrature class
################################################################################################################################

# May need to figure out a 2D version of the microsphere for 2D domains...

################################################################################################################################
# Full network microsphere quadrature class
################################################################################################################################

class MicrodiskQuadratureScheme: # add functionality to access any disk quadrature scheme that quadpy offers
    
    ############################################################################################################################
    # Initialization
    ############################################################################################################################

    def __init__(self, microdisk_quadrature_order):
        
        # qp_lebedev_order_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]

        # if microsphere_quadrature_order in qp_lebedev_order_list:
        #     scheme = self.lebedev(microsphere_quadrature_order)
        # else:
        #     scheme = qp.u3.get_good_scheme(microsphere_quadrature_order)
        
        # self.microsphere_quadrature_order = microsphere_quadrature_order
        # self.w = scheme.weights
        # self.lmbda_c_0_dir = np.transpose(scheme.points)
        # self.dim = scheme.points.shape[0]
        # self.points_num = scheme.points.shape[1]

        pass
    
    # ############################################################################################################################
    # # Function that stores lebedev quadrature schemes accessible with user-friendly syntax
    # ############################################################################################################################
    # def lebedev(self, quadrature_order):
    #     if quadrature_order == 3:
    #         return qp.u3._lebedev.lebedev_003a
    #     elif quadrature_order == 5:
    #         return qp.u3._lebedev.lebedev_005
    #     elif quadrature_order == 7:
    #         return qp.u3._lebedev.lebedev_007
    #     elif quadrature_order == 9:
    #         return qp.u3._lebedev.lebedev_009
    #     elif quadrature_order == 11:
    #         return qp.u3._lebedev.lebedev_011
    #     elif quadrature_order == 13:
    #         return qp.u3._lebedev.lebedev_013
    #     elif quadrature_order == 15:
    #         return qp.u3._lebedev.lebedev_015
    #     elif quadrature_order == 17:
    #         return qp.u3._lebedev.lebedev_017
    #     elif quadrature_order == 19:
    #         return qp.u3._lebedev.lebedev_019
    #     elif quadrature_order == 21:
    #         return qp.u3._lebedev.lebedev_021
    #     elif quadrature_order == 23:
    #         return qp.u3._lebedev.lebedev_023
    #     elif quadrature_order == 25:
    #         return qp.u3._lebedev.lebedev_025
    #     elif quadrature_order == 27:
    #         return qp.u3._lebedev.lebedev_027
    #     elif quadrature_order == 29:
    #         return qp.u3._lebedev.lebedev_029
    #     elif quadrature_order == 31:
    #         return qp.u3._lebedev.lebedev_031
    #     elif quadrature_order == 35:
    #         return qp.u3._lebedev.lebedev_035
    #     elif quadrature_order == 41:
    #         return qp.u3._lebedev.lebedev_041
    #     elif quadrature_order == 47:
    #         return qp.u3._lebedev.lebedev_047
    #     elif quadrature_order == 53:
    #         return qp.u3._lebedev.lebedev_053
    #     elif quadrature_order == 59:
    #         return qp.u3._lebedev.lebedev_059
    #     elif quadrature_order == 65:
    #         return qp.u3._lebedev.lebedev_065
    #     elif quadrature_order == 71:
    #         return qp.u3._lebedev.lebedev_071
    #     elif quadrature_order == 77:
    #         return qp.u3._lebedev.lebedev_077
    #     elif quadrature_order == 83:
    #         return qp.u3._lebedev.lebedev_083
    #     elif quadrature_order == 89:
    #         return qp.u3._lebedev.lebedev_089
    #     elif quadrature_order == 95:
    #         return qp.u3._lebedev.lebedev_095
    #     elif quadrature_order == 101:
    #         return qp.u3._lebedev.lebedev_101
    #     elif quadrature_order == 107:
    #         return qp.u3._lebedev.lebedev_107
    #     elif quadrature_order == 113:
    #         return qp.u3._lebedev.lebedev_113
    #     elif quadrature_order == 119:
    #         return qp.u3._lebedev.lebedev_119
    #     elif quadrature_order == 125:
    #         return qp.u3._lebedev.lebedev_125
    #     elif quadrature_order == 131:
    #         return qp.u3._lebedev.lebedev_131