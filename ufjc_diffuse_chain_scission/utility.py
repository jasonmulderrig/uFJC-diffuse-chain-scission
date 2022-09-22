# import necessary libraries
from __future__ import division
from dolfin import *
import os
import pathlib
import subprocess
from dolfin_utils.meshconvert import meshconvert
import matplotlib.pyplot as plt

def generate_savedir(namedir):
    savedir = "./"+namedir+"/"
    create_savedir(savedir)

    return savedir

def create_savedir(savedir):
    if MPI.rank(MPI.comm_world) == 0:
        if os.path.isdir(savedir) == False:
            pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    MPI.barrier(MPI.comm_world)

def fenics_mesher(fenics_mesh, subdir, mesh_name):

    def generate_mesh_dir(subdir):
        mesh_dir = "./"+subdir+"/"+"meshes"+"/"
        create_savedir(mesh_dir)

        return mesh_dir
    
    mesh_dir = generate_mesh_dir(subdir)

    geo_mesh = XDMFFile(MPI.comm_world, mesh_dir+mesh_name+".xdmf")
    geo_mesh.write(fenics_mesh)
    fenics_mesh.init()

    return fenics_mesh

def mesh_topologier(fenics_mesh_topology, subdir, mesh_topology_name):

    def generate_mesh_topology_dir(subdir):
        mesh_topology_dir = "./"+subdir+"/"+"mesh_topologies"+"/"
        create_savedir(mesh_topology_dir)

        return mesh_topology_dir
    
    mesh_topology_dir = generate_mesh_topology_dir(subdir)

    mesh_topology = XDMFFile(MPI.comm_world, mesh_topology_dir+mesh_topology_name+".xdmf")
    mesh_topology.write(fenics_mesh_topology)

def gmsh_mesher(gmsh_file, subdir, mesh_name):
    
    def generate_mesh_dir(subdir):
        mesh_dir = "./"+subdir+"/"+"meshes"+"/"
        create_savedir(mesh_dir)

        return mesh_dir
    
    mesh_dir = generate_mesh_dir(subdir)
    temp_mesh = Mesh() # create an empty mesh object

    if not os.path.isfile(mesh_dir+mesh_name+".xdmf"):

        if MPI.rank(MPI.comm_world) == 0:

            # Create a .geo file defining the mesh
            geo_file = open(mesh_dir+mesh_name+".geo", "w")
            geo_file.writelines(gmsh_file)
            geo_file.close()

            # Call gmsh to generate the mesh file and call dolfin-convert to generate the .xml file
            try:
                subprocess.call(["gmsh", "-2", "-o", mesh_dir+mesh_name+".msh", mesh_dir+mesh_name+".geo"])
            except OSError:
                print("-----------------------------------------------------------------------------")
                print(" Error: unable to generate the mesh using gmsh")
                print(" Make sure that you have gmsh installed and have added it to your system PATH")
                print("-----------------------------------------------------------------------------")
                return
            meshconvert.convert2xml(mesh_dir+mesh_name+".msh", mesh_dir+mesh_name+".xml", "gmsh")
        
        # Convert the .msh file to a .xdmf file
        MPI.barrier(MPI.comm_world)
        mesh = Mesh(mesh_dir+mesh_name+".xml")
        geo_mesh = XDMFFile(MPI.comm_world, mesh_dir+mesh_name+".xdmf")
        geo_mesh.write(mesh)
        geo_mesh.read(temp_mesh)
    
    else:
        geo_mesh = XDMFFile(MPI.comm_world, mesh_dir+mesh_name+".xdmf")
        geo_mesh.read(temp_mesh)
    
    return temp_mesh

def latex_formatting_figure(post_processing_parameters):

    ppp = post_processing_parameters

    plt.rcParams['axes.linewidth'] = ppp.axes_linewidth # set the value globally
    plt.rcParams['font.family']    = ppp.font_family
    plt.rcParams['text.usetex']    = ppp.text_usetex # comment this line out in WSL2, uncomment this line in native Linux on workstation
    
    plt.rcParams['ytick.right']     = ppp.ytick_right
    plt.rcParams['ytick.direction'] = ppp.ytick_direction
    plt.rcParams['xtick.top']       = ppp.xtick_top
    plt.rcParams['xtick.direction'] = ppp.xtick_direction
    
    plt.rcParams["xtick.minor.visible"] = ppp.xtick_minor_visible

def save_current_figure(savedir, xlabel, xlabelfontsize, ylabel, ylabelfontsize, name):
    plt.xlabel(xlabel, fontsize=xlabelfontsize)
    plt.ylabel(ylabel, fontsize=ylabelfontsize)
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()

def save_current_figure_no_labels(savedir, name):
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()