# PyParticles : Particles simulation in python
# Copyright (C) 2013  Ricardo Miranda
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import code # RCM lixo
import pdb # RCM lixo

import numpy                                            as np

import pyparticles.pset.particles_set                   as ps

import pyparticles.pset.opencl_context                  as occ

import pyparticles.ode.euler_solver                     as els
import pyparticles.ode.leapfrog_solver                  as lps
import pyparticles.ode.runge_kutta_solver               as rks
import pyparticles.ode.stormer_verlet_solver            as svs
import pyparticles.ode.midpoint_solver                  as mds

import pyparticles.forces.const_force                   as cf
import pyparticles.forces.drag                          as dr
import pyparticles.forces.multiple_force                as mf
import pyparticles.forces.navier_stokes                 as ns

import pyparticles.animation.animated_ogl               as aogl

import pyparticles.pset.default_boundary                as db
import pyparticles.pset.rebound_boundary                as rb
import pyparticles.pset.constrained_x                   as csx
import pyparticles.pset.constrained_force_interactions  as cfi

import pyparticles.geometry.mesh                        as msh

from pyparticles.utils.pypart_global import test_pyopencl

__INI_FLOAT = -9999.9 # To catch calculation error it is useful because sometimes zero is a value that makes sense


def default_pos( pset , indx ):

    t = default_pos.sim_time.time


def initial_pos      (pset, pcnt, pcntVol, pcntWall, ar, L):
    """
    In this example the particles are evenly put inside a cube size L, center=(L/2,L/2,L/2)
    """

    # Volume
    n = pcntWall
    for i in range(1, len(ar)-1):
        for j in range(1, len(ar)-1):
            for k in range(1, len(ar)):
                pset.X[n,0] = ar[i] * L
                pset.X[n,1] = ar[j] * L
                pset.X[n,2] = ar[k] * L
                n           = n + 1

def boundaryParticles(pset, pcnt, pcntVol, pcntWall, ar, L):
    """
    Virtual particles are put on the walls
    """

    costrs  = csx.ConstrainedX(pset)    # This is a list of virtual particles fixed in space.
    ci      = None                      # Array with indices of the boundary virtual particles
    cx      = None                      # Array with fixed positions of the boundary virtual particles

    # Bottom
    n = 0
    for i in range(0, len(ar)):
        for j in range(0, len(ar)):
            k           = 0
            cx_part     = __INI_FLOAT * np.ones((3), dtype=np.float64)

            pset.X[n,0] = ar[i] * L
            pset.X[n,1] = ar[j] * L
            pset.X[n,2] = ar[k] * L

            if ci == None:
                cx_part[0]  = pset.X[n,0]
                cx_part[1]  = pset.X[n,1]
                cx_part[2]  = pset.X[n,2]
                ci          =       [n]
                cx          = cx_part
            else:
                cx_part[0]  = pset.X[n,0]
                cx_part[1]  = pset.X[n,1]
                cx_part[2]  = pset.X[n,2]
                ci          = np.append(ci, [n],    axis=0)
                cx          = np.append(cx, cx_part, axis=1)

            n          = n + 1

    # Wall 1
    for j in range(0, len(ar)):
        for k in range(1, len(ar)):
            i           = 0
            pset.X[n,0] = ar[i] * L
            pset.X[n,1] = ar[j] * L
            pset.X[n,2] = ar[k] * L

            cx_part[0]  = pset.X[n,0]
            cx_part[1]  = pset.X[n,1]
            cx_part[2]  = pset.X[n,2]
            ci          = np.append(ci, [n]    , axis=0)
            cx          = np.append(cx, cx_part, axis=1)

            n           = n + 1

    # Wall 2
    for j in range(0, len(ar)):
        for k in range(1, len(ar)):
            i           = len(ar)-1
            pset.X[n,0] = ar[i] * L
            pset.X[n,1] = ar[j] * L
            pset.X[n,2] = ar[k] * L

            cx_part[0]  = pset.X[n,0]
            cx_part[1]  = pset.X[n,1]
            cx_part[2]  = pset.X[n,2]
            ci          = np.append(ci, [n]    , axis=0)
            cx          = np.append(cx, cx_part, axis=1)

            n           = n + 1

    # Wall 3
    for i in range(1, len(ar)-1):
        for k in range(1, len(ar)):
            j           = 0
            pset.X[n,0] = ar[i] * L
            pset.X[n,1] = ar[j] * L
            pset.X[n,2] = ar[k] * L

            cx_part[0]  = pset.X[n,0]
            cx_part[1]  = pset.X[n,1]
            cx_part[2]  = pset.X[n,2]
            ci          = np.append(ci, [n]    , axis=0)
            cx          = np.append(cx, cx_part, axis=1)

            n           = n + 1

    # Wall 4
    for i in range(1, len(ar)-1):
        for k in range(1, len(ar)):
            j           = len(ar)-1
            pset.X[n,0] = ar[i] * L
            pset.X[n,1] = ar[j] * L
            pset.X[n,2] = ar[k] * L

            cx_part[0]  = pset.X[n,0]
            cx_part[1]  = pset.X[n,1]
            cx_part[2]  = pset.X[n,2]
            ci          = np.append(ci, [n]    , axis=0)
            cx          = np.append(cx, cx_part, axis=1)

            n           = n + 1

    cx      = np.reshape(cx, (-1, 3))
    costrs.add_x_constraint(ci, cx)

    return costrs



def initial_vel      (pset, pcnt        ):
    """
    In this example the particles are at rest in the beginning
    """

    for i in range(0 , pcnt-1):
        pset.V[i,0] = 0.0
        pset.V[i,1] = 0.0
        pset.V[i,2] = 0.0


def initial_pressure (pset, pcnt,      L):
    for i in range(0, pcnt-1):
        pass


def cube_water():
    """
    Smoothed particle hydrodynamics cube of water exemple
    """

    steps   = 10000000                      # Number of steps
    dt      = 0.005                         # dt should be defined according to a numerical stability parameter, a simple one will be dt<2h/vmax
    dx      = 0.1                           # spacing between particles L/Nx
#    dx      = 0.025                         # spacing between particles L/Nx
    aux     = (int(1.0/dx)-1)
    pcntVol = aux*aux*(aux+1)               # Number of particles in volume
    pcntWall= (aux+2)*(aux+2)               # Number of particles in the bottom
    pcntWall= pcntWall+(aux+1)*(aux+1)*4    # Number of particles in wallls
    pcnt    = pcntVol+pcntWall
    L       =  1.0                          # Water cube size
    g       = -9.8                          # Gravity acceleration

    h_local = .15

    ar = np.arange(0, 1+dx, dx)


    fl = True
    if test_pyopencl() :
        print( "OpenCL is installed and enabled " )
        print( " Try, at least, 200000 particles " )

        while fl :
            try :
                print( " " )
                pcnt = int( raw_input('How many particles: ') )
            except :
                print( "Please insert a number! " )
            else :
                fl = False


    pset    = ps.ParticlesSet   (size = np.int64(pcnt), mass = True, density = True, dtype   = np.float64)
    pset.add_property_by_name   (property_name = "pressure",                         to_type = np.float64)

    costrs  = boundaryParticles (pset=pset, pcnt=np.int(pcnt), pcntVol=np.int(pcntVol), pcntWall=np.int(pcntWall), ar=ar, L=L)
    initial_pos                 (pset=pset, pcnt=np.int(pcnt), pcntVol=np.int(pcntVol), pcntWall=np.int(pcntWall), ar=ar, L=L)
    initial_vel                 (pset=pset, pcnt=np.int(pcnt)                                                                )
    initial_pressure            (pset=pset, pcnt=np.int(pcnt),                                                            L=L)

    nstk = ns.HPSNavierStokes   (pset=pset     )
    nstk.calc_density_water_rest(pset=pset, H=L)
    nstk.calc_pressure          (pset=pset, H=L)

    pset.M[:] = 1000.0*L*L*L / pcntVol

    fi      = cfi.ConstrainedForceInteractions(pset)
    sk      = ns.HPSSmothingKernels(pset=pset)
    sk.h    = h_local                                   # RCM, for tests
    h       = sk.h

    point1  = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    point2  = np.array([L,   L,   L  ], dtype=np.float64)
    mesh    = msh.Mesh                  (pset=pset, h = h, corner1 = point1, corner2 = point2)
    mesh.calc_mesh                      ()
    mesh.calc_particles_mesh_locations  (pset=pset, dtype=np.float64)
    f_conn = mesh.calc_particles_that_interact   (pset=pset)
    fi.add_connections                  (fc=f_conn)
    sk.calc_smothing_kernels            (pset=pset)
    grav   = cf.ConstForce( pset.size , dim=pset.dim , u_force=( 0.0 , 0.0 , g ) )

    occx = None
    if test_pyopencl() :
        occx = occ.OpenCLcontext( pset.size , pset.dim , ( occ.OCLC_X | occ.OCLC_V | occ.OCLC_A | occ.OCLC_M ) )
        drag = dr.DragOCL( pset.size , dim=pset.dim , Consts=0.01 , ocl_context=occx )
    else :
        drag = dr.Drag( pset.size , dim=pset.dim , Consts=0.01 )


    multi = mf.MultipleForce( pset.size , dim=pset.dim )

    multi.append_force( grav )
    multi.append_force( drag )

    multi.set_masses( pset.M )

    #solver = mds.MidpointSolver( multi , pset , dt )
    if test_pyopencl() :
        solver = els.EulerSolverOCL( multi , pset , dt , ocl_context=occx  )
    else :
        solver = els.EulerSolverConstrained( multi , pset , dt , costrs )

#    code.interact(local=locals()) # RCM lixo
#    pdb.pm() # RCM lixo
    solver.update_force()

    default_pos.sim_time = solver.get_sim_time()

#    bd = ( -100.0 , 100.0 , -100.0 , 100.0 , 0.0 , 100.0 )
    bd = ( -1.0 , 1.0 , -1.0 , 1.0 , 0.0 , 2.0 ) # RCM
    bound = db.DefaultBoundary( bd , dim=3 , defualt_pos=default_pos )

    pset.set_boundary( bound )

    a = aogl.AnimatedGl()

    a.ode_solver = solver
    a.pset = pset
    a.steps = steps

    a.draw_particles.set_draw_model( a.draw_particles.DRAW_MODEL_VECTOR )

    a.init_rotation( -80 , [ 0.7 , 0.05 , 0 ]  )

    a.build_animation()
    a.start()

    return


