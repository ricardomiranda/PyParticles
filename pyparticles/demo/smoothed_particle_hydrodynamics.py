# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
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

import numpy as                                 np

import pyparticles.pset.particles_set as        ps

import pyparticles.pset.opencl_context as       occ

import pyparticles.ode.euler_solver as          els
import pyparticles.ode.leapfrog_solver as       lps
import pyparticles.ode.runge_kutta_solver as    rks
import pyparticles.ode.stormer_verlet_solver as svs
import pyparticles.ode.midpoint_solver as       mds

import pyparticles.forces.const_force as        cf
import pyparticles.forces.drag as               dr
import pyparticles.forces.multiple_force as     mf
import pyparticles.forces.navier_stokes  as     ns

import pyparticles.animation.animated_ogl as    aogl

import pyparticles.pset.default_boundary as db
import pyparticles.pset.rebound_boundary as rb

from pyparticles.utils.pypart_global import test_pyopencl

def default_pos( pset , indx ):

    t = default_pos.sim_time.time


def initial_pos      (pset, pcnt, pcntVol, pcntWall, dx, L):
    """
    In this example the particles are evenly put inside a cube size L, center=(L/2,L/2,L/2)
    """

    # Volume
    i = 0
    x = dx
    while x < 1.0:
        y = dx
        while y < 1.0:
            z = dx
            while z < 1.0:
                pset.X[i,0] = x * L
                pset.X[i,1] = y * L
                pset.X[i,2] = z * L
                i           = i + 1
                z = z + dx
            y = y + dx
        x = x + dx

    # Bottom
    aux = i
    x = 0.0
    while x <= 1.0:
        y = 0.0
        while y <= 1.0:
            z = 0.0
            pset.X[i,0] = x * L
            pset.X[i,1] = y * L
            pset.X[i,2] = z
            i           = i + 1
            y = y + dx
        x = x + dx

    # Wall 1
    aux = i
    x = 0.0
    while x <= 1.0:
        y = 0.0
        z = dx
        while z <= 1.0:
            pset.X[i,0] = x * L
            pset.X[i,1] = y
            pset.X[i,2] = z * L
            i           = i + 1
            z = z + dx
        x = x + dx

    # Wall 2
    aux = i
    x = 0.0
    while x <= 1.0:
        y = 1.0
        z = dx
        while z <= 1.0:
            pset.X[i,0] = x * L
            pset.X[i,1] = y
            pset.X[i,2] = z * L
            i           = i + 1
            z = z + dx
        x = x + dx

    # Wall 3
    aux = i
    x = 0.0
    y = 0.0
    while y <= 1.0:
        z = dx
        while z <= 1.0:
            pset.X[i,0] = x
            pset.X[i,1] = y * L
            pset.X[i,2] = z * L
            i           = i + 1
            z = z + dx
        y = y + dx

    # Wall 4
    aux = i
    x = 1.0
    y = 0.0
    while y <= 1.0:
        z = dx
        while z <= 1.0:
            pset.X[i,0] = x
            pset.X[i,1] = y * L
            pset.X[i,2] = z * L
            i           = i + 1
            z = z + dx
        y = y + dx

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

    steps   = 10000000              # Number of steps
    dt      = 0.005                 # dt should be defined according to a numerical stability parameter, a simple one will be dt<2h/vmax
    dx      = 0.025                 # spacing between particles L/Nx
    aux     = ((1.0/dx)-1.0)
    pcntVol = aux*aux*aux           # Number of particles in
    aux     = aux+2.0
    pcntWall= aux*aux               # Number of particles in wallls
    pcnt    = pcntVol+pcntWall*5
    L       =  1.0                  # Water cube size
    g       = -9.8                  # Gravity acceleration


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


    pset = ps.ParticlesSet( size = np.int(pcnt) , mass = True , density = True , dtype=np.float64 )
    initial_pos               (pset, np.int(pcnt), np.int(pcntVol), np.int(pcntWall), dx, L)
    initial_vel               (pset, np.int(pcnt)                                          )
    initial_pressure          (pset, np.int(pcnt),                                        L)

    nstk = ns.HPSNavierStokes()
    nstk.calc_density_water_rest(pset,                                                    L)

    pset.M[:] = 1.0 / pcntVol

    grav = cf.ConstForce( pset.size , dim=pset.dim , u_force=( 0.0 , 0.0 , g ) )

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
        solver = els.EulerSolverOCL( multi , pset , dt , ocl_context=occx )
    else :
        solver = els.EulerSolver( multi , pset , dt )

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


