# PyParticles : Particles simulation in python
# Copyright (C) 2013  Ricardo Miranda
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,python arrays map   functional programmnig
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy                                                        as np
import pyparticles.pset.particles_set                               as ps
import pyparticles.pset.constrained_force_interactions  as cfi
import pyparticles.forces.navier_stokes                 as ns

from pyparticles.geometry.dist import distance


class Mesh(object):

    """
    Class to create a mesh of size h. It is used for Hydrodynamic Particles Simulation. h is the smoothing length.
    This mesh contains all the particles used in the simulation.
    Algorithm:
        1) every particle is mapped inside the mesh and assign to a cell;
        2) assume Particle A. For every particle inside the same cell, or adjacent cells, the distance between its center and
            Particle A's center is computed;
        3) if the distance between particles is less than h class constrained_force_interactions is updated;
        4) class constrained_force_interactions stores relations between particles that influence each other.
    """

    __INI_FLOAT = -9999.9           # To catch calculation error it is useful because sometimes zero is a value that makes sense
    __INI_INT   = -9999             # To catch calculation error it is useful because sometimes zero is a value that makes scene

    __ar_axis0      = None
    __ar_axis1      = None
    __ar_axis2      = None
    __h             = None          # h is the mesh spacing (equal to smoothing length)
    __corner1       = None          # Corner1 is the the first corner of the volume that contains all the particles
    __corner2       = None          # Corner2 is the the second corner of the volume that contains all the particles
    __cell          = "cell"        # Property added to particles_set

    # Init ---------------------------------------------------------------------

    def __init__(self, pset,
                       h         = None,
                       corner1   = None,
                       corner2   = None,
                       dtype     = np.float64):

        self.__dtype    = dtype

        if h        != None:
            self.__h        = float(h)
        if corner1   != None:
            self.__corner1   = np.array(corner1, self.__dtype)
        if corner2   != None:
            self.__corner2   = np.array(corner2, self.__dtype)

        self.__dtype = dtype

        # particles_set has 3D property 'cell'. Each particle is placed inside a cell in the grid mesh.
        pset.add_property_by_name(property_name = self.__cell, dim = 3, to_type = np.int64)

    # Get and Set --------------------------------------------------------------

    def geth(self   ):
        return self.__h
    def seth(self, h):
        self.__h = h
    h = property(geth, seth , doc="Manipulates smoothing length" )

    def getcorner1(self        ):
        return self.__corner1
    def setcorner1(self, corner1):
        self.__corner1 =  np.array(corner1, self.__dtype)
    corner1 = property(getcorner1, setcorner1 , doc="Corner1 is the the first corner of the volume that contains all the particles" )

    def getcorner2(self        ):
        return self.__corner2
    def setcorner2(self, corner2):
        self.__corner2 =  np.array(corner2, self.__dtype)
    corner2 = property(getcorner2, setcorner2 , doc="Corner2 is the the second corner of the volume that contains all the particles" )

    '''
    Methods --------------------------------------------------------------------
        calc_mesh
        calc_particle_mesh_location
        calc_particles_that_interact
            __particle_is_in_cell
    '''

    def calc_mesh(self, dtype = np.float64):
        L = self.__INI_FLOAT * np.ones((3), dtype=dtype)


        h       = self.__h
        corner1 = self.__corner1
        corner2 = self.__corner2
        L       = corner2 - corner1

        self.__ar_axis0     = np.arange(-h, L[0]+2*h, h, dtype=self.__dtype)
        self.__ar_axis1     = np.arange(-h, L[1]+2*h, h, dtype=self.__dtype)
        self.__ar_axis2     = np.arange(-h, L[2]+2*h, h, dtype=self.__dtype)

    def calc_particles_mesh_locations(self, pset, dtype = np.float64):
        '''
        Finds the mesh's cell where the each particle is located
        '''

        ar_axis0        = self.__ar_axis0
        ar_axis1        = self.__ar_axis1
        ar_axis2        = self.__ar_axis2
        corner1         = self.__corner1
        max0            = len(ar_axis0)-1
        max1            = len(ar_axis1)-1
        max2            = len(ar_axis2)-1
        particle_cell   = []                    # List of cells where each particle is

        for X in pset.X:
            cell        = [self.__INI_INT,self.__INI_INT,self.__INI_INT]
            cell0       = 0
            cell1       = 0
            cell2       = 0

            for x in range(0, max0):
                if (X[0] >= corner1[0]+ar_axis0[x  ] and     # This if is to guarantee that each particle is assign just to 1 cell
                    X[0] <= corner1[0]+ar_axis0[x+1]):
                        break
                else:
                    cell0 = cell0+1

            for x in range(0, max1):
                if (X[1] >= corner1[1]+ar_axis1[x  ] and
                    X[1] <= corner1[1]+ar_axis1[x+1]):
                    break
                else:
                    cell1 = cell1+1

            for x in range(0, max2):
                if (X[2] >= corner1[2]+ar_axis2[x  ] and
                    X[2] <= corner1[2]+ar_axis2[x+1]):
                        break
                else:
                    cell2 = cell2+1

            cell[0] = cell0
            cell[1] = cell1
            cell[2] = cell2

            particle_cell.append(cell)

        particle_cell=np.asarray(particle_cell, np.int64)

        pset.get_by_name(self.__cell)[:] = particle_cell[:]

    def __particle_is_in_cell(self, part_ij, cell_ij):
        if (part_ij[0] == cell_ij[0] and
            part_ij[1] == cell_ij[1] and
            part_ij[2] == cell_ij[2]):
                return True
        else:
            return False

    def calc_particles_that_interact(self, pset):
        '''
        Covers the entire mesh looking for particles. Algorithm:
            1) for the first particle in a cell (computed in Mesh.calc_particles_mesh_locations) looks for other particles
                in the same cell. Verifies if distance between centers is less than smoothing lenfth (h). If so adds relation
                to relations array (ConstrainedForceInteractions);
            2) looks for other particles in adjacent cells. Verifies if distance between centers is less than smoothing length (h).
                If so adds relation to relations array (ConstrainedForceInteractions);
            3) goes to the next cell.
        Distances between 2 particles in not computed twice.
        '''

        ar_axis0            = self.__ar_axis0
        ar_axis1            = self.__ar_axis1
        ar_axis2            = self.__ar_axis2

        '''
        2 additional columns had been added to make the algorithm easier to implement (beginning and end). There are no
            particles in the last cell in every direction.
        '''
        max0                = len(ar_axis0)-2
        max1                = len(ar_axis1)-2
        max2                = len(ar_axis2)-2

        h                   = self.__h
        particle_index      = []                    # List of particles to remember every particle already computed
        particle_cell       = None                  # List of cells where each particle is
        f_conn              = []                    # Connections list
        ij_dst              = []                    # List of distances between particles i and j

        particle_cell       = np.array(pset.get_by_name(self.__cell)[:])
        part_nbr            = len(particle_cell)-1

        # Scans the entire mesh
        for x0 in range(1, max0):
            for x1 in range(1, max1):
                for x2 in range(1, max2):
                    for i in range(0, part_nbr):
                        # Reference particle
                        if self.__particle_is_in_cell(particle_cell[i,:], (x0,x1,x2)):
                            particle_index.append(i)
                            j = i

                            # Scans particles_set
                            for j in range(i, part_nbr):
                                scnd_part_found = False

                                if (j in particle_index):
                                    pass
                                else:
                                    # Checks in 9 cells if there are particles
                                    if self.__particle_is_in_cell(particle_cell[j,:], (x0,  x1,  x2  )):
                                        x = np.array([pset.X[i,0],pset.X[i,1],pset.X[i,2]], dtype=np.float64)
                                        y = np.array([pset.X[j,0],pset.X[j,1],pset.X[j,2]], dtype=np.float64)
                                        scnd_part_found = True

                                    if scnd_part_found == False:
                                        if self.__particle_is_in_cell(particle_cell[j,:], (x0+1,x1,  x2  )):
                                            x = np.array([pset.X[i,0],pset.X[i,1],pset.X[i,2]], dtype=np.float64)
                                            y = np.array([pset.X[j,0],pset.X[j,1],pset.X[j,2]], dtype=np.float64)
                                            scnd_part_found = True

                                    if scnd_part_found == False:
                                        if self.__particle_is_in_cell(particle_cell[j,:], (x0+1,x1+1,x2  )):
                                            x = np.array([pset.X[i,0],pset.X[i,1],pset.X[i,2]], dtype=np.float64)
                                            y = np.array([pset.X[j,0],pset.X[j,1],pset.X[j,2]], dtype=np.float64)
                                            scnd_part_found = True

                                    if scnd_part_found == False:
                                        if self.__particle_is_in_cell(particle_cell[j,:], (x0+1,x1+1,x2+1)):
                                            x = np.array([pset.X[i,0],pset.X[i,1],pset.X[i,2]], dtype=np.float64)
                                            y = np.array([pset.X[j,0],pset.X[j,1],pset.X[j,2]], dtype=np.float64)
                                            scnd_part_found = True

                                    if scnd_part_found == False:
                                        if self.__particle_is_in_cell(particle_cell[j,:], (x0,  x1+1,x2  )):
                                            x = np.array([pset.X[i,0],pset.X[i,1],pset.X[i,2]], dtype=np.float64)
                                            y = np.array([pset.X[j,0],pset.X[j,1],pset.X[j,2]], dtype=np.float64)
                                            scnd_part_found = True

                                    if scnd_part_found == False:
                                        if self.__particle_is_in_cell(particle_cell[j,:], (x0,  x1+1,x2+1)):
                                            x = np.array([pset.X[i,0],pset.X[i,1],pset.X[i,2]], dtype=np.float64)
                                            y = np.array([pset.X[j,0],pset.X[j,1],pset.X[j,2]], dtype=np.float64)
                                            scnd_part_found = True

                                    if scnd_part_found == False:
                                        if self.__particle_is_in_cell(particle_cell[j,:], (x0,  x1,  x2+1)):
                                            x = np.array([pset.X[i,0],pset.X[i,1],pset.X[i,2]], dtype=np.float64)
                                            y = np.array([pset.X[j,0],pset.X[j,1],pset.X[j,2]], dtype=np.float64)
                                            scnd_part_found = True

                                    if scnd_part_found == False:
                                        if self.__particle_is_in_cell(particle_cell[j,:], (x0+1,x1  ,x2+1)):
                                            x = np.array([pset.X[i,0],pset.X[i,1],pset.X[i,2]], dtype=np.float64)
                                            y = np.array([pset.X[j,0],pset.X[j,1],pset.X[j,2]], dtype=np.float64)
                                            scnd_part_found = True


                                    if scnd_part_found:
                                        dist    = self.__INI_FLOAT
                                        dist    = distance(x, y)

                                        if dist <= h:
                                            conn    = [self.__INI_INT,  self.__INI_INT]
                                            conn    = [i,j]

                                            ij_dst.append(dist)
                                            f_conn.append(conn)

        return f_conn

