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

import numpy                                            as np
import pyparticles.pset.particles_set                   as ps
import pyparticles.pset.constrained_force_interactions  as cfi


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

    __INI_FLOAT = -9999.9           # To catch calculation error it is usefull because sometimes zero is a value that makes sence
    __INI_INT   = -9999             # To catch calculation error it is usefull because sometimes zero is a value that makes sence

    __ar_axis0      = None
    __ar_axis1      = None
    __ar_axis2      = None
    __h             = None          # h is the mesh spacing
    __point1        = None          # Point1 is the the first corner of the volume that contais all the particles
    __point2        = None          # Point2 is the the second corner of the volume that contais all the particles
    __cell          = "cell"        # Property added to particles_set

    # Init ---------------------------------------------------------------------

    def __init__(self, pset,
                       h        = None,
                       point1   = None,
                       point2   = None,
                       dtype    = np.float64):

        self.__dtype    = dtype

        if h        != None:
            self.__h        = h
        if point1   != None:
            self.__point1   = np.array(point1, self.__dtype)
        if point2   != None:
            self.__point2   = np.array(point2, self.__dtype)

        self.__dtype = dtype

        # particles_set has 3D property 'cell'. Each particle is placed inside a cell in the grid mesh.
        pset.add_property_by_name(property_name = self.__cell, dim = 3, to_type = np.int64)

    # Get and Set --------------------------------------------------------------

    def geth(self   ):
        return self.__h
    def seth(self, h):
        self.__h = h
    h = property(geth, seth , doc="Manipulates smothing length" )

    #-----------------------------------

    def getpoint1(self        ):
        return self.__point1
    def setpoint1(self, point1):
        self.__point1 = point1
    point1 = property(getpoint1, setpoint1 , doc="Point1 is the the first corner of the volume that contais all the particles" )

    #-----------------------------------

    def getpoint2(self        ):
        return self.__point2
    def setpoint2(self, point2):
        self.__point2 = point2
    point2 = property(getpoint2, setpoint2 , doc="Point2 is the the second corner of the volume that contais all the particles" )

    '''
    Methods --------------------------------------------------------------------
        calc_mesh
        calc_particle_mesh_location
    '''

    def calc_mesh(self, h         = None,
                        point1    = None,
                        point2    = None,
                        dtype     = np.float64):
        L = self.__INI_FLOAT * np.ones((3), dtype=dtype)


        if h        == None:
            h       = self.__h
        else:
            h       = h

        if point1   == None:
            point1  = np.array(self.__point1, self.__dtype)
        else:
            point1  = self.__point1

        if point2   == None:
            point2  = np.array(self.__point2, self.__dtype)
        else:
            point2  = self.__point2

        L               = point2 - point1

        self.__ar_axis0     = np.arange(-h, L[0]+2*h, h, dtype=self.__dtype)
        self.__ar_axis1     = np.arange(-h, L[1]+2*h, h, dtype=self.__dtype)
        self.__ar_axis2     = np.arange(-h, L[2]+2*h, h, dtype=self.__dtype)

    #-----------------------------------

    def calc_particles_mesh_locations(self, pset, dtype = np.float64):
        '''
        Finds the mesh cell where the each particle is located
        '''

        ar_axis0        = self.__ar_axis0
        ar_axis1        = self.__ar_axis1
        ar_axis2        = self.__ar_axis2
        point1          = self.__point1
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
                if (X[0] >= point1[0]+ar_axis0[x  ] and     # This if is to guarantee that each particle is assign just to 1 cell
                    X[0] <= point1[0]+ar_axis0[x+1]):
                        break
                else:
                    cell0 = cell0+1

            for x in range(0, max1):
                if (X[1] >= point1[1]+ar_axis1[x  ] and
                    X[1] <= point1[1]+ar_axis1[x+1]):
                    break
                else:
                    cell1 = cell1+1

            for x in range(0, max2):
                if (X[2] >= point1[2]+ar_axis2[x  ] and
                    X[2] <= point1[2]+ar_axis2[x+1]):
                        break
                else:
                    cell2 = cell2+1

            cell[0] = cell0
            cell[1] = cell1
            cell[2] = cell2

            particle_cell.append(cell)
        particle_cell=np.asarray(particle_cell)

        pset.get_by_name(self.__cell)[:] = particle_cell[:]
