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

import numpy                          as        np
import pyparticles.pset.particles_set as        ps


class HPSSmothingKernels(object):
    """
    The assumed smoothing kernel for HPS is the W function according to Monaghan (1992).

    These are the equations:

    .. math::
        Poly6 Kernel:
            W_{poly6}\left ( r,h \right )=
                \left\{\begin{matrix}
                    \frac{315}{64 \pi h^{9}} \left ( h^{2} - r^{2} \right )^{3} & , 0 \leq r \leq h \\
                    0                                                           & , otherwise
                \end{matrix}\right

        Gradient of Poly6 kernel:
            \triangledown W_{poly6}\left ( r,h \right ) =
                -r \frac{945}{32 \pi h^{9}} \left ( h^{2} - r^{2} \right )^{2}

        Laplacian of Poly6 kernel:
            \triangledown \cdot \triangledown W_{poly6}\left ( r,h \right ) =
                \frac{945}{8 \pi h^{9}} \left ( h^{2} - r^{2} \right )\left ( r^{2} -\frac{3}{4} \left ( h^{2} - r^{2} \right ) \right )


        Spiky kernel:
            W_{spiky}\left ( r,h \right )=
                \left\{\begin{matrix}
                    \frac{15}{\pi h^{6}} \left ( h-r \right )^{3} & , 0\leq r\leq h \\
                    0                        & , otherwise
                \end{matrix}\right

        Gradient of Spiky kernel:
           \triangledown  W_{spiky}\left ( r,h \right )=
                    -r \frac{45}{\pi h^{6} r} \left ( h-r \right )^{2}

        Smothing length:
            \begin{matrix}
                h \propto \frac{1}{\left \langle \rho  \right \rangle ^{1/\nu }} & ,where \left \langle \rho  \right \rangle = \frac{1}{n} \sum_{j} \rho _{j}
            \end{matrix}
    """

    # Init ---------------------------------------------------------------------

    def __init__(self, h                 = None,
                       alpha             = None):
        if h is None:
            self._h = 0.012 # For liquid water=0.012 m, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)
        else:
            self._h = h

        if alpha is None:
            self._alpha = 0.5 # For liquid water, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)
        else:
            self._alpha = alpha


    # Get and Set --------------------------------------------------------------

    def getalpha(self       ):
        return self._alpha
    def setalpha(self, alpha):
        self._alpha = alpha
    alpha = property(getalpha, setalpha , doc="Manipulates alpha" )


    def geth(self   ):
        return self._h
    def seth(self, h):
        self._h = h
    h = property(geth, seth , doc="Manipulates smothing length" )


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


class HPSNavierStokes(object):

    # Init ---------------------------------------------------------------------

    def __init__(self, reference_density = None,
                       coefficient       = None,
                       gamma             = None):
        if reference_density is None:
            self._reference_density = 1000.0 # For liquid water
        else:
            self._reference_density = reference_density


        if coefficient is None:
            self._coefficient = 16.0 # For liquid water, 10<coefficient<40, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)
        else:
            self._coefficient = coefficient


        if gamma is None:
            self._gamma = 7.0 # For liquid water, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)
        else:
            self._gamma = gamma


        self._gravity_acceleration =-9.8

    # Get and Set --------------------------------------------------------------

    def getcoefficient(self             ):
        return self._coefficient
    def setcoefficient(self, coefficient):
        self._coefficient = coefficient
    coefficient = property(getcoefficient, setcoefficient , doc="Manipulates coefficient" )


    def getgamma(self      ):
        return self._gamma
    def setgamma(self, gamma):
        self._gamma = gamma
    gamma = property(getgamma, setgamma , doc="Manipulates gamma" )


    def getreference_density(self                   ):
        return self._reference_density
    def setreference_density(self, reference_density):
        self._reference_density = reference_density
    reference_density = property(getreference_density, setreference_density , doc="Manipulates the refernce density" )


    def getgravity_acceleration(self                      ):
        return self._gravity_acceleration
    def setgravity_acceleration(self, gravity_acceleration):
        self._gravity_acceleration = gravity_acceleration
    gravity_acceleration = property(getgravity_acceleration, setgravity_acceleration , doc="Manipulates the gravity acceleration" )


    # Modifiers ----------------------------------------------------------------


    def calc_B(self, h_SWL):
        """
        The particles are assigned a pressure.
        For liquid water, SPHysics
        """
        # h_SWL is the maximum depth in the simulation
        gamma = self._gamma
        rd    = self._reference_density
        coef2 = self._coefficient**2
        return coef2*rd*h_SWL/gamma


    def calc_pressure(self, density):
        """
        The particles are assigned a pressure.
        For liquid water, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)

        .. math::
            P = B\left [ \left ( \frac{\rho }{\rho _{0}} \right )^{\gamma } -1 \right ]
        """
        gamma = self._gamma
        rd    = self._reference_density
        return B * ((density/reference_density)**gamma - 1)


    def calc_density_water_rest(self, p_set, H):
        """
        The particles are assigned an initial density based on hydrostatic pressure.
        For liquid water, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)

        .. math::
            \rho \left ( z \right ) = \rho _{0} \left ( 1+ \frac{\rho _{0} g \left ( H-z \right )}{B} \right )^{1/ \gamma }
        """
        rd    = self._reference_density
        g     = self._gravity_acceleration
        gamma = self._gamma
        B     = self.calc_B(H)
        p_set.D[:] = rd * np.power((1 + (rd*np.abs(g)*(H-p_set.X[:,2,np.newaxis]))/B), 1./gamma)



