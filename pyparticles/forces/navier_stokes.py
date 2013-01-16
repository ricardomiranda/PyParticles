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

import sys
import numpy                          as        np
import pyparticles.pset.particles_set as        ps


class HPSSmothingKernels(object):
    """
    The assumed smoothing kernel for HPS is the W function according to Monaghan (1992).
    Stefan Auer, Realtime particle-based fluid simulation, thesis, (2008).

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

        # ----------------------------------------------------------------------

        Spiky kernel:
            W_{spiky}\left ( r,h \right )=
                \left\{\begin{matrix}
                    \frac{15}{\pi h^{6}} \left ( h-r \right )^{3} & , 0\leq r\leq h \\
                    0                        & , otherwise
                \end{matrix}\right

        Gradient of Spiky kernel:
            \triangledown  W_{spiky}\left ( r,h \right )=
                    -r \frac{45}{\pi h^{6} r} \left ( h-r \right )^{2}

        # ----------------------------------------------------------------------

        Viscosity kernel:
            W_{viscosity}\left ( r,h \right )=
                \left\{\begin{matrix}
                    \frac{15}{2\pi h^{3}} \left ( -\frac{r^{3}}{2h^{3}}+\frac{r^{2}}{h^{2}}+\frac{h}{2r}-1  \right ) & , 0\leq r\leq h \\
                    0                        & , otherwise
                \end{matrix}\right

        Gradient of Viscosity kernel:
            \triangledown  W_{viscosity}\left ( r,h \right )=
                    r \frac{15}{2 \pi h^{3}} \left ( -\frac{3r}{2h^{3}}+\frac{2}{h^{2}}-\frac{h}{2r^{3}} \right )

        Laplacian of Viscosity kernel:
            \triangledown \cdot \triangledown W_{viscosity}\left ( r,h \right ) =
                     \frac{45}{\pi h^{5}} \left ( 1-\frac{r}{h} \right )

        # ----------------------------------------------------------------------

        Smothing length:
            \begin{matrix}
                h \propto \frac{1}{\left \langle \rho  \right \rangle ^{1/\nu }} & ,where \left \langle \rho  \right \rangle = \frac{1}{n} \sum_{j} \rho _{j}
            \end{matrix}
    """

    __INI_FLOAT = -9999.9 # To catch calculation error it is useful because sometimes zero is a value that makes sense

    # Columns of the __smothing_kernels 2D array
    __sk_col_i          =  0
    __sk_col_j          =  1
    __sk_col_w_p6       =  2
    __sk_col_w_p6_grd   =  3
    __sk_col_w_p6_lpl   =  4
    __sk_col_w_sp       =  5
    __sk_col_w_sp_grd   =  6
    __sk_col_w_vc       =  7
    __sk_col_w_vc_grd   =  8
    __sk_col_w_vc_lpl   =  9

    __smothing_kernels  = None

    # Init ---------------------------------------------------------------------

    def __init__(self, h                 = None,
                       alpha             = None):
        if h is None:
            self.__h = 0.012 # For liquid water=0.012 m, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)
        else:
            self.__h = h

        if alpha is None:
            self.__alpha = 0.5 # For liquid water, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)
        else:
            self.__alpha = alpha


    # Get and Set --------------------------------------------------------------

    def getalpha(self       ):
        return self.__alpha
    def setalpha(self, alpha):
        self.__alpha = alpha
    alpha = property(getalpha, setalpha , doc="Manipulates alpha" )

    #-----------------------------------

    def geth(self   ):
        return self.__h
    def seth(self, h):
        self.__h = h
    h = property(geth, seth , doc="Manipulates smothing length" )

    '''
    Methods --------------------------------------------------------------------
        calc_smothing_kernels
            __w_poly6
            __w_poly6_gradiend
            __w_poly6_laplace
            __w_spiky
            __w_spiky_gradiend
            __w_viscosity
            __w_viscosity_gradiend
            __w_viscosity_laplace
    '''

    def __w_poly6(self, r):
        h    = self.__h
        pi   = np.pi

        w_p6 = self.__INI_FLOAT
        if   0 <= r and r <= h:
            w_p6 = 315.0/(64.0*pi* np.power(h, 9)) * np.power(np.power(h, 2) - np.power(r, 2), 3)
            print 'r=',r,'h=',h,'w_p6=',w_p6
        elif r >  h:
            w_p6 = 0.0
        else:
            sys.exit(self, msg="navier_stokes.HPSSmothingKernels.w_poly6")

        return w_p6

    #-----------------------------------

    def __w_poly6_gradiend(self, r):
        h        = self.__h
        pi       = np.pi

        w_p6_grd = self.__INI_FLOAT
        if   0 <= r and r <= h:
            w_6_grd = -r * (945.0/(32.0*pi* np.power(h, 9))) * np.power(np.power(h, 2) - np.power(r, 2), 2)
        elif r >  h:
            w_6_grd = 0.0
        else:
            sys.exit(self, msg="navier_stokes.HPSSmothingKernels.w_poly6_gradiend")

        return w_p6_grd

    #-----------------------------------

    def __w_poly6_laplace(self, r):
        h        = self.__h
        pi       = np.pi

        w_p6_lpl = self.__INI_FLOAT
        if   0 <= r and r <= h:
            w_p6_lpl = ((945.0/(8.0*pi* np.power(h, 9))) * (np.power(h, 2) - np.power(r, 2)) *
                            (np.power(r, 2) - 3.0/4.0 * (np.power(h, 2) - np.power(r, 2))))
        elif r >  h:
            w_p6_lpl = 0.0
        else:
            sys.exit(self, msg="navier_stokes.HPSSmothingKernels.w_poly6_laplace")

        return w_p6_lpl

    #-----------------------------------

    def __w_spiky(self, r):
        h    = self.__h
        pi       = np.pi

        w_sp = self.__INI_FLOAT
        if   0 <= r and r <= h:
            w_sp = 15.0/(pi* np.power(h, 6)) * (np.power(h-r, 3))
        elif r >  h:
            w_sp = 0.0
        else:
            sys.exit(self, msg="navier_stokes.HPSSmothingKernels.w_spiky")

        return w_sp

    #-----------------------------------

    def __w_spiky_gradiend(self, r):
        h        = self.__h
        pi       = np.pi

        w_sp_grd = self.__INI_FLOAT
        if   0 <= r and r <= h:
            w_sp_grd = -r * (45.0/(r*pi* np.power(h, 6))) * (np.power(h-r, 2)) # RCM, verify if it is correct
        elif r >  h:
            w_sp_grd = 0.0
        else:
            sys.exit(self, msg="navier_stokes.HPSSmothingKernels.w_spiky_gradiend")

        return w_sp_grd

    #-----------------------------------

    def __w_viscosity(self, r):
        h    = self.__h
        pi       = np.pi

        w_vc = self.__INI_FLOAT
        if   0 < r and r <= h:
            a    = np.power(r, 3) / (2* np.power(h, 3))
            b    = np.power(r, 2) / np.power(h, 2)
            c    = h / (2.0*r)
            w_vc = 15.0/(2.0*pi* np.power(h, 3)) * (-a+b+c-1.0)
        elif r >  h:
            w_vc = 0.0
        elif r == 0.0:
            w_vc = 0.0
        else:
            sys.exit(self, msg="navier_stokes.HPSSmothingKernels.w_viscosity")

        return w_vc

    #-----------------------------------

    def __w_viscosity_gradiend(self, r):
        h        = self.__h
        pi       = np.pi

        w_vc_grd = self.__INI_FLOAT
        if   0 <= r and r <= h:
            a        = 3.0*r/ (2.0* np.power(h, 3))
            b        = 2.0/ np.power(h, 2)
            c        = h/(2.0* np.power(r, 3))
            w_vc_grd = r* 15.0/(2.0*pi* np.power(h, 3)) * (-a+b-c)
        elif r >  h:
            w_vc_grd = 0.0
        else:
            sys.exit(self, msg="navier_stokes.HPSSmothingKernels.w_viscosity_gradiend")

        return w_vc_grd

    #-----------------------------------

    def __w_viscosity_laplace(self, r):
        h        = self.__h
        pi       = np.pi

        w_vc_lpl = self.__INI_FLOAT
        if   0 <= r and r <= h:
            a        = r/h
            w_vc_lpl = 45.0/(pi* np.power(h, 5) * (1.0-a))
        elif r >  h:
            w_vc_lpl = 0.0
        else:
            sys.exit(self, msg="navier_stokes.HPSSmothingKernels.w_viscosity_laplace")

        return w_vc_lpl

    #-----------------------------------

    def calc_smothing_kernels(self, r, particle_i, particle_j, dtype=np.float64):
        '''
        r is the distance between particles 'i' and 'j'
        '''
        ij_smothing_kernels = self.__INI_FLOAT * np.ones((10), dtype=dtype)

        ij_smothing_kernels[self.__sk_col_i          ] = particle_i
        ij_smothing_kernels[self.__sk_col_j          ] = particle_j
        ij_smothing_kernels[self.__sk_col_w_p6       ] = self.__w_poly6              (r=r)
        ij_smothing_kernels[self.__sk_col_w_p6_grd   ] = self.__w_poly6_gradiend     (r=r)
        ij_smothing_kernels[self.__sk_col_w_p6_lpl   ] = self.__w_poly6_laplace      (r=r)
        ij_smothing_kernels[self.__sk_col_w_sp       ] = self.__w_spiky              (r=r)
        ij_smothing_kernels[self.__sk_col_w_sp_grd   ] = self.__w_spiky_gradiend     (r=r)
        ij_smothing_kernels[self.__sk_col_w_vc       ] = self.__w_viscosity          (r=r)
        ij_smothing_kernels[self.__sk_col_w_vc_grd   ] = self.__w_viscosity_gradiend (r=r)
        ij_smothing_kernels[self.__sk_col_w_vc_lpl   ] = self.__w_viscosity_laplace  (r=r)

        '''
        A NumPy array is a very different data structure from a list and is designed to be used in different ways. Your
        use of hstack is potentially very inefficient... every time you call it, all the data in the existing array is
        copied into a new one. (The append function will have the same issue.) If you want to build up your matrix one
        column at a time, you might be best off to keep it in a list until it is finished, and only then convert it into
        an array.

        mylist = []
        for item in data:
            mylist.append(item)
        mat = numpy.array(mylist)

        item can be a list, an array or any iterable, as long as each item has the same number of elements.
        '''

        if self.__smothing_kernels == None:
            self.__smothing_kernels = ij_smothing_kernels
        else:
            self.__smothing_kernels = np.append(self.__smothing_kernels, ij_smothing_kernels, axis = 1)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class HPSNavierStokes(object):

    __pressure = "pressure"        # Property added to particles_set

    # Init ---------------------------------------------------------------------

    def __init__(self, pset,
                       reference_density = None,
                       coefficient       = None,
                       gamma             = None):
        if reference_density is None:
            self.__reference_density = 1000.0 # For liquid water
        else:
            self.__reference_density = reference_density

        if coefficient is None:
            self.__coefficient = 16.0 # For liquid water, 10<coefficient<40, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)
        else:
            self.__coefficient = coefficient

        if gamma is None:
            self.__gamma = 7.0 # For liquid water, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)
        else:
            self.__gamma = gamma

        self.__gravity_acceleration =-9.8

        # particles_set has 1D property 'pressure'. Each particle has pressure.
        pset.add_property_by_name(property_name = self.__pressure, dim = 1, to_type = np.float64)

    # Get and Set --------------------------------------------------------------

    def getcoefficient(self             ):
        return self.__coefficient
    def setcoefficient(self, coefficient):
        self.__coefficient = coefficient
    coefficient = property(getcoefficient, setcoefficient , doc="Manipulates coefficient" )

    #-----------------------------------

    def getgamma(self      ):
        return self.__gamma
    def setgamma(self, gamma):
        self.__gamma = gamma
    gamma = property(getgamma, setgamma , doc="Manipulates gamma" )

    #-----------------------------------

    def getreference_density(self                   ):
        return self.__reference_density
    def setreference_density(self, reference_density):
        self.__reference_density = reference_density
    reference_density = property(getreference_density, setreference_density , doc="Manipulates the reference density" )

    #-----------------------------------

    def getgravity_acceleration(self                      ):
        return self.__gravity_acceleration
    def setgravity_acceleration(self, gravity_acceleration):
        self.__gravity_acceleration = gravity_acceleration
    gravity_acceleration = property(getgravity_acceleration, setgravity_acceleration , doc="Manipulates the gravity acceleration" )

    '''
    Methods --------------------------------------------------------------------
        calc_pressure
        calc_density_water_rest
            __calc_B
    '''

    def __calc_B(self, h_SWL):
        """
        The particles are assigned a pressure.
        For liquid water, SPHysics

        Parameters

        :param    h_SWL:        Mximum depth in the simulation
        """

        gamma = self.__gamma
        rd    = self.__reference_density
        coef2 = self.__coefficient**2
        return coef2*rd*h_SWL/gamma

    #-----------------------------------

    def calc_pressure(self, pset, H):
        """
        The particles are assigned a pressure.
        For liquid water, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)

        .. math::
            P = B\left [ \left ( \frac{\rho }{\rho _{0}} \right )^{\gamma } -1 \right ]

        Parameters

        :param    H:            Maximum water column height
        """
        gamma       = self.__gamma
        rd          = self.__reference_density
        B           = self.__calc_B(h_SWL=H)
        pressure    = self.__pressure

        pset.get_by_name(pressure)[:] = B * (np.power((pset.D[:]/rd), gamma) - 1)

    #-----------------------------------

    def calc_density_water_rest(self, pset, H):
        """
        The particles are assigned an initial density based on hydrostatic pressure.
        For liquid water, incompressible flow, Alejandro Jacobo Cabrera Crespo (2008)

        .. math::
            \rho \left ( z \right ) = \rho _{0} \left ( 1+ \frac{\rho _{0} g \left ( H-z \right )}{B} \right )^{1/ \gamma }

        Parameters

        :param    H:            Maximum water column heigth
        """
        rd          = self.__reference_density
        g           = self.__gravity_acceleration
        gamma       = self.__gamma
        B           = self.__calc_B(h_SWL=H)
        pset.D[:]   = rd * np.power((1 + (rd*np.abs(g)*(H-pset.X[:,2,np.newaxis]))/B), 1./gamma)





