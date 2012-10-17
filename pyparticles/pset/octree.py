# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva simone.rva {at} gmail {dot} com
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

import pyparticles.pset.particles_set as ps

import numpy as np

class TreeElement( object ):
    """
    The element of a tree node
    """
    def __init__( self ):
        
        self.__particle = None
        
        self.__ref_vertex = np.zeros((3))
        self.__edge_len = 0.0
        
        self.__M = 0.0
        self.__centre_of_mass = np.zeros((3))
        
        self.__part_cnt = 0 
        
        self.__tree = None
        
        
      

    def get_particle( self ) :
        return self.__particle
    

        
    particle = property( get_particle , doc="Get the particle index of the tree element" )


    def getM(self):
        return self.__M
    
    M = property( getM , doc="getter: return the total mass of the current element ")


    def get_centre_of_mass( self ):
        return self.__centre_of_mass
    
    centre_of_mass = property( get_centre_of_mass , doc="return the centre of mass of the particle" )
    

    def set_local_boundary( self , ref_vertex , edge_len ) :
        self.__ref_vertex[:] = ref_vertex
        self.__edge_len = edge_len


    def add_sub_trees( self ):        
        self.__tree = list( TreeElement() for i in range(8) )
        
        ## I - I
        self.__tree[0].set_local_boundary( self.__ref_vertex , self.__edge_len / 2.0  )
        
        d = np.array([ self.right[0] , 0.0 , 0.0 ]) * self.__edge_len
        self.__tree[1].set_local_boundary( self.__ref_vertex + d , self.__edge_len / 2.0 )
            
        d = np.array([ self.right[0] , self.far[0] , 0.0 ]) * self.__edge_len
        self.__tree[2].set_local_boundary( self.__ref_vertex + d , self.__edge_len / 2.0 )
        
        d = np.array([ self.left[0] , self.far[0] , 0.0 ]) * self.__edge_len
        self.__tree[3].set_local_boundary( self.__ref_vertex + d , self.__edge_len / 2.0 ) 
        
        ## II
        d = np.array([ self.left[0] , self.near[0] , self.up[0] ]) * self.__edge_len
        self.__tree[4].set_local_boundary( self.__ref_vertex + d , self.__edge_len / 2.0 )
        
        d = np.array([ self.right[0] , self.near[0] , self.up[0] ]) * self.__edge_len
        self.__tree[5].set_local_boundary( self.__ref_vertex + d , self.__edge_len / 2.0 )
        
        d = np.array([ self.right[0] , self.far[0] , self.up[0] ]) * self.__edge_len
        self.__tree[6].set_local_boundary( self.__ref_vertex + d , self.__edge_len / 2.0 )
        
        d = np.array([ self.left[0] , self.far[0] , self.up[0] ]) * self.__edge_len
        self.__tree[7].set_local_boundary( self.__ref_vertex + d , self.__edge_len / 2.0 )        
        
        
    def is_in( self , coord ):
        a = coord >= self.__ref_vertex
        b = coord < ( self.__ref_vertex + self.__edge_len )

        return np.all( np.logical_and( a , b ) )
    
    def insert_particle( self , pset , i ) :
        if self.particle == None and self.__tree == None :
            self.__particle = i
            self.__M = pset.M[i]
            self.__centre_of_mass = pset.X[i,:]
            self.__part_cnt = 1
            return
        
        elif self.particle != None and self.__tree == None :
            self.add_sub_trees()
            
        elif self.particle != None and self.__tree != None :
            pass ## it's only for testing the coherence of the tree structure
        
        else :
            raise 
        
        # update mass
        self.__M += pset.M[i]
            
        # update centre of mass
        self.__centre_of_mass = ( self.__centre_of_mass * self.__part_cnt + pset.X[i,:] ) / ( self.__part_cnt + 1 )
        self.__part_cnt += 1
        for st in self.__tree :
            if st.is_in( pset.X[i,:] ) :
                st.insert_particle( pset , i )
                break
        return
                

    def print_tree( self , pset , d=1 ):
        """
        Print the structure of the tree and return the maximal depth
        Args:
            # pset: the particles set
            # d = current depth (1 for the first node)
        """
        if self.__particle == None  :
            return 
        
        vrtx  = tuple(i for i in self.__ref_vertex)
        cm    = tuple(i for i in self.__centre_of_mass)
        coord = tuple(i for i in pset.X[self.__particle,:] )
        
        spc = " +"
        
        print( "" )
        print( "%s -------------------- " % ( spc*d ) )
        print( "%s depth :         %d " % ( spc*d , d ) )
        print( "%s part indx:      %d " % ( spc*d , self.__particle ) )
        print( "%s part coord:     %s " % ( spc*d , coord ) )
        print( "%s vertex:         %s " % ( spc*d , vrtx ) )
        print( "%s edge:           %s " % ( spc*d , self.__edge_len ) )
        print( "%s centre of mass: %s " % ( spc*d , cm ) )
        
        mx = d
        if self.__tree != None:
            for tr in self.__tree :
                mx = max ( tr.print_tree( pset , d+1 ) , mx )
            
        return mx
        

    def depth( self , d = 1 ):
        mx = d
        if self.__tree != None:
            for tr in self.__tree :
                mx = max ( tr.print_tree( pset , d+1 ) , mx )
            
        return mx
    

    def get_up(self):
        return ( 0.5 , 1.0 )
    
    def get_down(self):
        return ( 0.0 , 0.5 )
    
    def get_left(self):
        return ( 0.0 , 0.5 )
    
    def get_right(self):
        return ( 0.5 , 1.0 )
    
    def get_near(self):
        return ( 0.0 , 0.5 )
    
    def get_far(self):
        return ( 0.5 , 1.0 )
        
    up    = property( get_up    , doc="relative domanin for the up 'sub-space' " )
    down  = property( get_down  , doc="relative domanin for the down 'sub-space' " )
    left  = property( get_left  , doc="relative domanin for the left 'sub-space' " )
    right = property( get_right , doc="relative domanin for the right 'sub-space' " )
    near  = property( get_near  , doc="relative domanin for the near 'sub-space' " )
    far   = property( get_far   , doc="relative domanin for the far 'sub-space' " )            



class OcTree ( object ):
    """
    OcTree particles container class
    """
    def __init__( self ):
        self.__tree = None
        self.__pset = None
        self.__ref_vertex = np.zeros((3))
        self.__edge_len = 1.0
    
    def set_global_boundary( self ,
                            ref_vertex = np.array([ 0.0 , 0.0 , 0.0 ]) ,
                            edge_len   = 1.0 ) :
        
        """
        Define the size of the octree cube
        Arguments:
            ref_vertex : the ( down , near , left ) vertex
            edge_len   : leght of the edge 
        """
        
        self.__ref_vertex[:] = ref_vertex
        self.__edge_len = edge_len
      
        
    def get_centre_of_mass( self ):
        return self.__tree.centre_of_mass
        
    centre_of_mass = property( get_centre_of_mass , doc="return the centre of mass of the particles set" )
    
    def build_tree( self , pset ) :
        """
        Build the octree with the given particles set
            Arguments:
            pset: a ParticlesSet object used to build the tree
        """
        
        if self.__tree == None :
            self.__tree = TreeElement()
        else :
            del self.__tree
            self.__tree = TreeElement()
        
        self.__tree.set_local_boundary( self.__ref_vertex , self.__edge_len )
        
        self.__pset = pset
        
        for i in range( pset.size ):
            self.__tree.insert_particle( pset , i )
    
        
    def print_tree( self ):
        mx = self.__tree.print_tree( self.__pset , 1 )
        
        print("")
        print("Max depth: %d " % mx )
        print("")
        
    