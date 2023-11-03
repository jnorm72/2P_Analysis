"""
Function to find the offset between two fields of view

@author: jnorm
"""
import numpy as np


def find_offset(fov_1,fov_2,fov_dims,dimensionality_factor):
    """

     - will only work if FOVs are of the same size
    
    inputs:
     - FOV 1 & FOV 2
     - Dimension of the FOV
     - Dimesnionality Factor (How much of each dimension to match minimum) [i.e. to match start with 50% offset use 0.5]
    
    output: 
     - Coordinate offset of top left corner of FOV 2 with respect to top left corner of FOV 1.
     """
    #%% initialize constants
    error_arr = [] #initialize array of error values
    dimensionality_factor = 1 - dimensionality_factor #take inverse of dimensionality for coordinate space building
     
     
    scalar_term_rows = int(np.ceil(fov_dims[0]*dimensionality_factor) )    #calculate scalar for row operations
    scalar_term_columns = int(np.ceil(fov_dims[1]*dimensionality_factor) ) #calculate scalar for column operations
         
    #%% initalize static coordinate space for scanning against
    fov_static = np.zeros ( (int(fov_dims[0]+2*scalar_term_rows),int(fov_dims[1]+2*scalar_term_columns) ) )                #preallocate array of zeros
    fov_static[scalar_term_rows:fov_dims[0]+scalar_term_rows,scalar_term_columns:fov_dims[1]+scalar_term_columns] = fov_1  #write data in middle of coordinate space
     
    #%% iterate across possible coordinate positions for second fov and calculate error
    for i in range (0,2*scalar_term_rows+1):  #iterate through rows of coordinate space
         
        for j in range (0,2*scalar_term_columns+1): #iterate through columns of coordinate space
                 
            fov_dynamic = np.zeros ( (int(fov_dims[0]+2*scalar_term_rows),int(fov_dims[1]+2*scalar_term_columns) ) ) #allocate array of zeros the size of coordinate space 
            fov_dynamic [i:i+fov_dims[0],j:j+fov_dims[1]] = fov_2                                                    #write fov data in current interative position of coordinate space
            
            fov_static_overlap = fov_static[(fov_dynamic > 0) & (fov_static > 0)]
            fov_dynamic_overlap = fov_dynamic[(fov_dynamic > 0) & (fov_static > 0)]
            norm_square_error = np.sum(np.square(np.subtract(fov_dynamic_overlap,fov_static_overlap) ) )/ np.size(fov_static_overlap) #np.sum(np.square(np.subtract(fov_dynamic,fov_static) ) )/ np.size(fov_1) #calculate error of alignment for current position
            error_arr.append(norm_square_error) #append calculated error to list
   
    #%% find coordinate offset where coordinate returned is fov_2[0 0] in respect to fov_1 [0 0]
    error_arr_loci = np.reshape(np.array(error_arr) , (2*scalar_term_rows+1,2*scalar_term_columns+1) ) #transform list to coordinate space array
    offset_index = np.concatenate( np.where(error_arr_loci == np.amin(error_arr_loci) ), axis=None )  #numpy conversion for ease of use
    offset_corr = (offset_index - np.array([scalar_term_rows,scalar_term_columns]) ) * -1 #calculate coordinate of offset 
    
    return offset_corr
