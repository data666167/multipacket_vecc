# system imports
import io
import time
import datetime
import os
from os.path import abspath, join, dirname, basename
import sys
import cProfile
import pstats

# third party import
import numpy as np
import json


# import the path to the package
project_dir = '/home/oz_computron/Desktop/Research_idea/multipacket_VECC/t-amplitudes'
sys.path.insert(0, project_dir)

# local project directory packages 
import project
from project.vibronic_hamiltonian import vibronic_hamiltonian
from project.vibronic import vIO, VMK, vIO_wrapper
# breakpoint()

# initialize model 
def get_model_from_json_file(path, order):
    """ 
    path: string for json file location 
    order: int for the order of the hamiltonian used 
    """

    model = vIO.load_model_from_JSON(path)

    A, N = vIO._extract_dimensions_from_dictionary(model)
    

    #     # This will handle JSONS which match the coefficients of .op file which are written in the upper triangular format
    #     # You will need to talk to neil about trade offs for coefficient lengths  
    # if mctdh_comp := True:
    #     temp_g2 = model[VMK.G2].reshape(A,A,N,N)
    #     temp_g2 = symmetrize_upper_triangular(temp_g2)
    #     model[VMK.G2] = temp_g2.reshape(np.shape(model[VMK.G2]))
    #     # breakpoint()
    #     temp_g3 = model[VMK.G3].reshape(A,A,N,N,N)
    #     temp_g3 = symmetrize_upper_triangular(temp_g3)
    #     model[VMK.G3] = temp_g3.reshape(np.shape(model[VMK.G3]))
        
    #     temp_g4 = model[VMK.G4].reshape(A,A,N,N,N,N)
    #     temp_g4 = symmetrize_upper_triangular(temp_g4)
    #     model[VMK.G4] = temp_g4.reshape(np.shape(model[VMK.G4]))
                  
                  
    # probably more relevant later
    model[VMK.etdm] = np.array([[0.1+0.0j,0.1+0.0j]])
    model[VMK.mtdm] = np.array([[0.1+0.0j,0.1+0.0j]])

    # swap electron / vibrational dimensions
    vIO.prepare_model_for_cc_integration(model, order)
    
    return model

# get_model_from_op_file
def construct_hamiltonian(model_filename,model_path,model_order):
    # JSON only can probably read json 
    model_json_path = join(model_path,model_filename)
    
    model_name = basename(model_json_path) 
    model_name = os.path.splitext(model_name)[0] # splits path to text take the model name 0th element to retrieve name not ext   
    model = get_model_from_json_file(model_json_path, model_order)
    
    return(model,model_name)

def vecc_propagation(model,model_name,output_path,prop_length=10,h_order=1,z_order=2):
    # output path should be like like test or debug and named what kind of component is being debugged  

    # run model create ACF VECC
    ## initialize object 
    VECC_obj = vibronic_hamiltonian(model=model,
                                    model_name=model_name,
                                    cc_truncation_order=1,
                                    T_truncation_order=1,
                                    calculate_population_flag=False,
                                    hamiltonian_truncation_order=h_order,
                                    Z_truncation_order=z_order) 
    ## Construct acf for object 
    VECC_obj.rk45_integration(t_init=0,t_final=prop_length)

    # Construct output directory
    time_now = datetime.datetime.now()
    time_str = str(time_now.strftime("%Y_%m_%d__%H_%M_%S"))
    output_dir = join(output_path,time_str)    
    os.mkdir(output_dir)
    
    # save ACF for object
    file_output_path = VECC_obj.save_acf_data(model_name,output_path=output_dir)[0]

    # interpolate the result 
    norm_acf_path = project.spectra.generate_normalized_acf_results(dirname(file_output_path),basename(file_output_path),None,prop_length*0.5,0.1)
    
    # prepare input document
    prop_info = VECC_obj.run_info()
    prop_info['prop_time(fs)'] = prop_length
    
    output_info_path =join(output_dir,"output_info.json")
    with open(output_info_path, "w") as f:
        json.dump(prop_info,f)
     
    return(norm_acf_path)


def spectra_maker(acf_path):
    # should run autospec 
    return

# Write a proper main at some point
model_file_dir = '/home/oz_computron/Desktop/Research_idea/multipacket_VECC/t-amplitudes/data/input_json'
# output dirs should have a user initialized folder for easy extensions 
outputdir =  '/home/oz_computron/Desktop/Research_idea/multipacket_VECC/t-amplitudes/multipacket_data/initial_tests'

model_ham,model_name = construct_hamiltonian('displaced_1.json',model_file_dir,1)
acf_path = vecc_propagation(model=model_ham,model_name=model_name,output_path=outputdir,prop_length=10,h_order=1,z_order=2)

breakpoint()

