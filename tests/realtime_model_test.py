# system imports
import io
import time
import datetime
import os
from os.path import abspath, join, dirname, basename
import sys
import cProfile
import pstats
import pandas as pd 

# third party import
import numpy as np
import scipy as scp
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
    model[VMK.etdm] = np.array([[0.1+0.0j,0.0+0.0j]])
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

def interpolate_population_data(dataframe,prop_length,stepsize,):

    # first col is time
    raw_time = dataframe.iloc[:,0]
    # create data frame for inteprolated data include desired time propagation array  
    new_time = np.arange(0,prop_length,stepsize)
    interpolated_data = pd.DataFrame({'time(fs)':new_time})
    
    # iterate over the non time columns
    for col_idx in range(1,(dataframe.columns.shape)[0]):
        # grab raw population index
        raw_population = dataframe.iloc[:,col_idx]
        # strip column name 
        population_name = raw_population.name
        
        # interpolate raw results for desired range 
        interpolate_func = scp.interpolate.CubicSpline(raw_time,raw_population)
        interpolated_population = interpolate_func(new_time)
        
        # add interpolated data to interpolated dataframe
        interpolated_data[population_name] = interpolated_population
        
    return interpolated_data

def vecc_propagation(model,model_name,output_path,prop_length=10,h_order=1,z_order=2,population_flag=False):
    # output path should be like like test or debug and named what kind of component is being debugged  

    # run model create ACF VECC
    ## initialize object 
    VECC_obj = vibronic_hamiltonian(model=model,
                                    model_name=model_name,
                                    cc_truncation_order=1,
                                    T_truncation_order=1,
                                    calculate_population_flag=population_flag,
                                    hamiltonian_truncation_order=h_order,
                                    Z_truncation_order=z_order) 
    ## Construct acf for object 
    VECC_obj.rk45_integration(t_init=0,t_final=prop_length)

    # Construct output directory
    time_now = datetime.datetime.now()
    time_str = str(time_now.strftime("%Y_%m_%d__%H_%M_%S"))
    output_dir = join(output_path,f'VECC_{time_str}')    
    os.mkdir(output_dir)
    
    # Save ACF for object
    file_output_path = VECC_obj.save_acf_data(model_name,output_path=output_dir)[0]
    # interpolate ACF result 
    norm_acf_path = project.spectra.generate_normalized_acf_results(dirname(file_output_path),basename(file_output_path),None,prop_length*0.5,0.1)
  
    if population_flag: 
        # Save diabat object
        raw_diabatic_vecc = VECC_obj.diabatic_populations_data
        interp_diabat_data = interpolate_population_data(raw_diabatic_vecc,prop_length,stepsize=0.2) # stepsize is hard coded since stepsize for mctdh makes no sense with their implementation
        # construct paths for diabat files
        raw_diabat_path = join(output_dir,f'pop_db_{model_name}.csv')
        interp_diabat_path = join(output_dir,f'pop_db_{model_name}_normalized.csv')
        # write files to output path 
        raw_diabatic_vecc.to_csv(raw_diabat_path, index=False)
        interp_diabat_data.to_csv(interp_diabat_path, index=False)
    
    # prepare json file detailing calculation inputs
    prop_info = VECC_obj.run_info()
    prop_info['prop_time(fs)'] = prop_length
    prop_info['population'] = population_flag 
    prop_info['model_name'] = model_name
    
    output_info_path =join(output_dir,"output_info.json")
    with open(output_info_path, "w") as f:
        json.dump(prop_info,f)
     
    return

def ed_propagation(model,model_name,output_path,prop_length=10,basis_size=10,population_flag=False):
    VECC_obj = vibronic_hamiltonian(model=model,
                                    model_name=model_name,
                                    cc_truncation_order=1,
                                    T_truncation_order=1,
                                    calculate_population_flag=population_flag,
                                    hamiltonian_truncation_order=1,
                                    Z_truncation_order=2)
    
    # Construct output directory
    time_now = datetime.datetime.now()
    time_str = str(time_now.strftime("%Y_%m_%d__%H_%M_%S"))
    output_dir = join(output_path,f'ED_{time_str}')    
    os.mkdir(output_dir)
    
    # sum over states deals with all saving within VECC_obj's class  
    VECC_obj.sum_over_states(output_path=output_dir,prop_length=prop_length,basis_size=basis_size)
    
    # prepare json file detailing calculation inputs
    prop_info = {}
    prop_info['calculation type'] = 'ED'
    prop_info['basis size'] = basis_size
    prop_info['prop_time(fs)'] = prop_length
    prop_info['population'] = population_flag 
    prop_info['model_name'] = model_name
    
    output_info_path =join(output_dir,"output_info.json")
    with open(output_info_path, "w") as f:
        json.dump(prop_info,f)
    return

# # Write a proper main at some point
model_file_dir = '/home/oz_computron/Desktop/Research_idea/multipacket_VECC/t-amplitudes/data/input_json'
# # output dirs should have a user initialized folder for easy extensions 
outputdir =  '/home/oz_computron/Desktop/Research_idea/multipacket_VECC/t-amplitudes/multipacket_data/initial_tests'

model_ham,model_name = construct_hamiltonian('displaced_1.json',model_file_dir,1)
breakpoint()
vecc_propagation(model=model_ham,model_name=model_name,output_path=outputdir,
                 prop_length=10,h_order=1,z_order=2,population_flag=True)

ed_propagation(model=model_ham,model_name=model_name,output_path=outputdir, 
               prop_length=10,basis_size=40,population_flag=True)

breakpoint()

