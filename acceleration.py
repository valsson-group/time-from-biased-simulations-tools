#!/usr/bin/env python

import numpy as np
 
def read_colvar_header(filename, *column_names):
    variables = dict.fromkeys(column_names, 0)
    with open(filename) as f:
        head = next(f).strip().split() 
    for k in variables:
        variables[k] +=head.index(k)-2
    return variables

def read_colvar(filename, *column_names):
    column_numbers = read_colvar_header(filename, *column_names)
    colvar_data = dict.fromkeys(column_numbers.keys(),None)
    for k, v in column_numbers.items():
        colvar_data[k] = np.loadtxt(filename, comments="#!",usecols=v)
    return colvar_data

def write_header(*column_names):
    temp_string = ' '.join(column_names)
    header = "#! FIELDS "+temp_string
    return header

def write_colvar(filename, colvar_data, *column_names):
    header = write_header(*column_names)
    colvar_data_ordered = [colvar_data[var] for var in column_names]
    np.savetxt(filename,  np.column_stack(colvar_data_ordered), fmt='%8.6f', header=header,comments='')

def calc_acceleration(colvar_data, biasname,kT,timestep, time_unit, new_time_unit ):
    beta = 1/kT
    bias = colvar_data[biasname]
    rescaled_time = np.cumsum(np.exp(beta*bias)*timestep*time_unit/new_time_unit)
    acceleration = np.divide (np.cumsum(np.exp(beta*bias)),np.arange(1,len(bias)+1))
    colvar_data["rescaled_time"] = rescaled_time
    colvar_data["acceleration"] = acceleration




if __name__ == "__main__":
    import argparse
    import json
    import os
    from types import SimpleNamespace
    
    default_parameters = {'input_colvarfile': 'COLVAR', 
                          'output_colvarfile': 'COLVAR_acceleration', 
                          'wd': './', 
                          "time_column": "time",
                          "bias_column": "mtd.bias",
                          "variables": ["var"],
                          "kT": 2.494339,
                          "timestep": 1,
                          "time_unit":"ps",
                          "new_time_unit": "ps"}
    
    
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('-json_file')
     
    parser = argparse.ArgumentParser(parents=[config_parser], conflict_handler='resolve')
    parser.add_argument('-input_colvarfile', help='Colvar file name')
    parser.add_argument('-output_colvarfile', help='Output colvar file name')
    parser.add_argument('-wd', help='Working directory')
    parser.add_argument('-time_column', help='Name of the time column')
    parser.add_argument('-bias_column', help='Name of the bias column')
    parser.add_argument('-variables', help='Variables we need to distinguish between states')
    parser.add_argument('-kT', type=float)
    parser.add_argument('-timestep', help='Timestep used in the colvar file')
    parser.add_argument('-time_unit', help='Time units used in the colvar file')
    parser.add_argument('-new_time_unit', help='Time units for rescaled time')
    
    
    args, left_argv = config_parser.parse_known_args()
    if args.json_file is not None:
        json_dict = json.load(open(args.json_file))
        vars(args).update(json_dict)
    
    parser.parse_args(left_argv, args)
    default_parameters.update(vars(args))
    inp = SimpleNamespace(**default_parameters)
    
    time_units_dict = {'ps':10e-12, 'ns': 10e-9, 'us': 10e-6, 'ms': 10e-3, 's': 1 }

    os.chdir(inp.wd)
    colvar_data = read_colvar(inp.input_colvarfile, *inp.variables, inp.time_column, inp.bias_column)
    calc_acceleration(colvar_data,biasname=inp.bias_column, kT=inp.kT,timestep=inp.timestep, time_unit=time_units_dict[inp.time_unit], new_time_unit=time_units_dict[inp.new_time_unit])
    columns_to_write = [inp.time_column, *inp.variables, inp.bias_column]
    for key in colvar_data:
        if key not in columns_to_write:
            columns_to_write.append(key)
    
    write_colvar(inp.output_colvarfile, colvar_data, *columns_to_write)
