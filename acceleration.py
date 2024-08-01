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

def write_colvar(filename, colvar_data, *column_names, output_fmt="%15.9f"):
    header = write_header(*column_names)
    colvar_data_ordered = [colvar_data[var] for var in column_names]
    np.savetxt(filename,  np.column_stack(colvar_data_ordered), fmt=output_fmt, header=header,comments='')

def calc_acceleration(colvar_data, timename, biasname, kT, timestep, time_unit, new_time_unit):
    beta = 1/kT
    bias = colvar_data[biasname]
    time = colvar_data[timename]
    if(time[0]==0.0):
        rescaled_time=np.zeros(len(time))
        rescaled_time[0]=time[0]
        rescaled_time[1:] = np.cumsum(np.exp(beta*bias[1:])*timestep*time_unit/new_time_unit)
        acceleration=np.zeros(len(time))
        acceleration[0]=np.exp(beta*bias[0])
        acceleration[1:] = np.divide (np.cumsum(np.exp(beta*bias[1:])),np.arange(1,len(bias[1:])+1))
    else:
        rescaled_time = np.cumsum(np.exp(beta*bias)*timestep*time_unit/new_time_unit)
        acceleration = np.divide (np.cumsum(np.exp(beta*bias)),np.arange(1,len(bias)+1))
    colvar_data["rescaled_time"] = rescaled_time
    colvar_data["acceleration"] = acceleration




if __name__ == "__main__":
    import argparse
    import json
    import os
    
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('-json_file')
     
    parser = argparse.ArgumentParser(parents=[config_parser], conflict_handler='resolve')
    parser.add_argument('-input_colvarfile',default='COLVAR', help='Colvar file name')
    parser.add_argument('-output_colvarfile',default='COLVAR_acceleration', help='Output colvar file name')
    parser.add_argument('-wd',default=['./'], help='Working directory', nargs='*')
    parser.add_argument('-time_column',default='time', help='Name of the time column')
    parser.add_argument('-bias_column',default='mtd.bias', help='Name of the bias column')
    parser.add_argument('-variables',default=['var'], help='Variables we need to distinguish between states', nargs='*')
    parser.add_argument('-kT',default=2.494339, type=float)
    parser.add_argument('-timestep',default=0.1, help='Timestep used in the colvar file')
    parser.add_argument('-time_unit',default="ps", help='Time units used in the colvar file')
    parser.add_argument('-new_time_unit',default= "ns", help='Time units for rescaled time')
    parser.add_argument('-output_format',default= "%15.9f", help='Format of the output colvar file. Default is %15.9f')
    parser.add_argument('--write_json',action='store_true', help='Write json file and exit')
    

    args, left_argv = config_parser.parse_known_args()
    
    if args.json_file is not None:
        json_dict = json.load(open(args.json_file))
        vars(args).update(json_dict)
    
    parser.parse_args(left_argv, args)

    if args.write_json:
        with open('template.json', 'w') as fp:
            template_input = vars(args)
            template_input.pop('write_json', None)
            template_input.pop('json_file', None)
            json.dump(template_input,fp, indent=2)
    else:
        time_units_dict = {'ps':1e-12, 'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1 }
        for wd in args.wd:
            print("Processing {}".format(os.path.join(wd, args.input_colvarfile)))
            old_dir = os.getcwd()
            os.chdir(wd)
            colvar_data = read_colvar(args.input_colvarfile, *args.variables, args.time_column, args.bias_column)
            calc_acceleration(colvar_data, timename=args.time_column, biasname=args.bias_column, kT=args.kT,timestep=args.timestep, time_unit=time_units_dict[args.time_unit], new_time_unit=time_units_dict[args.new_time_unit])
            columns_to_write = [args.time_column, *args.variables, args.bias_column]
            for key in colvar_data:
                if key not in columns_to_write:
                    columns_to_write.append(key)
            
            write_colvar(args.output_colvarfile, colvar_data, *columns_to_write, output_fmt=args.output_format)
            os.chdir(old_dir)
