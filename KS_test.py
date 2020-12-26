from scipy.stats import expon
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
import math
import os

 
def plot_ecdf(outname, time_bins, ecdf, fitted_curve, theoretical_curve, p_value, rate, time_units):
    mpl.rcParams['grid.color'] = '#2C3043'
    mpl.rcParams['grid.alpha'] = 0.7
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['lines.markeredgewidth'] = 0.2

    fig = Figure(linewidth=4, figsize = (4.5,3), dpi =300)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=1, left =0.17, right=0.97, bottom=0.13, top=0.90,  wspace=0.2, hspace=0.2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_xscale('log')
    ax1.plot(time_bins, theoretical_curve, color='#e49b0f', linewidth=3, alpha=0.25)
    ax1.plot(time_bins, fitted_curve, color='#008000', linewidth=1.5, alpha = 0.35)
    ax1.plot(time_bins, ecdf, color='#c4201c', linewidth=0.7)

    ax1.set_xlabel('Time, {}'.format(time_units), fontsize=8)
    ax1.set_ylabel('ECDF', fontsize=8)
    ax1.text(1/rate/10000, 0.8, 'p-value = {:.2f}'.format(p_value), fontsize=10)
    ax1.text(1/rate/10000, 0.7, 'tau = {:.2f} {}'.format(1/rate, time_units), fontsize=10)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 

    fig.savefig(os.path.normpath('{}'.format(outname)))

def plot_cdf(outname, times, rate, time_units):


    mpl.rcParams['grid.color'] = '#2C3043'
    mpl.rcParams['grid.alpha'] = 0.7
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['lines.markeredgewidth'] = 0.2 
    
    fig = Figure(linewidth=4, figsize = (4.5,3), dpi =300)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=1, left =0.17, right=0.97, bottom=0.13, top=0.90,  wspace=0.2, hspace=0.2)
    ax1 = fig.add_subplot(gs[0,0])
    y, x = np.histogram(times, bins=int((len(times))**0.5))
    y = y/np.trapz(y, x=x[:-1])
    ax1.scatter(x[:-1], y, s=30, marker="o", facecolors='none', edgecolors='r', lw = 0.5)
    theoretical_pdf = expon.pdf(np.arange(np.amax(times)), loc=0, scale=1/rate)
    ax1.plot(np.arange(np.amax(times)), theoretical_pdf, color='#004953', linewidth=0.7)
    ax1.set_ylim(bottom=-0.05*max(y[0], theoretical_pdf[0]), top= max(y[0], theoretical_pdf[0])+0.1*max(y[0], theoretical_pdf[0]))
    ax1.set_xlabel('Time, {}'.format(time_units), fontsize=8)
    ax1.set_ylabel('$f_{{T1}}$(t) [1/{}]'.format(time_units), fontsize=8)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 

    fig.savefig(os.path.normpath('{}'.format(outname)))




if __name__ == "__main__":
    import argparse
    import json
    from scipy.stats import ks_2samp
    from scipy.optimize import curve_fit     
    

    
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('-json_file')
     
    parser = argparse.ArgumentParser(parents=[config_parser], conflict_handler='resolve')
    parser.add_argument('-data_file',default='data.dat', help='File with times')
    parser.add_argument('-statistics_file',default='statistics.dat', help='Output file for general statistics')
    parser.add_argument('-wd',default='./', help='Working directory')
    parser.add_argument('-out_file',default='out.dat', help='Output file for    ')
    parser.add_argument('-ECDF_plot',default='ECDF.png', help='ECDF pot')
    parser.add_argument('-CDF_plot',default='CDF.png', help='CDF plot')
    parser.add_argument('-range_factor',default=100, help='range factor for histogram')
    parser.add_argument('-time_unit',default= "ns", help='Time units used in the input file')
    parser.add_argument('-new_time_unit',default= "ns", help='Time units of tau')
    parser.add_argument('-nbins',default= 10000, help='bins number for histogram')
    parser.add_argument('-usecols',default= -1, help='Column with data')
    
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
    
        time_units_dict = {'ps':10e-12, 'ns': 10e-9, 'us': 10e-6, 'ms': 10e-3, 's': 1 }
        
        os.chdir(args.wd)
        
        data = np.loadtxt(args.data_file, usecols=args.usecols)
        data *=(time_units_dict[args.time_unit]/time_units_dict[args.new_time_unit])
        
        #Calculate general statistics
        mu, sigma, t_m = np.mean(data), np.std(data, ddof=1), np.median(data)
         
        #Calculate ECDF
        time_bins = np.logspace(np.log10(np.amin(data)/args.range_factor), np.log10(np.amax(data)*args.range_factor), num = args.nbins, base = 10)
        hist_values, _ = np.histogram(data, bins=time_bins)
        
        hist_values = np.append(hist_values,0)
        
        ecdf = np.cumsum(hist_values)/len(data)
        
        #Fit the ECDF with the func
        def func(t, rate=1/mu):
            return 1-np.exp(-rate*t)
        
        popt, pcov = curve_fit(func,xdata=time_bins, ydata=ecdf, ftol=1e-8, method='lm', p0=(1/mu))
        rate, tau = popt[0], 1/popt[0] 
        residuals =  ecdf - func(time_bins, rate)
        fitted_curve = func(time_bins, rate)  
        
        #KS Test
        sampling_from_theoretical_distribution = np.random.exponential(scale=1/rate, size=len(data)*1000000)
        D, p_value = ks_2samp(data, sampling_from_theoretical_distribution) 
        
        #theoretical_curve
        theoretical_curve=expon.cdf(time_bins, loc=0, scale=1/rate)
        
        #Save the results
        header = 'Time_bins   ECDF   Fitted_curve  Theoretical_curve'
        ordered_data = [time_bins, ecdf, fitted_curve, theoretical_curve]
        np.savetxt(args.out_file,  np.column_stack(ordered_data), fmt='%8.6f', header=header,comments='')
        
        #Plot the results
        plot_ecdf(outname=args.ECDF_plot, time_bins=time_bins, ecdf=ecdf, fitted_curve=fitted_curve, theoretical_curve=theoretical_curve,  p_value=p_value, rate=rate, time_units=args.new_time_unit)
        plot_cdf(outname=args.CDF_plot, times=data, rate=rate, time_units=args.new_time_unit)
        
        #Save statistics
        
        with open(args.statistics_file, "w") as out:
            out.write("mu: {:.5f}\n".format(mu))
            out.write("mu_sem: {:.5f}\n".format(sigma/np.sqrt(len(data))))
            out.write("sigma: {:.5f}\n".format(sigma))
            out.write("t_m: {:.5f}\n".format(t_m))
            out.write("tau: {:.5f}\n".format(tau))
            out.write("mu_sigma_ratio: {:.5f}; should be 1\n".format(mu/sigma))
            out.write("log2mu_median_ratio: {:.5f}; should be 1\n".format(mu*np.log(2)/t_m))
            out.write("tau_mu_ratio: {:.5f}; should be 1, because mu approximates tau\n".format(tau/mu))
            out.write("pvalue_KS_statistic: {:.5f}\n".format(p_value))
