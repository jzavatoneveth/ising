import sys
import os
import time
import csv
import click
import numpy as np
import logging
import matplotlib.pyplot as plt
from ising_c import run_ising  #import run_ising function from ising.py
import multiprocessing as mp
from IsingLattice import IsingLattice
from functools import partial
from math import ceil


def run_simulation(temp,
                   n,
                   num_steps,
                   num_burnin,
                   num_analysis,
                   flip_prop,
                   j,
                   b,
                   t0=-1,
                   b0=-1):

    try:
        #run the Ising model

        lattice = IsingLattice(n, flip_prop)

        Msamp, Esamp = run_ising(lattice, temp, num_steps, num_burnin, j, b,
                                 t0, b0)

        # calculate statistical values
        M_mean = np.average(Msamp[-num_analysis:])
        E_mean = np.average(Esamp[-num_analysis:])
        M_std = np.std(Msamp[-num_analysis:])
        E_std = np.std(Esamp[-num_analysis:])
        cv = (1 / (temp * temp)) * (E_std**2)
        chi = (1 / temp) * (M_std**2)

        data_array = [np.abs(M_mean), M_std, E_mean, E_std, cv, chi]

        corr = lattice.calc_auto_correlation()
        lattice.free_memory()

        return [temp, data_array, corr]

    except KeyboardInterrupt:
        print("\n\nProgram Terminated. Good Bye!")
        sys.exit()

    # except:
    #     logging.error("Temp="+str(temp)+": Simulation Failed. No Data Written")


# Simulation options (enter python main.py --help for details)
@click.command()
@click.option(
    '--t_min',
    default=2.0,
    prompt='Minimum Temp',
    help='Minimum Temperature (inclusive)',
    type=float)
@click.option(
    '--t_max',
    default=2.6,
    prompt='Maximum Temp',
    help='Maximum Temperature (inclusive)',
    type=float)
@click.option(
    '--t_step',
    default=0.1,
    prompt='Temp Step Size',
    help='Temperature Step Size',
    type=float)
@click.option(
    '--n', prompt='Lattice Size', help='Lattice Size (NxN)', type=int)
@click.option(
    '--num_steps', default=500000, help='Total Number of Steps', type=int)
@click.option(
    '--num_analysis',
    default=100000,
    help='Number of Steps used in Analysis',
    type=int)
@click.option(
    '--num_burnin',
    default=300000,
    help='Total Number of Burnin Steps',
    type=int)
@click.option('--j', default=1.0, help='Interaction Strength', type=float)
@click.option('--b', default=0.0, help='Applied Magnetic Field', type=float)
@click.option('--t0', default=-1, help='Initial temperature', type=float)
@click.option('--b0', default=0.1, help='Initial magnetic field', type=float)
@click.option(
    '--flip_prop',
    default=0.1,
    help='Proportion of Spins to Consider Flipping per Step',
    type=float)
@click.option(
    '--output',
    default='data',
    help='Directory Name for Data Output',
    type=str)
@click.option('--processes', default=mp.cpu_count(), help='', type=int)
def main(t_min, t_max, t_step, n, num_steps, num_analysis, num_burnin, j, b,
         t0, b0, flip_prop, output, processes):
    simulation_start_time = time.time()

    data_filename, corr_filename = initialize_simulation(
        n, num_steps, num_analysis, num_burnin, output, j, b, flip_prop)

    run_processes(processes, t_min, t_max, t_step, n, num_steps, num_burnin,
                  num_analysis, flip_prop, j, b, t0, b0, data_filename,
                  corr_filename)

    simulation_duration = round((time.time() - simulation_start_time) / 60.0,
                                2)

    print(
        '\n\nSimulation finished in {0} minutes. Data written to {1}.'.format(
            simulation_duration, data_filename))

    return None


def initialize_simulation(n, num_steps, num_analysis, num_burnin, output, j, b,
                          flip_prop):
    check_step_values(num_steps, num_analysis, num_burnin)
    data_filename, corr_filename = get_filenames(output)
    write_sim_parameters(data_filename, corr_filename, n, num_steps,
                         num_analysis, num_burnin, j, b, flip_prop)
    print('\nSimulation Started! Data will be written to ' + data_filename +
          '\n')
    return data_filename, corr_filename


def check_step_values(num_steps, num_analysis,
                      num_burnin):  #simulation size checks and exceptions
    if num_burnin > num_steps:
        raise ValueError(
            'num_burnin cannot be greater than available num_steps. Exiting simulation.'
        )

    if num_analysis > (num_steps - num_burnin):
        raise ValueError(
            'num_analysis cannot be greater than available num_steps after burnin. Exiting simulation.'
        )


def get_filenames(
        dirname):  #make data folder if doesn't exist, then specify filename
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        data_filename = os.path.join(
            dirname, 'data_' + str(time.strftime("%Y%m%d-%H%M%S")) + ".csv")
        corr_filename = os.path.join(
            dirname, 'corr_' + str(time.strftime("%Y%m%d-%H%M%S")) + ".csv")
        #Write simulation parameters to file
        return data_filename, corr_filename
    except:
        raise ValueError('Directory name not valid. Exiting simulation.')


def get_temp_array(t_min, t_max, t_step):
    if t_min > t_max:
        raise ValueError(
            'T_min cannot be greater than T_max. Exiting Simulation')
    elif (t_max - t_min) < t_step:
        return [t_min]
    else:
        # Pure Python replacement to deal with numpy.arange's bad handling of floating point round-off
        n_steps = int(ceil((float(t_max) - float(t_min)) / float(t_step)))
        T = [t_min + t_step * float(x) for x in range(0, n_steps)]
        return T


def write_sim_parameters(data_filename, corr_filename, n, num_steps,
                         num_analysis, num_burnin, j, b, flip_prop):
    try:
        with open(data_filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
            # Write simulations parameters to CSV file
            writer.writerow([
                'Lattice Size (NxN)', 'Total Steps', 'Steps Used in Analysis',
                'Burnin Steps', 'Interaction Strength', 'Applied Mag Field',
                'Spin Prop'
            ])
            writer.writerow(
                [n, num_steps, num_analysis, num_burnin, j, b, flip_prop])
            writer.writerow([])

        with open(corr_filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')

            # Write simulations parameters to CSV file
            writer.writerow([
                'Lattice Size (NxN)', 'Total Steps', 'Steps Used in Analysis',
                'Burnin Steps', 'Interaction Strength', 'Applied Mag Field',
                'Spin Prop'
            ])
            writer.writerow(
                [n, num_steps, num_analysis, num_burnin, j, b, flip_prop])
            writer.writerow([])
    except:
        logging.error(
            'Could not save simulation parameters. Exiting simulation')
        sys.exit()


def append_data_to_file(filename, data_array, temp=False):
    try:
        with open(filename, 'a') as csv_file:  # Appends to existing CSV File
            writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
            if temp:
                writer.writerow([temp] + data_array)
            else:
                writer.writerow(data_array)
    except:
        logging.error("Temp={0}: Error Writing to File".format(temp))


def run_processes(processes, t_min, t_max, t_step, n, num_steps, num_burnin,
                  num_analysis, flip_prop, j, b, t0, b0, data_filename,
                  corr_filename):

    # Get the temperature array
    T = get_temp_array(t_min, t_max, t_step)

    # Freeze the simulation function
    simfun = partial(
        run_simulation,
        n=n,
        num_steps=num_steps,
        num_burnin=num_burnin,
        num_analysis=num_analysis,
        flip_prop=flip_prop,
        j=j,
        b=b)

    # Run in parallel
    with mp.Pool(processes=processes) as pool:
        result = pool.map(simfun, T)

    # Save the data
    [
        append_data_to_file(data_filename, data_array, temp)
        for temp, data_array, _ in result
    ]
    [[
        append_data_to_file(corr_filename, corr_value, temp)
        for corr_value in corr
    ] for temp, _, corr in result]

    # Return none
    return None


if __name__ == "__main__":
    main()
