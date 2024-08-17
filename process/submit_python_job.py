# -*- coding: utf-8 -*-

"""submit_python_job.py
    
    Wrapper to submit a generic Python job.

    Example usage: python submit_python_job.py -J 5fps_6sats -N 1 -w 08:00:00 -l /g/g14/pochedls/tropics/logs/ -c rttov -e /g/g14/pochedls/tropics/process/infer_diurnal_cycle.py -A 5fps_6sats_0.25K

    Author: Stephen Po-Chedley
"""

import argparse
import os
import time

# specify argparse arguments
parser = argparse.ArgumentParser(description='Process arguments for submit_python_job.py.')

parser.add_argument('-q', dest='queue', type=str, required=False,
                    default='pbatch',
                    help='HPC queue')
parser.add_argument('-N', dest='nodes', type=str, required=False,
                    default='1',
                    help='Number of nodes to request')
parser.add_argument('-n', dest='cores', type=int, required=False,
                    default=None,
                    help='Number of cores to request')
parser.add_argument('-w', dest='walltime', type=str, required=False,
                    default='08:00:00',
                    help='Wallclock time in HH:MM:SS')
parser.add_argument('-a', dest='account', type=str, required=False,
                    default='cbronze',
                    help='HPC account to charge')
parser.add_argument('-l', dest='logpath', type=str, required=False,
                    default='$HOME/logs',
                    help='Logging directory')
parser.add_argument('-c', dest='conda', type=str, required=True,
                    help='Conda environment')
parser.add_argument('-e', dest='executable', type=str, required=True,
                    help='Full path to executable')
parser.add_argument('-A', dest='sargs', type=str, required=False,
                    default='',
                    help='Script arguments')
parser.add_argument('-J', dest='name', type=str, required=False,
                    default='',
                    help='Job name')


# get arguments
args = parser.parse_args()
queue = args.queue
nodes = args.nodes
cores = args.cores
walltime = args.walltime
account = args.account
logpath = args.logpath
conda = args.conda
executable = args.executable
name = args.name
sargs = args.sargs

# SBATCH ARGS
script = ['#!/bin/bash \n', '\n', '##### These lines are for Slurm\n']
abbr = {'-p': queue,
        '-N': nodes,
        '-J': name,
        '-n': cores,
        '-t': walltime,
        '--mail-type=ALL': '',
        '-A': account,
        '-o': logpath + '/slurm-%j.out',
        '-e': logpath + '/slurm-%j.out'}
for key in abbr.keys():
    if abbr[key] is not None:
        l = '#SBATCH ' + key + ' ' + abbr[key] + '\n'
        script.append(l)

script = script + ['\n',
                   '. /usr/workspace/pochedls/bin/miniconda3/etc/profile.d/conda.sh \n',
                   'conda activate ' + conda + '\n',
                   'python ' + executable + ' ' + sargs + '\n']

if os.path.exists('batch.sh'):
    raise ValueError('batch.sh exists...cannot overwrite this file')

with open('batch.sh', 'w') as f:
    f.writelines(script)

os.system('sbatch batch.sh')
time.sleep(2)

# clean up
os.remove("batch.sh")