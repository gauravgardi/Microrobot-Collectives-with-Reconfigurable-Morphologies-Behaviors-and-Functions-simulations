#!/usr/bin/env python
"""
This is for running simulation files in parallel.
"""
import multiprocessing
import subprocess
# import sys
import os
from multiprocessing.pool import ThreadPool
import numpy as np
import platform


def process_files():
    pool = multiprocessing.pool.ThreadPool(9)  # Number of processors to be used
    if platform.node() == 'NOTESIT43' and platform.system() == 'Windows':
        project_dir = "D:\\simulationFolder\\spinning_rafts_sim2"
    elif platform.node() == 'NOTESIT71' and platform.system() == 'Linux':
        project_dir = r'/media/wwang/shared/spinning_rafts_simulation/spinning_rafts_sim2'
    elif platform.node() == 'LABSIT27' and platform.system() == 'Linux':
        project_dir = '/home/gardi/Research_IS/spinning_rafts_sim2'    
    else:
        project_dir = os.getcwd()

    if project_dir != os.getcwd():
        os.chdir(project_dir)
        
    

#    print(os.getcwd())
    data_dir = os.path.join(project_dir, 'data')
    script_dir = os.path.join(project_dir, "scripts")
    filename = 'simulation_combined_lissajous_rotationWorking_2021-08-11_exportMatlabFile_parallel_stepoutAdded.py'
#    num_of_rafts = [7]  # num of rafts
#    spin_speeds = [26] # np.arange(-10, -20, -1)  # spin speeds, negative means clockwise in the rh coordinate
#    Omegax = [30] #[100,150,200,250,500]  # num of rafts
#    Omega = ([30, 30], [30, 31], [30, 60], [10, 0], [30, 0], [0, 10], [0, 30]) #[-13,-20,-26,-60] # np.arange(-10, -20, -1)  # spin speeds, negative means clockwise in the rh coordinate
#    Omega = np.arange(30, 75, 5)  # spin speeds, rotting and oscillating collective
    Omega = np.arange(20, 65, 5)  # spin speeds, static collective
#    Omega = np.arange(10, 75, 10)  # spin speeds, chains and GASPP
    
    arenaSize = [1.5e5] #[4.0e3,5.0e3,6.0e3,7.0e3,8.0e3,9.0e3,1.0e4,1.5e4] #[3.0e3,4.0e3,6.0e3,7.0e3,8.0e3,9e3] # arena size in microns
    
    for arg3 in arenaSize:
        for arg1 in Omega:
            script_file = os.path.join(script_dir, filename)
    #           print(script_file)
#            cmd = ["python", script_file, str(arg1[0]), str(arg1[1]), str(arg3)]
#            cmd = ["python", script_file, str(arg1), str(arg1), str(arg3)] # rotating collective
#            cmd = ["python", script_file, str(arg1), str(arg1 + 1), str(arg3)] # oscillating collective
#            cmd = ["python", script_file, str(arg1), str(arg1*2), str(arg3)] # static collective fx = 2*fy
#            cmd = ["python", script_file, str(arg1), str(0), str(arg3)] # Y chains
#            cmd = ["python", script_file, str(0), str(arg1), str(arg3)] # GASPP
    
            cmd = ["python", script_file, str(arg1*2), str(arg1*1), str(arg3)] # static collective fy = 2*fx
    
                #        p=subprocess.check_call(cmd ,stdout=subprocess.PIPE)
                #        print(p.communicate())
                #        print(script_file)
                # pool.apply_async(cmd) # supply command to system
            print(cmd)
            pool.apply_async(subprocess.check_call, (cmd,))  # supply command to system
    pool.close()
    pool.join()


if __name__ == '__main__':
    process_files()
