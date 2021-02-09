import os
import tempfile

import src.config.filepaths as filepaths
import src.config.constants as proc_const
import subprocess


def check_atx_year(ymd):
    if 'at2' in f.lower():
        if int(ymd[0:6]) <= 200306:
            return True
    elif 'at1' in f.lower():
        if int(ymd[0:6]) <= 199605:
            return True
    else:
        return True  # we process all ATS data and this checks for that


def make_outpath_atx(f, ymd):

    if 'N1' in f:
        sensor = 'ats'
    if 'E2' in f:
        sensor = 'at2'
    if 'E1' in f:
        sensor = 'at1'

    out_dir = os.path.join(filepaths.path_to_cems_output_l2, sensor,
                           ymd[0:4], ymd[4:6], ymd[6:8])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def submit_atx(root, f):

    if f.split('.')[-1] not in ['N1', 'E2', 'E1']:
        return

    data_path = os.path.join(root, f)

    # check year, month and sensor to see if we are going to process
    ymd = f[14:22]
    if not check_atx_year(ymd):
        return

    # construct ouptut path
    out_dir = make_outpath_atx(f, ymd)
    output_fname = f.split('.')[0] + '_hotspots.csv'
    out_path = os.path.join(out_dir, output_fname)
    if 'extract_hotspots' in python_exe:
        # check if we have already processed the file and skip if so
        if os.path.isfile(out_path):
            return
    elif 'flares_and_samples' in python_exe:
        # check if we have already processed the file and skip if so
        if os.path.isfile(os.path.join(out_dir, f.split('.')[0] + '_flares.csv')):
            return

    # for each ATSR file generate a bash script that calls ggf
    (gd, script_file) = tempfile.mkstemp('.sh', 'ggf.',
                                         out_dir, True)
    g = os.fdopen(gd, "w")

    g.write('#!/bin/bash\n')
    g.write('export PYTHONPATH=$PYTHONPATH:/home/users/dnfisher/projects/kcl-globalgasflaring/\n')
    g.write(filepaths.ggf_dir + python_exe +
            data_path + ' ' +
            out_path + " \n")
    g.write("rm -f " + script_file + "\n")
    g.close()
    os.chmod(script_file, 0o755)

    # generate slurm call
    info = '-o /group_workspaces/jasmin2/nceo_aerosolfire/data/temp/slurm_logs/info/%j.out '
    error = '-e /group_workspaces/jasmin2/nceo_aerosolfire/data/temp/slurm_logs/error/%j.out '

    cmd = 'sbatch -p short-serial ' + info + error + script_file

    # use subprocess to call the print batch command
    try:
        subprocess.call(cmd.split(' '))
    except Exception, e:
        print 'Subprocess failed with error:', str(e)


def make_outpath_sls(f, ymd):
    sensor = 'sls'
    out_dir = os.path.join(filepaths.path_to_cems_output_l2, sensor,
                           ymd[0:4], ymd[4:6], ymd[6:8])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def submit_sls(root, f):

    if '.zip' not in f:
        return

    ymd = f[16:24]
    path_to_data = os.path.join(root, f)
    out_dir = make_outpath_sls(f, ymd)

    # check if we have already processed the file and skip if so
    output_fname = f.split('.')[0] + '_hotspots.csv'
    out_path = os.path.join(out_dir, output_fname)
    print(out_path)
    if 'extract_hotspots' in python_exe:
        # check if we have already processed the file and skip if so
        if os.path.isfile(out_path):
            return
    elif 'flares_and_samples' in python_exe:
        # check if we have already processed the file and skip if so
        if os.path.isfile(os.path.join(out_dir, f.split('.')[0] + '_flares.csv')):
            return

    temp_dir = filepaths.path_to_temp
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    (gd, script_file) = tempfile.mkstemp('.sh', 'ggf.',
                                         out_dir, True)
    g = os.fdopen(gd, "w")
    g.write('#!/bin/bash\n')
    g.write('export PYTHONPATH=$PYTHONPATH:/home/users/dnfisher/projects/kcl-globalgasflaring/\n')
    g.write(filepaths.ggf_dir + python_exe +
            path_to_data + ' ' +
            out_path + ' ' +
            temp_dir + " \n")
    g.write("rm -f " + script_file + "\n")

    g.close()
    os.chmod(script_file, 0o755)

    # generate slurm call
 
    info = '-o /group_workspaces/jasmin2/nceo_aerosolfire/data/temp/slurm_logs/info/%j.out '
    error = '-e /group_workspaces/jasmin2/nceo_aerosolfire/data/temp/slurm_logs/error/%j.out '

    cmd = 'sbatch -p short-serial ' + info + error + script_file
    print(cmd)
    # use subprocess to call the print batch command
    try:
        subprocess.call(cmd.split(' '))
    except Exception, e:
        print 'Subprocess failed with error:', str(e)


# define python script to run
python_exe = 'ggf_extract_hotspots_sls.py '

if 'atx' in python_exe:
    paths = filepaths.paths_to_atx_data
else:
    paths = filepaths.path_to_sls_data

# iterate over all ATSR files in directory
for path_to_data in paths:
    years = os.listdir(path_to_data)
    for yr in years:
        if len(yr) > 4:
            continue
        if (int(yr) < 2020):
           continue
        print yr
        path = os.path.join(path_to_data,  yr)
        for root, dirs, files in os.walk(path, followlinks=True):
            for f in files:
                print root, f
                if 'atx' in python_exe:
                    submit_atx(root, f)
                else:
                    submit_sls(root, f)
