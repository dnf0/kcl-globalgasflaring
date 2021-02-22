import os
import sys
import tempfile
import subprocess
import glob

import src.config.filepaths as fp
from src.utils import build_outpath


def presubmission_checks(file_to_process,
                         sensor,
                         proc_flags):
    if not proc_flags['reprocess']:
        processed_file = build_outpath(sensor, file_to_process, proc_flags['stage'])
        if os.path.isfile(processed_file):
            return False
    return True


def submit(script, file_to_process, sensor):
    (gd, temp_file) = tempfile.mkstemp('.sh', 'ggf.', fp.script_temp, True)
    g = os.fdopen(gd, "w")
    g.write('#!/bin/bash\n')
    g.write('export PYTHONPATH=$PYTHONPATH:/home/users/dnfisher/projects/kcl-globalgasflaring/\n')
    g.write(" ".join([fp.script_temp, script, file_to_process, sensor + "\n"]))
    g.write(" ".join(["rm -f ", temp_file + "\n"]))
    g.close()
    os.chmod(temp_file, 0o755)

    cmd = ['sbatch -p short-serial', fp.slurm_info, fp.slurm_error, temp_file]
    try:
        subprocess.call(cmd)
    except Exception as e:
        print('Subprocess failed with error:', str(e))


def main():
    script = sys.argv[1]
    sensor = sys.argv[2]

    # check args
    if sensor not in ['ats', 'at2', 'at1', 'sls']:
        raise NotImplementedError(sensor)
    if script not in ['hotspots', 'flares']:
        raise NotImplementedError(script)

    # set processing flags
    proc_flags = {'reprocess': False, 'stage': script}

    # append filetype to script
    script += '.py'

    # TODO reverse filepath order so that most recent files are processed first
    filepaths = glob.glob(fp.products[sensor], recursive=True)
    for f in filepaths:
        if not presubmission_checks(f, sensor, proc_flags):
            continue
        submit(script, f, sensor)


if __name__ == "__main__":
    main()
