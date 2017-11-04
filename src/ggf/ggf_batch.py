import re
import os
import tempfile

import src.config.filepaths as filepaths
import subprocess

class BatchSystem:
    """Container for syntax to call a batch queuing system.
    Member variables:
    command      - The name of the batch queuing system.
    args         - A dictionary of string format functions, each taking one
                   argument, to produce an argument of the queuing function.
    regex        -  # Regex to parse command output.
    depend_arg   - Command for the job dependencies. This would be an element of
                   args, but has a more complicated structure.
    depend_delim - String require to space consequetive dependencies."""

    def __init__(self, command, regex, depend_arg, depend_delim, args):
        self.command = command
        self.args = args
        self.regex = re.compile(regex)
        self.depend_arg = depend_arg
        self.depend_delim = depend_delim
        self.args.update({'depend' : self.parse_depend})

    def parse_depend(self, item):
        """Deal with slightly more complex syntax for declaring dependencies"""
        if isinstance(item, str):
            return self.depend_arg.format(item)
        else:
            return self.depend_arg.format(self.depend_delim.join(item))

    def print_batch(self, values, exe=None):
        """Returns the queuing shell command. 'exe' is the thing to run."""
        arguments = [self.command]
        arguments.extend([ self.args[key](values[key])
                           for key in values.keys() if values[key] ])
        if type(exe) in [list, tuple]:
            arguments.extend(exe)
        elif exe:
            arguments.append(exe)
        return ' '.join(arguments)

    def parse_out(self, text, key=None):
        """Parse output of queuing function. Returns all regex groups unless
        'key' is specified, where it just returns that."""
        m = self.regex.match(text)
        if m == None:
            raise SyntaxError('Unexpected output from queue system ' + text)
        if key:
            return m.group(key)
        else:
            return m.groupdict()


# setup the batch running class
batch = BatchSystem('bsub',
                   'Job <(?P<ID>\d+)> is submitted to (?P<desc>\w*) queue '
                   '<(?P<queue>[\w\.-]+)>.',
                   '-w "done({})"', ') && done(',
                   {'duration' : '-W {}'.format,
                    'email'    : '-u {}'.format,
                    'err_file' : '-e {}'.format,
                    'job_name' : '-J {}'.format,
                    'log_file' : '-o {}'.format,
                    'order'    : '-R "order[{}]"'.format,
                    'procs'    : '-n {}'.format,
                    'priority' : '-p {}'.format,
                    'queue'    : '-q {}'.format,
                    'ram'      : '-R "rusage[mem={}]"'.format})
batch_values = {'email'    : 'danielfisher0@gmail.com'}

# iterate over all ATSR files in directory
years = os.listdir(filepaths.path_to_data)
for yr in years:
    if len(yr) > 4:
        continue
    if ('at2' in filepaths.path_to_data) & (yr > 2003):
        continue

    path = os.path.join(filepaths.path_to_data, yr)
    for root, dirs, files in os.walk(path):
        for f in files:
	    if f.split('.')[-1] not in ['N1', 'E2', 'E1']:
                continue
            
            path_to_data = os.path.join(root, f)

            # build path to output
            ymd = f[14:22]
            out_dir = os.path.join(filepaths.path_to_cems_output, ymd[0:4], ymd[4:6], ymd[6:8])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # for each ATSR file generate a bash script that calls ggf
            (gd, script_file) = tempfile.mkstemp('.sh', 'ggf.',
                                                 out_dir, True)
            g = os.fdopen(gd, "w")
            g.write('export PYTHONPATH=$PYTHONPATH:/home/users/dnfisher/projects/kcl-globalgasflaring/\n')
            g.write(filepaths.ggf_dir + 'ggf_processor.py ' +
                    path_to_data + ' ' +
                    out_dir + " \n")
            g.write("rm -f " + script_file + "\n")
            g.close()
            os.chmod(script_file, 0o755)

            # generate bsub call using print_batch
            cmd = batch.print_batch(batch_values, exe=script_file)

            # use subprocess to call the print batch command
            out = subprocess.check_output(cmd.split(' '))
            jid = batch.parse_out(out, 'ID')
