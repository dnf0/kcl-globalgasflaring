import re
import os
import tempfile

import src.ggf.ggf_processor as ggf_processor
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
        self.args.update({'depend' : self.ParseDepend})

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
batch = BatchSystem()
batch_values = {'email'    : 'daniel.fisher@kcl.ac.uk'}

# iterate over all ATSR files in directory
for root, dirs, files in os.walk(filepaths.path_to_data):
    for f in files:

        path_to_data = root + f
        print path_to_data
        continue

        # build path to output
        out_dir = filepaths.path_to_output + ''
        print out_dir
        continue
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # for each ATSR file generate a bash script that calls ggf
        (gd, script_file) = tempfile.mkstemp('.sh', 'ggf.',
                                             out_dir, True)
        print script_file
        continue
        g = os.fdopen(gd, "w")
        g.write(filepaths.ggf_dir + 'ggf_processor.py ' +
                path_to_data + ' ' +
                out_dir + " \n")
        g.write("rm -f " + script_file + "\n")
        g.close()
        os.chmod(script_file, 0o700)
        continue

        # generate bsub call using print_batch
        cmd = batch.PrintBatch(batch_values, exe=script_file)
        print cmd
        continue

        # use subprocess to call the print batch command
        out = subprocess.check_output(cmd.split(' '))
        jid = batch.ParseOut(out, 'ID')
        print jid