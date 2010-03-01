"""
Run directory
"""

import common.hyperparameters, common.options, common.dump

_rundir = None
def rundir():
    global _rundir
    if _rundir is None:
        HYPERPARAMETERS = common.hyperparameters.read("language-model")
        _rundir = common.dump.create_canonical_directory(HYPERPARAMETERS)
    return _rundir
