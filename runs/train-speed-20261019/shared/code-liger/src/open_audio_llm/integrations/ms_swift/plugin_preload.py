"""Import the --external_plugins files inside a forkserver process.

A worker unpickles the dataset before swift's worker_init_fn can replay the plugin
imports, so classes defined in a plugin (named `_swift_external_<hash>`) must already
be in sys.modules when the forkserver forks the worker.
"""

import os

from swift.utils.utils import import_external_file

PLUGIN_FILES_ENV = "OAL_FORKSERVER_PLUGIN_FILES"

for _path in filter(None, os.environ.get(PLUGIN_FILES_ENV, "").split(os.pathsep)):
    import_external_file(_path)
