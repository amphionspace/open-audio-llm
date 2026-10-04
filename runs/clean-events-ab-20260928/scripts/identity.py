"""Run the two official AntSpeaker shards, then apply the original screening rules."""

import os
import subprocess

from settings import SETTINGS, artifact, child

processes = []
try:
    for worker, gpu in enumerate(SETTINGS['runtime']['gpus']):
        with (artifact(f'identity-{worker}.log')).open('a') as log:
            process = subprocess.Popen(child('extract.py', '--worker', worker),
                                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                       stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
    for process in processes:
        code = process.wait()
        if code:
            raise RuntimeError(f'AntSpeaker shard exited {code}')
    subprocess.run(child('screen.py'), check=True)
finally:
    for process in processes:
        if process.poll() is None:
            process.terminate()
            process.wait()
