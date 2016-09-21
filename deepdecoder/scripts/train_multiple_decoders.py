# Copyright 2016 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import signal
import traceback
from subprocess import Popen, TimeoutExpired, DEVNULL
import json

import click


@click.command()
@click.option('--gt', '-g', type=click.Path(exists=True))
@click.option('--train-set', '-t', type=click.Path(exists=True))
@click.option('--test-set', type=click.Path(exists=True))
@click.option('--units', type=int)
@click.option('--model', default='resnet', type=str)
@click.option('--dataset-names', type=str)
@click.option('--force', is_flag=True)
@click.option('--max-processes', default=2, type=int)
@click.argument('output_dir', type=click.Path(resolve_path=True))
def main(gt, train_set, test_set, units, model, dataset_names,
         force, output_dir, max_processes):
    os.makedirs(output_dir, exist_ok=force)
    dataset_names = dataset_names.replace(' ', '').split(',')
    augmentations = [
        '',
        '--augmentation',
        '--augmentation --hist-eq',
        '--noise',
        '--augmentation --noise --hist-eq',
        '--hist-eq',
        '--noise --hist-eq',
    ]
    processes = []
    os.setpgrp()
    try:
        process_id = 1
        for dataset_name in dataset_names:
            for augmentation in augmentations:
                try:
                    prefix = "decoder_{}_{}".format(
                        dataset_name,
                        augmentation.replace('--', '').replace(' ', '_').replace('-', '_')
                    )
                    stdout_fname = os.path.join(output_dir, prefix + "_stdout.log")
                    stdout_log = open(stdout_fname, 'wb+')
                    cmd = [
                        'bb_train_decoder',
                        '--gt', gt,
                        '--train-set', train_set,
                        '--test-set', test_set,
                        '--units', units,
                        '--model', model,
                        '--dataset-name', dataset_name,
                        '--epoch', 1000,
                    ]
                    if augmentation != '':
                        cmd.extend(augmentation.split(' '))
                    cmd.append(output_dir)
                    cmd = list(map(str, cmd))
                    total_process = len(dataset_names) * len(augmentations)
                    print("Run {}/{}: ".format(process_id, total_process) +
                          " ".join(cmd).replace('--', '\n    --'))
                    processes.append(Popen(cmd, stdin=DEVNULL, stdout=stdout_log))
                    process_id += 1

                    while len(processes) >= max_processes:
                        for i in range(len(processes)):
                            try:
                                p = processes[i]
                                p.wait(1)
                                del processes[i]
                                break
                            except TimeoutExpired:
                                pass
                except KeyboardInterrupt:
                    return
                except Exception as err:
                    traceback.print_tb(err.__traceback__)
                    print(err)
        for i in range(len(processes)):
            processes[i].wait()
        processes = []
    finally:
        if len(processes) > 0:
            os.killpg(0, signal.SIGKILL)
