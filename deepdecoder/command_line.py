# Copyright 2015 Leon Sixt
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


from argparse import ArgumentParser
import sys


class NetworkArgparser(object):
    def __init__(self, train_cb, test_cb):
        self.parser = ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.parser_train = self.subparsers.add_parser(
            "train", help="train the network")
        self.parser_train.add_argument(
            "--weight-dir", default="weights",
            help="directory from where to load weights")
        self.parser_train.set_defaults(func=train_cb)
        self.parser_test = self.subparsers.add_parser(
            "test", help="test the network")
        self.parser_test.add_argument(
            "--weight-dir", default="weights",
            help="directory from where to load weights")
        self.parser_test.set_defaults(func=test_cb)

    def parse_args(self):
        args = self.parser.parse_args()
        if hasattr(args, 'func'):
            args.func(args)
        else:
            print("No a valid action {}.".format(sys.argv[1]))
            self.parser.print_help()
            exit(1)
