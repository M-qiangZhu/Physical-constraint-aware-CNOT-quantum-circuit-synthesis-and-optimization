# PyZX - Python library for quantum circuit rewriting 
#        and optimisation using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

usage_string = """python -m pyzx command [args]
Run one of the scripts supplied with PyZX.

The options for command are:
    opt    -- Optimise a circuit using PyZX
    tikz   -- Convert a circuit into a Tikz file
    mapper -- Map CNOT circuits onto restricted architectures
    router -- Map any circuit onto restricted architectures
    cnots  -- Generate random CNOT circuits 

For help on the arguments for these commands run for instance 'python -m pyzx opt --help'
"""

if __name__ == '__main__':

    import sys
    import argparse

    # from sys import exit

    try:
        from .scripts import circ2circ
        from .scripts import circ2tikz
        from .scripts import cnot2cnot
        from .scripts import circuit_router
        from .scripts import cnot_generator
    except SystemError:
        print("Please run as a module by using 'python -m pyzx'")
        exit(1)

    parser = argparse.ArgumentParser(prog="PyZX", description="PyZX commandline interface",
                                     usage=usage_string)

    parser.add_argument('command', help='Command to run')

    print("len(sys.argv) : ", len(sys.argv))

    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    print("============== 输出各个参数 ==============")
    for i in range(0, len(sys.argv)):
        print(i, sys.argv[i])
    print("==============  输出结束  ==============")

    print("sys.argv[1:2] : ", sys.argv[1:2])  # sys.argv[1:2] ['mapper'] 用列表存储
    args = parser.parse_args(sys.argv[1:2])
    print("parser.parse_args(sys.argv[1:2])=>args : ", args)
    print("args.command : ", args.command)

    if args.command not in ('opt', 'tikz', 'mapper', 'router', 'cnots'):
        print("Unrecognized command '{}'".format(args.command))
        parser.print_help()
        exit(1)

    if args.command == 'opt':
        circ2circ.main(sys.argv[2:])
    if args.command == 'tikz':
        circ2tikz.main(sys.argv[2:])
    if args.command == 'mapper':
        print("mapper found!")
        print("sys.argv[2]", sys.argv[2])
        cnot2cnot.main(sys.argv[2:])
    if args.command == 'router':
        circuit_router.main(sys.argv[2:])
    if args.command == 'cnots':
        cnot_generator.main(sys.argv[2:])
