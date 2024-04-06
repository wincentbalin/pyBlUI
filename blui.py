#!/usr/bin/env python3
"""Implementation of BlUI (http://doi.acm.org/10.1145/1294211.1294250)."""

import screeninfo
import sounddevice
import numpy


def list_monitors(args):
    for i, m in enumerate(screeninfo.get_monitors()):
        print('{index}: {name}{primary}'.format(index=i, name=m.name, primary=' (primary)' if m.is_primary else ''))


def list_microphones(args):
    for d in filter(lambda d: d['max_input_channels'] > 0, sounddevice.query_devices()):
        print('{index}: {name}'.format(index=d['index'], name=d['name']))


def train_model(args):
    all_monitors = screeninfo.get_monitors()
    all_microphones = list(filter(lambda d: d['max_input_channels'] > 0, sounddevice.query_devices()))

    from tkinter import Tk, Frame, Label

    class Trainer(Frame):
        def __init__(self, master, mic, res):
            super().__init__(master)
            grid_width, grid_height = res
            self.grid(column=0, row=0, sticky='NWSE')
            self.rowconfigure(0, weight=1)
            self.columnconfigure(0, weight=1)
            self.columnconfigure(tuple(range(grid_width)), weight=1)
            self.rowconfigure(tuple(range(grid_height)), weight=1)

            self.regions = []
            for grid_x in range(grid_width):
                for grid_y in range(grid_height):
                    text = 'Region {index}'.format(index=grid_y * grid_width + grid_x + 1)
                    region = Label(self, text='X')
                    region.grid(column=grid_x, row=grid_y, sticky='NWSE')
                    self.regions.append(region)

    monitor = all_monitors[args.monitor_index]
    microphone = next(filter(lambda d: d['index'] == args.microphone_index, all_microphones))
    resolution = tuple(map(int, args.grid_resolution.split('x')))

    root = Tk()
    root.title('Train BlUI')
    root.geometry('100x100+{x}+{y}'.format(x=monitor.x, y=monitor.y))
    root.state('zoomed')
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    app = Trainer(root, microphone, resolution)
    app.mainloop()


def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
    subparsers = parser.add_subparsers(help='Commands')
    parser1 = subparsers.add_parser('list_monitors', help='List monitors')
    parser1.set_defaults(func=list_monitors)
    parser2 = subparsers.add_parser('list_microphones', help='List microphones')
    parser2.set_defaults(func=list_microphones)
    parser3 = subparsers.add_parser('train_model', help='Record audio and train model')
    parser3.add_argument('monitor_index', type=int, help='Index of monitor (run command list_monitors)')
    parser3.add_argument('microphone_index', type=int, help='Index of microphone (run command list_microphones)')
    parser3.add_argument('grid_resolution', help='Grid resolution (WxH, for example 3x3)')
    parser3.add_argument('model_name', help='Name of the model in XXX format')
    parser3.set_defaults(func=train_model)
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        parser.print_usage()


if __name__ == '__main__':
    main()
