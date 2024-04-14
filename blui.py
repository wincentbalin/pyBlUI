#!/usr/bin/env python3
"""Implementation of BlUI (http://doi.acm.org/10.1145/1294211.1294250)."""

import pickle
import screeninfo
import sounddevice
from numpy import hanning, arange, split, pad, hstack, vstack
from scipy.fft import fft, fftshift
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

DATA_WIDTH = 1024


def list_monitors(args):
    for i, m in enumerate(screeninfo.get_monitors()):
        print('{index}: {name}{primary}'.format(index=i, name=m.name, primary=' (primary)' if m.is_primary else ''))


def list_microphones(args):
    for d in filter(lambda d: d['max_input_channels'] > 0, sounddevice.query_devices()):
        print('{index}: {name}'.format(index=d['index'], name=d['name']))


def train_model(args):
    all_monitors = screeninfo.get_monitors()
    all_microphones = list(filter(lambda d: d['max_input_channels'] > 0, sounddevice.query_devices()))

    import random
    from tkinter import Tk, Frame, Label

    class Trainer(Frame):
        def __init__(self, master, res, sample_rate: int):
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
                    region = Label(self, text='', highlightthickness=2)
                    region.grid(column=grid_x, row=grid_y, sticky='NWSE')
                    self.regions.append(region)

            self.recording_duration = 4.0
            self.pause_duration = 2.0
            self.fs = sample_rate
            self.training_queue = []
            self.training_index = None
            self.training_sample = None
            self.training_samples = []

        def start_training(self):
            random.seed()
            self.training_queue = random.sample(range(len(self.regions)), len(self.regions))
            self.master.after(1, self.start_step)

        def start_step(self):
            if self.training_index is not None:
                self.regions[self.training_index].config(text='', highlightbackground=self.master.cget('bg'))
            if not self.training_queue:
                self.master.destroy()
                return
            self.training_index = self.training_queue.pop()
            text = 'Blow at me\nRemaining samples: {}'.format(len(self.training_queue))
            self.regions[self.training_index].config(text=text, highlightbackground='yellow')
            self.training_sample = sounddevice.rec(int(self.recording_duration * self.fs), samplerate=self.fs, channels=1)
            self.master.after(int(self.recording_duration * 1000), self.end_step)

        def end_step(self):
            sounddevice.wait()
            self.regions[self.training_index].config(text='Inhale again, please!', highlightbackground='lightgreen')
            self.training_samples.append((self.training_sample, self.training_index))
            self.master.after(int(self.pause_duration * 1000), self.start_step)

        def get_recorded_samples(self):
            return self.training_samples

    microphone = next(filter(lambda d: d['index'] == args.microphone_index, all_microphones))
    sounddevice.default.device = microphone['name']
    sounddevice.default.samplerate = args.sample_rate

    monitor = all_monitors[args.monitor_index]
    resolution = tuple(map(int, args.grid_resolution.split('x')))

    root = Tk()
    root.title('Train BlUI')
    root.geometry('100x100+{x}+{y}'.format(x=monitor.x, y=monitor.y))
    root.state('zoomed')
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    app = Trainer(root, resolution, args.sample_rate)
    app.start_training()
    app.mainloop()

    data_vectors, targets = [], []
    window = hanning(DATA_WIDTH)
    for sample, sample_index in app.get_recorded_samples():
        chunks = split(sample[:, 0], indices_or_sections=list(range(DATA_WIDTH, len(sample), DATA_WIDTH)))
        if chunks[-1].size < DATA_WIDTH:
            chunks[-1] = pad(chunks[-1], (0, DATA_WIDTH - chunks[-1].size))
        for chunk in chunks:
            spectrum = fftshift(fft(chunk * window))
            data_vector = hstack((spectrum.real, spectrum.imag))
            data_vectors.append(data_vector)
            targets.append(sample_index)
    data = vstack(data_vectors)
    pca = PCA(n_components=32)
    pca.fit(data)
    data = pca.transform(data)
    # From https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': arange(1, 25)}
    knn_gscv = GridSearchCV(knn, param_grid)
    knn_gscv.fit(data, targets)
    n_neighbors = knn_gscv.best_params_['n_neighbors']
    print('Best KNN score: {score} ({n})'.format(score=knn_gscv.best_score_, n=n_neighbors))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(data, targets)
    with open(args.model_name, 'wb') as f:
        pickle.dump([pca, knn], f)


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
    parser3.add_argument('--sample_rate', type=int, default=44100, help='Sample rate')
    parser3.add_argument('monitor_index', type=int, help='Index of monitor (run command list_monitors)')
    parser3.add_argument('microphone_index', type=int, help='Index of microphone (run command list_microphones)')
    parser3.add_argument('grid_resolution', help='Grid resolution (WxH, for example 3x3)')
    parser3.add_argument('model_name', help='Name of the model in pickle format')
    parser3.set_defaults(func=train_model)
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        parser.print_usage()


if __name__ == '__main__':
    main()
