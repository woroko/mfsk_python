import numpy as np
import binascii
import random, string
from baudot import *
import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform
from kivy.clock import Clock
from kivy.properties import VariableListProperty
from kivy.metrics import *

from numpy.lib import stride_tricks
from PIL import Image

from txrxlib import main as testtx

import threading
import time

#Window.size = (600,320)

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return int2bytes(n).decode(encoding, errors)

def int2bytes(i):
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))

def interleaver(line, n):
    splits = []
    for i in range (0, n):
        splits.append("")

    for i in range (0, len(line)):
        splits[i % n] = splits[i % n] + line[i]

    return "".join(splits)

def splitCount(s, count):
     return [''.join(x) for x in zip(*[list(s[z::count]) for z in range(count)])]


def zipinterleaver(line, n):
    zipped = list(zip(*splitCount(line, n))) #split into n-character groups
    #print("zipped: " + str(zipped))
    stringlist = []
    for lst in zipped:
        stringlist.append(''.join(lst))

    return ''.join(stringlist)

def dezipinterleaver(line, n):
    split = splitCount(line, int(len(line)/n)) #split into interleaved groups
    unzipped = list(zip(*split))
    #print("unzipped: " + str(unzipped))

    stringlist = []
    for lst in unzipped:
        stringlist.append(''.join(lst))

    return ''.join(stringlist)

def spaced(line, n):
    lines = [line[i:i+n] for i in range(0, len(line), n)]
    line = ' '.join(lines)
    return line

def randomword(length):
   letters = string.printable
   return ''.join(random.choice(letters) for i in range(length))

def main():
    #while(true):
    textline = input("TX: ")

    #textline = randomword(10)
    textline = encodeBaudot(textline)
    #print(textline)
    #print(''.join(textline))
    original = ''.join(textline)
    n=5
    debug=True
    #original = str(text_to_bits(textline))
    if (debug):
        print("ORIG: " + spaced(original, n))

    interleaved = zipinterleaver(original, n)
    if (debug):
        print("ILV:  " + spaced(interleaved, n))

    decoded = dezipinterleaver(interleaved, n)
    if (debug):
        print("DEC:  " + spaced(decoded, n))
    print("LEN:  " + str(int(len(original)/4)))

    print("ORIG AND DEC MATCH? " + str(original == decoded))

    print("DECS: " + originaldecodeBaudot(decoded))

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


class MyApp(App):
    stop = threading.Event()

    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        softinput_mode = 'below_target'
        Window.softinput_mode = softinput_mode

    def _do_setup(self, *l):
        if platform == 'android':
            #pass
            self.root.padding = [0,0,0,dp(25)]

        #root.add_widget(FigureCanvasKivyAgg(plt.gcf()))


    def on_stop(self):
        self.stop.set()

    def testthread(self, *l):
        threading.Thread(target=testtx, args=(self,)).start()
        pass

    def build(self):
        self.root = Builder.load_file('guitest.kv')
        Clock.schedule_once(self._do_setup)
        Clock.schedule_once(self.testthread)
        return self.root

if __name__ == '__main__':
    MyApp().run()
