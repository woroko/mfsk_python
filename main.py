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
#from PIL import Image

from txrxlib import *

import threading
import time
from audiostream import get_output, AudioSample
from audiostream.sources.wave import SineSource
import struct
from functools import partial
import sounddevice as sd
import Queue

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

def testtone(context, text_to_tx):
    sendbits = string2bits(text_to_tx)
    bits = tx_bits(sendbits, None, M=2, CustomRs=50, CustomTsMult=1, CustomTsDiv=1, preamble=0)
    
    fill = np.zeros((7374), dtype=np.int16)
    #apply test offset
    bits = np.concatenate((np.squeeze(fill), bits))
    
    txg = bits.astype(np.int16).tolist()
    writeout = struct.pack('<' + 'h'*len(txg), *txg)
    
    context.sample.write(writeout)
    #print("---BITS---: " + str(bits))
    print("bits.shape: " + str(bits.shape))
    
    context.rec_queue.put(bits)
    
def record_process(context, rec_queue, duration):
    fs = 8000
    rec = None
    
    #while True:
    if platform == 'linux':
        rec = sd.rec(int(duration*fs), samplerate=fs, channels=1, blocking=True, dtype='int16')
        sd.wait()
        rec_queue.put(np.squeeze(rec))
     
def decode_sound(context, to_decode, dec_queue):
    decoded = rx_bits(to_decode, M=2, EbNodB=-1, CustomRs=50, CustomTsMult=1, CustomTsDiv=1)
    dec_queue.put(decoded)
    
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
        
        self.stream = get_output(channels=1, rate=8000, buffersize=128)
        self.sample = AudioSample()
        self.stream.add_sample(self.sample)
        self.sample.play()
        self.rec_queue = Queue.Queue()
        self.dec_queue = Queue.Queue()
        np.set_printoptions(threshold=np.inf)
        #root.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        
        #print(self.root.ids.ti0.ids)


    def on_stop(self):
        self.stop.set()

    def testthread(self, *l):
        #print(self.root.ids.txarea0.ids.msgIn.text)
        threading.Thread(target=testtone, args=(self,self.root.ids.txarea0.ids.msgIn.text)).start()
    
    def start_rec(self, duration):
        recordingThread = threading.Thread(target=record_process, args=(self,self.rec_queue,duration))
        recordingThread.start()
        
    def check_rec_dec_queue(self, dt):
        if (self.rec_queue.qsize() > 0):
            try:
                to_decode = self.rec_queue.get_nowait()
                print("to_decode shape: " + str(to_decode.shape))
                decodingThread = threading.Thread(target=decode_sound, args=(self,to_decode,self.dec_queue))
                decodingThread.start()
            except Queue.Empty:
                pass
            
        if (self.dec_queue.qsize() > 0):
            try:
                decode_result = self.dec_queue.get_nowait()
                self.root.ids.ti0.text += str(decode_result) + "\n\n---\n\n"
                #self.root.ids.ti0.text += "longest: " + 
            except Queue.Empty:
                pass
            

    def build(self):
        self.root = Builder.load_file('guitest.kv')
        Clock.schedule_once(self._do_setup)
        Clock.schedule_interval(self.check_rec_dec_queue, 0.1)
        #Clock.schedule_once(self.testthread)
        return self.root

if __name__ == '__main__':
    MyApp().run()
