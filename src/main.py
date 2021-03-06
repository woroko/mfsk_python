from kivy.config import Config
Config.set('graphics', 'maxfps', '20') #more cpu for decoding
Config.set('graphics', 'multisamples', '0')

import numpy as np
import binascii
import random, string
#from baudot import *
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
if platform == 'android' or platform == 'ios':
    from audiostream import get_input
'''if platform == 'android':
    from jnius import autoclass'''
import struct
from functools import partial
if platform != 'android' and platform != 'ios':
    import sounddevice as sd
import Queue

#Window.size = (600,320)

global_rec_queue = Queue.Queue()

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

'''def main():
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

    print("DECS: " + originaldecodeBaudot(decoded))'''

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
    bits = tx_bits(sendbits, None, M=32, CustomRs=10, CustomTsMult=1, CustomTsDiv=1, preamble=0)

    #fill = np.zeros((7374), dtype=np.int16)
    #apply test offset
    #bits = np.concatenate((np.squeeze(fill), bits))

    txg = bits.astype(np.int16).tolist()
    writeout = struct.pack('<' + 'h'*len(txg), *txg)

    context.sample.write(writeout)


def decode_sound(context, to_decode, dec_queue):
    rx_bits_threaded(to_decode, dec_queue, M=32, EbNodB=-1, CustomRs=10, CustomTsMult=1, CustomTsDiv=1)
    #dec_queue.put(decoded)

def record_callback(indata, frames, time, status):
    global_rec_queue.put(np.copy(indata[::,0]))
    #print("callback!")

def record_callback_mobile(buf):
    #res = struct.unpack('<' + 'h'*len(buf), str(buf))
    global_rec_queue.put(np.frombuffer(buf, dtype=np.int16))
    #global_rec_queue.put(np.asarray(res, dtype=np.int16))
    #print("callback!")

class MyApp(App):
    #stop = threading.Event()

    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        softinput_mode = 'below_target'
        Window.softinput_mode = softinput_mode

    def _do_setup(self, *l):
        if platform == 'android':
            self.root.padding = [0,0,0,dp(25)]

        self.stream = get_output(channels=1, rate=8000)
        self.sample = AudioSample()
        self.stream.add_sample(self.sample)
        self.sample.play()
        self.dec_queue = Queue.Queue()
        self.start_rec()
        self.decodingThread = threading.Thread(target=decode_sound, args=(self,global_rec_queue,self.dec_queue))
        self.decodingThread.start()
        np.set_printoptions(threshold=np.inf)
        #root.add_widget(FigureCanvasKivyAgg(plt.gcf()))

        #print(self.root.ids.ti0.ids)

    def readbuffer(self, *l):
        self.mobile_input_stream.poll()

    def on_stop(self):
        try:
            self.input_stream.abort()
        except:
            try:
                self.mobile_input_stream.stop()
            except:
                print("inputstream close failed")
        self.sample.stop()
        #empty Queue
        while not global_rec_queue.empty():
            try:
                global_rec_queue.get(False)
            except Queue.Empty:
                continue
            global_rec_queue.task_done()

        global_rec_queue.put(None) #stop thread
        #self.root.stop.set()

    def testthread(self, *l):
        #print(self.root.ids.txarea0.ids.msgIn.text)
        threading.Thread(target=testtone, args=(self,self.root.ids.txarea0.ids.msgIn.text)).start()
        pass

    def start_rec(self):

        if platform == 'android':
            '''self.MediaRecorder = autoclass('android.media.MediaRecorder')
            self.AudioSource = autoclass('android.media.MediaRecorder$AudioSource')
            self.AudioFormat = autoclass('android.media.AudioFormat')
            self.AudioRecord = autoclass('android.media.AudioRecord')
            # define our system
            self.SampleRate = 8000
            self.ChannelConfig = self.AudioFormat.CHANNEL_IN_MONO
            self.AudioEncoding = self.AudioFormat.ENCODING_PCM_16BIT
            #self.BufferSize = self.AudioRecord.getMinBufferSize(self.SampleRate, self.ChannelConfig, self.AudioEncoding)*8
            #if self.BufferSize < 16384:
            #    self.BufferSize = 16384'''
            self.BufferSize = 16384
            self.mobile_input_stream = get_input(callback=record_callback_mobile, source='default', buffersize=self.BufferSize, rate=8000)
            self.mobile_input_stream.start()
            Clock.schedule_interval(self.readbuffer, 1/float(10))
        elif platform == 'ios':
            self.mobile_input_stream = get_input(callback=record_callback_mobile, source='default', rate=8000)
            self.mobile_input_stream.start()
            Clock.schedule_interval(self.readbuffer, 1/float(60))
        else:
            self.input_stream = sd.InputStream(samplerate=8000, channels=1, dtype='int16'\
            , callback=record_callback)
            self.input_stream.start()
        #recordingThread = threading.Thread(target=record_process, args=(self,self.rec_queue,duration))
        #recordingThread.start()

    def check_rec_dec_queue(self, dt):
        if (self.dec_queue.qsize() > 0):
            try:
                decode_result = self.dec_queue.get_nowait()
                self.root.ids.ti0.text += str(decode_result) + "\n\n---\n\n"
                #self.root.ids.ti0.text += "longest: " +
            except Queue.Empty:
                print("Empty!!!")
                pass

    def print_dec_status(self, *l):
        print("global_rec_queue size: " + str(global_rec_queue.qsize()))

    def build(self):
        self.root = Builder.load_file('guitest.kv')
        Clock.schedule_once(self._do_setup)
        Clock.schedule_interval(self.check_rec_dec_queue, 0.1)
        Clock.schedule_interval(self.print_dec_status, 4.0)
        #Clock.schedule_once(self.testthread)
        return self.root

if __name__ == '__main__':
    try:
        app = MyApp()
        app.run()
    except KeyboardInterrupt:
        app.stop()
