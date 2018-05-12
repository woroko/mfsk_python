# mfsk_python
Multiplatform Kivy app for sending MFSK modulated text messages via (CB) radio.

Implemented: mfsk modulation and demodulation, simple baudot coding (needs baudot.py from https://github.com/bicycleprincess/AFSK , license unclear)
GUI for sending and receiving

Works on PC, Android and possibly iOS too).

TODO: Audio playback and recording, encryption (illegal in most countries on amateur, CB as well?)
Key exchange is out of question because of the low bit rate, perhaps AES-CTR with pre-shared key and a small number of bits for the counter in every message.

Uses parts of the mfsk octave scripts from the Codec2 project (https://svn.code.sf.net/p/freetel/code/codec2-dev/octave/), ported to Python

The aim of this project is to enable low symbol rate encrypted text messaging over long-distance, low-power, point-to-point radio links.
The targeted symbol rate is in the order of many seconds per symbol.

Never transmit if it is illegal in your country.

Info:
Error 'Didn't find class "org.audiostream.AudioIn" on path':

Solution: add AudioIn.java to dist org/audiostream/...
