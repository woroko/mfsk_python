from fsklib import *
import numpy as np
import struct
import Queue

def printf(format, *args):
    print(format % args)

def string2bits(s=''):
    #tempstring = str([bin(ord(x))[2:].zfill(8) for x in s])
    tempstring = ''.join(format(ord(x), 'b') for x in s)
    #print("tempstring: " + tempstring)
    ret = np.zeros(len(tempstring), dtype='uint8')
    i=0
    for c in tempstring:
        ret[i] = int(str(c))
        i += 1

    #print("string2bits bit array:\n" + str(ret))
    return ret

def fsk_init(Fs,Rs,M=2,TsMult=1,TsDiv=1):

    states = States(Fs,Rs,M, TsMult, TsDiv)
    #states.rtty = fsk_horus_init_rtty_uw(states)
    #states.binary = fsk_horus_init_binary_uw

    # Freq. estimator limits - keep these narrow to stop errors with low SNR 4FSK

    states.fest_fmin = 800
    states.fest_fmax = 3000
    states.fest_min_spacing = 10
    return states

def tx_bits(tx_bits, ftx=None, M=64, CustomRs=50, CustomTsMult=1, CustomTsDiv=1, preamble=0):
    timing_offset = 0.0 # see resample() for clock offset below
    fading = 0          # modulates tx power at 2Hz with 20dB fade depth,
                       # to simulate balloon rotating at end of mission
    df     = 0          # tx tone freq drift in Hz/s
    dA     = 1          # amplitude imbalance of tones (note this affects Eb so not a gd idea)

    #print("CustomTsDiv: " + CustomTsDiv)
    states = fsk_init(8000, CustomRs, M, CustomTsMult, CustomTsDiv)
    states.ftx = 900 + 2*states.Rs*np.arange(1,states.M+1) ##freqTx
    states.ntestframebits = states.nbit

    states.verbose = 0x1
    M = states.M
    printf("N: %i", states.N)
    N = states.N
    P = states.P
    Rs = states.Rs
    nsym = states.nsym
    nbit = states.nbit
    Fs = states.Fs
    states.df[:] = df
    states.dA[:] = dA

    #tx_bits = ""
    pin = 0

    '''if (preamble == 1):
        #list(bin(10)[2:])
        combos = np.binary_repr(0:2^(states.bitspersymbol) - 1 - '0')
        tx_bits = [reshape(combos,[1,numel(combos)]) tx_bits]
        print("preamble!")
        #tx_bits'''


    '''for i in range(0, size(tx_bytes)):
        #bitstream = dec2bin(uint8(tx_bytes(i)))-'0'
        bitstream = np.binary_repr(tx_bytes[i])
        #bitstream = [1 0 1 0 1 0 1 1]
        tx_bits = tx_bits + bitstream'''

    diff_mod = len(tx_bits) % states.nbit

    if (len(tx_bits) > states.ntestframebits):
        padding_nbits = states.ntestframebits - diff_mod
    else:
        padding_nbits = states.ntestframebits - len(tx_bits)

    if (diff_mod != 0 or len(tx_bits) < states.ntestframebits):
        for i in range(0, int(floor(padding_nbits/2))):
            tx_bits = np.append(tx_bits, 0)
        for i in range(int(floor(padding_nbits/2)), padding_nbits):
            tx_bits = np.insert(tx_bits, 0, 0)

    printf("tx_bits_len: %i ,ntestframebits: %i , padding_nbits: %i",
    len(tx_bits), states.ntestframebits, padding_nbits)


    #tx_bits_array = np.fromstring(tx_bits,'u1') - ord('0')
    #tx_bits_array = np.array(map(int, tx_bits))

    #print("tx_bits: " + str(tx_bits))
    #print("TESTING")

    tx = states.fsk_mod(tx_bits)
    #ftx=fopen("fsk_horus.raw","wb")
    #ftx is fileTx
    #tx = vertcat(zeros(32758, 1), tx) #pad front

    txg = tx*3000 #Eri voluumi dekoodatessa ei haittaa..
    txret = txg.astype(np.int16)
    #txg = txg.astype(np.int16).tolist()
    #print(txg)
    '''if (ftx != None):
        ftx.write(struct.pack('<' + 'h'*len(txg), *txg))'''# fclose(ftx)

    return txret


def rx_bits(s16le_in, M=64, EbNodB=-1, CustomRs=50,
 CustomTsMult=1, CustomTsDiv=1):
    timing_offset = 0.0 # see resample() for clock offset below
    fading = 0                  # modulates tx power at 2Hz with 20dB fade depth,
                                             # to simulate balloon rotating at end of mission
    df       = 0                    # tx tone freq drift in Hz/s
    dA       = 1                    # amplitude imbalance of tones (note this affects Eb so not a gd idea)

    states = fsk_init(8000, CustomRs, M, CustomTsMult, CustomTsDiv)
    states.ftx = 900 + 2*states.Rs*np.arange(1,states.M+1) ##freqTx
    states.verbose = 0x1 + 0x8
    states.ntestframebits = states.nbit

    if (EbNodB>0):
        EbNo = 10**(EbNodB/10)
        variance = states.Fs/(states.Rs*EbNo*states.bitspersymbol)

    N = states.N
    P = states.P
    Rs = states.Rs
    nsym = states.nsym
    nbit = states.nbit

    #rx = []
    rx_bits_log = []
    #rx_bits_sd_log = []
    norm_rx_timing_log = []
    f_int_resample_log = []
    EbNodB_log = []
    ppm_log = []
    f_log = []
    rx_bits_buf = np.zeros(nbit + states.ntestframebits)

    frames = 0
    finished = False
    printf("states.nin: %i", states.nin)

    #rx_index = 0
    #upper_bound = int(np.size(s16le_in) / states.nin)
    previous_slice_end = 0
    rx_index = 0
    while(not finished):
        nin = states.nin
        #[sf count] = fread(fin, nin, "short")
        slice_end = previous_slice_end + nin
        #if (rx_index >= upper_bound):
        #   finished = True
        #slice_end = np.size(s16le_in)

        if (rx_index == 0):
            sf = s16le_in[0:slice_end]
        else:
            sf = s16le_in[previous_slice_end:slice_end]

        previous_slice_end = slice_end
        sf = sf.astype(np.float_) / 1000.0
        sf_len = np.size(sf)
        #rx = [rx sf]
        #rx = rx


        if (EbNodB>0):
            noise = np.sqrt(variance)*np.random.randn(count)
            sf += noise

        #print("sf_size: " + str(np.size(sf)))
        #print("nin: " + str(nin))

        if (sf_len == nin):

            frames += 1

            states.f = 900 + 2*states.Rs*np.arange(1,states.M+1)

            #states.f = [1450 1590 1710 1850]
            rx_bits = states.fsk_demod(sf)

            rx_bits_buf[0:states.ntestframebits] = rx_bits_buf[nbit:states.ntestframebits+nbit]
            rx_bits_buf[states.ntestframebits:states.ntestframebits+nbit] = rx_bits
            #print(str(rx_bits) + "\n<-rx_bits")
            rx_bits_log = rx_bits_log + rx_bits.tolist()
            norm_rx_timing_log.append(states.norm_rx_timing)
            f_int_resample_log = f_int_resample_log + np.abs(states.f_int_resample).tolist()
            EbNodB_log.append(states.EbNodB)
            ppm_log.append(states.ppm)
            f_log = f_log + states.f.tolist()
        else:
            finished = True

        rx_index += 1
    #print (str(rx_bits_log))
    #print("<- rx_bits_log")
    print(str(f_log) + "\n-<f_log!")
    return np.asarray(rx_bits_log, dtype=np.uint8)

def rx_bits_threaded(to_demod_queue, received_bits_queue, M=64, EbNodB=-1, CustomRs=50,
 CustomTsMult=1, CustomTsDiv=1):
    timing_offset = 0.0 # see resample() for clock offset below
    fading = 0                  # modulates tx power at 2Hz with 20dB fade depth,
                                             # to simulate balloon rotating at end of mission
    df       = 0                    # tx tone freq drift in Hz/s
    dA       = 1                    # amplitude imbalance of tones (note this affects Eb so not a gd idea)

    states = fsk_init(8000, CustomRs, M, CustomTsMult, CustomTsDiv)
    states.ftx = 900 + 2*states.Rs*np.arange(1,states.M+1) ##freqTx
    states.verbose = 0x1 + 0x8
    states.ntestframebits = states.nbit

    if (EbNodB>0):
        EbNo = 10**(EbNodB/10)
        variance = states.Fs/(states.Rs*EbNo*states.bitspersymbol)

    N = states.N
    P = states.P
    Rs = states.Rs
    nsym = states.nsym
    nbit = states.nbit

    #rx = []
    #rx_bits_sd_log = []
    norm_rx_timing_log = []
    f_int_resample_log = []
    EbNodB_log = []
    ppm_log = []
    f_log = []
    rx_bits_buf = np.zeros(nbit + states.ntestframebits)

    frames = 0
    finished = False

    rx_index = 0
    unused_samples = None
    while(not finished):
        nin = states.nin
        rx_bits_log = []
        sf = None

        if unused_samples is None:
            while True:
                try:
                    sf = to_demod_queue.get_nowait()
                    if (sf is None):
                        return
                    break
                except Queue.Empty:
                    pass
        else:
            sf = unused_samples
            unused_samples = None
        while len(sf) < nin: #need more samples
            while True:
                try:
                    res = to_demod_queue.get_nowait()
                    if (res is None):
                        return
                    break
                except Queue.Empty:
                    pass
            sf = np.concatenate((sf, res))

        if (len(sf) > nin):
            unused_samples = sf[nin:]
            sf = sf[0:nin]



        #previous_slice_end = slice_end
        sf = sf.astype(np.float_) / 1000.0
        sf_len = np.size(sf)
        #rx = [rx sf]
        #rx = rx


        if (EbNodB>0):
            noise = np.sqrt(variance)*np.random.randn(count)
            sf += noise

        #print("sf_size: " + str(np.size(sf)))
        #print("nin: " + str(nin))

        if (sf_len == nin):

            frames += 1

            states.f = 900 + 2*states.Rs*np.arange(1,states.M+1)

            #states.f = [1450 1590 1710 1850]
            rx_bits = states.fsk_demod(sf)

            rx_bits_buf[0:states.ntestframebits] = rx_bits_buf[nbit:states.ntestframebits+nbit]
            rx_bits_buf[states.ntestframebits:states.ntestframebits+nbit] = rx_bits
            #print(str(rx_bits) + "\n<-rx_bits")
            rx_bits_log.append(rx_bits.tolist())
            norm_rx_timing_log.append(states.norm_rx_timing)
            f_int_resample_log = f_int_resample_log + np.abs(states.f_int_resample).tolist()
            EbNodB_log.append(states.EbNodB)
            ppm_log.append(states.ppm)
            f_log = f_log + states.f.tolist()
        else:
            finished = True
            print("Error, incorrect number of samples")

        received_bits_queue.put(np.asarray(rx_bits_log, dtype=np.uint8))
        rx_index += 1
    #print (str(rx_bits_log))
    #print("<- rx_bits_log")
    #print(str(f_log) + "\n-<f_log!")
    #return np.asarray(rx_bits_log, dtype=np.uint8)


def main(context):
    np.set_printoptions(threshold=np.inf)
    print("Testing txrxlib:")
    #states = States(8000, 50, 16, 1, 1)
    sendbits = string2bits("7834fjj23098mv?P=)QJ#H/F)N632f89009JV74328\
    (?=(!/=j9GVBM09A*\
    7834fjj23098mv?P=)QJ#H/F)N632f89009JV74328\
    (?=(!/=j9GVBM09A*\
    7834fjj23098mv?P=)QJ#H/F)N632f89009JV74328\
    (?=(!/=j9GVBM09A*\
    7834fjj23098mv?P=)QJ#H/F)N632f89009JV74328\
    (?=(!/=j9GVBM09A*")

    with open("testout.raw", 'wb') as ftx:
        bits = tx_bits(sendbits, ftx, M=128, CustomRs=8, CustomTsMult=1, CustomTsDiv=1, preamble=0)

    decoded = rx_bits(bits, M=128, EbNodB=-1, CustomRs=8, CustomTsMult=1, CustomTsDiv=1)
    print("\n\ncompare: ")
    print(str(sendbits) + "\n\n")
    print(str(decoded))

    #with open("testout.raw", 'rb') as frx:
    #do not port freq detection!!

if __name__ == '__main__':
    main(1)
