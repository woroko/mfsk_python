from math import ceil, floor
import numpy as np

def printf(format, *args):
    print(format % args)

def bitstring2array(bitstr):
    ret = np.zeros(len(bitstr), dtype=np.uint8)
    i=0
    for c in bitstr:
        if c == '0':
            ret[i] = 0
        else:
            ret[i] = 1
        i += 1
    return ret

class States:

    def __init__(self, Fs, Rs, M=2, TsMult=1, TsDiv=1):
        self.M = int(M)
        self.bitspersymbol = int(np.log2(M))
        self.Fs = int(Fs)
        self.Rs = int(Rs)

        self.nsym = 50                               # need enough symbols for good timing and freq offset est
        Ts = self.Ts = (Fs/Rs)*TsMult/TsDiv                         # number of samples per symbol
        #printf("Ts: #f\n", Ts)
        assert Ts == floor(Ts), "Fs/Rs must be an integer: "
        Ts = self.Ts = int(floor(Ts))

        N = self.N = Ts*self.nsym                  # processing buffer size, nice big window for timing est
        self.Ndft = np.minimum(1024, 2**ceil(np.log2(N)))      # find nearest power of 2 for efficient FFT
        self.nbit = self.nsym*self.bitspersymbol # number of bits per processing frame

        Nmem = self.Nmem  = int(N+2*Ts)                   # two symbol memory in down converted signals to allow for timing adj

        self.Sf = np.zeros(int(self.Ndft/2)) # current memory of dft mag samples
        self.f_dc = np.zeros((M,int(Nmem)), dtype=np.complex_)
        self.P = 8                                   # oversample rate out of filter
        #printf("Ts/P: #f\n", Ts/self.P)
        assert Ts/self.P == floor(Ts/self.P), "Ts/P must be an integer"

        self.nin = int(N)                                 # can be N +/- Ts/P samples to adjust for sample clock offsets
        self.verbose = 0
        self.phi = np.zeros(M)                       # keep down converter osc phase continuous

        #printf("M: #d Fs: #d Rs: #d Ts: #d nsym: #d nbit: #d\n", self.M, self.Fs, self.Rs, self.Ts, self.nsym, self.nbit)

        # BER stats

        self.ber_state = 0
        self.Tbits = 0
        self.Terrs = 0
        self.nerr_log = 0

        # extra simulation parameters

        self.tx_real = 1
        self.dA = np.zeros(M)
        self.dA[:] = 1
        self.df = np.zeros(M)
        self.df[:] = 0
        self.f = np.zeros(M)
        self.f[:] = 0
        self.norm_rx_timing = 0
        self.ppm = 0
        self.prev_pkt = []

        #python additions:
        self.ftx = 900 + 2*self.Rs*np.arange(1, self.M+1, dtype=np.float_) ##freqTx
        self.ntestframebits = self.nbit

        self.x = 1j
        self.timing_nl = 0.0
        self.rx_timing = 0.0
        self.norm_rx_timing = 0.0
        self.ppm = 0.0


    def fsk_mod(self, tx_bits):

        M  = self.M
        Ts = self.Ts
        Fs = self.Fs
        ftx  = self.ftx
        df = self.df # tone freq change in Hz/s
        dA = self.dA # amplitude of each tone

        num_bits = np.size(tx_bits)
        num_symbols = int(num_bits/self.bitspersymbol)
        tx = np.zeros(self.Ts*num_symbols)
        tx_phase = 0
        s = 1
        #tx = np.zeros()
        #print(tx)

        #for i=0:states.bitspersymbol:num_bits :
        for i in range(0, num_bits, self.bitspersymbol):
            # map bits to FSK symbol (tone number)
            K = self.bitspersymbol
            #np.dot treats as row or column vector based on argument position

            #print("arange: " + str(2**np.arange(K-1, -1, -1)))
            tone = np.dot(tx_bits[i:i+K], 2**np.arange(K-1, -1, -1))
            #print("bits: " + str(tx_bits[i:i+K]))
            #print("tone: " + str(tone))

            tx_phase_vec = tx_phase + np.arange(1,Ts+1)*2*np.pi*ftx[tone]/Fs
            tx_phase = tx_phase_vec[Ts-1] - np.floor(tx_phase_vec[Ts-1]/(2*np.pi))*2*np.pi

            #if states.tx_real == 1:
            #tx[(s-1)*Ts:s*Ts] = np.reshape(dA[tone]*2.0*np.cos(tx_phase_vec) , (Ts, 1))
            tx[(s-1)*Ts:s*Ts] = dA[tone]*2.0*np.cos(tx_phase_vec)
            #print("tx: " + str(tx))
            #else
            #    tx[(s-1)*Ts:s*Ts-1] = dA(tone)*exp(j*tx_phase_vec)
            s += 1
            # freq drift
            ftx = ftx + df*Ts/Fs

        self.ftx = ftx
        return tx

    # ------------------------------------------------------------------------------------
    # Given a buffer of nin input Rs baud FSK samples, returns nsym bits.
    #
    # nin is the number of input samples required by demodulator.  This is
    # time varying.  It will nominally be N (8000), and occasionally N +/-
    # Ts/2 (e.g. 8080 or 7920).  This is how we compensate for differences between the
    # remote tx sample clock and our sample clock.  This function always returns
    # N/Ts (e.g. 50) demodulated bits.  Variable number of input samples, constant number
    # of output bits.

    def fsk_demod(self, sf):
        debug = True
        M = self.M
        N = self.N
        Ndft = self.Ndft
        Fs = self.Fs
        Rs = self.Rs
        Ts = self.Ts
        nsym = self.nsym
        P = self.P
        nin = self.nin
        verbose = self.verbose
        Nmem = self.Nmem
        f = self.f

        assert np.size(sf) == nin

        # down convert and filter at rate P ------------------------------

        # update filter (integrator) memory by shifting in nin samples

        nold = Nmem-nin # number of old samples we retain

        f_dc = self.f_dc
        f_dc[:,0:nold] = f_dc[:,Nmem-nold:Nmem]

        # freq shift down to around DC, ensuring continuous phase from last frame

        phi_vec = np.zeros(N) #1, N

        for m in range(0,M):
            phi_vec = self.phi[m] + np.arange(1,nin+1)*2*np.pi*f[m]/Fs
            f_dc[m,nold:Nmem] = sf * np.exp(np.dot(1j,phi_vec))
            self.phi[m]  = phi_vec[nin-1]
            self.phi[m] -= 2*np.pi*floor(self.phi[m]/(2*np.pi))

        #print("f_dc: " + str(np.size(f_dc)))
        #print("self.phi: " + str(np.size(self.phi)))
        #print("Nmem: " + str(Nmem))

        print("freqshift!")

        # save filter (integrator) memory for next time

        self.f_dc = f_dc

        # integrate over symbol period, which is effectively a LPF, removing
        # the -2Fc frequency image.  Can also be interpreted as an ideal
        # integrate and dump, non-coherent demod.  We run the integrator at
        # rate P*Rs (1/P symbol offsets) to get outputs at a range of
        # different fine timing offsets.  We calculate integrator output
        # over nsym+1 symbols so we have extra samples for the fine timing
        # re-sampler at either end of the array.

        f_int = np.zeros((M, (nsym+1)*P), dtype=np.complex_)

        for i in range(0,(nsym+1)*P):
            st = int(0 + (i)*Ts/P)
            en = st+Ts
            f_int[0:M,i] = np.sum(f_dc[0:M,st:en], axis=1) #transpose removed

        self.f_int = f_int


        
        print("integrate!")

        # fine timing estimation -----------------------------------------------

        # Non linearity has a spectral line at Rs, with a phase
        # related to the fine timing offset.  See:
        #   http://www.rowetel.com/blog/?p=3573
        # We have sampled the integrator output at Fs=P samples/symbol, so
        # lets do a single point DFT at w = 2*pi*f/Fs = 2*pi*Rs/(P*Rs)
        #
        # Note timing non-lineariry derived by experiment.  Not quite sure what I'm doing here.....
        # but it gives 0dB impl loss for 2FSK Eb/No=9dB, testmode 1:
        #   Fs: 8000 Rs: 50 Ts: 160 nsym: 50
        #   frames: 200 Tbits: 9700 Terrs: 93 BER 0.010

        Np = np.size(f_int[0,:])
        w = 2*np.pi*(Rs)/(P*Rs)
        timing_nl = np.sum(np.abs(f_int[:,:])**2, axis=0)
        x = np.dot(timing_nl, np.exp(-1j*w*np.arange(0,Np)).transpose())
        norm_rx_timing = np.angle(x)/(2*np.pi)
        rx_timing = norm_rx_timing*P
        
        self.x = x
        self.timing_nl = timing_nl
        self.rx_timing = rx_timing
        prev_norm_rx_timing = self.norm_rx_timing
        self.norm_rx_timing = norm_rx_timing

        # estimate sample clock offset in ppm
        # d_norm_timing is fraction of symbol period shift over nsym symbols

        d_norm_rx_timing = norm_rx_timing - prev_norm_rx_timing

        # filter out big jumps due to nin changes
        #print("test!!!")
        #print(str(d_norm_rx_timing) + "\n <- d_norm_rx_timing")

        if abs(d_norm_rx_timing) < 0.2:
            appm = 1E6*d_norm_rx_timing/nsym
            self.ppm = 0.9*self.ppm + 0.1*appm

        # work out how many input samples we need on the next call. The aim
        # is to keep angle(x) away from the -pi/pi (+/- 0.5 fine timing
        # offset) discontinuity.  The side effect is to track sample clock
        # offsets

        next_nin = N
        if norm_rx_timing > 0.25:
            next_nin += Ts/2
            #print("next_nin + " + str(Ts/2))
        if norm_rx_timing < 0.25:
            next_nin -= Ts/2
            #print("next_nin - " + str(Ts/2))
        
        self.nin = int(next_nin)

        # Now we know the correct fine timing offset, Re-sample integrator
        # outputs using fine timing estimate and linear interpolation, then
        # extract the demodulated bits

        low_sample = int(floor(rx_timing))
        fract = rx_timing - low_sample
        high_sample = int(ceil(rx_timing))

        #if np.bitwise_and(verbose,0x2) == 1:
        print("rx_timing: {:3.2f} low_sample: {} high_sample: {} fract: {:3.3f} nin_next: {}\n".format(rx_timing, low_sample, high_sample, fract, next_nin))

        f_int_resample = np.zeros((M,nsym), dtype=np.complex_)
        rx_bits = np.zeros(nsym*self.bitspersymbol, dtype=np.uint8)
        tone_max = np.zeros(nsym, dtype=np.complex_)
        rx_bits_sd = np.zeros(nsym)
        #print(str(f_int) + "\n<- f_int")

        for i in range(0,nsym):
            st = i*P
            st_end = (i+1)*P
            if (st_end >= f_int.shape[1]):
                st_end = f_int.shape[1] - 1
                
            #print("st_end: " + str(st_end))
            #replaced by argmax above
            f_int_resample[:,i] = f_int[:,st+low_sample]*(1-fract) + f_int[:,st+high_sample]*fract
            #f_int_slice = f_int[:,st:st_end]
            #f_int_max_x, f_int_max_y = np.unravel_index(np.abs(f_int_slice).argmax(), f_int_slice.shape)
            
            #f_int_resample[:,i] = f_int[:,f_int_max_y]
            #print(str(f_int_resample[:,i]) + "<-detection: f_int_resample")
            #print(str(f_int[:,st+low_sample]) + "<-f_int low_sample")
            # Largest amplitude tone is the winner.  Map this FSK "symbol" back to a bunch-o-bits,
            # depending on M.
            # sample biggest magnitude, needs abs!!!!!
            tone_index = np.argmax(np.abs(f_int_resample[:,i]), axis=0)
            #tone_index = np.argmax(f_int_slice[:,5], axis=0)
            #print("f_int_max_y: " + str(f_int_max_y))
            #tone_index = f_int_max_x
            print("tone_index" + str(tone_index))
            #print("detection: tone_index=" + str(tone_index))
            tone_max[i] = f_int_resample[tone_index, i]
            st = (i)*self.bitspersymbol
            en = st + self.bitspersymbol
            arx_bits = bitstring2array(np.binary_repr(tone_index, self.bitspersymbol)) #- '0'
            #print("arx_bits\n" + str(arx_bits))
            rx_bits[st:en] = arx_bits


        print("detection!")

        self.f_int_resample = f_int_resample
        self.rx_bits_sd = rx_bits_sd

        # Eb/No estimation (todo: this needs some work, like calibration, low Eb/No perf)

        tone_max = np.abs(tone_max)
        self.EbNodB = -6 + 20*np.log10(1E-6+np.mean(tone_max)/(1E-6+np.std(tone_max)))

        return rx_bits


if __name__ == '__main__':
    print("Testing init:")
    states = States(8000, 50, 16, 1, 1)
    print(states.ftx)
