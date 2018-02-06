import binascii

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
    zipped = list(zip(*splitCount(line, n)))
    #print("zipped: " + str(zipped))
    stringlist = []
    for lst in zipped:
        stringlist.append(''.join(lst))

    return ''.join(stringlist)

def dezipinterleaver(line, n):
    split = splitCount(line, int(len(line)/n))
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

def main():
    #while(true):
    textline = input("TX: ")
    n=4
    original = str(text_to_bits(textline))
    print("ORIG: " + spaced(original, n))
    #txSplit = [line[i:i+n] for i in range(0, len(line), n)]
    #interleaved = "".join(i for j in zip(*txSplit) for i in j)
    interleaved = zipinterleaver(original, n)
    print("ILV:  " + spaced(interleaved, n))

    decoded = dezipinterleaver(interleaved, n)
    print("DEC:  " + spaced(decoded, n))
    print("LEN:  " + str(int(len(original)/4)))

    print("ORIG AND DEC MATCH? " + str(original == decoded))





if __name__ == "__main__":
    main()
