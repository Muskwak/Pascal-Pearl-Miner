"""Dev-fee address, stored XOR-obfuscated so it does not appear as a plaintext
string in the shipped binary (defeats `strings`). Decoded once at import. This is
defense-in-depth, not real protection — anyone can recover it at runtime; the
point is only to not hand it over for free. The address and the 2% fee are
disclosed in the README and at miner startup.
"""

_K = b"p40\x9akeyA\x11z"
_E = (b'\x00F\\\xab\x1b\x03\x0cvh\x08FA\x06\xf7\r\x0e\x124"\x17\x18\x06T\xff\x12'
      b'\x10\x0e$v\x19\x1eDQ\xef\x05\x0f\x03ug\x16\x03BQ\xf0Y\x16\x11&#\x0b\x1a_Q'
      b'\xef\x13W\x08v\'\x0f\tAT')

DEV_ADDRESS = bytes(c ^ _K[i % len(_K)] for i, c in enumerate(_E)).decode()
