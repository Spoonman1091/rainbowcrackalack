#!/usr/bin/env python3
"""
netntlmv1_hash_helper.py — compute the netntlmv1 hash of a 7-byte plaintext.

Algorithm mirrors the GPU implementation in CL/netntlmv1.cl (des_ecb_setkey_56 +
DES-ECB of the fixed challenge 1122334455667788) and the corrected CPU implementation
in cpu_rt_functions.c (setup_des_key + gcry_cipher_encrypt).

Usage:
    python3 netntlmv1_hash_helper.py <7-char plaintext>

    # numeric charset example — works on any box with pycryptodome:
    pip install pycryptodome
    python3 netntlmv1_hash_helper.py 1234567

Output line 1:  hash hex  (16 hex chars = 8 bytes)
Output line 2:  full lookup argument for crackalack_lookup
"""

import sys
from Crypto.Cipher import DES  # pip install pycryptodome


# Fixed 8-byte challenge — same as 'magic' in cpu_rt_functions.c and the
# pre-permuted constant X/Y in CL/netntlmv1.cl.
CHALLENGE = bytes.fromhex("1122334455667788")


def setup_des_key(key_56: bytes) -> bytes:
    """Expand a 7-byte (56-bit) DES key to 8 bytes with parity bits zeroed.

    Bit-for-bit identical to setup_des_key() in cpu_rt_functions.c and to
    des_ecb_setkey_56() in CL/netntlmv1.cl.
    """
    k = key_56
    return bytes([
        (((k[0] >> 1) & 0x7F) << 1),
        (((k[0] & 0x01) << 6 | ((k[1] >> 2) & 0x3F)) << 1),
        (((k[1] & 0x03) << 5 | ((k[2] >> 3) & 0x1F)) << 1),
        (((k[2] & 0x07) << 4 | ((k[3] >> 4) & 0x0F)) << 1),
        (((k[3] & 0x0F) << 3 | ((k[4] >> 5) & 0x07)) << 1),
        (((k[4] & 0x1F) << 2 | ((k[5] >> 6) & 0x03)) << 1),
        (((k[5] & 0x3F) << 1 | ((k[6] >> 7) & 0x01)) << 1),
        ((k[6] & 0x7F) << 1),
    ])


def netntlmv1_hash(plaintext: bytes) -> bytes:
    """Return the 8-byte netntlmv1 hash for a 7-byte plaintext."""
    if len(plaintext) != 7:
        raise ValueError(f"plaintext must be exactly 7 bytes, got {len(plaintext)}")
    key8 = setup_des_key(plaintext)
    return DES.new(key8, DES.MODE_ECB).encrypt(CHALLENGE)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <7-char plaintext>", file=sys.stderr)
        sys.exit(1)

    plaintext_str = sys.argv[1]
    try:
        plaintext_bytes = plaintext_str.encode("latin-1")
    except Exception as exc:
        print(f"Error encoding plaintext: {exc}", file=sys.stderr)
        sys.exit(1)

    if len(plaintext_bytes) != 7:
        print(
            f"Error: plaintext must be exactly 7 bytes, got {len(plaintext_bytes)}",
            file=sys.stderr,
        )
        sys.exit(1)

    hash_bytes = netntlmv1_hash(plaintext_bytes)
    hash_hex = hash_bytes.hex()

    print(f"plaintext : {plaintext_str!r}")
    print(f"hash      : {hash_hex}")
    print(f"challenge : 1122334455667788  (hardcoded; not part of lookup arg)")
    print(f"")
    print(f"Run lookup:")
    print(f"  ./crackalack_lookup <table_dir> {hash_hex}")
    print(f"")
    print(f"Expected output:")
    print(f"  HASH CRACKED => {hash_hex}:1122334455667788:{plaintext_bytes.hex()}")


if __name__ == "__main__":
    main()
