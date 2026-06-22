#!/usr/bin/env python3
"""
make_netntlmv1_test.py — Generate a tiny netntlmv1 rainbow table + test captures
for fast regression testing of crackalack_lookup without waiting for real tables.

Produces:
  <out_dir>/netntlmv1_numeric#7-7_0_<chain_len>x<num_chains>_0.rt
  <out_dir>/test_captures.txt   (5 lines: 3 valid, 1 ESS skip, 1 bad-challenge skip)
  <out_dir>/test_single.txt     (single-mode test — same capture as line 1)

Usage:
  pip install pycryptodome
  python3 make_netntlmv1_test.py [out_dir] [chain_len]

  Defaults: out_dir=. chain_len=100

Algorithm mirrors cpu_rt_functions.c (hash_to_index, index_to_plaintext) and
ntlmv1.c (ntlmv1_parse_capture, ntlmv1_recover_last2, ntlmv1_assemble).
"""

import os
import struct
import sys

from Crypto.Cipher import DES  # pip install pycryptodome

# ── Constants matching the C code ────────────────────────────────────────────
CHALLENGE = bytes.fromhex("1122334455667788")
CHARSET = b"0123456789"
CHARSET_LEN = 10
PT_LEN = 7
SPACE_TOTAL = 10 ** 7  # numeric#7-7: 10^7 = 10_000_000
REDUCTION_OFFSET = 0   # TABLE_INDEX_TO_REDUCTION_OFFSET(0) = 0 * 65536


# ── DES helpers (mirrors cpu_rt_functions.c: setup_des_key + netntlmv1_hash) ─

def setup_des_key(key_56: bytes) -> bytes:
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
    """8-byte DES-ECB of the fixed challenge under the 7-byte DES key."""
    assert len(plaintext) == 7
    return DES.new(setup_des_key(plaintext), DES.MODE_ECB).encrypt(CHALLENGE)


# ── Chain computation (mirrors cpu_rt_functions.c) ───────────────────────────

def index_to_plaintext(index: int) -> bytes:
    """Numeric base-10, 7 digits, left-padded with '0'. Right-to-left decode."""
    digits = []
    for _ in range(PT_LEN):
        digits.append(CHARSET[index % CHARSET_LEN])
        index //= CHARSET_LEN
    digits.reverse()
    return bytes(digits)


def plaintext_to_index(pt: bytes) -> int:
    """Inverse of index_to_plaintext for ASCII digit strings."""
    n = 0
    for b in pt:
        n = n * CHARSET_LEN + (b - ord('0'))
    return n


def hash_to_index(h: bytes, pos: int) -> int:
    """Little-endian uint64 of first 8 hash bytes + reduction_offset + pos, mod SPACE_TOTAL."""
    ret = int.from_bytes(h[:8], 'little')
    return (ret + REDUCTION_OFFSET + pos) % SPACE_TOTAL


def compute_chain_end(start_index: int, chain_len: int) -> int:
    """Run chain_len-1 hash/reduce steps, return the end index."""
    idx = start_index
    for pos in range(chain_len - 1):
        pt = index_to_plaintext(idx)
        h = netntlmv1_hash(pt)
        idx = hash_to_index(h, pos)
    return idx


# ── Capture synthesis (mirrors ntlmv1.c) ─────────────────────────────────────

def make_capture(username: str, k1: bytes, k2: bytes, k3: bytes = b'\x00' * 7):
    """
    Return (capture_line, expected_ntlm_hex).

    k3 defaults to all-zeros so ntlmv1_recover_last2 finds last2=[0,0] at v=0.
    LM response is all-zeros (no ESS trigger: any_nonzero_leading=0).
    """
    block1 = netntlmv1_hash(k1)
    block2 = netntlmv1_hash(k2)
    block3 = netntlmv1_hash(k3)
    nt_response = block1.hex() + block2.hex() + block3.hex()
    lm_response = "00" * 24
    capture = f"{username}::DOMAIN:{lm_response}:{nt_response}:1122334455667788"
    # ntlmv1_assemble: ntlm[0:7]=k1, ntlm[7:14]=k2, ntlm[14:16]=last2=k3[:2]
    ntlm16 = k1 + k2 + k3[:2]
    return capture, ntlm16.hex()


# ── Table file I/O (mirrors crackalack_tests.py: create_rt_table) ────────────

def write_rt_table(path: str, chains: list) -> None:
    """Write sorted (start_LE8, end_LE8) pairs — binary search requires end-sorted order."""
    sorted_chains = sorted(chains, key=lambda c: c[1])
    with open(path, 'wb') as f:
        for start, end in sorted_chains:
            f.write(struct.pack('<Q', start))
            f.write(struct.pack('<Q', end))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    chain_len = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    os.makedirs(out_dir, exist_ok=True)

    # Plantings: 3 distinct 7-byte DES keys (plaintexts) that will go in the table.
    # "1234567" and "7654321" → normal capture (2 distinct blocks, K1≠K2).
    # "5555555" → same-block capture (K1==K2, tests same_block path).
    plantings = [b"1234567", b"7654321", b"5555555"]

    print(f"Computing chains (chain_len={chain_len})...")
    chains = []
    for pt in plantings:
        start = plaintext_to_index(pt)
        end = compute_chain_end(start, chain_len)
        chains.append((start, end))
        print(f"  {pt.decode()!r}: start={start:>10}, end={end:>10}")

    # Write table
    num_chains = len(chains)
    table_name = f"netntlmv1_numeric#7-7_0_{chain_len}x{num_chains}_0.rt"
    table_path = os.path.join(out_dir, table_name)
    write_rt_table(table_path, chains)
    print(f"\nWrote table : {table_path}  ({num_chains} chains × 16 bytes = {num_chains * 16} bytes)")

    # Synthesize captures
    # 1. Normal (K1="1234567", K2="7654321") — 2 distinct block hashes
    cap1, nt1 = make_capture("Alice", b"1234567", b"7654321")
    # 2. Same-block (K1=K2="5555555") — tests same_block == 1 path
    cap2, nt2 = make_capture("Bob", b"5555555", b"5555555")
    # 3. Duplicate NT of cap1, different username — tests cross-line dedup:
    #    Alice and Carol share block1_hex and block2_hex → only 3 unique hashes total, not 6
    cap3, nt3 = make_capture("Carol", b"1234567", b"7654321")
    assert nt3 == nt1, "sanity: Carol == Alice"
    # 4. ESS line — first 8 LM bytes non-zero, trailing 16 all zero → NTLMV1_ERR_ESS
    ess_lm = "0102030405060708" + "00" * 16   # 16 + 32 = 48 hex chars
    cap4 = f"Dave::DOMAIN:{ess_lm}:{'aa' * 24}:1122334455667788"
    # 5. Wrong challenge → NTLMV1_ERR_CHALLENGE
    cap5 = f"Eve::DOMAIN:{'00' * 24}:{'bb' * 24}:deadbeefdeadbeef"

    captures_path = os.path.join(out_dir, "test_captures.txt")
    with open(captures_path, 'w') as f:
        for cap in [cap1, cap2, cap3, cap4, cap5]:
            f.write(cap + "\n")
    print(f"Wrote captures: {captures_path}  (5 lines)")

    single_path = os.path.join(out_dir, "test_single.txt")
    with open(single_path, 'w') as f:
        f.write(cap1 + "\n")
    print(f"Wrote single  : {single_path}")

    # Print summary
    b1 = netntlmv1_hash(b"1234567").hex()
    b2 = netntlmv1_hash(b"7654321").hex()
    b3 = netntlmv1_hash(b"5555555").hex()

    print(f"""
══════════════════════════════════════════════════════
EXPECTED BLOCK HASHES (what the table was built from):
  block("1234567") = {b1}
  block("7654321") = {b2}
  block("5555555") = {b3}

DEDUP CHECK:
  Total capture lines (valid):  3  (Alice, Bob, Carol)
  Total block references:        6  (Alice: 2, Bob: 2 same, Carol: 2 = Alice's)
  Unique block hashes:           3  ← "Pre-computing hash #1..#3" proves dedup

EXPECTED ASSEMBLED NT HASHES:
  Alice / Carol : {nt1}
  Bob           : {nt2}

TEST COMMANDS:
  Batch mode:
    ./crackalack_lookup {out_dir} -ntlmv1-file {captures_path}

  Single mode (regression — must match batch for Alice):
    ./crackalack_lookup {out_dir} -ntlmv1 '{cap1}'

EXPECTED BATCH OUTPUT:
  Pre-computing hash #1 of 3 ...   ← proves dedup (not "of 6")
  Pre-computing hash #2 of 3 ...
  Pre-computing hash #3 of 3 ...
  [Alice] Recovered NT hash: {nt1}
  [Bob]   Recovered NT hash: {nt2}
  [Carol] Recovered NT hash: {nt1}
  line 4 (Dave): skipped (ESS / NTLMv1-SSP)
  line 5 (Eve):  skipped (wrong challenge)
  NTLMv1 batch summary: 3 cracked, 0 partial, 0 malformed, 2 skipped (of 5 lines)
══════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
