/*
 * Rainbow Crackalack: cpu_rt_functions.c
 * Copyright (C) 2018-2019  Joe Testa <jtesta@positronsecurity.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms version 3 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include "cpu_rt_functions.h"
#include "shared.h"

#include <gcrypt.h>
#define GCRY_CIPHER GCRY_CIPHER_DES     // Use DES cipher
#define GCRY_MODE GCRY_CIPHER_MODE_ECB  // Use ECB mode
#define KEY_SIZE 8                      // DES key size in bytes
#define BLOCK_SIZE 8                    // DES block size in bytes
#ifdef _WIN32
#include <windows.h>
#endif



uint64_t fill_plaintext_space_table(unsigned int charset_len, unsigned int plaintext_len_min, unsigned int plaintext_len_max, uint64_t *plaintext_space_up_to_index) {
  uint64_t n = 1;
  int i;


  plaintext_space_up_to_index[0] = 0;
  for (i = 1; i <= plaintext_len_max; i++) {
    n = n * charset_len;
    if (i < plaintext_len_min)
      plaintext_space_up_to_index[i] = 0;
    else
      plaintext_space_up_to_index[i] = plaintext_space_up_to_index[i - 1] + n;
  }
  return plaintext_space_up_to_index[plaintext_len_max];
}


uint64_t hash_to_index(unsigned char *hash_value, unsigned int hash_len, unsigned int reduction_offset, uint64_t plaintext_space_total, unsigned int pos) {
  uint64_t ret = hash_value[7];
  ret <<= 8;
  ret |= hash_value[6];
  ret <<= 8;
  ret |= hash_value[5];
  ret <<= 8;
  ret |= hash_value[4];
  ret <<= 8;
  ret |= hash_value[3];
  ret <<= 8;
  ret |= hash_value[2];
  ret <<= 8;
  ret |= hash_value[1];
  ret <<= 8;
  ret |= hash_value[0];

  //printf("hash_to_index \treturn: %llu, ret: %llu, reduction_offset: %u, pos: %u, plaintext_space_total: %llu\n", (ret + reduction_offset + pos) % plaintext_space_total, ret, reduction_offset, pos, plaintext_space_total);

  return (ret + reduction_offset + pos) % plaintext_space_total;
}


void index_to_plaintext(uint64_t index, char *charset, unsigned int charset_len, unsigned int plaintext_len_min, unsigned int plaintext_len_max, uint64_t *plaintext_space_up_to_index, char *plaintext, unsigned int *plaintext_len) {
  int i;
  uint64_t index_x;

  //printf("************************************** this function is CPU not GPU, index: %llu\n", index);

  for (i = plaintext_len_max - 1; i >= plaintext_len_min - 1; i--) {
    if (index >= plaintext_space_up_to_index[i]) {
      *plaintext_len = i + 1;
      if (*plaintext_len >= MAX_PLAINTEXT_LEN)
	return;

      plaintext[*plaintext_len] = '\0';
      break;
    }
  }

  index_x = index - plaintext_space_up_to_index[*plaintext_len - 1];
  for (i = *plaintext_len - 1; i >= 0; i--) {
    plaintext[i] = charset[index_x % charset_len];
    //printf("appending %02x \n", plaintext[i]);
    index_x = index_x / charset_len;
  }

  return;
}


uint64_t generate_rainbow_chain(
    unsigned int hash_type,
    char *charset,
    unsigned int charset_len,
    unsigned int plaintext_len_min,
    unsigned int plaintext_len_max,
    unsigned int reduction_offset,
    unsigned int chain_len,
    uint64_t start,
    uint64_t *plaintext_space_up_to_index,
    uint64_t plaintext_space_total,
    char *plaintext,
    unsigned int *plaintext_len,
    unsigned char *hash,
    unsigned int *hash_len) {
  uint64_t index = start;
  unsigned int pos = 0;


  if ((hash_type != HASH_NTLM) && (hash_type != HASH_NETNTLMV1))
    fprintf(stderr, "\n\tWARNING: only NTLM and NetNTLMv1 hashes are currently supported!\n\n");

  /* NetNTLMv1 uses 8-byte DES hashes; NTLM uses 16-byte MD4 hashes. */
  if (hash_type == HASH_NETNTLMV1)
    *hash_len = 8;

  for (; pos < chain_len - 1; pos++) {
    index_to_plaintext(index, charset, charset_len, plaintext_len_min, plaintext_len_max, plaintext_space_up_to_index, plaintext, plaintext_len);
    if (hash_type == HASH_NETNTLMV1)
      netntlmv1_hash_nocheck((const unsigned char *)plaintext, hash);
    else
      ntlm_hash(plaintext, *plaintext_len, hash);
    index = hash_to_index(hash, *hash_len, reduction_offset, plaintext_space_total, pos);
  }
  return index;
}


/* Calculates the NTLM hash on the specified plaintext.  The result is stored in the hash
 * argument, which must be at least 16 bytes in size. */
void ntlm_hash(char *plaintext, unsigned int plaintext_len, unsigned char *hash) {
  unsigned int key[16] = {0};
  unsigned int output[4];
  int i = 0;


  if (plaintext_len > 27) {
    plaintext[27] = 0;
    plaintext_len = 27;
  }

  for (; i < (plaintext_len / 2); i++)
    key[i] = plaintext[i * 2] | (plaintext[(i * 2) + 1] << 16);

  if ((plaintext_len % 2) == 1)
    key[i] = plaintext[plaintext_len - 1] | 0x800000;
  else
    key[i] = 0x80;

  key[14] = plaintext_len << 4;

  md4_encrypt(output, key);

  i = 0;
  hash[i++] = ((output[0] >> 0) & 0xff);
  hash[i++] = ((output[0] >> 8) & 0xff);
  hash[i++] = ((output[0] >> 16) & 0xff);
  hash[i++] = ((output[0] >> 24) & 0xff);
  hash[i++] = ((output[1] >> 0) & 0xff);
  hash[i++] = ((output[1] >> 8) & 0xff);
  hash[i++] = ((output[1] >> 16) & 0xff);
  hash[i++] = ((output[1] >> 24) & 0xff);
  hash[i++] = ((output[2] >> 0) & 0xff);
  hash[i++] = ((output[2] >> 8) & 0xff);
  hash[i++] = ((output[2] >> 16) & 0xff);
  hash[i++] = ((output[2] >> 24) & 0xff);
  hash[i++] = ((output[3] >> 0) & 0xff);
  hash[i++] = ((output[3] >> 8) & 0xff);
  hash[i++] = ((output[3] >> 16) & 0xff);
  hash[i++] = ((output[3] >> 24) & 0xff);
}


/* The below copyright notice applies to the md4_encrypt() function only. */

/*
 * MD4 OpenCL kernel based on Solar Designer's MD4 algorithm implementation at:
 * http://openwall.info/wiki/people/solar/software/public-domain-source-code/md4
 * This code is in public domain.
 *
 * This software is Copyright (c) 2010, Dhiru Kholia <dhiru.kholia at gmail.com>
 * and Copyright (c) 2012, magnum
 * and Copyright (c) 2015, Sayantan Datta <std2048@gmail.com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted.
 *
 * Useful References:
 * 1  nt_opencl_kernel.c (written by Alain Espinosa <alainesp at gmail.com>)
 * 2. http://tools.ietf.org/html/rfc1320
 * 3. http://en.wikipedia.org/wiki/MD4
 */

#define F(x, y, z)	(z ^ (x & (y ^ z)))
#define G(x, y, z)	(((x) & ((y) | (z))) | ((y) & (z)))
#define H(x, y, z)	(((x) ^ (y)) ^ (z))
#define H2(x, y, z)	((x) ^ ((y) ^ (z)))

/* The MD4 transformation for all three rounds. */
#define STEP(f, a, b, c, d, x, s)	  \
	(a) += f((b), (c), (d)) + (x); \
	(a) = ((a << s) | (a >> (32 - s)))
	//(a) = rotate((a), (uint)(s)) //(a) = ((a << s) | (a >> (32 - s))) 

void md4_encrypt(unsigned int *hash, unsigned int *W)
{
	hash[0] = 0x67452301;
	hash[1] = 0xefcdab89;
	hash[2] = 0x98badcfe;
	hash[3] = 0x10325476;

	/* Round 1 */
	STEP(F, hash[0], hash[1], hash[2], hash[3], W[0], 3);
	STEP(F, hash[3], hash[0], hash[1], hash[2], W[1], 7);
	STEP(F, hash[2], hash[3], hash[0], hash[1], W[2], 11);
	STEP(F, hash[1], hash[2], hash[3], hash[0], W[3], 19);
	STEP(F, hash[0], hash[1], hash[2], hash[3], W[4], 3);
	STEP(F, hash[3], hash[0], hash[1], hash[2], W[5], 7);
	STEP(F, hash[2], hash[3], hash[0], hash[1], W[6], 11);
	STEP(F, hash[1], hash[2], hash[3], hash[0], W[7], 19);
	STEP(F, hash[0], hash[1], hash[2], hash[3], W[8], 3);
	STEP(F, hash[3], hash[0], hash[1], hash[2], W[9], 7);
	STEP(F, hash[2], hash[3], hash[0], hash[1], W[10], 11);
	STEP(F, hash[1], hash[2], hash[3], hash[0], W[11], 19);
	STEP(F, hash[0], hash[1], hash[2], hash[3], W[12], 3);
	STEP(F, hash[3], hash[0], hash[1], hash[2], W[13], 7);
	STEP(F, hash[2], hash[3], hash[0], hash[1], W[14], 11);
	STEP(F, hash[1], hash[2], hash[3], hash[0], W[15], 19);

	/* Round 2 */
	STEP(G, hash[0], hash[1], hash[2], hash[3], W[0] + 0x5a827999, 3);
	STEP(G, hash[3], hash[0], hash[1], hash[2], W[4] + 0x5a827999, 5);
	STEP(G, hash[2], hash[3], hash[0], hash[1], W[8] + 0x5a827999, 9);
	STEP(G, hash[1], hash[2], hash[3], hash[0], W[12] + 0x5a827999, 13);
	STEP(G, hash[0], hash[1], hash[2], hash[3], W[1] + 0x5a827999, 3);
	STEP(G, hash[3], hash[0], hash[1], hash[2], W[5] + 0x5a827999, 5);
	STEP(G, hash[2], hash[3], hash[0], hash[1], W[9] + 0x5a827999, 9);
	STEP(G, hash[1], hash[2], hash[3], hash[0], W[13] + 0x5a827999, 13);
	STEP(G, hash[0], hash[1], hash[2], hash[3], W[2] + 0x5a827999, 3);
	STEP(G, hash[3], hash[0], hash[1], hash[2], W[6] + 0x5a827999, 5);
	STEP(G, hash[2], hash[3], hash[0], hash[1], W[10] + 0x5a827999, 9);
	STEP(G, hash[1], hash[2], hash[3], hash[0], W[14] + 0x5a827999, 13);
	STEP(G, hash[0], hash[1], hash[2], hash[3], W[3] + 0x5a827999, 3);
	STEP(G, hash[3], hash[0], hash[1], hash[2], W[7] + 0x5a827999, 5);
	STEP(G, hash[2], hash[3], hash[0], hash[1], W[11] + 0x5a827999, 9);
	STEP(G, hash[1], hash[2], hash[3], hash[0], W[15] + 0x5a827999, 13);

	/* Round 3 */
	STEP(H, hash[0], hash[1], hash[2], hash[3], W[0] + 0x6ed9eba1, 3);
	STEP(H2, hash[3], hash[0], hash[1], hash[2], W[8] + 0x6ed9eba1, 9);
	STEP(H, hash[2], hash[3], hash[0], hash[1], W[4] + 0x6ed9eba1, 11);
	STEP(H2, hash[1], hash[2], hash[3], hash[0], W[12] + 0x6ed9eba1, 15);
	STEP(H, hash[0], hash[1], hash[2], hash[3], W[2] + 0x6ed9eba1, 3);
	STEP(H2, hash[3], hash[0], hash[1], hash[2], W[10] + 0x6ed9eba1, 9);
	STEP(H, hash[2], hash[3], hash[0], hash[1], W[6] + 0x6ed9eba1, 11);
	STEP(H2, hash[1], hash[2], hash[3], hash[0], W[14] + 0x6ed9eba1, 15);
	STEP(H, hash[0], hash[1], hash[2], hash[3], W[1] + 0x6ed9eba1, 3);
	STEP(H2, hash[3], hash[0], hash[1], hash[2], W[9] + 0x6ed9eba1, 9);
	STEP(H, hash[2], hash[3], hash[0], hash[1], W[5] + 0x6ed9eba1, 11);
	STEP(H2, hash[1], hash[2], hash[3], hash[0], W[13] + 0x6ed9eba1, 15);
	STEP(H, hash[0], hash[1], hash[2], hash[3], W[3] + 0x6ed9eba1, 3);
	STEP(H2, hash[3], hash[0], hash[1], hash[2], W[11] + 0x6ed9eba1, 9);
	STEP(H, hash[2], hash[3], hash[0], hash[1], W[7] + 0x6ed9eba1, 11);
	STEP(H2, hash[1], hash[2], hash[3], hash[0], W[15] + 0x6ed9eba1, 15);

	hash[0] = hash[0] + 0x67452301;
	hash[1] = hash[1] + 0xefcdab89;
	hash[2] = hash[2] + 0x98badcfe;
	hash[3] = hash[3] + 0x10325476;
}

void setup_des_key(char key_56[], unsigned char *key)
{
  //char key[8]= {0};
/*
  key[0] = key_56[0];
  key[1] = (key_56[0] << 7) | (key_56[1] >> 1);
  key[2] = (key_56[1] << 6) | (key_56[2] >> 2);
  key[3] = (key_56[2] << 5) | (key_56[3] >> 3);
  key[4] = (key_56[3] << 4) | (key_56[4] >> 4);
  key[5] = (key_56[4] << 3) | (key_56[5] >> 5);
  key[6] = (key_56[5] << 2) | (key_56[6] >> 6);
  key[7] = (key_56[6] << 1);
*/
  key[0] = (((key_56[0] >> 1) & 0x7f) << 1);
  key[1] = (((key_56[0] & 0x01) << 6 | ((key_56[1] >> 2) & 0x3f)) << 1);
  key[2] = (((key_56[1] & 0x03) << 5 | ((key_56[2] >> 3) & 0x1f)) << 1);
  key[3] = (((key_56[2] & 0x07) << 4 | ((key_56[3] >> 4) & 0x0f)) << 1);
  key[4] = (((key_56[3] & 0x0f) << 3 | ((key_56[4] >> 5) & 0x07)) << 1);
  key[5] = (((key_56[4] & 0x1f) << 2 | ((key_56[5] >> 6) & 0x03)) << 1);
  key[6] = (((key_56[5] & 0x3f) << 1 | ((key_56[6] >> 7) & 0x01)) << 1);
  key[7] = ((key_56[6] & 0x7f) << 1);
}

/*
void HashNetNTLMv1(
  unsigned char *pData,
  unsigned int  uLen,   // uLen == 7
  unsigned char Hash[8])
{
  */
void netntlmv1_hash(unsigned char *plaintext, unsigned int plaintext_len, unsigned char *hash) {
    gcry_control(GCRYCTL_DISABLE_SECMEM, 0); // Disable secure memory (optional)
    gcry_control(GCRYCTL_INITIALIZATION_FINISHED, 0);

    gcry_cipher_hd_t handle;
    gcry_error_t err;

    // Fixed 8-byte challenge used for all netntlmv1 hashes.
    unsigned char magic[KEY_SIZE] = { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88 };

    // Expand the 7-byte plaintext to an 8-byte DES key (parity bits zeroed).
    // This matches des_ecb_setkey_56() in CL/netntlmv1.cl, which is what the
    // GPU uses when generating chains.  Without this expansion gcry_cipher_setkey
    // would receive a 7-byte key and return GPG_ERR_INV_KEYLEN, causing the
    // function to return early and leave 'hash' unwritten.
    unsigned char des_key[KEY_SIZE];
    setup_des_key((char *)plaintext, des_key);

    // Open cipher context
    err = gcry_cipher_open(&handle, GCRY_CIPHER, GCRY_MODE, 0);
    if (err) {
        fprintf(stderr, "Failed to open cipher: %s\n", gcry_strerror(err));
        return;
    }

    // Allow weak DES keys (e.g., all-zero key from last2=[0,0] in block3 brute-force).
    // Without this, gcry_cipher_setkey returns GPG_ERR_WEAK_KEY and wipes the key
    // schedule, causing ntlmv1_recover_last2 to miss captures whose last2 bytes
    // expand to a DES weak key.
    gcry_cipher_ctl(handle, GCRYCTL_SET_ALLOW_WEAK_KEY, NULL, 0);

    // Set the expanded 8-byte DES key
    err = gcry_cipher_setkey(handle, des_key, KEY_SIZE);
    if (err) {
        fprintf(stderr, "Failed to set key: %s\n", gcry_strerror(err));
        gcry_cipher_close(handle);
        return;
    }

    // Encrypt the challenge
    err = gcry_cipher_encrypt(handle, hash, BLOCK_SIZE, magic, BLOCK_SIZE);
    if (err) {
        fprintf(stderr, "Encryption failed: %s\n", gcry_strerror(err));
        gcry_cipher_close(handle);
        return;
    }

    // Clean up
    gcry_cipher_close(handle);
}

/* --------------------------------------------------------------------------
 * Standalone table-driven DES ECB for ntlmv1_recover_last2 block3 brute-force.
 * Bypasses libgcrypt to avoid weak-key rejection on older versions.
 * Tables and key schedule ported directly from CL/netntlmv1.cl (GPU path).
 * -------------------------------------------------------------------------- */

static const uint32_t nocheck_SB1[64] = {
  0x01010400, 0x00000000, 0x00010000, 0x01010404,
  0x01010004, 0x00010404, 0x00000004, 0x00010000,
  0x00000400, 0x01010400, 0x01010404, 0x00000400,
  0x01000404, 0x01010004, 0x01000000, 0x00000004,
  0x00000404, 0x01000400, 0x01000400, 0x00010400,
  0x00010400, 0x01010000, 0x01010000, 0x01000404,
  0x00010004, 0x01000004, 0x01000004, 0x00010004,
  0x00000000, 0x00000404, 0x00010404, 0x01000000,
  0x00010000, 0x01010404, 0x00000004, 0x01010000,
  0x01010400, 0x01000000, 0x01000000, 0x00000400,
  0x01010004, 0x00010000, 0x00010400, 0x01000004,
  0x00000400, 0x00000004, 0x01000404, 0x00010404,
  0x01010404, 0x00010004, 0x01010000, 0x01000404,
  0x01000004, 0x00000404, 0x00010404, 0x01010400,
  0x00000404, 0x01000400, 0x01000400, 0x00000000,
  0x00010004, 0x00010400, 0x00000000, 0x01010004
};

static const uint32_t nocheck_SB2[64] = {
  0x80108020, 0x80008000, 0x00008000, 0x00108020,
  0x00100000, 0x00000020, 0x80100020, 0x80008020,
  0x80000020, 0x80108020, 0x80108000, 0x80000000,
  0x80008000, 0x00100000, 0x00000020, 0x80100020,
  0x00108000, 0x00100020, 0x80008020, 0x00000000,
  0x80000000, 0x00008000, 0x00108020, 0x80100000,
  0x00100020, 0x80000020, 0x00000000, 0x00108000,
  0x00008020, 0x80108000, 0x80100000, 0x00008020,
  0x00000000, 0x00108020, 0x80100020, 0x00100000,
  0x80008020, 0x80100000, 0x80108000, 0x00008000,
  0x80100000, 0x80008000, 0x00000020, 0x80108020,
  0x00108020, 0x00000020, 0x00008000, 0x80000000,
  0x00008020, 0x80108000, 0x00100000, 0x80000020,
  0x00100020, 0x80008020, 0x80000020, 0x00100020,
  0x00108000, 0x00000000, 0x80008000, 0x00008020,
  0x80000000, 0x80100020, 0x80108020, 0x00108000
};

static const uint32_t nocheck_SB3[64] = {
  0x00000208, 0x08020200, 0x00000000, 0x08020008,
  0x08000200, 0x00000000, 0x00020208, 0x08000200,
  0x00020008, 0x08000008, 0x08000008, 0x00020000,
  0x08020208, 0x00020008, 0x08020000, 0x00000208,
  0x08000000, 0x00000008, 0x08020200, 0x00000200,
  0x00020200, 0x08020000, 0x08020008, 0x00020208,
  0x08000208, 0x00020200, 0x00020000, 0x08000208,
  0x00000008, 0x08020208, 0x00000200, 0x08000000,
  0x08020200, 0x08000000, 0x00020008, 0x00000208,
  0x00020000, 0x08020200, 0x08000200, 0x00000000,
  0x00000200, 0x00020008, 0x08020208, 0x08000200,
  0x08000008, 0x00000200, 0x00000000, 0x08020008,
  0x08000208, 0x00020000, 0x08000000, 0x08020208,
  0x00000008, 0x00020208, 0x00020200, 0x08000008,
  0x08020000, 0x08000208, 0x00000208, 0x08020000,
  0x00020208, 0x00000008, 0x08020008, 0x00020200
};

static const uint32_t nocheck_SB4[64] = {
  0x00802001, 0x00002081, 0x00002081, 0x00000080,
  0x00802080, 0x00800081, 0x00800001, 0x00002001,
  0x00000000, 0x00802000, 0x00802000, 0x00802081,
  0x00000081, 0x00000000, 0x00800080, 0x00800001,
  0x00000001, 0x00002000, 0x00800000, 0x00802001,
  0x00000080, 0x00800000, 0x00002001, 0x00002080,
  0x00800081, 0x00000001, 0x00002080, 0x00800080,
  0x00002000, 0x00802080, 0x00802081, 0x00000081,
  0x00800080, 0x00800001, 0x00802000, 0x00802081,
  0x00000081, 0x00000000, 0x00000000, 0x00802000,
  0x00002080, 0x00800080, 0x00800081, 0x00000001,
  0x00802001, 0x00002081, 0x00002081, 0x00000080,
  0x00802081, 0x00000081, 0x00000001, 0x00002000,
  0x00800001, 0x00002001, 0x00802080, 0x00800081,
  0x00002001, 0x00002080, 0x00800000, 0x00802001,
  0x00000080, 0x00800000, 0x00002000, 0x00802080
};

static const uint32_t nocheck_SB5[64] = {
  0x00000100, 0x02080100, 0x02080000, 0x42000100,
  0x00080000, 0x00000100, 0x40000000, 0x02080000,
  0x40080100, 0x00080000, 0x02000100, 0x40080100,
  0x42000100, 0x42080000, 0x00080100, 0x40000000,
  0x02000000, 0x40080000, 0x40080000, 0x00000000,
  0x40000100, 0x42080100, 0x42080100, 0x02000100,
  0x42080000, 0x40000100, 0x00000000, 0x42000000,
  0x02080100, 0x02000000, 0x42000000, 0x00080100,
  0x00080000, 0x42000100, 0x00000100, 0x02000000,
  0x40000000, 0x02080000, 0x42000100, 0x40080100,
  0x02000100, 0x40000000, 0x42080000, 0x02080100,
  0x40080100, 0x00000100, 0x02000000, 0x42080000,
  0x42080100, 0x00080100, 0x42000000, 0x42080100,
  0x02080000, 0x00000000, 0x40080000, 0x42000000,
  0x00080100, 0x02000100, 0x40000100, 0x00080000,
  0x00000000, 0x40080000, 0x02080100, 0x40000100
};

static const uint32_t nocheck_SB6[64] = {
  0x20000010, 0x20400000, 0x00004000, 0x20404010,
  0x20400000, 0x00000010, 0x20404010, 0x00400000,
  0x20004000, 0x00404010, 0x00400000, 0x20000010,
  0x00400010, 0x20004000, 0x20000000, 0x00004010,
  0x00000000, 0x00400010, 0x20004010, 0x00004000,
  0x00404000, 0x20004010, 0x00000010, 0x20400010,
  0x20400010, 0x00000000, 0x00404010, 0x20404000,
  0x00004010, 0x00404000, 0x20404000, 0x20000000,
  0x20004000, 0x00000010, 0x20400010, 0x00404000,
  0x20404010, 0x00400000, 0x00004010, 0x20000010,
  0x00400000, 0x20004000, 0x20000000, 0x00004010,
  0x20000010, 0x20404010, 0x00404000, 0x20400000,
  0x00404010, 0x20404000, 0x00000000, 0x20400010,
  0x00000010, 0x00004000, 0x20400000, 0x00404010,
  0x00004000, 0x00400010, 0x20004010, 0x00000000,
  0x20404000, 0x20000000, 0x00400010, 0x20004010
};

static const uint32_t nocheck_SB7[64] = {
  0x00200000, 0x04200002, 0x04000802, 0x00000000,
  0x00000800, 0x04000802, 0x00200802, 0x04200800,
  0x04200802, 0x00200000, 0x00000000, 0x04000002,
  0x00000002, 0x04000000, 0x04200002, 0x00000802,
  0x04000800, 0x00200802, 0x00200002, 0x04000800,
  0x04000002, 0x04200000, 0x04200800, 0x00200002,
  0x04200000, 0x00000800, 0x00000802, 0x04200802,
  0x00200800, 0x00000002, 0x04000000, 0x00200800,
  0x04000000, 0x00200800, 0x00200000, 0x04000802,
  0x04000802, 0x04200002, 0x04200002, 0x00000002,
  0x00200002, 0x04000000, 0x04000800, 0x00200000,
  0x04200800, 0x00000802, 0x00200802, 0x04200800,
  0x00000802, 0x04000002, 0x04200802, 0x04200000,
  0x00200800, 0x00000000, 0x00000002, 0x04200802,
  0x00000000, 0x00200802, 0x04200000, 0x00000800,
  0x04000002, 0x04000800, 0x00000800, 0x00200002
};

static const uint32_t nocheck_SB8[64] = {
  0x10001040, 0x00001000, 0x00040000, 0x10041040,
  0x10000000, 0x10001040, 0x00000040, 0x10000000,
  0x00040040, 0x10040000, 0x10041040, 0x00041000,
  0x10041000, 0x00041040, 0x00001000, 0x00000040,
  0x10040000, 0x10000040, 0x10001000, 0x00001040,
  0x00041000, 0x00040040, 0x10040040, 0x10041000,
  0x00001040, 0x00000000, 0x00000000, 0x10040040,
  0x10000040, 0x10001000, 0x00041040, 0x00040000,
  0x00041040, 0x00040000, 0x10041000, 0x00001000,
  0x00000040, 0x10040040, 0x00001000, 0x00041040,
  0x10001000, 0x00000040, 0x10000040, 0x10040000,
  0x10040040, 0x10000000, 0x00040000, 0x10001040,
  0x00000000, 0x10041040, 0x00040040, 0x10000040,
  0x10040000, 0x10001000, 0x10001040, 0x00000000,
  0x10041040, 0x00041000, 0x00041000, 0x00001040,
  0x00001040, 0x00040040, 0x10000000, 0x10041000
};

static const uint32_t nocheck_LHs[16] = {
  0x00000000, 0x00000001, 0x00000100, 0x00000101,
  0x00010000, 0x00010001, 0x00010100, 0x00010101,
  0x01000000, 0x01000001, 0x01000100, 0x01000101,
  0x01010000, 0x01010001, 0x01010100, 0x01010101
};

static const uint32_t nocheck_RHs[16] = {
  0x00000000, 0x01000000, 0x00010000, 0x01010000,
  0x00000100, 0x01000100, 0x00010100, 0x01010100,
  0x00000001, 0x01000001, 0x00010001, 0x01010001,
  0x00000101, 0x01000101, 0x00010101, 0x01010101
};

static void nocheck_des_setkey(uint32_t SK[32], const unsigned char key[8])
{
  int i;
  uint32_t X, Y, T;

  X = ((uint32_t)key[0] << 24) | ((uint32_t)key[1] << 16)
    | ((uint32_t)key[2] <<  8) | ((uint32_t)key[3]);
  Y = ((uint32_t)key[4] << 24) | ((uint32_t)key[5] << 16)
    | ((uint32_t)key[6] <<  8) | ((uint32_t)key[7]);

  T = ((Y >> 4) ^ X) & 0x0F0F0F0Fu; X ^= T; Y ^= (T << 4);
  T = ((Y     ) ^ X) & 0x10101010u; X ^= T; Y ^= (T     );

  X = (nocheck_LHs[(X      ) & 0xF] << 3) | (nocheck_LHs[(X >>  8) & 0xF] << 2)
    | (nocheck_LHs[(X >> 16) & 0xF] << 1) | (nocheck_LHs[(X >> 24) & 0xF]     )
    | (nocheck_LHs[(X >>  5) & 0xF] << 7) | (nocheck_LHs[(X >> 13) & 0xF] << 6)
    | (nocheck_LHs[(X >> 21) & 0xF] << 5) | (nocheck_LHs[(X >> 29) & 0xF] << 4);

  Y = (nocheck_RHs[(Y >>  1) & 0xF] << 3) | (nocheck_RHs[(Y >>  9) & 0xF] << 2)
    | (nocheck_RHs[(Y >> 17) & 0xF] << 1) | (nocheck_RHs[(Y >> 25) & 0xF]     )
    | (nocheck_RHs[(Y >>  4) & 0xF] << 7) | (nocheck_RHs[(Y >> 12) & 0xF] << 6)
    | (nocheck_RHs[(Y >> 20) & 0xF] << 5) | (nocheck_RHs[(Y >> 28) & 0xF] << 4);

  X &= 0x0FFFFFFFu;
  Y &= 0x0FFFFFFFu;

  for (i = 0; i < 16; i++) {
    if (i < 2 || i == 8 || i == 15) {
      X = ((X <<  1) | (X >> 27)) & 0x0FFFFFFFu;
      Y = ((Y <<  1) | (Y >> 27)) & 0x0FFFFFFFu;
    } else {
      X = ((X <<  2) | (X >> 26)) & 0x0FFFFFFFu;
      Y = ((Y <<  2) | (Y >> 26)) & 0x0FFFFFFFu;
    }

    *SK++ = ((X <<  4) & 0x24000000u) | ((X << 28) & 0x10000000u)
          | ((X << 14) & 0x08000000u) | ((X << 18) & 0x02080000u)
          | ((X <<  6) & 0x01000000u) | ((X <<  9) & 0x00200000u)
          | ((X >>  1) & 0x00100000u) | ((X << 10) & 0x00040000u)
          | ((X <<  2) & 0x00020000u) | ((X >> 10) & 0x00010000u)
          | ((Y >> 13) & 0x00002000u) | ((Y >>  4) & 0x00001000u)
          | ((Y <<  6) & 0x00000800u) | ((Y >>  1) & 0x00000400u)
          | ((Y >> 14) & 0x00000200u) | ((Y      ) & 0x00000100u)
          | ((Y >>  5) & 0x00000020u) | ((Y >> 10) & 0x00000010u)
          | ((Y >>  3) & 0x00000008u) | ((Y >> 18) & 0x00000004u)
          | ((Y >> 26) & 0x00000002u) | ((Y >> 24) & 0x00000001u);

    *SK++ = ((X << 15) & 0x20000000u) | ((X << 17) & 0x10000000u)
          | ((X << 10) & 0x08000000u) | ((X << 22) & 0x04000000u)
          | ((X >>  2) & 0x02000000u) | ((X <<  1) & 0x01000000u)
          | ((X << 16) & 0x00200000u) | ((X << 11) & 0x00100000u)
          | ((X <<  3) & 0x00080000u) | ((X >>  6) & 0x00040000u)
          | ((X << 15) & 0x00020000u) | ((X >>  4) & 0x00010000u)
          | ((Y >>  2) & 0x00002000u) | ((Y <<  8) & 0x00001000u)
          | ((Y >> 14) & 0x00000808u) | ((Y >>  9) & 0x00000400u)
          | ((Y      ) & 0x00000200u) | ((Y <<  7) & 0x00000100u)
          | ((Y >>  7) & 0x00000020u) | ((Y >>  3) & 0x00000011u)
          | ((Y <<  2) & 0x00000004u) | ((Y >> 21) & 0x00000002u);
  }
}

void netntlmv1_hash_nocheck(const unsigned char *cand7, unsigned char *out8)
{
  uint32_t SK[32];
  uint32_t X, Y, T;
  const uint32_t *pSK;
  unsigned char des_key[8];

  setup_des_key((char *)cand7, des_key);
  nocheck_des_setkey(SK, des_key);

  /* IP-permuted state for fixed challenge 0x1122334455667788 (verified in netntlmv1.cl). */
  X = 0xf0aaf0aa;
  Y = 0x00cd00cd;

  pSK = SK;
#define NC_ROUND(XX, YY) do { \
    T = *pSK++ ^ (XX); \
    (YY) ^= nocheck_SB8[(T      ) & 0x3F] ^ nocheck_SB6[(T >>  8) & 0x3F] \
          ^ nocheck_SB4[(T >> 16) & 0x3F] ^ nocheck_SB2[(T >> 24) & 0x3F]; \
    T = *pSK++ ^ (((XX) << 28) | ((XX) >> 4)); \
    (YY) ^= nocheck_SB7[(T      ) & 0x3F] ^ nocheck_SB5[(T >>  8) & 0x3F] \
          ^ nocheck_SB3[(T >> 16) & 0x3F] ^ nocheck_SB1[(T >> 24) & 0x3F]; \
} while (0)

  NC_ROUND(Y, X); NC_ROUND(X, Y);
  NC_ROUND(Y, X); NC_ROUND(X, Y);
  NC_ROUND(Y, X); NC_ROUND(X, Y);
  NC_ROUND(Y, X); NC_ROUND(X, Y);
  NC_ROUND(Y, X); NC_ROUND(X, Y);
  NC_ROUND(Y, X); NC_ROUND(X, Y);
  NC_ROUND(Y, X); NC_ROUND(X, Y);
  NC_ROUND(Y, X); NC_ROUND(X, Y);
#undef NC_ROUND

  /* Final permutation: DES_FP(Y,X) — Y is the first macro argument, rotated first. */
  Y = ((Y << 31) | (Y >>  1));
  T = (Y ^ X) & 0xAAAAAAAAu; Y ^= T; X ^= T;
  X = ((X << 31) | (X >>  1));
  T = ((X >>  8) ^ Y) & 0x00FF00FFu; Y ^= T; X ^= (T <<  8);
  T = ((X >>  2) ^ Y) & 0x33333333u; Y ^= T; X ^= (T <<  2);
  T = ((Y >> 16) ^ X) & 0x0000FFFFu; X ^= T; Y ^= (T << 16);
  T = ((Y >>  4) ^ X) & 0x0F0F0F0Fu; X ^= T; Y ^= (T <<  4);

  out8[0] = (unsigned char)(Y >> 24); out8[1] = (unsigned char)(Y >> 16);
  out8[2] = (unsigned char)(Y >>  8); out8[3] = (unsigned char)(Y      );
  out8[4] = (unsigned char)(X >> 24); out8[5] = (unsigned char)(X >> 16);
  out8[6] = (unsigned char)(X >>  8); out8[7] = (unsigned char)(X      );
}
