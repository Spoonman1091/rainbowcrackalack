#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include "cpu_rt_functions.h"
#include "test_shared.h"
#include "ntlmv1.h"

static int is_hex_char(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static int is_hex_str(const char *s, unsigned int len) {
  unsigned int i;
  for (i = 0; i < len; i++)
    if (!is_hex_char(s[i])) return 0;
  return 1;
}

static void to_lowercase_inplace(char *s) {
  for (; *s; s++)
    *s = (char)tolower((unsigned char)*s);
}

int ntlmv1_parse_capture(const char *capture, ntlmv1_capture *out) {
  char buf[512];
  char *fields[6];
  int nfields = 0;
  char *p;
  unsigned char lm_bytes[24];
  int all_zero_trailing, any_nonzero_leading;
  int i;
  char *user, *lm_field, *nt_field, *chal_field;

  if (!capture || !out)
    return NTLMV1_ERR_FORMAT;

  if (strlen(capture) >= sizeof(buf))
    return NTLMV1_ERR_FORMAT;

  strncpy(buf, capture, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  /* Split on ':' into exactly 6 fields: user::domain:LM:NT:challenge */
  fields[0] = buf;
  nfields = 1;
  for (p = buf; *p; p++) {
    if (*p == ':') {
      *p = '\0';
      if (nfields >= 6)
        return NTLMV1_ERR_FORMAT;
      fields[nfields++] = p + 1;
    }
  }
  if (nfields != 6)
    return NTLMV1_ERR_FORMAT;

  user      = fields[0];
  lm_field  = fields[3];
  nt_field  = fields[4];
  chal_field = fields[5];

  if (strlen(lm_field) != 48 || strlen(nt_field) != 48 || strlen(chal_field) != 16)
    return NTLMV1_ERR_FORMAT;

  if (!is_hex_str(lm_field, 48) || !is_hex_str(nt_field, 48) || !is_hex_str(chal_field, 16))
    return NTLMV1_ERR_FORMAT;

  to_lowercase_inplace(chal_field);
  to_lowercase_inplace(nt_field);
  to_lowercase_inplace(lm_field);

  if (strcmp(chal_field, "1122334455667788") != 0)
    return NTLMV1_ERR_CHALLENGE;

  /* ESS check: decode 24-byte LM response; detect SSP signature
   * (first 8 bytes non-zero, trailing 16 bytes all zero). */
  hex_to_bytes(lm_field, 24, lm_bytes);
  any_nonzero_leading = 0;
  for (i = 0; i < 8; i++)
    if (lm_bytes[i] != 0) { any_nonzero_leading = 1; break; }
  all_zero_trailing = 1;
  for (i = 8; i < 24; i++)
    if (lm_bytes[i] != 0) { all_zero_trailing = 0; break; }
  if (any_nonzero_leading && all_zero_trailing)
    return NTLMV1_ERR_ESS;

  memset(out, 0, sizeof(*out));
  strncpy(out->block1_hex, nt_field,      16);
  out->block1_hex[16] = '\0';
  strncpy(out->block2_hex, nt_field + 16, 16);
  out->block2_hex[16] = '\0';
  hex_to_bytes(nt_field + 32, 8, out->block3);
  strncpy(out->username, user, sizeof(out->username) - 1);
  out->username[sizeof(out->username) - 1] = '\0';

  return NTLMV1_OK;
}

int ntlmv1_recover_last2(const unsigned char block3[8], unsigned char last2[2],
                         unsigned int *found_count) {
  unsigned int v;
  unsigned char cand[7];
  unsigned char h[8];

  *found_count = 0;
  memset(cand, 0, sizeof(cand));

  for (v = 0; v < 65536; v++) {
    cand[0] = (unsigned char)((v >> 8) & 0xff);
    cand[1] = (unsigned char)(v & 0xff);
    netntlmv1_hash_nocheck(cand, h);
    if (memcmp(h, block3, 8) == 0) {
      if (*found_count == 0) {
        last2[0] = cand[0];
        last2[1] = cand[1];
      }
      (*found_count)++;
    }
  }
  return (*found_count > 0) ? 1 : 0;
}

void ntlmv1_assemble(const unsigned char k1[7], const unsigned char k2[7],
                     const unsigned char last2[2], unsigned char ntlm[16]) {
  memcpy(ntlm,      k1,    7);
  memcpy(ntlm + 7,  k2,    7);
  memcpy(ntlm + 14, last2, 2);
}
