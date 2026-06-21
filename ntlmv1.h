#ifndef NTLMV1_H
#define NTLMV1_H

typedef struct {
  char block1_hex[17];        /* NT bytes 0-7   (16 hex + NUL) */
  char block2_hex[17];        /* NT bytes 8-15  (16 hex + NUL) */
  unsigned char block3[8];    /* NT bytes 16-23 (raw)          */
  char username[256];         /* capture field 0, for pwdump output */
} ntlmv1_capture;

#define NTLMV1_OK             0
#define NTLMV1_ERR_FORMAT    -1
#define NTLMV1_ERR_ESS       -2
#define NTLMV1_ERR_CHALLENGE -3

int  ntlmv1_parse_capture(const char *capture, ntlmv1_capture *out);
int  ntlmv1_recover_last2(const unsigned char block3[8], unsigned char last2[2],
                          unsigned int *found_count);
void ntlmv1_assemble(const unsigned char k1[7], const unsigned char k2[7],
                     const unsigned char last2[2], unsigned char ntlm[16]);

#endif
