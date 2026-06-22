#ifndef _STRING_CL
#define _STRING_CL

/* Performs standard strncpy() on __global source array to a local destination
 * array.  Unlike the traditional strncpy(), however, it returns the number of
 * bytes copied, not a pointer to the destination. */
inline unsigned int g_strncpy(char *dest, __global char *g_src, unsigned int n) {
  int i = 0;
  for (; i < n; i++) {
    dest[i] = g_src[i];
    if (dest[i] == 0)
      break;
  }
  return i;
}

/* The "byte" charset begins with a NUL, so g_strncpy() stops immediately and
 * reports length 0; a zero length is otherwise impossible, so treat it as the
 * byte charset and rebuild the full 0x00..0xFF sequence. */
inline unsigned int g_copy_charset(char *dest, __global char *g_src, unsigned int n) {
  unsigned int charset_len = g_strncpy(dest, g_src, n);
  if (charset_len == 0) {
    unsigned int i;
    for (i = 0; i < 256; i++)
      dest[i] = (char)i;
    charset_len = 256;
  }
  return charset_len;
}

inline unsigned int strlen(char *s) {
  unsigned int i = 0;
  for (; *s; i++, s++)
    ;
  return i;
}

inline void g_memcpy(unsigned char *dest, __global unsigned char *g_src, unsigned int n) {
  unsigned int i = 0;
  for (; i < n; i++)
    dest[i] = g_src[i];
}

inline void bzero(char *s, unsigned int n) {
  unsigned int i;
  for (i = 0; i < n; i++)
    s[i] = 0;
}

#endif
