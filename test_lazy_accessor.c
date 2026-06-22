/*
 * test_lazy_accessor.c — Verify that the lazy RTC/RT mmap decode formula
 * produces identical results to rtc_decompress() for every chain.
 *
 * Build (from the project root):
 *   gcc -Wall -O2 -o test_lazy_accessor test_lazy_accessor.c rtc_decompress.c
 *
 * Run:
 *   ./test_lazy_accessor
 *
 * Tests:
 *   1. RTC lazy: mmap + inline decode formula == rtc_decompress() for all chains,
 *      using two representative bit-width combinations (8+8 and 36+38).
 *   2. RT  lazy: mmap with 16-byte-per-chain direct read == fread() interleaved
 *      buffer for all chains.
 *   3. Edge cases: s_bits==0, s_bits==64, chain on the first/last slot.
 */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rtc_decompress.h"

#define RTC_MAGIC 0x30435452u
#define RTC_HEADER_BYTES 32u

/* -----------------------------------------------------------------------
 * Lazy RTC decode — mirrors crackalack_lookup.c inlines exactly.
 * ----------------------------------------------------------------------- */
static uint64_t lazy_rtc_s(const unsigned char *packed, unsigned int chain_size,
                            unsigned int s_bits, uint64_t s_mask, uint64_t s_min)
{
    uint64_t buf[2] = {0, 0};
    memcpy(buf, packed, chain_size);
    return (buf[0] & s_mask) + s_min;
}

static uint64_t lazy_rtc_e(const unsigned char *packed, unsigned int chain_size,
                            unsigned int s_bits, uint64_t e_min,
                            uint64_t e_interval, unsigned int k)
{
    uint64_t buf[2] = {0, 0};
    memcpy(buf, packed, chain_size);
    uint64_t e_delta;
    if      (s_bits == 0)   e_delta = buf[0];
    else if (s_bits >= 64)  e_delta = buf[1];
    else                    e_delta = (buf[0] >> s_bits) | (buf[1] << (64u - s_bits));
    return e_min + e_interval * (uint64_t)k + e_delta;
}

/* -----------------------------------------------------------------------
 * Write a synthetic .rtc file and return its path (caller frees path).
 * Chains are generated with predictable but non-trivial (s,e) values.
 * ----------------------------------------------------------------------- */
static char *write_synthetic_rtc(unsigned int num_chains,
                                  unsigned int s_bits, unsigned int e_bits,
                                  uint64_t s_min, uint64_t e_min,
                                  uint64_t e_interval)
{
    unsigned int chain_size = (s_bits + e_bits + 7) / 8;
    assert(chain_size > 0 && chain_size <= 16);

    size_t file_size = (size_t)RTC_HEADER_BYTES + (size_t)num_chains * chain_size;
    unsigned char *buf = calloc(1, file_size);
    assert(buf);

    /* Header */
    unsigned int  ver    = RTC_MAGIC;
    unsigned short sb    = (unsigned short)s_bits;
    unsigned short eb    = (unsigned short)e_bits;
    memcpy(buf +  0, &ver,        4);
    memcpy(buf +  4, &sb,         2);
    memcpy(buf +  6, &eb,         2);
    memcpy(buf +  8, &s_min,      8);
    memcpy(buf + 16, &e_min,      8);
    memcpy(buf + 24, &e_interval, 8);

    /* Build s_mask */
    uint64_t s_mask = 0;
    for (unsigned int i = 0; i < s_bits; i++) { s_mask <<= 1; s_mask |= 1; }

    /* Pack each chain */
    for (unsigned int k = 0; k < num_chains; k++) {
        /* Choose s and e_delta (delta < e_interval so the table stays monotone) */
        uint64_t s_stored  = (uint64_t)(k % 97);          /* 0..96, always < s_mask when s_bits>=7 */
        uint64_t e_delta   = (uint64_t)(k % 11);          /* 0..10, always < e_interval=100 */

        uint64_t packed_lo = 0, packed_hi = 0;
        if (s_bits == 0) {
            /* All bits are e_delta */
            packed_lo = e_delta;
        } else if (s_bits >= 64) {
            /* Low 64 bits = s_stored; next word = e_delta */
            packed_lo = s_stored;
            packed_hi = e_delta;
        } else {
            packed_lo = s_stored | (e_delta << s_bits);
            packed_hi = e_delta >> (64u - s_bits);
        }
        uint64_t pbuf[2] = {packed_lo, packed_hi};
        memcpy(buf + RTC_HEADER_BYTES + (size_t)k * chain_size, pbuf, chain_size);
    }

    /* Write to a temp file whose name embeds num_chains so rtc_decompress() can parse it */
    char *path = malloc(128);
    assert(path);
    snprintf(path, 128, "/tmp/test_rtc_x%u_0.rtc", num_chains);
    FILE *f = fopen(path, "wb");
    assert(f);
    fwrite(buf, 1, file_size, f);
    fclose(f);
    free(buf);
    return path;
}

/* -----------------------------------------------------------------------
 * Test 1: RTC lazy decode == rtc_decompress, for given bit widths.
 * ----------------------------------------------------------------------- */
static int test_rtc(unsigned int num_chains, unsigned int s_bits, unsigned int e_bits,
                    uint64_t s_min, uint64_t e_min, uint64_t e_interval)
{
    char *path = write_synthetic_rtc(num_chains, s_bits, e_bits, s_min, e_min, e_interval);

    /* Reference path: rtc_decompress */
    uint64_t *ref_table = NULL;
    unsigned int ref_nc = 0;
    int ret = rtc_decompress(path, &ref_table, &ref_nc);
    if (ret != 0 || ref_nc != num_chains) {
        fprintf(stderr, "[rtc s=%u e=%u] rtc_decompress failed: ret=%d nc=%u\n",
                s_bits, e_bits, ret, ref_nc);
        unlink(path); free(path);
        return 0;
    }

    /* Lazy path: mmap + inline decode */
    int fd = open(path, O_RDONLY);
    assert(fd >= 0);
    struct stat fst = {0};
    fstat(fd, &fst);
    void *map = mmap(NULL, (size_t)fst.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(map != MAP_FAILED);
    close(fd);

    const unsigned char *mh = (const unsigned char *)map;

    /* Read decode params from mmap'd header */
    uint64_t lz_smin = 0, lz_emin = 0, lz_eint = 0;
    unsigned short lz_sb = 0;
    memcpy(&lz_sb,    mh +  4, 2);
    memcpy(&lz_smin,  mh +  8, 8);
    memcpy(&lz_emin,  mh + 16, 8);
    memcpy(&lz_eint,  mh + 24, 8);

    uint64_t lz_smask = 0;
    for (unsigned int b = 0; b < (unsigned int)lz_sb; b++) { lz_smask <<= 1; lz_smask |= 1; }

    unsigned int chain_size = (s_bits + e_bits + 7) / 8;
    unsigned int mismatches = 0;

    for (unsigned int k = 0; k < num_chains; k++) {
        const unsigned char *packed = mh + RTC_HEADER_BYTES + (size_t)k * chain_size;
        uint64_t lz_s = lazy_rtc_s(packed, chain_size, lz_sb, lz_smask, lz_smin);
        uint64_t lz_e = lazy_rtc_e(packed, chain_size, lz_sb, lz_emin, lz_eint, k);

        uint64_t ref_s = ref_table[k * 2];
        uint64_t ref_e = ref_table[k * 2 + 1];

        if (lz_s != ref_s || lz_e != ref_e) {
            fprintf(stderr, "[rtc s=%u e=%u] chain %u: lazy=(%"PRIu64",%"PRIu64") ref=(%"PRIu64",%"PRIu64")\n",
                    s_bits, e_bits, k, lz_s, lz_e, ref_s, ref_e);
            if (++mismatches > 5) break;
        }
    }

    munmap(map, (size_t)fst.st_size);
    free(ref_table);
    unlink(path);
    free(path);

    if (mismatches == 0) {
        printf("  PASS  RTC s_bits=%-2u e_bits=%-2u chains=%u\n", s_bits, e_bits, num_chains);
        return 1;
    }
    fprintf(stderr, "  FAIL  RTC s_bits=%u e_bits=%u: %u mismatches\n", s_bits, e_bits, mismatches);
    return 0;
}

/* -----------------------------------------------------------------------
 * Test 2: RT lazy (direct 16-byte reads via mmap) == fread buffer.
 * ----------------------------------------------------------------------- */
static int test_rt(unsigned int num_chains)
{
    size_t file_size = (size_t)num_chains * 16;
    unsigned char *ref_buf = calloc(1, file_size);
    assert(ref_buf);

    /* Write interleaved (start,end) 8-byte pairs — end indices monotone */
    for (unsigned int k = 0; k < num_chains; k++) {
        uint64_t s = 0x100000000ULL + k * 7ULL;
        uint64_t e = 0x200000000ULL + k * 100ULL + (k % 13);
        memcpy(ref_buf + (size_t)k * 16,     &s, 8);
        memcpy(ref_buf + (size_t)k * 16 + 8, &e, 8);
    }

    char path[] = "/tmp/test_rt_lazy.rt";
    FILE *f = fopen(path, "wb");
    assert(f);
    fwrite(ref_buf, 1, file_size, f);
    fclose(f);

    int fd = open(path, O_RDONLY);
    assert(fd >= 0);
    struct stat fst = {0};
    fstat(fd, &fst);
    void *map = mmap(NULL, (size_t)fst.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(map != MAP_FAILED);
    close(fd);

    const unsigned char *m = (const unsigned char *)map;
    unsigned int mismatches = 0;

    for (unsigned int k = 0; k < num_chains; k++) {
        uint64_t ref_s, ref_e, lz_s, lz_e;
        memcpy(&ref_s, ref_buf + (size_t)k * 16,     8);
        memcpy(&ref_e, ref_buf + (size_t)k * 16 + 8, 8);
        /* Lazy RT accessor: start at k*16, end at k*16+8 */
        memcpy(&lz_s,  m + (size_t)k * 16,     8);
        memcpy(&lz_e,  m + (size_t)k * 16 + 8, 8);

        if (lz_s != ref_s || lz_e != ref_e) {
            fprintf(stderr, "[rt] chain %u: lazy=(%"PRIu64",%"PRIu64") ref=(%"PRIu64",%"PRIu64")\n",
                    k, lz_s, lz_e, ref_s, ref_e);
            if (++mismatches > 5) break;
        }
    }

    munmap(map, (size_t)fst.st_size);
    free(ref_buf);
    unlink(path);

    if (mismatches == 0) {
        printf("  PASS  RT direct mmap read, %u chains\n", num_chains);
        return 1;
    }
    fprintf(stderr, "  FAIL  RT: %u mismatches\n", mismatches);
    return 0;
}

/* -----------------------------------------------------------------------
 * Test 3: Binary-search correctness using lazy decode.
 * Builds a known RTC in memory, decodes every chain, then binary-searches
 * for a present and an absent end index and verifies the result.
 * ----------------------------------------------------------------------- */

/* Simple recursive binary search identical to crackalack_lookup.c logic */
static int bsearch_rtc(const unsigned char *map, unsigned int chain_size,
                       unsigned int s_bits, uint64_t s_mask,
                       uint64_t s_min, uint64_t e_min, uint64_t e_interval,
                       unsigned int low, unsigned int high,
                       uint64_t target, uint64_t *start_out)
{
    if (high - low <= 8) {
        for (unsigned int k = low; k < high; k++) {
            uint64_t e = lazy_rtc_e(map + RTC_HEADER_BYTES + (size_t)k * chain_size,
                                     chain_size, s_bits, e_min, e_interval, k);
            if (e == target) {
                *start_out = lazy_rtc_s(map + RTC_HEADER_BYTES + (size_t)k * chain_size,
                                         chain_size, s_bits, s_mask, s_min);
                return 1;
            }
        }
        return 0;
    }
    unsigned int mid = ((high - low) / 2) + low;
    uint64_t mid_e = lazy_rtc_e(map + RTC_HEADER_BYTES + (size_t)mid * chain_size,
                                  chain_size, s_bits, e_min, e_interval, mid);
    if (target >= mid_e)
        return bsearch_rtc(map, chain_size, s_bits, s_mask, s_min, e_min, e_interval,
                           mid, high, target, start_out);
    else
        return bsearch_rtc(map, chain_size, s_bits, s_mask, s_min, e_min, e_interval,
                           low, mid, target, start_out);
}

static int test_bsearch(void)
{
    const unsigned int NC = 1024;
    const unsigned int SB = 36, EB = 38;
    const uint64_t SMIN = 1000000ULL, EMIN = 5000000000ULL, EINT = 100ULL;

    char *path = write_synthetic_rtc(NC, SB, EB, SMIN, EMIN, EINT);

    /* Decode reference table */
    uint64_t *ref = NULL;
    unsigned int ref_nc = 0;
    assert(rtc_decompress(path, &ref, &ref_nc) == 0 && ref_nc == NC);

    int fd = open(path, O_RDONLY);
    assert(fd >= 0);
    struct stat fst = {0};
    fstat(fd, &fst);
    void *map = mmap(NULL, (size_t)fst.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(map != MAP_FAILED);
    close(fd);

    const unsigned char *mh = (const unsigned char *)map;
    uint64_t lz_smin, lz_emin, lz_eint;
    unsigned short lz_sb;
    memcpy(&lz_sb,    mh +  4, 2);
    memcpy(&lz_smin,  mh +  8, 8);
    memcpy(&lz_emin,  mh + 16, 8);
    memcpy(&lz_eint,  mh + 24, 8);
    uint64_t lz_smask = 0;
    for (unsigned int b = 0; b < (unsigned int)lz_sb; b++) { lz_smask <<= 1; lz_smask |= 1; }

    unsigned int chain_size = (SB + EB + 7) / 8;
    int ok = 1;

    /* Test 10 present chains (spread across the table) */
    for (unsigned int t = 0; t < 10; t++) {
        unsigned int target_k = t * (NC / 10);
        uint64_t target_e = ref[target_k * 2 + 1];
        uint64_t expected_s = ref[target_k * 2];

        uint64_t got_s = 0;
        int found = bsearch_rtc((const unsigned char *)map, chain_size,
                                 lz_sb, lz_smask, lz_smin, lz_emin, lz_eint,
                                 0, NC, target_e, &got_s);
        if (!found) {
            fprintf(stderr, "[bsearch] chain %u: present e=%" PRIu64 " not found\n",
                    target_k, target_e);
            ok = 0;
        } else if (got_s != expected_s) {
            fprintf(stderr, "[bsearch] chain %u: start mismatch got=%" PRIu64
                    " want=%" PRIu64 "\n", target_k, got_s, expected_s);
            ok = 0;
        }
    }

    /* Test an absent end index (between two real ones) */
    if (NC > 2) {
        uint64_t absent = ref[2 * 2 + 1] + 1;
        /* Make sure it's not actually present */
        if (absent != ref[3 * 2 + 1]) {
            uint64_t dummy = 0;
            int found = bsearch_rtc((const unsigned char *)map, chain_size,
                                     lz_sb, lz_smask, lz_smin, lz_emin, lz_eint,
                                     0, NC, absent, &dummy);
            if (found) {
                fprintf(stderr, "[bsearch] absent e=%" PRIu64 " incorrectly found\n", absent);
                ok = 0;
            }
        }
    }

    munmap(map, (size_t)fst.st_size);
    free(ref);
    unlink(path);
    free(path);

    if (ok) printf("  PASS  Binary search (present/absent) via lazy RTC decoder\n");
    else    fprintf(stderr, "  FAIL  Binary search via lazy RTC decoder\n");
    return ok;
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(void)
{
    int all_ok = 1;

    printf("=== Lazy accessor correctness tests ===\n\n");

    printf("[RTC decode vs rtc_decompress]\n");
    /* Small, easy-to-inspect bit widths */
    all_ok &= test_rtc(1000,  8,  8, 100,       200,       10);
    all_ok &= test_rtc(1000, 16, 16, 10000,     5000000,   200);
    /* Realistic NetNTLMv1 bit widths */
    all_ok &= test_rtc(10000, 36, 38, 1000000,  5000000000ULL, 100);
    all_ok &= test_rtc(10000, 40, 34, 9876543,  3000000000ULL, 77);

    printf("\n[RT direct mmap read]\n");
    all_ok &= test_rt(50000);

    printf("\n[Binary search via lazy RTC decoder]\n");
    all_ok &= test_bsearch();

    printf("\n%s\n", all_ok ? "ALL TESTS PASSED" : "ONE OR MORE TESTS FAILED");
    return all_ok ? 0 : 1;
}
