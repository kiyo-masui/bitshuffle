#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitshuffle_core.h"

int64_t bshuf_trans_bit_byte_scal(const void *in, void *out, const size_t size,
                                  const size_t elem_size);
int64_t bshuf_trans_bit_byte_NEON(const void *in, void *out, const size_t size,
                                  const size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_scal(const void *in, void *out,
                                         const size_t size,
                                         const size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_NEON(const void *in, void *out,
                                         const size_t size,
                                         const size_t elem_size);
int64_t bshuf_trans_bit_elem_scal(const void *in, void *out, const size_t size,
                                  const size_t elem_size);
int64_t bshuf_trans_bit_elem_NEON(const void *in, void *out, const size_t size,
                                  const size_t elem_size);
int64_t bshuf_untrans_bit_elem_scal(const void *in, void *out,
                                    const size_t size, const size_t elem_size);
int64_t bshuf_untrans_bit_elem_NEON(const void *in, void *out,
                                    const size_t size, const size_t elem_size);

typedef int64_t (*kernel_fn)(const void *, void *, const size_t,
                             const size_t);

typedef struct {
    const char *name;
    kernel_fn scal;
    kernel_fn neon;
} kernel_pair;

static const kernel_pair KERNELS[] = {
    {"trans_bit_byte", bshuf_trans_bit_byte_scal, bshuf_trans_bit_byte_NEON},
    {"shuffle_bit_eightelem", bshuf_shuffle_bit_eightelem_scal,
     bshuf_shuffle_bit_eightelem_NEON},
#ifndef BSHUF_UBSAN
    {"trans_bit_elem", bshuf_trans_bit_elem_scal, bshuf_trans_bit_elem_NEON},
    {"untrans_bit_elem", bshuf_untrans_bit_elem_scal,
     bshuf_untrans_bit_elem_NEON},
#endif
};

enum {
    GUARD_BYTES = 16,
    MAX_OFF = 7,
    CANARY = 0xAA,
    IN_FILL = 0x55,
    MAX_REPORT = 10,
};

typedef struct {
    long cases;
    long failures;
} run_stats;

static int check_guard(const uint8_t *buf, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        if (buf[i] != CANARY) {
            return 0;
        }
    }
    return 1;
}

static void record_failure(run_stats *st, const char *kind, const char *fn_name,
                           size_t es, size_t size, size_t in_off,
                           size_t out_off, const char *detail) {
    st->failures++;
    if (st->failures <= MAX_REPORT) {
        printf("FAIL [%s] fn=%s elem_size=%zu size=%zu in_off=%zu out_off=%zu (%s)\n",
               kind, fn_name, es, size, in_off, out_off, detail);
    }
}

static void run_case_pair(const kernel_pair *kp, const uint8_t *in_ptr,
                          size_t es, size_t size, size_t nb, size_t in_off,
                          size_t out_off, uint8_t *out_scal, uint8_t *out_neon,
                          size_t out_cap, const char *kind, run_stats *st) {
    size_t pre_end = GUARD_BYTES + out_off;
    size_t post_start = pre_end + nb;

    memset(out_scal, CANARY, out_cap);
    memset(out_neon, CANARY, out_cap);

    int64_t rc_scal = kp->scal(in_ptr, out_scal + pre_end, size, es);
    int64_t rc_neon = kp->neon(in_ptr, out_neon + pre_end, size, es);
    st->cases++;

    if (rc_scal != (int64_t)nb || rc_neon != (int64_t)nb) {
        record_failure(st, kind, kp->name, es, size, in_off, out_off,
                       "return code mismatch");
        return;
    }
    if (memcmp(out_scal + pre_end, out_neon + pre_end, nb) != 0) {
        record_failure(st, kind, kp->name, es, size, in_off, out_off,
                       "output mismatch");
    }
    if (!check_guard(out_neon, 0, pre_end) ||
        !check_guard(out_neon, post_start, out_cap) ||
        !check_guard(out_scal, 0, pre_end) ||
        !check_guard(out_scal, post_start, out_cap)) {
        record_failure(st, kind, kp->name, es, size, in_off, out_off,
                       "guard byte clobbered");
    }
}

static void run_case_all_kernels(const uint8_t *in_ptr, size_t es, size_t size,
                                 size_t nb, size_t in_off, size_t out_off,
                                 uint8_t *out_scal, uint8_t *out_neon,
                                 size_t out_cap, const char *kind,
                                 run_stats *st) {
    size_t nkernels = sizeof(KERNELS) / sizeof(KERNELS[0]);

    for (size_t k = 0; k < nkernels; k++) {
        run_case_pair(&KERNELS[k], in_ptr, es, size, nb, in_off, out_off,
                      out_scal, out_neon, out_cap, kind, st);
    }
}

static void test_size(size_t es, size_t size, run_stats *st) {
    size_t nb = es * size;
    size_t out_cap = GUARD_BYTES + MAX_OFF + nb + GUARD_BYTES;

    uint8_t *raw = malloc(nb);
    uint8_t *out_scal = malloc(out_cap);
    uint8_t *out_neon = malloc(out_cap);
    if (!raw || !out_scal || !out_neon) {
        fprintf(stderr, "allocation failed for es=%zu size=%zu\n", es, size);
        exit(2);
    }

    for (size_t i = 0; i < nb; i++) {
        raw[i] = (uint8_t)(rand() & 0xFF);
    }

    for (size_t in_off = 0; in_off <= MAX_OFF; in_off++) {
        uint8_t *in_buf = malloc(in_off + nb);
        if (!in_buf) {
            fprintf(stderr, "allocation failed for in_buf es=%zu size=%zu\n",
                    es, size);
            exit(2);
        }
        memset(in_buf, IN_FILL, in_off);
        memcpy(in_buf + in_off, raw, nb);
        for (size_t out_off = 0; out_off <= MAX_OFF; out_off++) {
            run_case_all_kernels(in_buf + in_off, es, size, nb, in_off, out_off,
                                 out_scal, out_neon, out_cap, "random", st);
        }
        free(in_buf);
    }

    free(raw);
    free(out_scal);
    free(out_neon);
}

static void test_basis_vectors(run_stats *st) {
    static const size_t BASIS_SIZES[] = {8, 16, 24, 64};
    size_t nsizes = sizeof(BASIS_SIZES) / sizeof(BASIS_SIZES[0]);

    for (size_t es = 1; es <= 17; es++) {
        for (size_t si = 0; si < nsizes; si++) {
            size_t size = BASIS_SIZES[si];
            if (size == 64 && es > 2) {
                continue;
            }
            size_t nb = es * size;
            size_t out_cap = GUARD_BYTES + nb + GUARD_BYTES;
            uint8_t *in_buf = malloc(nb);
            uint8_t *out_scal = malloc(out_cap);
            uint8_t *out_neon = malloc(out_cap);
            if (!in_buf || !out_scal || !out_neon) {
                fprintf(stderr, "allocation failed in basis test\n");
                exit(2);
            }

            for (size_t bit = 0; bit < nb * 8; bit++) {
                memset(in_buf, 0, nb);
                in_buf[bit / 8] = (uint8_t)(1u << (bit % 8));
                run_case_all_kernels(in_buf, es, size, nb, 0, 0, out_scal,
                                     out_neon, out_cap, "basis", st);
            }

            free(in_buf);
            free(out_scal);
            free(out_neon);
        }
    }
}

int main(void) {
    if (!bshuf_using_NEON()) {
        fprintf(stderr, "NEON not enabled at compile time\n");
        return 1;
    }

    run_stats st = {0, 0};
    srand(1);

    for (size_t es = 1; es <= 33; es++) {
        for (size_t size = 8; size <= 520; size += 8) {
            test_size(es, size, &st);
        }
        size_t base = (8192 / es) & ~(size_t)7;
        for (size_t add = 0; add <= 24; add += 8) {
            size_t extra = base + add;
            if (extra > 520) {
                test_size(es, extra, &st);
            }
        }
    }

    test_basis_vectors(&st);

    printf("test_neon_kernels: cases=%ld failures=%ld\n", st.cases,
           st.failures);
    return st.failures == 0 ? 0 : 1;
}
