import bitshuffle


def pytest_report_header(config):
    sse2 = bitshuffle.using_SSE2()
    avx2 = bitshuffle.using_AVX2()
    avx512 = bitshuffle.using_AVX512()
    neon = bitshuffle.using_NEON()
    return f"Bitshuffle instruction set: SSE2 {sse2}; AVX2 {avx2}; AVX512 {avx512}; NEON {neon}"
