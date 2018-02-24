#include <immintrin.h>
#include "avx_exp.h"

//ported for windows from avx_mathfun.h
//source http://software-lisc.fbk.eu/avx_mathfun/
__m256 exp256_ps(__m256 x) {
  __m256 tmp = _mm256_setzero_ps(), fx;
  __m256i imm0;
  float t = 1.;
  __m256 one = _mm256_broadcast_ss(&t);

  t = 88.3762626647949f;
  __m256 _ps256_exp_hi = _mm256_broadcast_ss(&t);
  x = _mm256_min_ps(x, _ps256_exp_hi);
  t = -88.3762626647949f;
  __m256 _ps256_exp_lo = _mm256_broadcast_ss(&t);
  x = _mm256_max_ps(x, _ps256_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  t = 1.44269504088896341;
  __m256 _ps256_cephes_LOG2EF = _mm256_broadcast_ss(&t);
  fx = _mm256_mul_ps(x, _ps256_cephes_LOG2EF);
  t = 0.5f;
  __m256 _ps256_0p5 = _mm256_broadcast_ss(&t);
  fx = _mm256_add_ps(fx, _ps256_0p5);

  /* how to perform a floorf with SSE: just below */
  //imm0 = _mm256_cvttps_epi32(fx);
  //tmp  = _mm256_cvtepi32_ps(imm0);

  tmp = _mm256_floor_ps(fx);

  /* if greater, substract 1 */
  //__m256 mask = _mm256_cmpgt_ps(tmp, fx);
  //int _CMP_GT_OS = 13;  //check this
  __m256 mask = _mm256_cmp_ps(tmp, fx, 13);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  t = 0.693359375;
  __m256 _ps256_cephes_exp_C1 = _mm256_broadcast_ss(&t);
  tmp = _mm256_mul_ps(fx, _ps256_cephes_exp_C1);
  t = -2.12194440e-4;
  __m256 _ps256_cephes_exp_C2 = _mm256_broadcast_ss(&t);
  __m256 z = _mm256_mul_ps(fx, _ps256_cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);

  /*
  _PS256_CONST(cephes_exp_C1, 0.693359375);
  _PS256_CONST(cephes_exp_C2, -2.12194440e-4);

  _PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
  _PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
  _PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
  _PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
  _PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
  _PS256_CONST(cephes_exp_p5, 5.0000001201E-1);
  */

  t = 1.9875691500E-4;
  __m256 _ps256_cephes_exp_p0 = _mm256_broadcast_ss(&t);
  t = 1.3981999507E-3;
  __m256 _ps256_cephes_exp_p1 = _mm256_broadcast_ss(&t);
  t = 8.3334519073E-3;
  __m256 _ps256_cephes_exp_p2 = _mm256_broadcast_ss(&t);
  t = 4.1665795894E-2;
  __m256 _ps256_cephes_exp_p3 = _mm256_broadcast_ss(&t);
  t = 1.6666665459E-1;
  __m256 _ps256_cephes_exp_p4 = _mm256_broadcast_ss(&t);
  t = 5.0000001201E-1;
  __m256 _ps256_cephes_exp_p5 = _mm256_broadcast_ss(&t);
  __m256 y = _ps256_cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _ps256_cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  int tt = 0x7f;
  __m256i _pi32_256_0x7f = _mm256_set1_epi32(tt);
  imm0 = _mm256_add_epi32(imm0, _pi32_256_0x7f);
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
