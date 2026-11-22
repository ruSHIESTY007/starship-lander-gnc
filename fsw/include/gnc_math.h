/**
 * @file gnc_math.h
 * @brief Single-precision vector / quaternion / small-matrix library for flight software.
 *
 * Conventions (used everywhere in the FSW):
 *  - Vectors are float[3]; matrices are row-major float arrays.
 *  - Quaternions are Hamilton, scalar-first: q = [w, x, y, z].
 *  - q_LB rotates body-frame vectors into the local-level (L) frame: v_L = R(q_LB) v_B.
 *  - Attitude kinematics: q_dot = 0.5 * q (x) [0, w_B].
 *
 * Design rules: no dynamic memory, no recursion, bounded loops, all outputs may alias
 * inputs unless stated otherwise, every division guarded.
 */
#ifndef GNC_MATH_H
#define GNC_MATH_H

#include <stdint.h>

#define GNC_PI        3.14159265358979f
#define GNC_DEG2RAD   (GNC_PI / 180.0f)
#define GNC_RAD2DEG   (180.0f / GNC_PI)
#define GNC_EPS       1.0e-9f

/* ---------- scalar ---------- */
float gnc_clampf(float x, float lo, float hi);
float gnc_wrap_pi(float a);
float gnc_sqrtf_safe(float x);
float gnc_signf(float x);

/* ---------- vec3 ---------- */
void  v3_set(float out[3], float x, float y, float z);
void  v3_copy(float out[3], const float a[3]);
void  v3_zero(float out[3]);
void  v3_add(float out[3], const float a[3], const float b[3]);
void  v3_sub(float out[3], const float a[3], const float b[3]);
void  v3_scale(float out[3], const float a[3], float s);
void  v3_axpy(float y[3], float alpha, const float x[3]);   /* y += alpha * x */
float v3_dot(const float a[3], const float b[3]);
void  v3_cross(float out[3], const float a[3], const float b[3]);
float v3_norm(const float a[3]);
float v3_normalize(float out[3], const float a[3]);          /* returns |a|; out = 0 if |a|~0 */
void  v3_skew(float S[9], const float a[3]);                  /* S*b = a x b */

/* ---------- 3x3 ---------- */
void m3_mul_v(float out[3], const float M[9], const float v[3]);
void m3t_mul_v(float out[3], const float M[9], const float v[3]);
void m3_mul(float out[9], const float A[9], const float B[9]);
void m3_transpose(float out[9], const float A[9]);
void m3_identity(float out[9]);

/* ---------- quaternion ---------- */
void  q_identity(float q[4]);
void  q_copy(float out[4], const float q[4]);
void  q_mul(float out[4], const float a[4], const float b[4]);
void  q_conj(float out[4], const float q[4]);
float q_normalize(float q[4]);
void  q_to_dcm(float R[9], const float q[4]);
void  q_from_dcm(float q[4], const float R[9]);
void  q_rotate(float out[3], const float q[4], const float v[3]);      /* R(q) v   */
void  q_rotate_inv(float out[3], const float q[4], const float v[3]);  /* R(q)^T v */
void  q_from_rotvec(float q[4], const float th[3]);                   /* exp map  */
void  q_to_rotvec(float th[3], const float q[4]);                     /* log map  */
/** Small-angle rotation vector taking frame a to frame b, expressed in a: q_b = q_a (x) exp(th). */
void  q_error(float th[3], const float q_a[4], const float q_b[4]);
void  q_to_euler321(float ypr[3], const float q[4]);
void  q_from_euler321(float q[4], float yaw, float pitch, float roll);

/**
 * Build a target attitude q_LB such that body axis `prim_axis` (0=x,1=y,2=z) points along
 * L-frame unit vector `prim_dir` (scaled by prim_sign = +/-1), and body axis `sec_axis`
 * points as close as possible to L-frame vector `sec_ref`. Returns 0 on success, -1 if
 * degenerate (sec_ref parallel to prim_dir), in which case q is left untouched.
 */
int att_from_two_axes(float q[4], int prim_axis, float prim_sign, const float prim_dir[3],
                      int sec_axis, const float sec_ref[3]);

/* ---------- general dense matrices (row-major) ---------- */
void mat_zero(float *A, int n, int m);
void mat_identity(float *A, int n);
void mat_mul(float *C, const float *A, const float *B, int n, int k, int m);     /* C=A*B    */
void mat_mul_abt(float *C, const float *A, const float *B, int n, int k, int m); /* C=A*B^T  */
void mat_symmetrize(float *A, int n);
/** Solve S x = b for symmetric positive-definite S (n <= 6) by Cholesky. Returns -1 if not PD. */
int  chol_solve(float *x, const float *S, const float *b, int n);

#endif /* GNC_MATH_H */
