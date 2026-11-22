/**
 * @file gnc_math.c
 * @brief Implementation of the single-precision GNC math library. See gnc_math.h.
 */
#include "gnc_math.h"

#include <math.h>

/* ======================= scalar ======================= */
float gnc_clampf(float x, float lo, float hi)
{
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

float gnc_wrap_pi(float a)
{
    /* bounded: atan2 form avoids loops */
    return atan2f(sinf(a), cosf(a));
}

float gnc_sqrtf_safe(float x)
{
    return (x > 0.0f) ? sqrtf(x) : 0.0f;
}

float gnc_signf(float x)
{
    return (x >= 0.0f) ? 1.0f : -1.0f;
}

/* ======================= vec3 ======================= */
void v3_set(float out[3], float x, float y, float z)
{
    out[0] = x; out[1] = y; out[2] = z;
}

void v3_copy(float out[3], const float a[3])
{
    out[0] = a[0]; out[1] = a[1]; out[2] = a[2];
}

void v3_zero(float out[3])
{
    out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f;
}

void v3_add(float out[3], const float a[3], const float b[3])
{
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
}

void v3_sub(float out[3], const float a[3], const float b[3])
{
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
}

void v3_scale(float out[3], const float a[3], float s)
{
    out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s;
}

void v3_axpy(float y[3], float alpha, const float x[3])
{
    y[0] += alpha * x[0]; y[1] += alpha * x[1]; y[2] += alpha * x[2];
}

float v3_dot(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void v3_cross(float out[3], const float a[3], const float b[3])
{
    const float x = a[1] * b[2] - a[2] * b[1];
    const float y = a[2] * b[0] - a[0] * b[2];
    const float z = a[0] * b[1] - a[1] * b[0];
    out[0] = x; out[1] = y; out[2] = z;
}

float v3_norm(const float a[3])
{
    return sqrtf(v3_dot(a, a));
}

float v3_normalize(float out[3], const float a[3])
{
    const float n = v3_norm(a);
    if (n > GNC_EPS) {
        v3_scale(out, a, 1.0f / n);
    } else {
        v3_zero(out);
    }
    return n;
}

void v3_skew(float S[9], const float a[3])
{
    S[0] = 0.0f;  S[1] = -a[2]; S[2] = a[1];
    S[3] = a[2];  S[4] = 0.0f;  S[5] = -a[0];
    S[6] = -a[1]; S[7] = a[0];  S[8] = 0.0f;
}

/* ======================= 3x3 ======================= */
void m3_mul_v(float out[3], const float M[9], const float v[3])
{
    const float x = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
    const float y = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
    const float z = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
    out[0] = x; out[1] = y; out[2] = z;
}

void m3t_mul_v(float out[3], const float M[9], const float v[3])
{
    const float x = M[0] * v[0] + M[3] * v[1] + M[6] * v[2];
    const float y = M[1] * v[0] + M[4] * v[1] + M[7] * v[2];
    const float z = M[2] * v[0] + M[5] * v[1] + M[8] * v[2];
    out[0] = x; out[1] = y; out[2] = z;
}

void m3_mul(float out[9], const float A[9], const float B[9])
{
    float t[9];
    int i, j;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            t[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
        }
    }
    for (i = 0; i < 9; ++i) { out[i] = t[i]; }
}

void m3_transpose(float out[9], const float A[9])
{
    float t[9];
    int i;
    t[0] = A[0]; t[1] = A[3]; t[2] = A[6];
    t[3] = A[1]; t[4] = A[4]; t[5] = A[7];
    t[6] = A[2]; t[7] = A[5]; t[8] = A[8];
    for (i = 0; i < 9; ++i) { out[i] = t[i]; }
}

void m3_identity(float out[9])
{
    mat_identity(out, 3);
}

/* ======================= quaternion ======================= */
void q_identity(float q[4])
{
    q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f;
}

void q_copy(float out[4], const float q[4])
{
    out[0] = q[0]; out[1] = q[1]; out[2] = q[2]; out[3] = q[3];
}

void q_mul(float out[4], const float a[4], const float b[4])
{
    const float w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    const float x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    const float y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    const float z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

void q_conj(float out[4], const float q[4])
{
    out[0] = q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = -q[3];
}

float q_normalize(float q[4])
{
    const float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > GNC_EPS) {
        const float s = 1.0f / n;
        q[0] *= s; q[1] *= s; q[2] *= s; q[3] *= s;
    } else {
        q_identity(q);
    }
    return n;
}

void q_to_dcm(float R[9], const float q[4])
{
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1.0f - 2.0f * (y * y + z * z);
    R[1] = 2.0f * (x * y - w * z);
    R[2] = 2.0f * (x * z + w * y);
    R[3] = 2.0f * (x * y + w * z);
    R[4] = 1.0f - 2.0f * (x * x + z * z);
    R[5] = 2.0f * (y * z - w * x);
    R[6] = 2.0f * (x * z - w * y);
    R[7] = 2.0f * (y * z + w * x);
    R[8] = 1.0f - 2.0f * (x * x + y * y);
}

void q_from_dcm(float q[4], const float R[9])
{
    /* Shepperd's method: pick the largest pivot for numerical robustness */
    const float tr = R[0] + R[4] + R[8];
    if (tr > R[0] && tr > R[4] && tr > R[8]) {
        const float s = 2.0f * sqrtf(1.0f + tr);
        q[0] = 0.25f * s;
        q[1] = (R[7] - R[5]) / s;
        q[2] = (R[2] - R[6]) / s;
        q[3] = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        const float s = 2.0f * sqrtf(1.0f + R[0] - R[4] - R[8]);
        q[0] = (R[7] - R[5]) / s;
        q[1] = 0.25f * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        const float s = 2.0f * sqrtf(1.0f + R[4] - R[0] - R[8]);
        q[0] = (R[2] - R[6]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25f * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        const float s = 2.0f * sqrtf(1.0f + R[8] - R[0] - R[4]);
        q[0] = (R[3] - R[1]) / s;
        q[1] = (R[2] + R[6]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25f * s;
    }
    if (q[0] < 0.0f) { q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; }
    (void)q_normalize(q);
}

void q_rotate(float out[3], const float q[4], const float v[3])
{
    /* v' = v + 2w (qv x v) + 2 qv x (qv x v) */
    float t[3], u[3];
    const float qv[3] = { q[1], q[2], q[3] };
    v3_cross(t, qv, v);
    v3_scale(t, t, 2.0f);
    v3_cross(u, qv, t);
    out[0] = v[0] + q[0] * t[0] + u[0];
    out[1] = v[1] + q[0] * t[1] + u[1];
    out[2] = v[2] + q[0] * t[2] + u[2];
}

void q_rotate_inv(float out[3], const float q[4], const float v[3])
{
    float qc[4];
    q_conj(qc, q);
    q_rotate(out, qc, v);
}

void q_from_rotvec(float q[4], const float th[3])
{
    const float a = v3_norm(th);
    if (a < 1.0e-6f) {
        q[0] = 1.0f; q[1] = 0.5f * th[0]; q[2] = 0.5f * th[1]; q[3] = 0.5f * th[2];
        (void)q_normalize(q);
    } else {
        const float s = sinf(0.5f * a) / a;
        q[0] = cosf(0.5f * a); q[1] = s * th[0]; q[2] = s * th[1]; q[3] = s * th[2];
    }
}

void q_to_rotvec(float th[3], const float q_in[4])
{
    float q[4];
    float vn, a;
    q_copy(q, q_in);
    if (q[0] < 0.0f) { q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3]; }
    vn = sqrtf(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (vn < 1.0e-7f) {
        th[0] = 2.0f * q[1]; th[1] = 2.0f * q[2]; th[2] = 2.0f * q[3];
        return;
    }
    a = 2.0f * atan2f(vn, q[0]);
    th[0] = a * q[1] / vn; th[1] = a * q[2] / vn; th[2] = a * q[3] / vn;
}

void q_error(float th[3], const float q_a[4], const float q_b[4])
{
    float qc[4], dq[4];
    q_conj(qc, q_a);
    q_mul(dq, qc, q_b);
    q_to_rotvec(th, dq);
}

void q_to_euler321(float ypr[3], const float q[4])
{
    float R[9];
    q_to_dcm(R, q);
    ypr[0] = atan2f(R[3], R[0]);
    ypr[1] = asinf(gnc_clampf(-R[6], -1.0f, 1.0f));
    ypr[2] = atan2f(R[7], R[8]);
}

void q_from_euler321(float q[4], float yaw, float pitch, float roll)
{
    const float cy = cosf(0.5f * yaw), sy = sinf(0.5f * yaw);
    const float cp = cosf(0.5f * pitch), sp = sinf(0.5f * pitch);
    const float cr = cosf(0.5f * roll), sr = sinf(0.5f * roll);
    q[0] = cy * cp * cr + sy * sp * sr;
    q[1] = cy * cp * sr - sy * sp * cr;
    q[2] = cy * sp * cr + sy * cp * sr;
    q[3] = sy * cp * cr - cy * sp * sr;
}

int att_from_two_axes(float q[4], int prim_axis, float prim_sign, const float prim_dir[3],
                      int sec_axis, const float sec_ref[3])
{
    float p[3], s[3], t[3], R[9];
    int k;
    float sgn;
    if (prim_axis < 0 || prim_axis > 2 || sec_axis < 0 || sec_axis > 2 || prim_axis == sec_axis) {
        return -1;
    }
    v3_scale(p, prim_dir, prim_sign);
    if (v3_normalize(p, p) < GNC_EPS) {
        return -1;
    }
    v3_copy(s, sec_ref);
    v3_axpy(s, -v3_dot(sec_ref, p), p);
    if (v3_normalize(s, s) < 1.0e-6f) {
        return -1;
    }
    k = 3 - prim_axis - sec_axis;
    sgn = (sec_axis == (prim_axis + 1) % 3) ? 1.0f : -1.0f;
    v3_cross(t, p, s);
    v3_scale(t, t, sgn);
    /* columns of R are the images of the body axes */
    R[0 + prim_axis] = p[0]; R[3 + prim_axis] = p[1]; R[6 + prim_axis] = p[2];
    R[0 + sec_axis]  = s[0]; R[3 + sec_axis]  = s[1]; R[6 + sec_axis]  = s[2];
    R[0 + k]         = t[0]; R[3 + k]         = t[1]; R[6 + k]         = t[2];
    q_from_dcm(q, R);
    return 0;
}

/* ======================= dense matrices ======================= */
void mat_zero(float *A, int n, int m)
{
    int i;
    for (i = 0; i < n * m; ++i) { A[i] = 0.0f; }
}

void mat_identity(float *A, int n)
{
    int i;
    mat_zero(A, n, n);
    for (i = 0; i < n; ++i) { A[i * n + i] = 1.0f; }
}

void mat_mul(float *C, const float *A, const float *B, int n, int k, int m)
{
    /* C must not alias A or B */
    int i, j, l;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            float acc = 0.0f;
            for (l = 0; l < k; ++l) { acc += A[i * k + l] * B[l * m + j]; }
            C[i * m + j] = acc;
        }
    }
}

void mat_mul_abt(float *C, const float *A, const float *B, int n, int k, int m)
{
    /* C (n x m) = A (n x k) * B^T, with B (m x k). C must not alias A or B */
    int i, j, l;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            float acc = 0.0f;
            for (l = 0; l < k; ++l) { acc += A[i * k + l] * B[j * k + l]; }
            C[i * m + j] = acc;
        }
    }
}

void mat_symmetrize(float *A, int n)
{
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            const float a = 0.5f * (A[i * n + j] + A[j * n + i]);
            A[i * n + j] = a;
            A[j * n + i] = a;
        }
    }
}

int chol_solve(float *x, const float *S, const float *b, int n)
{
    float L[36];
    float y[6];
    int i, j, k;
    if (n < 1 || n > 6) { return -1; }
    for (i = 0; i < n; ++i) {
        for (j = 0; j <= i; ++j) {
            float s = S[i * n + j];
            for (k = 0; k < j; ++k) { s -= L[i * 6 + k] * L[j * 6 + k]; }
            if (i == j) {
                if (s <= 0.0f) { return -1; }
                L[i * 6 + i] = sqrtf(s);
            } else {
                L[i * 6 + j] = s / L[j * 6 + j];
            }
        }
    }
    for (i = 0; i < n; ++i) {
        float s = b[i];
        for (k = 0; k < i; ++k) { s -= L[i * 6 + k] * y[k]; }
        y[i] = s / L[i * 6 + i];
    }
    for (i = n - 1; i >= 0; --i) {
        float s = y[i];
        for (k = i + 1; k < n; ++k) { s -= L[k * 6 + i] * x[k]; }
        x[i] = s / L[i * 6 + i];
    }
    return 0;
}
