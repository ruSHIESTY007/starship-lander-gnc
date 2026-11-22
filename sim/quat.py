"""Double-precision quaternion utilities (Hamilton, scalar-first, q_LB: body -> L)."""
from __future__ import annotations

import numpy as np


def mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize(q):
    return q / np.linalg.norm(q)


def to_dcm(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def from_dcm(R):
    tr = np.trace(R)
    if tr > max(R[0, 0], R[1, 1], R[2, 2]):
        s = 2 * np.sqrt(1 + tr)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return normalize(q if q[0] >= 0 else -q)


def from_euler321(yaw, pitch, roll):
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr])


def from_rotvec(th):
    a = np.linalg.norm(th)
    if a < 1e-9:
        return normalize(np.array([1.0, *(0.5 * np.asarray(th))]))
    return np.array([np.cos(a / 2), *(np.sin(a / 2) * np.asarray(th) / a)])


def to_rotvec(q):
    q = q if q[0] >= 0 else -q
    vn = np.linalg.norm(q[1:])
    if vn < 1e-12:
        return 2 * q[1:]
    return 2 * np.arctan2(vn, q[0]) * q[1:] / vn


def error(qa, qb):
    """Rotation vector of conj(qa) (x) qb (frame a -> frame b, expressed in a)."""
    return to_rotvec(mul(conj(qa), qb))


def tilt_deg(q, body_axis=0):
    """Angle between a body axis and L-frame up."""
    R = to_dcm(q)
    return np.degrees(np.arccos(np.clip(R[2, body_axis], -1, 1)))
