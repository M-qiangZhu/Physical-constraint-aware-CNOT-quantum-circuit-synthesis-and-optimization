// Initial wiring: [8, 0, 3, 7, 1, 4, 5, 2, 6]
// Resulting wiring: [8, 0, 3, 7, 1, 4, 5, 2, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[2];
cx q[2], q[3];
cx q[1], q[2];
cx q[3], q[4];
cx q[7], q[4];
cx q[4], q[3];
cx q[3], q[2];
cx q[4], q[3];
cx q[3], q[4];
cx q[4], q[1];
cx q[7], q[4];
cx q[4], q[7];
