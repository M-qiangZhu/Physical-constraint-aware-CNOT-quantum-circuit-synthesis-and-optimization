// Initial wiring: [5, 3, 2, 7, 6, 0, 4, 1, 8]
// Resulting wiring: [5, 3, 2, 7, 6, 0, 4, 1, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[1], q[4];
cx q[0], q[1];
cx q[5], q[6];
cx q[4], q[7];
cx q[3], q[4];
cx q[1], q[4];
cx q[4], q[7];
cx q[8], q[3];
cx q[7], q[8];
cx q[3], q[2];
