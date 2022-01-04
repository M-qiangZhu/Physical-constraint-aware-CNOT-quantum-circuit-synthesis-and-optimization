// Initial wiring: [2, 8, 5, 0, 1, 3, 4, 7, 6]
// Resulting wiring: [2, 8, 5, 0, 1, 3, 4, 7, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[2];
cx q[8], q[6];
cx q[6], q[5];
cx q[5], q[8];
cx q[4], q[6];
cx q[4], q[5];
cx q[5], q[4];
cx q[3], q[4];
cx q[4], q[3];
cx q[1], q[3];
cx q[0], q[3];
