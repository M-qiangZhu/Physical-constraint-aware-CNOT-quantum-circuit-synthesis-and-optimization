// Initial wiring: [5, 4, 3, 6, 2, 0, 8, 1, 7]
// Resulting wiring: [5, 4, 3, 6, 2, 0, 8, 1, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[6];
cx q[7], q[8];
cx q[5], q[4];
cx q[3], q[2];
cx q[2], q[1];
