// Initial wiring: [8, 5, 1, 4, 2, 6, 7, 3, 0]
// Resulting wiring: [8, 5, 1, 4, 2, 6, 7, 3, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[5], q[6];
cx q[7], q[8];
cx q[5], q[0];
cx q[4], q[5];
