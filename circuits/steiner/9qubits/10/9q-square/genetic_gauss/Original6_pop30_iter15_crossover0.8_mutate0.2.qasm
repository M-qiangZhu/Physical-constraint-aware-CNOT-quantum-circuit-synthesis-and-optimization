// Initial wiring: [6, 2, 1, 4, 3, 5, 8, 7, 0]
// Resulting wiring: [6, 2, 1, 4, 3, 5, 8, 7, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[0];
cx q[6], q[1];
cx q[7], q[6];
cx q[7], q[4];
cx q[3], q[5];
cx q[0], q[4];
cx q[4], q[0];
cx q[4], q[8];
