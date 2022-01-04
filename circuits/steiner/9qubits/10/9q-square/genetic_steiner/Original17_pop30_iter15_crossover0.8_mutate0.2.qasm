// Initial wiring: [6, 0, 4, 2, 3, 5, 7, 8, 1]
// Resulting wiring: [6, 0, 4, 2, 3, 5, 7, 8, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[2];
cx q[7], q[6];
cx q[6], q[5];
cx q[7], q[6];
cx q[8], q[3];
cx q[4], q[5];
cx q[1], q[2];
cx q[0], q[5];
