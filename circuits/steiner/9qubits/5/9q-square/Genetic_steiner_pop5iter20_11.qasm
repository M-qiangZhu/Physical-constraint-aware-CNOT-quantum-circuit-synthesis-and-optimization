// Initial wiring: [8, 3, 5, 4, 1, 7, 6, 2, 0]
// Resulting wiring: [8, 3, 5, 4, 1, 7, 6, 2, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[0], q[5];
cx q[5], q[6];
cx q[4], q[7];
cx q[6], q[5];
cx q[5], q[0];
cx q[6], q[5];
