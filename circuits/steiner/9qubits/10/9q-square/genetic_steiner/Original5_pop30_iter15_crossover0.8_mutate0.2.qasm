// Initial wiring: [6, 3, 1, 2, 5, 8, 4, 7, 0]
// Resulting wiring: [6, 3, 1, 2, 5, 8, 4, 7, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[0];
cx q[4], q[1];
cx q[5], q[0];
cx q[4], q[7];
cx q[4], q[5];
cx q[1], q[2];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[1];
