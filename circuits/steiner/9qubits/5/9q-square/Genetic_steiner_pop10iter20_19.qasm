// Initial wiring: [5, 4, 6, 1, 0, 3, 7, 8, 2]
// Resulting wiring: [5, 4, 6, 1, 0, 3, 7, 8, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[1], q[4];
cx q[4], q[5];
cx q[1], q[4];
cx q[5], q[6];
cx q[2], q[1];
