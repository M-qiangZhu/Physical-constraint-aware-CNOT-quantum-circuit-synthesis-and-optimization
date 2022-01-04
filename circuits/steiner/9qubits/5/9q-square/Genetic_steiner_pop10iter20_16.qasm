// Initial wiring: [8, 4, 1, 2, 3, 0, 7, 5, 6]
// Resulting wiring: [8, 4, 1, 2, 3, 0, 7, 5, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[4], q[5];
cx q[3], q[4];
cx q[5], q[6];
cx q[4], q[5];
