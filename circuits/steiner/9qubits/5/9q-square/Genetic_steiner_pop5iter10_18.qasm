// Initial wiring: [6, 7, 1, 8, 2, 3, 0, 5, 4]
// Resulting wiring: [6, 7, 1, 8, 2, 3, 0, 5, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[3], q[2];
cx q[4], q[3];
cx q[5], q[4];
cx q[5], q[0];
