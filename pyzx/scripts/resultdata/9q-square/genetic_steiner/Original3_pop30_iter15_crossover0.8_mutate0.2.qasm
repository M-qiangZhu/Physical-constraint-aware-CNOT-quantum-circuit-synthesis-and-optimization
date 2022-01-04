// Initial wiring: [8, 0, 7, 5, 6, 3, 1, 2, 4]
// Resulting wiring: [8, 0, 7, 5, 6, 3, 1, 2, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[3];
cx q[5], q[0];
cx q[2], q[3];
