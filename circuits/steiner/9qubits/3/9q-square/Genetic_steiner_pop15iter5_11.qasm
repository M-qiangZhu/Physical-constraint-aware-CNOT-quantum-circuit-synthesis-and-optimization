// Initial wiring: [8, 5, 7, 4, 0, 3, 6, 2, 1]
// Resulting wiring: [8, 5, 7, 4, 0, 3, 6, 2, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[0];
cx q[6], q[5];
cx q[4], q[5];
