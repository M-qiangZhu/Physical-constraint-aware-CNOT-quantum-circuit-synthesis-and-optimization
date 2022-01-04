// Initial wiring: [5, 4, 2, 3, 6, 7, 8, 0, 1]
// Resulting wiring: [5, 4, 2, 3, 6, 7, 8, 0, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[2];
cx q[1], q[2];
cx q[0], q[5];
