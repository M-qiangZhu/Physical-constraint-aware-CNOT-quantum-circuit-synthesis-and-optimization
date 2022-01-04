// Initial wiring: [8, 7, 0, 5, 6, 3, 4, 2, 1]
// Resulting wiring: [8, 7, 0, 5, 6, 3, 4, 2, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[1];
cx q[4], q[3];
cx q[2], q[3];
