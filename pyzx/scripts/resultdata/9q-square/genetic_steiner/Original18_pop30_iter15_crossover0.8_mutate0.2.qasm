// Initial wiring: [6, 7, 8, 5, 4, 3, 2, 0, 1]
// Resulting wiring: [6, 7, 8, 5, 4, 3, 2, 0, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[4];
cx q[7], q[6];
cx q[4], q[1];
