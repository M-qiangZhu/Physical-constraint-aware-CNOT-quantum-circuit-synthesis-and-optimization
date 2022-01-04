// Initial wiring: [0, 4, 8, 1, 6, 5, 2, 3, 7]
// Resulting wiring: [0, 4, 8, 1, 6, 5, 2, 3, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[6];
cx q[4], q[7];
cx q[1], q[2];
