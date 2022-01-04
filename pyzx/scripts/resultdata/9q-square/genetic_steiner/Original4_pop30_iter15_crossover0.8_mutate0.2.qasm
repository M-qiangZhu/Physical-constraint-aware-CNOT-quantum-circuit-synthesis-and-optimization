// Initial wiring: [0, 2, 3, 6, 8, 4, 7, 5, 1]
// Resulting wiring: [0, 2, 3, 6, 8, 4, 7, 5, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[4];
cx q[7], q[4];
cx q[5], q[6];
