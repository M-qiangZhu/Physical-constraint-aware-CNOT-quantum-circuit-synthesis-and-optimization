// Initial wiring: [1, 8, 5, 7, 4, 2, 3, 6, 0]
// Resulting wiring: [1, 8, 5, 7, 4, 2, 3, 6, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[6], q[5];
cx q[7], q[6];
