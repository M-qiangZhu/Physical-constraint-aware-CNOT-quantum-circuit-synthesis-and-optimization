// Initial wiring: [0, 6, 7, 2, 5, 1, 3, 4, 8]
// Resulting wiring: [0, 6, 7, 2, 5, 1, 3, 4, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[0];
cx q[7], q[6];
cx q[7], q[4];
