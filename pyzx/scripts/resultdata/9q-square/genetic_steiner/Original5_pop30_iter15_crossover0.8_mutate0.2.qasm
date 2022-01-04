// Initial wiring: [3, 1, 8, 2, 5, 6, 0, 7, 4]
// Resulting wiring: [3, 1, 8, 2, 5, 6, 0, 7, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[0];
cx q[5], q[6];
cx q[1], q[4];
