// Initial wiring: [1, 5, 4, 0, 7, 3, 8, 6, 2]
// Resulting wiring: [1, 5, 4, 0, 7, 3, 8, 6, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[4], q[3];
cx q[3], q[2];
cx q[5], q[0];
