// Initial wiring: [5, 6, 0, 8, 2, 1, 3, 7, 4]
// Resulting wiring: [5, 6, 0, 8, 2, 1, 3, 7, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[6];
cx q[7], q[4];
cx q[2], q[1];
cx q[5], q[0];
cx q[1], q[0];
