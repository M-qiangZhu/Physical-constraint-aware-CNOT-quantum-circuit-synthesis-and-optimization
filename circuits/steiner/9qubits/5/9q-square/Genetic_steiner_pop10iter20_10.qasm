// Initial wiring: [0, 8, 2, 6, 1, 7, 4, 3, 5]
// Resulting wiring: [0, 8, 2, 6, 1, 7, 4, 3, 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[2];
cx q[3], q[4];
cx q[4], q[7];
cx q[3], q[4];
cx q[7], q[6];
cx q[5], q[4];
