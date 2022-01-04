// Initial wiring: [0, 4, 1, 8, 3, 6, 5, 7, 2]
// Resulting wiring: [0, 4, 1, 8, 3, 6, 5, 7, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[5], q[6];
cx q[7], q[8];
cx q[3], q[8];
cx q[7], q[4];
cx q[4], q[3];
cx q[7], q[4];
