// Initial wiring: [0, 1, 6, 5, 3, 4, 7, 8, 2]
// Resulting wiring: [0, 1, 6, 5, 3, 4, 7, 8, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[0], q[5];
cx q[6], q[7];
cx q[7], q[8];
cx q[4], q[7];
