// Initial wiring: [5, 1, 6, 3, 7, 0, 2, 8, 4]
// Resulting wiring: [5, 1, 6, 3, 7, 0, 2, 8, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[2];
cx q[2], q[3];
cx q[1], q[2];
cx q[3], q[8];
cx q[7], q[6];
cx q[4], q[1];
