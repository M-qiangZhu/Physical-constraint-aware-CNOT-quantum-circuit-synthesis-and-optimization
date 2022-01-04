// Initial wiring: [2, 1, 7, 6, 8, 0, 3, 4, 5]
// Resulting wiring: [2, 1, 7, 6, 8, 0, 3, 4, 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[8], q[7];
cx q[4], q[3];
cx q[3], q[2];
cx q[4], q[3];
cx q[2], q[3];
cx q[1], q[0];
