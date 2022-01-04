// Initial wiring: [7, 5, 6, 0, 4, 8, 2, 1, 3]
// Resulting wiring: [7, 5, 6, 0, 4, 8, 2, 1, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[3], q[4];
cx q[0], q[5];
cx q[7], q[4];
cx q[4], q[1];
cx q[2], q[1];
cx q[1], q[0];
cx q[4], q[1];
cx q[2], q[1];
cx q[1], q[2];
