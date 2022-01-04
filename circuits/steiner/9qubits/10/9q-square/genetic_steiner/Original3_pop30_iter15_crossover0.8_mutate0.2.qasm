// Initial wiring: [1, 0, 5, 2, 8, 3, 6, 7, 4]
// Resulting wiring: [1, 0, 5, 2, 8, 3, 6, 7, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[1];
cx q[1], q[0];
cx q[2], q[1];
cx q[6], q[5];
cx q[7], q[6];
cx q[1], q[4];
cx q[4], q[7];
cx q[7], q[8];
cx q[4], q[1];
cx q[0], q[5];
