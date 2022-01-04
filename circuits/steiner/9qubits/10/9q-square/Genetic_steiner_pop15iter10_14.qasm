// Initial wiring: [0, 7, 2, 4, 3, 1, 5, 6, 8]
// Resulting wiring: [0, 7, 2, 4, 3, 1, 5, 6, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[1], q[4];
cx q[0], q[5];
cx q[5], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[4], q[1];
cx q[5], q[4];
cx q[4], q[5];
cx q[5], q[0];
cx q[6], q[5];
cx q[5], q[6];
