// Initial wiring: [8, 6, 7, 0, 2, 1, 5, 3, 4]
// Resulting wiring: [8, 6, 7, 0, 2, 1, 5, 3, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[1], q[4];
cx q[0], q[1];
cx q[0], q[5];
cx q[4], q[1];
cx q[7], q[4];
cx q[1], q[0];
cx q[4], q[1];
cx q[7], q[4];
cx q[1], q[4];
