// Initial wiring: [1, 4, 2, 0, 5, 3, 7, 8, 6]
// Resulting wiring: [1, 4, 2, 0, 5, 3, 7, 8, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[6];
cx q[4], q[5];
cx q[3], q[4];
cx q[5], q[6];
cx q[6], q[5];
cx q[5], q[6];
cx q[5], q[4];
cx q[4], q[3];
cx q[3], q[4];
cx q[5], q[0];
cx q[4], q[5];
cx q[3], q[4];
