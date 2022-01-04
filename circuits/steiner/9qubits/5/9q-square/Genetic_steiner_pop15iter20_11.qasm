// Initial wiring: [4, 0, 3, 1, 7, 5, 8, 6, 2]
// Resulting wiring: [4, 0, 3, 1, 7, 5, 8, 6, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[8];
cx q[7], q[6];
cx q[6], q[7];
cx q[7], q[4];
cx q[6], q[7];
cx q[4], q[7];
cx q[5], q[0];
