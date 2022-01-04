// Initial wiring: [3, 4, 6, 7, 5, 0, 2, 8, 1]
// Resulting wiring: [3, 4, 6, 7, 5, 0, 2, 8, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[1], q[2];
cx q[6], q[7];
cx q[5], q[6];
cx q[6], q[7];
cx q[8], q[3];
cx q[2], q[1];
cx q[1], q[2];
cx q[1], q[0];
cx q[2], q[1];
cx q[1], q[2];
