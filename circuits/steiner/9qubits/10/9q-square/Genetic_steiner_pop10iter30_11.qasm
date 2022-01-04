// Initial wiring: [8, 7, 6, 3, 2, 5, 0, 1, 4]
// Resulting wiring: [8, 7, 6, 3, 2, 5, 0, 1, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[1], q[4];
cx q[4], q[5];
cx q[3], q[4];
cx q[2], q[3];
cx q[4], q[5];
cx q[4], q[7];
cx q[1], q[4];
cx q[4], q[7];
cx q[3], q[8];
cx q[2], q[3];
cx q[3], q[8];
cx q[4], q[1];
cx q[2], q[1];
