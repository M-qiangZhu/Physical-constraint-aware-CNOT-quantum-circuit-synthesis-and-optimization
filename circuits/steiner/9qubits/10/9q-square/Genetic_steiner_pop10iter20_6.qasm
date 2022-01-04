// Initial wiring: [1, 7, 0, 6, 3, 4, 5, 8, 2]
// Resulting wiring: [1, 7, 0, 6, 3, 4, 5, 8, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[6];
cx q[7], q[8];
cx q[4], q[7];
cx q[1], q[4];
cx q[4], q[7];
cx q[5], q[4];
cx q[4], q[3];
cx q[5], q[4];
cx q[3], q[4];
cx q[3], q[2];
cx q[4], q[3];
cx q[3], q[4];
