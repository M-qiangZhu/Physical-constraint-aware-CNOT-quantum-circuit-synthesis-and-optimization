// Initial wiring: [5, 4, 2, 7, 3, 6, 8, 1, 0]
// Resulting wiring: [5, 4, 2, 7, 3, 6, 8, 1, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[4];
cx q[4], q[5];
cx q[1], q[4];
cx q[5], q[6];
cx q[4], q[5];
cx q[5], q[6];
cx q[4], q[7];
cx q[7], q[4];
cx q[4], q[7];
cx q[4], q[3];
cx q[7], q[4];
cx q[4], q[7];
cx q[5], q[0];
