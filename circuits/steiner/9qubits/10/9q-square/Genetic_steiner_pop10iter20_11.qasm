// Initial wiring: [5, 3, 0, 1, 2, 4, 6, 7, 8]
// Resulting wiring: [5, 3, 0, 1, 2, 4, 6, 7, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[4];
cx q[4], q[7];
cx q[8], q[7];
cx q[6], q[5];
cx q[7], q[4];
cx q[4], q[3];
cx q[7], q[4];
cx q[6], q[7];
cx q[1], q[0];
