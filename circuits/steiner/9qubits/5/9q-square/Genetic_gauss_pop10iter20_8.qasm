// Initial wiring: [0 1 2 3 4 5 8 6 7]
// Resulting wiring: [1 0 3 2 5 4 8 6 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[4], q[5];
cx q[4], q[5];
cx q[4], q[5];
cx q[0], q[1];
cx q[0], q[1];
cx q[0], q[1];
cx q[7], q[4];
cx q[8], q[3];
cx q[4], q[1];
cx q[2], q[3];
cx q[2], q[3];
cx q[2], q[3];
cx q[3], q[4];
