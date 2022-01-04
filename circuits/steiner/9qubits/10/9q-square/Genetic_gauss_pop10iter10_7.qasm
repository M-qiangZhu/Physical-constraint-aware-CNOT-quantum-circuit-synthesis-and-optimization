// Initial wiring: [0 4 2 3 1 6 5 8 7]
// Resulting wiring: [0 7 2 3 1 6 4 8 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[4];
cx q[3], q[4];
cx q[7], q[4];
cx q[7], q[4];
cx q[7], q[4];
cx q[2], q[3];
cx q[4], q[5];
cx q[4], q[5];
cx q[4], q[5];
cx q[0], q[5];
cx q[4], q[5];
cx q[3], q[4];
cx q[7], q[8];
