// Initial wiring: [0 1 2 8 4 5 6 7 3]
// Resulting wiring: [5 1 2 8 3 0 6 7 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[5];
cx q[1], q[4];
cx q[0], q[5];
cx q[0], q[5];
cx q[0], q[5];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[4], q[5];
cx q[7], q[6];
cx q[2], q[3];
cx q[5], q[6];
cx q[8], q[3];
