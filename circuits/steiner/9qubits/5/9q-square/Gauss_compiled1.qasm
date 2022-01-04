// Initial wiring: [0 1 2 4 7 3 6 8 5]
// Resulting wiring: [0 1 3 4 7 2 6 8 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[1];
cx q[2], q[3];
cx q[2], q[3];
cx q[2], q[3];
cx q[4], q[1];
cx q[4], q[3];
cx q[0], q[5];
cx q[6], q[5];
cx q[3], q[2];
