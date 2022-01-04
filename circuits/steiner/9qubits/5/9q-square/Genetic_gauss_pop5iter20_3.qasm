// Initial wiring: [0 4 2 5 8 1 6 7 3]
// Resulting wiring: [0 4 2 6 8 1 5 7 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[6];
cx q[6], q[5];
cx q[6], q[5];
cx q[6], q[5];
cx q[0], q[5];
cx q[2], q[3];
cx q[4], q[7];
