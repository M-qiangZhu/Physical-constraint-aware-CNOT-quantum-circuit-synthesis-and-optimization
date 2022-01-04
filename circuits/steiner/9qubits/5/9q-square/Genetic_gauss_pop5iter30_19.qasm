// Initial wiring: [0 1 3 2 4 5 7 6 8]
// Resulting wiring: [0 1 3 2 4 6 7 5 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[4];
cx q[6], q[5];
cx q[6], q[5];
cx q[6], q[5];
cx q[4], q[5];
cx q[0], q[5];
cx q[3], q[4];
