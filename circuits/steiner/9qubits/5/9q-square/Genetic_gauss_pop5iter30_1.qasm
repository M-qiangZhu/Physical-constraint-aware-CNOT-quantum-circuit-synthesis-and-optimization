// Initial wiring: [0 2 1 5 3 6 4 8 7]
// Resulting wiring: [0 1 2 5 3 6 4 8 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[0];
cx q[0], q[5];
cx q[1], q[2];
cx q[1], q[2];
cx q[1], q[2];
cx q[6], q[7];
cx q[0], q[1];
cx q[4], q[7];
