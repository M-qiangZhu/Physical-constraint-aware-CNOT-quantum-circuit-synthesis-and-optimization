// Initial wiring: [0 4 2 3 1 6 5 7 8]
// Resulting wiring: [1 4 2 3 0 6 5 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[7];
cx q[7], q[4];
cx q[2], q[1];
cx q[4], q[5];
cx q[0], q[1];
cx q[0], q[1];
cx q[0], q[1];
cx q[1], q[4];
