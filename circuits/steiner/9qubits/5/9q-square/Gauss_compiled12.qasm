// Initial wiring: [7 1 3 2 0 5 4 6 8]
// Resulting wiring: [7 4 3 2 0 1 5 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[0], q[1];
cx q[5], q[4];
cx q[5], q[4];
cx q[5], q[4];
cx q[0], q[5];
cx q[1], q[4];
cx q[1], q[4];
cx q[1], q[4];
cx q[8], q[3];
cx q[0], q[1];
cx q[7], q[8];
