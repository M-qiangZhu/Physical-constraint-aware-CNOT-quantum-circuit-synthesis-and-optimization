// Initial wiring: [1 4 2 3 0 5 8 6 7]
// Resulting wiring: [1 4 2 3 0 5 8 6 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[4];
cx q[4], q[1];
cx q[1], q[0];
cx q[5], q[6];
cx q[7], q[8];
