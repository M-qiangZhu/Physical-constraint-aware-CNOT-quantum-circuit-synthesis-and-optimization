// Initial wiring: [0 2 1 3 4 5 7 8 6]
// Resulting wiring: [0 2 1 3 4 5 6 8 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[0];
cx q[5], q[6];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[2], q[1];
cx q[5], q[6];
cx q[3], q[4];
