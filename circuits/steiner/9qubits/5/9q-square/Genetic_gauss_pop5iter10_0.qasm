// Initial wiring: [0 1 2 8 5 4 3 6 7]
// Resulting wiring: [0 1 2 8 5 4 3 6 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[2];
cx q[4], q[1];
cx q[5], q[0];
cx q[4], q[7];
cx q[3], q[8];
