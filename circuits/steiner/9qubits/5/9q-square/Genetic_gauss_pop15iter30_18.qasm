// Initial wiring: [5 0 2 3 4 1 8 6 7]
// Resulting wiring: [5 0 2 3 4 1 8 6 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[6], q[5];
cx q[5], q[4];
cx q[4], q[5];
cx q[4], q[1];
cx q[6], q[7];
