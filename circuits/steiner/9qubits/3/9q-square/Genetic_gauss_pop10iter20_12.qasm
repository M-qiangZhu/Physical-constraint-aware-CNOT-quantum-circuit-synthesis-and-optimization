// Initial wiring: [5 1 2 8 4 0 6 7 3]
// Resulting wiring: [5 1 2 8 4 0 6 7 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[4];
cx q[4], q[5];
cx q[3], q[2];
