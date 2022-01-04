// Initial wiring: [0 1 2 8 3 5 6 7 4]
// Resulting wiring: [0 1 2 3 8 5 6 7 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[3], q[8];
cx q[3], q[8];
cx q[3], q[8];
cx q[5], q[4];
cx q[2], q[3];
