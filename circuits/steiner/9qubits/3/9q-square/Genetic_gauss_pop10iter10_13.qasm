// Initial wiring: [0 1 2 8 7 5 4 6 3]
// Resulting wiring: [0 1 2 3 7 5 4 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[3], q[8];
cx q[3], q[8];
cx q[3], q[8];
cx q[1], q[4];
cx q[2], q[3];
