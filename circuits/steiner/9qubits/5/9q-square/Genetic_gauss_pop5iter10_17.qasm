// Initial wiring: [0 1 2 4 8 5 6 7 3]
// Resulting wiring: [0 1 2 4 3 5 6 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[7];
cx q[3], q[2];
cx q[3], q[8];
cx q[3], q[8];
cx q[3], q[8];
cx q[1], q[4];
cx q[2], q[3];
