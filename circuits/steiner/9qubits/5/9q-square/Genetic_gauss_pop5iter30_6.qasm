// Initial wiring: [0 1 2 3 4 6 5 7 8]
// Resulting wiring: [0 1 3 2 4 6 5 8 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[3];
cx q[8], q[7];
cx q[8], q[7];
cx q[8], q[7];
cx q[2], q[3];
cx q[2], q[3];
cx q[2], q[3];
cx q[0], q[1];
cx q[3], q[8];
cx q[6], q[7];
