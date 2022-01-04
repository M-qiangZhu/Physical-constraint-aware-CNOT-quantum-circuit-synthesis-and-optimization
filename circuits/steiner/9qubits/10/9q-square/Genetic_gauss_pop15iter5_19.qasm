// Initial wiring: [0 5 8 3 2 4 6 7 1]
// Resulting wiring: [5 0 3 8 2 4 6 7 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[8];
cx q[1], q[0];
cx q[0], q[5];
cx q[0], q[5];
cx q[0], q[5];
cx q[7], q[6];
cx q[4], q[7];
cx q[3], q[8];
cx q[6], q[5];
cx q[8], q[7];
cx q[1], q[2];
cx q[6], q[7];
