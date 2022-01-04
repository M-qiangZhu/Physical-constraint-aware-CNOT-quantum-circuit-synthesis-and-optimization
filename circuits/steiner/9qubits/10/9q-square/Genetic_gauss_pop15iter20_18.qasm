// Initial wiring: [0 1 2 3 5 4 6 7 8]
// Resulting wiring: [5 1 2 3 8 0 7 4 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[5];
cx q[4], q[5];
cx q[4], q[5];
cx q[4], q[5];
cx q[4], q[3];
cx q[4], q[7];
cx q[4], q[7];
cx q[4], q[7];
cx q[6], q[7];
cx q[0], q[5];
cx q[0], q[5];
cx q[0], q[5];
cx q[6], q[5];
cx q[7], q[8];
cx q[7], q[8];
cx q[7], q[8];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[4], q[1];
cx q[2], q[1];
cx q[8], q[7];
