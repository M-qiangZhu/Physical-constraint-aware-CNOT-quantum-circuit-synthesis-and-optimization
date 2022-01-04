// Initial wiring: [0 2 1 3 7 5 6 4 8]
// Resulting wiring: [5 3 1 2 7 0 6 4 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[8], q[7];
cx q[4], q[5];
cx q[4], q[3];
cx q[0], q[5];
cx q[0], q[5];
cx q[0], q[5];
cx q[7], q[6];
cx q[2], q[3];
cx q[2], q[3];
cx q[2], q[3];
cx q[1], q[4];
cx q[6], q[5];
cx q[4], q[3];
