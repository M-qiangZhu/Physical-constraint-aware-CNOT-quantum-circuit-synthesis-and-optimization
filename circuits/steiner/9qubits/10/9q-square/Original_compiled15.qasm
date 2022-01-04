// Initial wiring: [0 2 1 3 7 5 6 4 8]
// Resulting wiring: [5 2 1 3 7 0 6 4 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[6];
cx q[0], q[1];
cx q[0], q[5];
cx q[0], q[5];
cx q[0], q[5];
cx q[6], q[5];
cx q[7], q[8];
cx q[1], q[0];
cx q[5], q[4];
cx q[4], q[7];
cx q[0], q[5];
cx q[1], q[4];
cx q[3], q[2];
