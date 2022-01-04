// Initial wiring: [0 4 2 3 1 5 6 7 8]
// Resulting wiring: [0 6 2 8 1 5 7 4 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[7], q[4];
cx q[7], q[4];
cx q[7], q[4];
cx q[4], q[5];
cx q[4], q[5];
cx q[4], q[5];
cx q[0], q[5];
cx q[3], q[4];
cx q[6], q[5];
cx q[3], q[8];
cx q[3], q[8];
cx q[3], q[8];
cx q[5], q[4];
cx q[5], q[4];
cx q[5], q[4];
cx q[0], q[1];
cx q[1], q[0];
cx q[3], q[4];
cx q[1], q[4];
cx q[6], q[7];
cx q[6], q[7];
cx q[7], q[4];
