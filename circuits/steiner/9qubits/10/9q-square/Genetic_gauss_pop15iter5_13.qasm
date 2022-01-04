// Initial wiring: [0 4 2 3 1 5 6 7 8]
// Resulting wiring: [0 3 2 8 1 5 7 6 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[7], q[8];
cx q[0], q[5];
cx q[5], q[0];
cx q[3], q[8];
cx q[3], q[8];
cx q[3], q[8];
cx q[0], q[1];
cx q[7], q[6];
cx q[7], q[6];
cx q[7], q[6];
cx q[1], q[4];
cx q[8], q[7];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[7], q[4];
