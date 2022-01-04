// Initial wiring: [0 1 2 4 3 6 5 7 8]
// Resulting wiring: [4 1 2 3 5 6 0 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[5];
cx q[0], q[5];
cx q[0], q[5];
cx q[4], q[5];
cx q[4], q[1];
cx q[7], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[4], q[7];
cx q[1], q[4];
cx q[6], q[5];
cx q[3], q[4];
cx q[4], q[5];
cx q[4], q[5];
cx q[4], q[5];
cx q[4], q[3];
cx q[4], q[1];
cx q[7], q[8];
