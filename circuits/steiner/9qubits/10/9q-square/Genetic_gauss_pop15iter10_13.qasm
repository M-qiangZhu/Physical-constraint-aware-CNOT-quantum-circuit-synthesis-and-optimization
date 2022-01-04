// Initial wiring: [0 1 2 3 4 5 6 7 8]
// Resulting wiring: [5 7 1 8 2 0 6 4 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[4], q[7];
cx q[4], q[7];
cx q[4], q[7];
cx q[1], q[2];
cx q[1], q[2];
cx q[1], q[2];
cx q[1], q[0];
cx q[8], q[3];
cx q[8], q[3];
cx q[8], q[3];
cx q[1], q[4];
cx q[0], q[5];
cx q[0], q[5];
cx q[0], q[5];
cx q[7], q[6];
cx q[6], q[5];
cx q[4], q[3];
