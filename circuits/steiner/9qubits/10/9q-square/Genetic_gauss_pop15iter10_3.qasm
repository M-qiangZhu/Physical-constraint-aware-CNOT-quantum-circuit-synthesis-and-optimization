// Initial wiring: [0 3 2 4 7 5 6 1 8]
// Resulting wiring: [5 8 2 4 7 0 6 1 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[8], q[7];
cx q[1], q[2];
cx q[5], q[0];
cx q[4], q[5];
cx q[5], q[0];
cx q[5], q[0];
cx q[5], q[0];
cx q[1], q[4];
cx q[1], q[0];
cx q[8], q[3];
cx q[8], q[3];
cx q[8], q[3];
cx q[2], q[3];
cx q[2], q[1];
cx q[4], q[7];
