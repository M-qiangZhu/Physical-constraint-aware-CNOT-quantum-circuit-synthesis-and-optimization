// Initial wiring: [5 1 2 3 4 0 6 7 8]
// Resulting wiring: [0 1 2 3 7 5 6 8 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[6], q[7];
cx q[7], q[8];
cx q[7], q[8];
cx q[7], q[8];
cx q[4], q[7];
cx q[4], q[7];
cx q[4], q[7];
cx q[4], q[5];
cx q[5], q[0];
cx q[5], q[0];
cx q[5], q[0];
cx q[4], q[1];
cx q[1], q[0];
cx q[3], q[2];
