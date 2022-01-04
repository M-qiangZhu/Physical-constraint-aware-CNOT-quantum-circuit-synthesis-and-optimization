// Initial wiring: [0 1 2 4 5 3 6 7 8]
// Resulting wiring: [0 1 2 4 7 3 5 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[6];
cx q[5], q[6];
cx q[5], q[6];
cx q[5], q[4];
cx q[4], q[1];
cx q[5], q[6];
cx q[4], q[7];
cx q[4], q[5];
cx q[5], q[0];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[8], q[3];
cx q[5], q[6];
cx q[2], q[3];
