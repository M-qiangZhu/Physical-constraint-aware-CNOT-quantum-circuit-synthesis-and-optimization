OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[5],q[6];
u2(0,3.14159265358979) q[3];
cx q[6],q[2];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[2];
cx q[6],q[2];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[2];
cx q[6],q[2];
u2(0,3.14159265358979) q[6];
cx q[6],q[3];
u2(0,3.14159265358979) q[3];
u2(0,3.14159265358979) q[6];
cx q[4],q[1];
cx q[0],q[5];
