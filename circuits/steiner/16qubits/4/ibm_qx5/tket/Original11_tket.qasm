OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
u2(0,3.14159265358979) q[7];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[5];
u2(0,3.14159265358979) q[4];
cx q[6],q[4];
u2(0,3.14159265358979) q[4];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[3];
cx q[3],q[7];
u2(0,3.14159265358979) q[7];
u2(0,3.14159265358979) q[3];
u2(0,3.14159265358979) q[2];
cx q[2],q[5];
u2(0,3.14159265358979) q[5];
u2(0,3.14159265358979) q[2];
u2(0,3.14159265358979) q[1];
u2(0,3.14159265358979) q[0];
cx q[1],q[0];
u2(0,3.14159265358979) q[0];
u2(0,3.14159265358979) q[1];
