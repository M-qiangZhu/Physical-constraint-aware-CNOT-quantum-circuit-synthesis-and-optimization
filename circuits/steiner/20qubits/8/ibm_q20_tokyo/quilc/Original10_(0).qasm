// EXPECTED_REWIRING [9 1 2 3 4 5 6 7 0 8 10 11 12 13 14 15 17 16 18 19]
// CURRENT_REWIRING [9 1 2 4 3 5 6 7 0 8 10 11 12 13 14 15 17 16 18 19]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[2];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[13];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-1.5707963267948966*pi) q[17];
rx(1.5707963267948966*pi) q[17];
cz q[17], q[12];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(0.10344064106915161*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.3572636036508126*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.063603020062897*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.760407881182692*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[8];
cz q[9], q[8];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
cz q[14], q[13];
cz q[17], q[11];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(-2.4952420389150767*pi) q[3];
rz(-1.1645820567151595*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.1653856061068779*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.5146654427613733*pi) q[4];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(3.141592653589793*pi) q[11];
rz(3.141592653589793*pi) q[12];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[17];
rz(1.5707963267948966*pi) q[17];
