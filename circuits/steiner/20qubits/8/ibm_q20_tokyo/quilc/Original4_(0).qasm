// EXPECTED_REWIRING [1 9 2 3 4 5 6 7 8 0 19 11 12 13 14 15 16 17 18 10]
// CURRENT_REWIRING [1 9 13 8 4 5 6 2 3 0 19 11 12 7 14 15 16 17 18 10]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[2];
rz(-1.3754252720873812*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9489447954131095*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.637986794567288*pi) q[2];
rz(1.1448789610554453*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.7465354320955955*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[2], q[7];
rx(-1.5707963267948966*pi) q[2];
rz(-0.9874294873500755*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[2], q[7];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[7];
cz q[2], q[7];
rz(-2.087802470758894*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.3844841619731474*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.2762476260936904*pi) q[13];
rz(-2.5380195036829494*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.199287883159672*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.3492270773191928*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.7427018823858714*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.42096968930076595*pi) q[8];
cz q[8], q[2];
rz(-1.5244926902814477*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[2];
rz(-1.5707963267948966*pi) q[17];
rx(1.5707963267948966*pi) q[17];
cz q[17], q[11];
rz(2.4581734164557667*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.3053506845355098*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.1333754095704456*pi) q[2];
rz(-2.087802470758894*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.3844841619731474*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.2762476260936904*pi) q[3];
rz(2.7686212116126923*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.4602783523967934*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[13], q[7];
rz(2.5828937829287364*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[7];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
cz q[3], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-0.9941073278485633*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.2812393156934503*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[11];
rz(-2.828672371899947*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.284184088186584*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.06621968432810038*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.8478059182654604*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.4333331463394254*pi) q[8];
cz q[8], q[2];
rz(-1.3221726206226743*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[2];
rz(2.8264539594399785*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.120644673254948*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.7395318684441827*pi) q[8];
cz q[8], q[10];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[1];
cz q[8], q[9];
rz(1.9770105968746332*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.976207047482916*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[15];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[18];
rz(3.141592653589793*pi) q[1];
rz(-0.18778729442023123*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.8642689101048868*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.4589877529604713*pi) q[2];
rz(-1.1645820567151595*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.1653856061068779*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.5146654427613733*pi) q[3];
rz(2.217146941469614*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(1.257821817832986*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(3.141592653589793*pi) q[9];
rz(3.141592653589793*pi) q[10];
rz(3.141592653589793*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(0.5146654427613728*pi) q[13];
rx(3.141592653589793*pi) q[13];
rz(1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[17];
rz(1.5707963267948966*pi) q[17];
rz(3.141592653589793*pi) q[18];
