// EXPECTED_REWIRING [0 1 2 3 4 5 6 7 8]
// CURRENT_REWIRING [4 6 1 2 5 8 0 7 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[1];
rz(-2.087802470758894*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.3844841619731474*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.2762476260936904*pi) q[5];
rz(-3.134124002624414*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.5835429475234226*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.107630628825903*pi) q[1];
rz(-2.35464027174787*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.8079140319578353*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rx(-1.5707963267948966*pi) q[1];
rz(0.6014893863396429*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[1], q[4];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rz(-1.581420723291024*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.7392075162607028*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[5], q[4];
rz(-0.9610891585641862*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-1.1167003845935939*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.8225111847632454*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.8944800488406148*pi) q[1];
rz(2.4297546979908042*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.59400022872158*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.8777786409970366*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.4704242218168844*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.8599423857572064*pi) q[3];
rz(-0.8604704159860347*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.6953743794757219*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.9871494128524133*pi) q[4];
cz q[4], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.0728263700044014*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.9770105968746305*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.9762070474829145*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[0];
rz(2.0854617695562663*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[5];
cz q[0], q[5];
rz(0.24271325173162997*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.2615998376377684*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6734730460682392*pi) q[1];
rz(-0.2333673054954118*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.36740609059007684*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.05940610578303782*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4058908772645693*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.6205701280555707*pi) q[6];
cz q[6], q[5];
rz(-1.9721313145625796*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.5707963267948966*pi) q[0];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
rz(1.5189033571174495*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4033951232470652*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(1.467355685725745*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rz(-1.7843290499389812*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.077989633526896*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.8103884456122044*pi) q[8];
rz(-1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(2.9981513498031727*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.6354421079960095*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-1.8085656289318865*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.2576165841250162*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.4641361161433162*pi) q[5];
cz q[5], q[0];
rz(2.6868818006456667*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(-3.039559323861411*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.1879677360056835*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.1533223308177902*pi) q[6];
rz(-1.7843290499389812*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.077989633526896*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.8103884456122044*pi) q[2];
rz(1.674236967864048*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-3.075736053375836*pi) q[4];
rz(-1.449665980387412*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.6713085494098852*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[6];
rz(-2.5479125518443744*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.958108965734335*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.6015542728903497*pi) q[3];
cz q[8], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(3.0807514611584077*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.3844841619731474*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.436141354291*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(2.6905695414460977*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.4420038024051023*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
cz q[0], q[1];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(0.7880332668299803*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.0248602506292002*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.7574055653498764*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.802264426293914*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.9575363255792705*pi) q[2];
cz q[2], q[1];
rz(-1.8946399859925531*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.355385716812451*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.3500049525446647*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[4], q[1];
rz(1.6644704781617667*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(1.164631813787238*pi) q[1];
rz(-1.0883036250757443*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.5153182073417444*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.784937804352989*pi) q[2];
rz(0.6463506146747173*pi) q[3];
rz(2.217146941469614*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(-1.1645820567151588*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.16538560610687789*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.626927210828419*pi) q[5];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[6];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.6269272108284194*pi) q[8];