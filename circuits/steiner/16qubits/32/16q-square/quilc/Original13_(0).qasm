// EXPECTED_REWIRING [0 1 4 3 5 2 6 7 8 9 13 11 12 10 15 14]
// CURRENT_REWIRING [2 10 9 4 7 14 3 0 11 5 6 13 12 1 8 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(0.10344064106915161*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-3.075736053375836*pi) q[4];
rz(0.11520055559665443*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.236578819567307*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[7];
rz(0.10344064106915161*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.21353272314408345*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.077989633526895*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.381184772407102*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[6];
rz(1.357263603650812*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.0779896335268964*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.381184772407101*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(-0.40113382584624935*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.178000296541483*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.0785041912113424*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.545026407281064*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.1477914400169333*pi) q[6];
cz q[6], q[5];
rz(0.23362339419857747*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
cz q[8], q[7];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
rz(0.2951935340831708*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5929641358580808*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[7];
rz(2.0971207217406316*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.545088326018615*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-1.122743568159537*pi) q[11];
cz q[11], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.4953985695227985*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-2.768874955770027*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.238210140166909*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.4061670849019174*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.1151405284656875*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[4];
rz(-0.869838649084334*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.2159195906933558*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-3.069206940740111*pi) q[5];
rz(0.59368010174542*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.958108965734335*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.4741817804854853*pi) q[2];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.9770105968746388*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.9762070474829176*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(2.932152933304131*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.0636030200628979*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.7604078811826924*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[9], q[6];
rx(1.5707963267948966*pi) q[10];
rz(-2.236578819567308*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
cz q[9], q[10];
rz(-2.087802470758894*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.3844841619731474*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.2762476260936904*pi) q[14];
rz(3.141592653589793*pi) q[10];
rz(-1.0561308840335186*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[11];
cz q[5], q[4];
rz(-0.1034406410691524*pi) q[6];
rz(-2.6625757902999436*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.8385954038498082*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.387104966695441*pi) q[1];
rz(1.3572636036508112*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.0636030200628979*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.7604078811826911*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(0.10344064106915161*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[14], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
cz q[11], q[10];
rz(-1.1645820567151592*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687783*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.5146654427613733*pi) q[14];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.7590849181334149*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.077989633526896*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.8103884456122048*pi) q[3];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[10];
rz(-1.7843290499389812*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.077989633526896*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.8103884456122044*pi) q[11];
rz(2.217146941469614*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-3.0381520125206416*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[11], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(2.4038216736486877*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-2.320444232073701*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-2.7688749557700234*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.1645820567151637*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.1653856061068781*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[2];
cz q[3], q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.2511108496643026*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5797016675066762*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.6289285952229582*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7635807314863516*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.543779682520875*pi) q[10];
cz q[10], q[5];
rz(1.0495672358449597*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6463506146747164*pi) q[1];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.7411644049202186*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.6901856032353963*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.3891723554277626*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5445939793679184*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5659849117953182*pi) q[9];
rz(1.754978157360114*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.899375072830781*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.6180327100708016*pi) q[10];
cz q[10], q[9];
rz(-0.7568693458007694*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3930587681241233*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-0.24171432676996302*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.1832441131482887*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.6596456431279895*pi) q[11];
rz(-2.6625757902999436*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.8385954038498077*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.387104966695441*pi) q[9];
rz(-0.9061618764710467*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.7843290499389812*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.0779896335268964*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.381184772407101*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-0.1034406410691524*pi) q[7];
rz(-1.1645820567151581*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687808*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[7];
rz(-1.6139737325617298*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9634861998971104*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.7342273243371079*pi) q[9];
rz(-1.4947590430564706*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.7569819896711418*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(-1.5707963267948966*pi) q[9];
rz(-0.434481569489539*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(-1.5707963267948966*pi) q[10];
cz q[11], q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(0.17878345932519357*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.5119890908486053*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(0.6412827377442714*pi) q[14];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rz(-1.3919376616573098*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5296387560638005*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.461984464989155*pi) q[8];
cz q[15], q[14];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(3.141592653589793*pi) q[14];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(1.9770105968746388*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.9762070474829154*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[12];
rz(-2.6625757902999436*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.8385954038498077*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.387104966695441*pi) q[10];
cz q[13], q[14];
rz(3.141592653589793*pi) q[14];
rz(-1.7843290499389823*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.0636030200628976*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.760407881182692*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(-2.6625757902999436*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.8385954038498077*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.387104966695441*pi) q[5];
rz(-1.784329049938982*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.0779896335268955*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-2.381184772407101*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-0.9244457121201792*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.14358331375756608*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.45302580669267023*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[2];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(2.69149586560195*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.18348368785545838*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.1272958570777443*pi) q[1];
rz(-0.21353272314408464*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.077989633526896*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.3811847724071016*pi) q[2];
cz q[2], q[1];
rz(0.22391040338714507*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.1645820567151595*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.1653856061068779*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.5146654427613733*pi) q[3];
rz(-1.1645820567151592*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.16538560610687794*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.730367851897572*pi) q[2];
cz q[5], q[2];
rz(-1.1645820567151595*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.1653856061068779*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.5146654427613733*pi) q[10];
cz q[5], q[10];
rz(2.033247109083004*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.1834836878554576*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.7362880238478154*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.7870571649321592*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.1399611371027882*pi) q[9];
cz q[9], q[6];
rz(-0.9033854536905943*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.0451604648084385*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
cz q[3], q[2];
rz(-2.3918013978459243*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.4673556857257442*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[2], q[5];
rz(-0.9244457121201792*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[10];
cz q[4], q[11];
rz(-1.1645820567151632*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.165385606106878*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-0.4530258066926707*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6463506146747164*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(1.4564375502462912*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.426995486693993*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-3.059616298134544*pi) q[6];
rz(3.141592653589793*pi) q[7];
rz(0.2427132517316307*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.261599837637768*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.4681196075215537*pi) q[8];
rz(-1.1645820567151595*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.1653856061068779*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.5146654427613733*pi) q[9];
rz(0.5146654427613777*pi) q[11];
rx(3.141592653589793*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-1.5707963267948966*pi) q[12];
rz(-1.1645820567151595*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.1653856061068779*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.5146654427613733*pi) q[13];
rz(3.141592653589793*pi) q[14];
rz(1.0561308840335282*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(-1.5707963267948966*pi) q[15];
