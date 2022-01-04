// EXPECTED_REWIRING [5 0 2 3 4 1 6 7 8]
// CURRENT_REWIRING [4 0 3 2 6 1 5 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[7];
cz q[6], q[5];
rz(2.331804894062909*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.584515786856178*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.5479125518443846*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.958108965734335*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.9280599304457082*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.0779896335268964*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(2.2382071998991933*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-2.911372688912084*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.8821859851106535*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.1177947877471492*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.342827428409607*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(3.0981985283744855*pi) q[4];
cz q[4], q[1];
rz(2.0144602590291036*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(3.141592653589793*pi) q[5];
rz(1.6742369678640499*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.269862686889883*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.0947085426961376*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.1075828256054248*pi) q[4];
cz q[4], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-2.756526430571448*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-2.087802470758894*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.3844841619731474*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.2762476260936904*pi) q[8];
rz(1.9770105968746343*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.9762070474829163*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-3.110605393605025*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.9271039284917535*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.9399764813603676*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.233983839573878*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.47582165576509*pi) q[8];
cz q[8], q[3];
rz(1.1416800330324426*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.8097877595268841*pi) q[2];
rz(-1.6372194710945358*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.746521289294279*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.3186961835377247*pi) q[3];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
cz q[4], q[5];
cz q[3], q[2];
rz(-1.5442544464060648*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.0693804402184202*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.4281625409774943*pi) q[1];
rz(-2.4746421405539696*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rx(1.5707963267948966*pi) q[1];
rz(0.800405172689922*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[5];
rz(1.1083455445067942*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.9581089657343345*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(3.109498066299372*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.0105260612115377*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.584462671896449*pi) q[8];
cz q[8], q[7];
rz(2.2382071998992057*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.3564646052644109*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(3.141592653589793*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(-1.1645820567151595*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1653856061068779*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.5146654427613733*pi) q[8];
rz(0.338271642386573*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.3736915007876933*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.0918040036156227*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9089424723356221*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.5468781778925278*pi) q[4];
cz q[4], q[1];
rz(1.3225629095531115*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
cz q[6], q[5];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(-0.9244457121201797*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.6759543220020803*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.5667840156353474*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.5707963267948966*pi) q[6];
rz(-1.0632519655999602*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5335697922651068*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.8823171049008409*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.48766550834711614*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.4010921786567418*pi) q[5];
cz q[5], q[4];
rz(-2.491786612513719*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
cz q[2], q[3];
rz(-1.7000776645648947*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.6515950678558102*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
rz(-2.145656431773837*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(1.3772207884175731*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(1.5707963267948966*pi) q[4];
rz(-0.008076089343591164*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.0573771342155913*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.550077292875736*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.927878705377247*pi) q[8];
rz(-2.164476428540316*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.183483687855458*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.6674108731043075*pi) q[4];
rx(1.5707963267948966*pi) q[7];
rz(2.363778536040392*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(2.9128257749244946*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.6431416872111939*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.42465881181700216*pi) q[1];
rz(-1.205656996053951*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9186299685751856*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(2.926833255070073*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.0636030200628972*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.7604078811826923*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(0.07365734903162893*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.6978639754919513*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.13004364849779942*pi) q[1];
rz(0.8189354765689085*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.0056506763459478*pi) q[2];
rz(-3.0381520125206416*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[8], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
cz q[4], q[7];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.21353272314408578*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.0779896335268964*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.3811847724071016*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(1.9770105968746385*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.9762070474829154*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.70699528232417*pi) q[7];
cz q[6], q[7];
rz(1.4564375502462912*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4269954866939927*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.08197635545524956*pi) q[8];
rz(2.9280599304457073*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.0779896335268964*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.3811847724071016*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(0.24271325173163108*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.261599837637768*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[7];
rz(-2.3226571770208824*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[2];
rz(2.217146941469614*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-2.4681196075215532*pi) q[4];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.6463506146747173*pi) q[5];
rz(3.0381520125206407*pi) q[6];
rz(0.6508643982906501*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[8];
