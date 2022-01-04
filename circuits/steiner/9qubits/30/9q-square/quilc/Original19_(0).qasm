// EXPECTED_REWIRING [0 1 2 3 7 4 6 5 8]
// CURRENT_REWIRING [0 2 7 5 1 3 4 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[5];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[4];
rz(0.10344064106915161*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-3.075736053375836*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[7];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-3.075736053375836*pi) q[6];
rz(1.3572636036508112*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526896*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.8103884456122045*pi) q[7];
rz(-2.9280599304457073*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0636030200628968*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.760407881182692*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.495242038915076*pi) q[3];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
cz q[3], q[8];
rz(-2.9004621022578556*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5433740317527145*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.16311682313637377*pi) q[4];
rz(-1.4552255653600088*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[5];
rz(1.9770105968746368*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.976207047482915*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rx(-1.5707963267948966*pi) q[0];
cz q[1], q[0];
rz(1.674236967864048*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.7843290499389812*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.0779896335268964*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.381184772407101*pi) q[2];
cz q[2], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-0.34734009348347694*pi) q[5];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-0.0292881608500038*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.2183236063989553*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.05658878548833999*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.2451685808948216*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.255104124146284*pi) q[7];
cz q[7], q[4];
rz(1.0267048493344193*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(1.6818878091721041*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.2250718527224012*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.593459480039007*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rz(-1.1645820567151595*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.1653856061068779*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.5146654427613733*pi) q[2];
rz(-1.8734216173112568*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.1535324559627034*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.6463506146747164*pi) q[6];
cz q[8], q[7];
rz(0.49651206924412605*pi) q[4];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.547912551844372*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.958108965734335*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.474181780485484*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(-0.9244457121201792*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.3572636036508112*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.077989633526896*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.8103884456122045*pi) q[4];
cz q[2], q[3];
rz(1.4064587477307264*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.8220019369793847*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.972099917368147*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.41920979858509777*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.3490683979379892*pi) q[4];
cz q[4], q[1];
rz(-1.7712530916620748*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-0.38360487633445206*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7401880849566174*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.5535483121858533*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(0.10344064106915161*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.10374057071440759*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.8692885708165451*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.637887764149982*pi) q[4];
cz q[4], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-0.5668159222485896*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
cz q[1], q[2];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[5];
rz(1.9770105968746332*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.9762070474829154*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(-1.4891142355561604*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.3362264503142907*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.8424844978936856*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(0.21130219201097677*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.5707963267948966*pi) q[0];
rz(2.547912551844373*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.18348368785545804*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.8425981608894384*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0636030200628972*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-0.9033854536905883*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.7604078811826911*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[4];
rz(-2.7784866806896376*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.384484161973147*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.6410614979801341*pi) q[7];
cz q[7], q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.2059824549084524*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[3], q[8];
rz(2.2171469414696094*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.7633116952747185*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.8662457900317926*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9006644213688625*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.6940068969949138*pi) q[4];
cz q[4], q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(0.05267051292929693*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.18348368785545804*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.540038380699441*pi) q[3];
rz(-0.5939519083760968*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0636030200628952*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.887746383072439*pi) q[4];
cz q[4], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.1273385018897502*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.1645820567151652*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.1653856061068777*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-0.032720460772219795*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[4];
rz(-2.602467913520263*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731474*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.4361413542909993*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-1.1645820567151595*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.1653856061068779*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.5146654427613733*pi) q[7];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[3], q[8];
rx(-1.5707963267948966*pi) q[1];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.217146941469614*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rz(2.217146941469614*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-0.27168817109878796*pi) q[6];
rz(3.141592653589793*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
