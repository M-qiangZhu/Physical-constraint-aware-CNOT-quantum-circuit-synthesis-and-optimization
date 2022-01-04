// EXPECTED_REWIRING [0 7 2 3 1 5 6 4 8]
// CURRENT_REWIRING [8 3 7 2 0 5 1 6 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(0.5936801017454187*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6015542728903499*pi) q[0];
rz(-2.9280599304457082*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.0636030200628968*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.760407881182692*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(-2.5901077679358693*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.130843565491726*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
rz(2.1308435654917246*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-0.12775558440406531*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.6555425635497554*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.999399242235195*pi) q[5];
rz(-1.2959018413173293*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(1.5707963267948966*pi) q[5];
cz q[0], q[5];
rz(1.698017948417832*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.9279123624474037*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.3026280948923686*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.6078206621704134*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.7137854612660701*pi) q[2];
cz q[2], q[1];
rz(1.3323895609037866*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(2.217146941469614*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.31828854033336995*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.11981406464694813*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-1.78808772168094*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.2587803749192628*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.518099679758757*pi) q[2];
cz q[2], q[3];
rz(-0.9771162250494777*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4741817804854858*pi) q[0];
rz(1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[4];
cz q[1], q[4];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[8];
rz(-1.389593969462171*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.1447567340129075*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.5856767569008072*pi) q[4];
rz(1.1386450540576205*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.248492824591424*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[4], q[7];
rx(-1.5707963267948966*pi) q[4];
rz(-0.1101753796214382*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[4], q[7];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[7];
cz q[4], q[7];
rz(2.653165444968142*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.0779896335268964*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.8103884456122046*pi) q[5];
rz(0.24620628585472598*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.6699160232195086*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.3655793241091767*pi) q[7];
rz(3.141592653589793*pi) q[8];
rz(-1.4156544921979328*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.6691582740146005*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.6775219570448696*pi) q[6];
cz q[7], q[8];
rz(1.1934349010472056*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.0636030200628965*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.7604078811826902*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-3.0000451013810694*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.6505269541914358*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.5594471915786292*pi) q[4];
cz q[5], q[4];
rz(-1.0580788579204237*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-0.6435078855677968*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.1074841903175618*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.7143998407429077*pi) q[0];
rz(2.761369489712264*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.9641888827222767*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.9438241621069082*pi) q[5];
rz(-1.7843290499389823*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.0636030200628976*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.760407881182692*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.3572636036508112*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.077989633526896*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.8103884456122045*pi) q[8];
cz q[5], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(-1.1645820567151592*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.16538560610687794*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.730367851897572*pi) q[1];
rz(1.243009766619223*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rz(1.432368119281954*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9919829889765901*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.3741364628761206*pi) q[0];
rz(2.1083086675767078*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-1.1645820567151595*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.1653856061068779*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.5146654427613733*pi) q[5];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-0.9244457121201792*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[7];
cz q[8], q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
cz q[2], q[1];
rz(1.9770105968746334*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.9762070474829154*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.9997441003053645*pi) q[8];
cz q[3], q[8];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.495242038915076*pi) q[6];
rx(1.5707963267948966*pi) q[4];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[4], q[7];
rz(2.7067338403546013*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.6961978967896887*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.2689665954887484*pi) q[1];
rz(2.3072400458176126*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.627747406714956*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rx(-1.5707963267948966*pi) q[1];
rz(0.3608663680159072*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[1], q[4];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rz(2.547912551844375*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.958108965734335*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.21353272314408597*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.077989633526897*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(2.2382071998992012*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-2.381184772407101*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.241148243797139*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4553733359119096*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[7];
cz q[6], q[7];
rz(0.8711893515551128*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.217146941469612*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[4], q[7];
rx(-1.5707963267948966*pi) q[3];
rz(-2.9280599304457082*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.063603020062897*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.760407881182692*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.0857176692509034*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[8];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.2656962678891135*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.3787777837664*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rx(-1.5707963267948966*pi) q[4];
rz(0.21353272314408445*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.0636030200628974*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.7604078811826919*pi) q[7];
cz q[7], q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.495242038915076*pi) q[4];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
cz q[6], q[7];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[8], q[3];
cz q[4], q[7];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(2.217146941469613*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-0.7628148698233925*pi) q[0];
rz(-1.9286579485976747*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.2312905777274112*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.033527820764972*pi) q[1];
rz(0.2427132517316307*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.261599837637768*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.4681196075215537*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rz(1.2349434038497193*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rz(3.0381520125206407*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
