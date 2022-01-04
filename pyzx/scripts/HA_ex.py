from random import random, randint

error_list = [0.018282915538629818,
             0.02382597319762883,
             0.021199360577361798,
             0.016671016328373228,
             0.025117409134446322,
             0.013859579352843793]

gates_list = [9, 12, 17, 23, 50, 65, 130]

for gates in gates_list:
    result = 1
    for i in range(gates):
        index = randint(0, 5)
        # if index == 5:
        #     print(f"index = {index}")
        e = 1 - error_list[index]
        result *= e
    print(result)