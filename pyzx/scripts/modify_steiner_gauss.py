from pyzx.scripts import cnot2cnot
from pyzx.routing import cnot_mapper
from pyzx.scripts import my_experiment


def get_matrix():
    result = my_experiment.processor("9q_square", "3")



if __name__ == '__main__':
    get_matrix()