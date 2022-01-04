# PyZX - Python library for quantum circuit rewriting 
#        and optimisation using the ZX-calculus
# Copyright (C) 2019 - Aleks Kissinger, John van de Wetering,
#                      and Arianne Meijer-van de Griend

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np


def make_fitness_func(func, **func_args):
    """
    Creates and returns a fitness function to be used for the genetic algorithm that uses CNOT gate count as fitness.

    :param func: a function determining the fitness of a single permutation
    :param func_args: extra arguments for the fitness function
    :return: A fitness function that only requires a permutation.
    """
    def fitness_func(permutation):
        return func(permutation=permutation, **func_args)

    return fitness_func #返回函数入口

class GeneticAlgorithm():

    def __init__(self, population_size, crossover_prob, mutation_prob, fitness_func, maximize=False, quiet=True):
        self.population_size = population_size
        self.negative_population_size = int(np.sqrt(population_size))
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.fitness_func = fitness_func
        self._sort = lambda l: l.sort(key=lambda x:x[1], reverse=maximize)
        self.maximize = maximize
        self.n_qubits = 0
        self.population = None
        self.quiet=quiet

    def _select(self):  #从种群中选取两个样本
        fitness_scores = [f for c,f in self.population] #排序后的种群的30个适应度值
        #print("fitness_scores:",fitness_scores)
        
        total_fitness = sum(fitness_scores)
        
        if self.maximize:
            selection_chance = [f/total_fitness for f in fitness_scores]
        else:
            max_fitness = max(fitness_scores) + 1
            adjusted_scores = [max_fitness - f for f in fitness_scores]
            adjusted_total = sum(adjusted_scores)
            selection_chance = [ f/adjusted_total for f in adjusted_scores]
            
        return np.random.choice(self.population_size, size=2, replace=False, p=selection_chance)
        #p=selection_chance为选取的概率

    def _create_population(self, n):
        self.population = [np.random.permutation(n) for _ in range(self.population_size)]
        #np.arange(n)的输出是有序的array([0,1,2....,n-1])，而经过np.random.permutation()则变成乱序
        #range(n)  #从0开始到 n-1
        #print("self.population 1:",self.population)
        '''
        种群大小个数为30个，每次执行得到的序列可能不一样
        [array([0, 2, 4, 3, 1]), array([4, 1, 0, 2, 3]), array([1, 2, 4, 0, 3]), 
        array([4, 2, 3, 0, 1]), array([4, 2, 3, 1, 0]), array([3, 0, 4, 1, 2]), 
        array([1, 4, 2, 0, 3]), array([3, 0, 4, 2, 1]), array([0, 3, 4, 2, 1]), 
        array([3, 2, 0, 1, 4]), array([4, 1, 2, 0, 3]), array([2, 1, 0, 3, 4]), 
        array([4, 0, 3, 1, 2]), array([0, 1, 2, 3, 4]), array([2, 4, 1, 3, 0]), 
        array([1, 2, 0, 3, 4]), array([3, 2, 0, 1, 4]), array([4, 1, 0, 3, 2]), 
        array([2, 0, 1, 4, 3]), array([0, 4, 2, 3, 1]), array([3, 4, 2, 0, 1]), 
        array([4, 3, 2, 0, 1]), array([1, 0, 3, 4, 2]), array([4, 1, 2, 0, 3]), 
        array([0, 3, 4, 1, 2]), array([2, 0, 1, 4, 3]), array([0, 3, 2, 4, 1]), 
        array([1, 4, 0, 2, 3]), array([3, 2, 1, 4, 0]), array([0, 2, 4, 3, 1])]'''
        
        self.population = [(chromosome, self.fitness_func(chromosome)) for chromosome in self.population]
        #print("self.population 2:",self.population)
        '''
        [(array([0, 2, 4, 3, 1]), 8), (array([4, 1, 0, 2, 3]), 7), (array([1, 2, 4, 0, 3]), 6), 
        (array([4, 2, 3, 0, 1]), 12), (array([4, 2, 3, 1, 0]), 12), (array([3, 0, 4, 1, 2]), 8), 
        (array([1, 4, 2, 0, 3]), 10), (array([3, 0, 4, 2, 1]), 8), (array([0, 3, 4, 2, 1]), 8), 
        (array([3, 2, 0, 1, 4]), 7), (array([4, 1, 2, 0, 3]), 10), (array([2, 1, 0, 3, 4]), 6), 
        (array([4, 0, 3, 1, 2]), 3), (array([0, 1, 2, 3, 4]), 7), (array([2, 4, 1, 3, 0]), 7), 
        (array([1, 2, 0, 3, 4]), 6), (array([3, 2, 0, 1, 4]), 7), (array([4, 1, 0, 3, 2]), 7), 
        (array([2, 0, 1, 4, 3]), 6), (array([0, 4, 2, 3, 1]), 3), (array([3, 4, 2, 0, 1]), 12), 
        (array([4, 3, 2, 0, 1]), 12), (array([1, 0, 3, 4, 2]), 7), (array([4, 1, 2, 0, 3]), 10), 
        (array([0, 3, 4, 1, 2]), 8), (array([2, 0, 1, 4, 3]), 6), (array([0, 3, 2, 4, 1]), 10), 
        (array([1, 4, 0, 2, 3]), 7), (array([3, 2, 1, 4, 0]), 3), (array([0, 2, 4, 3, 1]), 8)]'''
        
        self._sort(self.population)
        #print("self.population 3:",self.population) #按适应度值由小到大排序
        '''
        [(array([4, 0, 3, 1, 2]), 3), (array([0, 4, 2, 3, 1]), 3), (array([3, 2, 1, 4, 0]), 3), 
        (array([1, 2, 4, 0, 3]), 6), (array([2, 1, 0, 3, 4]), 6), (array([1, 2, 0, 3, 4]), 6), 
        (array([2, 0, 1, 4, 3]), 6), (array([2, 0, 1, 4, 3]), 6), (array([4, 1, 0, 2, 3]), 7), 
        (array([3, 2, 0, 1, 4]), 7), (array([0, 1, 2, 3, 4]), 7), (array([2, 4, 1, 3, 0]), 7), 
        (array([3, 2, 0, 1, 4]), 7), (array([4, 1, 0, 3, 2]), 7), (array([1, 0, 3, 4, 2]), 7), 
        (array([1, 4, 0, 2, 3]), 7), (array([0, 2, 4, 3, 1]), 8), (array([3, 0, 4, 1, 2]), 8), 
        (array([3, 0, 4, 2, 1]), 8), (array([0, 3, 4, 2, 1]), 8), (array([0, 3, 4, 1, 2]), 8), 
        (array([0, 2, 4, 3, 1]), 8), (array([1, 4, 2, 0, 3]), 10), (array([4, 1, 2, 0, 3]), 10), 
        (array([4, 1, 2, 0, 3]), 10), (array([0, 3, 2, 4, 1]), 10), (array([4, 2, 3, 0, 1]), 12), 
        (array([4, 2, 3, 1, 0]), 12), (array([3, 4, 2, 0, 1]), 12), (array([4, 3, 2, 0, 1]), 12)]'''
        
        #print("self.negative_population_size:",self.negative_population_size) #5
        self.negative_population = self.population[-self.negative_population_size:]
        #print("self.negative_population:",self.negative_population) #取最后5个序列
        '''
        [(array([0, 3, 2, 4, 1]), 10), (array([4, 2, 3, 0, 1]), 12), 
        (array([4, 2, 3, 1, 0]), 12), (array([3, 4, 2, 0, 1]), 12), 
        (array([4, 3, 2, 0, 1]), 12)]'''

    def find_optimimum(self, n_qubits, n_generations, initial_order=None, n_child=None, continued=False):
    #找到最优排序，好像每次都不一样，随机序列
        self.n_qubits = n_qubits
        partial_solution = False        
             
        #print("self.population:",self.population) #返回None
        
        #print("cc") #执行一次
        if not continued or self.population is None: #实参continued=True
            if initial_order is None:
                print("Create population!")  #执行一次
                self._create_population(n_qubits)  #会计算30个染色体的适应度值，30次调用gauss函数
            elif n_qubits < len(initial_order):
                self._create_population(initial_order[:n_qubits])
                partial_solution = True
            else:                
                self._create_population(initial_order)

        if n_child is None:
            n_child = self.population_size  #30
        
        #print("bb") #上面循环30次后输出"bb"，种群共有30个染色体，计算了每个染色体的适应度值
        
        for i in range(n_generations): #n_generations，修改为1进行调试，n_generations=50
            self._update_population(n_child) #新得到的染色体要计算适应度值，导致这边调用gauss函数很多次
            (not self.quiet) and print("Iteration", i, "best fitness:", [p[1] for p in self.population[:5]])
        
        #print("dd") #输出一次
        
        if partial_solution:
            return self.population[0] + initial_order[n_qubits:]
        
        #print("self.population[0][0]:",self.population[0][0])
        return self.population[0][0] #返回最好的序列
    

    def _add_children(self, children):
        n_child = len(children) #孩子个数
        
        #print("old population size：",len(self.population))
        self.population.extend([(child, self.fitness_func(child)) for child in children])
        #计算新增加的children中所有孩子的适应度值
        #extend在列表末尾一次性追加另一个序列中的多个值
        #print("self.population_size:",self.population) #30,每个含染色体和适应度值
        #print("new population size：",len(self.population)) 
        #值不一样，不能输出self.population_size，这是事先设定的30
        
        self._sort(self.population)  #重新根据适应度值排序
        
        self.negative_population.extend(self.population[-n_child:])
        self.negative_population = [self.negative_population[i] for i in np.random.choice(self.negative_population_size + n_child, size=self.negative_population_size, replace=False)]
        #重新设置为原来最差的几个+孩子个数==>重新选择最差的几个
        
        self.population = self.population[:self.population_size] #重新去种群的前30个，实现种群的更新
        #print("self.population:",self.population)
        

    def _update_population(self, n_child): #更新种群
        children = []
        # Create a child from weak parents to avoid local optima
        p1, p2 = np.random.choice(self.negative_population_size, size=2, replace=False)
        #print(p1,p2)
        #numpy.random.choice(a, size=None, replace=True, p=None)
        #从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
        #replace:True表示可以取相同数字，False表示不可以取相同数字
        #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        #print("self.negative_population[p1]:",self.negative_population[p1])
        #print("self.negative_population[p2]:",self.negative_population[p2])
        child = self._crossover(self.negative_population[p1][0], self.negative_population[p2][0])
        #print("child ok:",child)
        
        children.append(child)
        #count=0
        
        for _ in range(n_child):  #n_child，n_child = self.population_size，更改为1进行调试
            if np.random.random() < self.crossover_prob:
                #count=count+1
                p1, p2 = self._select()  #从种群中选择两个相对好的样本
                
                child = self._crossover(self.population[p1][0], self.population[p2][0]) #量样本交叉
                #print("len(child):",len(child)) #5，每个染色体含有5个基因
                
                #print("self.mutation_prob",self.mutation_prob)                
                if np.random.random() < self.mutation_prob:                    
                    child = self._mutate(child) #对新产生的样本进行变异
                children.append(child)
        #print("count=",count)  #count输出15次，每次的值不定，小于30
                
        self._add_children(children) #共循环30次，添加新的孩子，但不一定每次都成功

    def _crossover(self, parent1, parent2):  #两个父节点交叉得到孩子结点
        crossover_start = np.random.choice(int(self.n_qubits/2))
        #print("crossover_start:",crossover_start )
        
        crossover_length = np.random.choice(self.n_qubits-crossover_start)
        #print("crossover_length:",crossover_length)
        
        crossover_end = crossover_start + crossover_length
        #print("crossover_end:",crossover_end)
        
        child = -1*np.ones_like(parent1)
        #print("child 1:",child);
        
        child[crossover_start:crossover_end] = parent1[crossover_start: crossover_end]
        child_idx = 0
        for parent_gen in parent2:
            if child_idx == crossover_start: # skip over the parent1 part in child
                child_idx = crossover_end
            if parent_gen not in child: # only add new genes
                child[child_idx] = parent_gen
                child_idx += 1
        #print("child 2:",child);                
        return child

    def _mutate(self, parent):
        #print("len(parent):",len(parent))
        gen1, gen2 = np.random.choice(len(parent), size=2, replace=False)
        _ = parent[gen1]
        parent[gen1] = parent[gen2]
        parent[gen2] = _
        return parent

class ParticleSwarmOptimization():

    def __init__(self, swarm_size, fitness_func, step_func, s_best_crossover, p_best_crossover, mutation, maximize=False):
        self.fitness_func = fitness_func
        self.step_func = step_func
        self.size = swarm_size
        self.s_crossover = s_best_crossover
        self.p_crossover = p_best_crossover
        self.mutation = mutation
        self.best_particle = None
        self.maximize = maximize

    def _create_swarm(self, n):
        self.swarm = [Particle(n, self.fitness_func, self.step_func, self.s_crossover, self.p_crossover, self.mutation, self.maximize, id=i) 
                        for i in range(self.size)]

    def find_optimimum(self, n_qubits, n_steps, quiet=True):
        self._create_swarm(n_qubits)
        self.best_particle = self.swarm[0]
        for i in range(n_steps):
            self._update_swarm()
            (not quiet) and print("Iteration", i, "best fitness:", self.best_particle.best, self.best_particle.best_point)
        return self.best_particle.best_solution

    def _update_swarm(self):
        for p in self.swarm:
            if p.step(self.best_particle) and self.best_particle.compare(p.best):
                #print(p.best, self.best_particle.best)
                self.best_particle = p

class Particle():

    def __init__(self, size, fitness_func, step_func, s_best_crossover, p_best_crossover, mutation, maximize=False, id=None):
        self.fitness_func = fitness_func
        self.step_func = step_func
        self.size = size
        self.current = np.random.permutation(size)
        self.best_point = self.current
        self.best = None#fitness_func(self.current)
        self.best_solution = None
        self.s_crossover = int(s_best_crossover*size)
        self.p_crossover = int(p_best_crossover*size)
        self.mutation = int(mutation*size)
        self.maximize = maximize
        self.id = id
    
    def compare(self, x):
        if self.maximize:
            return x > self.best
        else:
            return x < self.best 

    def step(self, swarm_best):
        new, solution, fitness = self.step_func(self.current)
        is_better = self.best is None or self.compare(fitness)
        if is_better:
            self.best = fitness
            self.best_point = self.current
            self.best_solution = solution
        elif all([self.current[i] == n for i, n in enumerate(new)]):
            new = self._mutate(self.current)
            new = self._crossover(new, self.best_point, self.p_crossover)
            new = self._crossover(new, swarm_best.best_point, self.s_crossover)
            # Sanity check
            if any([i not in new for i in range(self.size)]): raise Exception("The new particle point is not a permutation anymore!" + str(self.current))
        self.current = new
        return is_better

    def _mutate(self, particle):
        new_particle = particle.copy()
        m_idxs = np.random.choice(self.size, size=self.mutation, replace=False)
        m_perm = np.random.permutation(self.mutation)
        for old_i, new_i in enumerate(m_perm):
            new_particle[m_idxs[old_i]] = particle[m_idxs[new_i]]
        return new_particle

    def _crossover(self, particle, best_particle, n):
        cross_idxs = np.random.choice(self.size, size=n, replace=False)
        new_particle = -1*np.ones_like(particle)
        new_particle[cross_idxs] = best_particle[cross_idxs]
        idx = 0
        for i, gen in enumerate(new_particle):
            if gen == -1: # skip over the parent1 part in child
                while(particle[idx] in new_particle):
                    idx += 1
                    if idx == len(particle):
                        break
                if idx < len(particle):
                    new_particle[i] = particle[idx]
        return new_particle


if __name__ == '__main__':
    def fitness_func(chromosome):
        t1 = 1
        t2 = 1
        size = len(chromosome)
        f1 = [chromosome[i]-i for i in range(size)]
        f2 = [size - g for g in f1]
        f1.sort()
        f2.sort()
        for i in range(1, size):
            t1 += int(f1[i] == f1[i-1])
            t2 += int(f2[i] == f2[i-1])
        return t1 + t2


    optimizer = GeneticAlgorithm(1000, 0.8, 0.2, fitness_func)
    optimizer.find_optimimum(8, 300)
    print(optimizer.population)

