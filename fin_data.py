import sys
sys.path.append('./lib')
import numpy as np
from data import snp500,snp500_individual
# s.prepare_pd()
# table = s.get_pd_table()
# size = table['Log Return'].as_matrix().size
# k = table['Log Return'].as_matrix()

class data_generator():
        def __init__(self):
                self.sequence_length = 1024*8
                self.batch_size = 12
                self.learning_phase = "train"
                self.train_split = 0.8

        def snp500_index(self):
                s = snp500()
                s.prepare_pd()
                table = s.get_pd_table()
                return table['Log Return'].as_matrix()
        def real_data(self,mode = "individuals"):
                if mode == "individuals":
                        return self.individuals_data_random_picker()
                elif mode == "index":
                        return self.index_data_random_picker()

        def individuals_data_random_picker(self):
                data = []
                for i in range(self.batch_size):
                        random_code = self.choose_random_code()
                        si = snp500_individual(random_code)
                        si.prepare_pd()
                        table = si.get_pd_table()
                        sequence = table['Log Return'].as_matrix()
                        size = table['Log Return'].as_matrix().size
                        random_pos = np.random.randint(0,size-self.sequence_length)
                        data.append([sequence[random_pos:random_pos+self.sequence_length]])
                data = np.array(data)
                #data /= max(data.max(),-data.min())
                data = np.reshape(data,(self.batch_size,self.sequence_length,1))
                return data

        def choose_random_code(self):
                s = snp500()
                codes = s.get_code_list()
                codes_size = len(codes)
                size = 0
                while size < self.sequence_length:
                        if self.train_split == -1:
                                random_code = codes[np.random.randint(0,codes_size)]
                        else:
                                if self.learning_phase == "train":
                                        random_code = codes[np.random.randint(0,int(self.train_split*codes_size))]
                                else:
                                        random_code = codes[np.random.randint(int(self.train_split*codes_size),codes_size)]
                        si = snp500_individual(random_code)
                        si.prepare_pd()
                        table = si.get_pd_table()
                        size = table['Log Return'].as_matrix().size
                return random_code

        def index_data_random_picker(self):
                s = snp500()
                s.prepare_pd()
                table = s.get_pd_table()
                sequence = table['Log Return'].as_matrix()
                size = table['Log Return'].as_matrix().size
                data = []
                for i in range(self.batch_size):
                        random_pos = np.random.randint(0,size-self.sequence_length)
                        data.append([sequence[random_pos:random_pos+self.sequence_length]])
                data = np.array(data)
                #data /= max(data.max(),-data.min())
                data = np.reshape(data,(self.batch_size,self.sequence_length,1))
                return data

