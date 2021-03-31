import numpy

def shuffle_same(a,b):
    
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b) 

a = [1.1, 2.2, 3.3, 4.4, 5.5]
b = [1, 2, 3, 4, 5]

shuffle_same(a,b)

print(f'{a}\t{b}')

