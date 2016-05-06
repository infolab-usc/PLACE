__author__ = 'ubriela'


from Main import shanon_entropy_list, cut_list
import random

N = 20
low = 1
high = 100

count = 0
while True:
    count = count + 1
    x = random.sample(range(low, high), N)
    x_c = cut_list(x, random.randint(low, N))

    entropy_x = shanon_entropy_list(x)
    entropy_x_c = shanon_entropy_list(x_c)
    if entropy_x_c < entropy_x:
        print "----"
        print x, entropy_x
        print x_c, entropy_x_c

    if count % 10000 == 0:
        print count